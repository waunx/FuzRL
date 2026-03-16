import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys
import os
import spinup.algos.fuzppolag.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
import torch.nn as nn
import torch.nn.functional as F
import warnings
from spinup.utils.fuzzynet import (
    FuzzyNet,
    choquet_integral,
    choquet_integral_dual,
    PerturbationSampler,
)


class FuzBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97, outlevel=10, device="cuda:0"):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.true_ret_buf = np.zeros(size, dtype=np.float32)
        self.true_cret_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.v_disturb_buf = np.zeros(core.combined_shape(size, outlevel), dtype=np.float32)
        self.vc_disturb_buf = np.zeros(core.combined_shape(size, outlevel), dtype=np.float32)

        if isinstance(obs_dim, tuple):
            state_dim = obs_dim[0]
        else:
            state_dim = obs_dim
        
        self.fvnext_buf = np.zeros(size, dtype=np.float32)
        self.fcvnext_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, crew, val, cval, v_disturb, vc_disturb,
              fvnext, fcvnext, logp):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.v_disturb_buf[self.ptr] = v_disturb
        self.vc_disturb_buf[self.ptr] = vc_disturb

        self.fvnext_buf[self.ptr] = fvnext
        self.fcvnext_buf[self.ptr] = fcvnext

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        fvals_next = self.fvnext_buf[path_slice]
        fcvals_next = self.fcvnext_buf[path_slice]
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = rews[:-1] + self.gamma * fvals_next
        self.cret_buf[path_slice] = crews[:-1] + self.gamma * fcvals_next

        self.true_ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.true_cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) / cadv_std

        data = dict(
            obs=self.obs_buf, 
            act=self.act_buf, 
            next_obs=self.next_obs_buf,
            rew=self.rew_buf, 
            crew=self.crew_buf,
            ret=self.ret_buf, 
            cret=self.cret_buf, 
            t_ret=self.true_ret_buf, 
            t_cret=self.true_cret_buf,
            v=self.val_buf, 
            vc=self.cval_buf,
            v_disturb=self.v_disturb_buf, 
            vc_disturb=self.vc_disturb_buf,
            adv=self.adv_buf, 
            cadv=self.cadv_buf, 
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in data.items()}


# --------------------------
# Fuz-PPOL Algorithm Implementation
# --------------------------
def fuzzyppolag(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    level=10,
    mode='ANFIS',
    env_name='fuzzytest',
    device="cuda:0",
    task="stab",
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    fuzzy_lr=1e-4,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    repeat_times=1,
    logger_kwargs=dict(),
    fuz_freq=2,
    save_freq=100,
    eps=0.1,
    train_eps=1.0,
    cost_limit=25.0,
):
    warnings.filterwarnings('ignore')
    # Setup MPI and PyTorch
    setup_pytorch_for_mpi()
    # Logger configuration with SwanLab support
    swanlab_logger = logger_kwargs.pop('swanlab_logger', None)
    if swanlab_logger and swanlab_logger.enabled:
        try:
            # Import here to avoid circular imports
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            from scripts.train import SwanLabEpochLogger
            logger = SwanLabEpochLogger(swanlab_logger=swanlab_logger, **logger_kwargs)
        except ImportError:
            # Fallback to standard logger if import fails (e.g., in multi-process mode)
            logger = EpochLogger(**logger_kwargs)
    else:
        logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Create fuzzy logic system (Unified FuzzyNet)
    fuzzy_logic_system = FuzzyNet(
        state_dim=obs_dim[0],
        n_singletons=level,
        hidden_dim=32,
        use_softmax=True,
        device=device,
    ).to(device)
    # Perturbation sampler for generating neighborhood perturbations s'

    perturbation_sampler = PerturbationSampler(strategy='stratified', state_dim=obs_dim[0])
    # perturbation_sampler = PerturbationSampler(strategy='uniform', state_dim=obs_dim[0])
    # Sync parameters across processes
    sync_params(ac)
    sync_params(fuzzy_logic_system)

    # Count parameters
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, ac.vc])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\t vc: %d,'%var_counts)
    var_count_fuzzy = core.count_vars(fuzzy_logic_system)
    logger.log('Number of parameters in fuzzy logic system: %d\n' % var_count_fuzzy)

    # Setup experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = FuzBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, outlevel=level, device=device)

    # Calculate PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'] ,data['logp']
        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']
        # cost_limit = 25.0
        
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()

        loss_cpi = ratio*cadv
        loss_cpi = loss_cpi.mean()
        
        p = softplus(penalty_param)
        penalty_item = p.item()

        # Balance reward and safety constraints (Paper Appendix B.1)
        pi_objective = loss_rpi - penalty_item*loss_cpi
        pi_objective = pi_objective/(1+penalty_item)
        loss_pi = -pi_objective

        cost_deviation = (cur_cost - cost_limit)
        # Additional information
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Calculate value loss
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        return ((ac.v(obs) - ret)**2).mean(), ((ac.vc(obs) - cret)** 2).mean()
    

    # # Calculate fuzzy loss (based on fuzzy Bellman operator)
    def compute_loss_fuzzy(data):
        # Use conditional measure g(s) from next state only (no per-perturb input)
        r, c, t_ret, t_cret = data['rew'], data['crew'], data['t_ret'], data['t_cret']
        v_disturb, vc_disturb = data['v_disturb'], data['vc_disturb']
        next_o = data['next_obs']  # [batch, state_dim]

        # fuzzy_measure: [batch, K]
        fuzzy_measure = fuzzy_logic_system(next_o)  # [batch, K]
        fuzzy_measure = torch.nan_to_num(fuzzy_measure, nan=1e-4, posinf=0.99, neginf=1e-4)

        fuzzy_v_next = choquet_integral(fuzzy_measure, v_disturb)  # [batch, 1] choquet_integral_dual

        fuzzy_vc_next = choquet_integral_dual(fuzzy_measure, vc_disturb)  # [batch, 1]

        fuzzy_v_next = torch.nan_to_num(fuzzy_v_next, nan=0.0, posinf=1e5, neginf=-1e5)
        fuzzy_vc_next = torch.nan_to_num(fuzzy_vc_next, nan=0.0, posinf=1e5, neginf=-1e5)

        fuzzy_v_next = torch.clamp(fuzzy_v_next, min=-1e5, max=1e5)
        fuzzy_vc_next = torch.clamp(fuzzy_vc_next, min=-1e5, max=1e5)

        pred_v = r.unsqueeze(1) + gamma * fuzzy_v_next
        pred_c = c.unsqueeze(1) + gamma * fuzzy_vc_next

        target_v = t_ret.unsqueeze(1)
        target_c = t_cret.unsqueeze(1)

        loss_r = F.smooth_l1_loss(pred_v, target_v)
        loss_c = F.smooth_l1_loss(pred_c, target_c)
        
        fuzzy_entropy = -torch.sum(fuzzy_measure * torch.log(fuzzy_measure + 1e-8), dim=1).mean()
        
        total_loss = loss_r + loss_c
        total_loss = total_loss.mean()

        return total_loss, loss_r, loss_c, fuzzy_entropy, pred_v.mean().item(), target_v.mean().item()
    
    # Setup optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
    penalty = softplus(penalty_param)
    
    penalty_lr = 5e-2
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    
    # Lower learning rate for fuzzy network than value network (Paper Section 5.2)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)
    fuzzy_optimizer = Adam(fuzzy_logic_system.parameters(), lr=fuzzy_lr)

    # Model saving setup
    logger.setup_pytorch_saver(ac)

    def update(epoch):
        try:
            cur_cost = logger.get_stats('EpRisk')[0]
        except (KeyError, IndexError):
            # If no EpRisk data yet, use default cost limit
            cur_cost = cost_limit
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)

        # Update penalty parameter
        loss_penalty = penalty_param * cost_dev 
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()

        data['cur_penalty'] = penalty_param

        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item() 

        # Train policy
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, _, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d (KL exceeds threshold)'%i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function and fuzzy system learning (fuzzy system updated less frequently)
        for i in range(train_v_iters):
            # Update value networks
            loss_v, loss_vc = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

            # Update fuzzy system (lower frequency)
            if epoch % fuz_freq == 0:
                loss_fuzzy, loss_fuzzy_r, loss_fuzzy_c, fuzzy_entropy, pred_v, target_v = compute_loss_fuzzy(data)
                fuzzy_optimizer.zero_grad()
                loss_fuzzy.backward()
                mpi_avg_grads(fuzzy_logic_system)
                fuzzy_optimizer.step()
            else:
                # Only calculate loss for logging
                with torch.no_grad():
                    loss_fuzzy, loss_fuzzy_r, loss_fuzzy_c, fuzzy_entropy, pred_v, target_v = compute_loss_fuzzy(data)

        # Record update information
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossFuzzy=loss_fuzzy.item(),
                     LossFuzzyR=loss_fuzzy_r.item(), LossFuzzyC=loss_fuzzy_c.item(),
                     FuzzyEntropy=fuzzy_entropy.item(), FuzzyPredV=pred_v, FuzzyTargetV=target_v,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
    
    # Main loop for interacting with the environment
    start_time = time.time()
    o, ep_ret, ep_cret, ep_len = env.reset()[0], 0, 0, 0
    
    best_ep_ret = -np.inf
    best_epoch = -1

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
            

            if env_name in ['DoubleIntegrator']:
                truncated = False
                next_o, r, d, infos = env.step(a)
                c = infos['constraint_violation']
            elif env_name in ["SafetyPointGoal1-v0", "SafetyPointButton1-v0", "SafetyPointPush1-v0", "SafetyPointCircle1-v0", 'SafetyWalker2dVelocity-v1', 'SafetyHalfCheetahVelocity-v1', 'SafetyHopperVelocity-v1', 'SafetySwimmerVelocity-v1']:
                next_o, r, c, d, truncated, infos = env.step(a)
            elif env_name in ['CartPole', 'QuadRotor', 'QuadRotor3D']:
                truncated = False
                next_o, r, d, infos = env.step(a)
                c = infos['constraint_violation']

            batch_size = level * repeat_times
            next_o_np = next_o.cpu().numpy() if isinstance(next_o, torch.Tensor) else next_o
            next_obs_batch_np = np.repeat(next_o_np.reshape(1, -1), batch_size, axis=0)

            
            next_obs_batch_perb_np = perturbation_sampler.sample(
                state=next_o_np,
                n_levels=level,
                n_samples_per_level=repeat_times,
                eps_max=eps,
            )
            
            with torch.no_grad():
                _, next_v_np, next_vc_np, _ = ac.step(
                    torch.as_tensor(next_obs_batch_perb_np, dtype=torch.float32).to(device)
                )
            
            next_v_distrib_np = next_v_np.reshape(level, repeat_times).mean(axis=1)  # [K]
            next_vc_distrib_np = next_vc_np.reshape(level, repeat_times).mean(axis=1)  # [K]
            
            with torch.no_grad():
                next_obs_batch = torch.as_tensor(next_o_np, dtype=torch.float32).unsqueeze(0).to(device)
                fuzzy_measure = fuzzy_logic_system(next_obs_batch)  # [1, K]

                fuzzy_v_next_np = choquet_integral(
                    fuzzy_measure,
                    torch.tensor(next_v_distrib_np).unsqueeze(0).to(device),
                ).squeeze().item()
                
                fuzzy_vc_next_np = choquet_integral_dual(
                    fuzzy_measure,
                    torch.tensor(next_vc_distrib_np).unsqueeze(0).to(device),
                ).squeeze().item()

            ep_ret += r
            ep_cret += c
            ep_len += 1

            buf.store(
                obs=o, 
                next_obs=next_o,
                act=a, 
                rew=r, 
                crew=c, 
                val=v, 
                cval=vc, 
                v_disturb=next_v_distrib_np,      # [K]
                vc_disturb=next_vc_distrib_np,    # [K]
                fvnext=fuzzy_v_next_np, 
                fcvnext=fuzzy_vc_next_np, 
                logp=logp
            )
            
            logger.store(VVals=v)
            logger.store(CVVals=vc)
            
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout or truncated
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory truncated at {ep_len} steps', flush=True)
                
                if timeout or epoch_ended:
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v = 0
                    vc = 0
                buf.finish_path(last_val=v, last_cval=vc)
                
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRisk=ep_cret)
                
                o, ep_ret, ep_cret, ep_len = env.reset()[0], 0, 0, 0
        
        try:
            current_ep_ret = logger.get_stats('EpRet')[0]
        except (KeyError, IndexError):
            current_ep_ret = -float('inf')
        if current_ep_ret > best_ep_ret:
            logger.save_state({'env': env}, itr='best')
            best_ep_ret = current_ep_ret
            if proc_id() == 0:  
                print(f'✓ New best model saved at epoch {epoch} with EpRet={best_ep_ret:.3f}')
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, epoch)

        update(epoch)
        
        logger.log_tabular('Epoch', epoch)
        try:
            logger.log_tabular('EpRet', average_only=True)
        except KeyError:
            pass  # No episode completed this epoch
        try:
            logger.log_tabular('EpRisk', average_only=True)
        except KeyError:
            pass  # No episode completed this epoch
        try:
            logger.log_tabular('EpLen', average_only=True)
        except KeyError:
            pass  # No episode completed this epoch
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossFuzzy', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('LossFuzzyR', average_only=True)
        logger.log_tabular('LossFuzzyC', average_only=True)
        logger.log_tabular('FuzzyEntropy', average_only=True)
        logger.log_tabular('FuzzyPredV', average_only=True)
        logger.log_tabular('FuzzyTargetV', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        

"""
Quick check for FuzzyNet module when running this file directly.
"""
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    state_dim, K = 4, 10
    model = FuzzyNet(state_dim=state_dim, n_singletons=K, device=device, use_softmax=True)
    # Test input
    state = torch.randn(32, state_dim, device=device)
    g = model(state)
    print(f"FuzzyNet g shape: {g.shape}, sample row sum: {g[0].sum().item():.4f}")