import numpy as np
import sys
import os
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.cvarppolag.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
torch.autograd.set_detect_anomaly(True)
import safety_gymnasium
from spinup.utils.run_utils import setup_logger_kwargs


def compute_cvar(values, alpha=0.1):
    """Compute lower-tail CVaR (expected worst alpha fraction).
    values: 1D numpy or torch array shape (K,)
    """
    if isinstance(values, torch.Tensor):
        vals = values.detach().cpu().numpy()
    else:
        vals = np.array(values)
    K = vals.size
    if K == 0:
        return 0.0
    k = max(1, int(np.ceil(alpha * K)))
    sorted_vals = np.sort(vals)
    worst = sorted_vals[:k]
    return float(worst.mean())

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97, device="cuda:0"):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)
        
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

        #buf.store(   o, a, r, c, v,vc, logp)
    def store(self, obs, act, rew, crew, val,cval, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) #/ adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
                    adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in data.items()}



def cvarppolag(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, level=10, alpha=0.1, env_name="fuzzytest", device="cuda:0", task="stab",
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=100, cost_limit=25.0):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
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

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, device=device)

    # Set up function for computing PPO policy loss
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

        # clip_cadv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * cadv
        # loss_cpi = (torch.min(ratio * cadv, clip_cadv)).mean()
        loss_cpi = ratio*cadv
        loss_cpi = loss_cpi.mean()
        
        p = softplus(penalty_param)
        penalty_item = p.item()
      
        pi_objective = loss_rpi - penalty_item*loss_cpi
        pi_objective = pi_objective/(1+penalty_item)
        loss_pi = -pi_objective

        cost_deviation = (cur_cost - cost_limit)
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).to(device).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        # Use CVaR-style aggregation - focus on worst alpha fraction of returns
        v_pred = ac.v(obs)
        vc_pred = ac.vc(obs)

        # Simple CVaR implementation: sort returns and average worst alpha fraction
        ret_sorted = torch.sort(ret)[0]
        cret_sorted = torch.sort(cret)[0]
        k = max(1, int(alpha * len(ret_sorted)))
        ret_cvar = ret_sorted[:k].mean()
        cret_cvar = cret_sorted[:k].mean()

        return ((v_pred - ret_cvar)**2).mean(), ((vc_pred - cret_cvar)**2).mean()
    # Set up optimizers for policy and value function
    # pi_lr = 3e-4
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = torch.tensor(1.0,requires_grad=True).float()
    penalty = softplus(penalty_param)
    

    penalty_lr = 5e-2
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    # vf_lr = 1e-3
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(),lr=vf_lr)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        cur_cost = logger.get_stats('EpRisk')[0]
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)

        loss_penalty = penalty_param * cost_dev 
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()

        data['cur_penalty'] = penalty_param

        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item() 

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, _, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            loss_v, loss_vc = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)   # average grads across MPI processes
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossVc=cv_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        
    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_cret, ep_len = env.reset()[0], 0, 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
            if env_name in ['DoubleIntegrator']:
                truncated = False
                next_o, r, d, infos = env.step(a)
                c = infos['constraint_violation']
            elif env_name in ['SafetyPointGoal1-v0', 'SafetyPointPush1-v0', 'SafetyPointButton1-v0', 'SafetyPointCircle1-v0', 'SafetyWalker2dVelocity-v1', 'SafetyHalfCheetahVelocity-v1', 'SafetyHopperVelocity-v1', 'SafetySwimmerVelocity-v1']:
                next_o, r, c, d, truncated, infos = env.step(a)
            elif env_name in ['CartPole', 'QuadRotor', 'QuadRotor3D']:
                truncated = False
                next_o, r, d, infos = env.step(a)
                c = infos['constraint_violation']
            ep_ret += r
            ep_cret += c
            ep_len += 1

            # save and log
            buf.store(o, a, r, c, v,vc, logp)
            logger.store(VVals=v)
            logger.store(CVVals=vc)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout or truncated
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                        _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v = 0
                    vc = 0
                buf.finish_path(last_val=v, last_cval=vc)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRisk=ep_cret)
                    o, ep_ret, ep_cret, ep_len = env.reset()[0], 0, 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpRisk',average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CVVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossVc', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
