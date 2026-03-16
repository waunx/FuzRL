"""
FuzzyNet: Unified Fuzzy Network for Robust RL with Choquet Integral

This module provides:
1. Neural Fuzzy Network (NFN) for state-conditioned fuzzy densities
2. Sugeno λ-fuzzy measure computation
3. Choquet integral aggregation (standard and dual)
4. Utility functions for robust value estimation

Author: Xu Wan
Date: 2024-11-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FuzzyNet(nn.Module):
    """
    Neural Fuzzy Network for generating state-conditioned singleton fuzzy densities.
    
    This network maps state observations to K fuzzy density values g_i ∈ (0, 1),
    which parameterize a Sugeno λ-fuzzy measure for Choquet integral aggregation.
    
    Args:
        state_dim: Dimension of state observation
        n_singletons: Number of fuzzy singletons (K perturbation levels)
        hidden_dim: Hidden layer dimension
        use_softmax: If True, apply softmax to normalize g_i; otherwise use sigmoid
        device: Computation device
    """
    
    def __init__(self, state_dim, n_singletons=10, hidden_dim=64, 
                 use_softmax=False, device="cuda:0"):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_singletons = n_singletons
        self.use_softmax = use_softmax
        self.device = device
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Density head for generating fuzzy densities
        self.density_head = nn.Linear(hidden_dim, n_singletons)
        
        # Small epsilon to avoid numerical issues
        self._eps = 1e-6
        
        self.to(device)
    
    def forward(self, state):
        """
        Generate fuzzy densities from state.
        
        Args:
            state: [batch_size, state_dim] or [batch_size, K, state_dim]
            
        Returns:
            g: [batch_size, K] fuzzy densities in (ε, 1-ε)
        """
        state = state.to(self.device)
        
        # Handle batch input
        original_shape = state.shape
        if len(original_shape) == 3:  # [B, K, D]
            B, K, D = original_shape
            state = state.view(-1, D)  # [B*K, D]
        
        # Encode and generate logits
        h = self.encoder(state)  # [B(*K), hidden_dim]
        logits = self.density_head(h)  # [B(*K), n_singletons]
        
        # Apply activation
        if self.use_softmax:
            # Softmax normalization: sum to 1
            g = F.softmax(logits, dim=-1)
            # Add small noise to avoid exact 0 or 1
            g = g * (1 - 2 * self._eps) + self._eps
        else:
            # Sigmoid: each g_i ∈ (0, 1) independently
            g = torch.sigmoid(logits)
            g = g * (1 - 2 * self._eps) + self._eps
        
        # Reshape if needed
        if len(original_shape) == 3:
            g = g.view(B, K, self.n_singletons)
            # If input is [B, K, D], aggregate over K dimension
            g = g.mean(dim=1)  # [B, n_singletons]
        
        return g



def solve_sugeno_lambda(g: torch.Tensor, max_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
    """
    Solve for the Sugeno lambda parameter using Newton's method.
    
    Args:
        g: Fuzzy singleton densities with shape [batch_size, K].
           Each g_i represents the measure of singleton {x_i}.
        max_iter: Maximum number of Newton iterations.
        tol: Convergence tolerance for early stopping.
    
    Returns:
        Lambda parameter with shape [batch_size, 1].
    """
    if g.dim() == 3:
        g = g.squeeze(-1)
    
    batch_size, K = g.shape
    device = g.device
    
    g_sum = g.sum(dim=1, keepdim=True)
    
    # Initialize lambda based on sum of densities
    lam = torch.where(
        torch.abs(g_sum - 1.0) < 0.01,
        torch.zeros_like(g_sum),
        torch.where(
            g_sum < 1.0,
            torch.full_like(g_sum, 0.5),   # superadditive case
            torch.full_like(g_sum, -0.5)   # subadditive case
        )
    )
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        # f(lambda) = prod(1 + lambda * g_i) - 1 - lambda
        terms = 1.0 + lam * g
        terms = torch.clamp(terms, min=1e-8)
        
        prod = terms.prod(dim=1, keepdim=True)
        f = prod - 1.0 - lam
        
        # f'(lambda) = prod * sum(g_i / (1 + lambda * g_i)) - 1
        # Using log-sum-exp for numerical stability
        log_terms = torch.log(terms)
        sum_log = log_terms.sum(dim=1, keepdim=True)
        f_prime = (g * torch.exp(sum_log.expand_as(g) - log_terms)).sum(dim=1, keepdim=True) - 1.0
        
        # Newton update
        delta = f / (f_prime + 1e-10)
        lam = lam - delta
        
        # Enforce constraint: lambda > -1
        lam = torch.clamp(lam, min=-0.99, max=10.0)
        
        # Check convergence
        if torch.abs(delta).max().item() < tol:
            break
    
    return lam


def sugeno_measure(g: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative Sugeno lambda-measures for nested sets.
    
    Complexity:
    Time: O(K), single pass through elements
    Space: O(K) for storing intermediate measures
    
    Args:
        g: Singleton densities with shape [batch_size, K], ordered according
           to the sorted values (typically descending for Choquet integral).
        lam: Lambda parameter with shape [batch_size, 1].
    
    Returns:
        Cumulative measures with shape [batch_size, K], where output[:, k]
        represents mu(A_{k+1}) = mu({x_1, ..., x_{k+1}}).
    """
    batch_size, K = g.shape
    
    mu_list = []
    
    # Base case: mu(A_1) = mu({x_1}) = g_1
    mu = g[:, 0:1].clone()
    mu_list.append(mu)
    
    # Recursive case: mu(A_k) = mu(A_{k-1}) + g_k + lambda * mu(A_{k-1}) * g_k
    for k in range(1, K):
        g_k = g[:, k:k+1]
        mu = mu + g_k + lam * mu * g_k
        mu = torch.clamp(mu, min=0.0, max=10.0)  # numerical stability
        mu_list.append(mu)
    
    return torch.cat(mu_list, dim=1)


def compute_fuzzy_measure(g: torch.Tensor, lam: torch.Tensor, 
                          subset_indices: list = None) -> torch.Tensor:
    """
    Compute Sugeno lambda-fuzzy measure for an arbitrary subset.

    Args:
        g: Singleton densities with shape [batch_size, K].
        lam: Lambda parameter with shape [batch_size, 1].
        subset_indices: List of indices defining the subset. If None, computes
                        mu(X) for the full set.
    
    Returns:
        Fuzzy measure of the subset with shape [batch_size, 1].
    """
    if subset_indices is None:
        subset_indices = list(range(g.shape[1]))
    
    if len(subset_indices) == 0:
        return torch.zeros(g.shape[0], 1, device=g.device)
    
    mu = g[:, subset_indices[0]:subset_indices[0]+1]
    
    for idx in subset_indices[1:]:
        g_i = g[:, idx:idx+1]
        mu = g_i + mu + lam * g_i * mu
        mu = torch.clamp(mu, min=0.0)
    
    return mu


def choquet_integral(g: torch.Tensor, values: torch.Tensor, 
                     use_lambda: bool = True) -> torch.Tensor:
    """
    Compute the discrete Choquet integral with Sugeno lambda-measure.
    
    The Choquet integral is a nonlinear aggregation operator that generalizes
    the weighted average by allowing interactions between criteria. Using the
    descending order formulation:
    
        C_mu(f) = sum_{k=1}^{K} [f_(k) - f_(k+1)] * mu(A_k)
    
    where:
        - f_(1) >= f_(2) >= ... >= f_(K) are values sorted in descending order
        - A_k = {x_(1), x_(2), ..., x_(k)} is the set of indices with top-k values
        - f_(K+1) = 0 by convention
        - mu is a Sugeno lambda-fuzzy measure
    
    Args:
        g: Fuzzy singleton densities with shape [batch_size, K] or [batch_size, K, 1].
           Values should be in (0, 1).
        values: Values to aggregate with shape [batch_size, K] or [batch_size, K, 1].
        use_lambda: If True, compute lambda via Sugeno's equation.
                    If False, use additive measure (lambda = 0).
    
    Returns:
        Aggregated values with shape [batch_size, 1].
    """
    # Handle 3D input tensors
    if g.dim() == 3:
        g = g.squeeze(-1)
    if values.dim() == 3:
        values = values.squeeze(-1)
    
    batch_size, K = values.shape
    device = values.device
    
    # Handle non-finite values gracefully
    if not torch.isfinite(values).all() or not torch.isfinite(g).all():
        return values.nan_to_num().mean(dim=1, keepdim=True)
    
    # Sort values in descending order: f_(1) >= f_(2) >= ... >= f_(K)
    sorted_values, sort_idx = torch.sort(values, dim=1, descending=True)
    
    # Reorder densities to match sorted values
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, K)
    sorted_g = g[batch_idx, sort_idx]
    sorted_g = torch.clamp(sorted_g, min=1e-6, max=1.0 - 1e-6)
    
    # Compute lambda parameter for Sugeno measure
    if use_lambda:
        lam = solve_sugeno_lambda(sorted_g)
    else:
        lam = torch.zeros(batch_size, 1, device=device)
    
    # Compute cumulative measures: mu(A_k) where A_k = {x_(1), ..., x_(k)}
    # A_1 = {x_(1)}, A_2 = {x_(1), x_(2)}, ..., A_K = {x_(1), ..., x_(K)}
    mu_tensor = sugeno_measure(sorted_g, lam)  # [batch_size, K]
    
    # Compute value differences: f_(k) - f_(k+1), with f_(K+1) = 0
    f_next = torch.cat([
        sorted_values[:, 1:],
        torch.zeros(batch_size, 1, device=device)
    ], dim=1)
    value_diffs = sorted_values - f_next  # [batch_size, K], all non-negative
    
    # Choquet integral: sum of (value_diff * measure)
    integral = (value_diffs * mu_tensor).sum(dim=1, keepdim=True)
    integral = torch.nan_to_num(integral, nan=0.0)
    
    return integral


def choquet_integral_dual(g: torch.Tensor, values: torch.Tensor, 
                          use_lambda: bool = True) -> torch.Tensor:
    """
    Compute the dual (conjugate) Choquet integral.
    
    The dual Choquet integral uses the conjugate measure mu* defined as:
        mu*(A) = mu(X) - mu(A^c)
    
    where A^c denotes the complement of A with respect to the universal set X.
    
    The dual integral is computed as:
        C*_mu(f) = sum_{k=1}^{K} [f_(k) - f_(k+1)] * mu*(A_k)
    
    Args:
        g: Fuzzy singleton densities with shape [batch_size, K] or [batch_size, K, 1].
        values: Values to aggregate with shape [batch_size, K] or [batch_size, K, 1].
        use_lambda: If True, use Sugeno lambda-measure; if False, use additive measure.
    
    Returns:
        Aggregated values with shape [batch_size, 1].
    """
    # Handle 3D input tensors
    if g.dim() == 3:
        g = g.squeeze(-1)
    if values.dim() == 3:
        values = values.squeeze(-1)
    
    batch_size, K = values.shape
    device = values.device
    
    # Handle non-finite values gracefully
    if not torch.isfinite(values).all() or not torch.isfinite(g).all():
        return values.nan_to_num().mean(dim=1, keepdim=True)
    
    # Sort values in descending order
    sorted_values, sort_idx = torch.sort(values, dim=1, descending=True)
    
    # Reorder densities to match sorted values
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, K)
    sorted_g = g[batch_idx, sort_idx]
    sorted_g = torch.clamp(sorted_g, min=1e-6, max=1.0 - 1e-6)
    
    # Compute lambda parameter
    if use_lambda:
        lam = solve_sugeno_lambda(sorted_g)
    else:
        lam = torch.zeros(batch_size, 1, device=device)
    
    # Compute mu(X) - measure of the universal set (last element of cumulative)
    mu_X = sugeno_measure(sorted_g, lam)[:, -1:]  # [batch_size, 1]
    
    # Compute mu(A_k^c) for complement sets
    # A_k^c = {x_(k+1), ..., x_(K)} contains the (K-k) smallest elements
    # 
    # Strategy: Reverse g and compute cumulative measures from the tail.
    # If sorted_g = [g_(1), g_(2), ..., g_(K)] (descending by value),
    # then reversed_g = [g_(K), g_(K-1), ..., g_(1)]
    # and mu_reversed[:, j] = mu({x_(K), x_(K-1), ..., x_(K-j)})
    reversed_g = torch.flip(sorted_g, dims=[1])
    mu_reversed = sugeno_measure(reversed_g, lam)  # [batch_size, K]
    
    # Map reversed measures to complement measures:
    # mu(A_k^c) = mu({x_(k+1), ..., x_(K)}) = measure of (K-k) tail elements
    # For k=1: A_1^c = {x_(2), ..., x_(K)}, size = K-1
    # For k=K: A_K^c = {}, size = 0, mu = 0
    mu_complements = []
    for k in range(K):
        complement_size = K - k - 1
        if complement_size <= 0:
            # A_K^c = empty set
            mu_complements.append(torch.zeros(batch_size, 1, device=device))
        else:
            # mu_reversed[:, complement_size - 1] gives measure of complement_size elements
            mu_complements.append(mu_reversed[:, complement_size - 1: complement_size])
    
    mu_comp_tensor = torch.cat(mu_complements, dim=1)  # [batch_size, K]
    
    # Dual measure: mu*(A_k) = mu(X) - mu(A_k^c)
    mu_dual = mu_X - mu_comp_tensor  # [batch_size, K]
    
    # Compute value differences: f_(k) - f_(k+1), with f_(K+1) = 0
    f_next = torch.cat([
        sorted_values[:, 1:],
        torch.zeros(batch_size, 1, device=device)
    ], dim=1)
    value_diffs = sorted_values - f_next
    
    # Dual Choquet integral
    integral = (value_diffs * mu_dual).sum(dim=1, keepdim=True)
    integral = torch.nan_to_num(integral, nan=0.0)
    
    return integral


class PerturbationSampler:
    """
    Advanced perturbation sampler for generating meaningful state perturbations.
    
    Supports multiple sampling strategies:
    - uniform: Uniform random noise in range
    - gaussian: Gaussian noise with adaptive variance
    - adversarial: Gradient-based adversarial perturbations
    - importance: Importance sampling based on state sensitivity
    """
    
    def __init__(self, strategy='stratified', state_dim=None):
        """
        Args:
            strategy: Sampling strategy ('uniform', 'gaussian', 'stratified', 'adaptive')
            state_dim: Dimension of state space
        """
        self.strategy = strategy
        self.state_dim = state_dim
    
    def sample(self, state, n_levels, n_samples_per_level, eps_max, 
               gradient_fn=None, **kwargs):
        """
        Sample perturbations around given state.
        
        Args:
            state: [state_dim] base state
            n_levels: Number of perturbation levels (K)
            n_samples_per_level: Samples per level (M)
            eps_max: Maximum perturbation magnitude
            gradient_fn: Optional function to compute state gradients
            
        Returns:
            perturbations: [K*M, state_dim] perturbed states
        """
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        else:
            state_np = np.array(state)
        
        if self.strategy == 'uniform':
            return self._sample_uniform(state_np, n_levels, n_samples_per_level, eps_max)
        elif self.strategy == 'gaussian':
            return self._sample_gaussian(state_np, n_levels, n_samples_per_level, eps_max)
        elif self.strategy == 'stratified':
            return self._sample_stratified(state_np, n_levels, n_samples_per_level, eps_max)
        elif self.strategy == 'adaptive':
            return self._sample_adaptive(state_np, n_levels, n_samples_per_level, eps_max, gradient_fn)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _sample_uniform(self, state, n_levels, n_samples, eps_max):
        """Uniform sampling in layered balls."""
        eps_levels = np.linspace(0, eps_max, n_levels + 1)[1:]
        
        perturbations = []
        for eps in eps_levels:
            for _ in range(n_samples):
                noise = np.random.randn(len(state))
                noise = noise / (np.linalg.norm(noise) + 1e-8) * eps * np.random.rand()
                perturbations.append(state + noise)
        
        return np.array(perturbations)
    
    def _sample_gaussian(self, state, n_levels, n_samples, eps_max):
        """Gaussian sampling with increasing variance."""
        perturbations = []
        
        for level in range(n_levels):
            sigma = eps_max * (level + 1) / n_levels
            for _ in range(n_samples):
                noise = np.random.randn(len(state)) * sigma
                perturbations.append(state + noise)
        
        return np.array(perturbations)
    
    def _sample_stratified(self, state, n_levels, n_samples, eps_max):
        """Stratified sampling in concentric shells."""
        perturbations = []
        eps_bounds = np.linspace(0, eps_max, n_levels + 1)
        
        for i in range(n_levels):
            eps_low, eps_high = eps_bounds[i], eps_bounds[i+1]
            
            for _ in range(n_samples):
                direction = np.random.randn(len(state))
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                radius = np.random.uniform(eps_low, eps_high)
                noise = direction * radius
                
                perturbations.append(state + noise)
        
        return np.array(perturbations)
    
    def _sample_adaptive(self, state, n_levels, n_samples, eps_max, gradient_fn):
        """Adaptive sampling based on gradient/sensitivity."""
        if gradient_fn is None:
            return self._sample_stratified(state, n_levels, n_samples, eps_max)
        
        grad = gradient_fn(state)
        if isinstance(grad, torch.Tensor):
            grad = grad.cpu().numpy()
        
        importance = np.abs(grad) + 1e-8
        importance = importance / importance.sum()
        
        perturbations = []
        eps_bounds = np.linspace(0, eps_max, n_levels + 1)
        
        for i in range(n_levels):
            eps_low, eps_high = eps_bounds[i], eps_bounds[i+1]
            
            for _ in range(n_samples):
                noise = np.random.randn(len(state)) * np.sqrt(importance)
                noise = noise / (np.linalg.norm(noise) + 1e-8)
                radius = np.random.uniform(eps_low, eps_high)
                
                perturbations.append(state + noise * radius)
        
        return np.array(perturbations)


# ==============================================================================
# Unit Tests
# ==============================================================================

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("FuzzyNet Module Test Suite")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Test FuzzyNet
    # -------------------------------------------------------------------------
    print("\n[Test 1] FuzzyNet Forward Pass")
    print("-" * 50)
    
    state_dim, K = 10, 5
    fuzzy_net = FuzzyNet(state_dim, n_singletons=K, use_softmax=False, device=device)
    
    test_states = torch.randn(3, state_dim).to(device)
    g = fuzzy_net(test_states)
    
    print(f"Input shape: {test_states.shape}")
    print(f"Output shape: {g.shape}")
    print(f"Density range: [{g.min().item():.4f}, {g.max().item():.4f}]")
    print(f"Density sum per sample: {g.sum(dim=1).tolist()}")
    
    # -------------------------------------------------------------------------
    # Test Choquet Integral (additive measure)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Choquet Integral (Additive Measure, lambda=0)")
    print("-" * 50)
    
    g_test = torch.tensor([[0.2, 0.5, 0.3]]).to(device)
    values_test = torch.tensor([[1.0, 5.0, 3.0]]).to(device)
    
    result = choquet_integral(g_test, values_test, use_lambda=False)
    expected = 3.6
    
    print(f"g = {g_test.tolist()}")
    print(f"values = {values_test.tolist()}")
    print(f"Computed: {result.item():.4f}")
    print(f"Expected: {expected:.4f}")
    print(f"Status: {'PASS' if abs(result.item() - expected) < 1e-4 else 'FAIL'}")
    
    # -------------------------------------------------------------------------
    # Test Choquet Integral (non-additive measure)
    # -------------------------------------------------------------------------
    print("\n[Test 3] Choquet Integral (Sugeno Lambda-Measure)")
    print("-" * 50)
    
    g_nonadditive = torch.tensor([[0.4, 0.5, 0.4]]).to(device)  # sum > 1
    values_nonadditive = torch.tensor([[1.0, 5.0, 3.0]]).to(device)
    
    lam = solve_sugeno_lambda(g_nonadditive)
    result_nonadditive = choquet_integral(g_nonadditive, values_nonadditive, use_lambda=True)
    
    print(f"g = {g_nonadditive.tolist()}, sum = {g_nonadditive.sum().item():.2f}")
    print(f"Computed lambda: {lam.item():.6f} (expected < 0 for sum > 1)")
    print(f"Choquet integral: {result_nonadditive.item():.4f}")
    
    # Verify lambda satisfies characteristic equation
    prod = torch.prod(1 + lam * g_nonadditive, dim=1)
    residual = (prod - 1 - lam).abs().item()
    print(f"Lambda equation residual: {residual:.2e}")
    print(f"Status: {'PASS' if residual < 1e-5 and lam.item() < 0 else 'FAIL'}")
    
    # -------------------------------------------------------------------------
    # Test Dual Choquet Integral
    # -------------------------------------------------------------------------
    print("\n[Test 4] Dual Choquet Integral")
    print("-" * 50)
    
    result_std = choquet_integral(g_test, values_test, use_lambda=False)
    result_dual = choquet_integral_dual(g_test, values_test, use_lambda=False)
    
    print(f"Standard Choquet (lambda=0): {result_std.item():.4f}")
    print(f"Dual Choquet (lambda=0): {result_dual.item():.4f}")
    print(f"Difference: {abs(result_std.item() - result_dual.item()):.2e}")
    print(f"Status: {'PASS' if abs(result_std.item() - result_dual.item()) < 1e-4 else 'FAIL'}")
    print("(Expected: equal for additive measure)")
    
    # -------------------------------------------------------------------------
    # Test with non-additive measure (standard vs dual should differ)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Standard vs Dual (Non-Additive Measure)")
    print("-" * 50)
    
    result_std_na = choquet_integral(g_nonadditive, values_nonadditive, use_lambda=True)
    result_dual_na = choquet_integral_dual(g_nonadditive, values_nonadditive, use_lambda=True)
    
    print(f"Standard Choquet: {result_std_na.item():.4f}")
    print(f"Dual Choquet: {result_dual_na.item():.4f}")
    print(f"Difference: {abs(result_std_na.item() - result_dual_na.item()):.4f}")
    differs = abs(result_std_na.item() - result_dual_na.item()) > 0.01
    print(f"Status: {'PASS' if differs else 'FAIL'} (expected: different for non-additive)")
    
    # -------------------------------------------------------------------------
    # Test batch processing
    # -------------------------------------------------------------------------
    print("\n[Test 6] Batch Processing")
    print("-" * 50)
    
    batch_g = torch.tensor([
        [0.2, 0.5, 0.3],
        [0.33, 0.33, 0.34],
        [0.1, 0.6, 0.3]
    ]).to(device)
    batch_values = torch.tensor([
        [1.0, 5.0, 3.0],
        [2.0, 2.0, 2.0],
        [0.0, 10.0, 5.0]
    ]).to(device)
    
    batch_result = choquet_integral(batch_g, batch_values, use_lambda=False)
    
    print(f"Batch size: {batch_values.shape[0]}")
    print(f"Results: {batch_result.squeeze().tolist()}")
    print(f"Uniform values [2,2,2] -> {batch_result[1].item():.4f} (expected: 2.0)")
    print(f"Status: {'PASS' if abs(batch_result[1].item() - 2.0) < 1e-4 else 'FAIL'}")
    
    # -------------------------------------------------------------------------
    # Test PerturbationSampler
    # -------------------------------------------------------------------------
    print("\n[Test 7] PerturbationSampler")
    print("-" * 50)
    
    sampler = PerturbationSampler(strategy='stratified', state_dim=state_dim)
    base_state = np.random.randn(state_dim)
    
    perturbed = sampler.sample(base_state, n_levels=3, n_samples_per_level=2, eps_max=0.1)
    distances = np.linalg.norm(perturbed - base_state, axis=1)
    
    print(f"Generated {len(perturbed)} perturbed states")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"Status: {'PASS' if distances.max() <= 0.1 + 1e-6 else 'FAIL'}")
    
    print("\n" + "=" * 70)
    print("All Tests Completed!")
    print("=" * 70)