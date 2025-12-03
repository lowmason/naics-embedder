# -------------------------------------------------------------------------------------------------
# PyTorch Compile Utilities
# -------------------------------------------------------------------------------------------------
'''
Utilities for torch.compile integration to fuse element-wise operations and improve throughput.

This module provides:
  - Configuration for torch.compile settings
  - Compiled versions of common Lorentz operations
  - Decorator utilities for conditional compilation

Usage:
    from naics_embedder.utils.compile import CompiledLorentzOps, get_compile_config

    # Get compiled operations (respects global config)
    ops = CompiledLorentzOps.get_instance()
    result = ops.exp_map_zero(tangent_vectors, c=1.0)

    # Or use standalone compiled functions
    from naics_embedder.utils.compile import compiled_exp_map_zero
    result = compiled_exp_map_zero(tangent_vectors, c=1.0)

Note: torch.compile requires PyTorch 2.0+. On older versions, operations fall back to eager mode.
'''

import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional, TypeVar

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Compile Configuration
# -------------------------------------------------------------------------------------------------

CompileMode = Literal['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']
CompileBackend = Literal['inductor', 'eager', 'aot_eager', 'cudagraphs']

@dataclass
class CompileConfig:
    '''Configuration for torch.compile behavior.

    Attributes:
        enabled: Whether to use torch.compile (requires PyTorch 2.0+)
        mode: Compilation mode - affects speed vs compile time tradeoff
            - 'default': Balanced mode
            - 'reduce-overhead': Best for small tensors and repeated calls
            - 'max-autotune': Maximum optimization, longer compile time
            - 'max-autotune-no-cudagraphs': Max optimization without CUDA graphs
        backend: Compilation backend
            - 'inductor': Default, best performance on most hardware
            - 'eager': No compilation (for debugging)
            - 'aot_eager': Ahead-of-time compilation with eager execution
            - 'cudagraphs': CUDA graphs optimization
        fullgraph: Whether to require full graph compilation (no graph breaks)
        dynamic: Whether to enable dynamic shape support
        disable_on_cpu: Disable compilation when running on CPU
        cache_size_limit: Maximum number of cached compilations per function
    '''

    enabled: bool = False
    mode: CompileMode = 'reduce-overhead'
    backend: CompileBackend = 'inductor'
    fullgraph: bool = False
    dynamic: bool = True  # Important for varying batch sizes
    disable_on_cpu: bool = False
    cache_size_limit: int = 64

    def __post_init__(self) -> None:
        # Check environment variables for overrides
        if os.environ.get('NAICS_DISABLE_COMPILE', '').lower() in ('1', 'true', 'yes'):
            self.enabled = False
            logger.info('torch.compile disabled via NAICS_DISABLE_COMPILE environment variable')

        # Check PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version < (2, 0):
            self.enabled = False
            logger.warning(
                f'torch.compile requires PyTorch 2.0+, found {torch.__version__}. '
                'Falling back to eager mode.'
            )

# Global configuration instance
_compile_config: Optional[CompileConfig] = None

def get_compile_config() -> CompileConfig:
    '''Get the global compile configuration.'''
    global _compile_config
    if _compile_config is None:
        _compile_config = CompileConfig()
    return _compile_config

def set_compile_config(config: CompileConfig) -> None:
    '''Set the global compile configuration.'''
    global _compile_config
    _compile_config = config

# -------------------------------------------------------------------------------------------------
# Compile Decorator
# -------------------------------------------------------------------------------------------------

F = TypeVar('F', bound=Callable)

def maybe_compile(
    mode: Optional[CompileMode] = None,
    fullgraph: Optional[bool] = None,
    dynamic: Optional[bool] = None,
) -> Callable[[F], F]:
    '''
    Decorator that conditionally applies torch.compile based on global config.

    Args:
        mode: Override compile mode (uses global config if None)
        fullgraph: Override fullgraph setting
        dynamic: Override dynamic shapes setting

    Returns:
        Decorated function that may be compiled

    Example:
        @maybe_compile(mode='reduce-overhead')
        def my_tensor_op(x: Tensor) -> Tensor:
            return x * 2 + 1
    '''

    def decorator(fn: F) -> F:
        config = get_compile_config()

        if not config.enabled:
            return fn

        compile_mode = mode if mode is not None else config.mode
        compile_fullgraph = fullgraph if fullgraph is not None else config.fullgraph
        compile_dynamic = dynamic if dynamic is not None else config.dynamic

        try:
            compiled_fn = torch.compile(
                fn,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                backend=config.backend,
            )
            logger.debug(f'Compiled function: {fn.__name__} with mode={compile_mode}')
            return compiled_fn  # type: ignore
        except Exception as e:
            logger.warning(f'Failed to compile {fn.__name__}: {e}. Using eager mode.')
            return fn

    return decorator

# -------------------------------------------------------------------------------------------------
# Compiled Lorentz Operations (Standalone Functions)
# -------------------------------------------------------------------------------------------------
# These are the core element-wise operations that benefit most from fusion.
# They're defined as standalone functions for maximum flexibility.

def _exp_map_zero_impl(v_spatial: Tensor, sqrt_c: Tensor) -> Tensor:
    '''Core exponential map computation - highly fusible element-wise ops.'''
    # Compute norm of tangent vector (spatial part)
    norm_v = torch.norm(v_spatial, p=2, dim=-1, keepdim=True)
    norm_v = torch.clamp(norm_v, min=1e-8)

    # Clamp to avoid overflow in sinh/cosh
    theta = torch.clamp(sqrt_c * norm_v, max=40.0)

    # Exponential map formula
    x0 = torch.cosh(theta) / sqrt_c
    sinh_term = torch.sinh(theta) / sqrt_c
    x_spatial = (sinh_term / norm_v) * v_spatial

    return torch.cat([x0, x_spatial], dim=-1)

def _log_map_zero_impl(x0: Tensor, x_spatial: Tensor, sqrt_c: Tensor) -> Tensor:
    '''Core logarithmic map computation - highly fusible element-wise ops.'''
    # Distance from origin
    theta = torch.acosh(torch.clamp(sqrt_c * x0, min=1.0 + 1e-5))

    # Scale factor
    sinh_theta = torch.sinh(theta)
    sinh_theta = torch.clamp(sinh_theta, min=1e-8)
    scale = theta / sinh_theta
    scale = torch.where(theta > 1e-8, scale, torch.ones_like(scale))

    # Tangent vector
    v_spatial = scale * x_spatial
    v_time = torch.zeros_like(x0)

    return torch.cat([v_time, v_spatial], dim=-1)

def _lorentz_distance_impl(uv_time: Tensor, uv_spatial_sum: Tensor, sqrt_c: Tensor) -> Tensor:
    '''Core Lorentz distance computation - highly fusible element-wise ops.'''
    # Lorentz inner product: sum of spatial - time
    dot_product = uv_spatial_sum - uv_time

    # Clamp for valid arccosh argument
    arccosh_arg = torch.clamp(-dot_product, min=1.0)

    # Distance
    return sqrt_c * torch.acosh(arccosh_arg)

def _minkowski_dot_impl(x: Tensor, y: Tensor) -> Tensor:
    '''Core Minkowski inner product - element-wise multiplication and reduction.'''
    xy = x * y
    return torch.sum(xy[..., 1:], dim=-1) - xy[..., 0]

def _project_to_hyperboloid_impl(spatial: Tensor, c_inv: float) -> Tensor:
    '''Project points onto hyperboloid - element-wise ops.'''
    spatial_norm_sq = torch.sum(spatial**2, dim=-1, keepdim=True)
    x0_new = torch.sqrt(spatial_norm_sq + c_inv)
    return torch.cat([x0_new, spatial], dim=-1)

def _sech_adaptive_margin_impl(time_coord: Tensor, sqrt_c: Tensor, base_margin: float) -> Tensor:
    '''Compute sech-based adaptive margin - element-wise ops.'''
    arg = sqrt_c * time_coord
    arg_clamped = torch.clamp(arg, min=1.0 + 1e-6)
    lorentz_norm = torch.acosh(arg_clamped)
    cosh_norms = torch.cosh(lorentz_norm)
    sech_norms = 1.0 / (cosh_norms + 1e-8)
    return base_margin * sech_norms

# Apply compilation to core implementations
_compiled_exp_map = maybe_compile(mode='reduce-overhead')(_exp_map_zero_impl)
_compiled_log_map = maybe_compile(mode='reduce-overhead')(_log_map_zero_impl)
_compiled_lorentz_dist = maybe_compile(mode='reduce-overhead')(_lorentz_distance_impl)
_compiled_minkowski_dot = maybe_compile(mode='reduce-overhead')(_minkowski_dot_impl)
_compiled_project = maybe_compile(mode='reduce-overhead')(_project_to_hyperboloid_impl)
_compiled_sech_margin = maybe_compile(mode='reduce-overhead')(_sech_adaptive_margin_impl)

# -------------------------------------------------------------------------------------------------
# CompiledLorentzOps - Drop-in replacement for LorentzOps
# -------------------------------------------------------------------------------------------------

class CompiledLorentzOps:
    '''
    Compiled static utility class for Lorentz model operations.

    This is a drop-in replacement for LorentzOps with torch.compile applied
    to fuse element-wise operations for better throughput.

    All methods are static and stateless.
    '''

    _instance: Optional['CompiledLorentzOps'] = None

    @classmethod
    def get_instance(cls) -> 'CompiledLorentzOps':
        '''Get singleton instance.'''
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def exp_map_zero(x_tan: Tensor, c: float = 1.0) -> Tensor:
        '''
        Exponential map from tangent space at origin to hyperboloid.

        Args:
            x_tan: Tangent vector, shape [..., D+1]
            c: Curvature parameter (default: 1.0)

        Returns:
            Point on hyperboloid, shape [..., D+1]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_tan.device, dtype=x_tan.dtype))
        v_spatial = x_tan[..., 1:]
        return _compiled_exp_map(v_spatial, sqrt_c)

    @staticmethod
    def log_map_zero(x_hyp: Tensor, c: float = 1.0) -> Tensor:
        '''
        Logarithmic map from hyperboloid to tangent space at origin.

        Args:
            x_hyp: Point on hyperboloid, shape [..., D+1]
            c: Curvature parameter (default: 1.0)

        Returns:
            Tangent vector, shape [..., D+1]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_hyp.device, dtype=x_hyp.dtype))
        x0 = x_hyp[..., 0:1]
        x_spatial = x_hyp[..., 1:]
        return _compiled_log_map(x0, x_spatial, sqrt_c)

    @staticmethod
    def lorentz_distance(u: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        '''
        Compute Lorentzian distance between two points on the hyperboloid.

        Args:
            u: First point on hyperboloid, shape [..., D+1]
            v: Second point on hyperboloid, shape [..., D+1]
            c: Curvature parameter (default: 1.0)

        Returns:
            Distances, shape [...]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=u.device, dtype=u.dtype))
        uv = u * v
        uv_time = uv[..., 0]
        uv_spatial_sum = torch.sum(uv[..., 1:], dim=-1)
        return _compiled_lorentz_dist(uv_time, uv_spatial_sum, sqrt_c)

    @staticmethod
    def minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
        '''
        Compute Minkowski inner product: ⟨x, y⟩_L = -x_0*y_0 + Σ x_i*y_i

        Args:
            x: First tensor [..., D+1]
            y: Second tensor [..., D+1]

        Returns:
            Inner product [...]
        '''
        return _compiled_minkowski_dot(x, y)

    @staticmethod
    def project_to_hyperboloid(x: Tensor, c: float = 1.0) -> Tensor:
        '''
        Project points onto the Lorentz hyperboloid.

        Args:
            x: Points to project [..., D+1]
            c: Curvature parameter

        Returns:
            Projected points on hyperboloid
        '''
        spatial = x[..., 1:]
        return _compiled_project(spatial, 1.0 / c)

    @staticmethod
    def sech_adaptive_margin(x: Tensor, c: float = 1.0, base_margin: float = 0.5) -> Tensor:
        '''
        Compute sech-based adaptive margin for embeddings.

        Args:
            x: Hyperbolic embeddings [..., D+1]
            c: Curvature parameter
            base_margin: Base margin value

        Returns:
            Adaptive margins [...]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        time_coord = x[..., 0]
        return _compiled_sech_margin(time_coord, sqrt_c, base_margin)

# -------------------------------------------------------------------------------------------------
# Exported compiled functions for direct use
# -------------------------------------------------------------------------------------------------

def compiled_exp_map_zero(x_tan: Tensor, c: float = 1.0) -> Tensor:
    '''Compiled exponential map from tangent space at origin to hyperboloid.'''
    return CompiledLorentzOps.exp_map_zero(x_tan, c)

def compiled_log_map_zero(x_hyp: Tensor, c: float = 1.0) -> Tensor:
    '''Compiled logarithmic map from hyperboloid to tangent space at origin.'''
    return CompiledLorentzOps.log_map_zero(x_hyp, c)

def compiled_lorentz_distance(u: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
    '''Compiled Lorentzian distance computation.'''
    return CompiledLorentzOps.lorentz_distance(u, v, c)

def compiled_minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
    '''Compiled Minkowski inner product.'''
    return CompiledLorentzOps.minkowski_dot(x, y)

def compiled_project_to_hyperboloid(x: Tensor, c: float = 1.0) -> Tensor:
    '''Compiled projection to hyperboloid.'''
    return CompiledLorentzOps.project_to_hyperboloid(x, c)

# -------------------------------------------------------------------------------------------------
# Benchmark Utilities
# -------------------------------------------------------------------------------------------------

def benchmark_compile_speedup(
    batch_size: int = 256,
    embedding_dim: int = 768,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> dict:
    '''
    Benchmark compiled vs eager Lorentz operations.

    Args:
        batch_size: Batch size for benchmarking
        embedding_dim: Embedding dimension (will use D+1 for Lorentz)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        device: Device to run benchmark on

    Returns:
        Dictionary with timing results
    '''
    import time

    # Create test tensors
    lorentz_dim = embedding_dim + 1
    x = torch.randn(batch_size, lorentz_dim, device=device)
    y = torch.randn(batch_size, lorentz_dim, device=device)

    # Project to hyperboloid for valid test data
    x = CompiledLorentzOps.project_to_hyperboloid(x)
    y = CompiledLorentzOps.project_to_hyperboloid(y)

    # Tangent vectors
    v = torch.randn(batch_size, lorentz_dim, device=device)
    v[:, 0] = 0  # Time component should be 0 for tangent at origin

    results = {}

    # Define eager versions (without compilation)
    def eager_exp_map(v_spatial: Tensor, sqrt_c: Tensor) -> Tensor:
        norm_v = torch.norm(v_spatial, p=2, dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=1e-8)
        theta = torch.clamp(sqrt_c * norm_v, max=40.0)
        x0 = torch.cosh(theta) / sqrt_c
        sinh_term = torch.sinh(theta) / sqrt_c
        x_spatial = (sinh_term / norm_v) * v_spatial
        return torch.cat([x0, x_spatial], dim=-1)

    def eager_distance(u: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        sqrt_c = torch.sqrt(torch.tensor(c, device=u.device, dtype=u.dtype))
        uv = u * v
        dot_product = torch.sum(uv[..., 1:], dim=-1) - uv[..., 0]
        arccosh_arg = torch.clamp(-dot_product, min=1.0)
        return sqrt_c * torch.acosh(arccosh_arg)

    sqrt_c = torch.sqrt(torch.tensor(1.0, device=device))
    v_spatial = v[:, 1:]

    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark exp_map - Eager
    for _ in range(num_warmup):
        _ = eager_exp_map(v_spatial, sqrt_c)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = eager_exp_map(v_spatial, sqrt_c)
    if device == 'cuda':
        torch.cuda.synchronize()
    eager_exp_time = (time.perf_counter() - start) / num_iterations

    # Benchmark exp_map - Compiled
    for _ in range(num_warmup):
        _ = _compiled_exp_map(v_spatial, sqrt_c)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = _compiled_exp_map(v_spatial, sqrt_c)
    if device == 'cuda':
        torch.cuda.synchronize()
    compiled_exp_time = (time.perf_counter() - start) / num_iterations

    results['exp_map'] = {
        'eager_ms': eager_exp_time * 1000,
        'compiled_ms': compiled_exp_time * 1000,
        'speedup': eager_exp_time / compiled_exp_time if compiled_exp_time > 0 else float('inf'),
    }

    # Benchmark distance - Eager
    for _ in range(num_warmup):
        _ = eager_distance(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = eager_distance(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()
    eager_dist_time = (time.perf_counter() - start) / num_iterations

    # Benchmark distance - Compiled
    for _ in range(num_warmup):
        _ = CompiledLorentzOps.lorentz_distance(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = CompiledLorentzOps.lorentz_distance(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()
    compiled_dist_time = (time.perf_counter() - start) / num_iterations

    results['lorentz_distance'] = {
        'eager_ms': eager_dist_time * 1000,
        'compiled_ms': compiled_dist_time * 1000,
        'speedup': eager_dist_time / compiled_dist_time if compiled_dist_time > 0 else float('inf'),
    }

    results['config'] = {
        'batch_size': batch_size,
        'embedding_dim': embedding_dim,
        'device': device,
        'compile_enabled': get_compile_config().enabled,
    }

    return results
