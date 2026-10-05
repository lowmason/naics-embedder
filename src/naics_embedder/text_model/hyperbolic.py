# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Utilities
# Shared module for hyperbolic embeddings, distances, and manifold operations
# With optional torch.compile support for fused operations
# -------------------------------------------------------------------------------------------------

import math
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

# Import compile utilities
try:
    from naics_embedder.utils.compile import maybe_compile

    _COMPILE_AVAILABLE = True
except ImportError:
    _COMPILE_AVAILABLE = False

    def maybe_compile(*args, **kwargs):  # type: ignore[misc]
        '''Fallback decorator when compile module not available.'''

        def decorator(fn):
            return fn

        return decorator

# -------------------------------------------------------------------------------------------------
# The geometry head and the polar distance (spec 4.2)
# -------------------------------------------------------------------------------------------------

class HeadPoints(NamedTuple):
    '''
    The head's points for a batch of B vectors in dimension d.

    Attributes:
        tangent: The bounded tangent vector at the origin, r · û (B, d), which the export writes.
        embedding: The Lorentz point exp_o(r · û) = (cosh r, sinh r · û) at c = 1 (B, d + 1).
        radius: r, the point's geodesic distance from the origin (B,).
        direction: û, the unit direction of the head's input (B, d); zero where the input is zero.
    '''

    tangent: torch.Tensor
    embedding: torch.Tensor
    radius: torch.Tensor
    direction: torch.Tensor

def _where_positive(values: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    '''``values`` where ``positive`` holds, else 1: a safe divisor or root whose gradient is 0.'''

    return torch.where(positive, values, torch.ones_like(values))

class HyperbolicHead(nn.Module):
    '''
    The hyperbolic head (spec 4.2): a gradient-passing bound on the radius, then the exp map at the
    origin of the c = 1 hyperboloid. It has no parameters and no curvature.

    From the projection's output v, with ν = ‖v‖ and the bound R, the radius is r = R · tanh(ν / R)
    and the direction is û = v / ν, so r ≤ R for every v. The radius takes gradient at any length
    (Req 13): dr/dν = sech²(ν / R) is positive, about 0.61 at a six-digit code's target r = 5 under
    R = 8. The interim cap at norm 2 passed about 1e-7 at its saturated points. In float32 the
    derivative rounds to 0 only from ν ≈ 8.7R, where tanh rounds to 1. At v = 0 the direction and
    the radius are 0, so the point is the origin, and every gradient there is finite. ``distance``
    names the decoding distance for its points.

    Args:
        radius_bound: R, the bound on every radius (``model.radius_bound``).

    Raises:
        ValueError: If ``radius_bound`` is not a positive finite number.
    '''

    distance = 'lorentz'

    def __init__(self, radius_bound: float = 8.0):
        super().__init__()
        if not (math.isfinite(radius_bound) and radius_bound > 0):
            raise ValueError(f'radius_bound must be a positive finite number, not {radius_bound!r}')
        self.radius_bound = float(radius_bound)

    def forward(self, vectors: torch.Tensor) -> HeadPoints:
        '''
        Bound each vector's radius and map it to the hyperboloid, in the vectors' dtype.

        Args:
            vectors: The projection's outputs, (B, d).

        Returns:
            The bounded tangent, the Lorentz point, the radius and the direction.
        '''

        norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        radius = self.radius_bound * torch.tanh(norm / self.radius_bound)
        # Guarded by where, not a clamp: at ν = 0 the direction and its gradient are 0, not 1/ε
        moving = norm > 0
        direction = torch.where(
            moving, vectors / _where_positive(norm, moving), torch.zeros_like(vectors)
        )
        # The tangent is computed once, here, so the export and every reader see the same r · û
        tangent = radius * direction
        embedding = torch.cat([torch.cosh(radius), torch.sinh(radius) * direction], dim=1)
        return HeadPoints(tangent, embedding, radius.squeeze(1), direction)

def _refuse_unpaired_shapes(*tensors: torch.Tensor) -> None:
    '''Refuse anything but radii (A,) and (B,) with directions (A, d) and (B, d).'''

    shapes = [tuple(tensor.shape) for tensor in tensors]
    paired = [len(shape) for shape in shapes] == [1, 2, 1, 2]
    if paired:
        (count_a, ), (rows_a, width_a), (count_b, ), (rows_b, width_b) = shapes
        paired = (rows_a, rows_b, width_a) == (count_a, count_b, width_b)
    if not paired:
        raise ValueError(
            'polar_distance takes radii (A,) and (B,) with directions (A, d) and (B, d); got '
            + ', '.join(str(shape) for shape in shapes)
        )

def polar_distance(
    radius_a: torch.Tensor,
    direction_a: torch.Tensor,
    radius_b: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    '''
    The geodesic distance at c = 1 between every point of one set and every point of another,
    from each point's radius r and direction û (spec 4.2).

    The distance is the hyperbolic law of cosines in half-angle form, so it equals
    arcosh(−⟨x, y⟩_L) exactly:

        d(x, y) = 2 · asinh(√(sinh²((r_x − r_y) / 2) + sinh r_x · sinh r_y · ‖û_x − û_y‖² / 4))

    Every term is non-negative, so nothing cancels, and float32 resolves it at every radius the
    head's bound allows. ‖û_x − û_y‖² comes from explicit differences, not from 2 − 2 û_x · û_y,
    which cancels at small angles. The ops are elementwise, so under autocast float32 inputs give
    float32 distances. The square root is guarded at zero: a zero separation has distance 0, and
    the value and its gradient stay finite.

    Args:
        radius_a: The first set's radii, (A,).
        direction_a: Its directions, (A, d): unit vectors, or 0 at the origin.
        radius_b: The second set's radii, (B,).
        direction_b: Its directions, (B, d).

    Returns:
        The distances, (A, B), in the inputs' dtype.

    Raises:
        ValueError: If the shapes are not (A,), (A, d), (B,) and (B, d).
    '''

    _refuse_unpaired_shapes(radius_a, direction_a, radius_b, direction_b)
    half_gap = torch.sinh((radius_a.unsqueeze(1) - radius_b.unsqueeze(0)) / 2)
    chord = (direction_a.unsqueeze(1) - direction_b.unsqueeze(0)).square().sum(dim=2)
    sinh_product = torch.sinh(radius_a).unsqueeze(1) * torch.sinh(radius_b).unsqueeze(0)
    squared = half_gap.square() + sinh_product * chord / 4
    separated = squared > 0
    root = torch.where(
        separated, torch.sqrt(_where_positive(squared, separated)), torch.zeros_like(squared)
    )
    return 2 * torch.asinh(root)

# -------------------------------------------------------------------------------------------------
# Hyperbolic Manifold Validation and Diagnostics
# -------------------------------------------------------------------------------------------------

def check_lorentz_manifold_validity(
    embeddings: torch.Tensor, curvature: float = 1.0, tolerance: float = 1e-3
) -> Tuple[bool, torch.Tensor, torch.Tensor]:
    '''
    Check if embeddings satisfy the Lorentz hyperboloid constraint.

    For valid points: -x₀² + x₁² + ... + xₙ² = -1/c

    Args:
        embeddings: Hyperbolic embeddings of shape (batch_size, embedding_dim+1)
        curvature: Curvature parameter c
        tolerance: Tolerance for constraint violation

    Returns:
        Tuple of:
            - is_valid: Boolean indicating if all points are valid
            - lorentz_norms: Lorentz inner product for each point (should be -1/c)
            - violations: Magnitude of constraint violations
    '''
    # Compute Lorentz inner product with itself: ⟨x, x⟩_L
    time_coord = embeddings[:, 0]  # x₀
    spatial_coords = embeddings[:, 1:]  # x₁...xₙ

    spatial_norm_sq = torch.sum(spatial_coords**2, dim=1)
    time_norm_sq = time_coord**2

    lorentz_norms = spatial_norm_sq - time_norm_sq  # Should be -1/c

    target_value = -1.0 / curvature
    violations = torch.abs(lorentz_norms - target_value)

    is_valid = bool(torch.all(violations < tolerance).item())

    return is_valid, lorentz_norms, violations

def compute_hyperbolic_radii(embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Extract hyperbolic radii (time coordinates) from Lorentz embeddings.

    The time coordinate x₀ represents the hyperbolic radius (distance from origin).

    Args:
        embeddings: Hyperbolic embeddings of shape (batch_size, embedding_dim+1)

    Returns:
        Hyperbolic radii of shape (batch_size,)
    '''
    return embeddings[:, 0]

# -------------------------------------------------------------------------------------------------
# Compiled Lorentz Operations Core Functions
# -------------------------------------------------------------------------------------------------

@maybe_compile(mode='reduce-overhead')
def _log_map_zero_ops_compiled(
    x0: torch.Tensor, x_spatial: torch.Tensor, sqrt_c: torch.Tensor
) -> torch.Tensor:
    '''Core logarithmic map computation for LorentzOps - highly fusible.'''
    theta = torch.acosh(torch.clamp(sqrt_c * x0, min=1.0 + 1e-5))
    sinh_theta = torch.sinh(theta)
    sinh_theta = torch.clamp(sinh_theta, min=1e-8)
    scale = theta / sinh_theta
    scale = torch.where(theta > 1e-8, scale, torch.ones_like(scale))
    v_spatial = scale * x_spatial
    v_time = torch.zeros_like(x0)
    return torch.cat([v_time, v_spatial], dim=1)

@maybe_compile(mode='reduce-overhead')
def _exp_map_zero_ops_compiled(v_spatial: torch.Tensor, sqrt_c: torch.Tensor) -> torch.Tensor:
    '''Core exponential map computation for LorentzOps - highly fusible.'''
    norm_v = torch.norm(v_spatial, p=2, dim=1, keepdim=True)
    norm_v = torch.clamp(norm_v, min=1e-8)
    theta = torch.clamp(sqrt_c * norm_v, max=40.0)
    x0 = torch.cosh(theta) / sqrt_c
    sinh_term = torch.sinh(theta) / sqrt_c
    x_spatial = (sinh_term / norm_v) * v_spatial
    return torch.cat([x0, x_spatial], dim=1)

@maybe_compile(mode='reduce-overhead')
def _lorentz_distance_ops_compiled(
    uv_time: torch.Tensor, uv_spatial_sum: torch.Tensor, sqrt_c: torch.Tensor
) -> torch.Tensor:
    '''Core Lorentz distance computation for LorentzOps - highly fusible.'''
    dot_product = uv_spatial_sum - uv_time
    arccosh_arg = torch.clamp(-dot_product, min=1.0)
    return sqrt_c * torch.acosh(arccosh_arg)

# -------------------------------------------------------------------------------------------------
# Lorentz Operations Utility Class
# -------------------------------------------------------------------------------------------------

class LorentzOps:
    '''
    Static utility class for Lorentz model operations.
    Provides functions for mapping between hyperboloid and tangent space, and computing distances.

    When torch.compile is enabled (PyTorch 2.0+), core operations are compiled
    for better throughput through kernel fusion.
    '''

    @staticmethod
    def log_map_zero(x_hyp: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Logarithmic map from hyperboloid to tangent space at origin.

        Maps a point on the Lorentz hyperboloid to the tangent space at the origin.
        Inverse of exp_map_zero.

        Uses compiled operations when torch.compile is enabled.

        Args:
            x_hyp: Point on hyperboloid, shape (batch_size, embedding_dim+1)
                   Must satisfy ||x_spatial||^2 - x0^2 = -1/c
            c: Curvature parameter (default: 1.0)

        Returns:
            Tangent vector, shape (batch_size, embedding_dim+1)
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_hyp.device, dtype=x_hyp.dtype))
        x0 = x_hyp[:, 0:1]  # (batch_size, 1)
        x_spatial = x_hyp[:, 1:]  # (batch_size, embedding_dim)
        return _log_map_zero_ops_compiled(x0, x_spatial, sqrt_c)

    @staticmethod
    def exp_map_zero(x_tan: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Exponential map from tangent space at origin to hyperboloid.

        Maps a tangent vector at the origin to a point on the Lorentz hyperboloid.
        The output satisfies the Lorentz constraint: ||x_spatial||^2 - x0^2 = -1/c

        Uses compiled operations when torch.compile is enabled.

        Args:
            x_tan: Tangent vector, shape (batch_size, embedding_dim+1)
            c: Curvature parameter (default: 1.0)

        Returns:
            Point on hyperboloid, shape (batch_size, embedding_dim+1)
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_tan.device, dtype=x_tan.dtype))
        v_spatial = x_tan[:, 1:]  # (batch_size, embedding_dim)
        return _exp_map_zero_ops_compiled(v_spatial, sqrt_c)

    @staticmethod
    def lorentz_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Compute Lorentzian distance between two points on the hyperboloid.

        Uses compiled operations when torch.compile is enabled.

        Args:
            u: First point on hyperboloid, shape (batch_size, embedding_dim+1)
            v: Second point on hyperboloid, shape (batch_size, embedding_dim+1)
            c: Curvature parameter (default: 1.0)

        Returns:
            Distances, shape (batch_size,)
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=u.device, dtype=u.dtype))
        uv = u * v
        uv_time = uv[:, 0]
        uv_spatial_sum = torch.sum(uv[:, 1:], dim=1)
        return _lorentz_distance_ops_compiled(uv_time, uv_spatial_sum, sqrt_c)
