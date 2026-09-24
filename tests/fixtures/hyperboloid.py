'''
Synthetic hyperboloid points for testing the precision of evaluation-only Lorentz distances.

The builders mimic real exports: points built exactly in float64 at hyperbolic radius 2-4 with 384
spatial dimensions, which tests then store as float32. That far from the origin, x0 * y0 - <xs, ys>
cancels catastrophically, and float32 rounding of x0 alone moves -<x, x>_L off 1 by ~5e-5, which
acosh near 1 turns into distances of ~1e-2.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

SPATIAL_DIM = 384

# -------------------------------------------------------------------------------------------------
# Points
# -------------------------------------------------------------------------------------------------

def hyperboloid_points(
    count: int,
    *,
    seed: int,
    min_radius: float = 2.0,
    max_radius: float = 4.0,
    spatial_dim: int = SPATIAL_DIM,
) -> torch.Tensor:
    '''Exact float64 points on the c = 1 hyperboloid at hyperbolic radii in [min, max].'''
    generator = torch.Generator().manual_seed(seed)
    unit = torch.rand(count, 1, generator=generator, dtype=torch.float64)
    radius = min_radius + (max_radius - min_radius) * unit
    direction = torch.randn(count, spatial_dim, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    return torch.cat([torch.cosh(radius), torch.sinh(radius) * direction], dim=1)

def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x[:, 1:] * y[:, 1:]).sum(dim=1) - x[:, 0] * y[:, 0]

def points_at_distance(points: torch.Tensor, distance: float, *, seed: int) -> torch.Tensor:
    '''Move each float64 hyperboloid point exactly `distance` along a random geodesic.

    A distance of 0 returns exact copies, since cosh(0) = 1 and sinh(0) = 0.
    '''
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(points.shape, generator=generator, dtype=torch.float64)
    # Project onto the tangent space at each point: v = w + <x, w>_L x satisfies <x, v>_L = 0.
    tangent = noise + minkowski_dot(points, noise).unsqueeze(1) * points
    tangent = tangent / minkowski_dot(tangent, tangent).sqrt().unsqueeze(1)
    return math.cosh(distance) * points + math.sinh(distance) * tangent

def fortran_order(points: torch.Tensor) -> torch.Tensor:
    '''The same values in column-major memory, as torch.tensor keeps Polars' to_numpy() layout.'''
    f_order = points.T.contiguous().T
    assert torch.equal(points, f_order)
    assert points.stride() != f_order.stride()
    return f_order

# -------------------------------------------------------------------------------------------------
# Memory
# -------------------------------------------------------------------------------------------------

class LargestStorage(TorchDispatchMode):
    '''Record the largest tensor storage that any op produces while the mode is active.'''

    def __init__(self):
        super().__init__()
        self.nbytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in out if isinstance(out, (tuple, list)) else (out, ):
            if isinstance(tensor, torch.Tensor):
                self.nbytes = max(self.nbytes, tensor.untyped_storage().nbytes())
        return out
