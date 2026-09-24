'''Precision and memory of GraphDownstreamEvaluator's pairwise Lorentz distances.

The fixtures mimic real exports: points built exactly in float64 at hyperbolic radius 2-4 with 384
spatial dimensions, then stored as float32. That far from the origin, x0 * y0 - <xs, ys> cancels
catastrophically, and float32 rounding of x0 alone moves -<x, x>_L off 1 by ~5e-5, which acosh
near 1 turns into distances of ~1e-2.
'''

import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from naics_embedder.metrics import GraphDownstreamEvaluator, GraphEmbeddingDataset

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

SPATIAL_DIM = 384

def _hyperboloid_points(
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

def _minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x[:, 1:] * y[:, 1:]).sum(dim=1) - x[:, 0] * y[:, 0]

def _points_at_distance(points: torch.Tensor, distance: float, *, seed: int) -> torch.Tensor:
    '''Move each float64 hyperboloid point exactly `distance` along a random geodesic.'''
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(points.shape, generator=generator, dtype=torch.float64)
    # Project onto the tangent space at each point: v = w + <x, w>_L x satisfies <x, v>_L = 0.
    tangent = noise + _minkowski_dot(points, noise).unsqueeze(1) * points
    tangent = tangent / _minkowski_dot(tangent, tangent).sqrt().unsqueeze(1)
    return math.cosh(distance) * points + math.sinh(distance) * tangent

def _evaluator(embeddings: torch.Tensor) -> GraphDownstreamEvaluator:
    count = embeddings.size(0)
    codes = [str(i) for i in range(count)]
    dataset = GraphEmbeddingDataset(embeddings=embeddings, codes=codes, levels=[6] * count)
    return GraphDownstreamEvaluator(dataset)

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return _evaluator(embeddings)._pairwise_distances()

# -------------------------------------------------------------------------------------------------
# Precision
# -------------------------------------------------------------------------------------------------

def test_copies_of_a_point_are_zero_apart():
    # Identical rows at distinct indices, as for codes whose text is identical, test the
    # arithmetic itself; the diagonal alone could pass by being overwritten.
    points = _hyperboloid_points(32, seed=0).float()
    distances = _pairwise_distances(torch.cat([points, points]))

    copies = torch.arange(32)
    same_point = torch.cat([distances.diagonal(), distances[copies, copies + 32]])
    torch.testing.assert_close(same_point, torch.zeros_like(same_point), rtol=0, atol=1e-5)

def test_close_pairs_far_from_the_origin_keep_their_distance():
    # Parent/child and sibling pairs sit this close, and their order decides the metrics.
    parents = _hyperboloid_points(32, seed=1, min_radius=3.0)
    children = _points_at_distance(parents, 0.01, seed=2)
    distances = _pairwise_distances(torch.cat([parents, children]).float())

    pairs = torch.arange(32)
    close = distances[pairs, pairs + 32]
    torch.testing.assert_close(close, torch.full_like(close, 0.01), rtol=1e-3, atol=0)

def test_distances_do_not_depend_on_memory_layout():
    c_order = _hyperboloid_points(64, seed=3).float()
    f_order = c_order.T.contiguous().T
    assert torch.equal(c_order, f_order)
    assert c_order.stride() != f_order.stride()

    assert torch.equal(_pairwise_distances(c_order), _pairwise_distances(f_order))

# -------------------------------------------------------------------------------------------------
# Memory
# -------------------------------------------------------------------------------------------------

class _LargestStorage(TorchDispatchMode):
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

def test_distances_never_materialize_an_n_by_n_by_d_tensor():
    # At N of about 2,100 codes and D = 384, an (N, N, D) float32 intermediate is about 6.8 GB.
    count, spatial_dim = 24, 40
    evaluator = _evaluator(_hyperboloid_points(count, seed=4, spatial_dim=spatial_dim).float())

    with _LargestStorage() as largest:
        evaluator._pairwise_distances()

    # Nothing bigger than a float64 (N, N) matrix or a float64 copy of the (N, D + 1) input.
    assert largest.nbytes <= 8 * max(count * count, count * (spatial_dim + 1))
