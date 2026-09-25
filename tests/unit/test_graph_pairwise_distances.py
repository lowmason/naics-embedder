'''Precision and memory of the pairwise Lorentz distances of ``lorentz_distance_matrix``.

The fixtures (tests/fixtures/hyperboloid.py) mimic real exports: points built exactly in float64 at
hyperbolic radius 2-4 with 384 spatial dimensions, then stored as float32.
'''

import torch

from naics_embedder.metrics.core import lorentz_distance_matrix
from tests.fixtures.hyperboloid import (
    LargestStorage,
    fortran_order,
    hyperboloid_points,
    points_at_distance,
)

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return lorentz_distance_matrix(embeddings)

# -------------------------------------------------------------------------------------------------
# Precision
# -------------------------------------------------------------------------------------------------

def test_copies_of_a_point_are_zero_apart():
    # Identical rows at distinct indices, as for codes whose text is identical, test the
    # arithmetic itself; the diagonal alone could pass by being overwritten.
    points = hyperboloid_points(32, seed=0).float()
    distances = _pairwise_distances(torch.cat([points, points]))

    copies = torch.arange(32)
    same_point = torch.cat([distances.diagonal(), distances[copies, copies + 32]])
    torch.testing.assert_close(same_point, torch.zeros_like(same_point), rtol=0, atol=1e-5)

def test_close_pairs_far_from_the_origin_keep_their_distance():
    # Parent/child and sibling pairs sit this close, and their order decides the metrics.
    parents = hyperboloid_points(32, seed=1, min_radius=3.0)
    children = points_at_distance(parents, 0.01, seed=2)
    distances = _pairwise_distances(torch.cat([parents, children]).float())

    pairs = torch.arange(32)
    close = distances[pairs, pairs + 32]
    torch.testing.assert_close(close, torch.full_like(close, 0.01), rtol=1e-3, atol=0)

def test_distances_do_not_depend_on_memory_layout():
    c_order = hyperboloid_points(64, seed=3).float()

    assert torch.equal(_pairwise_distances(c_order), _pairwise_distances(fortran_order(c_order)))

# -------------------------------------------------------------------------------------------------
# Memory
# -------------------------------------------------------------------------------------------------

def test_distances_never_materialize_an_n_by_n_by_d_tensor():
    # At N of about 2,100 codes and D = 384, an (N, N, D) float32 intermediate is about 6.8 GB.
    count, spatial_dim = 24, 40
    points = hyperboloid_points(count, seed=4, spatial_dim=spatial_dim).float()

    with LargestStorage() as largest:
        _pairwise_distances(points)

    # Nothing bigger than a float64 (N, N) matrix or a float64 copy of the (N, D + 1) input.
    assert largest.nbytes <= 8 * max(count * count, count * (spatial_dim + 1))
