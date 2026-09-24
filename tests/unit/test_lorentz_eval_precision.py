'''Precision, layout independence and memory of the evaluation-only Lorentz distance paths.

EmbeddingEvaluator.compute_pairwise_distances(metric='lorentz') feeds both stages' hierarchy
validation metrics (Spearman, cophenetic, NDCG, retrieval); compute_validation_metrics feeds HGCN's
triplet validation metrics. The fixtures (tests/fixtures/hyperboloid.py) mimic real exports: points
built exactly in float64 at hyperbolic radius 2-4 with 384 spatial dimensions, then stored as
float32.
'''

import math

import pytest
import torch

from naics_embedder.metrics import EmbeddingEvaluator, compute_validation_metrics
from tests.fixtures.hyperboloid import (
    LargestStorage,
    fortran_order,
    hyperboloid_points,
    points_at_distance,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

# MPS has no float64, so the embeddings must reach the CPU before the cast.
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='needs an Apple GPU (MPS)'
)

# -------------------------------------------------------------------------------------------------
# Pairwise distances: EmbeddingEvaluator.compute_pairwise_distances(metric='lorentz')
# -------------------------------------------------------------------------------------------------

def _evaluator() -> EmbeddingEvaluator:
    evaluator = EmbeddingEvaluator()
    evaluator.device = 'cpu'
    return evaluator

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return _evaluator().compute_pairwise_distances(embeddings, metric='lorentz', curvature=1.0)

def test_pairwise_copies_of_a_point_are_zero_apart():
    # Identical rows at distinct indices, as for codes whose text is identical, test the
    # arithmetic itself; the diagonal alone could pass by being overwritten.
    points = hyperboloid_points(32, seed=0).float()
    distances = _pairwise_distances(torch.cat([points, points]))

    copies = torch.arange(32)
    same_point = torch.cat(
        [
            distances.diagonal(),
            distances[copies, copies + 32],
            distances[copies + 32, copies],
        ]
    )
    torch.testing.assert_close(same_point, torch.zeros_like(same_point), rtol=0, atol=1e-5)

def test_pairwise_close_pairs_far_from_the_origin_keep_their_distance():
    # Parent/child and sibling pairs sit this close, and their order decides the metrics.
    parents = hyperboloid_points(32, seed=1, min_radius=3.0)
    children = points_at_distance(parents, 0.01, seed=2)
    distances = _pairwise_distances(torch.cat([parents, children]).float())

    pairs = torch.arange(32)
    close = distances[pairs, pairs + 32]
    torch.testing.assert_close(close, torch.full_like(close, 0.01), rtol=1e-3, atol=0)

def test_pairwise_distances_do_not_depend_on_memory_layout():
    c_order = hyperboloid_points(64, seed=3).float()

    assert torch.equal(_pairwise_distances(c_order), _pairwise_distances(fortran_order(c_order)))

def test_pairwise_distances_are_symmetric_float32_on_the_evaluator_device():
    # Downstream metrics move the matrix to their own device, which may be MPS (no float64), and
    # structural Spearman rejects matrices that are not symmetric within float32 tolerance.
    points = hyperboloid_points(48, seed=4)
    distances = _pairwise_distances(torch.cat([points, points[:8]]))

    assert distances.dtype == torch.float32
    assert distances.device.type == 'cpu'
    assert torch.equal(distances, distances.T)

@requires_mps
def test_pairwise_distances_accept_mps_embeddings():
    points = hyperboloid_points(16, seed=6).float()
    evaluator = _evaluator()
    evaluator.device = 'mps'

    distances = evaluator.compute_pairwise_distances(points.to('mps'), metric='lorentz')

    assert distances.device.type == 'mps'
    assert torch.equal(distances.cpu(), _pairwise_distances(points))

def test_pairwise_distances_never_materialize_an_n_by_n_by_d_tensor():
    # At N = 2,125 codes and D = 384, an (N, N, D + 1) float32 intermediate is 6.95 GB.
    count, spatial_dim = 24, 40
    embeddings = hyperboloid_points(count, seed=5, spatial_dim=spatial_dim).float()
    evaluator = _evaluator()

    with LargestStorage() as largest:
        evaluator.compute_pairwise_distances(embeddings, metric='lorentz', curvature=1.0)

    # Nothing bigger than a float64 (N, N) matrix or a float64 copy of the (N, D + 1) input.
    assert largest.nbytes <= 8 * max(count * count, count * (spatial_dim + 1))

# -------------------------------------------------------------------------------------------------
# Triplet validation metrics: compute_validation_metrics
# -------------------------------------------------------------------------------------------------

ANCHORS = 64
POSITIVE_DISTANCE = 0.02
NEGATIVE_DISTANCES = torch.linspace(0.022, 0.029, 8, dtype=torch.float64)

Triplets = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def _triplets(positive_distance: float = POSITIVE_DISTANCE) -> Triplets:
    '''Float32 embeddings plus anchor, positive and (anchor, k) negative indices.

    Every anchor's positive sits at `positive_distance` and its hard negatives just beyond it, so
    exact distances rank every positive first.
    '''
    anchors = hyperboloid_points(ANCHORS, seed=10)
    positives = points_at_distance(anchors, positive_distance, seed=11)
    negatives = [
        points_at_distance(anchors, float(distance), seed=12 + k)
        for k, distance in enumerate(NEGATIVE_DISTANCES)
    ]
    # Row b * K + k of the negative block is anchor b's k-th negative.
    negative_block = torch.stack(negatives, dim=1).reshape(-1, anchors.size(1))
    embeddings = torch.cat([anchors, positives, negative_block]).float()

    k_negatives = len(NEGATIVE_DISTANCES)
    anchor_idx = torch.arange(ANCHORS)
    positive_idx = ANCHORS + anchor_idx
    negative_idx = 2 * ANCHORS + torch.arange(ANCHORS * k_negatives).view(ANCHORS, k_negatives)
    return embeddings, anchor_idx, positive_idx, negative_idx

def test_validation_metrics_rank_close_positives_ahead_of_hard_negatives():
    metrics = compute_validation_metrics(*_triplets(), c=1.0, top_k=5)

    assert metrics['relation_accuracy'] == 1.0
    assert metrics['top_k_relation_accuracy'] == 1.0
    assert metrics['mean_positive_rank'] == 0.0

def test_validation_metrics_keep_close_distances():
    metrics = compute_validation_metrics(*_triplets(), c=1.0)

    assert math.isclose(metrics['avg_positive_dist'], POSITIVE_DISTANCE, rel_tol=1e-3)
    assert math.isclose(
        metrics['avg_negative_dist'], float(NEGATIVE_DISTANCES.mean()), rel_tol=1e-3
    )

def test_validation_metrics_put_copies_zero_apart():
    # Each positive is a float32 copy of its anchor, stored at a different index.
    metrics = compute_validation_metrics(*_triplets(positive_distance=0.0), c=1.0)

    assert metrics['avg_positive_dist'] <= 1e-5
    assert metrics['relation_accuracy'] == 1.0

def test_validation_metrics_do_not_depend_on_memory_layout():
    # Python floats keep the float64 aggregates, whose last bits would show layout-dependent
    # rounding that a float32 conversion hides.
    embeddings, anchors, positives, negatives = _triplets()
    c_order = compute_validation_metrics(embeddings, anchors, positives, negatives, c=1.0)
    f_order = compute_validation_metrics(
        fortran_order(embeddings), anchors, positives, negatives, c=1.0
    )

    assert c_order == f_order

def test_validation_metric_tensors_are_float32_on_the_embedding_device():
    # Lightning logs these tensors from the model's device.
    embeddings, anchors, positives, negatives = _triplets()
    metrics = compute_validation_metrics(
        embeddings, anchors, positives, negatives, c=1.0, as_tensors=True
    )

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), name
        assert value.dtype == torch.float32, name
        assert value.device == embeddings.device, name

@requires_mps
def test_validation_metrics_accept_mps_tensors():
    triplets = _triplets()
    on_cpu = compute_validation_metrics(*triplets, c=1.0, as_tensors=True)
    on_mps = compute_validation_metrics(
        *(tensor.to('mps') for tensor in triplets), c=1.0, as_tensors=True
    )

    for name, value in on_mps.items():
        assert value.device.type == 'mps', name
        assert torch.equal(value.cpu(), on_cpu[name]), name
