'''
Unit tests for graph_model/evaluation.py

Tests cover:
- compute_validation_metrics function
- Hyperbolic distance computation for graph embeddings
- Relation accuracy (positive closer than negatives)
- Mean positive rank computation
- Distance spread metrics
'''

import logging

import pytest
import torch

from naics_embedder.graph_model.evaluation import compute_validation_metrics
from naics_embedder.text_model.hyperbolic import LorentzOps

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_graph_embeddings(test_device, n_nodes=50, embedding_dim=384):
    '''Generate sample graph node embeddings on Lorentz manifold.'''

    # Create tangent vectors and map to hyperboloid
    tangent = torch.randn(n_nodes, embedding_dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component = 0
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0

    embeddings = LorentzOps.exp_map_zero(tangent, c=1.0)

    return embeddings

@pytest.fixture
def sample_triplet_indices(test_device, batch_size=16, k_negatives=8, n_nodes=50):
    '''Generate sample anchor-positive-negative triplet indices.'''

    # Create anchors
    anchors = torch.randint(0, n_nodes, (batch_size,), device=test_device)

    # Create positives (different from anchors)
    positives = (anchors + 1) % n_nodes

    # Create negatives (different from both anchors and positives)
    negatives = torch.randint(0, n_nodes, (batch_size, k_negatives), device=test_device)

    return anchors, positives, negatives

# -------------------------------------------------------------------------------------------------
# Test: compute_validation_metrics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestValidationMetrics:
    '''Test graph validation metrics computation.'''

    def test_metrics_computation_basic(
        self, sample_graph_embeddings, sample_triplet_indices, test_device
    ):
        '''Test basic validation metrics computation.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
            top_k=1,
        )

        # Check all expected metrics are present
        assert 'avg_positive_dist' in metrics
        assert 'avg_negative_dist' in metrics
        assert 'distance_spread' in metrics
        assert 'relation_accuracy' in metrics
        assert 'mean_positive_rank' in metrics

        # Check metrics are scalars
        assert isinstance(metrics['avg_positive_dist'], float)
        assert isinstance(metrics['avg_negative_dist'], float)
        assert isinstance(metrics['distance_spread'], float)
        assert isinstance(metrics['relation_accuracy'], float)
        assert isinstance(metrics['mean_positive_rank'], float)

    def test_positive_distance_computation(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test that positive distances are computed correctly.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        avg_positive_dist = metrics['avg_positive_dist']

        # Average positive distance should be positive and finite
        assert avg_positive_dist > 0
        assert not torch.isinf(torch.tensor(avg_positive_dist))
        assert not torch.isnan(torch.tensor(avg_positive_dist))

    def test_negative_distance_computation(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test that negative distances are computed correctly.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        avg_negative_dist = metrics['avg_negative_dist']

        # Average negative distance should be positive and finite
        assert avg_negative_dist > 0
        assert not torch.isinf(torch.tensor(avg_negative_dist))
        assert not torch.isnan(torch.tensor(avg_negative_dist))

    def test_relation_accuracy_range(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test that relation accuracy is in valid range [0, 1].'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        relation_accuracy = metrics['relation_accuracy']

        # Should be in [0, 1]
        assert 0.0 <= relation_accuracy <= 1.0

    def test_mean_positive_rank(self, sample_graph_embeddings, sample_triplet_indices):
        '''Test mean positive rank computation.'''

        anchors, positives, negatives = sample_triplet_indices
        k_negatives = negatives.shape[1]

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        mean_positive_rank = metrics['mean_positive_rank']

        # Rank should be in [0, k_negatives]
        # (rank 0 = positive is closest, rank k_negatives = positive is furthest)
        assert 0.0 <= mean_positive_rank <= k_negatives

    def test_distance_spread_positive(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test that distance spread is positive.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        distance_spread = metrics['distance_spread']

        # Spread (coefficient of variation) should be positive
        assert distance_spread > 0

# -------------------------------------------------------------------------------------------------
# Test: Different Curvatures
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDifferentCurvatures:
    '''Test validation metrics with different curvature values.'''

    def test_small_curvature(self, sample_graph_embeddings, sample_triplet_indices):
        '''Test metrics with small curvature (c = 0.1).'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=0.1,
        )

        # Should produce valid metrics
        assert not torch.isnan(torch.tensor(metrics['avg_positive_dist']))
        assert not torch.isnan(torch.tensor(metrics['avg_negative_dist']))

    def test_large_curvature(self, sample_graph_embeddings, sample_triplet_indices):
        '''Test metrics with large curvature (c = 10.0).'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=10.0,
        )

        # Should produce valid metrics
        assert not torch.isnan(torch.tensor(metrics['avg_positive_dist']))
        assert not torch.isnan(torch.tensor(metrics['avg_negative_dist']))

    def test_curvature_affects_distances(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test that different curvatures produce different distances.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics_c1 = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        metrics_c5 = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=5.0,
        )

        # Different curvatures should produce different average distances
        assert metrics_c1['avg_positive_dist'] != metrics_c5['avg_positive_dist']

# -------------------------------------------------------------------------------------------------
# Test: Edge Cases
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Test edge cases and boundary conditions.'''

    def test_single_anchor(self, sample_graph_embeddings, test_device):
        '''Test with single anchor-positive-negative triplet.'''

        batch_size = 1
        k_negatives = 4

        anchors = torch.tensor([0], device=test_device)
        positives = torch.tensor([1], device=test_device)
        negatives = torch.tensor([[2, 3, 4, 5]], device=test_device)

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # Should produce valid metrics
        assert metrics['avg_positive_dist'] > 0
        assert metrics['avg_negative_dist'] > 0

    def test_single_negative(self, sample_graph_embeddings, test_device):
        '''Test with single negative per anchor.'''

        batch_size = 4
        anchors = torch.tensor([0, 1, 2, 3], device=test_device)
        positives = torch.tensor([4, 5, 6, 7], device=test_device)
        negatives = torch.tensor([[8], [9], [10], [11]], device=test_device)

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # Should work with k_negatives = 1
        assert 0.0 <= metrics['relation_accuracy'] <= 1.0

    def test_many_negatives(self, sample_graph_embeddings, test_device):
        '''Test with many negatives per anchor.'''

        batch_size = 4
        k_negatives = 20

        anchors = torch.randint(0, 50, (batch_size,), device=test_device)
        positives = (anchors + 1) % 50
        negatives = torch.randint(0, 50, (batch_size, k_negatives), device=test_device)

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # Should handle many negatives
        assert metrics['mean_positive_rank'] <= k_negatives

# -------------------------------------------------------------------------------------------------
# Test: Perfect vs Random Ranking
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRankingQuality:
    '''Test that metrics distinguish between good and bad rankings.'''

    def test_perfect_ranking_high_accuracy(self, test_device):
        '''Test that perfect ranking (positive always closest) gives accuracy = 1.0.'''

        # Create embeddings where positives are always closer than negatives
        n_nodes = 20
        embedding_dim = 384

        # Place nodes on a line in hyperbolic space
        # Anchors at position 0, positives at position 1, negatives at positions 2+
        tangent = torch.zeros(n_nodes, embedding_dim + 1, device=test_device)
        tangent[:, 1] = torch.arange(n_nodes, dtype=torch.float32, device=test_device)
        embeddings = LorentzOps.exp_map_zero(tangent, c=1.0)

        # Anchors: 0, 4, 8, 12
        # Positives: 1, 5, 9, 13
        # Negatives: 2-3, 6-7, 10-11, 14-15
        batch_size = 4
        anchors = torch.tensor([0, 4, 8, 12], device=test_device)
        positives = torch.tensor([1, 5, 9, 13], device=test_device)
        negatives = torch.tensor([[2, 3], [6, 7], [10, 11], [14, 15]], device=test_device)

        metrics = compute_validation_metrics(
            emb=embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # With perfect ordering, relation accuracy should be 1.0
        assert metrics['relation_accuracy'] == 1.0

        # Mean positive rank should be 0 (always closest)
        assert metrics['mean_positive_rank'] == 0.0

    def test_random_ranking_lower_accuracy(self, sample_graph_embeddings, test_device):
        '''Test that random embeddings have lower accuracy.'''

        # With random embeddings, we expect lower (but non-zero) accuracy
        batch_size = 16
        k_negatives = 8

        anchors = torch.randint(0, 50, (batch_size,), device=test_device)
        positives = torch.randint(0, 50, (batch_size,), device=test_device)
        negatives = torch.randint(0, 50, (batch_size, k_negatives), device=test_device)

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # Relation accuracy should be less than 1.0 for random rankings
        assert metrics['relation_accuracy'] < 1.0

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test numerical stability of validation metrics.'''

    def test_no_nan_or_inf(self, sample_graph_embeddings, sample_triplet_indices):
        '''Test that metrics don\'t produce NaN or Inf values.'''

        anchors, positives, negatives = sample_triplet_indices

        metrics = compute_validation_metrics(
            emb=sample_graph_embeddings,
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            c=1.0,
        )

        # Check all metrics are finite
        for key, value in metrics.items():
            assert not torch.isnan(torch.tensor(value)), f'{key} is NaN'
            assert not torch.isinf(torch.tensor(value)), f'{key} is Inf'

    def test_identical_embeddings_handled(self, test_device):
        '''Test that identical embeddings (degenerate case) don\'t crash.'''

        # Create identical embeddings
        n_nodes = 20
        embedding_dim = 384

        tangent = torch.zeros(1, embedding_dim + 1, device=test_device)
        tangent[:, 1] = 1.0
        single_emb = LorentzOps.exp_map_zero(tangent, c=1.0)

        # Repeat to create identical embeddings
        embeddings = single_emb.repeat(n_nodes, 1)

        anchors = torch.tensor([0, 1, 2, 3], device=test_device)
        positives = torch.tensor([4, 5, 6, 7], device=test_device)
        negatives = torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]], device=test_device)

        # Should not crash (though metrics may not be meaningful)
        metrics = compute_validation_metrics(
            emb=embeddings, anchors=anchors, positives=positives, negatives=negatives, c=1.0
        )

        # Should produce some output without crashing
        assert 'avg_positive_dist' in metrics

    def test_extreme_curvature_values(
        self, sample_graph_embeddings, sample_triplet_indices
    ):
        '''Test with extreme curvature values.'''

        anchors, positives, negatives = sample_triplet_indices

        for curvature in [0.01, 100.0]:
            metrics = compute_validation_metrics(
                emb=sample_graph_embeddings,
                anchors=anchors,
                positives=positives,
                negatives=negatives,
                c=curvature,
            )

            # Should not produce NaN or Inf
            for key, value in metrics.items():
                assert not torch.isnan(torch.tensor(value)), f'{key} is NaN at c={curvature}'
                assert not torch.isinf(torch.tensor(value)), f'{key} is Inf at c={curvature}'
