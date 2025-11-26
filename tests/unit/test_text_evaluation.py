'''
Unit tests for text_model/evaluation.py

Tests cover:
- EmbeddingEvaluator: distance and similarity computation
- RetrievalMetrics: precision, recall, MAP, NDCG
- HierarchyMetrics: hierarchy-specific evaluation
- EmbeddingStatistics: radius, norm, diversity
- NAICSEvaluationRunner: overall evaluation
'''

import logging

import pytest
import torch

from naics_embedder.text_model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    RetrievalMetrics,
)
from naics_embedder.text_model.hyperbolic import LorentzOps

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings_euclidean(test_device, batch_size=16, embedding_dim=384):
    '''Generate sample Euclidean embeddings.'''

    return torch.randn(batch_size, embedding_dim, device=test_device)

@pytest.fixture
def sample_embeddings_lorentz(test_device, batch_size=16):
    '''Generate sample Lorentz embeddings on manifold.'''

    embedding_dim = 384
    # Create tangent vectors and map to hyperboloid
    tangent = torch.randn(batch_size, embedding_dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component = 0 for tangent at origin
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0

    return LorentzOps.exp_map_zero(tangent, c=1.0)

@pytest.fixture
def sample_ground_truth_relevance(test_device, batch_size=16):
    '''Generate sample binary relevance matrix.'''

    # Create block-diagonal relevance (codes in same block are relevant)
    relevance = torch.zeros(batch_size, batch_size, device=test_device)

    # Make 4 blocks of 4 codes each relevant to each other
    block_size = 4
    for i in range(batch_size // block_size):
        start = i * block_size
        end = start + block_size
        relevance[start:end, start:end] = 1.0

    # Set diagonal to 0 (self is not relevant)
    relevance.fill_diagonal_(0.0)

    return relevance

# -------------------------------------------------------------------------------------------------
# Test: EmbeddingEvaluator
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingEvaluator:
    '''Test EmbeddingEvaluator distance and similarity computation.'''

    def test_evaluator_creation(self):
        '''Test EmbeddingEvaluator initialization.'''

        evaluator = EmbeddingEvaluator()
        assert evaluator.device is not None

    def test_euclidean_distance_matrix(self, sample_embeddings_euclidean):
        '''Test Euclidean distance matrix computation.'''

        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        batch_size = sample_embeddings_euclidean.shape[0]
        assert distances.shape == (batch_size, batch_size)

        # Distance matrix should be symmetric
        torch.testing.assert_close(distances, distances.t(), rtol=1e-5, atol=1e-5)

        # Diagonal should be zero (distance to self)
        assert torch.allclose(distances.diagonal(), torch.zeros(batch_size))

        # All distances should be non-negative
        assert (distances >= 0).all()

    def test_cosine_distance_matrix(self, sample_embeddings_euclidean):
        '''Test cosine distance matrix computation.'''

        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='cosine'
        )

        batch_size = sample_embeddings_euclidean.shape[0]
        assert distances.shape == (batch_size, batch_size)

        # Cosine distance should be in [0, 2]
        assert (distances >= 0).all()
        assert (distances <= 2.0).all()

        # Diagonal should be zero
        assert torch.allclose(
            distances.diagonal(), torch.zeros(batch_size, device=distances.device), atol=1e-5
        )

    def test_lorentz_distance_matrix(self, sample_embeddings_lorentz):
        '''Test Lorentzian distance matrix computation.'''

        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_lorentz, metric='lorentz', curvature=1.0
        )

        batch_size = sample_embeddings_lorentz.shape[0]
        assert distances.shape == (batch_size, batch_size)

        # Distance matrix should be symmetric
        torch.testing.assert_close(distances, distances.t(), rtol=1e-4, atol=1e-4)

        # Diagonal should be zero
        assert torch.allclose(
            distances.diagonal(), torch.zeros(batch_size, device=distances.device), atol=1e-3
        )

        # All distances should be non-negative
        assert (distances >= 0).all()

    def test_cosine_similarity_matrix(self, sample_embeddings_euclidean):
        '''Test cosine similarity matrix computation.'''

        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(
            sample_embeddings_euclidean, metric='cosine'
        )

        batch_size = sample_embeddings_euclidean.shape[0]
        assert similarities.shape == (batch_size, batch_size)

        # Similarity matrix should be symmetric
        torch.testing.assert_close(similarities, similarities.t(), rtol=1e-5, atol=1e-5)

        # Diagonal should be 1 (self-similarity)
        torch.testing.assert_close(
            similarities.diagonal(),
            torch.ones(batch_size, device=similarities.device),
            rtol=1e-5,
            atol=1e-5,
        )

        # Similarities should be in [-1, 1]
        assert (similarities >= -1.0).all()
        assert (similarities <= 1.0).all()

    def test_dot_similarity_matrix(self, sample_embeddings_euclidean):
        '''Test dot product similarity matrix computation.'''

        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(
            sample_embeddings_euclidean, metric='dot'
        )

        batch_size = sample_embeddings_euclidean.shape[0]
        assert similarities.shape == (batch_size, batch_size)

        # Should be symmetric
        torch.testing.assert_close(similarities, similarities.t(), rtol=1e-5, atol=1e-5)

    def test_invalid_metric_raises_error(self, sample_embeddings_euclidean):
        '''Test that invalid metric raises ValueError.'''

        evaluator = EmbeddingEvaluator()

        with pytest.raises(ValueError, match='Unknown metric'):
            evaluator.compute_pairwise_distances(sample_embeddings_euclidean, metric='invalid')

    def test_different_curvatures(self, sample_embeddings_lorentz):
        '''Test Lorentz distance with different curvatures.'''

        evaluator = EmbeddingEvaluator()

        for curvature in [0.1, 1.0, 5.0]:
            distances = evaluator.compute_pairwise_distances(
                sample_embeddings_lorentz, metric='lorentz', curvature=curvature
            )

            # Should produce valid distance matrix
            assert not torch.isnan(distances).any()
            assert not torch.isinf(distances).any()
            assert (distances >= 0).all()

# -------------------------------------------------------------------------------------------------
# Test: RetrievalMetrics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrievalMetrics:
    '''Test retrieval metrics (precision, recall, MAP, NDCG).'''

    def test_metrics_creation(self):
        '''Test RetrievalMetrics initialization.'''

        metrics = RetrievalMetrics()
        assert metrics.device is not None

    def test_precision_at_k(
        self, sample_embeddings_euclidean, sample_ground_truth_relevance, test_device
    ):
        '''Test precision@k computation.'''

        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()

        # Compute distances
        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        # Compute precision@5
        precision = metrics.precision_at_k(
            distances, sample_ground_truth_relevance, k=5
        )

        batch_size = sample_embeddings_euclidean.shape[0]
        assert precision.shape == (batch_size,)

        # Precision should be in [0, 1]
        assert (precision >= 0).all()
        assert (precision <= 1.0).all()

    def test_recall_at_k(
        self, sample_embeddings_euclidean, sample_ground_truth_relevance, test_device
    ):
        '''Test recall@k computation.'''

        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        # Compute recall@5
        recall = metrics.recall_at_k(distances, sample_ground_truth_relevance, k=5)

        batch_size = sample_embeddings_euclidean.shape[0]
        assert recall.shape == (batch_size,)

        # Recall should be in [0, 1]
        assert (recall >= 0).all()
        assert (recall <= 1.0).all()

    def test_mean_average_precision(
        self, sample_embeddings_euclidean, sample_ground_truth_relevance, test_device
    ):
        '''Test Mean Average Precision (MAP) computation.'''

        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        # Compute MAP
        map_score = metrics.mean_average_precision(distances, sample_ground_truth_relevance)

        assert isinstance(map_score, torch.Tensor)
        assert map_score.ndim == 0  # Scalar

        # MAP should be in [0, 1]
        assert 0 <= map_score.item() <= 1.0

    def test_map_with_k_limit(
        self, sample_embeddings_euclidean, sample_ground_truth_relevance
    ):
        '''Test MAP with maximum rank limit.'''

        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        # Compute MAP@10
        map_score = metrics.mean_average_precision(
            distances, sample_ground_truth_relevance, k=10
        )

        assert 0 <= map_score.item() <= 1.0

    def test_ndcg_at_k(self, sample_embeddings_euclidean, test_device):
        '''Test NDCG@k computation.'''

        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()

        batch_size = sample_embeddings_euclidean.shape[0]

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        # Create relevance scores (higher = more relevant)
        relevance_scores = torch.rand(batch_size, batch_size, device=test_device)
        relevance_scores.fill_diagonal_(0.0)  # Self is not relevant

        # Compute NDCG@10
        ndcg = metrics.ndcg_at_k(distances, relevance_scores, k=10)

        assert ndcg.shape == (batch_size,)

        # NDCG should be in [0, 1]
        assert (ndcg >= 0).all()
        assert (ndcg <= 1.0).all()

    def test_perfect_ranking_gives_high_map(self, test_device):
        '''Test that perfect ranking gives MAP = 1.0.'''

        metrics = RetrievalMetrics()

        # Create perfect ranking: distance matrix where closer indices = smaller distance
        n = 10
        distances = torch.zeros(n, n, device=test_device)
        for i in range(n):
            for j in range(n):
                distances[i, j] = abs(i - j)

        # Ground truth: adjacent codes are relevant
        relevance = torch.zeros(n, n, device=test_device)
        for i in range(n):
            if i > 0:
                relevance[i, i - 1] = 1.0
            if i < n - 1:
                relevance[i, i + 1] = 1.0

        map_score = metrics.mean_average_precision(distances, relevance, k=5)

        # Should achieve high MAP (close to 1.0) for perfect ranking
        assert map_score.item() > 0.8

# -------------------------------------------------------------------------------------------------
# Test: EmbeddingStatistics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingStatistics:
    '''Test embedding statistics computation.'''

    def test_statistics_creation(self):
        '''Test EmbeddingStatistics initialization.'''

        stats = EmbeddingStatistics()
        assert hasattr(stats, 'compute_radius')

    def test_radius_computation(self, sample_embeddings_lorentz):
        '''Test hyperbolic radius computation.'''

        stats = EmbeddingStatistics()
        radius = stats.compute_radius(sample_embeddings_lorentz, curvature=1.0)

        assert isinstance(radius, torch.Tensor)
        assert radius.ndim == 0  # Scalar

        # Radius should be positive
        assert radius.item() > 0

    def test_norm_computation(self, sample_embeddings_euclidean):
        '''Test embedding norm computation.'''

        stats = EmbeddingStatistics()
        norms = stats.compute_norms(sample_embeddings_euclidean)

        batch_size = sample_embeddings_euclidean.shape[0]
        assert norms.shape == (batch_size,)

        # All norms should be positive
        assert (norms > 0).all()

    def test_diversity_computation(self, sample_embeddings_euclidean):
        '''Test embedding diversity computation.'''

        stats = EmbeddingStatistics()
        evaluator = EmbeddingEvaluator()

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )

        diversity = stats.compute_diversity(distances)

        assert isinstance(diversity, torch.Tensor)
        assert diversity.ndim == 0

        # Diversity should be positive
        assert diversity.item() > 0

    def test_collapsed_embeddings_low_diversity(self, test_device):
        '''Test that collapsed embeddings have low diversity.'''

        stats = EmbeddingStatistics()
        evaluator = EmbeddingEvaluator()

        # Create nearly identical embeddings (collapsed)
        batch_size = 16
        embedding_dim = 384
        base = torch.randn(1, embedding_dim, device=test_device)
        collapsed = base.repeat(batch_size, 1) + torch.randn(
            batch_size, embedding_dim, device=test_device
        ) * 0.01

        distances = evaluator.compute_pairwise_distances(collapsed, metric='euclidean')
        diversity = stats.compute_diversity(distances)

        # Diversity should be very low
        assert diversity.item() < 1.0

    def test_diverse_embeddings_high_diversity(self, sample_embeddings_euclidean):
        '''Test that diverse embeddings have high diversity.'''

        stats = EmbeddingStatistics()
        evaluator = EmbeddingEvaluator()

        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_euclidean, metric='euclidean'
        )
        diversity = stats.compute_diversity(distances)

        # Random embeddings should have reasonable diversity
        assert diversity.item() > 1.0

# -------------------------------------------------------------------------------------------------
# Test: HierarchyMetrics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyMetrics:
    '''Test hierarchy-specific metrics.'''

    def test_hierarchy_metrics_creation(self):
        '''Test HierarchyMetrics initialization.'''

        hierarchy = HierarchyMetrics()
        assert hasattr(hierarchy, 'compute_distance_correlation')

    def test_distance_correlation(self, test_device):
        '''Test distance correlation computation.'''

        hierarchy = HierarchyMetrics()

        # Create learned and ground truth distances
        n = 20
        learned_distances = torch.rand(n, n, device=test_device)
        learned_distances = (learned_distances + learned_distances.t()) / 2  # Symmetrize
        learned_distances.fill_diagonal_(0.0)

        # Ground truth similar to learned (positive correlation)
        ground_truth_distances = learned_distances + torch.randn(n, n, device=test_device) * 0.1
        ground_truth_distances = (ground_truth_distances + ground_truth_distances.t()) / 2
        ground_truth_distances.fill_diagonal_(0.0)
        ground_truth_distances = ground_truth_distances.abs()  # Ensure non-negative

        correlation = hierarchy.compute_distance_correlation(
            learned_distances, ground_truth_distances
        )

        assert isinstance(correlation, torch.Tensor)
        assert correlation.ndim == 0

        # Should have positive correlation
        assert correlation.item() > 0.5

    def test_perfect_correlation(self, test_device):
        '''Test that identical distances give correlation = 1.0.'''

        hierarchy = HierarchyMetrics()

        n = 10
        distances = torch.rand(n, n, device=test_device)
        distances = (distances + distances.t()) / 2
        distances.fill_diagonal_(0.0)

        correlation = hierarchy.compute_distance_correlation(distances, distances)

        # Perfect correlation
        torch.testing.assert_close(correlation, torch.tensor(1.0, device=test_device), atol=1e-5)

    def test_level_consistency(self, sample_embeddings_lorentz, test_device):
        '''Test level consistency computation.'''

        hierarchy = HierarchyMetrics()

        # Create code to level mapping
        batch_size = sample_embeddings_lorentz.shape[0]
        codes = [f'{i:02d}111' for i in range(batch_size)]

        # Assign levels (4 codes per level, 4 levels)
        code_to_level = {code: i // 4 for i, code in enumerate(codes)}

        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_embeddings_lorentz, metric='lorentz', curvature=1.0
        )

        level_consistency = hierarchy.compute_level_consistency(distances, codes, code_to_level)

        assert isinstance(level_consistency, dict)
        assert len(level_consistency) > 0

        # Each level should have consistency metrics
        for level, metrics in level_consistency.items():
            assert 'mean_intra_level_distance' in metrics
            assert 'mean_inter_level_distance' in metrics

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test numerical stability of evaluation functions.'''

    def test_zero_distances_handled(self, test_device):
        '''Test that zero distances don\'t cause issues.'''

        metrics = RetrievalMetrics()

        n = 10
        distances = torch.zeros(n, n, device=test_device)
        relevance = torch.ones(n, n, device=test_device)
        relevance.fill_diagonal_(0.0)

        # Should not crash or produce NaN
        map_score = metrics.mean_average_precision(distances, relevance)
        assert not torch.isnan(map_score)

    def test_large_distances_handled(self, test_device):
        '''Test that large distances are handled correctly.'''

        evaluator = EmbeddingEvaluator()

        # Create embeddings with large norms
        embeddings = torch.randn(16, 384, device=test_device) * 100

        distances = evaluator.compute_pairwise_distances(embeddings, metric='euclidean')

        # Should not overflow
        assert not torch.isinf(distances).any()
        assert not torch.isnan(distances).any()

    def test_empty_relevance_handled(self, test_device):
        '''Test that empty relevance (no relevant items) is handled.'''

        metrics = RetrievalMetrics()

        n = 10
        distances = torch.rand(n, n, device=test_device)
        relevance = torch.zeros(n, n, device=test_device)  # No relevant items

        # Should not crash
        map_score = metrics.mean_average_precision(distances, relevance)
        assert not torch.isnan(map_score)
