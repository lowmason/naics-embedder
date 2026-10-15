'''
Unit tests for text model evaluation module.

Tests cover:
- EmbeddingEvaluator (distance and similarity computation)
- HierarchyMetrics (cophenetic correlation, Spearman correlation, distortion)
'''

import pytest
import torch

from naics_embedder.metrics import EmbeddingEvaluator, HierarchyMetrics

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings(test_device):
    '''Generate sample embeddings for testing.'''
    torch.manual_seed(42)
    return torch.randn(20, 128, device=test_device)

@pytest.fixture
def sample_lorentz_embeddings(test_device):
    '''Generate valid Lorentz embeddings for testing.'''
    torch.manual_seed(42)
    # Create embeddings on hyperboloid: x_0^2 - sum(x_i^2) = -1/c
    # Start with spatial components
    spatial = torch.randn(20, 128, device=test_device)
    spatial_norm = torch.norm(spatial, dim=1, keepdim=True)
    # Set time component to satisfy constraint
    time = torch.sqrt(1.0 + spatial_norm**2)
    return torch.cat([time, spatial], dim=1)

@pytest.fixture
def sample_tree_distances(test_device):
    '''Generate sample tree distance matrix.'''
    torch.manual_seed(42)
    # Create symmetric distance matrix
    distances = torch.rand(20, 20, device=test_device) * 5.0
    distances = (distances + distances.t()) / 2.0
    distances.fill_diagonal_(0.0)
    return distances

# -------------------------------------------------------------------------------------------------
# EmbeddingEvaluator Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingEvaluator:
    '''Test suite for EmbeddingEvaluator.'''

    def test_init(self, test_device):
        '''Test evaluator initialization.'''
        evaluator = EmbeddingEvaluator()
        # get_device() returns a string, not torch.device
        assert isinstance(evaluator.device, str)
        assert evaluator.device in ['cpu', 'cuda', 'mps']

    def test_compute_pairwise_distances_euclidean(self, sample_embeddings):
        '''Test Euclidean distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        assert distances.shape == (20, 20)
        assert torch.allclose(distances, distances.t())  # Symmetric
        # Zero diagonal
        assert torch.allclose(torch.diag(distances), torch.zeros(20, device=distances.device))

    def test_compute_pairwise_distances_cosine(self, sample_embeddings):
        '''Test cosine distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='cosine')

        assert distances.shape == (20, 20)
        # Allow small numerical errors (cosine distance should be >= -1e-6)
        assert torch.all(distances >= -1e-6)  # Non-negative (with numerical tolerance)
        assert torch.all(distances <= 2.0 + 1e-6)  # Cosine distance is in [0, 2]
        assert torch.allclose(distances, distances.t())  # Symmetric

    def test_compute_pairwise_distances_lorentz(self, sample_lorentz_embeddings):
        '''Test Lorentz distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_lorentz_embeddings, metric='lorentz', curvature=1.0
        )

        assert distances.shape == (20, 20)
        assert torch.all(distances >= 0.0)  # Non-negative
        assert torch.allclose(distances, distances.t())  # Symmetric
        # Lorentz distances: diagonal should be very small (not exactly zero due to numerical precision)
        diag = torch.diag(distances)
        assert torch.all(diag < 0.01)  # Diagonal should be very small

    def test_compute_pairwise_distances_invalid_metric(self, sample_embeddings):
        '''Test error handling for invalid metric.'''
        evaluator = EmbeddingEvaluator()
        with pytest.raises(ValueError, match='Unknown metric'):
            evaluator.compute_pairwise_distances(sample_embeddings, metric='invalid')

    def test_compute_similarity_matrix_cosine(self, sample_embeddings):
        '''Test cosine similarity computation.'''
        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(sample_embeddings, metric='cosine')

        assert similarities.shape == (20, 20)
        # Allow small numerical errors (cosine similarity should be in [-1, 1] with tolerance)
        assert torch.all(similarities >= -1.0 - 1e-6) and torch.all(similarities <= 1.0 + 1e-6)
        assert torch.allclose(similarities, similarities.t())  # Symmetric
        # Self-similarity = 1
        assert torch.allclose(
            torch.diag(similarities),
            torch.ones(20, device=similarities.device),
            atol=1e-5,
        )

    def test_compute_similarity_matrix_dot(self, sample_embeddings):
        '''Test dot product similarity computation.'''
        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(sample_embeddings, metric='dot')

        assert similarities.shape == (20, 20)
        assert torch.allclose(similarities, similarities.t())  # Symmetric

    def test_compute_similarity_matrix_invalid_metric(self, sample_embeddings):
        '''Test error handling for invalid similarity metric.'''
        evaluator = EmbeddingEvaluator()
        with pytest.raises(ValueError, match='Unknown metric'):
            evaluator.compute_similarity_matrix(sample_embeddings, metric='invalid')

# -------------------------------------------------------------------------------------------------
# HierarchyMetrics Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyMetrics:
    '''Test suite for HierarchyMetrics.'''

    def test_init(self, test_device):
        '''Test metrics initialization.'''
        metrics = HierarchyMetrics()
        # get_device() returns a string, not torch.device
        assert isinstance(metrics.device, str)
        assert metrics.device in ['cpu', 'cuda', 'mps']

    def test_cophenetic_correlation(self, sample_embeddings, sample_tree_distances):
        '''Test cophenetic correlation computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.cophenetic_correlation(emb_distances, sample_tree_distances)

        assert 'correlation' in result
        assert 'n_pairs' in result
        assert 'n_total' in result
        assert isinstance(result['correlation'], torch.Tensor)
        assert -1.0 <= result['correlation'].item() <= 1.0  # Correlation in [-1, 1]

    def test_cophenetic_correlation_with_min_distance(
        self, sample_embeddings, sample_tree_distances
    ):
        '''Test cophenetic correlation with minimum distance filter.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.cophenetic_correlation(
            emb_distances, sample_tree_distances, min_distance=1.0
        )

        assert 'correlation' in result
        assert result['n_pairs'] <= result['n_total']  # Filtered pairs should be <= total

    def test_cophenetic_correlation_insufficient_pairs(self, test_device):
        '''Test cophenetic correlation with insufficient valid pairs.'''
        metrics = HierarchyMetrics()
        # Create very small tree distances (all filtered out)
        emb_distances = torch.rand(5, 5, device=test_device)
        emb_distances.fill_diagonal_(0.0)
        tree_distances = torch.ones(5, 5, device=test_device) * 0.05  # All below threshold
        tree_distances.fill_diagonal_(0.0)

        result = metrics.cophenetic_correlation(emb_distances, tree_distances, min_distance=0.1)

        correlation = result['correlation']
        if isinstance(correlation, torch.Tensor):
            assert correlation.item() == 0.0
        else:
            assert correlation == 0.0
        assert result['n_pairs'] < 2

    def test_spearman_correlation(self, sample_embeddings, sample_tree_distances):
        '''Test Spearman correlation computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.spearman_correlation(emb_distances, sample_tree_distances)

        assert 'correlation' in result
        assert 'n_pairs' in result
        assert isinstance(result['correlation'], torch.Tensor)
        assert -1.0 <= result['correlation'].item() <= 1.0

    def test_spearman_correlation_perfect_rank(self, test_device):
        '''Test Spearman correlation with perfect rank preservation.'''
        metrics = HierarchyMetrics()
        # Create distances that perfectly preserve ranking
        tree_distances = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ],
            device=test_device,
        )

        # Embedding distances are proportional (perfect rank preservation)
        emb_distances = tree_distances * 2.0

        result = metrics.spearman_correlation(emb_distances, tree_distances)
        # Should be close to 1.0 for perfect rank preservation
        correlation = result['correlation']
        if isinstance(correlation, torch.Tensor):
            assert correlation.item() > 0.9
        else:
            assert correlation > 0.9

    def test_distortion(self, sample_embeddings, sample_tree_distances):
        '''Test distortion metrics computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.distortion(emb_distances, sample_tree_distances)

        assert 'mean_distortion' in result
        assert 'max_distortion' in result
        assert 'min_distortion' in result
        assert 'std_distortion' in result
        assert 'median_distortion' in result
        assert result['mean_distortion'] > 0.0

    def test_distortion_no_stretch(self, test_device):
        '''Test distortion with no stretching (proportional distances).'''
        metrics = HierarchyMetrics()
        tree_distances = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            device=test_device,
        )

        # Embedding distances are exactly proportional
        emb_distances = tree_distances * 2.0

        result = metrics.distortion(emb_distances, tree_distances)
        # Mean distortion should be close to 2.0 (scaling factor)
        assert abs(result['mean_distortion'].item() - 2.0) < 0.1

    def test_ndcg_ranking(self, sample_embeddings, sample_tree_distances):
        '''Test NDCG ranking computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.ndcg_ranking(emb_distances, sample_tree_distances, k_values=[5, 10])

        assert 'ndcg@5' in result
        assert 'ndcg@10' in result
        assert result['ndcg@5'] >= 0.0 and result['ndcg@5'] <= 1.0
        assert result['ndcg@10'] >= 0.0 and result['ndcg@10'] <= 1.0

    def test_ndcg_ranking_without_enough_valid_codes_is_zero_over_no_queries(self, test_device):
        '''An anchor needs max(k_values) valid codes. When none has them, each NDCG@k is a zero
        tensor (HGCN detaches it) over 0 queries.'''
        metrics = HierarchyMetrics()
        # Each anchor has 2 valid codes, fewer than max(k_values) = 3, though NDCG@2 is defined
        tree_distances = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            device=test_device,
        )

        result = metrics.ndcg_ranking(tree_distances * 2.0, tree_distances, k_values=[2, 3])

        assert set(result) == {'ndcg@2', 'ndcg@2_n_queries', 'ndcg@3', 'ndcg@3_n_queries'}
        for k in [2, 3]:
            assert isinstance(result[f'ndcg@{k}'], torch.Tensor)
            assert result[f'ndcg@{k}'].item() == 0.0
            assert result[f'ndcg@{k}_n_queries'] == 0

    def test_ndcg_ranking_counts_only_the_anchors_it_keeps(self, test_device):
        '''An anchor with fewer valid codes than max(k_values) is skipped at every k, and the
        queries counted are the anchors kept.'''
        metrics = HierarchyMetrics()
        # Anchor 0's codes 1 and 2 lie below min_distance (0.1), so it has 1 valid code, fewer than
        # max(k_values) = 2; anchors 1 and 2 have exactly 2, and anchor 3 has 3
        tree_distances = torch.tensor(
            [
                [0.0, 0.05, 0.05, 2.0],
                [0.05, 0.0, 1.0, 2.0],
                [0.05, 1.0, 0.0, 2.0],
                [2.0, 2.0, 2.0, 0.0],
            ],
            device=test_device,
        )

        # Embedding distances proportional to the tree's rank each kept anchor's codes perfectly
        result = metrics.ndcg_ranking(tree_distances * 2.0, tree_distances, k_values=[1, 2])

        for k in [1, 2]:
            assert result[f'ndcg@{k}'].item() == pytest.approx(1.0)
            assert result[f'ndcg@{k}_n_queries'] == 3

# -------------------------------------------------------------------------------------------------
# Level Consistency Tests (Per-level embedding quality)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLevelConsistency:
    '''Test suite for per-level embedding quality metrics.

    Tests that embeddings maintain quality and consistency across different
    NAICS hierarchy levels (2-digit through 6-digit codes).
    '''

    @pytest.fixture
    def level_embeddings(self, test_device):
        '''Generate embeddings for each hierarchy level.

        Returns dict mapping level -> embeddings tensor.
        '''
        torch.manual_seed(42)
        return {
            2: torch.randn(5, 128, device=test_device),  # 5 sector codes
            3: torch.randn(10, 128, device=test_device),  # 10 subsector codes
            4: torch.randn(15, 128, device=test_device),  # 15 industry group codes
            5: torch.randn(20, 128, device=test_device),  # 20 industry codes
            6: torch.randn(25, 128, device=test_device),  # 25 national codes
        }

    @pytest.fixture
    def level_tree_distances(self, test_device):
        '''Generate tree distances for each hierarchy level.'''
        torch.manual_seed(42)
        return {
            2: torch.rand(5, 5, device=test_device) * 2.0,
            3: torch.rand(10, 10, device=test_device) * 3.0,
            4: torch.rand(15, 15, device=test_device) * 4.0,
            5: torch.rand(20, 20, device=test_device) * 5.0,
            6: torch.rand(25, 25, device=test_device) * 6.0,
        }

    def test_level_consistency_hierarchy_metrics_per_level(
        self, level_embeddings, level_tree_distances
    ):
        '''Test hierarchy preservation metrics for each level.'''
        evaluator = EmbeddingEvaluator()
        hierarchy_metrics = HierarchyMetrics()

        for level in [2, 3, 4, 5, 6]:
            embeddings = level_embeddings[level]
            tree_dists = level_tree_distances[level]

            # Make tree distances symmetric with zero diagonal
            tree_dists = (tree_dists + tree_dists.t()) / 2
            tree_dists.fill_diagonal_(0.0)

            emb_distances = evaluator.compute_pairwise_distances(embeddings, metric='euclidean')
            result = hierarchy_metrics.cophenetic_correlation(
                emb_distances, tree_dists, min_distance=0.05
            )

            assert 'correlation' in result
            assert 'n_pairs' in result
            # Correlation should be in valid range
            corr = result['correlation']
            if isinstance(corr, torch.Tensor):
                assert -1.0 <= corr.item() <= 1.0
            else:
                assert -1.0 <= corr <= 1.0

    def test_level_consistency_cross_level_coherence(self, test_device):
        '''Test that embeddings from different levels maintain coherent relationships.

        Parent codes should generally be closer to their children than to
        codes from different branches in a well-trained embedding space.
        '''
        torch.manual_seed(42)
        evaluator = EmbeddingEvaluator()

        # Create parent and child embeddings with expected relationship
        # Parent embedding
        parent = torch.randn(1, 128, device=test_device)
        # Child embeddings (similar to parent with some noise)
        children = parent + torch.randn(3, 128, device=test_device) * 0.5
        # Unrelated codes (different random embeddings)
        unrelated = torch.randn(3, 128, device=test_device) * 2.0

        all_embeddings = torch.cat([parent, children, unrelated], dim=0)
        distances = evaluator.compute_pairwise_distances(all_embeddings, metric='euclidean')

        # Distance from parent (idx 0) to children (idx 1,2,3) should generally be
        # smaller than to unrelated (idx 4,5,6)
        parent_to_children = distances[0, 1:4].mean()
        parent_to_unrelated = distances[0, 4:7].mean()

        # In a well-structured embedding, children should be closer
        # (with high probability given our construction)
        assert parent_to_children < parent_to_unrelated
