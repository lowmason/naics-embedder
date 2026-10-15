# -------------------------------------------------------------------------------------------------
# Core Metrics Classes
# -------------------------------------------------------------------------------------------------
'''
Core metrics classes for embedding evaluation.

Contains:
- lorentz_distance_matrix: Float64 pairwise Lorentz distances for evaluation
- EmbeddingEvaluator: Float32 pairwise Lorentz distances on the evaluation device
- HierarchyMetrics: Cophenetic/Spearman correlation, NDCG ranking, distortion
'''

import logging
from typing import Dict, List, Union

import torch

from naics_embedder.metrics.structural_spearman import (
    StructuralSpearmanResult,
    compute_structural_spearman,
)
from naics_embedder.utils.backend import get_device

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Embedding similarity and distance utilities
# -------------------------------------------------------------------------------------------------

def lorentz_distance_matrix(embeddings: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    '''
    Pairwise Lorentz distances in float64 on the CPU, for evaluation (no gradients flow).

    Far from the origin, -<x, y>_L = x0 * y0 - <xs, ys> cancels catastrophically for nearby
    points, and acosh near 1 amplifies the error. Float64 arithmetic alone is not enough: float32
    rounding of the stored x0 already moves -<x, x>_L off 1/c by up to x0 * ulp(x0), about 5e-5
    at radius 4, i.e. distances of about 1e-2. So x0 is re-derived in float64 from the spatial
    coordinates, which assumes the points lie on the curvature-c hyperboloid.

    Like every distance formula here, sqrt(c) * acosh(-<x, y>_L) is only correct for c = 1;
    see "One curvature everywhere" in specs/naics-embedding-methodology.md.

    Args:
        embeddings: Hyperbolic embeddings of shape (N, D+1), on any device
        curvature: Curvature parameter c

    Returns:
        Float64 CPU distance matrix of shape (N, N), exactly symmetric with a zero diagonal
    '''
    # Move before casting, as MPS has no float64. One fixed layout makes the reductions round
    # identically for C- and Fortran-ordered inputs.
    emb = embeddings.detach().cpu().to(torch.float64).contiguous()
    spatial = emb[:, 1:]
    gram = spatial @ spatial.T
    # BLAS need not return A @ A.T bitwise symmetric, and structural Spearman rejects asymmetric
    # distance matrices. Averaging leaves the diagonal unchanged.
    gram = (gram + gram.T) / 2
    # Taking x0 from the same product as the dot products lets copies of a point at different
    # indices cancel as closely as the point does with itself.
    time = torch.sqrt(1.0 / curvature + gram.diagonal())
    neg_dot = torch.outer(time, time) - gram
    # Clamp only to acosh's domain: a 1 + eps floor would also floor every distance at
    # acosh(1 + eps), e.g. 4.5e-3 for eps = 1e-5.
    distances = curvature**0.5 * torch.acosh(torch.clamp(neg_dot, min=1.0))
    distances.fill_diagonal_(0.0)
    return distances

class EmbeddingEvaluator:

    def __init__(self):
        '''
        Evaluator for embedding quality metrics.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def compute_pairwise_distances(
        self, embeddings: torch.Tensor, metric: str = 'lorentz', curvature: float = 1.0
    ) -> torch.Tensor:
        '''
        Compute pairwise Lorentz distances between embeddings.

        HGCN's hyperbolic points are the only input; a flat arm's distances come from
        ``panels.decoding``.

        Args:
            embeddings: Lorentz embeddings of shape (N, D+1)
            metric: Distance metric; only 'lorentz' is accepted
            curvature: Curvature parameter for the Lorentz metric (default: 1.0)

        Returns:
            Float32 distance matrix of shape (N, N) on self.device

        Raises:
            ValueError: If metric is not 'lorentz'.
        '''

        if metric != 'lorentz':
            raise ValueError(f'Unknown metric: {metric}')
        # Computed in float64 from the embeddings as given, then cast for the hierarchy metrics
        distances = lorentz_distance_matrix(embeddings, curvature=curvature)
        return distances.to(device=self.device, dtype=torch.float32)

# -------------------------------------------------------------------------------------------------
# Hierarchy preservation metrics
# -------------------------------------------------------------------------------------------------

class HierarchyMetrics:

    def __init__(self):
        '''
        Metrics for evaluating hierarchy preservation.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def cophenetic_correlation(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        min_distance: float = 0.1,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        '''
        Compute cophenetic correlation coefficient with better handling.

        Measures how well embedding distances preserve hierarchical tree distances.
        Filters out pairs with very small tree distances to avoid noise.

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)
            min_distance: Minimum tree distance to include (filters out same-level codes)

        Returns:
            Dictionary with correlation and metadata
        '''

        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        # Get upper triangular indices (excluding diagonal)
        N = embedding_distances.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1, device=self.device)

        # Extract upper triangular values
        emb_dists = embedding_distances[triu_indices[0], triu_indices[1]]
        tree_dists = tree_distances[triu_indices[0], triu_indices[1]]

        # Filter out pairs with very small tree distances
        valid_mask = tree_dists >= min_distance
        emb_dists_filtered = emb_dists[valid_mask]
        tree_dists_filtered = tree_dists[valid_mask]

        # Check if we have enough valid pairs
        if len(emb_dists_filtered) < 2:
            return {
                'correlation': torch.tensor(0.0, device=self.device),
                'n_pairs': len(emb_dists_filtered),
                'n_total': len(emb_dists),
                'mean_tree_dist': tree_dists.mean(),
                'mean_emb_dist': emb_dists.mean(),
            }

        # Compute Pearson correlation on filtered pairs
        correlation = self._pearson_correlation(emb_dists_filtered, tree_dists_filtered)

        return {
            'correlation': correlation,
            'n_pairs': len(emb_dists_filtered),
            'n_total': len(emb_dists),
            'mean_tree_dist': tree_dists_filtered.mean(),
            'mean_emb_dist': emb_dists_filtered.mean(),
        }

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Compute Pearson correlation coefficient.

        Args:
            x: First variable (shape: N)
            y: Second variable (shape: N)

        Returns:
            Correlation coefficient (scalar)
        '''

        x_mean = x.mean()
        y_mean = y.mean()

        x_centered = x - x_mean
        y_centered = y - y_mean

        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())

        denominator = torch.clamp(denominator, min=1e-10)
        correlation = numerator / denominator

        return correlation

    def spearman_correlation(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        min_distance: float = 0.1,
    ) -> StructuralSpearmanResult:
        '''Compute structural-spearman-v1 over unique unordered non-self pairs.

        Inputs must be same-shaped square, real numeric tensors. The calculation
        validates both off-diagonal orientations, averages mirrored values on CPU
        in float64, filters canonical targets >= min_distance, and uses SciPy's
        average ranks for exact ties. Diagonal values are ignored.

        Returns:
            A detached float32 scalar on self.device, pair counts, definition,
            status, and reason. Undefined correlations are NaN with an explicit
            reason; malformed input raises StructuralMetricInputError.
        '''
        result = compute_structural_spearman(
            embedding_distances, tree_distances, min_distance=min_distance
        )
        return {
            'correlation': torch.tensor(
                result.correlation, dtype=torch.float32, device=self.device
            ),
            'n_pairs': result.n_pairs,
            'n_total': result.n_total,
            'definition': result.definition,
            'status': result.status,
            'reason': result.reason,
        }

    def ndcg_ranking(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        k_values: List[int] = [5, 10, 20],
        min_distance: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        '''
        Compute NDCG@k for ranking evaluation.

        For each anchor code, ranks all other codes by embedding distance and
        evaluates using NDCG based on tree distance relevance.

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)
            k_values: List of k values for NDCG@k
            min_distance: Minimum tree distance to consider

        Returns:
            Dictionary with NDCG@k for each k value
        '''
        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        N = embedding_distances.shape[0]
        results = {}

        # For each anchor, compute NDCG
        all_ndcg_scores = {k: [] for k in k_values}

        for anchor_idx in range(N):
            # Get distances from anchor to all others
            anchor_emb_dists = embedding_distances[anchor_idx]  # (N,)
            anchor_tree_dists = tree_distances[anchor_idx]  # (N,)

            # Filter out self and very small tree distances
            not_self = torch.arange(N, device=self.device) != anchor_idx
            valid_mask = (anchor_tree_dists >= min_distance) & not_self

            if valid_mask.sum() < max(k_values):
                continue

            # Get valid indices
            valid_indices = torch.where(valid_mask)[0]

            # Get distances and tree distances for valid codes
            valid_emb_dists = anchor_emb_dists[valid_indices]
            valid_tree_dists = anchor_tree_dists[valid_indices]

            # Convert tree distances to relevance scores
            # Lower tree distance = higher relevance
            max_tree_dist = valid_tree_dists.max()
            relevance_scores = (max_tree_dist - valid_tree_dists + 1e-6) / (max_tree_dist + 1e-6)

            # Compute NDCG@k for each k
            for k in k_values:
                k_actual = min(k, len(valid_emb_dists))

                # Sort by embedding distance (ascending = most relevant first)
                _, sorted_indices = torch.sort(valid_emb_dists, descending=False)
                sorted_relevance = relevance_scores[sorted_indices[:k_actual]]

                # Compute DCG
                positions = torch.arange(1, k_actual + 1, dtype=torch.float32, device=self.device)
                dcg = torch.sum(sorted_relevance / torch.log2(positions + 1))

                # Compute ideal DCG (sort by relevance descending)
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_dcg = torch.sum(ideal_relevance[:k_actual] / torch.log2(positions + 1))

                # NDCG = DCG / IDCG
                if ideal_dcg > 0:
                    ndcg = dcg / ideal_dcg
                    all_ndcg_scores[k].append(ndcg)

        # Average NDCG across all anchors
        for k in k_values:
            if len(all_ndcg_scores[k]) > 0:
                results[f'ndcg@{k}'] = torch.stack(all_ndcg_scores[k]).mean()
                results[f'ndcg@{k}_n_queries'] = len(all_ndcg_scores[k])
            else:
                results[f'ndcg@{k}'] = torch.tensor(0.0, device=self.device)
                results[f'ndcg@{k}_n_queries'] = 0

        return results

    def distortion(self, embedding_distances: torch.Tensor,
                   tree_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Compute distortion metrics (how much distances are stretched/compressed).

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)

        Returns:
            Dictionary with distortion metrics
        '''

        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        # Get upper triangular values
        N = embedding_distances.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1, device=self.device)

        emb_dists = embedding_distances[triu_indices[0], triu_indices[1]]
        tree_dists = tree_distances[triu_indices[0], triu_indices[1]]

        # Avoid division by zero
        tree_dists = torch.clamp(tree_dists, min=1e-10)

        # Distortion ratio
        ratios = emb_dists / tree_dists

        return {
            'mean_distortion': ratios.mean(),
            'max_distortion': ratios.max(),
            'min_distortion': ratios.min(),
            'std_distortion': ratios.std(),
            'median_distortion': ratios.median(),
        }
