# -------------------------------------------------------------------------------------------------
# Evaluation Runner
# -------------------------------------------------------------------------------------------------
'''
Complete evaluation runner for NAICS embeddings.

Contains:
- NAICSEvaluationRunner: Comprehensive evaluation runner that uses all metric classes
'''

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from naics_embedder.metrics.core import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    RetrievalMetrics,
)
from naics_embedder.utils.backend import get_device

if TYPE_CHECKING:
    from naics_embedder.text_model.naics_model import NAICSContrastiveModel

logger = logging.getLogger(__name__)

class NAICSEvaluationRunner:

    def __init__(self, model: 'NAICSContrastiveModel'):
        '''
        Complete evaluation runner for NAICS embeddings.

        Args:
            model: Trained NAICSContrastiveModel instance

        Device is automatically detected via get_device().
        '''

        self.model = model
        self.device, _, _ = get_device()

        self.embedding_eval = EmbeddingEvaluator()
        self.retrieval_metrics = RetrievalMetrics()
        self.hierarchy_metrics = HierarchyMetrics()
        self.embedding_stats = EmbeddingStatistics()

    def evaluate(
        self,
        embeddings: torch.Tensor,
        tree_distances: Optional[torch.Tensor] = None,
        ground_truth_relevance: Optional[torch.Tensor] = None,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict[str, Any]:
        '''
        Run comprehensive evaluation.

        Args:
            embeddings: Learned embeddings (N, D)
            tree_distances: Ground truth tree distances (N, N), optional
            ground_truth_relevance: Binary relevance matrix (N, N), optional
            k_values: k values for precision@k and recall@k

        Returns:
            Dictionary of all evaluation metrics
        '''

        results = {}

        # Embedding statistics
        logger.info('Computing embedding statistics...')
        results['statistics'] = self.embedding_stats.compute_statistics(embeddings)
        results['collapse_check'] = self.embedding_stats.check_collapse(embeddings)

        # Compute embedding distances
        logger.info('Computing pairwise distances...')
        emb_distances = self.embedding_eval.compute_pairwise_distances(
            embeddings, metric='euclidean'
        )

        # Hierarchy preservation
        if tree_distances is not None:
            logger.info('Evaluating hierarchy preservation...')
            results['cophenetic_correlation'] = self.hierarchy_metrics.cophenetic_correlation(
                emb_distances, tree_distances
            )
            results['spearman_correlation'] = self.hierarchy_metrics.spearman_correlation(
                emb_distances, tree_distances
            )
            results['distortion'] = self.hierarchy_metrics.distortion(emb_distances, tree_distances)

        # Retrieval metrics
        if ground_truth_relevance is not None:
            logger.info('Computing retrieval metrics...')
            results['retrieval'] = {}

            for k in k_values:
                precision = self.retrieval_metrics.precision_at_k(
                    emb_distances, ground_truth_relevance, k
                )
                recall = self.retrieval_metrics.recall_at_k(
                    emb_distances, ground_truth_relevance, k
                )

                results['retrieval'][f'precision@{k}'] = precision.mean()
                results['retrieval'][f'recall@{k}'] = recall.mean()

            results['retrieval']['map'] = self.retrieval_metrics.mean_average_precision(
                emb_distances, ground_truth_relevance
            )

        return results
