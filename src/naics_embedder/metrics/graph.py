# -------------------------------------------------------------------------------------------------
# Graph Model Metrics
# -------------------------------------------------------------------------------------------------
'''
Graph-specific metrics.

Contains:
- compute_validation_metrics: Triplet-based validation metrics for hyperbolic embeddings
- GraphEmbeddingDataset: Container for hyperbolic graph embeddings
'''

import logging
from dataclasses import dataclass
from typing import Dict, Sequence, Union

import polars as pl
import torch

from naics_embedder.utils.utilities import (
    STAGE3_EMBEDDING_PREFIX,
    STAGE4_EMBEDDING_PREFIX,
    sorted_embedding_columns,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Validation Metrics
# -------------------------------------------------------------------------------------------------

def _anchor_distances(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    c: float,
) -> torch.Tensor:
    '''Float64 CPU Lorentz distances from each anchor to each of its candidates.

    As in lorentz_distance_matrix, x0 is re-derived in float64 from the spatial coordinates:
    float32 rounding of the stored x0 alone puts distances of about 1e-2 on pairs that are close
    far from the origin.

    Args:
        emb: Embeddings of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        candidates: Indices to measure from each anchor, shape ``(batch_size, m)``.
        c: Curvature parameter.

    Returns:
        Distances of shape ``(batch_size, m)``.
    '''
    # Move before casting, as MPS has no float64. One fixed layout makes the reductions round
    # identically for C- and Fortran-ordered inputs.
    points = emb.detach().cpu().to(torch.float64).contiguous()
    spatial = points[:, 1:]
    anchors = anchors.cpu()
    candidates = candidates.cpu()
    # Squared norms and dot products share one reduction, so copies of a point at different
    # indices cancel as closely as the point does with itself.
    time = torch.sqrt(1.0 / c + (spatial * spatial).sum(dim=1))
    dot = (spatial[candidates] * spatial[anchors].unsqueeze(1)).sum(dim=-1)
    neg_dot = time[candidates] * time[anchors].unsqueeze(1) - dot
    return c**0.5 * torch.acosh(torch.clamp(neg_dot, min=1.0))

def compute_validation_metrics(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    c: float = 1.0,
    top_k: int = 1,
    *,
    as_tensors: bool = False,
) -> Union[Dict[str, float], Dict[str, torch.Tensor]]:
    '''Compute validation metrics for hyperbolic embeddings.

    Distances are computed in float64 on the CPU (see ``_anchor_distances``). Like every distance
    formula here, sqrt(c) * acosh(-<x, y>_L) is only correct for c = 1.

    Args:
        emb: Embeddings tensor of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        positives: Positive indices, shape ``(batch_size,)``.
        negatives: Negative indices, shape ``(batch_size, k_negatives)``.
        c: Curvature parameter (default: 1.0).
        top_k: Number of top negatives to consider for auxiliary accuracy.
        as_tensors: Return torch scalars instead of Python floats (for Lightning logging).

    Returns:
        Mapping of metric names to values: float32 tensors on ``emb``'s device, or floats.
    '''
    k_negatives = negatives.size(1)
    effective_top_k = max(1, min(top_k, k_negatives))

    # Column 0 holds each anchor's distance to its positive, the rest to its negatives.
    candidates = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    all_dists_per_anchor = _anchor_distances(emb, anchors, candidates, c)
    positive_dist = all_dists_per_anchor[:, 0]
    negative_dist = all_dists_per_anchor[:, 1:]

    avg_positive_dist = positive_dist.mean()
    avg_negative_dist = negative_dist.mean()

    all_distances = torch.cat([positive_dist, negative_dist.reshape(-1)], dim=0)
    distance_spread = torch.div(all_distances.std(), all_distances.mean().clamp_min(1e-8))

    relation_accuracy = (positive_dist.unsqueeze(1) < negative_dist).all(dim=1).float().mean()

    closest_negatives = torch.topk(negative_dist, k=effective_top_k, dim=1, largest=False).values
    top_k_relation_accuracy = (positive_dist.unsqueeze(1)
                               < closest_negatives).all(dim=1).float().mean()

    order = torch.argsort(all_dists_per_anchor, dim=1)
    positive_rank_tensor = torch.argmax((order == 0).int(), dim=1)
    mean_positive_rank = positive_rank_tensor.float().mean()

    metrics = {
        'avg_positive_dist': avg_positive_dist,
        'avg_negative_dist': avg_negative_dist,
        'distance_spread': distance_spread,
        'relation_accuracy': relation_accuracy,
        'top_k_relation_accuracy': top_k_relation_accuracy,
        'mean_positive_rank': mean_positive_rank,
    }

    if as_tensors:
        # Lightning logs these from the model's device.
        return {k: v.to(device=emb.device, dtype=torch.float32) for k, v in metrics.items()}

    return {k: float(v) for k, v in metrics.items()}

# -------------------------------------------------------------------------------------------------
# Graph embeddings
# -------------------------------------------------------------------------------------------------

@dataclass
class GraphEmbeddingDataset:
    '''Container for a set of hyperbolic graph embeddings.'''

    embeddings: torch.Tensor
    codes: Sequence[str]
    levels: Sequence[int]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError('embeddings tensor must be 2D')

        num_nodes = self.embeddings.size(0)
        if num_nodes != len(self.codes) or num_nodes != len(self.levels):
            raise ValueError(
                'Embeddings, codes, and levels must have the same first dimension '
                f'(got embeddings={num_nodes}, codes={len(self.codes)}, levels={len(self.levels)})'
            )

        self.codes = list(self.codes)
        self.levels = [int(level) for level in self.levels]

    @classmethod
    def from_dataframe(
        cls,
        frame: pl.DataFrame,
        *,
        embedding_prefix: str = STAGE4_EMBEDDING_PREFIX,
        code_column: str = 'code',
        level_column: str = 'level',
    ) -> 'GraphEmbeddingDataset':
        '''Build a dataset from a parquet dataframe.'''

        if code_column not in frame.columns:
            raise ValueError(f'Expected column "{code_column}" in embeddings parquet')
        if level_column not in frame.columns:
            raise ValueError(f'Expected column "{level_column}" in embeddings parquet')

        embed_cols = sorted_embedding_columns(frame.columns, embedding_prefix)
        if not embed_cols:
            raise ValueError(
                f'No embedding columns found with prefix "{embedding_prefix}". '
                f'Set embedding_prefix to match {STAGE4_EMBEDDING_PREFIX}* or '
                f'{STAGE3_EMBEDDING_PREFIX}* columns.'
            )

        # to_numpy() returns a read-only view when the columns happen to sit back-to-back in
        # memory. torch.tensor copies either way and keeps the Fortran layout for both, whereas
        # to_numpy(writable=True) would copy only views to C order.
        tensor = torch.tensor(frame.select(embed_cols).to_numpy(), dtype=torch.float32)
        codes = frame.get_column(code_column).to_list()
        levels = frame.get_column(level_column).to_list()

        return cls(embeddings=tensor, codes=codes, levels=levels)
