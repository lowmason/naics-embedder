'''Utilities for configurable false-negative handling strategies (Issue #45).'''

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from naics_embedder.utils.config import FalseNegativeConfig

def apply_false_negative_strategy(
    config: FalseNegativeConfig,
    anchor_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    false_negative_mask: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    '''
    Adjust false-negative handling based on the configured strategy.

    Args:
        config: FalseNegativeConfig describing the desired behavior.
        anchor_embeddings: Tensor of shape (batch_size, dim).
        negative_embeddings: Tensor of shape (batch_size, k_negatives, dim).
        false_negative_mask: Optional boolean mask (batch_size, k_negatives).

    Returns:
        Tuple of (updated_mask, auxiliary_loss). Auxiliary loss is a scalar tensor suitable for
        adding to the training objective or None if no extra loss is required.
    '''

    if false_negative_mask is None or not false_negative_mask.any():
        return false_negative_mask, None

    if config.strategy == 'eliminate':
        return false_negative_mask, None

    # Collect anchor/negative pairs that were marked as false negatives
    anchor_pairs = anchor_embeddings.unsqueeze(1).expand_as(negative_embeddings)[false_negative_mask]
    negative_pairs = negative_embeddings[false_negative_mask]

    if anchor_pairs.numel() == 0:
        return false_negative_mask, None

    if config.attraction_metric == 'cosine':
        labels = torch.ones(anchor_pairs.shape[0], device=anchor_embeddings.device)
        attraction = F.cosine_embedding_loss(anchor_pairs, negative_pairs, labels, reduction='mean')
    else:
        attraction = F.mse_loss(anchor_pairs, negative_pairs, reduction='mean')

    auxiliary_loss = config.attraction_weight * attraction

    updated_mask: Optional[torch.Tensor]
    if config.strategy == 'attract':
        updated_mask = None  # keep negatives but add attraction loss
    else:
        updated_mask = false_negative_mask  # hybrid: mask + attraction

    return updated_mask, auxiliary_loss
