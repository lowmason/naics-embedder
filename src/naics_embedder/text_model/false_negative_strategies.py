'''Utilities for configurable false-negative handling strategies (Issue #45).'''

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from naics_embedder.text_model.loss import effective_false_negative_mask
from naics_embedder.utils.config import FalseNegativeConfig


def apply_false_negative_strategy(
    config: FalseNegativeConfig,
    anchor_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    false_negative_mask: Optional[torch.Tensor],
    *,
    explicit_exclusion_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    '''
    Adjust false-negative handling based on the configured strategy.

    The pseudo-related ``false_negative_mask`` is first reduced to its effective form
    (``pseudo_related AND NOT explicit_exclusion AND valid``), so explicit exclusions are never
    eliminated from the contrastive denominator or attracted toward the anchor, and invalid padding
    never participates.

    Args:
        config: FalseNegativeConfig describing the desired behavior.
        anchor_embeddings: Tensor of shape (batch_size, dim).
        negative_embeddings: Tensor of shape (batch_size, k_negatives, dim).
        false_negative_mask: Optional pseudo-related boolean mask (batch_size, k_negatives).
        explicit_exclusion_mask: Explicit-exclusion flags (batch_size, k_negatives).
        valid_mask: Candidate validity (batch_size, k_negatives).

    Returns:
        Tuple of (updated_mask, auxiliary_loss). Auxiliary loss is a scalar tensor suitable for
        adding to the training objective or None if no extra loss is required.

    Raises:
        ValueError: If a mask does not align with ``negative_embeddings``.
    '''

    candidate_shape = negative_embeddings.shape[:2]
    masks = (false_negative_mask, explicit_exclusion_mask, valid_mask)
    if any(mask is not None and mask.shape != candidate_shape for mask in masks):
        raise ValueError('false-negative masks must align with negative embeddings')

    effective = effective_false_negative_mask(
        false_negative_mask,
        explicit_exclusion_mask,
        valid_mask,
    )
    if effective is None or not effective.any():
        return effective, None

    if config.strategy == 'eliminate':
        return effective, None

    # Collect anchor/negative pairs that are effective false negatives
    anchor_pairs = anchor_embeddings.unsqueeze(1).expand_as(negative_embeddings)[effective]
    negative_pairs = negative_embeddings[effective]

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
        updated_mask = effective  # hybrid: mask + attraction

    return updated_mask, auxiliary_loss
