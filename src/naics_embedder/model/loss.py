# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from naics_embedder.model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Hyperbolic InfoNCE Loss
# -------------------------------------------------------------------------------------------------

class HyperbolicInfoNCELoss(nn.Module):
    '''
    Hyperbolic InfoNCE loss operating directly on Lorentz-model embeddings.
    
    The encoder now returns hyperbolic embeddings directly, so this loss function
    works with them without additional projection.
    '''
    
    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.07,
        curvature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.curvature = curvature
        
        # Use shared Lorentz distance computation
        self.lorentz_distance = LorentzDistance(curvature)
    
    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
        batch_size: int,
        k_negatives: int,
        false_negative_mask: Optional[torch.Tensor] = None 
    ) -> torch.Tensor:
        '''
        Compute Hyperbolic InfoNCE loss.
        
        Args:
            anchor_emb: Anchor hyperbolic embeddings (batch_size, embedding_dim+1)
            positive_emb: Positive hyperbolic embeddings (batch_size, embedding_dim+1)
            negative_embs: Negative hyperbolic embeddings (batch_size * k_negatives, embedding_dim+1)
            batch_size: Batch size
            k_negatives: Number of negatives per anchor
            false_negative_mask: Optional mask for false negatives (batch_size, k_negatives)
        
        Returns:
            Loss scalar
        '''
        # Embeddings are already in hyperbolic space (Lorentz model)
        anchor_hyp = anchor_emb
        positive_hyp = positive_emb
        negative_hyp = negative_embs
        
        pos_distances = self.lorentz_distance(anchor_hyp, positive_hyp)
        
        # Compute negative distances using batched operations
        # Reshape negative_hyp from (batch_size * k_negatives, embedding_dim+1)
        # to (batch_size, k_negatives, embedding_dim+1)
        negative_hyp_reshaped = negative_hyp.view(batch_size, k_negatives, -1)
        
        # Use batched forward to compute all anchor-negative distances at once
        # anchor_hyp: (batch_size, embedding_dim+1) -> (batch_size, 1, embedding_dim+1)
        #   via broadcasting
        # negative_hyp_reshaped: (batch_size, k_negatives, embedding_dim+1)
        # Result: (batch_size, k_negatives)
        neg_distances = self.lorentz_distance.batched_forward(
            anchor_hyp, 
            negative_hyp_reshaped
        )
        
        pos_similarities = -pos_distances / self.temperature
        neg_similarities = -neg_distances / self.temperature

        if false_negative_mask is not None:
            neg_similarities = neg_similarities.masked_fill(
                false_negative_mask,
                -torch.finfo(neg_similarities.dtype).max
            )
        
        logits = torch.cat([
            pos_similarities.unsqueeze(1),
            neg_similarities
        ], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss