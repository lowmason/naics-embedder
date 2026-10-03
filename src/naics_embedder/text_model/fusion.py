'''
Fusion of a code's channel vectors into one vector (Req 14; Req 9; spec 4.1).

Every option reads only the present channels: an absent channel's slot is masked whatever it
holds, and a row with no present channel fuses to a finite vector with a finite gradient.

- ``masked_mean``, the default: the sum over present channels, divided by max(1, number present).
- ``attention``: a learned vector scores each present channel by its dot product, and a softmax
  over the present channels weights them.
- ``moe``, an ablation only (R12): the masked mean, then the mixture of experts on that vector.
  Only this option emits gate probabilities and expert indices.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from naics_embedder.text_model.moe import MixtureOfExperts

FUSIONS = ('masked_mean', 'attention', 'moe')

# -------------------------------------------------------------------------------------------------
# Output and the masked mean
# -------------------------------------------------------------------------------------------------

@dataclass
class FusionOutput:
    '''One fused vector per row, with the experts' gates under ``moe`` only.'''

    vector: torch.Tensor
    gate_probs: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None

def masked_mean(vectors: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    '''
    The mean over present channels, (B, F, H) and (B, F) to (B, H).

    A row with no present channel is zeros, divided by one, never by zero.
    '''

    weights = present.to(vectors.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return (vectors * weights).sum(dim=1) / count

# -------------------------------------------------------------------------------------------------
# Fusion options
# -------------------------------------------------------------------------------------------------

class MaskedMeanFusion(nn.Module):
    '''The default fusion. It has no parameters, so no absent channel can contribute.'''

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        return FusionOutput(masked_mean(vectors, present))

class AttentionFusion(nn.Module):
    '''
    Attention pooling over the present channels.

    The learned vector starts at zeros, so the pooling starts as the masked mean. An absent
    channel scores the dtype's minimum and its weight is then zeroed, so a row with no present
    channel has zero weights, a zero vector and a finite gradient.

    Args:
        hidden_size: The width of the channel vectors.
    '''

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        scores = vectors @ self.query
        # The score's own dtype bounds the fill: under autocast it can be narrower than the input
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * present.to(scores.dtype)
        return FusionOutput((weights.unsqueeze(-1) * vectors).sum(dim=1))

class MoEFusion(nn.Module):
    '''
    The MoE ablation (R12): the masked mean, then the mixture of experts on the fused vector.

    The experts belong to the fusion step; the load-balancing term reads their gates (R11).

    Args:
        hidden_size: The width of the channel vectors, the experts' input and output.
        num_experts: The number of experts.
        top_k: The experts each row is routed to.
        hidden_dim: The experts' hidden width.
    '''

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.moe = MixtureOfExperts(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        output, gate_probs, top_k_indices = self.moe(masked_mean(vectors, present))
        return FusionOutput(output, gate_probs, top_k_indices)

def build_fusion(
    name: str,
    hidden_size: int,
    *,
    num_experts: int = 4,
    top_k: int = 2,
    moe_hidden_dim: int = 1024,
) -> nn.Module:
    '''
    The fusion module that ``model.fusion`` names.

    Args:
        name: One of ``FUSIONS``.
        hidden_size: The width of the channel vectors.
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each row is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.

    Raises:
        ValueError: If the name is outside ``FUSIONS``.
    '''

    if name == 'masked_mean':
        return MaskedMeanFusion()
    if name == 'attention':
        return AttentionFusion(hidden_size)
    if name == 'moe':
        return MoEFusion(
            hidden_size, num_experts=num_experts, top_k=top_k, hidden_dim=moe_hidden_dim
        )
    raise ValueError(f'unknown fusion {name!r}; expected one of {list(FUSIONS)}')
