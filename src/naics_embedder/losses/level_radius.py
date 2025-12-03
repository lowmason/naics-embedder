from __future__ import annotations

import torch
import torch.nn.functional as F


def level_radius_loss(
    embeddings: torch.Tensor,
    levels: torch.Tensor,
    *,
    curvature: float = 1.0,
    base_level: float = 2.0,
    level_scale: float = 0.5,
    reduction: str = 'mean',
) -> torch.Tensor:
    '''
    Encourage embeddings to occupy progressively larger hyperbolic radii as the
    NAICS hierarchy level increases.
    '''
    if embeddings.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    if reduction not in {'mean', 'sum'}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    curvature = max(curvature, 1e-6)
    radial_sq = torch.clamp(embeddings[:, 0].pow(2) - 1.0 / curvature, min=1e-8)
    radial = torch.sqrt(radial_sq)
    targets = (levels.float() - base_level) * level_scale
    loss = F.smooth_l1_loss(radial, targets, reduction='none')

    if reduction == 'sum':
        return loss.sum()
    return loss.mean()

