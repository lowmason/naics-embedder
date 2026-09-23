# -------------------------------------------------------------------------------------------------
# Phase 2: Lorentzian Hard Negative Mining (HNM)
# With torch.compile support for fused operations
# -------------------------------------------------------------------------------------------------

import logging
from typing import Tuple

import torch
import torch.nn as nn

from naics_embedder.supervision.candidates import CandidateProposal, NegativeCandidateBatch
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.text_model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)

def _ineligible(candidates: NegativeCandidateBatch) -> torch.Tensor:
    '''Candidates no strategy may propose: invalid padding and explicit exclusions.'''
    return ~candidates.valid_mask | candidates.is_explicit_exclusion

def _top_k_proposal(scores: torch.Tensor, k: int, reason: SelectionReason) -> CandidateProposal:
    k_actual = min(k, scores.shape[1])
    top_scores, top_indices = torch.topk(scores, k=k_actual, dim=1, largest=True)
    return CandidateProposal(source_indices=top_indices, scores=top_scores, reason=reason)

# Import compile utilities
try:
    from naics_embedder.utils.compile import maybe_compile

    _COMPILE_AVAILABLE = True
except ImportError:
    _COMPILE_AVAILABLE = False

    def maybe_compile(*args, **kwargs):  # type: ignore[misc]
        '''Fallback decorator when compile module not available.'''

        def decorator(fn):
            return fn

        return decorator

# -------------------------------------------------------------------------------------------------
# Compiled core operations for hard negative mining
# -------------------------------------------------------------------------------------------------

@maybe_compile(mode='reduce-overhead')
def _lorentz_norm_compiled(
    time_norm_sq: torch.Tensor, spatial_norm_sq: torch.Tensor
) -> torch.Tensor:
    '''Core Lorentz norm computation - highly fusible.'''
    lorentz_norm_sq = time_norm_sq - spatial_norm_sq
    return torch.sqrt(torch.clamp(lorentz_norm_sq, min=1e-8))

@maybe_compile(mode='reduce-overhead')
def _sech_margin_compiled(
    time_coord: torch.Tensor, sqrt_c: torch.Tensor, base_margin: float
) -> torch.Tensor:
    '''Core sech-based adaptive margin computation - highly fusible.'''
    arg = sqrt_c * time_coord
    arg_clamped = torch.clamp(arg, min=1.0 + 1e-6)
    lorentz_norm = torch.acosh(arg_clamped)
    cosh_norms = torch.cosh(lorentz_norm)
    sech_norms = 1.0 / (cosh_norms + 1e-8)
    return base_margin * sech_norms

@maybe_compile(mode='reduce-overhead')
def _kl_divergence_compiled(
    anchor_probs: torch.Tensor, negative_probs: torch.Tensor
) -> torch.Tensor:
    '''Core KL-divergence computation - highly fusible.'''
    eps = 1e-8
    log_ratio = torch.log(anchor_probs + eps) - torch.log(negative_probs + eps)
    return (anchor_probs * log_ratio).sum(dim=2)

@maybe_compile(mode='reduce-overhead')
def _cosine_similarity_compiled(
    anchor_normalized: torch.Tensor, negative_normalized: torch.Tensor
) -> torch.Tensor:
    '''Core cosine similarity computation - highly fusible.'''
    return (anchor_normalized * negative_normalized).sum(dim=2)

# -------------------------------------------------------------------------------------------------
# Lorentzian Hard Negative Mining
# -------------------------------------------------------------------------------------------------

class LorentzianHardNegativeMiner(nn.Module):
    '''
    Phase 2 Hard Negative Mining: Propose negatives that are geometrically close
    in the learned hyperbolic space using Lorentzian distance.

    For each anchor, proposes the top-k eligible candidates with the smallest Lorentzian distance
    as source indices; the selection coordinator performs the only gather.
    '''

    def __init__(self, curvature: float = 1.0, safety_epsilon: float = 1e-5):
        '''
        Initialize hard negative miner.

        Args:
            curvature: Hyperbolic curvature parameter c
            safety_epsilon: Small epsilon for safety checks to prevent NaN
        '''
        super().__init__()
        self.curvature = curvature
        self.safety_epsilon = safety_epsilon

        # Use shared Lorentz distance computation
        self.lorentz_distance = LorentzDistance(curvature)

    def propose(
        self,
        anchor_emb: torch.Tensor,
        candidates: NegativeCandidateBatch,
        k: int,
    ) -> CandidateProposal:
        '''
        Propose the ``k`` geometrically closest eligible candidates per anchor.

        Returns source indices into the canonical pool with detached ``-distance`` scores;
        invalid padding and explicit exclusions score ``-inf`` and are never eligible. No
        candidate field is gathered here.

        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim+1)
            candidates: The canonical candidate pool
            k: Maximum number of proposals per anchor
        '''
        with torch.no_grad():
            distances = self.lorentz_distance.batched_forward(anchor_emb, candidates.embedding)
            scores = (-distances).masked_fill(_ineligible(candidates), -torch.inf)
        return _top_k_proposal(scores, k, SelectionReason.GEOMETRIC)

    def compute_lorentz_norm(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentz norm: ||x||_L = sqrt(⟨x, x⟩_L)

        For a point on the hyperboloid: ⟨x, x⟩_L = -1/c
        The norm is the hyperbolic radius: sqrt(x0^2 - 1/c)

        Uses compiled operations when torch.compile is enabled.

        Args:
            x: Hyperbolic embeddings (batch_size, embedding_dim+1)

        Returns:
            Lorentz norms (batch_size,)
        '''
        time_coord = x[:, 0]  # x₀
        spatial_coords = x[:, 1:]  # x₁...xₙ
        spatial_norm_sq = torch.sum(spatial_coords**2, dim=1)
        time_norm_sq = time_coord**2
        return _lorentz_norm_compiled(time_norm_sq, spatial_norm_sq)

    def check_lorentz_inner_product_safety(self, u: torch.Tensor,
                                           v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Safety check: Ensure ⟨u, v⟩_L < -1 to prevent NaN in gradients.

        Args:
            u: First point on hyperboloid (batch_size, embedding_dim+1)
            v: Second point on hyperboloid (batch_size, k, embedding_dim+1) or
                (batch_size, embedding_dim+1)

        Returns:
            Tuple of (safe_dot_product, is_valid)
            - safe_dot_product: Clamped Lorentz inner product
            - is_valid: Boolean tensor indicating if all pairs are valid
        '''
        # Compute Lorentz inner product
        if v.dim() == 3:
            # Batched case: u (batch_size, D+1), v (batch_size, k, D+1)
            uv = u.unsqueeze(1) * v  # (batch_size, k, D+1)
            dot_product = torch.sum(uv[:, :, 1:], dim=2) - uv[:, :, 0]  # (batch_size, k)
        else:
            # Pairwise case: u (batch_size, D+1), v (batch_size, D+1)
            uv = u * v  # (batch_size, D+1)
            dot_product = torch.sum(uv[:, 1:], dim=1) - uv[:, 0]  # (batch_size,)

        # Safety check: ⟨u, v⟩_L must be < -1 for valid arccosh
        # Clamp to ensure: dot_product <= -1 - epsilon
        safe_dot_product = torch.clamp(dot_product, max=-1.0 - self.safety_epsilon)

        # Check validity: all pairs should satisfy constraint
        is_valid = torch.all(dot_product < -1.0 - self.safety_epsilon)

        return safe_dot_product, is_valid

# -------------------------------------------------------------------------------------------------
# Router-Guided Negative Mining
# -------------------------------------------------------------------------------------------------

class RouterGuidedNegativeMiner(nn.Module):
    '''
    Router-Guided Negative Mining: Propose negatives that confuse the Gating Network.

    Prevents "Expert Collapse" where a single expert handles all easy negatives by
    mining negatives where the router assigns high probability to the same experts as the anchor.

    Uses KL-Divergence or Cosine Similarity to measure confusion between gate distributions.
    '''

    def __init__(self, metric: str = 'kl_divergence', temperature: float = 1.0):
        '''
        Initialize router-guided negative miner.

        Args:
            metric: Confusion metric to use ('kl_divergence' or 'cosine_similarity')
            temperature: Temperature for KL-divergence computation (higher = more uniform)
        '''
        super().__init__()
        self.metric = metric
        self.temperature = temperature

        if metric not in ['kl_divergence', 'cosine_similarity']:
            raise ValueError(f"metric must be 'kl_divergence' or 'cosine_similarity', got {metric}")

    def propose(
        self,
        anchor_gate_probs: torch.Tensor,
        candidates: NegativeCandidateBatch,
        k: int,
    ) -> CandidateProposal:
        '''
        Propose the ``k`` eligible candidates whose gate distributions most confuse the router.

        Returns source indices into the canonical pool with detached confusion scores; invalid
        padding and explicit exclusions score ``-inf`` and are never eligible.

        Raises:
            ValueError: If the candidate pool carries no router gate probabilities.
        '''
        if candidates.router_gate_probs is None:
            raise ValueError('router-guided proposals require candidate gate probabilities')
        with torch.no_grad():
            scores = self.compute_confusion_scores(
                anchor_gate_probs, candidates.router_gate_probs
            ).masked_fill(_ineligible(candidates), -torch.inf)
        return _top_k_proposal(scores, k, SelectionReason.ROUTER)

    def compute_kl_divergence(
        self, anchor_gate_probs: torch.Tensor, negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute KL-divergence between anchor and negative gate distributions.

        KL(P_anchor || P_negative) = sum(P_anchor * log(P_anchor / P_negative))

        Lower KL-divergence means similar distributions
        (router assigns similar expert probabilities).
        We want negatives with similar gate distributions (low KL-divergence)
        to confuse the router,
        as they make the router think the negative is similar to the anchor.

        Uses compiled operations when torch.compile is enabled.

        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)

        Returns:
            KL-divergence scores (batch_size, k_negatives) - lower = more confusion
        '''
        eps = 1e-8

        # Normalize to ensure valid probability distributions
        anchor_probs = anchor_gate_probs + eps
        anchor_probs = anchor_probs / anchor_probs.sum(dim=1, keepdim=True)

        negative_probs = negative_gate_probs + eps
        negative_probs = negative_probs / negative_probs.sum(dim=2, keepdim=True)

        # Expand anchor_probs for broadcasting: (batch_size, 1, num_experts)
        anchor_probs_expanded = anchor_probs.unsqueeze(1)

        # Use compiled KL-divergence computation
        return _kl_divergence_compiled(anchor_probs_expanded, negative_probs)

    def compute_cosine_similarity(
        self, anchor_gate_probs: torch.Tensor, negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute cosine similarity between anchor and negative gate distributions.

        Higher cosine similarity means more confusion (similar distributions).

        Uses compiled operations when torch.compile is enabled.

        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)

        Returns:
            Cosine similarity scores (batch_size, k_negatives)
        '''
        # Normalize to unit vectors
        anchor_norm = torch.norm(anchor_gate_probs, dim=1, keepdim=True)
        anchor_normalized = anchor_gate_probs / (anchor_norm + 1e-8)

        negative_norm = torch.norm(negative_gate_probs, dim=2, keepdim=True)
        negative_normalized = negative_gate_probs / (negative_norm + 1e-8)

        # Expand anchor for broadcasting: (batch_size, 1, num_experts)
        anchor_expanded = anchor_normalized.unsqueeze(1)

        # Use compiled cosine similarity computation
        return _cosine_similarity_compiled(anchor_expanded, negative_normalized)

    def compute_confusion_scores(
        self, anchor_gate_probs: torch.Tensor, negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute confusion scores between anchor and negative gate distributions.

        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)

        Returns:
            Confusion scores (batch_size, k_negatives)
            - For KL-divergence: lower scores = more confusion (similar distributions)
            - For cosine similarity: higher scores = more confusion (similar distributions)
        '''
        if self.metric == 'kl_divergence':
            scores = self.compute_kl_divergence(anchor_gate_probs, negative_gate_probs)
            # Lower KL-divergence = more confusion, so we negate for consistency
            # (we want to select negatives with high confusion = low KL-divergence)
            return -scores
        elif self.metric == 'cosine_similarity':
            scores = self.compute_cosine_similarity(anchor_gate_probs, negative_gate_probs)
            # Higher cosine similarity = more confusion
            return scores
        else:
            raise ValueError(f'Unknown metric: {self.metric}')

# -------------------------------------------------------------------------------------------------
# Norm-Adaptive Margin
# -------------------------------------------------------------------------------------------------

class NormAdaptiveMargin(nn.Module):
    '''
    Norm-adaptive margin for triplet loss that decays as anchor's norm increases.

    Formula: m(a) = m_0 * sech(||a||_L)

    This ensures that anchors near the leaf boundary (large norm) have smaller margins,
    making the loss more adaptive to the hyperbolic geometry.
    '''

    def __init__(self, base_margin: float = 0.5, curvature: float = 1.0):
        '''
        Initialize norm-adaptive margin.

        Args:
            base_margin: Base margin m_0
            curvature: Hyperbolic curvature parameter c
        '''
        super().__init__()
        self.base_margin = base_margin
        self.curvature = curvature

        # Use Lorentz distance for computing norms
        self.lorentz_distance = LorentzDistance(curvature)

    def compute_lorentz_norm(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentz norm (hyperbolic distance from origin) for embeddings.

        The hyperbolic distance from the origin to a point x on the hyperboloid is:
        d = arccosh(sqrt(c) * x₀)

        Args:
            x: Hyperbolic embeddings (batch_size, embedding_dim+1)

        Returns:
            Hyperbolic distances from origin (batch_size,)
        '''
        time_coord = x[:, 0]  # x₀
        sqrt_c = torch.sqrt(torch.tensor(self.curvature, device=x.device, dtype=x.dtype))

        # Hyperbolic distance from origin: d = arccosh(sqrt(c) * x₀)
        # Clamp to avoid numerical issues (arccosh requires argument >= 1)
        arg = sqrt_c * time_coord
        arg_clamped = torch.clamp(arg, min=1.0 + 1e-6)
        lorentz_norm = torch.acosh(arg_clamped)

        return lorentz_norm

    def forward(self, anchor_emb: torch.Tensor) -> torch.Tensor:
        '''
        Compute norm-adaptive margin for each anchor.

        Uses compiled operations when torch.compile is enabled.

        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim+1)

        Returns:
            Adaptive margins (batch_size,)
        '''
        time_coord = anchor_emb[:, 0]
        sqrt_c = torch.sqrt(
            torch.tensor(self.curvature, device=anchor_emb.device, dtype=anchor_emb.dtype)
        )
        return _sech_margin_compiled(time_coord, sqrt_c, self.base_margin)
