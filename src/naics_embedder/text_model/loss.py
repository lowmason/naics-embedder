# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from naics_embedder.supervision.candidates import SelectedNegativeBatch
from naics_embedder.text_model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# False-negative eligibility
# -------------------------------------------------------------------------------------------------

def effective_false_negative_mask(
    pseudo_related: Optional[torch.Tensor],
    is_explicit_exclusion: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    '''
    Pseudo-related candidates that may be masked or attracted as false negatives.

    ``effective = pseudo_related AND NOT is_explicit_exclusion AND valid``: an explicit exclusion
    is always a repulsive negative, and invalid padding is never a candidate.

    Raises:
        ValueError: If the masks are not aligned on ``[batch, candidate]``.
    '''
    if is_explicit_exclusion.shape != valid_mask.shape:
        raise ValueError('explicit-exclusion and validity masks must align')
    if pseudo_related is None:
        return None
    if pseudo_related.shape != valid_mask.shape:
        raise ValueError('pseudo-related mask must align with selected candidates')
    return pseudo_related & ~is_explicit_exclusion & valid_mask

# -------------------------------------------------------------------------------------------------
# Hyperbolic InfoNCE Loss
# -------------------------------------------------------------------------------------------------

class HyperbolicInfoNCELoss(nn.Module):
    '''
    Hyperbolic InfoNCE loss operating directly on Lorentz-model embeddings.

    The encoder now returns hyperbolic embeddings directly, so this loss function
    works with them without additional projection.
    '''

    def __init__(self, embedding_dim: int, temperature: float = 0.07, curvature: float = 1.0):
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
        *,
        valid_mask: torch.Tensor,
        is_explicit_exclusion: torch.Tensor,
        pseudo_related_mask: Optional[torch.Tensor] = None,
        adaptive_margins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Compute Hyperbolic InfoNCE loss over selected negatives.

        Only eligible pseudo-related candidates leave the contrastive denominator, never a flagged
        explicit exclusion, though training and validation now admit none (Req 8). Invalid padding
        never contributes to the loss or its gradients. Anchors without any eligible negative are
        skipped.

        Args:
            anchor_emb: Anchor hyperbolic embeddings (batch_size, embedding_dim+1)
            positive_emb: Positive hyperbolic embeddings (batch_size, embedding_dim+1)
            negative_embs: Selected negative embeddings (batch_size, selected, embedding_dim+1)
            valid_mask: Negative validity (batch_size, selected)
            is_explicit_exclusion: Explicit-exclusion flags (batch_size, selected)
            pseudo_related_mask: Optional model-derived relatedness (batch_size, selected)
            adaptive_margins: Optional per-anchor adaptive margins (batch_size,)

        Returns:
            Loss scalar
        '''
        if negative_embs.ndim != 3:
            raise ValueError('negative embeddings must have shape [batch, selected, dim]')
        if valid_mask.shape != negative_embs.shape[:2]:
            raise ValueError('valid mask must align with negative embeddings')

        pos_distances = self.lorentz_distance(anchor_emb, positive_emb)
        neg_distances = self.lorentz_distance.batched_forward(anchor_emb, negative_embs)

        if adaptive_margins is not None:
            # Subtract per-anchor margin from negative distances (triplet-style)
            neg_distances = torch.clamp(neg_distances - adaptive_margins.unsqueeze(1), min=0.0)

        pos_similarities = -pos_distances / self.temperature
        neg_similarities = -neg_distances / self.temperature

        effective_false_negative = effective_false_negative_mask(
            pseudo_related_mask,
            is_explicit_exclusion,
            valid_mask,
        )
        eligible = valid_mask
        if effective_false_negative is not None:
            eligible = eligible & ~effective_false_negative
        neg_similarities = neg_similarities.masked_fill(~eligible, -torch.inf)
        has_negative = eligible.any(dim=1)
        if not has_negative.any():
            return (anchor_emb.sum() + positive_emb.sum() + negative_embs.sum()) * 0.0

        # Decoupled Contrastive Learning (DCL) loss: -pos_sim + logsumexp(neg_sims). Rows without
        # an eligible negative are excluded *before* logsumexp so an all -inf row cannot produce a
        # NaN gradient.
        per_anchor = -pos_similarities[has_negative] + torch.logsumexp(
            neg_similarities[has_negative], dim=1
        )
        return per_anchor.mean()

# -------------------------------------------------------------------------------------------------
# Hierarchy Preservation Loss
# -------------------------------------------------------------------------------------------------

class HierarchyPreservationLoss(nn.Module):
    '''
    Loss component that encourages embedding distances to match tree distances.
    This directly optimizes hierarchy preservation by penalizing deviations from
    ground truth tree structure.
    '''

    def __init__(
        self,
        tree_distances: torch.Tensor,
        code_to_idx: Dict[str, int],
        weight: float = 0.1,
        min_distance: float = 0.1,
    ):
        super().__init__()
        # Register as buffer so it moves with model to correct device
        self.register_buffer('tree_distances', tree_distances)
        self.code_to_idx = code_to_idx
        self.weight = weight
        self.min_distance = min_distance

    def forward(
        self,
        embeddings: torch.Tensor,
        codes: List[str],
        lorentz_distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        '''
        Compute hierarchy preservation loss.

        Args:
            embeddings: Hyperbolic embeddings (N, D+1)
            codes: List of NAICS codes corresponding to embeddings
            lorentz_distance_fn: Function to compute Lorentz distances

        Returns:
            Loss scalar
        '''
        # Get indices for codes that exist in ground truth
        valid_indices = []
        valid_codes = []
        for i, code in enumerate(codes):
            if code in self.code_to_idx:
                valid_indices.append(i)
                valid_codes.append(code)

        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Get embeddings for valid codes
        valid_embeddings = embeddings[valid_indices]

        # Get ground truth distance matrix indices
        gt_indices = torch.tensor(
            [self.code_to_idx[code] for code in valid_codes], device=embeddings.device
        )

        # Get ground truth distances for these codes
        gt_dists = self.tree_distances[gt_indices][:, gt_indices]  # type: ignore[index]

        # Compute embedding distances
        N = valid_embeddings.shape[0]
        emb_dists = torch.zeros((N, N), device=embeddings.device)

        for i in range(N):
            for j in range(i + 1, N):
                dist = lorentz_distance_fn(valid_embeddings[i:i + 1], valid_embeddings[j:j + 1])
                emb_dists[i, j] = dist
                emb_dists[j, i] = dist

        # Get upper triangular values (excluding diagonal)
        triu_indices = torch.triu_indices(N, N, offset=1, device=embeddings.device)
        emb_dists_flat = emb_dists[triu_indices[0], triu_indices[1]]
        gt_dists_flat = gt_dists[triu_indices[0], triu_indices[1]]

        # Filter out pairs with very small tree distances
        valid_mask = gt_dists_flat >= self.min_distance
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb_dists_filtered = emb_dists_flat[valid_mask]
        gt_dists_filtered = gt_dists_flat[valid_mask]

        # Normalize distances to similar scales for stable training
        emb_mean = emb_dists_filtered.mean()
        gt_mean = gt_dists_filtered.mean()

        emb_dists_norm = emb_dists_filtered / (emb_mean + 1e-8)
        gt_dists_norm = gt_dists_filtered / (gt_mean + 1e-8)

        # MSE loss between normalized distances
        mse_loss = torch.mean((emb_dists_norm - gt_dists_norm)**2)

        return self.weight * mse_loss

# -------------------------------------------------------------------------------------------------
# Structural Preference Loss (replaces LambdaRank)
# -------------------------------------------------------------------------------------------------

def structural_preference_from_distances(
    *,
    learned_distances: torch.Tensor,
    structural_distances: torch.Tensor,
    candidate_code_ids: torch.Tensor,
    anchor_code_ids: torch.Tensor,
    is_explicit_exclusion: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
    temperature: float,
    tie_tolerance: float,
    pair_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Pairwise structural-preference loss with a correction-direction gradient.

    For every unordered pair of eligible candidates with unequal structural distance, orient the
    pair so candidate ``i`` is structurally closer than ``j`` and penalize
    ``softplus((d_i - d_j + margin) / temperature)`` on learned distances. The gradient is
    positive for ``d_i`` and negative for ``d_j``, so descent pulls the structurally closer
    candidate in and pushes the farther one out.

    Explicit exclusions, invalid padding, self-candidates, and repeated code identities never
    participate; structural ties within ``tie_tolerance`` are ignored; optional importance weights
    are detached. Comparisons are normalized per anchor, then averaged over anchors with at least
    one comparison; with none, a finite differentiable zero is returned.

    Args:
        learned_distances: ``[batch, candidate]`` learned anchor-candidate distances
        structural_distances: ``[batch, candidate]`` raw structural distances
        candidate_code_ids: ``[batch, candidate]`` canonical code IDs
        anchor_code_ids: ``[batch]`` anchor code IDs
        is_explicit_exclusion: ``[batch, candidate]`` explicit-exclusion flags
        valid_mask: ``[batch, candidate]`` validity
        margin: Nonnegative ordering margin
        temperature: Positive softplus temperature
        tie_tolerance: Nonnegative structural tie tolerance
        pair_weights: Optional ``[batch, pairs]`` importance weights (upper-triangle order)
    '''
    if temperature <= 0:
        raise ValueError('structural preference temperature must be positive')
    if margin < 0 or tie_tolerance < 0:
        raise ValueError('structural preference margin and tie tolerance must be nonnegative')
    shape = learned_distances.shape
    aligned = (
        structural_distances,
        candidate_code_ids,
        is_explicit_exclusion,
        valid_mask,
    )
    if learned_distances.ndim != 2 or any(value.shape != shape for value in aligned):
        raise ValueError(
            'structural preference candidate tensors must share [batch, candidate] shape'
        )
    if anchor_code_ids.shape != (shape[0], ):
        raise ValueError('structural preference requires one anchor code ID per row')

    count = shape[1]
    left, right = torch.triu_indices(count, count, offset=1, device=learned_distances.device)
    learned_left = learned_distances[:, left]
    learned_right = learned_distances[:, right]
    structural_delta = structural_distances[:, left] - structural_distances[:, right]

    closer_left = structural_delta < -tie_tolerance
    closer_right = structural_delta > tie_tolerance
    learned_close = torch.where(closer_left, learned_left, learned_right)
    learned_far = torch.where(closer_left, learned_right, learned_left)

    duplicate_matrix = candidate_code_ids.unsqueeze(2).eq(candidate_code_ids.unsqueeze(1))
    seen_before = torch.tril(duplicate_matrix, diagonal=-1).any(dim=2)
    non_self = candidate_code_ids.ne(anchor_code_ids.unsqueeze(1))
    candidate_eligible = valid_mask & ~is_explicit_exclusion & ~seen_before & non_self
    pair_mask = (
        candidate_eligible[:, left]
        & candidate_eligible[:, right]
        & (closer_left | closer_right)
        & candidate_code_ids[:, left].ne(candidate_code_ids[:, right])
    )

    penalties = F.softplus((learned_close - learned_far + margin) / temperature)
    if pair_weights is None:
        weights = torch.ones_like(penalties)
    else:
        if pair_weights.shape != penalties.shape:
            raise ValueError('pair importance weights must align with unordered comparisons')
        weights = pair_weights.detach().to(penalties)
    weights = weights * pair_mask
    denominator = weights.sum(dim=1)
    contributing = denominator.gt(0)
    if not contributing.any():
        return learned_distances.sum() * 0.0
    per_anchor = (penalties * weights).sum(dim=1) / torch.where(
        contributing, denominator, torch.ones_like(denominator)
    )
    return per_anchor[contributing].mean()

class StructuralPreferenceLoss(nn.Module):
    '''
    Structural preference over each anchor's positive plus its selected negatives.

    Explicit exclusions are carried as candidates but never compared; see
    :func:`structural_preference_from_distances`.
    '''

    def __init__(
        self,
        *,
        curvature: float,
        margin: float,
        temperature: float,
        tie_tolerance: float,
        weight: float,
    ):
        super().__init__()
        self.lorentz_distance = LorentzDistance(curvature)
        self.margin = margin
        self.temperature = temperature
        self.tie_tolerance = tie_tolerance
        self.weight = weight

    def forward(
        self,
        *,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        anchor_code_id: torch.Tensor,
        positive_code_id: torch.Tensor,
        positive_structural_distance: torch.Tensor,
        selected: SelectedNegativeBatch,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positive_learned = self.lorentz_distance(anchor_emb, positive_emb).unsqueeze(1)
        negative_learned = self.lorentz_distance.batched_forward(anchor_emb, selected.embedding)
        learned = torch.cat([positive_learned, negative_learned], dim=1)
        structural = torch.cat(
            [
                positive_structural_distance.to(selected.structural_distance.dtype).unsqueeze(1),
                selected.structural_distance,
            ],
            dim=1,
        )
        code_ids = torch.cat([positive_code_id.unsqueeze(1), selected.code_id], dim=1)
        explicit = torch.cat(
            [
                torch.zeros_like(positive_code_id.unsqueeze(1), dtype=torch.bool),
                selected.is_explicit_exclusion,
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                torch.ones_like(positive_code_id.unsqueeze(1), dtype=torch.bool),
                selected.valid_mask,
            ],
            dim=1,
        )
        return self.weight * structural_preference_from_distances(
            learned_distances=learned,
            structural_distances=structural,
            candidate_code_ids=code_ids,
            anchor_code_ids=anchor_code_id,
            is_explicit_exclusion=explicit,
            valid_mask=valid,
            margin=self.margin,
            temperature=self.temperature,
            tie_tolerance=self.tie_tolerance,
            pair_weights=pair_weights,
        )

# -------------------------------------------------------------------------------------------------
# Req 11's three terms and the learned logit scales (spec 4.1)
# -------------------------------------------------------------------------------------------------

def _refuse_malformed_logits(
    term: str, rows: str, distances: torch.Tensor, scale: torch.Tensor
) -> None:
    '''Refuse distances that are not a non-empty (rows, codes) matrix, or a scale not 0-d.'''

    if distances.ndim != 2 or distances.numel() == 0:
        raise ValueError(
            f'{term} needs a non-empty ({rows}, codes) distance matrix, '
            f'not {tuple(distances.shape)}'
        )
    if not isinstance(scale, torch.Tensor) or scale.ndim != 0:
        raise ValueError(f'{term} needs a 0-d logit scale, not {scale!r}')

def _refuse_unaligned_mask(
    term: str, name: str, mask: torch.Tensor, distances: torch.Tensor
) -> None:
    '''Refuse a mask that is not bool or not shaped like the distances.'''

    if mask.dtype != torch.bool or mask.shape != distances.shape:
        raise ValueError(
            f'{term}: {name} must be a bool mask shaped like the distances '
            f'{tuple(distances.shape)}, not {mask.dtype} {tuple(mask.shape)}'
        )

def _refuse_unless_positive(term: str, name: str, value: float) -> None:
    '''Refuse a setting that is not a positive finite number.'''

    if not (math.isfinite(value) and value > 0):
        raise ValueError(f'{term}: {name} must be a positive finite number, not {value!r}')

def _refuse_rows(failing: torch.Tensor, message: str) -> None:
    '''Refuse the step at the first row where ``failing`` holds, naming the row in ``message``.'''

    if failing.any():
        raise ValueError(message.format(int(failing.nonzero()[0])))

def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Each row's log-sum-exp over the entries its mask holds; the others contribute nothing.'''

    return torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)

def task_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    candidates: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    '''
    The task term (spec 4.1(i)): the mean over the step's queries of
    −log Σ_{t ∈ T} softmax_C(−s_q · d(q, c))_t.

    The probability is summed over the query's targets T, so a query with two targets pays nothing
    for how it splits the probability between them. The softmax runs over exactly the candidates
    C: a code outside them carries no probability and takes no gradient. Spec 4.1 builds C from
    the codes at the query's level and the query's forced negatives N, so a cross-reference query
    always scores its referencing code (Req 8(b)).

    Args:
        distances: d(q, c) from every query to every code, (Q, N), finite everywhere, outside C too.
        scale: s_q, the task term's logit scale, a 0-d tensor.
        candidates: C, bool (Q, N).
        targets: T, bool (Q, N): for each query, a non-empty subset of its candidates.

    Returns:
        The term, a 0-d tensor in the distances' dtype.

    Raises:
        ValueError: If the step has no query, the shapes or masks are malformed, a query has no
            target, or a target is not one of its query's candidates.
    '''

    _refuse_malformed_logits('task_loss', 'queries', distances, scale)
    _refuse_unaligned_mask('task_loss', 'candidates', candidates, distances)
    _refuse_unaligned_mask('task_loss', 'targets', targets, distances)
    _refuse_rows(~targets.any(dim=1), 'task_loss: query {} has no target')
    _refuse_rows(
        (targets & ~candidates).any(dim=1),
        'task_loss: query {} has a target outside its candidates',
    )
    logits = -scale * distances
    return (_masked_logsumexp(logits, candidates) - _masked_logsumexp(logits, targets)).mean()

def code_code_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    structural: torch.Tensor,
    keep: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    '''
    The code–code listwise term (spec 4.1(ii)): the mean over the step's anchors of
    CE(p_a, softmax over J_a of −s_c · d(a, j)), with the target p_a = softmax over J_a of
    −D*_{aj} / τ_t.

    J_a is every code but the anchor and, for a unary pair, its partner, so a unary pair is neither
    a positive nor a negative (Req 9). A code outside J_a carries no probability on either side:
    its D* never reaches the target, its d never reaches the model's softmax, and it takes no
    gradient. The term reads no exclusion data, so no exclusion pair can act as a code–code
    negative (Req 8(c)).

    Args:
        distances: d(a, j) from every anchor to every code, (A, N), finite everywhere, off J_a too.
        scale: s_c, the code–code term's logit scale, a 0-d tensor.
        structural: D*_{aj}, the tree metric from every anchor to every code, (A, N).
        keep: J_a, bool (A, N).
        target_temperature: τ_t, the temperature of the target's softmax.

    Returns:
        The term, a 0-d tensor in the distances' dtype.

    Raises:
        ValueError: If the step has no anchor, the shapes or the mask are malformed, an anchor
            keeps no code, or ``target_temperature`` is not a positive finite number.
    '''

    _refuse_malformed_logits('code_code_loss', 'anchors', distances, scale)
    if structural.shape != distances.shape:
        raise ValueError(
            'code_code_loss: structural must be shaped like the distances '
            f'{tuple(distances.shape)}, not {tuple(structural.shape)}'
        )
    _refuse_unaligned_mask('code_code_loss', 'keep', keep, distances)
    _refuse_unless_positive('code_code_loss', 'target_temperature', target_temperature)
    _refuse_rows(~keep.any(dim=1), 'code_code_loss: anchor {} keeps no code')
    target_logits = -structural.to(distances.dtype) / target_temperature
    target = torch.softmax(target_logits.masked_fill(~keep, -torch.inf), dim=1)
    log_model = torch.log_softmax((-scale * distances).masked_fill(~keep, -torch.inf), dim=1)
    # Off J_a the target is 0 and the log-probability is -inf: zero the latter first, so neither
    # the value nor its gradient meets 0 * -inf
    return -(target * log_model.masked_fill(~keep, 0.0)).sum(dim=1).mean()

def radial_loss(radius: torch.Tensor, levels: torch.Tensor, radial_step: float) -> torch.Tensor:
    '''
    The radial term (spec 4.1(iii)): the mean over the step's anchors of (r_a − ρ · (λ(a) − 1))².

    The virtual root sits at the origin, the sectors (level 2) at r = ρ and the six-digit codes at
    r = 5ρ. r is the geodesic distance from the origin, the text stage's one radial coordinate
    (spec 4.2).

    Args:
        radius: r_a, the anchors' radii, (A,).
        levels: λ(a), the anchors' code levels, (A,).
        radial_step: ρ, the radius from one level to the next.

    Returns:
        The term, a 0-d tensor in the radii's dtype.

    Raises:
        ValueError: If the radii and levels are not non-empty and aligned (A,), or ``radial_step``
            is not a positive finite number.
    '''

    if radius.ndim != 1 or radius.numel() == 0 or levels.shape != radius.shape:
        raise ValueError(
            'radial_loss needs non-empty radii (A,) and levels (A,), not '
            f'{tuple(radius.shape)} and {tuple(levels.shape)}'
        )
    _refuse_unless_positive('radial_loss', 'radial_step', radial_step)
    target = radial_step * (levels.to(radius.dtype) - 1)
    return (radius - target).square().mean()

class LogitScale(nn.Module):
    '''
    A learned logit scale (spec 4.1): s = exp(θ), starting at ``init`` and clamped to
    [``low``, ``high``].

    Its one parameter is θ, ``log_scale``, so s is always positive. The clamp acts on θ in the
    forward pass: s never leaves the range, and θ takes no gradient while it lies outside it.

    Args:
        init: s at the start, inside the range.
        low: The least scale, positive.
        high: The greatest scale, above ``low``.

    Raises:
        ValueError: If a value is not finite, the range does not satisfy 0 < low < high, or
            ``init`` lies outside it.
    '''

    def __init__(self, init: float, low: float, high: float):
        super().__init__()
        if not all(math.isfinite(value) for value in (init, low, high)):
            raise ValueError(
                f'LogitScale takes finite numbers, not init={init!r}, low={low!r}, high={high!r}'
            )
        if not 0 < low < high:
            raise ValueError(
                f'a logit-scale range must satisfy 0 < low < high, not [{low!r}, {high!r}]'
            )
        if not low <= init <= high:
            raise ValueError(
                f'a logit scale must start inside its range [{low!r}, {high!r}], not at {init!r}'
            )
        self.low = float(low)
        self.high = float(high)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init)))

    def forward(self) -> torch.Tensor:
        '''
        The scale: exp(θ), with θ clamped to [log ``low``, log ``high``].

        Returns:
            A 0-d tensor in the parameter's dtype.
        '''

        return self.log_scale.clamp(math.log(self.low), math.log(self.high)).exp()
