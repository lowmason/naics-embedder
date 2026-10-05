'''
Req 11's three terms and their learned logit scales (spec 4.1).

The task term, the code-code term and the radial term; ``LogitScale`` holds each of the two
listwise terms' scale. Nothing else remains of the six-term objective (spec 4.5).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math

import torch
import torch.nn as nn

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
