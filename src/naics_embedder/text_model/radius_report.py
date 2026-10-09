'''
Radius and gradient checks for a selected arm (spec 4.2, Verification Radius, No inert terms).

The radius checks are the hyperbolic arm's, the one arm with the radial term and the live-radius
head (Req 12, 13). Every arm's terms and scales are checked for gradient.
'''

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import polars as pl
import torch

from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.text_model.hyperbolic import exp_map_origin, polar_distance

# polar_distance materializes (rows, codes, dimension), so bound the first axis (spec 4.2).
PAIR_CHUNK_ROWS = 32
MIN_LEVEL_SD = 1e-3
MAX_RELATIVE_ERROR = 1e-3
MANIFOLD_RELATIVE_TOLERANCE = 1e-9
ANCHOR_GRADIENT_PREFIX = 'anchor_radius/'

@dataclass(frozen=True)
class RadiusReport:
    '''
    The selected table's radius checks and the batch's signed anchor-radius gradients.

    ``level_sd`` is the population SD at each observed level. ``pairs`` counts every ordered
    pair, self pairs included. Exact coincident tangent rows have mathematical distance zero:
    their float32 distance must be exactly zero, and their float64 read cancellation residual
    is recorded separately, since relative error at zero is undefined. Every other pair must
    have a positive read distance and relative error at most 1e-3; there is no denominator floor.
    A table-only report leaves the gradient unmeasured and cannot pass all the checks.
    '''

    anchor_radius_gradient: Optional[Tuple[Optional[float], ...]]
    anchor_gradient_nonzero: Optional[bool]
    level_sd: Dict[int, float]
    sector_radii: Dict[str, float]
    sector_min_gap: Optional[float]
    max_radius: float
    max_radius_code: str
    manifold_error: float
    manifold_tolerance: float
    pairs: int
    zero_distance_pairs: int
    zero_training_max_error: float
    zero_read_max_error: float
    nonzero_pairs_read_as_zero: int
    max_relative_error: Optional[float]
    failures: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        '''Whether every required check was measured and passed.'''

        return not self.failures

@dataclass(frozen=True)
class _PairAgreement:
    pairs: int
    zero_distance_pairs: int
    zero_training_max_error: float
    zero_read_max_error: float
    nonzero_pairs_read_as_zero: int
    max_relative_error: Optional[float]

def _pair_agreement(tangent: torch.Tensor, points: torch.Tensor) -> _PairAgreement:
    '''Compare float32 training and float64 panel-read distances over every ordered pair.'''

    vectors = tangent.float()
    radius = torch.linalg.vector_norm(vectors, dim=1)
    moving = radius > 0
    divisor = torch.where(moving, radius, torch.ones_like(radius))
    direction = vectors / divisor[:, None]
    pairs = zeros = unreadable = 0
    training_zero_error = read_zero_error = 0.0
    relative_errors = []
    for start in range(0, len(tangent), PAIR_CHUNK_ROWS):
        stop = start + PAIR_CHUNK_ROWS
        training = polar_distance(radius[start:stop], direction[start:stop], radius, direction)
        reading = lorentz_distances(points[start:stop], points)
        if not (torch.isfinite(training).all() and torch.isfinite(reading).all()):
            raise ValueError('the all-pairs distances are not finite')
        # Exact equality of the exported tangent rows defines mathematical zero, including
        # off-diagonal duplicate points. Do not mistake a cancelled read distance for equality.
        coincident = (tangent[start:stop, None, :] == tangent[None, :, :]).all(dim=2)
        pairs += training.numel()
        zeros += int(coincident.sum())
        if coincident.any():
            training_zero_error = max(training_zero_error, float(training[coincident].abs().max()))
            read_zero_error = max(read_zero_error, float(reading[coincident].abs().max()))
        nonzero = ~coincident
        unreadable += int((nonzero & (reading == 0)).sum())
        comparable = nonzero & (reading > 0)
        if comparable.any():
            error = (training.double()[comparable] - reading[comparable]).abs() / reading[comparable]
            relative_errors.append(float(error.max()))
    return _PairAgreement(
        pairs, zeros, training_zero_error, read_zero_error, unreadable,
        max(relative_errors) if relative_errors else None
    )

def radius_report(
    table: pl.DataFrame, *, anchor_radius_gradient: Optional[np.ndarray] = None
) -> RadiusReport:
    '''
    Check Verification Radius's quantities on a table in Req 2's tangent-coordinate form.

    The manifold residual is measured at the largest observed radius, on the float64 points
    the arm's reads use. All distances are checked in bounded row chunks. ``anchor_radius_gradient``
    is the real batch's signed dL/dr_a, one per anchor; missing, empty, zero or nonfinite gradients
    fail the corresponding check. Nonfinite supplied entries are represented by None in the report.

    Raises:
        ValueError: If the coordinate table is invalid, empty, has incorrect level metadata,
            the gradient is not one-dimensional, or the resulting points or distances are not finite.
    '''

    codes, matrix = coordinate_matrix(table)
    if not codes:
        raise ValueError('the radius report needs a nonempty coordinate table')
    levels = np.array([len(code) for code in codes])
    if 'level' in table.columns and table['level'].to_list() != levels.tolist():
        raise ValueError('the table level metadata differs from its code lengths')
    tangent = torch.from_numpy(matrix)
    radius = torch.linalg.vector_norm(tangent, dim=1).numpy()
    points = exp_map_origin(tangent)
    if not torch.isfinite(points).all():
        raise ValueError('the table maps to nonfinite Lorentz points')
    failures = []
    gradient_values = None
    gradient_nonzero = None
    if anchor_radius_gradient is None:
        failures.append('the anchor radius gradient was not supplied')
    else:
        gradient = np.asarray(anchor_radius_gradient, dtype=np.float64)
        if gradient.ndim != 1:
            raise ValueError('anchor_radius_gradient must be one-dimensional, one per anchor')
        gradient_values = tuple(float(value) if np.isfinite(value) else None for value in gradient)
        gradient_nonzero = bool(
            gradient.size and np.isfinite(gradient).all() and (gradient != 0).all()
        )
        if not gradient_nonzero:
            failures.append('every anchor needs a finite nonzero radius gradient')
    level_sd = {int(level): float(radius[levels == level].std()) for level in np.unique(levels)}
    for level, sd in level_sd.items():
        if sd <= MIN_LEVEL_SD:
            failures.append(f'level {level} radius SD {sd} does not exceed {MIN_LEVEL_SD}')
    sector_radii = {
        code: float(value)
        for code, value, level in zip(codes, radius, levels) if level == 2
    }
    sector_min_gap = None
    if len(sector_radii) < 2:
        failures.append('at least two sectors are needed to measure their least radius gap')
    else:
        ordered = sorted(sector_radii.values())
        sector_min_gap = float(np.diff(ordered).min())
        if sector_min_gap <= 0:
            failures.append('sector radii are not pairwise distinct')
    if any(value <= 0 for value in sector_radii.values()):
        failures.append('every sector radius must be positive')
    largest = int(radius.argmax())
    point = points[largest]
    error = float((point[1:].square().sum() - point[0].square() + 1).abs())
    tolerance = MANIFOLD_RELATIVE_TOLERANCE * float(point[0].square())
    if error > tolerance:
        failures.append(f'the largest-radius manifold error {error} exceeds {tolerance}')
    agreement = _pair_agreement(tangent, points)
    if agreement.zero_training_max_error != 0:
        failures.append(
            'the float32 training distance is nonzero at a mathematically coincident pair'
        )
    if agreement.nonzero_pairs_read_as_zero:
        failures.append(
            f'{agreement.nonzero_pairs_read_as_zero} noncoincident pairs have zero float64 read distance'
        )
    if agreement.max_relative_error is not None and agreement.max_relative_error > MAX_RELATIVE_ERROR:
        failures.append(
            f'all-pairs relative error {agreement.max_relative_error} exceeds {MAX_RELATIVE_ERROR}'
        )
    return RadiusReport(
        gradient_values, gradient_nonzero, level_sd, sector_radii, sector_min_gap,
        float(radius[largest]), codes[largest], error, tolerance, agreement.pairs, agreement
        .zero_distance_pairs, agreement.zero_training_max_error, agreement.zero_read_max_error,
        agreement.nonzero_pairs_read_as_zero, agreement.max_relative_error, tuple(failures)
    )

@contextmanager
def _without_expert_logs(model: torch.nn.Module) -> Iterator[None]:
    '''Suppress diagnostic MoE utilization logs and restore the original instance attribute.'''

    name = '_log_expert_utilization'
    if model.fusion != 'moe':
        yield
        return
    original = vars(model).get(name)
    present = name in vars(model)
    setattr(model, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        if present:
            setattr(model, name, original)
        else:
            delattr(model, name)

def _gradient_norm(term: torch.Tensor, parameters: Tuple[torch.Tensor, ...]) -> float:
    '''The L2 norm of a term's gradient, without accumulating into parameter.grad.'''

    if not term.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    squared = sum(
        float(gradient.detach().cpu().double().square().sum()) for gradient in gradients
        if gradient is not None
    )
    return float(np.sqrt(squared))

def term_gradients(model: torch.nn.Module, batch: Mapping[str, Any]) -> Dict[str, float]:
    '''
    Measure No inert terms and, in the hyperbolic arm, signed dL/dr_a on one two-stream batch
    (spec 6, P26).

    Each term's norm is taken over the trainable encoder parameters, with its contribution's
    weight. The two scale gradients come from the total, so a zero-weight code-code term also
    leaves its scale inert. The radial term and ``anchor_radius/<row>``, each signed total-loss
    radius gradient, are the hyperbolic arm's alone (Req 12): a flat arm reports neither.
    Existing parameter gradients and training flags are preserved. MoE utilization logging is
    suppressed for this computation only, so no Trainer, log warning or histogram is produced.
    '''

    parameters = tuple(
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    with torch.enable_grad(), _without_expert_logs(model):
        losses = model.compute_losses(batch)
        terms = {
            'task': losses.task,
            'code_code': model.hparams.code_code_weight * losses.code_code,
        }
        if losses.radial is not None:
            terms['radial'] = model.hparams.radial_weight * losses.radial
        if losses.load_balancing is not None:
            terms['load_balancing'] = model.hparams.load_balancing_coef * losses.load_balancing
        result = {name: _gradient_norm(term, parameters) for name, term in terms.items()}
        for name, scale in (
            ('logit_scale_task', model.logit_scale_task), (
                'logit_scale_code', model.logit_scale_code
            )
        ):
            result[name] = _gradient_norm(losses.total, (scale.log_scale, ))
        if model.encoder.head.radial:
            gradient = torch.autograd.grad(losses.total, losses.anchor_radius, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(losses.anchor_radius)
            result.update(
                {
                    f'{ANCHOR_GRADIENT_PREFIX}{row}': float(value)
                    for row, value in enumerate(gradient.detach().cpu())
                }
            )
    return result
