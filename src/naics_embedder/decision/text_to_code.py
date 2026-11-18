'''
Text→code's decision on the ITA panel (spec ``text-to-code-ita-panel``, 4.5).

- **Reference.** The reference arms are ``lexical-stub`` and ``euclidean-d16``. The reference is
  the arm with the higher validation headline MRR as a point estimate, ``euclidean-d16``'s the mean
  over its seeds. A tie goes to the stub, the cheaper arm.
- **The test (U40).** Δ = candidate MRR − reference MRR on the headline. A candidate beats the
  reference when the lower bound of the 95 % interval on Δ exceeds +δ. The interval is
  ``percentile_interval(replicates, superiority_level(1))``: the headline is the one decision
  panel, so n = 1.
- **Intervals.** The paired two-stage bootstrap of ``decision/resampling.py`` on the panel
  ``ita_text``: description groups are the units and the headline pairs (those not copied) the
  items, with each arm's seeds nested within the shared unit draw. The stub is a one-seed arm. The
  comparison enforces no seed floor (P6): Stage 5's read applies it to candidates.
- **Margin.** δ = r × the chosen reference's validation headline MRR (U38), fixed from the margin
  record's own fresh reference reads under a new run id, before any candidate read. Stage 4's reads
  stay gap evidence and are not reused. ``check_ita_margin_first`` refuses a record when the log
  holds an ``ita_text`` validation read with role ``candidate``, or with no role, made before it was
  fixed: training-monitor reads count as candidate reads.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.decision.decide import read_time
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import superiority_level
from naics_embedder.decision.sweep import _log_records
from naics_embedder.panels.ita import ItaPanel
from naics_embedder.panels.ita_scoring import ItaResult
from naics_embedder.panels.ita_split import ITA_PANEL, ItaPart
from naics_embedder.panels.outcome import QueryCodeEncoder
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

LEXICAL_STUB = 'lexical-stub'
EUCLIDEAN_D16 = 'euclidean-d16'
REFERENCE_ARMS = (LEXICAL_STUB, EUCLIDEAN_D16)

# -------------------------------------------------------------------------------------------------
# Reference and verdict
# -------------------------------------------------------------------------------------------------

def headline_values(result: ItaResult) -> pl.DataFrame:
    '''
    The headline's values for the bootstrap: one row per pair that is not copied.

    Args:
        result: A part's ``ItaResult``.

    Returns:
        ``unit`` (the pair's ``description_key``), ``item`` (its ``pair_id``) and ``value`` (its
        ``reciprocal_rank``).
    '''

    return result.per_pair.filter(~pl.col('copied')).select(
        pl.col('description_key').alias('unit'),
        pl.col('pair_id').alias('item'),
        pl.col('reciprocal_rank').alias('value'),
    )

def arm_replicates(
    items: PanelItems,
    arm: str,
    seeds: Sequence[ItaResult],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> Tuple[float, np.ndarray]:
    '''
    One arm's headline MRR as a point estimate and on each paired resample.

    Unit draws are on the stream ``ita_text``, shared by every arm, and the arm's seed draws on
    ``(ita_text, arm)``.

    Args:
        items: The headline's items, shared by every arm compared.
        arm: The arm's name.
        seeds: The arm's validation results, one per seed.
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.

    Returns:
        The mean over seeds of each seed's MRR, and the MRR on each resample, (replicates,).

    Raises:
        ValueError: If the arm has no seed, or a seed's headline holds other items than ``items``
            or puts one in another description group.
    '''

    if not seeds:
        raise ValueError(f'arm {arm!r} has no seed')
    values = np.stack([items.values(headline_values(seed)) for seed in seeds])
    sums = items.sums(values)
    units = unit_draws(ITA_PANEL, len(items.units), replicates, bootstrap_seed)
    seed_counts = seed_draws(ITA_PANEL, arm, len(seeds), replicates, bootstrap_seed)
    return (
        point_statistic(sums, items.sizes),
        replicate_statistics(sums, items.sizes, units, seed_counts),
    )

def choose_reference(mrr: Mapping[str, float]) -> str:
    '''
    The reference arm: the higher validation headline MRR, the stub on a tie.

    Args:
        mrr: Each reference arm's headline MRR, keyed by exactly ``REFERENCE_ARMS``.

    Returns:
        ``euclidean-d16`` when its MRR is higher, and otherwise ``lexical-stub``.

    Raises:
        ValueError: If the keys are not exactly ``REFERENCE_ARMS``, or an MRR is not finite.
    '''

    if set(mrr) != set(REFERENCE_ARMS):
        raise ValueError(f'the reference arms are {list(REFERENCE_ARMS)}, not {sorted(mrr)}')
    for arm in REFERENCE_ARMS:
        if not math.isfinite(mrr[arm]):
            raise ValueError(f'arm {arm!r} has an MRR that is not finite: {mrr[arm]}')
    return EUCLIDEAN_D16 if mrr[EUCLIDEAN_D16] > mrr[LEXICAL_STUB] else LEXICAL_STUB

@dataclass(frozen=True)
class ItaVerdict:
    '''
    A candidate's verdict against the reference on the validation headline.

    Attributes:
        candidate: The candidate arm.
        reference: The reference arm.
        delta: Δ, the candidate's MRR minus the reference's, as point estimates.
        interval: The two-sided percentile interval on Δ at ``level``.
        level: The interval's level, ``superiority_level(1)``.
        margin: δ.
        beats: Whether the interval's lower bound exceeds δ: only then does the candidate go
            forward.
    '''

    candidate: str
    reference: str
    delta: float
    interval: Tuple[float, float]
    level: float
    margin: float
    beats: bool

def beat_by_margin(
    candidate: str,
    reference: str,
    delta: float,
    replicates: np.ndarray,
    margin: float,
) -> ItaVerdict:
    '''
    Whether the candidate beats the reference by δ (U40).

    Args:
        candidate: The candidate arm.
        reference: The reference arm.
        delta: The point estimate of Δ, candidate minus reference.
        replicates: Δ on each paired resample, oriented the same way.
        margin: δ.

    Returns:
        The verdict: it beats when the lower bound of the 95 % interval on Δ exceeds δ.
    '''

    level = superiority_level(1)
    interval = percentile_interval(replicates, level)
    return ItaVerdict(
        candidate=candidate,
        reference=reference,
        delta=float(delta),
        interval=interval,
        level=level,
        margin=float(margin),
        beats=interval[0] > margin,
    )

def compare_to_reference(
    candidate: str,
    candidate_seeds: Sequence[ItaResult],
    reference: str,
    reference_seeds: Sequence[ItaResult],
    margin: float,
    *,
    replicates: int,
    bootstrap_seed: int,
) -> ItaVerdict:
    '''
    A candidate's verdict against the reference on paired resamples of the validation headline.

    Args:
        candidate: The candidate arm.
        candidate_seeds: Its validation results, one per seed.
        reference: The reference arm.
        reference_seeds: The reference's validation results, one per seed (one for the stub).
        margin: δ.
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.

    Returns:
        The verdict on Δ = candidate − reference.

    Raises:
        ValueError: If an arm has no seed, or a seed of either arm is scored on other headline
            pairs than the reference's first seed, or groups them otherwise.
    '''

    if not reference_seeds:
        raise ValueError(f'arm {reference!r} has no seed')
    items = PanelItems.from_values(headline_values(reference_seeds[0]))
    settings = {'replicates': replicates, 'bootstrap_seed': bootstrap_seed}
    candidate_point, candidate_draws = arm_replicates(items, candidate, candidate_seeds, **settings)
    reference_point, reference_draws = arm_replicates(items, reference, reference_seeds, **settings)
    return beat_by_margin(
        candidate,
        reference,
        candidate_point - reference_point,
        candidate_draws - reference_draws,
        margin,
    )

# -------------------------------------------------------------------------------------------------
# Margin
# -------------------------------------------------------------------------------------------------

class ItaMarginRecord(BaseModel):
    '''
    Text→code's δ, fixed from the reference arms' own validation reads before any candidate read
    (spec 4.5; U38).

    Attributes:
        name: Names the margin in the verdicts that use it.
        r: δ as a multiple of the reference's MRR.
        run_id: The run the margin's reads are logged under.
        reference_mrr: Each reference arm's validation headline MRR, the mean over its seeds.
        reference: The chosen reference arm (``choose_reference``).
        margin: δ, ``r`` times the reference's MRR.
        split_fingerprint: The panel's split fingerprint.
        data_fingerprint: The validation part's data fingerprint.
        log_records: The margin's reads, as the selection log appended them.
        fixed_at: When δ was fixed, after its reads.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    kind: Literal['ita_margin'] = 'ita_margin'
    name: str
    r: float = Field(gt=0)
    run_id: str
    reference_mrr: Dict[str, float]
    reference: str
    margin: float
    split_fingerprint: str
    data_fingerprint: str
    log_records: List[Dict[str, Any]]
    fixed_at: datetime

def fix_ita_margin(
    panel: ItaPanel,
    reference_arms: Mapping[str, Sequence[Tuple[Optional[int], QueryCodeEncoder]]],
    r: float,
    name: str,
    *,
    purpose: str,
) -> ItaMarginRecord:
    '''
    Read each reference arm's seeds on the validation part, and fix δ as ``r`` times the chosen
    reference's headline MRR (spec 4.5; U38).

    Every read is logged under one fresh run id with role ``reference``. An arm's MRR is the mean
    over its seeds of each seed's headline MRR.

    Args:
        panel: The ITA panel.
        reference_arms: Each reference arm's seeds, keyed by exactly ``REFERENCE_ARMS``: the seed
            (None for a seedless arm such as the stub) and the arm's encoder for it.
        r: δ as a multiple of the reference's MRR (U38: 0.05).
        name: Names the margin in the verdicts that use it.
        purpose: Why the reads happen, which the selection log records.

    Returns:
        The margin record, fixed after its reads.

    Raises:
        ValueError: If ``r`` is not a positive finite number, or the arms are not exactly
            ``REFERENCE_ARMS`` or one has no seed, before anything is read.
    '''

    if not (math.isfinite(r) and r > 0):
        raise ValueError(f'the margin multiple r must be positive and finite, got {r}')
    if set(reference_arms) != set(REFERENCE_ARMS):
        raise ValueError(
            f'the reference arms are {list(REFERENCE_ARMS)}, not {sorted(reference_arms)}'
        )
    for arm in REFERENCE_ARMS:
        if not reference_arms[arm]:
            raise ValueError(f'arm {arm!r} has no seed')
    run_id = f'ita-margin/{uuid.uuid4().hex}'
    reference_mrr: Dict[str, float] = {}
    for arm in REFERENCE_ARMS:
        values = []
        for seed, encoder in reference_arms[arm]:
            detail: Dict[str, Any] = {'arm': arm}
            if seed is not None:
                detail['seed'] = seed
            detail.update(role='reference', run=run_id)
            result = panel.score(encoder, ItaPart.VALIDATION, purpose, detail=detail)
            values.append(result.headline['mrr'])
        reference_mrr[arm] = math.fsum(values) / len(values)
    reference = choose_reference(reference_mrr)
    return ItaMarginRecord(
        name=name,
        r=r,
        run_id=run_id,
        reference_mrr=reference_mrr,
        reference=reference,
        margin=r * reference_mrr[reference],
        split_fingerprint=panel.fingerprint,
        data_fingerprint=panel.data_fingerprint(ItaPart.VALIDATION),
        log_records=_log_records([panel.log], run_id),
        fixed_at=datetime.now(timezone.utc),
    )

def check_ita_margin_first(margin: ItaMarginRecord, log: SelectionLog) -> None:
    '''
    Require the margin's reads to precede it, and no candidate read of ITA validation to precede
    it (spec 4.5).

    A read counts as a candidate read when its role is ``candidate`` or it names no role:
    training-monitor reads count. Sealed-part reads and other panels' reads do not count.

    Args:
        margin: The margin record.
        log: The selection log the reads went to.

    Raises:
        ValueError: If ``fixed_at`` names no timezone; if one of the margin's reads is later than
            ``fixed_at``; or if the log holds an ``ita_text`` validation read whose role is not
            ``reference`` before ``fixed_at``, naming its purpose and time.
    '''

    name = f'the margin {margin.name!r}'
    if margin.fixed_at.utcoffset() is None:
        raise ValueError(f'{name} was fixed at {margin.fixed_at.isoformat()}, with no timezone')
    for record in margin.log_records:
        time = read_time(record, f'{name}: its read of {record.get("detail", {}).get("arm")}')
        if time > margin.fixed_at:
            raise ValueError(
                f'{name} was fixed at {margin.fixed_at.isoformat()}, before its read at '
                f'{time.isoformat()} (spec 4.5)'
            )
    for record in log.records():
        if (record.get('event'), record.get('panel'), record.get('split')) != (
            SelectionEvent.READ.value, ITA_PANEL, ItaPart.VALIDATION.value
        ):
            continue
        if record.get('detail', {}).get('role') == 'reference':
            continue
        time = read_time(record, f'the read {record.get("purpose")!r}')
        if time < margin.fixed_at:
            raise ValueError(
                f'{name} was fixed at {margin.fixed_at.isoformat()}, after the candidate read '
                f'{record.get("purpose")!r} at {time.isoformat()}: the margin comes first '
                '(spec 4.5)'
            )
