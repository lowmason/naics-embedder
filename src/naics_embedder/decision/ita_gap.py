'''
The gap between Census phrases and owner text for text→code's reference arms (redesign roadmap
Stage 4, plan 20; D1, D2, P5, P6).

For each reference arm, three MRRs: C on the outcome panel's held-out Census index entries
(validation), K on the ITA validation part's copied pairs, and H on its headline pairs, those not
copied. The gaps are C − H, C − K and K − H, and a code-matched C − H that gives each code one
weight: over the codes M that report at least one headline pair and hold at least one outcome
validation query, each code's mean census value against the mean reciprocal rank of the headline
pairs that report it. A pair with several codes counts under each, as ``ItaResult.per_code`` counts
it. A code that one panel lacks is left out of M and counted, and nothing fails.

Intervals come from the paired two-stage bootstrap of ``decision/resampling.py`` (P5). Each
stratum draws its own units on its own stream: H on ``ita_text`` (so that it equals
``arm_replicates``), K on ``ita_text/copied``, C on ``outcome``, and the matched pair on
``ita_text/matched``. One seed draw per arm, on ``(ita_text, arm)``, is shared by every stratum.
Copied and headline keys are disjoint, so the strata do not overlap. A gap's interval is the
percentile interval of the difference of the two strata's replicates.

D2: the finding recommends a later ITA training part when, for both reference arms, the lower bound
of the 95 % interval on the code-matched headline gap exceeds r × that arm's C. Nothing here reads a
panel or logs a read: the inputs are Stage 4's read records and the Stage 8 record of
``euclidean-d16``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from naics_embedder.decision.ita_reads import ItaReferenceRead
from naics_embedder.decision.records import ArmRecord, Estimate, read_record
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.text_to_code import EUCLIDEAN_D16, LEXICAL_STUB, REFERENCE_ARMS
from naics_embedder.panels.ita_evidence import ItaEvidence
from naics_embedder.panels.ita_split import ITA_PANEL
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.supervision.artifacts import sha256_file

# Every interval's level
GAP_LEVEL = 0.95

# The unit-draw streams of the copied stratum and of the code-matched pair (P5)
COPIED_STREAM = f'{ITA_PANEL}/copied'
MATCHED_STREAM = f'{ITA_PANEL}/matched'

# The census rows a read keeps from its outcome scores
_CENSUS_STATISTIC = 'mrr'

# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GapSeed:
    '''
    One seed's frames for the gap.

    Attributes:
        seed: The ``euclidean-d16`` seed; None for the stub.
        ita: The ITA read's per-pair frame.
        census: One row per outcome validation query: ``unit`` (its code), ``item`` (its query id)
            and ``value`` (its reciprocal rank).
    '''

    seed: Optional[int]
    ita: pl.DataFrame
    census: pl.DataFrame

def _census_rows(scores: pl.DataFrame) -> pl.DataFrame:
    '''An outcome scores frame's census rows: panel ``outcome``, statistic ``mrr``.'''

    return scores.filter(
        (pl.col('panel') == OUTCOME_PANEL) & (pl.col('statistic') == _CENSUS_STATISTIC)
    ).select('unit', 'item', 'value')

def _require_one(reads: Sequence[ItaReferenceRead], field: str) -> None:
    '''Refuse reads that name more than one value of a field.'''

    values = {
        value.model_dump_json() if isinstance(value, BaseModel) else value
        for value in (getattr(read, field) for read in reads)
    }
    if len(values) > 1:
        raise ValueError(
            f'the reads name {len(values)} values of {field}: a gap pairs reads of one run, one '
            'pin and one table build'
        )

def gap_seeds(
    reads: Sequence[ItaReferenceRead],
    store: ArtifactStore,
    stage8: ArmRecord,
) -> Dict[str, List[GapSeed]]:
    '''
    Each reference arm's seeds, from Stage 4's reads and the Stage 8 record of ``euclidean-d16``.

    The stub's census rows come from its stored outcome scores, and each ``euclidean-d16`` seed's
    from the record run's stored ``scores`` (``store.read_frame`` checks the hash), filtered to
    panel ``outcome`` and statistic ``mrr``.

    Args:
        reads: The six read records: the stub with an outcome read beside, and d16's seeds.
        store: The artifact store the reads and the record name.
        stage8: The Stage 8 record of ``euclidean-d16``.

    Returns:
        ``lexical-stub``'s one seed, then ``euclidean-d16``'s seeds in seed order.

    Raises:
        ValueError: When the arms are not exactly ``REFERENCE_ARMS``; the reads name more than one
            run, evidence, split fingerprint or data fingerprint; the stub has other than one read
            or its read has no outcome; a d16 read has an outcome; d16's seeds repeat or are not
            the record's; a d16 read's checkpoint or table is not its seed's in the record; or the
            stub's outcome fingerprints are not the record's outcome panel.
    '''

    arms = {read.arm for read in reads}
    if arms != set(REFERENCE_ARMS):
        raise ValueError(f'the reads hold arms {sorted(arms)}, not {list(REFERENCE_ARMS)}')
    for field in ('run', 'evidence', 'split_fingerprint', 'data_fingerprint'):
        _require_one(reads, field)
    stubs = [read for read in reads if read.arm == LEXICAL_STUB]
    if len(stubs) != 1:
        raise ValueError(f'the stub has {len(stubs)} reads, not one')
    [stub] = stubs
    if stub.outcome is None:
        raise ValueError('the stub read has no outcome read beside it: C needs one (D1)')
    if (stub.outcome.fingerprint, stub.outcome.data_fingerprint) != (
        stage8.panels.outcome, stage8.panels.outcome_data
    ):
        raise ValueError(
            "the stub's outcome read is not on the Stage 8 record's outcome split: fingerprints "
            f'{stub.outcome.fingerprint} and {stub.outcome.data_fingerprint}, not '
            f'{stage8.panels.outcome} and {stage8.panels.outcome_data}'
        )
    d16 = [read for read in reads if read.arm == EUCLIDEAN_D16]
    for read in d16:
        if read.outcome is not None:
            raise ValueError(
                f'euclidean-d16 seed {read.seed} has an outcome read: its C is the Stage 8 '
                "record's stored scores (P4)"
            )
    seeds = [read.seed for read in d16]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'euclidean-d16 repeats a seed: {sorted(seeds)}')
    runs = {run.seed: run for run in stage8.runs}
    if set(seeds) != set(runs):
        raise ValueError(
            f"euclidean-d16's reads are seeds {sorted(seeds)}, not the Stage 8 record's "
            f'{sorted(runs)}'
        )
    for read in d16:
        run = runs[read.seed]
        if read.checkpoint_sha256 != run.checkpoint.sha256:
            raise ValueError(
                f'euclidean-d16 seed {read.seed} read checkpoint {read.checkpoint_sha256}, not the '
                f"Stage 8 record's {run.checkpoint.sha256}"
            )
        if read.table_fingerprint != run.table.matrix_fingerprint:
            raise ValueError(
                f'euclidean-d16 seed {read.seed} read table {read.table_fingerprint}, not the '
                f"Stage 8 record's {run.table.matrix_fingerprint}"
            )
    stub_seed = GapSeed(
        seed=None,
        ita=store.read_frame(stub.per_pair),
        census=_census_rows(store.read_frame(stub.outcome.scores)),
    )
    d16_seeds = [
        GapSeed(
            seed=read.seed,
            ita=store.read_frame(read.per_pair),
            census=_census_rows(store.read_frame(runs[read.seed].scores)),
        ) for read in sorted(d16, key=lambda read: read.seed)
    ]
    return {LEXICAL_STUB: [stub_seed], EUCLIDEAN_D16: d16_seeds}

# -------------------------------------------------------------------------------------------------
# One arm's gap
# -------------------------------------------------------------------------------------------------

class ArmGap(BaseModel):
    '''
    One reference arm's MRRs and gaps, each with its 95 % interval (D1, D2).

    Attributes:
        arm: The arm.
        n_seeds: Its seeds: 1 for the stub.
        census: C, on the outcome validation queries.
        copied: K, on the ITA validation copied pairs.
        headline: H, on the ITA validation headline pairs.
        census_matched: C over the matched codes, each code one weight.
        headline_matched: H over the matched codes, each code one weight.
        n_matched_codes: The codes M with a headline pair and an outcome validation query.
        n_unmatched_codes: The headline pairs' codes outside M.
        headline_gap: C − H.
        copied_gap: C − K.
        wording_gap: K − H.
        matched_headline_gap: C − H over the matched codes.
        threshold: r × C's point.
        large: Whether the matched gap's lower bound exceeds the threshold (D2).
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    arm: str
    n_seeds: int
    census: Estimate
    copied: Estimate
    headline: Estimate
    census_matched: Estimate
    headline_matched: Estimate
    n_matched_codes: int
    n_unmatched_codes: int
    headline_gap: Estimate
    copied_gap: Estimate
    wording_gap: Estimate
    matched_headline_gap: Estimate
    threshold: float
    large: bool

_Stratum = Tuple[float, np.ndarray]

def _stratum(
    frames: Sequence[pl.DataFrame],
    stream: str,
    seeds: np.ndarray,
    replicates: int,
    bootstrap_seed: int,
) -> _Stratum:
    '''One stratum's point and replicates: its own unit draws, the arm's shared seed draws.'''

    items = PanelItems.from_values(frames[0])
    sums = items.sums(np.stack([items.values(frame) for frame in frames]))
    units = unit_draws(stream, len(items.units), replicates, bootstrap_seed)
    return (
        point_statistic(sums, items.sizes),
        replicate_statistics(sums, items.sizes, units, seeds),
    )

def _pair_values(per_pair: pl.DataFrame, *, copied: bool) -> pl.DataFrame:
    '''The copied or headline pairs as (unit, item, value): description key, pair, rank.'''

    return per_pair.filter(pl.col('copied') == copied).select(
        unit=pl.col('description_key'),
        item=pl.col('pair_id'),
        value=pl.col('reciprocal_rank'),
    )

def _code_means(
    frame: pl.DataFrame, code_column: str, value_column: str, codes: Sequence[str]
) -> pl.DataFrame:
    '''Each code's correctly rounded mean value, as (unit, item, value) with unit = item = code.'''

    values: Dict[str, List[float]] = {code: [] for code in codes}
    for code, value in frame.select(code_column, value_column).iter_rows():
        if code in values:
            values[code].append(value)
    empty = [code for code in codes if not values[code]]
    if empty:
        raise ValueError(f'a seed holds no {value_column} for the matched codes {empty}')
    return pl.DataFrame(
        {
            'unit': list(codes),
            'item': list(codes),
            'value': [math.fsum(values[code]) / len(values[code]) for code in codes],
        },
        schema={
            'unit': pl.Utf8,
            'item': pl.Utf8,
            'value': pl.Float64
        },
    )

def _estimate(stratum: _Stratum) -> Estimate:
    return Estimate(point=stratum[0], interval=percentile_interval(stratum[1], GAP_LEVEL))

def _gap(first: _Stratum, second: _Stratum) -> Estimate:
    return Estimate(
        point=first[0] - second[0],
        interval=percentile_interval(first[1] - second[1], GAP_LEVEL),
    )

def arm_gap(
    arm: str,
    seeds: Sequence[GapSeed],
    *,
    r: float,
    replicates: int,
    bootstrap_seed: int,
) -> ArmGap:
    '''
    One arm's C, K and H, the gaps between them, and the code-matched gap (D1, D2, P5, P6).

    Args:
        arm: The arm.
        seeds: Its seeds' frames, from ``gap_seeds``.
        r: The relative margin, ``conf/data/ita_panel.yaml``'s 0.05.
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.

    Returns:
        The arm's gap.

    Raises:
        ValueError: When the arm has no seed, the part has no copied or no headline pair, no code
            is matched, or a seed's frames hold other items than the first seed's.
    '''

    if not seeds:
        raise ValueError(f'arm {arm!r} has no seed')
    first = seeds[0]
    headline_frames = [_pair_values(seed.ita, copied=False) for seed in seeds]
    copied_frames = [_pair_values(seed.ita, copied=True) for seed in seeds]
    for name, frames in (('headline', headline_frames), ('copied', copied_frames)):
        if frames[0].is_empty():
            raise ValueError(f'the ITA validation part has no {name} pair: no {name} MRR')
    headline_codes = set(first.ita.filter(~pl.col('copied')).get_column('codes').explode())
    matched = sorted(headline_codes & set(first.census.get_column('unit')))
    if not matched:
        raise ValueError('no code has both a headline pair and an outcome validation query')
    matched_census = []
    matched_headline = []
    for seed in seeds:
        headline = seed.ita.filter(~pl.col('copied')).explode('codes')
        matched_census.append(_code_means(seed.census, 'unit', 'value', matched))
        matched_headline.append(_code_means(headline, 'codes', 'reciprocal_rank', matched))
    draws = seed_draws(ITA_PANEL, arm, len(seeds), replicates, bootstrap_seed)
    settings = {'seeds': draws, 'replicates': replicates, 'bootstrap_seed': bootstrap_seed}
    headline = _stratum(headline_frames, ITA_PANEL, **settings)
    copied = _stratum(copied_frames, COPIED_STREAM, **settings)
    census = _stratum([seed.census for seed in seeds], OUTCOME_PANEL, **settings)
    census_matched = _stratum(matched_census, MATCHED_STREAM, **settings)
    headline_matched = _stratum(matched_headline, MATCHED_STREAM, **settings)
    matched_gap = _gap(census_matched, headline_matched)
    threshold = r * census[0]
    return ArmGap(
        arm=arm,
        n_seeds=len(seeds),
        census=_estimate(census),
        copied=_estimate(copied),
        headline=_estimate(headline),
        census_matched=_estimate(census_matched),
        headline_matched=_estimate(headline_matched),
        n_matched_codes=len(matched),
        n_unmatched_codes=len(headline_codes) - len(matched),
        headline_gap=_gap(census, headline),
        copied_gap=_gap(census, copied),
        wording_gap=_gap(copied, headline),
        matched_headline_gap=matched_gap,
        threshold=threshold,
        large=matched_gap.interval[0] > threshold,
    )

# -------------------------------------------------------------------------------------------------
# The record
# -------------------------------------------------------------------------------------------------

class ItaGapRecord(BaseModel):
    '''
    Stage 4's gap: both reference arms' MRRs and gaps, and D2's recommendation.

    Attributes:
        kind: The record's kind.
        run: The run id of the six reads.
        evidence: The pin the reads' tables were checked against (C1).
        split_fingerprint: The ITA split fingerprint.
        data_fingerprint: The ITA validation part's data fingerprint: the table build read.
        outcome_fingerprint: The outcome panel's role fingerprint.
        outcome_data_fingerprint: Its validation split's data fingerprint.
        stage8_sha256: The SHA-256 of the Stage 8 record of ``euclidean-d16``.
        read_sha256: Each read record's file name and its SHA-256.
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.
        level: The intervals' level.
        r: The relative margin D2 ties the threshold to.
        arms: Each arm's gap, the stub first.
        recommend_training_part: Whether every arm's gap is large (D2).
        created_at: When the record was made, in UTC.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    kind: Literal['ita_gap'] = 'ita_gap'
    run: str
    evidence: ItaEvidence
    split_fingerprint: str
    data_fingerprint: str
    outcome_fingerprint: str
    outcome_data_fingerprint: str
    stage8_sha256: str
    read_sha256: Dict[str, str]
    replicates: int
    bootstrap_seed: int
    level: float
    r: float
    arms: List[ArmGap]
    recommend_training_part: bool
    created_at: datetime

def gap_record(
    read_paths: Sequence[Path],
    store: ArtifactStore,
    stage8_path: Path,
    *,
    r: float,
    replicates: int,
    bootstrap_seed: int,
) -> ItaGapRecord:
    '''
    The gap record from Stage 4's six read records and the Stage 8 record of ``euclidean-d16``.

    Args:
        read_paths: The read records.
        store: The artifact store they name.
        stage8_path: The Stage 8 record.
        r: The relative margin (``conf/data/ita_panel.yaml``).
        replicates: The number of resamples (``conf/data/decision.yaml``).
        bootstrap_seed: The bootstrap's seed (``conf/data/decision.yaml``).

    Returns:
        The record. It reads no panel and logs nothing.

    Raises:
        ValueError: When two read records share a file name, or as ``gap_seeds`` and ``arm_gap``.
    '''

    paths = [Path(path) for path in read_paths]
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError(f'two read records share a file name: {sorted(names)}')
    read_sha256 = {path.name: sha256_file(path) for path in paths}
    reads = [read_record(path, ItaReferenceRead) for path in paths]
    stage8 = read_record(stage8_path, ArmRecord)
    seeds = gap_seeds(reads, store, stage8)
    settings = {'r': r, 'replicates': replicates, 'bootstrap_seed': bootstrap_seed}
    arms = [arm_gap(arm, seeds[arm], **settings) for arm in REFERENCE_ARMS]
    first = reads[0]
    return ItaGapRecord(
        run=first.run,
        evidence=first.evidence,
        split_fingerprint=first.split_fingerprint,
        data_fingerprint=first.data_fingerprint,
        outcome_fingerprint=stage8.panels.outcome,
        outcome_data_fingerprint=stage8.panels.outcome_data,
        stage8_sha256=sha256_file(stage8_path),
        read_sha256=read_sha256,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        level=GAP_LEVEL,
        r=r,
        arms=arms,
        recommend_training_part=all(arm.large for arm in arms),
        created_at=datetime.now(timezone.utc),
    )
