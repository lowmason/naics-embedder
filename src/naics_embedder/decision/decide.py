'''
Margins from a reference arm, and a decision over arms (Req 5; D8, D10, D11).

Before any number is computed, every arm is checked:

- it has at least ``min_seeds`` seeds, each read once on each of the three panels;
- every stored artifact still hashes to its reference;
- its text-only table's provenance matches the arm's backbone, revision, descriptions and
  window (D9);
- each run's log records are validation reads that name the run, its table and its text-only
  table by the fingerprints the store recorded;
- all arms read the same panels with the same fit settings, so Δ pairs item for item.

A decision also requires every run other than the margin record's own reference runs to have
read nothing before the margins were fixed.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.decision.records import (
    ArmRecord,
    ArmReport,
    ArmSpec,
    Comparison,
    DecisionRecord,
    DecisionSettings,
    Estimate,
    MarginRecord,
    PanelMargin,
    SeedRun,
    TextOnlyRef,
)
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import (
    NONINFERIORITY_LEVEL,
    SUPERIORITY_LEVEL,
    compare,
    compare_panel,
    non_dominated,
    tie_order,
)
from naics_embedder.decision.scores import (
    DECISION_STATISTIC,
    EMBEDDING_COMPARATOR,
    HELDOUT_PANEL,
    ORIENTATION,
    PANELS,
    REGRESSOR_PANELS,
    SPARSE_COMPARATOR,
    STATISTIC_DEFINITIONS,
    panel_statistic,
    statistic_means,
    statistic_values,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import VALIDATION

# Req 5: each arm runs at least 5 seeds; a caller can ask for more, never fewer
MIN_SEEDS = 5

# -------------------------------------------------------------------------------------------------
# Guards
# -------------------------------------------------------------------------------------------------

def check_text_only(spec: ArmSpec, text_only: TextOnlyRef) -> None:
    '''
    Require the text-only table to come from the arm's backbone reading the arm's text (D9).

    Raises:
        ValueError: If the provenance's backbone, revision, descriptions sha256 or window
            differs from the arm's.
    '''

    built = (
        text_only.backbone, text_only.revision, text_only.descriptions_sha256, text_only.max_length
    )
    reads = (spec.backbone, spec.backbone_revision, spec.descriptions_sha256, spec.max_length)
    if built != reads:
        raise ValueError(
            f'{spec.name}: the text-only table was built from {built}, the arm reads {reads} '
            "(D9: the arm's own backbone reading the arm's text)"
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
    Require an arm record to be complete, intact and read as the decision reads it.

    Raises:
        ValueError: If ``min_seeds`` is below Req 5's floor of ``MIN_SEEDS``, or on the first
            problem found.
    '''

    if min_seeds < MIN_SEEDS:
        raise ValueError(
            f'min_seeds is {min_seeds}: Req 5 needs at least {MIN_SEEDS} seeds per arm'
        )
    name = arm.spec.name
    seeds = [run.seed for run in arm.runs]
    if len(set(seeds)) < min_seeds:
        raise ValueError(f'{name}: {len(set(seeds))} distinct seeds, fewer than {min_seeds}')
    if len(set(seeds)) != len(seeds) or len({run.run_id for run in arm.runs}) != len(arm.runs):
        raise ValueError(f'{name}: a seed or run id repeats')
    check_text_only(arm.spec, arm.text_only)
    store.resolve(arm.text_only.table)
    store.resolve(arm.text_only.provenance)
    for run in arm.runs:
        for reference in (run.checkpoint, run.table, run.scores, run.decoding, run.predictions):
            store.resolve(reference)
        _check_log_records(arm, run)

def _check_log_records(arm: ArmRecord, run: SeedRun) -> None:
    name = f'{arm.spec.name} seed {run.seed}'
    panels = sorted(record['panel'] for record in run.log_records)
    if panels != sorted(PANELS):
        raise ValueError(f'{name}: the run read {panels}, not each of {sorted(PANELS)} once')
    for record in run.log_records:
        detail = record['detail']
        if record['event'] != 'read' or record['split'] != VALIDATION:
            raise ValueError(
                f'{name}: a {record["event"]} of the {record["split"]} split; a decision reads '
                'validation splits only (Req 4)'
            )
        if detail.get('run') != run.run_id:
            raise ValueError(f'{name}: a log record names another run')
        if record['panel'] == OUTCOME_PANEL:
            named = {'fingerprint': arm.panels.outcome, 'table': run.table.matrix_fingerprint}
            logged = {'fingerprint': record['fingerprint'], 'table': detail.get('table')}
        else:
            named = {
                'fingerprint': arm.panels.regressor,
                'arm': run.table.matrix_fingerprint,
                'text_only': arm.text_only.table.matrix_fingerprint,
            }
            logged = {key: detail.get(key) for key in ('arm', 'text_only')}
            logged['fingerprint'] = record['fingerprint']
        wrong = sorted(key for key in named if logged[key] != named[key])
        if wrong:
            raise ValueError(f'{name}: a {record["panel"]} read names another {wrong}')

def check_pairing(arms: Sequence[ArmRecord], margins: MarginRecord) -> None:
    '''
    Require every arm, and the margins' reference, to have read the same panels under the same
    fit settings.

    Raises:
        ValueError: If two arms share a name, or one read other panels.
    '''

    names = [arm.spec.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f'arm names repeat: {names}')
    others = [(arm.spec.name, arm.panels) for arm in arms[1:]]
    others.append((f'the margin reference {margins.reference.spec.name}', margins.reference.panels))
    for name, panels in others:
        if panels != arms[0].panels:
            raise ValueError(
                f'{name} read other panels or fit settings than {arms[0].spec.name}: paired '
                'arms must be scored on the same resample units (Req 5)'
            )

def check_margins_first(arms: Sequence[ArmRecord], margins: MarginRecord) -> None:
    '''
    Require every run but the margin record's reference runs to read after the margins were fixed.

    Raises:
        ValueError: If a run read before ``margins.fixed_at``.
    '''

    reference = {run.run_id for run in margins.reference.runs}
    for arm in arms:
        for run in arm.runs:
            if run.run_id in reference:
                continue
            first = min(datetime.fromisoformat(record['time']) for record in run.log_records)
            if first < margins.fixed_at:
                raise ValueError(
                    f'{arm.spec.name} seed {run.seed} read at {first.isoformat()}, before the '
                    f'margins were fixed at {margins.fixed_at.isoformat()} (Req 5)'
                )

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def fix_margins(
    reference: ArmRecord,
    multiple: float,
    name: str,
    store: ArtifactStore,
    *,
    min_seeds: int,
) -> MarginRecord:
    '''
    Each panel's δ: ``multiple`` times the reference arm's across-seed standard deviation of the
    panel's decision statistic (Req 5).

    Raises:
        ValueError: If the multiple is not positive, the reference fails ``check_arm``, or a
            panel's statistic does not vary across seeds.
    '''

    if multiple <= 0:
        raise ValueError(f'the margin multiple must be positive, got {multiple}')
    check_arm(reference, store, min_seeds)
    scores = [store.read_frame(run.scores) for run in reference.runs]
    margins: List[PanelMargin] = []
    for panel in PANELS:
        statistic = DECISION_STATISTIC[panel]
        per_seed = [panel_statistic(frame, panel, statistic) for frame in scores]
        sd = float(np.std(per_seed, ddof=1))
        if sd == 0.0:
            raise ValueError(f"{panel}: the reference's {statistic} does not vary across seeds")
        margins.append(
            PanelMargin(
                panel=panel, statistic=statistic, per_seed=per_seed, sd=sd, margin=multiple * sd
            )
        )
    return MarginRecord(
        name=name,
        multiple=multiple,
        reference=reference,
        margins=margins,
        fixed_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Resampling every arm
# -------------------------------------------------------------------------------------------------

def _values(
    store: ArtifactStore,
    arms: Sequence[ArmRecord],
    panel: str,
    statistic: str,
    items: Optional[PanelItems] = None,
) -> Tuple[PanelItems, Dict[str, np.ndarray]]:
    '''
    Each arm's (seeds, items) values of one statistic, on items every arm shares: ``items``, or
    else the first seed's.
    '''

    frames = {
        arm.spec.name: [
            statistic_values(store.read_frame(run.scores), panel, statistic) for run in arm.runs
        ]
        for arm in arms
    }
    items = items or PanelItems.from_values(next(iter(frames.values()))[0])
    return items, {
        name: np.stack([items.values(frame) for frame in seeds])
        for name, seeds in frames.items()
    }

def _replicates(
    items: PanelItems,
    values: np.ndarray,
    panel: str,
    arm: str,
    units: np.ndarray,
    settings: DecisionSettings,
) -> Tuple[float, np.ndarray]:
    sums = items.sums(values)
    seeds = seed_draws(panel, arm, values.shape[0], settings.replicates, settings.bootstrap_seed)
    return point_statistic(sums, items.sizes), replicate_statistics(sums, items.sizes, units, seeds)

# -------------------------------------------------------------------------------------------------
# Decision
# -------------------------------------------------------------------------------------------------

def decide(
    name: str,
    question: str,
    arms: Sequence[ArmRecord],
    margins: MarginRecord,
    store: ArtifactStore,
    *,
    replicates: int,
    bootstrap_seed: int,
    min_seeds: int,
) -> DecisionRecord:
    '''
    Req 5's decision over two or more arms.

    Raises:
        ValueError: If fewer than two arms are given, or a guard fails (module docstring).
        TieUnresolvedError: If the tie order cannot separate the surviving arms.
    '''

    if len(arms) < 2:
        raise ValueError('a decision compares at least two arms')
    for arm in arms:
        check_arm(arm, store, min_seeds)
    check_pairing(arms, margins)
    check_margins_first(arms, margins)
    settings = DecisionSettings(
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        min_seeds=min_seeds,
        noninferiority_level=NONINFERIORITY_LEVEL,
        superiority_level=SUPERIORITY_LEVEL,
    )

    names = [arm.spec.name for arm in arms]
    points: Dict[str, Dict[str, float]] = {panel: {} for panel in PANELS}
    draws: Dict[str, Dict[str, np.ndarray]] = {panel: {} for panel in PANELS}
    gains: Dict[str, Dict[str, Estimate]] = {arm: {} for arm in names}
    for panel in PANELS:
        items, values = _values(store, arms, panel, DECISION_STATISTIC[panel])
        units = unit_draws(panel, len(items.units), replicates, bootstrap_seed)
        for arm in names:
            points[panel][arm], draws[panel][arm] = _replicates(
                items, values[arm], panel, arm, units, settings
            )
        if panel in REGRESSOR_PANELS:
            _, sparse = _values(store, arms, panel, SPARSE_COMPARATOR[panel], items)
            for arm in names:
                point, reps = _replicates(
                    items, sparse[arm] - values[arm], panel, arm, units, settings
                )
                gains[arm][panel] = Estimate(
                    point=point, interval=percentile_interval(reps, NONINFERIORITY_LEVEL)
                )

    comparisons: List[Comparison] = []
    for a in names:
        for b in names:
            if a == b:
                continue
            panels = [
                compare_panel(
                    panel,
                    ORIENTATION[panel] * (points[panel][a] - points[panel][b]),
                    ORIENTATION[panel] * (draws[panel][a] - draws[panel][b]),
                    margins.margin(panel),
                ) for panel in PANELS
            ]
            comparisons.append(compare(a, b, panels))

    survivors, cycle = non_dominated(names, comparisons)
    heldout_gain = {arm: gains[arm][HELDOUT_PANEL].point for arm in names}
    order = tie_order([arm.spec for arm in arms if arm.spec.name in survivors], heldout_gain)
    return DecisionRecord(
        name=name,
        question=question,
        created_at=datetime.now(timezone.utc),
        statistics=dict(STATISTIC_DEFINITIONS),
        settings=settings,
        margins=margins,
        arms=list(arms),
        comparisons=comparisons,
        non_dominated=survivors,
        cycle=cycle,
        tie_order=order,
        heldout_gain=heldout_gain,
        chosen=order[0],
        reports=[
            _report(
                arm, store, {panel: points[panel][arm.spec.name]
                             for panel in PANELS}, gains[arm.spec.name]
            ) for arm in arms
        ],
    )

def _report(
    arm: ArmRecord,
    store: ArtifactStore,
    statistics: Mapping[str, float],
    gain: Mapping[str, Estimate],
) -> ArmReport:
    '''Every other statistic D10 reports, each a mean over the arm's seeds.'''

    scores = [store.read_frame(run.scores) for run in arm.runs]
    means = [statistic_means(frame) for frame in scores]
    averaged = {
        panel: {
            statistic: float(np.mean([seed[panel][statistic] for seed in means]))
            for statistic in means[0][panel]
        }
        for panel in means[0]
    }
    by_year: Dict[str, Dict[str, float]] = {}
    heldout = pl.concat([frame.filter(pl.col('panel') == HELDOUT_PANEL)
                         for frame in scores]).group_by('feature_year',
                                                        'statistic').agg(pl.col('value').mean())
    # Each seed scores the same rows, so the pooled mean is the mean over seeds
    for year, statistic, value in heldout.sort('feature_year', 'statistic').iter_rows():
        if statistic in (EMBEDDING_COMPARATOR, SPARSE_COMPARATOR[HELDOUT_PANEL]):
            by_year.setdefault(str(year), {})[statistic] = float(value)
    for values in by_year.values():
        values['gain'] = values[SPARSE_COMPARATOR[HELDOUT_PANEL]] - values[EMBEDDING_COMPARATOR]
    return ArmReport(
        arm=arm.spec.name,
        statistics=dict(statistics),
        outcome_metrics=averaged[OUTCOME_PANEL],
        comparator_mse={panel: averaged[panel]
                        for panel in REGRESSOR_PANELS},
        gain=dict(gain),
        heldout_by_feature_year=by_year,
    )
