'''
The economic artifact's decision (Req 24; Req 23; Req 18): the margin read, the candidates' reads
and the decision over the two growth panels.

- **Panels.** ``growth_seen`` and ``growth_heldout``, the growth outcome's regimes at level 6
  (Req 21), and no other: no artifact's decision reads another artifact's panel (Req 18).
- **Statistic.** D10's, applied to growth: each row's squared error averaged over its repeats
  (``decision.scores.regressor_scores``); a candidate's is ``covariates+embedding``'s MSE.
- **Reference.** The comparator ``covariates+ancestors`` (U9). It reads no candidate, and the
  folds depend on the regime, level, repeat and group alone, so the margin read's row errors are
  those every candidate's read computes, and they pair exactly.
- **Margins.** δ = r × the reference's validation MSE on each panel (U13), from a margin read that
  reads no candidate, fixed before any candidate is read.
- **Candidates.** A deterministic candidate, such as the text-only table or a tree embedding, is
  one replicate: its table, with the provenance its builder wrote, read once on each growth panel
  the margins read, with no checkpoint or monitor reads (Req 24; Req 23). Its reads pair with
  the margin read only where the reference predicts bit for bit as it did there.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

import polars as pl

from naics_embedder.decision.decide import _same, check_recorded_text_only, check_text_only
from naics_embedder.decision.records import (
    ArmSpec,
    EconomicArmRecord,
    EconomicRun,
    GrowthPanelSet,
    ReadEnvironment,
    ReferenceRead,
    RelativeMargin,
    RelativeMarginRecord,
)
from naics_embedder.decision.scores import EMBEDDING_COMPARATOR, panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import _fit_settings, _log_records
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    READ_NAMES,
    VALIDATION,
    ArmTables,
    Regime,
    RegressorPanel,
)
from naics_embedder.supervision.artifacts import sha256_file

# The growth outcome's two regimes: the economic decision's panels (Req 24)
GROWTH_PANELS = tuple(READ_NAMES[RegressorOutcome.GROWTH][regime] for regime in Regime)
# The hierarchy's ancestor indicators beside the covariates (U9)
REFERENCE_COMPARATOR = 'covariates+ancestors'

# -------------------------------------------------------------------------------------------------
# Reads
# -------------------------------------------------------------------------------------------------

def read_environment(lock: Union[str, Path]) -> ReadEnvironment:
    '''
    This process's platform and Python, and the sha256 of ``lock``, the ``uv.lock`` it runs under.

    Raises:
        FileNotFoundError: If ``lock`` is not a file.
    '''

    path = Path(lock)
    if not path.is_file():
        raise FileNotFoundError(f'{path}: no uv.lock to name the read by')
    return ReadEnvironment(
        platform=platform.platform(),
        python=platform.python_version(),
        lock_sha256=sha256_file(path),
    )

def growth_panel_set(panel: RegressorPanel) -> GrowthPanelSet:
    '''The growth panel's identity at the decision level, which every read of a decision shares.'''

    return GrowthPanelSet(
        regressor=panel.fingerprint,
        regressor_data=panel.data_fingerprint(DECISION_LEVEL),
        fit_settings=_fit_settings(panel),
    )

def _require_growth(panel: RegressorPanel, reader: str) -> None:
    '''
    Require the growth panel at the decision level: the economic decision's (Req 24; U8).

    Raises:
        ValueError: If the panel reads another outcome or lacks the decision level.
    '''

    if panel.outcome is not RegressorOutcome.GROWTH:
        raise ValueError(
            f'{reader}: the economic decision reads growth, not {panel.outcome.value} '
            '(Req 24; U8)'
        )
    if DECISION_LEVEL not in panel.levels:
        raise ValueError(
            f'{reader}: the economic decision reads level {DECISION_LEVEL}, which the panel has '
            'not loaded'
        )

def _require_same_panels(name: str, panels: GrowthPanelSet, margins: RelativeMarginRecord) -> None:
    '''
    Require a read of the growth panel the margins read, with its data and fit settings.

    Raises:
        ValueError: Naming what differs.
    '''

    expected = margins.read.panels.model_dump()
    wrong = sorted(key for key, value in panels.model_dump().items() if value != expected[key])
    if wrong:
        raise ValueError(
            f'{name} read other panels or fit settings than the margins {margins.name!r}, '
            f'differing in {wrong}: paired reads must score the same rows (Req 24)'
        )

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def fix_relative_margins(
    panel: RegressorPanel,
    r: float,
    name: str,
    store: ArtifactStore,
    *,
    purpose: str,
    lock: Union[str, Path],
) -> RelativeMarginRecord:
    '''
    Read the reference alone on each growth panel's validation split, and fix each panel's δ as
    ``r`` times the reference comparator's MSE there (Req 24).

    The read logs and scores only the comparators that read no candidate
    (``reference_comparators``), so no candidate's growth error exists before δ is recorded.

    Args:
        panel: The growth panel.
        r: Each δ as a multiple of the reference's MSE (U13).
        name: Names the margins in the decisions that use them.
        store: Where the read's predictions and scores go.
        purpose: Why the read happens, which the selection log records.
        lock: The ``uv.lock`` the read runs under, which the record names.

    Raises:
        ValueError: If ``r`` is not positive, or the panel is not the growth panel at the decision
            level, before anything is read.
        FileNotFoundError: If ``lock`` is not a file, before anything is read.
    '''

    if r <= 0:
        raise ValueError(f'the margin multiple r must be positive, got {r}')
    _require_growth(panel, 'the margins')
    environment = read_environment(lock)
    run_id = f'margins/{uuid.uuid4().hex}'
    predictions = pl.concat(
        [
            panel.reference_validation(regime, DECISION_LEVEL, purpose, detail={'run': run_id})
            for regime in Regime
        ]
    )
    scores = regressor_scores(predictions, panel.settings.repeats, GROWTH_PANELS)
    read = ReferenceRead(
        run_id=run_id,
        panels=growth_panel_set(panel),
        predictions=store.put_frame(predictions, 'predictions.parquet'),
        scores=store.put_frame(scores, 'scores.parquet'),
        log_records=_log_records([panel.log], run_id),
        environment=environment,
    )
    margins: List[RelativeMargin] = []
    for growth in GROWTH_PANELS:
        mse = panel_statistic(scores, growth, REFERENCE_COMPARATOR)
        margins.append(
            RelativeMargin(
                panel=growth, statistic=REFERENCE_COMPARATOR, reference_mse=mse, margin=r * mse
            )
        )
    return RelativeMarginRecord(
        name=name,
        r=r,
        reference=REFERENCE_COMPARATOR,
        read=read,
        margins=margins,
        fixed_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Candidates
# -------------------------------------------------------------------------------------------------

def read_deterministic_arm(
    spec: ArmSpec,
    table: Union[str, Path],
    *,
    regressor_panel: RegressorPanel,
    text_only_table: Union[str, Path],
    margins: RelativeMarginRecord,
    store: ArtifactStore,
    purpose: str,
    lock: Union[str, Path],
) -> EconomicArmRecord:
    '''
    Read a deterministic candidate once on each growth panel's validation split: a single
    replicate (Req 24), recorded with its table's provenance (Req 23).

    Every check runs before the first read.

    Args:
        spec: The candidate, with what the tie order and D9 read.
        table: Its 2,125-code table in the export form, with its builder's provenance beside it
            (``panels.text_only.provenance_path``).
        regressor_panel: The growth panel the margins read.
        text_only_table: The text-only table its reads score as ``covariates+text_only`` (D9).
        margins: The margins the candidate will be judged against.
        store: Where the tables, the provenance and the read's outputs go.
        purpose: Why the read happens, which the selection log records.
        lock: The ``uv.lock`` the read runs under, which the record names.

    Raises:
        ValueError: If the panel is not the growth panel the margins read, with their data and
            fit settings; if the provenance does not describe the table; if the text-only table
            was not built from what the candidate reads (D9); or if the table's width is not the
            spec's dimension.
        FileNotFoundError: If the table has no provenance beside it, or ``lock`` is not a file.
    '''

    _require_growth(regressor_panel, spec.name)
    panels = growth_panel_set(regressor_panel)
    _require_same_panels(spec.name, panels, margins)
    environment = read_environment(lock)
    stored, provenance = store.put_candidate(table)
    text_only = store.put_text_only(text_only_table)
    check_text_only(spec, text_only)
    arm = ArmTables.from_tables(
        pl.read_parquet(store.resolve(stored)), pl.read_parquet(store.resolve(text_only.table))
    )
    if arm.dimension != spec.dimension:
        raise ValueError(
            f"{spec.name}: the table has dimension {arm.dimension}, not the spec's "
            f'{spec.dimension}'
        )
    regressor_panel.require_arm(arm)

    run_id = f'{spec.name}/{uuid.uuid4().hex}'
    detail = {'run': run_id, 'arm_name': spec.name}
    predictions = pl.concat(
        [
            regressor_panel.validation(regime, DECISION_LEVEL, arm, purpose, detail=detail)
            for regime in Regime
        ]
    )
    scores = regressor_scores(predictions, regressor_panel.settings.repeats, GROWTH_PANELS)
    run = EconomicRun(
        run_id=run_id,
        table=stored,
        provenance=provenance,
        scores=store.put_frame(scores, 'scores.parquet'),
        predictions=store.put_frame(predictions, 'predictions.parquet'),
        statistics={
            panel: panel_statistic(scores, panel, EMBEDDING_COMPARATOR)
            for panel in GROWTH_PANELS
        },
        log_records=_log_records([regressor_panel.log], run_id),
    )
    return EconomicArmRecord(
        spec=spec,
        text_only=text_only,
        store=str(store.root),
        panels=panels,
        environment=environment,
        runs=[run],
        created_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def check_economic_arm(arm: EconomicArmRecord, store: ArtifactStore) -> None:
    '''
    Require an economic arm record to be a deterministic candidate's single replicate, intact,
    with its provenance, read once on each growth panel as the decision reads it (Req 24; Req 23).

    Raises:
        ValueError: On the first problem found.
    '''

    name = arm.spec.name
    if len(arm.runs) != 1 or arm.runs[0].seed is not None:
        raise ValueError(
            f'{name}: an economic arm is one deterministic replicate with no seed (Req 24), not '
            f'{len(arm.runs)} runs with seeds {[run.seed for run in arm.runs]}'
        )
    check_recorded_text_only(arm.spec, arm.text_only, store)
    (run, ) = arm.runs
    for reference in (run.table, run.scores, run.predictions):
        store.resolve(reference)
    if store.candidate_provenance(run.table, run.provenance.file) != run.provenance:
        raise ValueError(f"{name}: the record's provenance differs from its stored provenance")
    _check_growth_reads(arm, run)

def _check_growth_reads(arm: EconomicArmRecord, run: EconomicRun) -> None:
    '''Require the run's reads to be its table's validation reads, once on each growth panel.'''

    name = arm.spec.name
    panels = sorted(str(record.get('panel')) for record in run.log_records)
    if panels != sorted(GROWTH_PANELS):
        raise ValueError(f'{name}: the run read {panels}, not each of {sorted(GROWTH_PANELS)} once')
    named = {
        'fingerprint': arm.panels.regressor,
        'run': run.run_id,
        'level': DECISION_LEVEL,
        'arm': run.table.matrix_fingerprint,
        'text_only': arm.text_only.table.matrix_fingerprint,
        'dimension': arm.spec.dimension,
    }
    for record in run.log_records:
        if record.get('event') != 'read' or record.get('split') != VALIDATION:
            raise ValueError(
                f'{name}: a {record.get("event")} of the {record.get("split")} split; a decision '
                'reads validation splits only (Req 4)'
            )
        detail = record.get('detail') or {}
        logged = {key: detail.get(key) for key in named}
        logged['fingerprint'] = record.get('fingerprint')
        wrong = sorted(key for key in named if not _same(logged[key], named[key]))
        if wrong:
            raise ValueError(f'{name}: a {record["panel"]} read names another {wrong}')

def _reference_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    '''The reference's predictions, one per panel, repeat and row.'''

    return predictions.filter(pl.col('comparator') == REFERENCE_COMPARATOR
                              ).sort('panel', 'repeat', 'code', 'feature_year')

def _where(environment: ReadEnvironment) -> str:
    return (
        f'{environment.platform}, Python {environment.python}, uv.lock '
        f'{environment.lock_sha256[:12]}'
    )

def check_reference_pairing(
    margins: RelativeMarginRecord, arm: EconomicArmRecord, store: ArtifactStore
) -> None:
    '''
    Require the arm to have read the growth panel the margins read, and the reference to predict
    in each of its runs exactly as in the margin read, so its errors pair with δ's (Req 24).

    Only the predictions decide; the environments name where each read ran.

    Raises:
        ValueError: If the panels, data or fit settings differ, or the reference's predictions
            in a run differ from the margin read's.
    '''

    name = arm.spec.name
    _require_same_panels(name, arm.panels, margins)
    reference = _reference_rows(store.read_frame(margins.read.predictions))
    for run in arm.runs:
        if not _reference_rows(store.read_frame(run.predictions)).equals(reference):
            raise ValueError(
                f"{name}: its {REFERENCE_COMPARATOR} predictions differ from the margin read's, "
                f'so its errors do not pair with the reference (Req 24). It was read on '
                f'{_where(arm.environment)}; the margins on {_where(margins.read.environment)}'
            )
