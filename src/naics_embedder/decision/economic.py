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

from naics_embedder.decision.records import (
    GrowthPanelSet,
    ReadEnvironment,
    ReferenceRead,
    RelativeMargin,
    RelativeMarginRecord,
)
from naics_embedder.decision.scores import panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import _fit_settings, _log_records
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import DECISION_LEVEL, READ_NAMES, Regime, RegressorPanel
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
        raise ValueError(f'{reader} read growth, not {panel.outcome.value} (Req 24; U8)')
    if DECISION_LEVEL not in panel.levels:
        raise ValueError(f'{reader} read level {DECISION_LEVEL}, which the panel has not loaded')

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
