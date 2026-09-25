'''
The seed-sweep driver (roadmap Stage 4): run a configuration for N seeds, read each seed once on
each of D8's three panels, and keep everything a decision record references.

A runner trains (or loads) one seed of a configuration and returns its encoder checkpoint, its
2,125-code table in the export form and a ``QueryCodeEncoder``. Until Stage 6 adds the export and
the query path, only synthetic runners exist, in tests.

Every read goes through the panels, so it is logged, and it carries the run's id, which is how
the arm record picks its runs' records out of the selection log. The checkpoint, the table, the
text-only table with its provenance, the scores, the decoding and the predictions go to the
artifact store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Union

import polars as pl

from naics_embedder.decision.decide import check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import DECISION_LEVEL, ArmTables, Regime, RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.schema import IndexRole

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Runner interface
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedArtifacts:
    '''
    What one seed of a configuration produces.

    Attributes:
        checkpoint: The encoder checkpoint file.
        table: The 2,125-code table in the export form (tangent coordinates if hyperbolic).
        encoder: Queries and codes embedded in one space, for the outcome panel.
        distance: The arm's decoding distance (``panels.decoding.DISTANCES``).
    '''

    checkpoint: Path
    table: Path
    encoder: QueryCodeEncoder
    distance: str

class ArmRunner(Protocol):
    '''Trains or loads one seed of a configuration.'''

    def run(self, spec: ArmSpec, seed: int) -> SeedArtifacts:
        ...

# -------------------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------------------

def _fit_settings(panel: RegressorPanel) -> Dict[str, Any]:
    settings = asdict(panel.settings)
    return {**settings, 'alphas': list(settings['alphas'])}

def _log_records(logs: Sequence[SelectionLog], run_id: str) -> List[Dict[str, Any]]:
    '''The run's records, from every distinct log file the panels write to.'''

    paths = dict.fromkeys(log.path.resolve() for log in logs)
    return [
        record for path in paths for record in SelectionLog(path).records()
        if record['detail'].get('run') == run_id
    ]

def run_seed_sweep(
    spec: ArmSpec,
    seeds: Sequence[int],
    runner: ArmRunner,
    *,
    outcome_panel: OutcomePanel,
    regressor_panel: RegressorPanel,
    text_only_table: Union[str, Path],
    store: ArtifactStore,
    purpose: str,
) -> ArmRecord:
    '''
    Run a configuration for each seed and read each seed once on each panel's validation split.

    Raises:
        ValueError: If a seed repeats, or the text-only table was not built from the arm's
            backbone, revision, descriptions and window (D9).
    '''

    if len(set(seeds)) != len(seeds):
        raise ValueError(f'a seed repeats: {list(seeds)}')
    text_only = store.put_text_only(text_only_table)
    check_text_only(spec, text_only)
    text_frame = pl.read_parquet(store.resolve(text_only.table))
    logs = [outcome_panel.log, regressor_panel.log]

    runs: List[SeedRun] = []
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
        table = store.put_table(artifacts.table)
        detail = {'run': run_id, 'arm_name': spec.name, 'seed': seed}
        arm = ArmTables.from_tables(pl.read_parquet(store.resolve(table)), text_frame)
        regressor_panel.require_arm(arm)
        decoding = outcome_panel.score(
            artifacts.encoder,
            IndexRole.VALIDATION,
            purpose,
            distance=artifacts.distance,
            detail={
                **detail, 'table': table.matrix_fingerprint
            },
        )
        predictions = pl.concat(
            [
                regressor_panel.validation(regime, DECISION_LEVEL, arm, purpose, detail=detail)
                for regime in Regime
            ]
        )
        scores = seed_scores(decoding.per_query, predictions, regressor_panel.settings.repeats)
        runs.append(
            SeedRun(
                seed=seed,
                run_id=run_id,
                checkpoint=checkpoint,
                table=table,
                scores=store.put_frame(scores, 'scores.parquet'),
                decoding=store.put_frame(decoding.per_query, 'decoding.parquet'),
                predictions=store.put_frame(predictions, 'predictions.parquet'),
                statistics={
                    panel: panel_statistic(scores, panel, DECISION_STATISTIC[panel])
                    for panel in PANELS
                },
                log_records=_log_records(logs, run_id),
            )
        )
        logger.info(f'{spec.name} seed {seed}: {runs[-1].statistics}')

    return ArmRecord(
        spec=spec,
        text_only=text_only,
        store=str(store.root),
        panels=PanelSet(
            outcome=outcome_panel.fingerprint,
            regressor=regressor_panel.fingerprint,
            fit_settings=_fit_settings(regressor_panel),
        ),
        runs=runs,
        created_at=datetime.now(timezone.utc),
    )
