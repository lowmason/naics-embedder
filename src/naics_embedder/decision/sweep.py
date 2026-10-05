'''
The seed-sweep driver (roadmap Stage 4): run a configuration for N seeds, read each seed once on
each of D8's three panels, and keep everything a decision record references.

A runner trains (or loads) one seed of a configuration and returns its encoder checkpoint, its
2,125-code table in the export form and a ``QueryCodeEncoder``; for a trained seed, also its
training run, its checkpoint's epoch and the monitor reads that selected it, which the seed's
record carries (spec 4.4).

Every read goes through the panels, so it is logged, and it carries the run's id, which is how
the arm record picks its runs' records out of the selection log. The checkpoint, the table, the
text-only table with its provenance, the scores, the decoding and the predictions go to the
artifact store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union

import polars as pl

from naics_embedder.decision.decide import check_seed_table, check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
from naics_embedder.decision.store import ArtifactStore, provenance_fields
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    ArmTables,
    Regime,
    RegressorPanel,
    table_fingerprint,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
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
        table: The 2,125-code table in the export form (tangent coordinates if hyperbolic), with
            its export provenance beside it.
        encoder: Queries and codes embedded in one space, for the outcome panel.
        distance: The arm's decoding distance (``panels.decoding.DISTANCES``).
        training_run: The id of the training run the checkpoint is from; None for a seed that
            was not trained.
        checkpoint_epoch: The checkpoint's epoch, for a trained seed: the earliest with the
            highest monitor MRR.
        monitor_records: A trained seed's ``monitor_reads.jsonl`` records, oldest first: the
            reads that selected its checkpoint.
    '''

    checkpoint: Path
    table: Path
    encoder: QueryCodeEncoder
    distance: str
    training_run: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    monitor_records: Sequence[Dict[str, Any]] = ()

class ArmRunner(Protocol):
    '''Trains or loads one seed of a configuration.'''

    def run(self, spec: ArmSpec, seed: int) -> SeedArtifacts:
        ...

# -------------------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------------------

def _seed_table_fields(table: Path) -> Mapping[str, Any]:
    '''
    The D9 fields of a seed's table, from the export provenance beside it.

    Raises:
        ValueError: If the provenance is missing, or as ``provenance_fields``.
    '''

    provenance = provenance_path(table)
    if not provenance.is_file():
        raise ValueError(f'{table} has no export provenance at {provenance}')
    return provenance_fields(
        json.loads(provenance.read_text(encoding='utf-8')),
        sha256_file(table),
        table_fingerprint(pl.read_parquet(table)),
        str(provenance),
    )

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
        ValueError: If a seed repeats; if the text-only table, or a seed's table by its export
            provenance, was not built from the arm's backbone, revision, descriptions, summaries
            and window (D9); or if a seed's table width is not the arm spec's dimension.
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
        # What the seed read, before anything is stored or any panel is read
        check_seed_table(spec, seed, _seed_table_fields(Path(artifacts.table)))
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
        table = store.put_table(artifacts.table)
        detail = {'run': run_id, 'arm_name': spec.name, 'seed': seed}
        arm = ArmTables.from_tables(pl.read_parquet(store.resolve(table)), text_frame)
        if arm.dimension != spec.dimension:
            raise ValueError(
                f'{spec.name} seed {seed}: the table has dimension {arm.dimension}, not the '
                f"spec's dimension {spec.dimension}"
            )
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
                training_run=artifacts.training_run,
                checkpoint_epoch=artifacts.checkpoint_epoch,
                monitor_records=list(artifacts.monitor_records),
            )
        )
        logger.info(f'{spec.name} seed {seed}: {runs[-1].statistics}')

    return ArmRecord(
        spec=spec,
        text_only=text_only,
        store=str(store.root),
        panels=PanelSet(
            outcome=outcome_panel.fingerprint,
            outcome_data=outcome_panel.data_fingerprint(IndexRole.VALIDATION),
            regressor=regressor_panel.fingerprint,
            regressor_data=regressor_panel.data_fingerprint(DECISION_LEVEL),
            fit_settings=_fit_settings(regressor_panel),
        ),
        runs=runs,
        created_at=datetime.now(timezone.utc),
    )
