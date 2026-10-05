'''
Read trained seeds through their selected checkpoints (spec 4.6).

``check`` verifies the durable monitor records and both checkpoints without loading a model or
exporting a table. ``tools sweep`` checks every seed before its first decision read. ``run`` then
exports the selected checkpoint's table and returns the encoder and the reads that selected it.
'''

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import torch

from naics_embedder.decision.decide import _monitor_record_problem
from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.sweep import SeedArtifacts
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
    shared_encoder_architecture,
    validate_supervision_contract,
)
from naics_embedder.text_model.arm_encoder import ArmEncoder
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.text_model.monitor import MONITOR_RECORDS, read_monitor_records
from naics_embedder.text_model.naics_model import TRAINING_RUN_KEY
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import (
    outcome_checkpoint,
    read_checkpoint,
    refuse_a_resume_under_other_settings,
)

# -------------------------------------------------------------------------------------------------
# A selection, checked without encoding
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedSelection:
    '''A seed's earliest best epoch, its checkpoint and the monitor records through last.ckpt.'''

    checkpoint: Path
    epoch: int
    training_run: str
    mrr: float
    monitor_records: Tuple[Dict[str, Any], ...]

# -------------------------------------------------------------------------------------------------
# The checkpoint runner
# -------------------------------------------------------------------------------------------------

class CheckpointRunner:
    '''
    Load trained seeds from their run directories, refusing incomplete or inconsistent runs.

    Args:
        cfg: The arm's configuration, including the descriptions and tokenization window.
        bundle: Its validated supervision bundle.
        run_directory: Resolve each seed's checkpoint directory, as pulled from training.
        device: Where exports and query encoding run; independent of the training accelerator.
    '''

    def __init__(
        self, cfg: Config, bundle: ValidatedSupervisionBundle, *,
        run_directory: Callable[[int], Path], device: Union[str, torch.device]
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.run_directory = run_directory
        self.device = device

    def _check_checkpoint(self, saved: Dict[str, Any], spec: ArmSpec, seed: int) -> None:
        contract = validate_supervision_contract(
            saved.get(CHECKPOINT_KEY),
            self.bundle.manifest,
            summaries=summaries_identity(self.cfg.data_loader.tokenization.tokenizer_name)
        )
        expected = shared_encoder_architecture(
            fusion=spec.settings.get('fusion', self.cfg.model.fusion),
            dimension=spec.dimension,
            backbone=spec.backbone
        )
        if contract.encoder != expected:
            raise ValueError('the checkpoint encoder contract differs from the arm (D2)')
        refuse_a_resume_under_other_settings(saved, spec.settings, seed=seed)

    def check(self, spec: ArmSpec, seed: int) -> SeedSelection:
        '''
        Check one seed without exporting or reading a decision panel (P25, spec 5).

        Records must cover every epoch through ``last.ckpt`` exactly once and name its training
        run. The earliest epoch with the highest MRR must have one unambiguous checkpoint,
        whose identity, contract, settings and saved best score agree exactly.

        Raises:
            ValueError: Naming the seed and its first missing or inconsistent artifact.
        '''

        try:
            return self._selection(spec, seed)
        except (OSError, ValueError) as exc:
            raise ValueError(f'{spec.name} seed {seed}: {exc}') from exc

    def _selection(self, spec: ArmSpec, seed: int) -> SeedSelection:
        directory = Path(self.run_directory(seed))
        records_path = directory / MONITOR_RECORDS
        if not records_path.is_file():
            raise ValueError(f'the run has no {MONITOR_RECORDS} at {records_path}')
        records = read_monitor_records(records_path)
        last_path = directory / 'last.ckpt'
        if not last_path.is_file():
            raise ValueError(f'the run has no last.ckpt at {last_path}')
        last = read_checkpoint(last_path)
        self._check_checkpoint(last, spec, seed)
        last_epoch = last.get('epoch')
        if isinstance(last_epoch, bool) or not isinstance(last_epoch, int) or last_epoch < 0:
            raise ValueError('last.ckpt names no non-negative integer epoch')
        training_run = last.get(TRAINING_RUN_KEY)
        if not isinstance(training_run, str) or not training_run.strip():
            raise ValueError('last.ckpt names no training run')
        epochs = []
        for record in records:
            problem = _monitor_record_problem(record)
            if problem is not None:
                raise ValueError(f'a monitor record is malformed: {problem}')
            detail = record['read']['detail']
            if detail.get('training_run') != training_run:
                raise ValueError('a monitor record names another training run than last.ckpt')
            epochs.append(detail['epoch'])
        repeated = sorted(epoch for epoch, count in Counter(epochs).items() if count > 1)
        missing = sorted(set(range(last_epoch + 1)) - set(epochs))
        extra = sorted(set(epochs) - set(range(last_epoch + 1)))
        if repeated or missing or extra:
            raise ValueError(
                f'monitor records must cover epochs 0 through {last_epoch} exactly once: '
                f'missing {missing}, repeated {repeated}, extra {extra}'
            )
        best = max(record['mrr'] for record in records)
        epoch = min(
            record['read']['detail']['epoch'] for record in records if record['mrr'] == best
        )
        checkpoint = directory / f'epoch={epoch:03d}.ckpt'
        if not checkpoint.is_file():
            raise ValueError(f'the selected epoch {epoch} has no checkpoint at {checkpoint}')
        siblings = sorted(directory.glob(f'epoch={epoch:03d}-v*.ckpt'))
        if siblings:
            raise ValueError(
                f'the selected checkpoint has version siblings {[path.name for path in siblings]}; '
                'the selected epoch is ambiguous after an interrupted save'
            )
        saved = read_checkpoint(checkpoint)
        self._check_checkpoint(saved, spec, seed)
        if saved.get('epoch') != epoch or isinstance(saved.get('epoch'), bool):
            raise ValueError(f'the selected checkpoint is not from epoch {epoch}')
        if saved.get(TRAINING_RUN_KEY) != training_run:
            raise ValueError('the selected checkpoint names another training run than last.ckpt')
        key = outcome_checkpoint(directory).state_key
        score = saved.get('callbacks', {}).get(key, {}).get('best_model_score')
        if isinstance(score, torch.Tensor) and score.numel() == 1:
            score = score.item()
        if (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or score != best
        ):
            raise ValueError(
                f'the selected checkpoint best_model_score {score!r} is not the monitor MRR '
                f'{best!r} exactly'
            )
        return SeedSelection(checkpoint, epoch, training_run, float(best), tuple(records))

    def run(self, spec: ArmSpec, seed: int) -> SeedArtifacts:
        '''Export the checked selected epoch, replacing any existing table and provenance pair.'''

        selected = self.check(spec, seed)
        token_config = code_token_config(self.cfg)
        output = selected.checkpoint.parent / f'arm_table_epoch={selected.epoch:03d}.parquet'
        table = export_code_table(
            selected.checkpoint, self.bundle, token_config, output, device=self.device
        )
        encoder = ArmEncoder.from_files(
            selected.checkpoint, table, self.bundle, token_config, device=self.device
        )
        return SeedArtifacts(
            checkpoint=selected.checkpoint,
            table=table,
            encoder=encoder,
            distance=encoder.distance,
            training_run=selected.training_run,
            checkpoint_epoch=selected.epoch,
            monitor_records=selected.monitor_records
        )
