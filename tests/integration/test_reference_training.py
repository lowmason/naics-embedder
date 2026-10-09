'''
Training the reference bundle through ``create_trainer``: the Trainer-driven tests of spec 6's
Cache and Monitor criteria and of its schedule and health logs, and P27's resumed permutation.

Each run is built as ``train`` builds it, on the CPU and the tiny backbone: the model from
``build_model_from_config``, the two-stream datamodule from ``build_datamodule_from_config`` and the
Trainer from ``create_trainer``. The reference bundle (``tests/fixtures/supervision.py``) has 17
codes and 11 task queries, so at 4 queries a step an epoch has 3 steps. Its token cache is
``code_token_config``'s, moved under ``tmp_path``.

Every run has a monitor, since ModelCheckpoint and EarlyStopping read ``val/outcome_mrr`` (P30):
the real ``OutcomeMonitor``, built by ``build_monitor_from_config`` with its selection log under
``tmp_path`` (P28); a scripted one, which gives each epoch's MRR; or the real one with scripted
MRRs, which logs and records its reads as the real one does.

An exact resume builds a new model, datamodule, monitor and Trainer, as ``train`` does, and resumes
from the ``last.ckpt`` that ModelCheckpoint wrote during the interrupted fit. The interruption is a
lost instance: it dies saving an epoch's checkpoint, after the module's epoch-end hook has read and
recorded that epoch. Every resume test checks that no epoch-end hook replays, since a replay would
read the monitor, step the plateau and append to ``monitor_reads.jsonl`` a second time. A resume
P19's guards refuse never reaches the fit.
'''

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest
import pytorch_lightning as pyl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import Logger

from naics_embedder.cli.commands import training
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import matrix_fingerprint
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import CheckpointContract, validate_exact_resume
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader import datamodule as two_stream
from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, read_epoch_summary
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.text_model.mixins import OUTCOME_MRR
from naics_embedder.text_model.monitor import (
    MONITOR_RECORDS,
    CodeCache,
    MonitorRead,
    OutcomeMonitor,
    read_monitor_records,
)
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig
from naics_embedder.utils.training import (
    HardwareInfo,
    create_trainer,
    outcome_early_stopping,
    read_checkpoint,
    refuse_a_fresh_start_into_a_used_directory,
    refuse_a_resume_from_another_directory,
    refuse_a_resume_of_a_stopped_run,
    refuse_a_resume_under_other_settings,
    run_settings,
)
from tests.fixtures.shared_encoder import REFERENCE_QUERIES_PER_STEP, reference_step_dataset

pytestmark = pytest.mark.integration

# The run's seed: every epoch's two permutations are drawn from it, and every read names it
SEED = 7
# The reference bundle's steps an epoch, at 4 queries a step (spec 4.3)
STEPS = 3
LEVELS = range(2, 7)
TERMS = ('task', 'code_code', 'radial', 'total')
# P20's health keys at the reference bundle's levels
HEALTH_KEYS = {
    *(f'loss/{term}' for term in TERMS),
    'logit_scale/task',
    'logit_scale/code_code',
    *(f'radius/{statistic}/level_{level}' for statistic in ('mean', 'sd') for level in LEVELS),
}

# -------------------------------------------------------------------------------------------------
# The monitor, the lost instance and the recorders
# -------------------------------------------------------------------------------------------------

class ScriptedMonitor:
    '''
    A stand-in for ``OutcomeMonitor``: epoch e's read returns ``mrrs[e]``, so a resumed run's
    monitor goes on with the same script. It reads no split and writes nothing (P28), and
    ``events`` records each call in order.
    '''

    def __init__(self, mrrs: Sequence[float]):
        self.mrrs = tuple(mrrs)
        self.events: List[Tuple[str, Optional[int]]] = []

    def start(self, *, resumed_epoch: Optional[int]) -> None:
        self.events.append(('start', resumed_epoch))

    def read(self, model, cache, *, training_run: str, seed: int, epoch: int) -> MonitorRead:
        self.events.append(('read', epoch))
        return MonitorRead(epoch=epoch, mrr=self.mrrs[epoch], record={'detail': {'epoch': epoch}})

    def append(self, read: MonitorRead) -> None:
        self.events.append(('append', read.epoch))

class _ScriptedReads:
    '''
    The real monitor with scripted MRRs: each epoch's read is scored, logged and recorded in
    ``monitor_reads.jsonl`` by ``monitor``, but its MRR is ``mrrs[epoch]``, so early stopping ends
    the run where the script says. A logged record holds no MRR, so each line stays consistent.
    '''

    def __init__(self, monitor: OutcomeMonitor, mrrs: Sequence[float]):
        self.monitor = monitor
        self.mrrs = tuple(mrrs)

    def start(self, *, resumed_epoch: Optional[int]) -> None:
        self.monitor.start(resumed_epoch=resumed_epoch)

    def read(self, model, cache, *, training_run: str, seed: int, epoch: int) -> MonitorRead:
        read = self.monitor.read(model, cache, training_run=training_run, seed=seed, epoch=epoch)
        return MonitorRead(epoch=read.epoch, mrr=self.mrrs[epoch], record=read.record)

    def append(self, read: MonitorRead) -> None:
        self.monitor.append(read)

class InstanceLost(RuntimeError):
    '''The instance a run trained on was lost.'''

class _LoseTheInstance(pyl.Callback):
    '''
    Lose the instance while it saves epoch ``epoch``'s first checkpoint.

    By then the module's epoch-end hook has read the epoch and appended its record, and no
    checkpoint of the epoch is on disk: Lightning calls ``on_save_checkpoint`` before it writes
    the file.
    '''

    def __init__(self, epoch: int):
        self.epoch = epoch

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if checkpoint['epoch'] == self.epoch:
            raise InstanceLost(f'the instance was lost saving epoch {self.epoch}')

class _SaveMidEpoch(pyl.Callback):
    '''Save a checkpoint after epoch 0's second step, as the shipped ModelCheckpoint never does.'''

    def __init__(self, path: Path):
        self.path = path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.current_epoch, batch_idx) == (0, 1):
            trainer.save_checkpoint(self.path)

class _StepRecorder(pyl.Callback):
    '''Each optimizer step as it ran: its epoch and index, each group's rate and its anchors.'''

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Read after the step, the rates are the ones it ran at: the warmup sets them just before
        # a step, and the plateau only at an epoch's end
        self.steps.append(
            {
                'epoch': trainer.current_epoch,
                'step': batch_idx,
                'rates': [group['lr'] for group in trainer.optimizers[0].param_groups],
                'codes': batch['codes']['ids'].tolist(),
            }
        )

class _ModuleHooks:
    '''
    The module's hooks as they ran: each step's terms (a flat arm's steps have no radial term),
    each epoch whose end ran with the logit scales at that end, and each code-cache refresh with
    the hook and epoch it ran in.
    '''

    def __init__(self, model: NAICSContrastiveModel):
        self.terms: List[Dict[str, Any]] = []
        self.epoch_ends: List[int] = []
        self.scales: List[Tuple[float, float]] = []
        self.refreshes: List[Tuple[Optional[str], int, CodeCache]] = []
        self._hook: Optional[str] = None
        self._wrap(model, 'on_train_start', 'start')
        self._wrap(model, 'on_train_epoch_end', 'end')
        compute_losses, refresh = model.compute_losses, model.refresh_code_cache

        def recorded_losses(batch):
            losses = compute_losses(batch)
            terms = {
                term: getattr(losses, term).item()
                for term in TERMS if getattr(losses, term) is not None
            }
            self.terms.append({'epoch': model.current_epoch, **terms})
            return losses

        def recorded_refresh(code_rows):
            cache = refresh(code_rows)
            self.refreshes.append((self._hook, model.current_epoch, cache))
            return cache

        model.compute_losses = recorded_losses
        model.refresh_code_cache = recorded_refresh

    def _wrap(self, model: NAICSContrastiveModel, name: str, hook: str) -> None:
        run = getattr(model, name)

        def recorded() -> None:
            if hook == 'end':
                self.epoch_ends.append(model.current_epoch)
                with torch.no_grad():
                    scales = (model.logit_scale_task().item(), model.logit_scale_code().item())
                self.scales.append(scales)
            self._hook = hook
            try:
                run()
            finally:
                self._hook = None

        setattr(model, name, recorded)

    def cache(self, hook: str, epoch: int) -> CodeCache:
        '''The cache the refresh in ``hook`` (``start`` or ``end``) of ``epoch`` built.'''

        caches = [cache for at, when, cache in self.refreshes if (at, when) == (hook, epoch)]
        (cache, ) = caches
        return cache

class _MemoryLogger(Logger):
    '''A logger that keeps every metrics dict Lightning logs, in order.'''

    def __init__(self):
        super().__init__()
        self.logged: List[Dict[str, float]] = []

    @property
    def name(self) -> str:
        return 'memory'

    @property
    def version(self) -> int:
        return 0

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.logged.append(dict(metrics))

def _every_epoch(directory: Path) -> ModelCheckpoint:
    '''
    A checkpoint of every epoch, saved as the run's own are, after the module's epoch-end hook.

    Only runs that are never resumed take it: its state lands in every checkpoint, and a resume
    without it would warn of a missing callback.
    '''

    return ModelCheckpoint(
        dirpath=directory,
        filename='epoch={epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

def _names(directory: Path) -> List[str]:
    return sorted(path.name for path in directory.iterdir())

def _epochs(records: Iterable[Mapping[str, Any]]) -> List[int]:
    return [record['read']['detail']['epoch'] for record in records]

def _earliest_best(mrrs: Sequence[float]) -> int:
    '''The earliest epoch with the highest MRR.'''

    return max(range(len(mrrs)), key=lambda epoch: (mrrs[epoch], -epoch))

# -------------------------------------------------------------------------------------------------
# Runs of the reference bundle
# -------------------------------------------------------------------------------------------------

@dataclass
class _Run:
    '''One fit's pieces, built as train builds them, and what the recorders saw of it.'''

    cfg: Config
    contract: CheckpointContract
    settings: Dict[str, Any]
    model: NAICSContrastiveModel
    datamodule: NAICSDataModule
    trainer: pyl.Trainer
    checkpoint: ModelCheckpoint
    early_stopping: EarlyStopping
    steps: _StepRecorder
    hooks: _ModuleHooks
    checkpoint_dir: Path

    @property
    def last(self) -> Path:
        return self.checkpoint_dir / 'last.ckpt'

    @property
    def plateau(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return self.trainer.lr_scheduler_configs[0].scheduler

    def fit(self, ckpt_path: Optional[Path] = None) -> '_Run':
        '''
        ``train``'s fit, after its P19 guards: an exact resume passes the checkpoint, a fresh run
        None.
        '''

        if ckpt_path is None:
            refuse_a_fresh_start_into_a_used_directory(self.checkpoint_dir)
        else:
            validate_exact_resume(ckpt_path, self.contract)
            saved = read_checkpoint(ckpt_path)
            refuse_a_resume_from_another_directory(saved, self.checkpoint_dir)
            refuse_a_resume_under_other_settings(saved, self.settings, seed=self.cfg.seed)
            refuse_a_resume_of_a_stopped_run(saved, self.cfg.training.early_stopping_patience)
        checkpoint = None if ckpt_path is None else str(ckpt_path)
        self.trainer.fit(self.model, self.datamodule, ckpt_path=checkpoint)
        return self

class _ReferenceRuns:
    '''Builds runs of the reference bundle as ``train`` builds them, every file under tmp_path.'''

    def __init__(self, root: Path, manifest: Path, bundle: ValidatedSupervisionBundle):
        self.root = root
        self.manifest = manifest
        self.bundle = bundle

    def config(self, name: str, overrides: Mapping[str, Any]) -> Config:
        '''
        The shipped defaults, with the reference bundle at 4 queries a step, the run's seed, one log
        line a step and every directory under ``<tmp_path>/<name>``. ``overrides`` are dotted keys,
        as on train's command line.
        '''

        parameters = self.bundle.manifest.generation_parameters
        settings = {
            'seed': SEED,
            'supervision.manifest_path': str(self.manifest),
            'data_loader.queries_per_step': REFERENCE_QUERIES_PER_STEP,
            'data_loader.streaming.descriptions_parquet': parameters['descriptions_parquet'],
            'training.trainer.log_every_n_steps': 1,
            'dirs.checkpoint_dir': str(self.root / name / 'checkpoints'),
            'dirs.output_dir': str(self.root / name / 'outputs'),
            **overrides,
        }
        return Config().override(settings)

    @staticmethod
    def checkpoint_dir(cfg: Config) -> Path:
        return Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name

    def monitor(self, cfg: Config, log: Path) -> OutcomeMonitor:
        '''The real monitor, as train builds it, with its selection log at ``log`` (P28).'''

        return training.build_monitor_from_config(
            cfg, self.bundle, self.checkpoint_dir(cfg), selection_log=log
        )

    def build(
        self,
        cfg: Config,
        monitor: Any,
        *,
        callbacks: Sequence[pyl.Callback] = (),
        logger: Optional[Logger] = None,
        precision: str = '32-true',
    ) -> _Run:
        '''
        The datamodule, model and Trainer, as train builds them on a CPU host, after its seed;
        ``logger`` replaces the TensorBoard logger, and ``precision`` the host's.
        '''

        pyl.seed_everything(cfg.seed, verbose=False)
        hardware = HardwareInfo(accelerator='cpu', precision=precision, num_devices=1)
        settings = run_settings(cfg, accelerator=hardware.accelerator, precision=hardware.precision)
        contract = training.runtime_contract_for(cfg, self.bundle)
        datamodule = training.build_datamodule_from_config(cfg, self.bundle)
        model = training.build_model_from_config(
            cfg, contract, self.bundle, run_settings=settings, monitor=monitor
        )
        hooks = _ModuleHooks(model)
        checkpoint_dir = self.checkpoint_dir(cfg)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        steps = _StepRecorder()
        trainer, checkpoint, early_stopping = create_trainer(
            cfg, hardware, checkpoint_dir, callbacks=[steps, *callbacks], tb_logger=logger
        )
        return _Run(
            cfg,
            contract,
            settings,
            model,
            datamodule,
            trainer,
            checkpoint,
            early_stopping,
            steps,
            hooks,
            checkpoint_dir,
        )

@pytest.fixture
def reference_runs(
    tmp_path, monkeypatch, tiny_backbone, reference_manifest, reference_bundle
) -> _ReferenceRuns:
    '''
    Runs of the reference bundle on the tiny backbone.

    ``code_token_config``'s cache is ``./data/token_cache/token_cache.pt``, which no test may
    write, so it moves under tmp_path, for the datamodule and the monitor alike. Every segment of a
    test's run reads that one cache, as a resume on the same instance would.
    '''

    token_cache = tmp_path / 'token_cache' / 'token_cache.pt'

    def moved(cfg: Config) -> TokenizationConfig:
        return code_token_config(cfg).model_copy(update={'output_path': str(token_cache)})

    monkeypatch.setattr(training, 'code_token_config', moved)
    return _ReferenceRuns(tmp_path, reference_manifest, reference_bundle)

# -------------------------------------------------------------------------------------------------
# The monitor's reads (spec 4.4; spec 6, Monitor and records)
# -------------------------------------------------------------------------------------------------

def test_each_epochs_read_is_logged_as_an_outcome_validation_read_and_recorded(
    reference_runs, tmp_path
):
    '''
    One read an epoch, logged as an outcome validation read that names the training run, the seed,
    the epoch and the table of that epoch's refreshed cache. ``monitor_reads.jsonl`` holds each
    read as logged with its MRR, the one value logged as ``val/outcome_mrr``, which ModelCheckpoint
    keeps in float64 (P18).
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 3})
    log = tmp_path / 'logs' / 'selection_log.jsonl'
    monitor = reference_runs.monitor(cfg, log)
    memory = _MemoryLogger()

    run = reference_runs.build(cfg, monitor, logger=memory).fit()

    reads = SelectionLog(log).records()
    records = read_monitor_records(run.checkpoint_dir / MONITOR_RECORDS)
    assert _epochs(records) == [0, 1, 2]
    assert [record['read'] for record in records] == reads
    training_run = run.model.training_run
    for epoch, read in enumerate(reads):
        assert (read['event'], read['panel'], read['split']) == ('read', 'outcome', 'validation')
        assert read['purpose'] == monitor.purpose
        assert read['fingerprint'] == monitor.panel.fingerprint
        cache = run.hooks.cache('end', epoch)
        named = {name: read['detail'][name] for name in ('training_run', 'seed', 'epoch', 'table')}
        assert named == {
            'training_run': training_run,
            'seed': SEED,
            'epoch': epoch,
            'table': matrix_fingerprint(cache.codes, cache.tangent.numpy()),
        }
    # One value: the MRR logged each epoch is the record's, and it stays float64
    mrrs = [record['mrr'] for record in records]
    assert [metrics[OUTCOME_MRR] for metrics in memory.logged if OUTCOME_MRR in metrics] == mrrs
    logged = run.trainer.callback_metrics[OUTCOME_MRR]
    assert logged.dtype == torch.float64 and logged.item() == mrrs[-1]
    kept = _earliest_best(mrrs)
    assert Path(run.checkpoint.best_model_path).name == f'epoch={kept:03d}.ckpt'
    assert run.checkpoint.best_model_score.dtype == torch.float64
    assert run.checkpoint.best_model_score.item() == mrrs[kept]
    # The checkpoints name the run the reads name
    last = read_checkpoint(run.last)
    assert (last['epoch'], last['training_run']) == (2, training_run)

@pytest.mark.parametrize('precision', ['32-true', 'bf16-mixed'])
def test_an_epochs_monitor_mrr_is_the_read_of_that_epochs_exported_checkpoint(
    reference_runs, reference_bundle, tmp_path, precision
):
    '''
    Spec 4.4's agreement, on the CPU and exactly: each epoch's monitor MRR is
    ``read_outcome_validation``'s on the checkpoint saved at that epoch's end, through the table
    exported from it, and the epoch's record names that table. Under ``bf16-mixed`` only the
    training steps autocast: the refresh, the monitor and the export encode in float32 (spec 4.2).
    ``train`` trains a CPU run at ``32-true`` (P31); CPU ``bf16-mixed`` stands in for CUDA's.
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 2})
    monitor = reference_runs.monitor(cfg, tmp_path / 'logs' / 'monitor_log.jsonl')
    every_epoch = tmp_path / 'every_epoch'

    run = reference_runs.build(
        cfg, monitor, callbacks=[_every_epoch(every_epoch)], precision=precision
    ).fit()

    records = read_monitor_records(run.checkpoint_dir / MONITOR_RECORDS)
    assert _epochs(records) == [0, 1]
    # Training moved the codes, so each epoch's read is of its own table
    assert records[0]['read']['detail']['table'] != records[1]['read']['detail']['table']
    panel = OutcomePanel.from_bundle(reference_bundle, tmp_path / 'logs' / 'arm_log.jsonl')
    token_config = run.datamodule.token_config
    for epoch, record in enumerate(records):
        checkpoint = every_epoch / f'epoch={epoch:03d}.ckpt'
        table = export_code_table(
            checkpoint,
            reference_bundle,
            token_config,
            tmp_path / 'tables' / f'epoch={epoch:03d}.parquet',
        )
        arm = ArmEncoder.from_files(checkpoint, table, reference_bundle, token_config)
        exported = read_outcome_validation(arm, panel, 'the monitor read of the exported epoch')
        assert exported.summary['mrr'] == record['mrr'], epoch
        assert arm.table_fingerprint == record['read']['detail']['table'], epoch

@pytest.mark.parametrize('precision', ['32-true', 'bf16-mixed'])
@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arm_trains_without_the_radial_term_and_reads_by_its_own_distance(
    reference_runs, reference_bundle, tmp_path, geometry, precision
):
    '''
    Req 12 through the Trainer: a flat arm trains without the radial term, records no
    ``loss/radial``, and each epoch's monitor read decodes by its own distance. As in the
    hyperbolic arm, that read's MRR is the read of the table exported from the epoch's checkpoint,
    exactly, on the CPU (spec 4.4).
    '''

    cfg = reference_runs.config(
        'run', {
            'training.trainer.max_epochs': 2,
            'model.geometry': geometry
        }
    )
    monitor = reference_runs.monitor(cfg, tmp_path / 'logs' / 'monitor_log.jsonl')
    every_epoch = tmp_path / 'every_epoch'

    run = reference_runs.build(
        cfg, monitor, callbacks=[_every_epoch(every_epoch)], precision=precision
    ).fit()

    assert run.model.encoder.head.geometry == geometry
    assert len(run.hooks.terms) == 2 * STEPS
    assert all('radial' not in terms for terms in run.hooks.terms)
    summary = read_epoch_summary(run.checkpoint_dir / EPOCH_SUMMARY)
    assert [row['epoch'] for row in summary] == [0, 1]
    assert all('loss/task' in row and 'loss/radial' not in row for row in summary)
    records = read_monitor_records(run.checkpoint_dir / MONITOR_RECORDS)
    distance = GEOMETRY_DISTANCES[geometry]
    assert [record['read']['detail']['distance'] for record in records] == [distance] * 2
    panel = OutcomePanel.from_bundle(reference_bundle, tmp_path / 'logs' / 'arm_log.jsonl')
    token_config = run.datamodule.token_config
    for epoch, record in enumerate(records):
        checkpoint = every_epoch / f'epoch={epoch:03d}.ckpt'
        table = export_code_table(
            checkpoint,
            reference_bundle,
            token_config,
            tmp_path / 'tables' / f'epoch={epoch:03d}.parquet',
        )
        arm = ArmEncoder.from_files(checkpoint, table, reference_bundle, token_config)
        exported = read_outcome_validation(arm, panel, 'the monitor read of the exported epoch')
        assert arm.distance == distance
        assert exported.summary['mrr'] == record['mrr'], epoch
        assert arm.table_fingerprint == record['read']['detail']['table'], epoch

# -------------------------------------------------------------------------------------------------
# What the monitor drives: the kept epoch, early stopping and the schedule (spec 4.4; P16, P17)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('mrrs', 'kept'),
    [
        ((0.5, 0.7, 0.7, 0.6), 1),
        ((0.5, 0.7, 0.7 + 1e-9, 0.6), 2),
        ((0.5, 0.4, 0.6, 0.6), 2),
    ],
    ids=['a-tie-keeps-the-earlier', 'a-float64-gain-is-a-gain', 'a-later-best-replaces'],
)
def test_the_kept_checkpoint_is_the_earliest_epoch_with_the_highest_mrr(reference_runs, mrrs, kept):
    '''
    P17: one kept epoch, the earliest with the highest ``val/outcome_mrr``, and ``last.ckpt``. A
    later epoch replaces it only by beating it, by any float64 margin: the MRR is logged in float64
    (P18), where a float32 log would tie 0.7 and 0.7 + 1e-9.
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': len(mrrs)})

    run = reference_runs.build(cfg, ScriptedMonitor(mrrs)).fit()

    assert _names(run.checkpoint_dir) == [
        f'epoch={kept:03d}.ckpt', 'epoch_summary.jsonl', 'last.ckpt'
    ]
    assert read_checkpoint(run.checkpoint_dir / f'epoch={kept:03d}.ckpt')['epoch'] == kept
    assert read_checkpoint(run.last)['epoch'] == len(mrrs) - 1
    assert Path(run.checkpoint.best_model_path).name == f'epoch={kept:03d}.ckpt'
    assert run.checkpoint.best_model_score.item() == mrrs[kept]

def test_early_stopping_counts_a_tie_against_its_patience(reference_runs):
    '''
    Spec 4.4: early stopping on ``val/outcome_mrr``, mode max. An equal MRR is no gain, so at
    patience 2 the run stops once epochs 2 and 3 fail to beat epoch 1, and never reads epoch 4.
    '''

    cfg = reference_runs.config(
        'run',
        {
            'training.trainer.max_epochs': 6,
            'training.early_stopping_patience': 2,
        },
    )
    monitor = ScriptedMonitor((0.5, 0.7, 0.7, 0.6, 0.9, 0.9))

    run = reference_runs.build(cfg, monitor).fit()

    assert [epoch for event, epoch in monitor.events if event == 'read'] == [0, 1, 2, 3]
    assert run.early_stopping.stopped_epoch == 3
    assert _names(run.checkpoint_dir) == ['epoch=001.ckpt', 'epoch_summary.jsonl', 'last.ckpt']
    assert read_checkpoint(run.last)['epoch'] == 3

def test_the_warmup_ramps_each_step_and_the_plateau_cuts_once_its_patience_is_spent(
    reference_runs, tmp_path
):
    '''
    P16 under the Trainer's own step count (S = 3) and global step. For W = 2 warmup epochs, step t
    runs at base · (t + 1) / (W · S); after the warmup only the plateau moves the rate. The module
    steps the plateau once an epoch with the MRR, before the epoch's checkpoints are saved, so each
    holds that epoch's step; Lightning's own step of it changes nothing. At patience 1 the plateau
    cuts the rate at the second epoch in a row without a gain, a tie included.
    '''

    base = 1e-3
    cfg = reference_runs.config(
        'run',
        {
            'training.trainer.max_epochs': 7,
            'training.learning_rate': base,
            'training.warmup_epochs': 2,
            'training.lr_plateau_factor': 0.5,
            'training.lr_plateau_patience': 1,
            'training.early_stopping_patience': 7,
        },
    )
    # Epochs 1 and 2 do not beat epoch 0, so the plateau cuts at epoch 2's end, after the warmup;
    # epoch 3 beats it, and epochs 4 (a tie) and 5 do not, so it cuts again at epoch 5's end
    mrrs = (0.5, 0.4, 0.5, 0.6, 0.6, 0.55, 0.7)
    every_epoch = tmp_path / 'every_epoch'
    callbacks = [_every_epoch(every_epoch)]

    run = reference_runs.build(cfg, ScriptedMonitor(mrrs), callbacks=callbacks).fit()

    steps = run.steps.steps
    ran = [(step['epoch'], step['step']) for step in steps]
    assert ran == [(epoch, step) for epoch in range(7) for step in range(STEPS)]
    # Steps 0 to 5 are the warmup's; epoch 2 runs at the base rate, until the two cuts
    warmup = [base * (step + 1) / 6 for step in range(6)]
    after = [base] * 3 + [base / 2] * 9 + [base / 4] * 3
    expected = [[pytest.approx(rate, rel=1e-12)] * 2 for rate in warmup + after]
    assert [step['rates'] for step in steps] == expected
    # At each epoch's end: the plateau's best and bad epochs, and the rate after its step
    ends = [
        (0.5, 0, base / 2),  # The warmup's third step ran at base · 3 / 6
        (0.5, 1, base),
        (0.5, 0, base / 2),  # Cut: a second epoch without a gain
        (0.6, 0, base / 2),
        (0.6, 1, base / 2),  # A tie is no gain
        (0.6, 0, base / 4),  # Cut again
        (0.7, 0, base / 4),
    ]
    for epoch, (best, bad, rate) in enumerate(ends):
        saved = read_checkpoint(every_epoch / f'epoch={epoch:03d}.ckpt')
        (plateau, ) = saved['lr_schedulers']
        # Stepped once an epoch, by the module alone
        assert (plateau['last_epoch'], plateau['best'], plateau['num_bad_epochs']) == (
            epoch + 1,
            best,
            bad,
        ), epoch
        (optimizer, ) = saved['optimizer_states']
        rates = [group['lr'] for group in optimizer['param_groups']]
        assert rates == [pytest.approx(rate, rel=1e-12)] * 2, epoch
    assert run.plateau.last_epoch == len(mrrs)
    assert [group['lr'] for group in run.trainer.optimizers[0].param_groups] == [base / 4] * 2

def test_the_health_logs_reach_the_logger_once_an_epoch_as_plain_means(reference_runs):
    '''
    P20 through the Trainer's logger. Each epoch logs, once and beside ``val/outcome_mrr``, its
    terms' means, the two logit scales, and r's mean and SD at each level of its end-of-epoch
    cache. A term's mean is the plain mean of the epoch's steps, which differ in size (6, 6 and 5
    anchors; 4, 4 and 3 queries), so a mean weighted by either size would differ from it. Lightning
    logs the values in float32.
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 2})
    mrrs = (0.5, 0.6)
    memory = _MemoryLogger()

    run = reference_runs.build(cfg, ScriptedMonitor(mrrs), logger=memory).fit()

    epochs = [metrics for metrics in memory.logged if 'loss/total' in metrics]
    assert [metrics['epoch'] for metrics in epochs] == [0, 1]
    levels = run.model.code_levels.cpu()
    for epoch, metrics in enumerate(epochs):
        assert set(metrics) == HEALTH_KEYS | {OUTCOME_MRR, 'epoch'}
        assert metrics[OUTCOME_MRR] == mrrs[epoch]
        steps = [terms for terms in run.hooks.terms if terms['epoch'] == epoch]
        assert len(steps) == STEPS
        for term in TERMS:
            mean = statistics.fmean(terms[term] for terms in steps)
            assert metrics[f'loss/{term}'] == pytest.approx(mean, rel=1e-6), (epoch, term)
        # The total the progress bar shows each step is the step's total
        totals = [
            logged['loss/step'] for logged in memory.logged
            if 'loss/step' in logged and logged['epoch'] == epoch
        ]
        assert totals == [pytest.approx(terms['total'], rel=1e-6) for terms in steps]
        task, code = run.hooks.scales[epoch]
        assert metrics['logit_scale/task'] == pytest.approx(task, rel=1e-6)
        assert metrics['logit_scale/code_code'] == pytest.approx(code, rel=1e-6)
        radius = run.hooks.cache('end', epoch).radius.cpu().to(torch.float64)
        for level in LEVELS:
            at_level = radius[levels == level]
            mean, sd = at_level.mean().item(), at_level.std(correction=0).item()
            assert metrics[f'radius/mean/level_{level}'] == pytest.approx(mean, rel=1e-6), level
            assert metrics[f'radius/sd/level_{level}'] == pytest.approx(sd, rel=1e-5, abs=1e-6)

# -------------------------------------------------------------------------------------------------
# Exact resume from the last.ckpt written during the fit (spec 4.3, 4.4; spec 6; P27)
# -------------------------------------------------------------------------------------------------

# A schedule a resume could break: a 2-epoch warmup, a plateau at patience 1 and early stopping at
# patience 3, over 8 epochs at most
RESUME_SCHEDULE = {
    'training.trainer.max_epochs': 8,
    'training.learning_rate': 1e-3,
    'training.warmup_epochs': 2,
    'training.lr_plateau_factor': 0.5,
    'training.lr_plateau_patience': 1,
    'training.early_stopping_patience': 3,
}
# Epoch 0 is kept until epoch 3 beats it; the plateau cuts at the ends of epochs 2 and 5; epochs 4,
# 5 and 6 do not beat epoch 3, so the run stops after epoch 6 and never reads epoch 7
RESUME_MRRS = (0.5, 0.4, 0.5, 0.6, 0.6, 0.55, 0.5, 0.7)

def _reads_and_appends(epochs: Iterable[int]) -> List[Tuple[str, int]]:
    return [event for epoch in epochs for event in (('read', epoch), ('append', epoch))]

@pytest.mark.parametrize('lost', [1, 5], ids=['in-the-warmup', 'while-early-stopping-waits'])
def test_an_exact_resume_follows_the_uninterrupted_run(reference_runs, lost):
    '''
    Exact resume under spec 4.4's schedule (P16, P17, P27). The instance is lost saving epoch
    ``lost``'s checkpoint, and the run resumes from the epoch before. It then goes on as if never
    interrupted: every step at the same rate and on the same anchors, and the same plateau, early
    stop and kept epoch.

    - Lost in the warmup: the resumed steps take their rates from the restored global step, and
      the restored kept epoch, epoch 0, gives way to a later best.
    - Lost two epochs after the last gain: the restored early stopping and plateau count on from
      their waits, so the plateau cuts at epoch 5's end and the run stops after epoch 6.

    No epoch-end hook replays: the resumed run reads, steps the plateau and appends once for each
    epoch after the restored one.
    '''

    straight = reference_runs.build(
        reference_runs.config('straight', RESUME_SCHEDULE), ScriptedMonitor(RESUME_MRRS)
    ).fit()
    cfg = reference_runs.config('resumed', RESUME_SCHEDULE)
    interrupted = reference_runs.build(
        cfg, ScriptedMonitor(RESUME_MRRS), callbacks=[_LoseTheInstance(lost)]
    )
    with pytest.raises(InstanceLost):
        interrupted.fit()
    restored = lost - 1
    assert read_checkpoint(interrupted.last)['epoch'] == restored
    monitor = ScriptedMonitor(RESUME_MRRS)

    resumed = reference_runs.build(cfg, monitor).fit(ckpt_path=interrupted.last)

    # The uninterrupted run: stopped after epoch 6, with epoch 3 kept
    assert straight.hooks.epoch_ends == list(range(7))
    assert straight.early_stopping.stopped_epoch == 6
    assert _names(straight.checkpoint_dir) == ['epoch=003.ckpt', 'epoch_summary.jsonl', 'last.ckpt']
    # No replay: one read, plateau step and append for each epoch after the restored one
    assert resumed.hooks.epoch_ends == list(range(lost, 7))
    assert monitor.events == [('start', restored), *_reads_and_appends(range(lost, 7))]
    assert resumed.plateau.last_epoch == straight.plateau.last_epoch == 7
    # Every step at the same rate and on the same anchors, the first resumed epoch's included
    kept_steps = [step for step in interrupted.steps.steps if step['epoch'] <= restored]
    assert kept_steps + resumed.steps.steps == straight.steps.steps
    # The same plateau, early stop and kept epoch
    assert resumed.plateau.state_dict() == straight.plateau.state_dict()
    for name in ('wait_count', 'stopped_epoch', 'patience'):
        assert getattr(resumed.early_stopping, name) == getattr(straight.early_stopping, name)
    assert resumed.early_stopping.best_score.item() == straight.early_stopping.best_score.item()
    assert _names(resumed.checkpoint_dir) == ['epoch=003.ckpt', 'epoch_summary.jsonl', 'last.ckpt']
    assert Path(resumed.checkpoint.best_model_path).name == 'epoch=003.ckpt'
    assert resumed.checkpoint.best_model_score.item() == 0.6
    resumed_last, straight_last = read_checkpoint(resumed.last), read_checkpoint(straight.last)
    for key in ('epoch', 'global_step'):
        assert resumed_last[key] == straight_last[key], key
    assert (resumed_last['epoch'], resumed_last['global_step']) == (6, 7 * STEPS)
    # One training run: the resumed run keeps the interrupted run's id
    assert resumed.model.training_run == interrupted.model.training_run
    assert resumed_last['training_run'] == interrupted.model.training_run

def test_the_first_resumed_epoch_reads_its_steps_only_once_its_epoch_is_set(
    reference_runs, reference_bundle, minilm_tokenizer, monkeypatch
):
    '''
    P27, by the order of events on the step dataset. Lightning takes no batch ahead from a loader
    whose length it knows, and with ``num_workers=0`` every step is read in the main process, so
    the epoch-start callback sets each epoch before its first step is read. That holds on a resume
    too, where Lightning asks for the loader while the trainer's epoch is still the restored one.
    Nothing else sets an epoch, so each resumed step reads its (seed, epoch) permutations.
    '''

    events: List[Tuple[Any, ...]] = []
    set_epoch, read = two_stream.StepDataset.set_epoch, two_stream.StepDataset.__getitem__

    def recorded_set_epoch(dataset, epoch):
        events.append(('set_epoch', epoch))
        return set_epoch(dataset, epoch)

    def recorded_read(dataset, step):
        events.append(('read', dataset.epoch, step))
        return read(dataset, step)

    monkeypatch.setattr(two_stream.StepDataset, 'set_epoch', recorded_set_epoch)
    monkeypatch.setattr(two_stream.StepDataset, '__getitem__', recorded_read)

    def epoch_events(epochs: Iterable[int]) -> List[Tuple[Any, ...]]:
        return [
            event for epoch in epochs
            for event in [('set_epoch', epoch), *(('read', epoch, step) for step in range(STEPS))]
        ]

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 4})
    mrrs = (0.5, 0.6, 0.7, 0.8)
    interrupted = reference_runs.build(cfg, ScriptedMonitor(mrrs), callbacks=[_LoseTheInstance(2)])
    with pytest.raises(InstanceLost):
        interrupted.fit()
    assert events == epoch_events(range(3))
    events.clear()

    monitor = ScriptedMonitor(mrrs)
    resumed = reference_runs.build(cfg, monitor).fit(ckpt_path=interrupted.last)

    assert events == epoch_events(range(2, 4))
    # No epoch-end hook replayed
    assert resumed.hooks.epoch_ends == [2, 3]
    assert monitor.events == [('start', 1), *_reads_and_appends(range(2, 4))]
    expected = reference_step_dataset(
        reference_bundle, list(resumed.datamodule.code_rows), minilm_tokenizer, seed=SEED
    )
    assert [step['epoch'] for step in resumed.steps.steps] == [2] * STEPS + [3] * STEPS
    for step in resumed.steps.steps:
        expected.set_epoch(step['epoch'])
        assert step['codes'] == expected[step['step']]['codes']['ids'].tolist()

def test_an_exact_resume_keeps_the_run_and_its_records_dropping_the_read_past_its_epoch(
    reference_runs, tmp_path
):
    '''
    Spec 4.4 and spec 6, Monitor and records. The instance lost saving epoch 2's checkpoint had read
    and recorded epoch 2, so the resumed run, restored to epoch 1, drops that record and reads
    epoch 2 again. The records file then holds each epoch of the surviving run once, epochs 0 and
    1 as first written, and the dropped read stays only in the lost instance's selection log.
    Every read names the one training run, which the last checkpoint names too.
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 4})
    lost_log = tmp_path / 'logs' / 'lost_instance.jsonl'
    resumed_log = tmp_path / 'logs' / 'resumed_instance.jsonl'
    interrupted = reference_runs.build(
        cfg, reference_runs.monitor(cfg, lost_log), callbacks=[_LoseTheInstance(2)]
    )
    with pytest.raises(InstanceLost):
        interrupted.fit()
    records_path = interrupted.checkpoint_dir / MONITOR_RECORDS
    written = records_path.read_text(encoding='utf-8').splitlines()
    assert _epochs(read_monitor_records(records_path)) == [0, 1, 2]
    assert read_checkpoint(interrupted.last)['epoch'] == 1
    training_run = interrupted.model.training_run

    resumed = reference_runs.build(cfg, reference_runs.monitor(cfg, resumed_log)).fit(
        ckpt_path=interrupted.last
    )

    records = read_monitor_records(records_path)
    assert _epochs(records) == [0, 1, 2, 3]
    assert records_path.read_text(encoding='utf-8').splitlines()[:2] == written[:2]
    # The resumed run read epochs 2 and 3 once each: no epoch-end hook replayed
    assert resumed.hooks.epoch_ends == [2, 3]
    assert [record['read'] for record in records[2:]] == SelectionLog(resumed_log).records()
    # The dropped read is the lost instance's alone
    lost_reads = SelectionLog(lost_log).records()
    assert [read['detail']['epoch'] for read in lost_reads] == [0, 1, 2]
    assert [record['read'] for record in records[:2]] == lost_reads[:2]
    assert lost_reads[2] not in [record['read'] for record in records]
    # One training run
    assert resumed.model.training_run == training_run
    assert {record['read']['detail']['training_run'] for record in records} == {training_run}
    assert read_checkpoint(resumed.last)['training_run'] == training_run

def test_an_exact_resume_rebuilds_the_cache_its_checkpoint_was_read_on(reference_runs, tmp_path):
    '''
    Spec 6, Cache. The cache is never saved, so a resume rebuilds it at fit start from the restored
    weights. It is the cache the restored epoch ended with, bit for bit: the one that epoch's read
    was scored on, whose table the epoch's record names.
    '''

    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 3})
    interrupted = reference_runs.build(
        cfg,
        reference_runs.monitor(cfg, tmp_path / 'logs' / 'lost_instance.jsonl'),
        callbacks=[_LoseTheInstance(2)],
    )
    with pytest.raises(InstanceLost):
        interrupted.fit()
    # The instance was lost at epoch 2's end; its last checkpoint is epoch 1's
    ended = interrupted.hooks.cache('end', 1)

    resumed = reference_runs.build(
        cfg, reference_runs.monitor(cfg, tmp_path / 'logs' / 'resumed_instance.jsonl')
    ).fit(ckpt_path=interrupted.last)

    # No epoch-end hook replayed: the resumed run refreshed at its start and at epoch 2's end
    assert resumed.hooks.epoch_ends == [2]
    refreshes = [(hook, epoch) for hook, epoch, _ in resumed.hooks.refreshes]
    assert refreshes == [('start', 2), ('end', 2)]
    rebuilt = resumed.hooks.cache('start', 2)
    assert rebuilt.codes == ended.codes
    for name in ('radius', 'direction', 'tangent'):
        assert torch.equal(getattr(rebuilt, name), getattr(ended, name)), name
    records = read_monitor_records(interrupted.checkpoint_dir / MONITOR_RECORDS)
    table = matrix_fingerprint(rebuilt.codes, rebuilt.tangent.numpy())
    assert records[1]['read']['detail']['table'] == table
    # The lost epoch had moved the codes on: the rebuilt cache is not its
    assert not torch.equal(interrupted.hooks.cache('end', 2).tangent, rebuilt.tangent)

def test_a_resume_from_a_mid_epoch_checkpoint_is_refused_at_its_first_step(
    reference_runs, tmp_path
):
    '''
    The one resume here from a checkpoint ModelCheckpoint did not write: it saves only at an epoch's
    end, so this pins what happens to one saved mid-epoch elsewhere. Lightning restarts mid-epoch
    without ``on_train_epoch_start``, so no epoch is set, and the step dataset refuses the first
    step rather than read an epoch it was never given (P27).
    '''

    mid_epoch = tmp_path / 'mid_epoch.ckpt'
    cfg = reference_runs.config('run', {'training.trainer.max_epochs': 1})
    reference_runs.build(cfg, ScriptedMonitor((0.5, )), callbacks=[_SaveMidEpoch(mid_epoch)]).fit()
    resumed = reference_runs.build(cfg, ScriptedMonitor((0.5, )))

    with pytest.raises(RuntimeError, match='StepDataset has no epoch: call set_epoch first'):
        resumed.fit(ckpt_path=mid_epoch)
    assert resumed.steps.steps == []
    assert resumed.hooks.epoch_ends == []

def test_a_resume_of_a_run_early_stopping_ended_is_refused_leaving_its_files_as_they_were(
    reference_runs, tmp_path
):
    '''
    P19's stopped-run guard, on the checkpoint a fit wrote. At patience 2 early stopping ends the
    run at epoch 3, and the ``last.ckpt`` saved at that epoch's end records the stop: early
    stopping checks before ModelCheckpoint saves, which Lightning runs last. Lightning restores
    that state on a resume but not the trainer's stop, so a resumed run would train epochs 4 and
    5, keep epoch 4 and append their records. The guard refuses the resume before it fits: the
    checkpoint directory, ``monitor_reads.jsonl`` included, stays byte for byte as the run left
    it, and the resumed segment reads nothing.
    '''

    cfg = reference_runs.config(
        'run',
        {
            'training.trainer.max_epochs': 6,
            'training.early_stopping_patience': 2,
        },
    )
    # Epochs 2 (a tie) and 3 do not beat epoch 1, so early stopping ends the run at epoch 3;
    # epoch 4 would beat it
    mrrs = (0.5, 0.7, 0.7, 0.6, 0.9, 0.9)
    monitor = _ScriptedReads(reference_runs.monitor(cfg, tmp_path / 'logs' / 'run.jsonl'), mrrs)

    run = reference_runs.build(cfg, monitor).fit()

    assert run.early_stopping.stopped_epoch == 3
    # The stop is in the last checkpoint, which the epoch that stopped the run saved
    last = read_checkpoint(run.last)
    stopper = last['callbacks'][outcome_early_stopping(2).state_key]
    assert (last['epoch'], stopper['stopped_epoch'], stopper['wait_count']) == (3, 3, 2)
    records = read_monitor_records(run.checkpoint_dir / MONITOR_RECORDS)
    assert _epochs(records) == [0, 1, 2, 3]
    assert [record['mrr'] for record in records] == list(mrrs[:4])
    files = {path.name: path.read_bytes() for path in run.checkpoint_dir.iterdir()}
    assert sorted(files) == ['epoch=001.ckpt', 'epoch_summary.jsonl', 'last.ckpt', MONITOR_RECORDS]
    resumed_log = tmp_path / 'logs' / 'resumed.jsonl'
    resumed = reference_runs.build(
        cfg, _ScriptedReads(reference_runs.monitor(cfg, resumed_log), mrrs)
    )

    with pytest.raises(ValueError, match='early stopping ended the run at epoch 3'):
        resumed.fit(ckpt_path=run.last)

    assert {path.name: path.read_bytes() for path in run.checkpoint_dir.iterdir()} == files
    assert resumed.steps.steps == []
    assert resumed.hooks.epoch_ends == []
    assert not resumed_log.exists()

def test_each_epoch_writes_the_exact_health_values_and_mrr_to_the_summary(reference_runs):
    import importlib
    module = importlib.import_module('naics_embedder.text_model.epoch_summary')
    cfg = reference_runs.config('summary-run', {'training.trainer.max_epochs': 2})
    mrrs = (0.123456789012345, 0.6)
    run = reference_runs.build(cfg, ScriptedMonitor(mrrs)).fit()
    rows = module.read_epoch_summary(run.checkpoint_dir / module.EPOCH_SUMMARY)
    assert [row['epoch'] for row in rows] == [0, 1]
    for epoch, row in enumerate(rows):
        assert set(row) == HEALTH_KEYS | {'epoch', 'mrr'}
        assert row['mrr'] == mrrs[epoch]
        steps = [terms for terms in run.hooks.terms if terms['epoch'] == epoch]
        for term in TERMS:
            assert row[f'loss/{term}'] == statistics.fmean(step[term] for step in steps)
        radius = run.hooks.cache('end', epoch).radius.cpu().to(torch.float64)
        for level in LEVELS:
            values = radius[run.model.code_levels.cpu() == level]
            assert row[f'radius/mean/level_{level}'] == values.mean().item()
            assert row[f'radius/sd/level_{level}'] == values.std(correction=0).item()

def test_an_exact_resume_prunes_the_interrupted_epochs_summary_and_continues(reference_runs):
    import importlib
    module = importlib.import_module('naics_embedder.text_model.epoch_summary')
    cfg = reference_runs.config('summary-resume', {'training.trainer.max_epochs': 4})
    mrrs = (0.3, 0.4, 0.5, 0.6)
    interrupted = reference_runs.build(cfg, ScriptedMonitor(mrrs), callbacks=[_LoseTheInstance(2)])
    with pytest.raises(InstanceLost):
        interrupted.fit()
    path = interrupted.checkpoint_dir / module.EPOCH_SUMMARY
    before = path.read_text().splitlines()
    assert [row['epoch'] for row in module.read_epoch_summary(path)] == [0, 1, 2]
    resumed = reference_runs.build(cfg, ScriptedMonitor(mrrs)).fit(ckpt_path=interrupted.last)
    rows = module.read_epoch_summary(path)
    assert [row['epoch'] for row in rows] == [0, 1, 2, 3]
    assert [row['mrr'] for row in rows] == list(mrrs)
    assert path.read_text().splitlines()[:2] == before[:2]
    assert resumed.hooks.epoch_ends == [2, 3]
