# -------------------------------------------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------------------------------------------
'''
Training orchestration utilities for NAICS Embedder.

This module provides helper functions to simplify training setup by extracting
common operations into reusable components. These utilities handle hardware
detection, configuration parsing, checkpoint management, and trainer creation.

Functions:
    detect_hardware: Detect the accelerator and the precision the trainer runs at.
    get_gpu_memory_info: Query current GPU memory usage.
    parse_config_overrides: Parse and validate command-line config overrides.
    effective_precision: The precision a run trains at on an accelerator (P31).
    run_settings: A run's free settings, epoch budget, fusion, accumulation and clipping (P21).
    resolve_checkpoint: Resolve checkpoint path from user input.
    read_checkpoint: A Lightning checkpoint's contents, on the CPU.
    refuse_a_fresh_start_into_a_used_directory: P19's guard on a fresh run's directory.
    refuse_a_resume_from_another_directory: P19's guard on a resume's directory.
    refuse_a_resume_under_other_settings: P19's guard on a resume's settings and seed.
    refuse_a_resume_of_a_stopped_run: P19's guard on a resume of a run early stopping ended.
    outcome_checkpoint: The ModelCheckpoint the monitor's MRR drives (P17).
    outcome_early_stopping: The EarlyStopping the monitor's MRR drives (P17).
    create_trainer: Create a configured PyTorch Lightning Trainer.
    TrainingResult: Structured result from a training run.
'''

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pytorch_lightning as pyl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from naics_embedder.text_model.dataloader.datamodule import TrainDatasetEpochCallback
from naics_embedder.text_model.mixins import OUTCOME_MRR
from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import Config, parse_override_value

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    '''
    Hardware configuration detected for training.

    Attributes:
        accelerator: The accelerator type (cuda, mps, cpu).
        precision: The precision the trainer runs at: the configured CUDA precision on CUDA,
            32-true elsewhere.
        num_devices: Number of available devices.
        gpu_memory: Optional GPU memory information dictionary.
    '''

    accelerator: str
    precision: str
    num_devices: int
    gpu_memory: Optional[Dict[str, float]] = None

@dataclass
class CheckpointInfo:
    '''
    Resolved checkpoint information.

    Attributes:
        path: Resolved filesystem path to the checkpoint, or None.
        is_same_stage: Whether checkpoint is from the same experiment stage.
        exists: Whether the checkpoint file exists.
    '''

    path: Optional[str]
    is_same_stage: bool
    exists: bool

@dataclass
class TrainingResult:
    '''
    Structured result from a training run.

    Provides a clean interface for accessing training outputs, metrics, and
    paths for downstream processing or testing.

    Attributes:
        best_checkpoint_path: Path to the kept checkpoint: the earliest epoch with the highest
            ``val/outcome_mrr``.
        last_checkpoint_path: Path to the last model checkpoint.
        config_path: Path to the saved configuration file.
        best_score: The kept epoch's ``val/outcome_mrr``, the monitor's validation MRR.
        stopped_epoch: Epoch at which training stopped (early stopping or max).
        early_stopped: Whether early stopping was triggered.
        metrics: Dictionary of final metrics.
    '''

    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    best_score: Optional[float] = None
    stopped_epoch: int = 0
    early_stopped: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------------------------------
# Hardware Detection
# -------------------------------------------------------------------------------------------------

def detect_hardware(log_info: bool = False, *, cuda_precision: str = 'bf16-mixed') -> HardwareInfo:
    '''
    Detect available hardware and the precision the trainer runs at.

    Queries the system for CUDA, MPS, or CPU availability and returns the
    accelerator and precision settings for PyTorch Lightning.

    Args:
        log_info: If True, log detailed hardware information.
        cuda_precision: The precision on CUDA; ``train`` passes ``training.trainer.precision``
            (spec 4.2). Off CUDA the trainer runs at ``32-true``.

    Returns:
        HardwareInfo with detected accelerator, precision, device count,
        and optional GPU memory information.

    Example:
        >>> hw = detect_hardware(log_info=True, cuda_precision='bf16-mixed')
        >>> print(f'Training on {hw.accelerator} with {hw.precision} precision')
    '''
    accelerator, precision, num_devices = get_device(
        log_info=log_info, cuda_precision=cuda_precision
    )
    gpu_memory = None

    if accelerator in ['cuda', 'gpu'] and torch.cuda.is_available():
        gpu_memory = get_gpu_memory_info()

    return HardwareInfo(
        accelerator=accelerator,
        precision=precision,
        num_devices=num_devices,
        gpu_memory=gpu_memory
    )

def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    '''
    Query current GPU memory usage.

    Returns memory statistics including total, reserved, allocated, and free
    memory in gigabytes, along with utilization percentage.

    Returns:
        Dictionary with memory statistics, or None if CUDA unavailable.

    Example:
        >>> info = get_gpu_memory_info()
        >>> if info:
        ...     print(f'GPU Memory: {info["free_gb"]:.1f} GB free')
    '''
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        free = total - reserved

        return {
            'total_gb': total,
            'reserved_gb': reserved,
            'allocated_gb': allocated,
            'free_gb': free,
            'utilization_pct': (reserved / total) * 100 if total > 0 else 0,
        }
    except Exception as e:
        logger.debug(f'Could not get GPU memory info: {e}')
        return None

# -------------------------------------------------------------------------------------------------
# Configuration Parsing
# -------------------------------------------------------------------------------------------------

def parse_config_overrides(overrides: Optional[List[str]]) -> Tuple[Dict[str, Any], List[str]]:
    '''
    Parse and validate command-line configuration overrides.

    Converts a list of ``key=value`` strings into a dictionary suitable for
    use with ``Config.override()``. Invalid overrides are collected and
    returned for warning messages.

    Args:
        overrides: List of override strings like ``training.learning_rate=1e-4``.

    Returns:
        Tuple of (valid_overrides_dict, list_of_invalid_override_strings).

    Example:
        >>> overrides = ['training.learning_rate=1e-4', 'invalid', 'batch_size=32']
        >>> valid, invalid = parse_config_overrides(overrides)
        >>> print(valid)  # {'training.learning_rate': 0.0001, 'batch_size': 32}
        >>> print(invalid)  # ['invalid']
    '''
    if not overrides:
        return {}, []

    override_dict: Dict[str, Any] = {}
    invalid_overrides: List[str] = []

    for override in overrides:
        if '=' not in override:
            invalid_overrides.append(override)
            continue

        key, value_str = override.split('=', 1)
        value = parse_override_value(value_str)
        override_dict[key] = value
        logger.info(f'  • {key} = {value} ({type(value).__name__})')

    return override_dict, invalid_overrides

# -------------------------------------------------------------------------------------------------
# A run's precision and settings
# -------------------------------------------------------------------------------------------------

# The precision a trainer runs at off CUDA (spec 4.2)
FULL_PRECISION = '32-true'

def effective_precision(cfg: Config, accelerator: str) -> str:
    '''
    The precision a run trains at on ``accelerator`` (P4, P31).

    The configured ``training.trainer.precision`` on CUDA, and ``32-true`` everywhere else: the
    rule ``detect_hardware`` resolves for the trainer, so the precision a run records is the one
    it trains at, and ``tools sweep`` names a run's precision from its accelerator alone.

    Args:
        cfg: The run's configuration.
        accelerator: ``cuda``, ``mps`` or ``cpu``.

    Returns:
        A Lightning precision.
    '''

    return cfg.training.trainer.precision if accelerator == 'cuda' else FULL_PRECISION

def run_settings(cfg: Config, *, accelerator: str, precision: str) -> Dict[str, Any]:
    '''
    A run's free settings (R7), its epoch budget, its fusion and the trainer's accumulation and
    clipping, with where it trains (P21).

    ``train`` records them in the model's hyperparameters, so an exact resume under other settings
    is refused (P19), and ``tools sweep`` writes them as the arm's settings. The two trainer
    settings change a run mid-way too: the warmup counts optimizer steps, so another
    ``accumulate_grad_batches`` would stretch it, and another ``gradient_clip_val`` would change
    every later step. Every value is a JSON type, the logit-scale range a list, so an arm record's
    settings equal a run's after the record's round trip.

    Args:
        cfg: The run's configuration.
        accelerator: The accelerator the run trains on.
        precision: The precision it trains at (``effective_precision``).

    Returns:
        The settings, in a fixed order.
    '''

    loss, training = cfg.loss, cfg.training
    return {
        'fusion': cfg.model.fusion,
        'dimension': cfg.model.dimension,
        'radius_bound': cfg.model.radius_bound,
        'code_code_weight': loss.code_code_weight,
        'radial_weight': loss.radial_weight,
        'target_temperature': loss.target_temperature,
        'radial_step': loss.radial_step,
        'logit_scale_init': loss.logit_scale_init,
        'logit_scale_range': list(loss.logit_scale_range),
        'learning_rate': training.learning_rate,
        'weight_decay': training.weight_decay,
        'warmup_epochs': training.warmup_epochs,
        'lr_plateau_factor': training.lr_plateau_factor,
        'lr_plateau_patience': training.lr_plateau_patience,
        'early_stopping_patience': training.early_stopping_patience,
        'max_epochs': training.trainer.max_epochs,
        'queries_per_step': cfg.data_loader.queries_per_step,
        'accumulate_grad_batches': training.trainer.accumulate_grad_batches,
        'gradient_clip_val': training.trainer.gradient_clip_val,
        'accelerator': accelerator,
        'precision': precision,
    }

# -------------------------------------------------------------------------------------------------
# Checkpoint Resolution
# -------------------------------------------------------------------------------------------------

def resolve_checkpoint(
    ckpt_path: Optional[str], checkpoint_dir: Path, experiment_name: str
) -> CheckpointInfo:
    '''
    Resolve a checkpoint path from user input.

    Handles three cases:
    1. ``None``: No checkpoint specified, start fresh.
    2. ``"last"`` or ``"last.ckpt"``: Auto-detect last checkpoint in experiment dir.
    3. Explicit path: Validate and resolve the provided path.

    Args:
        ckpt_path: User-provided checkpoint path or keyword.
        checkpoint_dir: Base directory for checkpoints.
        experiment_name: Name of the current experiment.

    Returns:
        CheckpointInfo with resolved path and metadata.

    Example:
        >>> info = resolve_checkpoint('last', Path('checkpoints'), '01_text')
        >>> if info.exists:
        ...     print(f'Resuming from {info.path}')
    '''
    if not ckpt_path:
        return CheckpointInfo(path=None, is_same_stage=False, exists=False)

    experiment_dir = checkpoint_dir / experiment_name
    ckpt_path_lower = ckpt_path.lower()

    # Handle 'last' keyword
    if ckpt_path_lower in ('last', 'last.ckpt'):
        last_ckpt = experiment_dir / 'last.ckpt'
        if last_ckpt.exists():
            logger.info(f'Auto-detected last checkpoint: {last_ckpt}')
            return CheckpointInfo(path=str(last_ckpt), is_same_stage=True, exists=True)
        else:
            logger.warning(f'Last checkpoint not found at {last_ckpt}')
            return CheckpointInfo(path=None, is_same_stage=False, exists=False)

    # Handle explicit path
    checkpoint_path_obj = Path(ckpt_path)
    if checkpoint_path_obj.exists():
        resolved_path = str(checkpoint_path_obj.resolve())

        # Determine if this is from the same stage
        try:
            is_same_stage = checkpoint_path_obj.resolve().parent == experiment_dir.resolve()
        except Exception:
            is_same_stage = False

        logger.info(f'Using checkpoint: {resolved_path}')
        return CheckpointInfo(path=resolved_path, is_same_stage=is_same_stage, exists=True)
    else:
        logger.warning(f'Checkpoint not found at {ckpt_path}')
        return CheckpointInfo(path=None, is_same_stage=False, exists=False)

def read_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    '''
    A Lightning checkpoint's contents, on the CPU.

    Lightning checkpoints carry pickled hyperparameters and loop state, so they load with
    ``weights_only=False``: they are trusted artifacts of this project's own training runs.
    '''

    return torch.load(Path(path), map_location='cpu', weights_only=False)

# -------------------------------------------------------------------------------------------------
# The guards on a run's checkpoints (P19)
# -------------------------------------------------------------------------------------------------

def refuse_a_fresh_start_into_a_used_directory(checkpoint_dir: Path) -> None:
    '''
    Refuse a fresh run into a checkpoint directory that exists and is not empty (P19).

    The fresh run's checkpoints would sit beside the other run's as ``-v1`` siblings, which
    ModelCheckpoint never deletes, and ``--ckpt-path last`` would resolve to the other run's
    ``last.ckpt``; the monitor would also find its records file taken.

    Raises:
        ValueError: If the directory is not empty, or is not a directory.
    '''

    path = Path(checkpoint_dir)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f'the checkpoint directory {path} is not a directory')
    if any(path.iterdir()):
        raise ValueError(
            f'the checkpoint directory {path} exists and is not empty: a fresh run would train '
            "beside another run's checkpoints and monitor records. Resume that run with "
            '--ckpt-path last, or set another experiment_name'
        )

def refuse_a_resume_from_another_directory(
    checkpoint: Mapping[str, Any], checkpoint_dir: Path
) -> None:
    '''
    Refuse an exact resume from a checkpoint that ModelCheckpoint saved in another directory
    (P19).

    Lightning 2.5.5 restores ModelCheckpoint's best-k state only into the directory it was saved
    from: elsewhere it keeps no earlier epoch, so a lower later epoch would be saved as the best,
    and ``last.ckpt`` would become ``last-v1.ckpt``. The directories compare as ModelCheckpoint
    stores them, as real paths, and the state is the one this run's callback reads.

    Raises:
        ValueError: If the checkpoint holds no state of this run's ModelCheckpoint, or holds one
            saved in another directory.
    '''

    callback = outcome_checkpoint(checkpoint_dir)
    state = (checkpoint.get('callbacks') or {}).get(callback.state_key) or {}
    saved = state.get('dirpath')
    if saved is None:
        raise ValueError(
            f'the checkpoint holds no ModelCheckpoint state on {OUTCOME_MRR}, so an exact resume '
            'would restore no kept epoch: resume a checkpoint this training saved'
        )
    if saved != callback.dirpath:
        raise ValueError(
            f"the checkpoint was saved in another checkpoint directory, {saved}, not this run's "
            f'{callback.dirpath}: ModelCheckpoint would restore none of its kept epochs. Resume it '
            'from the directory it was saved in'
        )

# How a setting one side lacks reads in a refusal
_ABSENT = object()

def _described(value: Any) -> str:
    return 'absent' if value is _ABSENT else repr(value)

def refuse_a_resume_under_other_settings(
    checkpoint: Mapping[str, Any], settings: Mapping[str, Any], *, seed: int
) -> None:
    '''
    Refuse an exact resume whose checkpoint was saved under other run settings or another seed
    (P19).

    ``train`` builds the model and the data from the config, and the checkpoint contract's encoder
    record holds only the fusion, the dimension and the backbone, so a changed setting (the radius
    bound, a learning rate, the epoch budget) would change the run mid-way. The seed draws every
    epoch's permutations and names every monitor read, so it must not change either.

    Args:
        checkpoint: The checkpoint's contents (``read_checkpoint``).
        settings: This run's ``run_settings``.
        seed: This run's seed.

    Raises:
        ValueError: If the checkpoint records no run settings, or a setting or the seed differs,
            naming each difference with its saved and current values.
    '''

    hparams = checkpoint.get('hyper_parameters') or {}
    saved = hparams.get('run_settings')
    if saved is None:
        raise ValueError(
            'the checkpoint records no run settings, so an exact resume could change them '
            'unseen: resume a checkpoint this training saved'
        )
    differences = [
        (key, saved.get(key, _ABSENT), settings.get(key, _ABSENT))
        for key in [*settings, *(key for key in saved if key not in settings)]
        if saved.get(key, _ABSENT) != settings.get(key, _ABSENT)
    ]
    saved_seed = hparams.get('seed', _ABSENT)
    if saved_seed != seed:
        differences.append(('seed', saved_seed, seed))
    if differences:
        named = '; '.join(
            f'{key}: saved {_described(old)}, now {_described(new)}'
            for key, old, new in differences
        )
        raise ValueError(
            f'exact resume under other run settings ({named}): train builds the model from the '
            'config, so the run would change mid-way. Resume with the settings it started with, '
            'or start a new run under another experiment_name'
        )

def refuse_a_resume_of_a_stopped_run(checkpoint: Mapping[str, Any], patience: int) -> None:
    '''
    Refuse an exact resume of a run that early stopping ended (P19, spec 4.4).

    Lightning 2.5.5 restores EarlyStopping's state, ``stopped_epoch`` with it, but not the
    trainer's stop, so a resumed run would train the epochs after its stop: it would append more
    monitor records and could keep another epoch, changing the run's selection after the fact.
    Early stopping checks at an epoch's end before ModelCheckpoint saves, so the ``last.ckpt`` of
    a stopped run records the stop, and its kept checkpoint and ``monitor_reads.jsonl`` are final.
    A run that spent its epoch budget without an early stop is not refused, deliberately: its
    resume trains nothing. The state read is the one this run's EarlyStopping
    (``outcome_early_stopping``) restores.

    Args:
        checkpoint: The checkpoint's contents (``read_checkpoint``).
        patience: This run's ``training.early_stopping_patience``.

    Raises:
        ValueError: If the checkpoint records no state of this run's EarlyStopping, or one whose
            ``stopped_epoch`` is above 0, naming that epoch.
    '''

    callback = outcome_early_stopping(patience)
    state = (checkpoint.get('callbacks') or {}).get(callback.state_key) or {}
    stopped = state.get('stopped_epoch')
    if stopped is None:
        raise ValueError(
            f'the checkpoint records no EarlyStopping state on {OUTCOME_MRR}, so an exact resume '
            'could not tell whether early stopping ended the run: resume a checkpoint this '
            'training saved'
        )
    if stopped > 0:
        raise ValueError(
            f'early stopping ended the run at epoch {stopped}: its kept checkpoint and its monitor '
            'records are final, and it has nothing left to train. Lightning restores early '
            "stopping's state but not its stop, so a resume would train on and change the run's "
            'selection. Start a new run under another experiment_name'
        )

# -------------------------------------------------------------------------------------------------
# Trainer Creation
# -------------------------------------------------------------------------------------------------

def outcome_checkpoint(checkpoint_dir: Path) -> ModelCheckpoint:
    '''
    The ModelCheckpoint the monitor's MRR drives (P17, spec 4.4).

    It keeps one epoch, ``epoch=<NNN>.ckpt``: the earliest with the highest ``val/outcome_mrr``,
    since a later epoch replaces it only by beating it. It also keeps ``last.ckpt`` for exact
    resume. Both save at each training epoch's end, after the module's hook has logged the MRR.
    The exact-resume guard reads the saved state of this same callback.

    Args:
        checkpoint_dir: The run's checkpoint directory.
    '''

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch={epoch:03d}',
        auto_insert_metric_name=False,
        monitor=OUTCOME_MRR,
        mode='max',
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=True,
    )

def outcome_early_stopping(patience: int) -> EarlyStopping:
    '''
    The EarlyStopping the monitor's MRR drives (P17, spec 4.4).

    It ends the run after ``patience`` epochs without a higher ``val/outcome_mrr``, checked at each
    training epoch's end, before ModelCheckpoint saves: Lightning runs checkpoint callbacks last.
    An equal MRR is no gain (``min_delta`` 0), so a tie counts against the patience. The
    stopped-run guard reads the saved state of this same callback.

    Args:
        patience: The run's ``training.early_stopping_patience``.
    '''

    return EarlyStopping(
        monitor=OUTCOME_MRR,
        mode='max',
        patience=patience,
        check_on_train_epoch_end=True,
    )

def create_trainer(
    cfg: Config,
    hardware: HardwareInfo,
    checkpoint_dir: Path,
    callbacks: Optional[List[Callback]] = None,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Tuple[pyl.Trainer, ModelCheckpoint, EarlyStopping]:
    '''
    Create the text stage's Trainer (P17), which ``train`` fits.

    The monitor's ``val/outcome_mrr`` drives the kept checkpoint (``outcome_checkpoint``) and
    early stopping (``outcome_early_stopping``), both checked at each training epoch's end: there
    is no validation loop, since validation is the outcome monitor (spec 4.4).
    ``TrainDatasetEpochCallback`` hands each epoch to the step dataset. The trainer runs on one
    device at the precision ``detect_hardware`` resolved.

    Args:
        cfg: Training configuration.
        hardware: Detected hardware information.
        checkpoint_dir: The run's checkpoint directory.
        callbacks: Optional additional callbacks to include.
        tb_logger: Optional TensorBoard logger (created if not provided).

    Returns:
        Tuple of (Trainer, ModelCheckpoint callback, EarlyStopping callback).

    Example:
        >>> hw = detect_hardware()
        >>> trainer, ckpt_cb, es_cb = create_trainer(cfg, hw, Path('checkpoints'))
        >>> trainer.fit(model, datamodule)
    '''
    checkpoint_callback = outcome_checkpoint(checkpoint_dir)
    early_stopping = outcome_early_stopping(cfg.training.early_stopping_patience)

    # Setup TensorBoard logger if not provided
    if tb_logger is None:
        tb_log_dir = Path(cfg.dirs.output_dir) / cfg.experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorBoardLogger(save_dir=cfg.dirs.output_dir, name=cfg.experiment_name)

    # The step dataset reads the epoch this callback sets at each epoch's start (P27)
    all_callbacks: List[Callback] = [
        checkpoint_callback,
        early_stopping,
        TrainDatasetEpochCallback(),
    ]
    if callbacks:
        all_callbacks.extend(callbacks)

    # Create trainer on one device: the config refuses devices > 1, because the code cache is per
    # process (spec 4.5), and no strategy is passed, so Lightning never picks DDP. No validation
    # loop runs, so training.trainer.val_check_interval is not passed
    trainer = pyl.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=hardware.accelerator,
        devices=1,
        precision=hardware.precision,  # type: ignore
        gradient_clip_val=cfg.training.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=all_callbacks,
        logger=tb_logger,
        default_root_dir=cfg.dirs.output_dir,
    )

    return trainer, checkpoint_callback, early_stopping

# -------------------------------------------------------------------------------------------------
# Summary Artifacts
# -------------------------------------------------------------------------------------------------

def save_training_summary(
    result: TrainingResult,
    config: Config,
    hardware: HardwareInfo,
    output_dir: Path,
    format: str = 'both',
) -> Dict[str, str]:
    '''
    Save training summary artifacts for downstream evaluation and documentation.

    Creates YAML and/or JSON summary files containing training results,
    configuration snapshot, and hardware information. These artifacts can
    be used for evaluation scripts, MkDocs documentation, or CI/CD pipelines.

    Args:
        result: TrainingResult from the completed training run.
        config: Configuration used for training.
        hardware: Hardware information used during training.
        output_dir: Directory to save summary files.
        format: Output format - 'yaml', 'json', or 'both'.

    Returns:
        Dictionary mapping format to output file path.

    Example:
        >>> paths = save_training_summary(result, cfg, hw, Path('outputs'))
        >>> print(f'Summary saved to: {paths}')
    '''
    import json
    from datetime import datetime

    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary data
    summary = {
        'training_run': {
            'experiment': config.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'completed': True,
            'early_stopped': result.early_stopped,
            'stopped_epoch': result.stopped_epoch,
        },
        'results': {
            'best_score': result.best_score,
            'best_checkpoint': result.best_checkpoint_path,
            'last_checkpoint': result.last_checkpoint_path,
            'config_path': result.config_path,
        },
        'metrics': result.metrics,
        'hardware': {
            'accelerator': hardware.accelerator,
            'precision': hardware.precision,
            'num_devices': hardware.num_devices,
            'gpu_memory_gb': (hardware.gpu_memory['total_gb'] if hardware.gpu_memory else None),
        },
        'config_snapshot': {
            'model': {
                'base_model': config.model.base_model_name,
                'lora_rank': config.model.lora.r,
                'num_experts': config.model.moe.num_experts,
            },
            # The settings the run recorded, by the rule train records them with (P21, P31)
            'run_settings': run_settings(
                config,
                accelerator=hardware.accelerator,
                precision=effective_precision(config, hardware.accelerator),
            ),
        },
    }

    output_paths: Dict[str, str] = {}

    if format in ('yaml', 'both'):
        yaml_path = output_dir / 'training_summary.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        output_paths['yaml'] = str(yaml_path)
        logger.info(f'Saved training summary (YAML): {yaml_path}')

    if format in ('json', 'both'):
        json_path = output_dir / 'training_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        output_paths['json'] = str(json_path)
        logger.info(f'Saved training summary (JSON): {json_path}')

    return output_paths
