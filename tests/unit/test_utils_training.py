import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from naics_embedder.text_model.dataloader.datamodule import TrainDatasetEpochCallback
from naics_embedder.utils import training as utils_training
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import (
    HardwareInfo,
    TrainingResult,
    create_trainer,
    detect_hardware,
    get_gpu_memory_info,
    parse_config_overrides,
    resolve_checkpoint,
    save_training_summary,
)

OUTCOME_MRR = 'val/outcome_mrr'

def _build_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.experiment_name = 'unit-test'
    cfg.dirs.output_dir = str(tmp_path / 'outputs')
    cfg.dirs.checkpoint_dir = str(tmp_path / 'checkpoints')
    Path(cfg.dirs.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dirs.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.training.trainer.max_epochs = 1
    cfg.training.trainer.devices = 1
    cfg.training.trainer.gradient_clip_val = 0.5
    cfg.training.trainer.accumulate_grad_batches = 1
    cfg.training.trainer.log_every_n_steps = 1
    cfg.training.trainer.val_check_interval = 1.0
    return cfg

@pytest.mark.unit
def test_detect_hardware_cuda_collects_gpu_memory(monkeypatch):
    # The device check returns the CUDA precision it is given (utils/backend.get_device)
    monkeypatch.setattr(
        'naics_embedder.utils.training.get_device',
        lambda log_info=False, *, cuda_precision: ('cuda', cuda_precision, 2),
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(
        'naics_embedder.utils.training.get_gpu_memory_info',
        lambda: {'total_gb': 24.0},
    )

    info = detect_hardware(log_info=True, cuda_precision='32')

    assert info.accelerator == 'cuda'
    assert info.precision == '32'
    assert info.gpu_memory == {'total_gb': 24.0}

@pytest.mark.unit
def test_detect_hardware_cpu_fallback(monkeypatch):
    received = []

    def fake_get_device(log_info=False, *, cuda_precision):
        received.append(cuda_precision)
        return 'cpu', '32-true', 1

    monkeypatch.setattr('naics_embedder.utils.training.get_device', fake_get_device)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)

    info = detect_hardware()

    assert info.accelerator == 'cpu'
    assert info.precision == '32-true'
    assert info.gpu_memory is None
    # The CUDA precision a caller that names none passes on: the shipped bf16-mixed
    assert received == ['bf16-mixed']

@pytest.mark.unit
def test_get_gpu_memory_info_returns_stats(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda device: SimpleNamespace(total_memory=8 * 1024**3),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 2 * 1024**3, raising=False)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 1 * 1024**3, raising=False)

    stats = get_gpu_memory_info()

    assert stats is not None
    assert pytest.approx(stats['total_gb'], rel=1e-3) == 8.0
    assert pytest.approx(stats['reserved_gb'], rel=1e-3) == 2.0
    assert pytest.approx(stats['allocated_gb'], rel=1e-3) == 1.0
    assert pytest.approx(stats['free_gb'], rel=1e-3) == 6.0

@pytest.mark.unit
def test_get_gpu_memory_info_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    assert get_gpu_memory_info() is None

@pytest.mark.unit
def test_parse_config_overrides_handles_invalid_entries():
    overrides = ['training.learning_rate=1e-4', 'bad_override', 'trainer.devices=2']
    parsed, invalid = parse_config_overrides(overrides)

    assert parsed['training.learning_rate'] == pytest.approx(1e-4)
    assert parsed['trainer.devices'] == 2
    assert invalid == ['bad_override']

@pytest.mark.unit
def test_resolve_checkpoint_last_keyword(tmp_path):
    experiment_dir = tmp_path / 'exp'
    experiment_dir.mkdir()
    last_ckpt = experiment_dir / 'last.ckpt'
    last_ckpt.write_text('checkpoint')

    info = resolve_checkpoint('last', tmp_path, 'exp')
    assert info.exists
    assert info.is_same_stage
    assert info.path == str(last_ckpt)

@pytest.mark.unit
def test_resolve_checkpoint_explicit_path(tmp_path):
    ckpt = tmp_path / 'custom.ckpt'
    ckpt.write_text('ckpt')

    info = resolve_checkpoint(str(ckpt), tmp_path, 'exp')
    assert info.exists
    assert info.path == str(ckpt.resolve())

@pytest.mark.unit
def test_resolve_checkpoint_missing_path(tmp_path):
    info = resolve_checkpoint('missing.ckpt', tmp_path, 'exp')
    assert info.exists is False
    assert info.path is None

@pytest.mark.unit
def test_create_trainer_uses_cpu_defaults(tmp_path):
    cfg = _build_config(tmp_path)
    # Nothing reads it: there is no validation loop
    cfg.training.trainer.val_check_interval = 0.25
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer, ckpt_cb, es_cb = create_trainer(cfg, hardware, checkpoint_dir)

    assert trainer.max_epochs == cfg.training.trainer.max_epochs
    assert (trainer.precision, trainer.num_devices) == ('32-true', 1)
    assert trainer.accumulate_grad_batches == cfg.training.trainer.accumulate_grad_batches
    assert trainer.gradient_clip_val == cfg.training.trainer.gradient_clip_val
    assert trainer.log_every_n_steps == cfg.training.trainer.log_every_n_steps
    # No validation loop: validation is the outcome monitor (spec 4.4)
    assert (trainer.limit_val_batches, trainer.num_sanity_val_steps) == (0, 0)
    assert trainer.val_check_interval == 1.0
    assert trainer.logger is not None
    assert ckpt_cb in trainer.callbacks and es_cb in trainer.callbacks

@pytest.mark.unit
def test_create_trainer_keeps_the_earliest_best_epoch_and_the_last_on_the_outcome_mrr(tmp_path):
    '''P17: one kept epoch, the first with the highest val/outcome_mrr (a later epoch replaces it
    only by beating it), saved at each training epoch's end as epoch=<NNN>.ckpt, plus last.ckpt.'''

    cfg = _build_config(tmp_path)
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name

    _, ckpt_cb, _ = create_trainer(cfg, hardware, checkpoint_dir)

    assert isinstance(ckpt_cb, ModelCheckpoint)
    assert ckpt_cb.dirpath == os.path.realpath(checkpoint_dir)
    assert (ckpt_cb.monitor, ckpt_cb.mode, ckpt_cb.save_top_k) == (OUTCOME_MRR, 'max', 1)
    assert ckpt_cb.save_last is True
    # Saved at the training epoch's end, whatever the trainer's validation settings
    assert ckpt_cb._save_on_train_epoch_end is True
    assert ckpt_cb.format_checkpoint_name({'epoch': torch.tensor(7)}
                                          ) == os.path.join(ckpt_cb.dirpath, 'epoch=007.ckpt')
    # The same callback the exact-resume guard reads the saved state of
    assert ckpt_cb.state_key == utils_training.outcome_checkpoint(checkpoint_dir).state_key

@pytest.mark.unit
def test_create_trainer_stops_early_on_the_outcome_mrr_at_the_configured_patience(tmp_path):
    cfg = _build_config(tmp_path)
    cfg.training.early_stopping_patience = 7
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name

    trainer, _, es_cb = create_trainer(cfg, hardware, checkpoint_dir)

    assert isinstance(es_cb, EarlyStopping)
    assert (es_cb.monitor, es_cb.mode, es_cb.patience) == (OUTCOME_MRR, 'max', 7)
    # A tie is no improvement: an equal MRR counts against the patience
    assert es_cb.min_delta == 0.0
    assert es_cb._check_on_train_epoch_end is True
    early_stoppers = [cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)]
    checkpointers = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    assert (len(early_stoppers), len(checkpointers)) == (1, 1)

@pytest.mark.unit
def test_create_trainer_stops_early_with_the_callback_the_stopped_run_guard_reads(tmp_path):
    '''P19: the stopped-run guard reads the state of outcome_early_stopping's callback, so
    create_trainer's EarlyStopping is that callback: the same class, state key and settings.'''

    cfg = _build_config(tmp_path)
    cfg.training.early_stopping_patience = 7
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name

    _, _, es_cb = create_trainer(cfg, hardware, checkpoint_dir)

    helper = utils_training.outcome_early_stopping(7)
    # A subclass would save its state under another key: the key names the class
    assert type(es_cb) is type(helper) is EarlyStopping
    assert es_cb.state_key == helper.state_key
    settings, helper_settings = dict(vars(es_cb)), dict(vars(helper))
    assert torch.equal(settings.pop('best_score'), helper_settings.pop('best_score'))
    assert settings == helper_settings

@pytest.mark.unit
def test_create_trainer_propagates_train_epoch_to_datamodule(tmp_path):
    '''Lightning never calls datamodule epoch hooks, so the trainer must carry the callback.'''
    cfg = _build_config(tmp_path)
    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer, _, _ = create_trainer(cfg, hardware, checkpoint_dir)

    assert any(isinstance(cb, TrainDatasetEpochCallback) for cb in trainer.callbacks)

@pytest.mark.unit
def test_create_trainer_runs_on_one_device_at_the_detected_precision(monkeypatch, tmp_path):
    '''No DDP (spec 4.5: the code cache is per process): one device on a two-GPU host, even for a
    devices value that skipped the config's refusal, at the precision detect_hardware resolved.'''

    cfg = _build_config(tmp_path)
    cfg.training.trainer.devices = 2  # Attribute assignment skips the validator
    hardware = HardwareInfo(accelerator='cuda', precision='32', num_devices=2)
    checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    class DummyTrainer:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('naics_embedder.utils.training.pyl.Trainer', DummyTrainer)

    create_trainer(cfg, hardware, checkpoint_dir)

    assert (captured['accelerator'], captured['devices']) == ('cuda', 1)
    # Lightning's own choice on one device, never a DDP strategy
    assert captured.get('strategy', 'auto') == 'auto'
    assert captured['precision'] == '32'

@pytest.mark.unit
def test_save_training_summary_writes_files(tmp_path):
    result = TrainingResult(
        best_checkpoint_path='epoch=003.ckpt',
        last_checkpoint_path='last.ckpt',
        config_path='config.yaml',
        best_score=0.42,
        stopped_epoch=5,
        early_stopped=True,
        metrics={'best_val_outcome_mrr': 0.42},
    )
    cfg = Config()
    hw = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)

    paths = save_training_summary(result, cfg, hw, tmp_path, format='both')

    assert 'yaml' in paths and Path(paths['yaml']).exists()
    assert 'json' in paths and Path(paths['json']).exists()
    summary = json.loads(Path(paths['json']).read_text())
    # The snapshot holds the settings the run records, by train's rule (P21, P31), so it reads none
    # of the old objective's keys (spec 4.5)
    settings = utils_training.run_settings(cfg, accelerator='cpu', precision='32-true')
    model = {
        'base_model': cfg.model.base_model_name,
        'lora_rank': cfg.model.lora.r,
        'num_experts': cfg.model.moe.num_experts,
    }
    assert summary['config_snapshot'] == {'model': model, 'run_settings': settings}
    from_yaml = yaml.safe_load(Path(paths['yaml']).read_text())
    assert from_yaml['config_snapshot'] == summary['config_snapshot']
    # The best score is the kept epoch's validation MRR, not a loss
    assert summary['results']['best_score'] == 0.42
    assert 'best_loss' not in summary['results']
    assert summary['metrics'] == {'best_val_outcome_mrr': 0.42}

# -------------------------------------------------------------------------------------------------
# One accelerator and precision rule, and the run's settings (P21, P31)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    ('accelerator', 'expected'), [('cuda', '16-mixed'), ('mps', '32-true'), ('cpu', '32-true')]
)
def test_effective_precision_is_the_configured_precision_on_cuda_only(accelerator, expected):
    cfg = Config().override({'training.trainer.precision': '16-mixed'})

    assert utils_training.effective_precision(cfg, accelerator) == expected

@pytest.mark.unit
@pytest.mark.parametrize('host', ['cuda', 'mps', 'cpu'])
def test_effective_precision_is_the_precision_the_hardware_check_resolves(monkeypatch, host):
    '''P31: the precision a run records is the one detect_hardware hands its trainer.'''

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: host == 'cuda', raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', SimpleNamespace(is_available=lambda: host == 'mps'))
    monkeypatch.setattr(utils_training, 'get_gpu_memory_info', lambda: None)
    cfg = Config()

    hardware = detect_hardware(cuda_precision=cfg.training.trainer.precision)

    assert hardware.accelerator == host
    assert utils_training.effective_precision(cfg, hardware.accelerator) == hardware.precision

# Every run setting at a value other than its default, so each one is seen to be read
RUN_OVERRIDES = {
    'model.fusion': 'attention',
    'model.dimension': 8,
    'model.radius_bound': 5.0,
    'loss.code_code_weight': 0.25,
    'loss.radial_weight': 2.0,
    'loss.target_temperature': 0.5,
    'loss.radial_step': 0.75,
    'loss.logit_scale_init': 2.0,
    'loss.logit_scale_range': [0.05, 50.0],
    'training.learning_rate': 3e-4,
    'training.weight_decay': 0.02,
    'training.warmup_epochs': 2,
    'training.lr_plateau_factor': 0.25,
    'training.lr_plateau_patience': 3,
    'training.early_stopping_patience': 6,
    'training.trainer.max_epochs': 12,
    'data_loader.queries_per_step': 64,
    'training.trainer.accumulate_grad_batches': 2,
    'training.trainer.gradient_clip_val': 0.5,
}

@pytest.mark.unit
def test_run_settings_are_the_free_settings_the_epoch_budget_and_the_fusion():
    '''P21: R7's free settings, the epoch budget, the fusion, the accumulation and the clipping,
    the accelerator and the precision, in that order, the logit-scale range a list.'''

    cfg = Config().override(RUN_OVERRIDES)

    settings = utils_training.run_settings(cfg, accelerator='cuda', precision='bf16-mixed')

    assert list(settings.items()) == [
        ('fusion', 'attention'),
        ('dimension', 8),
        ('radius_bound', 5.0),
        ('code_code_weight', 0.25),
        ('radial_weight', 2.0),
        ('target_temperature', 0.5),
        ('radial_step', 0.75),
        ('logit_scale_init', 2.0),
        ('logit_scale_range', [0.05, 50.0]),
        ('learning_rate', 3e-4),
        ('weight_decay', 0.02),
        ('warmup_epochs', 2),
        ('lr_plateau_factor', 0.25),
        ('lr_plateau_patience', 3),
        ('early_stopping_patience', 6),
        ('max_epochs', 12),
        ('queries_per_step', 64),
        ('accumulate_grad_batches', 2),
        ('gradient_clip_val', 0.5),
        ('accelerator', 'cuda'),
        ('precision', 'bf16-mixed'),
    ]
    assert type(settings['logit_scale_range']) is list

@pytest.mark.unit
def test_run_settings_survive_a_records_json_unchanged():
    '''An arm record holds the settings as JSON, so the record's equal the run's.'''

    settings = utils_training.run_settings(Config(), accelerator='cuda', precision='bf16-mixed')

    assert json.loads(json.dumps(settings)) == settings

# -------------------------------------------------------------------------------------------------
# The guards on a run's checkpoint directory (P19)
# -------------------------------------------------------------------------------------------------

# The shipped training.early_stopping_patience (spec 4.4)
PATIENCE = 5

def _saved_state(
    checkpoint_dir: Path,
    settings,
    *,
    seed: int = 42,
    dirpath=None,
    wait_count: int = 0,
    stopped_epoch: int = 0,
) -> dict:
    '''
    What a checkpoint saved in ``checkpoint_dir`` holds of what the guards read: the states of the
    run's ModelCheckpoint and EarlyStopping, under the keys the trainer restores them by, and the
    hyperparameters.
    '''

    callback = utils_training.outcome_checkpoint(checkpoint_dir)
    stopper = utils_training.outcome_early_stopping(PATIENCE)
    return {
        'callbacks': {
            callback.state_key: {
                'dirpath': callback.dirpath if dirpath is None else dirpath,
                'best_model_score': torch.tensor(0.5),
            },
            stopper.state_key: {
                'wait_count': wait_count,
                'stopped_epoch': stopped_epoch,
                'best_score': torch.tensor(0.5, dtype=torch.float64),
                'patience': PATIENCE,
            },
        },
        'hyper_parameters': {
            'seed': seed,
            'run_settings': settings
        },
    }

@pytest.fixture
def current_settings():
    return utils_training.run_settings(Config(), accelerator='cpu', precision='32-true')

@pytest.mark.unit
@pytest.mark.parametrize('state', ['missing', 'empty'])
def test_a_fresh_start_into_a_missing_or_empty_checkpoint_directory_is_allowed(tmp_path, state):
    checkpoint_dir = tmp_path / 'checkpoints' / 'reference'
    if state == 'empty':
        checkpoint_dir.mkdir(parents=True)

    utils_training.refuse_a_fresh_start_into_a_used_directory(checkpoint_dir)

@pytest.mark.unit
@pytest.mark.parametrize('entry', ['last.ckpt', 'monitor_reads.jsonl', 'a-subdirectory'])
def test_a_fresh_start_into_a_used_checkpoint_directory_is_refused(tmp_path, entry):
    '''A fresh run would leave its checkpoints as -v1 siblings of the other run's, and resolve
    --ckpt-path last to the other run's last.ckpt.'''

    checkpoint_dir = tmp_path / 'reference'
    checkpoint_dir.mkdir()
    if entry == 'a-subdirectory':
        (checkpoint_dir / entry).mkdir()
    else:
        (checkpoint_dir / entry).write_text('')

    with pytest.raises(ValueError, match='exists and is not empty') as excinfo:
        utils_training.refuse_a_fresh_start_into_a_used_directory(checkpoint_dir)

    assert str(checkpoint_dir) in str(excinfo.value)
    assert '--ckpt-path last' in str(excinfo.value)

@pytest.mark.unit
def test_a_fresh_start_into_a_file_in_place_of_the_checkpoint_directory_is_refused(tmp_path):
    checkpoint_dir = tmp_path / 'reference'
    checkpoint_dir.write_text('')

    with pytest.raises(ValueError, match='is not a directory'):
        utils_training.refuse_a_fresh_start_into_a_used_directory(checkpoint_dir)

@pytest.mark.unit
def test_a_resume_from_the_runs_own_checkpoint_directory_is_allowed(tmp_path, current_settings):
    '''The saved directory is compared as ModelCheckpoint stores it, its real path: a symlinked
    path to the same directory resumes.'''

    checkpoint_dir = tmp_path / 'checkpoints' / 'reference'
    checkpoint_dir.mkdir(parents=True)
    linked = tmp_path / 'linked'
    linked.symlink_to(tmp_path / 'checkpoints')
    saved = _saved_state(checkpoint_dir, current_settings)

    utils_training.refuse_a_resume_from_another_directory(saved, linked / 'reference')

@pytest.mark.unit
def test_a_resume_from_another_checkpoint_directory_is_refused(tmp_path, current_settings):
    '''Lightning 2.5.5 restores ModelCheckpoint's best-k state only from its own directory, so
    such a resume would lose the kept epoch.'''

    elsewhere = tmp_path / 'elsewhere' / 'reference'
    checkpoint_dir = tmp_path / 'checkpoints' / 'reference'
    saved = _saved_state(elsewhere, current_settings)

    with pytest.raises(ValueError, match='another checkpoint directory') as excinfo:
        utils_training.refuse_a_resume_from_another_directory(saved, checkpoint_dir)

    message = str(excinfo.value)
    assert os.path.realpath(elsewhere) in message
    assert os.path.realpath(checkpoint_dir) in message

@pytest.mark.unit
@pytest.mark.parametrize(
    'corruption',
    ['no-callbacks', 'another-monitor', 'no-dirpath'],
)
def test_a_resume_whose_checkpoint_holds_no_checkpoint_state_of_this_run_is_refused(
    tmp_path, current_settings, corruption
):
    '''Without the state, Lightning would restore no best-k state and keep no earlier epoch.'''

    checkpoint_dir = tmp_path / 'reference'
    saved = _saved_state(checkpoint_dir, current_settings)
    state_key = utils_training.outcome_checkpoint(checkpoint_dir).state_key
    if corruption == 'no-callbacks':
        del saved['callbacks']
    elif corruption == 'another-monitor':
        other = ModelCheckpoint(dirpath=checkpoint_dir, monitor='val/contrastive_loss')
        saved['callbacks'] = {other.state_key: saved['callbacks'][state_key]}
    else:
        del saved['callbacks'][state_key]['dirpath']

    with pytest.raises(ValueError, match='no ModelCheckpoint state'):
        utils_training.refuse_a_resume_from_another_directory(saved, checkpoint_dir)

@pytest.mark.unit
def test_a_resume_under_the_runs_own_settings_is_allowed(tmp_path, current_settings):
    saved = _saved_state(tmp_path, dict(current_settings))

    utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

@pytest.mark.unit
def test_a_resume_under_other_run_settings_is_refused_naming_each_difference(
    tmp_path, current_settings
):
    '''P19: train builds the model from the config, so a changed setting would change the run
    mid-way, with nothing in the checkpoint contract to catch it.'''

    saved_settings = {**current_settings, 'learning_rate': 2e-4, 'max_epochs': 10}
    saved_settings['logit_scale_range'] = [0.05, 50.0]
    saved = _saved_state(tmp_path, saved_settings)

    with pytest.raises(ValueError, match='other run settings') as excinfo:
        utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

    message = str(excinfo.value)
    assert 'learning_rate: saved 0.0002, now 0.0001' in message
    assert 'max_epochs: saved 10, now 40' in message
    assert 'logit_scale_range: saved [0.05, 50.0], now [0.01, 100.0]' in message
    assert 'radius_bound' not in message

@pytest.mark.unit
@pytest.mark.parametrize(
    ('key', 'value', 'named'),
    [
        ('training.trainer.accumulate_grad_batches', 2, 'accumulate_grad_batches: saved 2, now 1'),
        ('training.trainer.gradient_clip_val', 0.5, 'gradient_clip_val: saved 0.5, now 1.0'),
    ],
    ids=['accumulate_grad_batches', 'gradient_clip_val'],
)
def test_a_resume_under_another_accumulation_or_clipping_is_refused(
    tmp_path, current_settings, key, value, named
):
    '''P21: the warmup counts optimizer steps, so another accumulation would stretch it mid-run,
    and another clipping would change every later step. Both are run settings, refused as any.'''

    saved_settings = utils_training.run_settings(
        Config().override({key: value}), accelerator='cpu', precision='32-true'
    )
    saved = _saved_state(tmp_path, saved_settings)

    with pytest.raises(ValueError, match='other run settings') as excinfo:
        utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

    assert named in str(excinfo.value)

@pytest.mark.unit
def test_a_resume_under_a_setting_one_side_lacks_is_refused(tmp_path, current_settings):
    saved_settings = {key: value for key, value in current_settings.items() if key != 'fusion'}
    saved_settings['curriculum_phase1_end'] = 0.5
    saved = _saved_state(tmp_path, saved_settings)

    with pytest.raises(ValueError, match='other run settings') as excinfo:
        utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

    message = str(excinfo.value)
    assert "fusion: saved absent, now 'masked_mean'" in message
    assert 'curriculum_phase1_end: saved 0.5, now absent' in message

@pytest.mark.unit
def test_a_resume_under_another_seed_is_refused(tmp_path, current_settings):
    '''The seed draws every epoch's permutations and names every monitor read.'''

    saved = _saved_state(tmp_path, current_settings, seed=7)

    with pytest.raises(ValueError, match='seed: saved 7, now 42'):
        utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

@pytest.mark.unit
@pytest.mark.parametrize('hparams', ['no-run-settings', 'no-hyperparameters'])
def test_a_resume_from_a_checkpoint_that_records_no_run_settings_is_refused(
    tmp_path, current_settings, hparams
):
    saved = _saved_state(tmp_path, None)
    if hparams == 'no-hyperparameters':
        del saved['hyper_parameters']

    with pytest.raises(ValueError, match='records no run settings'):
        utils_training.refuse_a_resume_under_other_settings(saved, current_settings, seed=42)

@pytest.mark.unit
@pytest.mark.parametrize('wait_count', [0, PATIENCE - 1], ids=['improving', 'waiting'])
def test_a_resume_of_a_run_early_stopping_has_not_ended_is_allowed(
    tmp_path, current_settings, wait_count
):
    '''A stopped_epoch of 0 is a run early stopping has not ended, however long it has waited.'''

    saved = _saved_state(tmp_path, current_settings, wait_count=wait_count)

    utils_training.refuse_a_resume_of_a_stopped_run(saved, PATIENCE)

@pytest.mark.unit
@pytest.mark.parametrize('stopped_epoch', [1, 3])
def test_a_resume_of_a_run_early_stopping_ended_is_refused_naming_its_epoch(
    tmp_path, current_settings, stopped_epoch
):
    '''P19: Lightning 2.5.5 restores early stopping's state but not the trainer's stop, so a
    resumed run would train past its stop and change its kept checkpoint and records. Epoch 1 is
    the earliest stop there can be, at a patience of 1.'''

    saved = _saved_state(
        tmp_path, current_settings, wait_count=PATIENCE, stopped_epoch=stopped_epoch
    )

    with pytest.raises(
        ValueError, match=f'early stopping ended the run at epoch {stopped_epoch}'
    ) as excinfo:
        utils_training.refuse_a_resume_of_a_stopped_run(saved, PATIENCE)

    assert 'another experiment_name' in str(excinfo.value)

@pytest.mark.unit
@pytest.mark.parametrize('corruption', ['no-callbacks', 'another-monitor', 'no-stopped-epoch'])
def test_a_resume_whose_checkpoint_records_no_early_stopping_state_of_this_run_is_refused(
    tmp_path, current_settings, corruption
):
    '''Without the state, the guard could not tell a run early stopping ended from one it has
    not.'''

    saved = _saved_state(tmp_path, current_settings)
    state_key = utils_training.outcome_early_stopping(PATIENCE).state_key
    if corruption == 'no-callbacks':
        del saved['callbacks']
    elif corruption == 'another-monitor':
        # The early stopping of the runs before Req 11's objective
        other = EarlyStopping(monitor='val/contrastive_loss', mode='min')
        saved['callbacks'][other.state_key] = saved['callbacks'].pop(state_key)
    else:
        del saved['callbacks'][state_key]['stopped_epoch']

    with pytest.raises(ValueError, match='records no EarlyStopping state'):
        utils_training.refuse_a_resume_of_a_stopped_run(saved, PATIENCE)

@pytest.mark.unit
def test_read_checkpoint_loads_a_saved_checkpoint_on_the_cpu(tmp_path, current_settings):
    path = tmp_path / 'last.ckpt'
    saved = {**_saved_state(tmp_path, current_settings), 'state_dict': {'w': torch.ones(2)}}
    torch.save(saved, path)

    loaded = utils_training.read_checkpoint(path)

    assert loaded['hyper_parameters'] == saved['hyper_parameters']
    assert loaded['state_dict']['w'].device.type == 'cpu'
    assert torch.equal(loaded['state_dict']['w'], torch.ones(2))

@pytest.mark.unit
@pytest.mark.parametrize('fusion', ['masked_mean', 'attention', 'moe'])
def test_saved_constructor_controls_match_without_changing_the_21_key_identity(fusion):
    cfg = Config().override({'model.fusion': fusion})
    saved = {
        'hyper_parameters': {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'num_experts': 4,
            'top_k': 2,
            'moe_hidden_dim': 1024,
            'load_balancing_coef': 0.01,
        }
    }
    before = utils_training.run_settings(cfg, accelerator='cpu', precision='32-true')
    utils_training.refuse_other_constructor_settings(saved, cfg)
    assert utils_training.run_settings(cfg, accelerator='cpu', precision='32-true') == before
    assert len(before) == 21

@pytest.mark.unit
@pytest.mark.parametrize('fusion', ['masked_mean', 'attention'])
def test_inactive_moe_controls_do_not_change_constructor_identity(fusion):
    cfg = Config().override(
        {
            'model.fusion': fusion,
            'model.moe.num_experts': 8,
            'model.moe.top_k': 1,
            'model.moe.hidden_dim': 512,
            'model.moe.load_balancing_coef': 0.2,
        }
    )
    # Inactive controls need not even be saved; neither fusion constructs experts (R11).
    saved = {'hyper_parameters': {'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1}}
    utils_training.refuse_other_constructor_settings(saved, cfg)

@pytest.mark.unit
@pytest.mark.parametrize(
    'name', [
        'lora_r',
        'lora_alpha',
        'lora_dropout',
        'num_experts',
        'top_k',
        'moe_hidden_dim',
        'load_balancing_coef',
    ]
)
def test_missing_active_constructor_controls_fail_closed(name):
    cfg = Config().override({'model.fusion': 'moe'})
    saved = {
        'hyper_parameters': {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'num_experts': 4,
            'top_k': 2,
            'moe_hidden_dim': 1024,
            'load_balancing_coef': 0.01,
        }
    }
    del saved['hyper_parameters'][name]
    with pytest.raises(ValueError, match=f'{name}: saved absent'):
        utils_training.refuse_other_constructor_settings(saved, cfg)
