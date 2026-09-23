from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.supervision.checkpoints import CheckpointContract, MigrationReport
from naics_embedder.text_model.dataloader.datamodule import TrainDatasetEpochCallback
from naics_embedder.utils.config import CheckpointLoadMode, Config
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationError, ValidationResult


@pytest.fixture
def cli_runner():
    return CliRunner()

@pytest.fixture
def training_env(monkeypatch, tmp_path):
    context = SimpleNamespace()
    context.checkpoint_info = CheckpointInfo(path=None, is_same_stage=False, exists=False)
    context.validation_result = ValidationResult.success()
    context.save_summary_calls = []
    context.fail_during_fit = False
    context.trainer = None
    context.events = []
    context.exact_resume_calls = []
    context.bundle = SimpleNamespace(
        manifest=SimpleNamespace(
            contract_version='stage3-supervision-v1',
            bundle_id='bundle-a',
            codebook_fingerprint='a' * 64,
        )
    )

    def fake_gate(cfg):
        context.events.append('supervision_gate')
        return None if cfg.supervision.mode == 'legacy_containment' else context.bundle

    monkeypatch.setattr(training, 'require_valid_supervision_bundle', fake_gate)

    def fake_validate_exact_resume(path, runtime):
        context.exact_resume_calls.append((path, runtime))

    monkeypatch.setattr(training, 'validate_exact_resume', fake_validate_exact_resume)

    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    monkeypatch.setattr(training, 'detect_hardware', lambda log_info=False: hardware)

    def build_cfg():
        cfg = Config()
        cfg.experiment_name = 'cli-test'
        outputs_dir = tmp_path / 'outputs'
        checkpoints_dir = tmp_path / 'checkpoints'
        outputs_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        cfg.dirs.output_dir = str(outputs_dir)
        cfg.dirs.checkpoint_dir = str(checkpoints_dir)
        desc_path = tmp_path / 'descriptions.parquet'
        desc_path.write_text('data')
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir(exist_ok=True)
        cfg.data_loader.streaming.descriptions_parquet = str(desc_path)
        cfg.data_loader.streaming.triplets_parquet = str(triplets_dir)
        cfg.supervision.manifest_path = str(tmp_path / 'bundle' / 'manifest.json')
        cfg.training.trainer.max_epochs = 1
        cfg.training.trainer.devices = 1
        cfg.training.trainer.log_every_n_steps = 1
        cfg.training.trainer.val_check_interval = 1.0
        cfg.training.trainer.gradient_clip_val = 0.5
        cfg.training.trainer.accumulate_grad_batches = 1
        cfg.loss.__dict__['base_margin'] = 1.0
        return cfg

    monkeypatch.setattr(training.Config, 'from_yaml', classmethod(lambda cls, path: build_cfg()))
    monkeypatch.setattr(training, 'validate_training_config', lambda cfg: context.validation_result)

    original_override = Config.override

    def override_with_margin(self, overrides):
        new_cfg = original_override(self, overrides)
        new_cfg.loss.__dict__['base_margin'] = 1.0
        return new_cfg

    monkeypatch.setattr(Config, 'override', override_with_margin, raising=False)

    def fake_resolve(ckpt_path, checkpoint_dir, experiment_name):
        context.events.append('resolve_checkpoint')
        context.resolve_args = (ckpt_path, str(checkpoint_dir), experiment_name)
        return context.checkpoint_info

    monkeypatch.setattr(training, 'resolve_checkpoint', fake_resolve)

    class DummyDataModule:

        def __init__(self, *args, **kwargs):
            context.events.append('datamodule')
            self.kwargs = kwargs

    monkeypatch.setattr(training, 'NAICSDataModule', DummyDataModule)

    class DummyModel:

        def __init__(self, **kwargs):
            context.events.append('model')
            self.kwargs = kwargs
            self.loaded_from_ckpt = False

        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            instance = cls(**kwargs)
            instance.loaded_from_ckpt = True
            instance.ckpt_path = path  # pyright: ignore[reportAttributeAccessIssue]
            return instance

    monkeypatch.setattr(training, 'NAICSContrastiveModel', DummyModel)

    class DummyTrainer:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_calls = []
            context.trainer = self

        def fit(self, model, datamodule, ckpt_path=None):
            if context.fail_during_fit:
                raise RuntimeError('boom')
            self.fit_calls.append(
                {
                    'model': model,
                    'datamodule': datamodule,
                    'ckpt_path': ckpt_path
                }
            )

    monkeypatch.setattr(training.pyl, 'Trainer', DummyTrainer)

    def fake_save_summary(**kwargs):
        context.save_summary_calls.append(kwargs)
        return {'json': str(tmp_path / 'summary.json')}

    monkeypatch.setattr(training, 'save_training_summary', fake_save_summary)
    monkeypatch.setattr(training.typer, 'confirm', lambda *_, **__: False)

    return context

@pytest.mark.unit
def test_cli_train_runs_with_defaults(cli_runner, training_env):
    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    assert training_env.trainer is not None
    assert training_env.trainer.fit_calls[0]['ckpt_path'] is None
    assert training_env.save_summary_calls

@pytest.mark.unit
def test_training_propagates_train_epoch_to_datamodule(training_env):
    '''Lightning never calls datamodule epoch hooks, so the trainer must carry the callback.'''
    training.train(skip_validation=True)

    callbacks = training_env.trainer.kwargs['callbacks']
    assert any(isinstance(cb, TrainDatasetEpochCallback) for cb in callbacks)

@pytest.mark.unit
def test_cli_train_applies_overrides(cli_runner, training_env, monkeypatch):
    captured = {}

    def fake_parse(overrides):
        captured['overrides'] = overrides
        return {'training.learning_rate': 0.5}, []

    monkeypatch.setattr(training, 'parse_config_overrides', fake_parse)

    result = cli_runner.invoke(
        cli_app, ['train', 'training.learning_rate=0.5'], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert captured['overrides'] == ['training.learning_rate=0.5']

@pytest.mark.unit
def test_training_workflow_validation_failure(training_env):
    training_env.validation_result = ValidationResult(valid=False, errors=['boom'], warnings=[])

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=False)

    assert excinfo.value.exit_code == 1

@pytest.mark.unit
def test_training_checkpoint_resume_passes_ckpt(training_env):
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    training.train(ckpt_path='last', skip_validation=True)

    assert training_env.trainer.fit_calls[0]['ckpt_path'] == 'foo.ckpt'
    [(path, runtime)] = training_env.exact_resume_calls
    assert path == 'foo.ckpt'
    assert runtime == CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )

@pytest.mark.unit
def test_exact_resume_contract_mismatch_fails_before_training(training_env, monkeypatch):
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    def reject(path, runtime):
        raise ValueError('exact resume contract mismatch')

    monkeypatch.setattr(training, 'validate_exact_resume', reject)

    with pytest.raises(typer.Exit) as excinfo:
        training.train(ckpt_path='last', skip_validation=True)

    assert excinfo.value.exit_code == 1
    assert training_env.trainer is None

@pytest.mark.unit
def test_weights_only_never_passes_checkpoint_to_trainer(training_env, monkeypatch):
    training_env.checkpoint_info = CheckpointInfo(
        path='legacy.ckpt',
        is_same_stage=False,
        exists=True,
    )
    reports = []

    def fake_load_weights_only(model, path):
        assert path == 'legacy.ckpt'
        report = MigrationReport(
            loaded=('encoder.weight', ),
            skipped=('loss_fn.buffer', ),
            missing=(),
            unexpected=(),
        )
        reports.append(report)
        return report

    monkeypatch.setattr(training, 'load_weights_only', fake_load_weights_only)

    training.train(
        ckpt_path='last',
        checkpoint_load_mode=CheckpointLoadMode.WEIGHTS_ONLY,
        skip_validation=True,
    )

    assert reports
    assert training_env.exact_resume_calls == []
    assert training_env.trainer.fit_calls[0]['ckpt_path'] is None

@pytest.mark.unit
def test_weights_only_without_an_existing_checkpoint_is_fatal(training_env, monkeypatch):
    training_env.checkpoint_info = CheckpointInfo(path=None, is_same_stage=False, exists=False)
    monkeypatch.setattr(
        training,
        'load_weights_only',
        lambda *_args: pytest.fail('nothing to migrate'),
    )

    with pytest.raises(typer.Exit) as excinfo:
        training.train(
            ckpt_path='missing.ckpt',
            checkpoint_load_mode=CheckpointLoadMode.WEIGHTS_ONLY,
            skip_validation=True,
        )

    assert excinfo.value.exit_code == 1
    assert training_env.trainer is None

@pytest.mark.unit
def test_supervision_gate_runs_before_datamodule_checkpoint_and_model(training_env):
    training.train(skip_validation=True)

    assert training_env.events[0] == 'supervision_gate'
    assert training_env.events.index('supervision_gate') < training_env.events.index('model')

@pytest.mark.unit
def test_supervision_gate_failure_stops_before_any_construction(training_env, monkeypatch):

    def fail(cfg):
        training_env.events.append('supervision_gate')
        raise ValidationError('Repaired Stage-3 training requires supervision.manifest_path')

    monkeypatch.setattr(training, 'require_valid_supervision_bundle', fail)

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True)

    assert excinfo.value.exit_code == 1
    assert training_env.events == ['supervision_gate']

@pytest.mark.unit
def test_cli_legacy_containment_is_prominently_tagged(
    cli_runner, training_env, monkeypatch, caplog
):
    cfg = training.Config.from_yaml('unused.yaml')
    cfg.supervision.mode = 'legacy_containment'
    cfg.supervision.manifest_path = None
    cfg.loss.rank_order_weight = 0.35
    cfg.data_loader.streaming.phase1_exclusion_weight = 100.0
    monkeypatch.setattr(
        training.Config,
        'from_yaml',
        classmethod(lambda cls, path: cfg),
    )

    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    assert 'LEGACY CONTAINMENT' in caplog.text or 'LEGACY CONTAINMENT' in result.output
    model = training_env.trainer.fit_calls[0]['model']
    assert model.kwargs['supervision_mode'] == 'legacy_containment'
    assert model.kwargs['checkpoint_contract'].bundle_id == 'legacy-containment'

@pytest.mark.unit
def test_legacy_containment_uses_legacy_inputs_without_a_bundle(training_env, monkeypatch):
    cfg = training.Config.from_yaml('unused.yaml')
    cfg.supervision.mode = 'legacy_containment'
    cfg.supervision.manifest_path = None
    monkeypatch.setattr(training.Config, 'from_yaml', classmethod(lambda cls, path: cfg))

    training.train(skip_validation=True)

    fit = training_env.trainer.fit_calls[0]
    model_kwargs = fit['model'].kwargs
    assert model_kwargs['supervision_manifest_path'] is None
    assert model_kwargs['supervision_bundle'] is None
    assert model_kwargs['distance_matrix_path'] == cfg.data_loader.streaming.distance_matrix_parquet
    assert model_kwargs['relations_parquet_path'] == cfg.data_loader.streaming.relations_parquet
    assert fit['datamodule'].kwargs['supervision_mode'] == 'legacy_containment'
    assert fit['datamodule'].kwargs['supervision_bundle'] is None

@pytest.mark.unit
def test_repaired_model_and_datamodule_receive_bundle_supervision(training_env):
    training.train(skip_validation=True)

    fit = training_env.trainer.fit_calls[0]
    model_kwargs = fit['model'].kwargs
    datamodule_kwargs = fit['datamodule'].kwargs
    assert model_kwargs['supervision_mode'] == 'repaired'
    assert model_kwargs['supervision_manifest_path'].endswith('manifest.json')
    assert model_kwargs['supervision_bundle'] is training_env.bundle
    assert model_kwargs['checkpoint_contract'].bundle_id == 'bundle-a'
    assert model_kwargs['structural_preference_weight'] == 0.35
    assert model_kwargs['selection_seed'] == 42
    for legacy_key in ('rank_order_weight', 'distance_matrix_path', 'relations_parquet_path'):
        assert legacy_key not in model_kwargs
    assert datamodule_kwargs['supervision_mode'] == 'repaired'
    assert datamodule_kwargs['supervision_bundle'] is training_env.bundle

@pytest.mark.unit
def test_training_error_handling_exits(training_env):
    training_env.fail_during_fit = True

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True)

    assert excinfo.value.exit_code == 1

@pytest.mark.unit
def test_training_passes_curriculum_horizon_to_datamodule(training_env):
    '''Phase1MapDataset's difficulty ramp must end where the model's curriculum Phase 1 ends.'''
    training.train(
        skip_validation=True,
        overrides=['training.trainer.max_epochs=10', 'curriculum.phase1_end=0.5'],
    )

    fit_call = training_env.trainer.fit_calls[0]
    datamodule = fit_call['datamodule']
    model = fit_call['model']

    assert datamodule.kwargs.get('max_epochs') == 10
    assert datamodule.kwargs.get('phase1_end') == 0.5
    # The model's CurriculumScheduler derives Phase 1 from trainer.max_epochs and this hparam
    assert datamodule.kwargs['max_epochs'] == training_env.trainer.kwargs['max_epochs']
    assert datamodule.kwargs['phase1_end'] == model.kwargs['curriculum_phase1_end']
