import io
import os
from types import SimpleNamespace

import click
import pytest
import torch
import typer
from typer.testing import CliRunner

from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    shared_encoder_architecture,
)
from naics_embedder.text_model.dataloader.datamodule import TrainDatasetEpochCallback
from naics_embedder.utils import training as utils_training
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationError, ValidationResult

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM
)
HGCN_QUESTION = 'Generate embeddings parquet file from this checkpoint?'

def _stub_host(monkeypatch, *, cuda: bool) -> None:
    '''Run train's real hardware check on a stubbed host: one CUDA GPU, or a CPU (no MPS).'''

    monkeypatch.setattr(training, 'detect_hardware', utils_training.detect_hardware)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda, raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: int(cuda), raising=False)
    monkeypatch.setattr(torch.backends, 'mps', SimpleNamespace(is_available=lambda: False))
    # The one call detect_hardware makes into the CUDA runtime
    monkeypatch.setattr(utils_training, 'get_gpu_memory_info', lambda: None)

def _closed_stream() -> io.StringIO:
    stream = io.StringIO()
    stream.close()
    return stream

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
            contract_version='stage3-supervision-v2',
            bundle_id='bundle-a',
            codebook_fingerprint='a' * 64,
        )
    )

    def fake_gate(cfg):
        context.events.append('supervision_gate')
        return context.bundle

    monkeypatch.setattr(training, 'require_valid_supervision_bundle', fake_gate)

    def fake_validate_exact_resume(path, runtime):
        context.exact_resume_calls.append((path, runtime))

    monkeypatch.setattr(training, 'validate_exact_resume', fake_validate_exact_resume)

    hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
    monkeypatch.setattr(
        training, 'detect_hardware', lambda log_info=False, cuda_precision=None: hardware
    )

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
        cfg.data_loader.streaming.descriptions_parquet = str(desc_path)
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
def test_the_train_banner_headlines_no_structural_statistic(cli_runner, training_env):
    '''Req 6: the structural statistics are logged for the record, never announced as the
    evaluation.'''

    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output.replace('\n', '')
    for name in ('Cophenetic', 'NDCG', 'Distortion'):
        assert name not in output
    assert 'Structural statistics, for the record only' in output

@pytest.mark.unit
def test_the_train_banner_names_the_fusion_and_dimension(cli_runner, training_env):
    result = cli_runner.invoke(cli_app, ['train', 'model.dimension=8'], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output.replace('\n', '')
    assert 'Fusion: masked_mean' in output
    assert 'Dimension: 8' in output

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
        encoder=CONFIGURED_ENCODER,
        # The seam's dummy pin for the configured tokenizer (tests/conftest.py)
        summaries=summaries_identity(MINILM),
    )

@pytest.mark.unit
def test_the_runtime_contract_records_the_configured_encoder(training_env):
    training.train(skip_validation=True, overrides=['model.fusion=attention', 'model.dimension=8'])

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)

@pytest.mark.unit
def test_the_model_and_its_contract_record_the_tokenizers_summaries(training_env):
    training.train(skip_validation=True)

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['summaries'] == summaries_identity(MINILM)
    assert model_kwargs['checkpoint_contract'].summaries == summaries_identity(MINILM)
    assert model_kwargs['summaries'] is not None

@pytest.mark.unit
def test_the_summaries_follow_the_tokenizer_and_not_the_base_model(training_env):
    # The seam pins MiniLM's summaries alone, so a key taken from the base model finds no pin
    training.train(skip_validation=True, overrides=['model.base_model_name=other/backbone'])

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    # The override took, so the two names differ
    assert model_kwargs['base_model_name'] == 'other/backbone'
    assert model_kwargs['checkpoint_contract'].encoder.backbone == 'other/backbone'
    assert model_kwargs['summaries'] == summaries_identity(MINILM)
    assert model_kwargs['checkpoint_contract'].summaries == summaries_identity(MINILM)
    assert model_kwargs['summaries'] is not None

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
def test_weights_only_is_no_longer_a_checkpoint_load_mode(cli_runner, training_env):
    '''D2: --checkpoint-load-mode keeps one value, exact; weights_only is a usage error.'''

    result = cli_runner.invoke(
        cli_app, ['train', '--ckpt-path', 'last', '--checkpoint-load-mode', 'weights_only']
    )

    assert result.exit_code == 2
    assert 'weights_only' in click.unstyle(result.output).replace('\n', '')
    assert training_env.trainer is None

@pytest.mark.unit
def test_the_remote_launch_line_still_parses(cli_runner, training_env):
    '''`--ckpt-path last --checkpoint-load-mode exact` exact-resumes, as it did before D2.'''

    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    result = cli_runner.invoke(
        cli_app,
        ['train', '--ckpt-path', 'last', '--checkpoint-load-mode', 'exact'],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert training_env.trainer.fit_calls[0]['ckpt_path'] == 'foo.ckpt'
    assert [path for path, _ in training_env.exact_resume_calls] == ['foo.ckpt']

@pytest.mark.unit
def test_train_refuses_the_removed_supervision_mode(training_env):
    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True, overrides=['supervision.mode=legacy_containment'])

    assert excinfo.value.exit_code == 1
    # The config refuses the key before the supervision gate runs
    assert training_env.events == []
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
def test_repaired_model_and_datamodule_receive_bundle_supervision(training_env):
    training.train(skip_validation=True)

    fit = training_env.trainer.fit_calls[0]
    model_kwargs = fit['model'].kwargs
    datamodule_kwargs = fit['datamodule'].kwargs
    assert model_kwargs['supervision_manifest_path'].endswith('manifest.json')
    assert model_kwargs['supervision_bundle'] is training_env.bundle
    assert model_kwargs['checkpoint_contract'].bundle_id == 'bundle-a'
    assert model_kwargs['structural_preference_weight'] == 0.35
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('masked_mean', 16)
    assert 'selection_seed' not in model_kwargs
    # D2: training is always repaired, so neither receives a mode
    for legacy_key in (
        'supervision_mode', 'rank_order_weight', 'distance_matrix_path', 'relations_parquet_path'
    ):
        assert legacy_key not in model_kwargs
    assert 'supervision_mode' not in datamodule_kwargs
    assert datamodule_kwargs['supervision_bundle'] is training_env.bundle

@pytest.mark.unit
def test_training_error_handling_exits(training_env):
    training_env.fail_during_fit = True

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True)

    assert excinfo.value.exit_code == 1

@pytest.mark.unit
@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [(None, 'bf16-mixed'), (["training.trainer.precision='32'"], '32')],
    ids=['default', 'configured'],
)
def test_train_honors_the_configured_precision_on_cuda(
    training_env, monkeypatch, overrides, expected
):
    '''Spec 4.2 and section 6: on CUDA (a stubbed device check) the trainer runs at
    training.trainer.precision, full precision included, on one device.'''

    _stub_host(monkeypatch, cuda=True)

    training.train(skip_validation=True, overrides=overrides)

    trainer_kwargs = training_env.trainer.kwargs
    assert (trainer_kwargs['accelerator'], trainer_kwargs['precision']) == ('cuda', expected)
    assert trainer_kwargs['devices'] == 1
    assert trainer_kwargs.get('strategy', 'auto') == 'auto'

@pytest.mark.unit
def test_train_runs_at_32_true_off_cuda(training_env, monkeypatch):
    '''Off CUDA the trainer keeps 32-true, whatever training.trainer.precision says (spec 4.2).'''

    _stub_host(monkeypatch, cuda=False)

    training.train(skip_validation=True, overrides=['training.trainer.precision=16-mixed'])

    trainer_kwargs = training_env.trainer.kwargs
    assert (trainer_kwargs['accelerator'], trainer_kwargs['precision']) == ('cpu', '32-true')

@pytest.mark.unit
def test_train_refuses_more_than_one_device(training_env):
    '''Spec 4.5 and section 5: devices > 1 exits 1 before anything is built.'''

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True, overrides=['training.trainer.devices=2'])

    assert excinfo.value.exit_code == 1
    # The config refuses the value before the supervision gate runs
    assert training_env.events == []
    assert training_env.trainer is None

@pytest.mark.unit
def test_train_does_not_prompt_without_a_terminal(cli_runner, training_env, monkeypatch):
    '''A remote launch reads stdin from /dev/null: a finished run asks nothing and exits 0
    (spec 4.5, "Stays"; section 5).'''

    questions = []
    monkeypatch.setattr(training.typer, 'confirm', lambda *args, **_: questions.append(args))

    result = cli_runner.invoke(cli_app, ['train'], input='', catch_exceptions=False)

    assert result.exit_code == 0
    assert training_env.trainer.fit_calls
    assert questions == []
    assert 'Generate embeddings' not in click.unstyle(result.output).replace('\n', '')

@pytest.mark.unit
def test_train_asks_the_hgcn_question_on_a_terminal(training_env, monkeypatch):
    '''The question stays (Stage 11): on a terminal it is asked once, defaulting to no.'''

    questions = []

    def confirm(text, default):
        questions.append((text, default))
        return False

    monkeypatch.setattr(training.typer, 'confirm', confirm)
    monkeypatch.setattr(training, '_stdin_is_terminal', lambda: True)

    training.train(skip_validation=True)

    assert questions == [(HGCN_QUESTION, False)]

@pytest.mark.unit
@pytest.mark.parametrize(
    ('make_stdin', 'expected'),
    [
        (lambda: SimpleNamespace(isatty=lambda: True), True),
        (lambda: open(os.devnull), False),
        (lambda: None, False),
        (_closed_stream, False),
    ],
    ids=['terminal', 'dev-null', 'no-stdin', 'closed'],
)
def test_stdin_is_a_terminal_only_when_it_is_one(request, monkeypatch, make_stdin, expected):
    stdin = make_stdin()
    if hasattr(stdin, 'close'):
        request.addfinalizer(stdin.close)
    monkeypatch.setattr(training, 'sys', SimpleNamespace(stdin=stdin))

    assert training._stdin_is_terminal() is expected

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

@pytest.mark.unit
@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [(None, 100), (['data_loader.n_epochs=1'], 1)],
    ids=['shipped', 'overridden'],
)
def test_training_passes_the_pre_sampled_epoch_count_to_datamodule(
    training_env, overrides, expected
):
    '''data_loader.n_epochs reaches the datamodule, and a run without it keeps the shipped 100.'''
    training.train(skip_validation=True, overrides=overrides)

    datamodule = training_env.trainer.fit_calls[0]['datamodule']

    assert datamodule.kwargs['n_epochs'] == expected
