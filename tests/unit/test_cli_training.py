import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typer.testing import CliRunner

from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    shared_encoder_architecture,
)
from naics_embedder.text_model.dataloader.datamodule import (
    NAICSDataModule,
    TrainDatasetEpochCallback,
)
from naics_embedder.text_model.export import code_token_config
from naics_embedder.text_model.monitor import MONITOR_RECORDS, OutcomeMonitor
from naics_embedder.utils import training as utils_training
from naics_embedder.utils.config import Config, OutcomePanelConfig
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationError, ValidationResult

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
OUTCOME_MRR = 'val/outcome_mrr'
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

    # P28: the outcome panel's selection log is tmp_path's, and the monitor is a stand-in, so no
    # test that runs train reads a panel or writes a log
    context.selection_log = tmp_path / 'logs' / 'selection_log.jsonl'
    context.load_config_calls = []

    def fake_load_config(config_class, yaml_path):
        context.load_config_calls.append((config_class, yaml_path))
        return OutcomePanelConfig(selection_log=str(context.selection_log))

    monkeypatch.setattr(training, 'load_config', fake_load_config)

    context.monitor = SimpleNamespace(name='the run monitor')
    context.monitor_calls = []

    def fake_build_monitor(cfg, bundle, checkpoint_dir, *, selection_log):
        context.events.append('monitor')
        context.monitor_calls.append(
            {
                'cfg': cfg,
                'bundle': bundle,
                'checkpoint_dir': checkpoint_dir,
                'selection_log': selection_log,
            }
        )
        return context.monitor

    monkeypatch.setattr(training, 'build_monitor_from_config', fake_build_monitor)

    # An exact resume reads the checkpoint's saved directory, settings and early-stopping state
    # (P19). By default the checkpoint is one this environment's run saved, early stopping not
    # having ended it; a test sets saved_checkpoint to another.
    def matching_checkpoint():
        cfg = build_cfg()
        checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
        callback = utils_training.outcome_checkpoint(checkpoint_dir)
        stopper = utils_training.outcome_early_stopping(cfg.training.early_stopping_patience)
        return {
            'callbacks': {
                callback.state_key: {
                    'dirpath': callback.dirpath
                },
                stopper.state_key: stopper.state_dict(),
            },
            'hyper_parameters': {
                'seed': cfg.seed,
                'run_settings': utils_training.run_settings(
                    cfg, accelerator=hardware.accelerator, precision=hardware.precision
                ),
            },
        }

    context.matching_checkpoint = matching_checkpoint
    context.saved_checkpoint = None
    context.read_checkpoint_calls = []

    def fake_read_checkpoint(path):
        context.read_checkpoint_calls.append(path)
        if context.saved_checkpoint is not None:
            return context.saved_checkpoint
        return matching_checkpoint()

    monkeypatch.setattr(training, 'read_checkpoint', fake_read_checkpoint)

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
    '''Req 6: the text stage's validation computes no structural statistic, and the banner names
    the monitor's MRR as what selects (spec 4.4).'''

    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output.replace('\n', '')
    for name in ('Cophenetic', 'NDCG', 'Distortion', 'Structural statistics', 'Collapse'):
        assert name not in output
    assert OUTCOME_MRR in output
    assert 'Batch size' not in output
    assert 'Queries per step: 128' in output

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
@pytest.mark.parametrize(('overrides', 'bound'), [([], 8.0), (['model.radius_bound=5.0'], 5.0)])
def test_the_model_takes_the_configured_radius_bound(training_env, overrides, bound):
    training.train(skip_validation=True, overrides=overrides)

    assert training_env.trainer.fit_calls[0]['model'].kwargs['radius_bound'] == bound

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
    assert model_kwargs['seed'] == 42
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('masked_mean', 16)
    assert 'selection_seed' not in model_kwargs
    # D2: training is always repaired, so neither receives a mode; and the old objective's
    # settings are gone (Req 10, 11)
    for legacy_key in (
        'supervision_mode',
        'rank_order_weight',
        'distance_matrix_path',
        'relations_parquet_path',
        'temperature',
        'curvature',
        'hierarchy_weight',
        'radius_reg_weight',
        'level_radius_weight',
        'warmup_steps',
        'base_margin',
        'structural_preference_weight',
        'false_negative_config',
        'curriculum_anneal',
        'eval_sample_size',
    ):
        assert legacy_key not in model_kwargs
    assert 'supervision_mode' not in datamodule_kwargs
    assert datamodule_kwargs['supervision_bundle'] is training_env.bundle

@pytest.mark.unit
def test_the_configured_model_builds_on_a_real_bundle(
    tiny_backbone, reference_manifest, reference_bundle
):
    '''The stand-in model above takes any argument; the real one takes only its own (P15).'''

    cfg = Config()
    cfg.supervision.manifest_path = str(reference_manifest)
    contract = training.runtime_contract_for(cfg, reference_bundle)
    settings = utils_training.run_settings(cfg, accelerator='cpu', precision='32-true')
    monitor = SimpleNamespace(name='the run monitor')

    model = training.build_model_from_config(
        cfg, contract, reference_bundle, run_settings=settings, monitor=monitor
    )

    assert model.checkpoint_contract == contract
    hparams = model.hparams
    assert (hparams['fusion'], hparams['dimension'], hparams['radius_bound']) == (
        cfg.model.fusion,
        cfg.model.dimension,
        cfg.model.radius_bound,
    )
    assert (hparams['learning_rate'], hparams['weight_decay']) == (
        cfg.training.learning_rate,
        cfg.training.weight_decay,
    )
    assert hparams['seed'] == cfg.seed
    # The run's settings are saved, for the exact-resume guard; the monitor is not (P15)
    assert hparams['run_settings'] == settings
    assert model.monitor is monitor
    assert 'monitor' not in hparams

@pytest.mark.unit
def test_the_datamodule_receives_the_token_cache_the_seed_and_the_bundle(training_env):
    '''Exactly its own arguments (spec 4.3): the token cache the export reads, the run's seed and
    the validated bundle, and none of the old data path's settings.'''

    training.train(skip_validation=True)

    cfg = training.Config.from_yaml('config.yaml')  # the environment's config, built again
    datamodule = training_env.trainer.fit_calls[0]['datamodule']
    assert datamodule.kwargs == {
        'token_config': code_token_config(cfg),
        'seed': cfg.seed,
        'queries_per_step': cfg.data_loader.queries_per_step,
        'supervision_manifest_path': cfg.supervision.manifest_path,
        'supervision_contract_version': cfg.supervision.contract_version,
        'supervision_bundle': training_env.bundle,
    }

@pytest.mark.unit
def test_the_configured_datamodule_builds_on_a_real_bundle(reference_manifest, reference_bundle):
    '''The stand-in datamodule above takes any argument; the real one takes only its own.'''

    cfg = Config().override({'data_loader.queries_per_step': 4})
    cfg.supervision.manifest_path = str(reference_manifest)

    datamodule = training.build_datamodule_from_config(cfg, reference_bundle)

    assert isinstance(datamodule, NAICSDataModule)
    assert datamodule.token_config == code_token_config(cfg)
    assert (datamodule.seed, datamodule.queries_per_step) == (cfg.seed, 4)
    assert datamodule.supervision_manifest_path == str(reference_manifest)
    # Construction reads nothing: the step dataset is built in setup
    assert datamodule.train_dataset is None

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

# -------------------------------------------------------------------------------------------------
# The run's monitor, settings, trainer and checkpoint directory (spec 4.4; P17, P19, P21, P31)
# -------------------------------------------------------------------------------------------------

def _checkpoint_dir(tmp_path: Path) -> Path:
    '''The checkpoint directory of training_env's run.'''

    return tmp_path / 'checkpoints' / 'cli-test'

@pytest.mark.unit
def test_train_builds_the_monitor_from_the_bundle_the_outcome_panel_log_and_the_checkpoint_dir(
    training_env, tmp_path
):
    '''Spec 4.4: the monitor reads the bundle's outcome panel, logs to
    conf/data/outcome_panel.yaml's selection_log, and keeps its records in the run's checkpoint
    directory.'''

    training.train(skip_validation=True)

    assert training_env.load_config_calls == [(OutcomePanelConfig, 'data/outcome_panel.yaml')]
    [call] = training_env.monitor_calls
    assert call['bundle'] is training_env.bundle
    assert call['selection_log'] == str(training_env.selection_log)
    assert Path(call['checkpoint_dir']) == _checkpoint_dir(tmp_path)
    assert call['cfg'].experiment_name == 'cli-test'
    assert training_env.trainer.fit_calls[0]['model'].kwargs['monitor'] is training_env.monitor
    # Built after the supervision gate, before the model
    events = training_env.events
    assert events.index('supervision_gate') < events.index('monitor') < events.index('model')

@pytest.mark.unit
def test_the_configured_monitor_builds_on_a_real_bundle(
    tmp_path, reference_manifest, reference_bundle
):
    '''The stand-in monitor above takes any argument; the real one reads the bundle's panel and
    tokenizes queries as the token cache does. Building it reads and writes no log.'''

    cfg = Config()
    cfg.supervision.manifest_path = str(reference_manifest)
    log = tmp_path / 'logs' / 'selection_log.jsonl'
    checkpoint_dir = tmp_path / 'checkpoints' / cfg.experiment_name

    monitor = training.build_monitor_from_config(
        cfg, reference_bundle, checkpoint_dir, selection_log=log
    )

    assert isinstance(monitor, OutcomeMonitor)
    assert monitor.panel.log.path == log
    assert monitor.panel.fingerprint == OutcomePanel.from_bundle(reference_bundle, log).fingerprint
    assert monitor.records_path == checkpoint_dir / MONITOR_RECORDS
    token_config = code_token_config(cfg)
    assert monitor.max_length == token_config.max_length == 128
    assert monitor.tokenizer.name_or_path == token_config.tokenizer_name
    assert cfg.experiment_name in monitor.purpose
    assert not log.exists() and not checkpoint_dir.exists()

# Every added key at a value other than its default
ADDED_KEY_OVERRIDES = {
    'loss.code_code_weight': 0.25,
    'loss.radial_weight': 2.0,
    'loss.target_temperature': 0.5,
    'loss.radial_step': 0.75,
    'loss.logit_scale_init': 2.0,
    'loss.logit_scale_range': [0.05, 50.0],
    'training.warmup_epochs': 3,
    'training.lr_plateau_factor': 0.25,
    'training.lr_plateau_patience': 4,
}

@pytest.mark.unit
def test_the_model_takes_every_added_key_the_run_settings_and_the_monitor(training_env):
    '''Spec 4.5's added keys reach the model, from the command line too, and so do the run's
    settings (P21) and its monitor.'''

    overrides = [f'{key}={value}' for key, value in ADDED_KEY_OVERRIDES.items()]

    training.train(skip_validation=True, overrides=overrides)

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    for key, value in ADDED_KEY_OVERRIDES.items():
        name = key.split('.')[-1]
        expected = tuple(value) if isinstance(value, list) else value
        assert model_kwargs[name] == expected, name
    cfg = training.Config.from_yaml('config.yaml').override(ADDED_KEY_OVERRIDES)
    assert model_kwargs['run_settings'] == utils_training.run_settings(
        cfg, accelerator='cpu', precision='32-true'
    )
    assert model_kwargs['monitor'] is training_env.monitor

@pytest.mark.unit
def test_train_records_the_run_settings_a_sweep_builds_for_its_accelerator(
    training_env, monkeypatch
):
    '''P31: on a stubbed CUDA host, the settings train records equal the ones tools sweep builds
    for --accelerator cuda, the logit-scale range a list on both sides, and they name the
    precision the trainer runs at.'''

    _stub_host(monkeypatch, cuda=True)

    training.train(skip_validation=True)

    recorded = training_env.trainer.fit_calls[0]['model'].kwargs['run_settings']
    cfg = training.Config.from_yaml('config.yaml')
    swept = utils_training.run_settings(
        cfg, accelerator='cuda', precision=utils_training.effective_precision(cfg, 'cuda')
    )
    assert recorded == swept
    assert type(recorded['logit_scale_range']) is list
    assert type(swept['logit_scale_range']) is list
    assert (recorded['accelerator'], recorded['precision']) == ('cuda', 'bf16-mixed')
    assert recorded['precision'] == training_env.trainer.kwargs['precision']
    assert json.loads(json.dumps(recorded)) == recorded

@pytest.mark.unit
def test_train_checkpoints_and_stops_on_the_outcome_mrr_through_create_trainer(
    training_env, monkeypatch, tmp_path
):
    '''P17: train's trainer is create_trainer's: the kept epoch and last.ckpt on
    val/outcome_mrr in the run's checkpoint directory, early stopping at the configured
    patience, the epoch callback, and no validation loop.'''

    calls = []
    real_create_trainer = training.create_trainer

    def spy(cfg, hardware, checkpoint_dir, **kwargs):
        calls.append(Path(checkpoint_dir))
        return real_create_trainer(cfg, hardware, checkpoint_dir, **kwargs)

    monkeypatch.setattr(training, 'create_trainer', spy)

    training.train(skip_validation=True, overrides=['training.early_stopping_patience=3'])

    assert calls == [_checkpoint_dir(tmp_path)]
    kwargs = training_env.trainer.kwargs
    [checkpointer] = [cb for cb in kwargs['callbacks'] if isinstance(cb, ModelCheckpoint)]
    [stopper] = [cb for cb in kwargs['callbacks'] if isinstance(cb, EarlyStopping)]
    assert any(isinstance(cb, TrainDatasetEpochCallback) for cb in kwargs['callbacks'])
    assert checkpointer.dirpath == os.path.realpath(_checkpoint_dir(tmp_path))
    assert (checkpointer.monitor, checkpointer.mode, checkpointer.save_top_k) == (
        OUTCOME_MRR, 'max', 1
    )
    assert (stopper.monitor, stopper.mode, stopper.patience) == (OUTCOME_MRR, 'max', 3)
    assert (kwargs['limit_val_batches'], kwargs['num_sanity_val_steps']) == (0, 0)
    assert 'val_check_interval' not in kwargs
    assert kwargs['accumulate_grad_batches'] == 1

@pytest.mark.unit
@pytest.mark.parametrize('entry', ['last.ckpt', MONITOR_RECORDS])
def test_train_refuses_a_fresh_start_into_a_used_checkpoint_directory(
    training_env, tmp_path, entry
):
    '''P19: refused before the datamodule, the monitor, the model or the trainer is built.'''

    _checkpoint_dir(tmp_path).mkdir(parents=True)
    (_checkpoint_dir(tmp_path) / entry).write_text('')

    with pytest.raises(typer.Exit) as excinfo:
        training.train(skip_validation=True)

    assert excinfo.value.exit_code == 1
    assert training_env.trainer is None
    assert 'datamodule' not in training_env.events
    assert 'monitor' not in training_env.events and 'model' not in training_env.events
    assert training_env.read_checkpoint_calls == []

@pytest.mark.unit
def test_train_starts_fresh_into_an_empty_checkpoint_directory(training_env, tmp_path):
    _checkpoint_dir(tmp_path).mkdir(parents=True)

    training.train(skip_validation=True)

    assert training_env.trainer.fit_calls[0]['ckpt_path'] is None

def _early_stopping_key() -> str:
    '''The key a checkpoint holds the run's EarlyStopping state under.'''

    stopper = utils_training.outcome_early_stopping(Config().training.early_stopping_patience)
    return stopper.state_key

def _another_directory(saved):
    for state in saved['callbacks'].values():
        if 'dirpath' in state:
            state['dirpath'] = '/elsewhere/checkpoints/cli-test'

def _other_settings(saved):
    saved['hyper_parameters']['run_settings']['learning_rate'] = 2e-4

def _another_seed(saved):
    saved['hyper_parameters']['seed'] = 7

def _no_checkpoint_state(saved):
    del saved['callbacks']

def _no_run_settings(saved):
    del saved['hyper_parameters']['run_settings']

def _no_early_stopping_state(saved):
    del saved['callbacks'][_early_stopping_key()]

@pytest.mark.unit
@pytest.mark.parametrize(
    'corrupt',
    [
        _another_directory,
        _other_settings,
        _another_seed,
        _no_checkpoint_state,
        _no_run_settings,
        _no_early_stopping_state,
    ],
    ids=[
        'another-directory',
        'other-settings',
        'another-seed',
        'no-checkpoint-state',
        'no-settings',
        'no-early-stopping-state',
    ],
)
def test_train_refuses_a_resume_it_cannot_continue(training_env, corrupt):
    '''P19: an exact resume from another checkpoint directory, under other run settings or another
    seed, or from a checkpoint with no EarlyStopping state, exits 1 after the contract check and
    before the datamodule, the monitor or the model.'''

    saved = training_env.matching_checkpoint()
    corrupt(saved)
    training_env.saved_checkpoint = saved
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    with pytest.raises(typer.Exit) as excinfo:
        training.train(ckpt_path='last', skip_validation=True)

    assert excinfo.value.exit_code == 1
    assert [path for path, _ in training_env.exact_resume_calls] == ['foo.ckpt']
    assert training_env.read_checkpoint_calls == ['foo.ckpt']
    assert training_env.trainer is None
    assert 'datamodule' not in training_env.events
    assert 'monitor' not in training_env.events and 'model' not in training_env.events

@pytest.mark.unit
def test_train_refuses_a_resume_of_a_run_early_stopping_ended(training_env):
    '''P19: Lightning restores early stopping's state but not its stop, so a finished run resumed
    from its last.ckpt, as the remote launch line always resumes, would train on and change its
    selection. Refused after the other resume guards, before the datamodule, the monitor or the
    model is built.'''

    saved = training_env.matching_checkpoint()
    saved['callbacks'][_early_stopping_key()]['stopped_epoch'] = 3
    training_env.saved_checkpoint = saved
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)

    with pytest.raises(typer.Exit) as excinfo:
        training.train(ckpt_path='last', skip_validation=True)

    assert excinfo.value.exit_code == 1
    # train exits 1 from the stopped-run guard's own refusal
    refusal = excinfo.value.__context__
    assert isinstance(refusal, ValueError)
    assert 'early stopping ended the run at epoch 3' in str(refusal)
    assert [path for path, _ in training_env.exact_resume_calls] == ['foo.ckpt']
    assert training_env.read_checkpoint_calls == ['foo.ckpt']
    assert training_env.trainer is None
    assert 'datamodule' not in training_env.events
    assert 'monitor' not in training_env.events and 'model' not in training_env.events

@pytest.mark.unit
@pytest.mark.parametrize('end', ['low', 'high'])
def test_a_config_at_the_edge_of_every_added_key_builds_the_model(
    tiny_backbone, reference_manifest, reference_bundle, end
):
    '''P22: a config that validates never fails at model construction, at the edges included.'''

    low, high = 0.01, 100.0
    cfg = Config().override(
        {
            'loss.code_code_weight': 0.0,
            'loss.radial_weight': 0.0,
            'loss.target_temperature': 1e-6,
            'loss.radial_step': 1e-6,
            'loss.logit_scale_init': low if end == 'low' else high,
            'loss.logit_scale_range': [low, high],
            'training.warmup_epochs': 0,
            'training.lr_plateau_factor': 1e-6 if end == 'low' else 1 - 1e-6,
            'training.lr_plateau_patience': 0,
            'training.early_stopping_patience': 1,
        }
    )
    cfg.supervision.manifest_path = str(reference_manifest)
    contract = training.runtime_contract_for(cfg, reference_bundle)
    settings = utils_training.run_settings(cfg, accelerator='cpu', precision='32-true')

    model = training.build_model_from_config(
        cfg, contract, reference_bundle, run_settings=settings, monitor=None
    )

    assert float(model.logit_scale_task().detach()) == pytest.approx(cfg.loss.logit_scale_init)
    assert model.hparams['lr_plateau_patience'] == 0
