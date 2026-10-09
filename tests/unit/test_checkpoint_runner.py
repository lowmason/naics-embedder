'''The trained-seed runner and the sweep's guards, on fixture files only (spec 4.6 and 5).'''

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch
from typer.testing import CliRunner

from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.decision.decide import check_arm
from naics_embedder.decision.records import ArmRecord, ArmSpec, read_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.export import code_token_config
from naics_embedder.text_model.monitor import MONITOR_RECORDS, read_monitor_records
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import outcome_checkpoint, read_checkpoint, run_settings
from tests.fixtures.checkpoint_runs import REVISION, cached_tiny_backbone
from tests.fixtures.regressor_panel import SETTINGS
from tests.fixtures.shared_encoder import (
    PRE_STAGE_7_REFUSAL,
    lightning_checkpoint,
    pre_stage_7_contract,
)

pytestmark = pytest.mark.unit

def _spec(cfg, *, revision=REVISION):
    return ArmSpec(
        name='reference',
        components=1,
        dimension=cfg.model.dimension,
        geometry='hyperbolic',
        backbone=cfg.model.base_model_name,
        backbone_revision=revision,
        descriptions_sha256=sha256_file(cfg.data_loader.streaming.descriptions_parquet),
        summaries_sha256=summaries_identity(cfg.data_loader.tokenization.tokenizer_name),
        max_length=cfg.data_loader.streaming.max_length,
        settings=run_settings(cfg, accelerator='cpu', precision='32-true')
    )

def _write_records(directory, records):
    (directory / MONITOR_RECORDS).write_text(
        ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records)
    )

@pytest.fixture
def fixture_run(tmp_path, shared_model, validated_bundle):
    cfg = Config().override(
        {
            'model.dimension': shared_model.hparams.dimension,
            'model.lora.r': shared_model.hparams.lora_r,
            'model.lora.alpha': shared_model.hparams.lora_alpha,
            'model.lora.dropout': shared_model.hparams.lora_dropout,
        }
    )
    # _spec's descriptions hash is a file identity; this fixture never exports a table.
    cfg.data_loader.streaming.descriptions_parquet = str(tmp_path / 'descriptions.parquet')
    Path(cfg.data_loader.streaming.descriptions_parquet).write_bytes(b'fixture-descriptions')
    spec = _spec(cfg)
    directory = tmp_path / 'seed-7'
    directory.mkdir()
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint.update(epoch=2, training_run='training-7')
    checkpoint['hyper_parameters'].update(seed=7, run_settings=spec.settings)
    key = outcome_checkpoint(directory).state_key
    checkpoint['callbacks'] = {key: {'best_model_score': torch.tensor(0.7, dtype=torch.float64)}}
    torch.save(checkpoint, directory / 'last.ckpt')
    checkpoint['epoch'] = 1
    torch.save(checkpoint, directory / 'epoch=001.ckpt')
    records = [
        {
            'mrr': mrr,
            'read': {
                'event': 'read',
                'panel': 'outcome',
                'split': 'validation',
                'fingerprint': 'fixture',
                'detail': {
                    'epoch': epoch,
                    'seed': 7,
                    'training_run': 'training-7',
                    'table': 'cache'
                },
            }
        } for epoch, mrr in enumerate((0.4, 0.7, 0.7))
    ]
    _write_records(directory, records)
    return SimpleNamespace(
        cfg=cfg, spec=spec, directory=directory, bundle=validated_bundle, records=records, key=key
    )

def _runner(fixture):
    module = importlib.import_module('naics_embedder.text_model.checkpoint_runner')
    return module.CheckpointRunner(
        fixture.cfg, fixture.bundle, run_directory=lambda seed: fixture.directory, device='cpu'
    )

def test_check_selects_the_earliest_tied_best_epoch_without_export(fixture_run, monkeypatch):
    module = importlib.import_module('naics_embedder.text_model.checkpoint_runner')
    monkeypatch.setattr(module, 'export_code_table', lambda *args, **kwargs: pytest.fail('export'))
    # Record order cannot act as the epoch: the earliest tied best is last in this list.
    _write_records(fixture_run.directory, [fixture_run.records[i] for i in (2, 0, 1)])
    selected = _runner(fixture_run).check(fixture_run.spec, 7)
    assert selected.epoch == 1
    assert selected.checkpoint == fixture_run.directory / 'epoch=001.ckpt'
    assert selected.training_run == 'training-7'
    assert selected.mrr == 0.7
    assert len(selected.monitor_records) == 3
    assert not list(fixture_run.directory.glob('*.parquet'))

@pytest.mark.parametrize(
    'problem', [
        'no-records-file',
        'empty-records',
        'missing-epoch',
        'repeated-epoch',
        'extra-epoch',
        'no-last',
        'invalid-last-epoch',
        'missing-selected',
        'version-sibling',
        'selected-epoch',
        'selected-run',
        'last-run',
        'records-run',
        'selected-seed',
        'selected-settings',
        'last-seed',
        'last-settings',
        'missing-score',
        'wrong-score',
        'no-training-run',
        'contract-bundle',
        'contract-encoder',
        'old-objective',
    ]
)
def test_check_refuses_an_incomplete_or_inconsistent_seed(fixture_run, problem):
    directory = fixture_run.directory
    selected = directory / 'epoch=001.ckpt'
    records = fixture_run.records
    if problem == 'no-records-file':
        (directory / MONITOR_RECORDS).unlink()
    elif problem == 'empty-records':
        _write_records(directory, [])
    elif problem == 'missing-epoch':
        _write_records(directory, records[1:])
    elif problem == 'repeated-epoch':
        _write_records(directory, records + [records[0]])
    elif problem == 'extra-epoch':
        extra = json.loads(json.dumps(records[0]))
        extra['read']['detail']['epoch'] = 3
        _write_records(directory, records + [extra])
    elif problem == 'no-last':
        (directory / 'last.ckpt').unlink()
    elif problem == 'missing-selected':
        selected.unlink()
    elif problem == 'version-sibling':
        shutil.copyfile(selected, directory / 'epoch=001-v1.ckpt')
    elif problem == 'records-run':
        records[0]['read']['detail']['training_run'] = 'other-training'
        _write_records(directory, records)
    else:
        path = directory / 'last.ckpt' if problem.startswith('last-') or problem in (
            'invalid-last-epoch', 'no-training-run'
        ) else selected
        saved = read_checkpoint(path)
        if problem in ('selected-run', 'last-run'):
            saved['training_run'] = 'other-training'
        elif problem == 'no-training-run':
            saved.pop('training_run')
        elif problem in ('selected-seed', 'last-seed'):
            saved['hyper_parameters']['seed'] = 8
        elif problem in ('selected-settings', 'last-settings'):
            saved['hyper_parameters']['run_settings']['radius_bound'] = 9.0
        elif problem == 'selected-epoch':
            saved['epoch'] = 2
        elif problem == 'invalid-last-epoch':
            saved['epoch'] = True
            # True would otherwise pass as epoch 1; leave its coverage valid to isolate the type.
            _write_records(directory, records[:2])
        elif problem == 'missing-score':
            saved['callbacks'][fixture_run.key].pop('best_model_score')
        elif problem == 'wrong-score':
            saved['callbacks'][fixture_run.key]['best_model_score'] = torch.tensor(
                0.7 + 1e-10, dtype=torch.float64
            )
        elif problem == 'contract-bundle':
            saved['stage3_supervision']['bundle_id'] = 'other-bundle'
        elif problem == 'contract-encoder':
            saved['stage3_supervision']['encoder']['dimension'] = 8
        elif problem == 'old-objective':
            saved['stage3_supervision'] = pre_stage_7_contract(saved['stage3_supervision'])
        torch.save(saved, path)
    with pytest.raises(
        ValueError,
        match=(
            PRE_STAGE_7_REFUSAL if problem == 'old-objective' else 'non-negative integer epoch'
            if problem == 'invalid-last-epoch' else '.'
        )
    ):
        _runner(fixture_run).check(fixture_run.spec, 7)

def test_check_refuses_a_seed_of_another_geometry(fixture_run):
    '''P7: the arm's geometry is its spec's, and a checkpoint of another arm's head is refused.'''

    spec = fixture_run.spec.model_copy(update={'geometry': 'euclidean'})

    with pytest.raises(ValueError, match='seed 7: the checkpoint encoder contract differs'):
        _runner(fixture_run).check(spec, 7)

def test_check_reads_a_seed_saved_before_stage_8_as_hyperbolic(fixture_run):
    '''P7: Stage 7's checkpoints name no geometry, and the reference arm reads them unchanged.'''

    for name in ('last.ckpt', 'epoch=001.ckpt'):
        path = fixture_run.directory / name
        saved = read_checkpoint(path)
        del saved['stage3_supervision']['encoder']['geometry']
        del saved['hyper_parameters']['geometry']
        torch.save(saved, path)

    assert _runner(fixture_run).check(fixture_run.spec, 7).epoch == 1

@pytest.fixture
def sweep_env(tmp_path, monkeypatch, trained_seeds, regressor_rows):
    root = tmp_path / 'runs'
    for seed in trained_seeds.seeds:
        shutil.copytree(trained_seeds.directory(seed), root / f'seed-{seed}')
    cfg = trained_seeds.cfg
    monkeypatch.setattr(
        'naics_embedder.text_model.shared_encoder.load_base_model', cached_tiny_backbone
    )
    monkeypatch.setattr(tools_cli, 'configure_logging', lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_cli.Config, 'from_yaml', classmethod(lambda cls, path: cfg))
    monkeypatch.setattr(tools_cli, 'pick_device', lambda value: 'cpu')
    monkeypatch.setattr(
        tools_cli,
        'load_backbone',
        lambda name: (cached_tiny_backbone(name), None, REVISION),
        raising=False
    )
    # The runner's export must share the cache training used, under pytest's temp root.
    token_config = code_token_config(cfg).model_copy(
        update={'output_path': str(trained_seeds.root / 'tokens' / 'cache.pt')}
    )
    monkeypatch.setattr(
        'naics_embedder.text_model.export.code_token_config', lambda cfg: token_config
    )
    monkeypatch.setattr(tools_cli, 'code_token_config', lambda cfg: token_config)
    module_name = 'naics_embedder.text_model.checkpoint_runner'
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass
    else:
        monkeypatch.setattr(module, 'code_token_config', lambda cfg: token_config)
    codes = ['311111', '311211', '321111', '441111']
    originals = ['111111', '111211', '321111', '311111']
    rows = regressor_rows[6].filter(pl.col('code').is_in(originals)).with_columns(
        pl.col('code').replace_strict(dict(zip(originals, codes)))
    )
    log = tmp_path / 'selection_log.jsonl'
    monkeypatch.setattr(
        tools_cli, 'load_regressor_panel', lambda cfg, codebook, **kwargs: RegressorPanel(
            {6: rows}, (), SelectionLog(Path(kwargs['log_path'])), SETTINGS
        )
    )
    args = [
        'sweep', '--runs',
        str(root / 'seed-{seed}'), '--text-only',
        str(trained_seeds.text_only), '--store',
        str(tmp_path / 'store'), '--output',
        str(tmp_path / 'arm.json'), '--purpose', 'fixture sweep', '--accelerator', 'cpu', '--log',
        str(log)
    ]
    for seed in trained_seeds.seeds:
        args.extend(['--seed', str(seed)])
    return SimpleNamespace(args=args, root=root, log=log, output=tmp_path / 'arm.json', cfg=cfg)

def test_tools_sweep_reads_five_trained_seeds_and_writes_a_checked_arm(sweep_env):
    # A training cache encoded on another device need not fingerprint as the CPU export does.
    for seed in range(1, 6):
        directory = sweep_env.root / f'seed-{seed}'
        records = read_monitor_records(directory / MONITOR_RECORDS)
        for record in records:
            record['read']['detail']['table'] = 'cache-from-another-device'
        _write_records(directory, records)
    result = CliRunner().invoke(tools_cli.app, sweep_env.args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    arm = read_record(sweep_env.output, ArmRecord)
    assert arm.spec.settings == run_settings(sweep_env.cfg, accelerator='cpu', precision='32-true')
    assert arm.spec.backbone_revision == REVISION
    assert len(arm.runs) == 5
    assert len(SelectionLog(sweep_env.log).records()) == 15
    for run in arm.runs:
        records = read_monitor_records(sweep_env.root / f'seed-{run.seed}' / MONITOR_RECORDS)
        assert run.monitor_records == records
        assert run.training_run == records[0]['read']['detail']['training_run']
        assert run.checkpoint_epoch == min(
            record['read']['detail']['epoch'] for record in records
            if record['mrr'] == max(item['mrr'] for item in records)
        )
    check_arm(arm, ArtifactStore(sweep_env.output.parent / 'store'), min_seeds=5)

@pytest.mark.parametrize(
    'problem', ['missing-epoch', 'fingerprint', 'seed', 'split', 'training_run']
)
def test_tools_sweep_checks_a_later_seed_before_the_first_decision_read(sweep_env, problem):
    directory = sweep_env.root / 'seed-5'
    records = read_monitor_records(directory / MONITOR_RECORDS)
    if problem == 'missing-epoch':
        records = records[1:]
    elif problem in ('seed', 'training_run'):
        records[0]['read']['detail'][problem] = 999 if problem == 'seed' else 'other-training'
    else:
        records[0]['read'][problem] = 'other-panel' if problem == 'fingerprint' else 'test'
    _write_records(directory, records)
    result = CliRunner().invoke(tools_cli.app, sweep_env.args)
    assert result.exit_code == 1, result.output
    assert 'seed 5' in result.output
    assert SelectionLog(sweep_env.log).records() == []
    assert not sweep_env.output.exists()
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_tools_sweep_refuses_seeds_of_another_geometry_before_the_first_read(sweep_env):
    '''P7: the sweep's config names the arm's geometry, and every seed is checked against it.'''

    result = CliRunner().invoke(tools_cli.app, [*sweep_env.args, 'model.geometry=euclidean'])

    assert result.exit_code == 1, result.output
    output = result.output.replace('\n', '')
    assert 'seed 1' in output and 'encoder contract differs' in output
    assert SelectionLog(sweep_env.log).records() == []
    assert not sweep_env.output.exists()
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_tools_sweep_refuses_a_text_only_revision_from_another_backbone(sweep_env, tmp_path):
    args = list(sweep_env.args)
    original = Path(args[args.index('--text-only') + 1])
    table = tmp_path / 'other-text.parquet'
    shutil.copyfile(original, table)
    provenance = json.loads(provenance_path(original).read_text())
    provenance['revision'] = 'another-revision'
    provenance_path(table).write_text(json.dumps(provenance))
    args[args.index('--text-only') + 1] = str(table)
    result = CliRunner().invoke(tools_cli.app, args)
    assert result.exit_code == 1
    assert 'D9' in result.output
    assert SelectionLog(sweep_env.log).records() == []
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_runner_reexports_an_existing_table_and_provenance(sweep_env, trained_seeds):
    fixture = SimpleNamespace(
        cfg=sweep_env.cfg, bundle=trained_seeds.bundle, directory=sweep_env.root / 'seed-1'
    )
    runner = _runner(fixture)
    first = runner.run(_spec(sweep_env.cfg), 1)
    assert first.encoder.distance == first.distance == 'lorentz'
    assert first.table.name == f'arm_table_epoch={first.checkpoint_epoch:03d}.parquet'
    first.table.write_bytes(b'stale-table')
    provenance_path(first.table).write_text('{}')
    second = runner.run(_spec(sweep_env.cfg), 1)
    assert pl.read_parquet(second.table).height == 17
    assert json.loads(provenance_path(second.table).read_text())['checkpoint']['sha256'] == (
        sha256_file(second.checkpoint)
    )
    assert second.monitor_records == first.monitor_records

@pytest.mark.parametrize(
    'problem', ['output-exists', 'bad-pattern', 'duplicate-seed', 'blank-purpose']
)
def test_tools_sweep_refuses_bad_arguments_before_loading_a_panel(sweep_env, monkeypatch, problem):
    args = list(sweep_env.args)
    if problem == 'output-exists':
        sweep_env.output.write_text('existing')
    elif problem == 'bad-pattern':
        args[args.index('--runs') + 1] = str(sweep_env.root)
    elif problem == 'duplicate-seed':
        args.extend(['--seed', '1'])
    else:
        args[args.index('--purpose') + 1] = ' '
    monkeypatch.setattr(tools_cli, '_run_bundle', lambda cfg: pytest.fail('loaded a bundle'))
    result = CliRunner().invoke(tools_cli.app, args)
    assert result.exit_code == 1
    assert SelectionLog(sweep_env.log).records() == []

@pytest.mark.parametrize('checkpoint_name', ['last.ckpt', 'epoch=001.ckpt'])
@pytest.mark.parametrize('name, value', [('lora_alpha', 32), ('lora_dropout', 0.6)])
def test_preflight_refuses_constructor_changes_in_both_checkpoints(
    fixture_run, monkeypatch, checkpoint_name, name, value
):
    module = importlib.import_module('naics_embedder.text_model.checkpoint_runner')
    monkeypatch.setattr(module, 'export_code_table', lambda *args, **kwargs: pytest.fail('export'))
    path = fixture_run.directory / checkpoint_name
    saved = read_checkpoint(path)
    saved['hyper_parameters'][name] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match=name):
        _runner(fixture_run).run(fixture_run.spec, 7)
    assert not list(fixture_run.directory.glob('*.parquet'))

@pytest.mark.parametrize('checkpoint_name', ['last.ckpt', 'selected'])
def test_all_seed_preflight_refuses_constructor_changes_before_any_export_or_panel(
    sweep_env, monkeypatch, checkpoint_name
):
    directory = sweep_env.root / 'seed-5'
    if checkpoint_name == 'selected':
        records = read_monitor_records(directory / MONITOR_RECORDS)
        best = max(record['mrr'] for record in records)
        epoch = min(
            record['read']['detail']['epoch'] for record in records if record['mrr'] == best
        )
        checkpoint_name = f'epoch={epoch:03d}.ckpt'
    path = directory / checkpoint_name
    saved = read_checkpoint(path)
    saved['hyper_parameters']['lora_alpha'] *= 2
    torch.save(saved, path)
    module = importlib.import_module('naics_embedder.text_model.checkpoint_runner')
    monkeypatch.setattr(module, 'export_code_table', lambda *args, **kwargs: pytest.fail('export'))
    monkeypatch.setattr(
        tools_cli, 'load_regressor_panel', lambda *args, **kwargs: pytest.fail('panel')
    )
    result = CliRunner().invoke(tools_cli.app, sweep_env.args)
    assert result.exit_code == 1, result.output
    assert 'seed 5' in result.output and 'lora_alpha' in result.output
    assert SelectionLog(sweep_env.log).records() == []
    assert not sweep_env.output.exists()
    assert not list(sweep_env.root.glob('**/*.parquet'))
