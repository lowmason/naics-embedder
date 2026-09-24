from pathlib import Path

import pytest
from typer.testing import CliRunner

from naics_embedder.cli.commands import data as data_cli
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.supervision.artifacts import load_validated_bundle

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def no_logging(monkeypatch):
    monkeypatch.setattr(data_cli, 'configure_logging', lambda *_, **__: None)
    monkeypatch.setattr(tools_cli, 'configure_logging', lambda *_, **__: None)

def test_data_preprocess_invokes_download(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: calls.append((cfg, force))
    )

    result = runner.invoke(data_cli.app, ['preprocess'])

    assert result.exit_code == 0
    [(cfg, force)] = calls
    assert cfg.source_dir is None
    assert cfg.index_roles_csv == './conf/data/index_roles.csv'
    assert force is False

def test_data_preprocess_passes_source_dir_and_force(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: calls.append((cfg, force))
    )

    result = runner.invoke(data_cli.app, ['preprocess', '--source-dir', '/sources', '--force'])

    assert result.exit_code == 0
    assert [(cfg.source_dir, force) for cfg, force in calls] == [('/sources', True)]

def test_data_preprocess_reports_a_refused_overwrite(monkeypatch, runner):

    def refuse(cfg, force):
        raise FileExistsError('pinned; pass --force to overwrite it')

    monkeypatch.setattr(data_cli, 'download_preprocess_data', refuse)

    result = runner.invoke(data_cli.app, ['preprocess'])

    assert result.exit_code == 1
    assert '--force' in result.output

def test_data_all_runs_preprocess_then_one_supervision_build(monkeypatch, runner, tmp_path):
    order = []
    manifest = tmp_path / 'bundle' / 'manifest.json'
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: order.append('preprocess')
    )

    def fake_generate(cfg):
        order.append('supervision')
        return manifest

    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', fake_generate)

    result = runner.invoke(data_cli.app, ['all'])

    assert result.exit_code == 0
    assert order == ['preprocess', 'supervision']
    assert str(manifest) in result.output

def test_data_supervision_prints_the_manifest_path(monkeypatch, runner, tmp_path):
    manifest = tmp_path / 'bundle-id' / 'manifest.json'
    configs = []

    def fake_generate(cfg):
        configs.append(cfg)
        return manifest

    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', fake_generate)

    result = runner.invoke(data_cli.app, ['supervision'])

    assert result.exit_code == 0
    assert str(manifest) in result.output
    assert configs[0].contract_version == 'stage3-supervision-v1'
    assert configs[0].relation_id['cross_sector'] == 99

@pytest.mark.parametrize('command', ['relations', 'distances', 'triplets'])
def test_legacy_stage_commands_build_the_complete_bundle(monkeypatch, runner, tmp_path, command):
    manifest = tmp_path / 'bundle-id' / 'manifest.json'
    calls = []

    def fake_generate(cfg):
        calls.append(cfg)
        return manifest

    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', fake_generate)

    result = runner.invoke(data_cli.app, [command])

    assert result.exit_code == 0
    assert 'data supervision' in result.output
    assert len(calls) == 1
    assert str(manifest) in result.output

def test_data_roles_draws_the_table_with_both_configs(monkeypatch, runner, tmp_path):
    calls = []

    def fake_generate(download_cfg, panel_cfg, force):
        calls.append((download_cfg, panel_cfg, force))
        return tmp_path / 'index_roles.csv'

    monkeypatch.setattr(data_cli, 'generate_index_role_table', fake_generate)

    result = runner.invoke(data_cli.app, ['roles', '--source-dir', '/sources'])

    assert result.exit_code == 0
    [(download_cfg, panel_cfg, force)] = calls
    assert download_cfg.source_dir == '/sources'
    assert panel_cfg.seed == 20260924
    assert force is False
    assert 'index_roles.csv' in result.output

def test_data_roles_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(download_cfg, panel_cfg, force):
        raise FileExistsError('the role table exists; pass --force to redraw it')

    monkeypatch.setattr(data_cli, 'generate_index_role_table', refuse)

    result = runner.invoke(data_cli.app, ['roles'])

    assert result.exit_code == 1
    assert '--force' in result.output

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
    captured = {}
    monkeypatch.setattr(
        tools_cli, 'show_current_config', lambda cfg_path: captured.setdefault('path', cfg_path)
    )
    config_path = tmp_path / 'custom.yaml'
    config_path.write_text('foo: bar')

    result = runner.invoke(tools_cli.app, ['config', '--config', str(config_path)])

    assert result.exit_code == 0
    assert captured['path'] == str(config_path)

def test_tools_visualize_handles_exception(monkeypatch, runner):

    def boom(**_kwargs):
        raise RuntimeError('boom')

    monkeypatch.setattr(tools_cli, 'visualize_metrics', boom)

    result = runner.invoke(tools_cli.app, ['visualize'])

    assert result.exit_code == 1
    assert 'Error' in result.output

def test_tools_investigate_success(monkeypatch, runner):
    monkeypatch.setattr(
        tools_cli,
        'investigate_hierarchy',
        lambda **_: {
            'reason': 'ok',
            'suggestion': 'none'
        },
    )

    result = runner.invoke(tools_cli.app, ['investigate'])

    assert result.exit_code == 0
    assert 'Investigation complete' in result.output

def test_verify_stage4_failure_sets_exit_code(monkeypatch, runner):

    def fake_verify(*_args, **_kwargs):
        return {
            'pre': {
                'metric': 0.8
            },
            'post': {
                'metric': 0.7
            },
            'delta': {
                'metric': -0.1
            },
            'checks': {
                'cophenetic': False
            },
            'passed': False,
        }

    monkeypatch.setattr(tools_cli, 'verify_stage4', fake_verify)

    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code == 1
    assert 'Verification failed' in result.output or 'failed thresholds' in result.output

@pytest.mark.unit
@pytest.mark.parametrize('undefined', [False, True])
def test_verify_stage4_formats_versioned_spearman(monkeypatch, runner, undefined):
    key = 'structural_spearman_v1'
    value = None if undefined else 0.87831006565368
    delta = None if undefined else 0.125
    payload = {
        'pre': {
            key: value
        },
        'post': {
            key: value
        },
        'delta': {
            key: delta
        },
        'checks': {
            'cophenetic': True,
            'ndcg': True,
            'local_improvement': True
        },
        'passed': True,
    }
    monkeypatch.setattr(tools_cli, 'verify_stage4', lambda *_, **__: payload)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 0, result.output
    if undefined:
        assert result.output.count(f'{key}: N/A') == 3
    else:
        assert result.output.count(f'{key}: 0.8783') == 2
        assert f'{key}: +0.1250' in result.output
    assert 'spearman_correlation' not in result.output

@pytest.mark.unit
def test_verify_stage4_input_error_is_fatal(monkeypatch, runner):

    def invalid(*_args, **_kwargs):
        raise StructuralMetricInputError('structural-spearman-v1: invalid tree_distances')

    monkeypatch.setattr(tools_cli, 'verify_stage4', invalid)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 1
    assert 'Verification failed' in result.output
    assert 'invalid tree_distances' in result.output

@pytest.mark.unit
def test_verify_stage4_has_no_spearman_threshold_option(runner):
    result = runner.invoke(tools_cli.app, ['verify-stage4', '--help'])
    assert result.exit_code == 0
    assert '--max-spearman-drop' not in result.output
    assert '--min-spearman' not in result.output

@pytest.fixture
def verify_inputs(monkeypatch):
    '''The structural input paths verify-stage4 hands to verify_stage4.'''
    seen = {}

    def fake_verify(_stage3, _stage4, distance_matrix, relations, _cfg):
        seen.update(distance_matrix=distance_matrix, relations=relations)
        return {'pre': {}, 'post': {}, 'delta': {}, 'checks': {}, 'passed': True}

    monkeypatch.setattr(tools_cli, 'verify_stage4', fake_verify)
    return seen

@pytest.mark.unit
def test_verify_stage4_defaults_to_the_legacy_structural_files(runner, verify_inputs):
    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code == 0, result.output
    assert verify_inputs == {
        'distance_matrix': Path('./data/naics_distance_matrix.parquet'),
        'relations': Path('./data/naics_relations.parquet'),
    }

@pytest.mark.unit
def test_verify_stage4_reads_structure_from_its_supervision_bundle(
    runner, verify_inputs, generated_bundle
):
    result = runner.invoke(
        tools_cli.app, ['verify-stage4', '--supervision-manifest',
                        str(generated_bundle)]
    )

    bundle = load_validated_bundle(generated_bundle)
    assert result.exit_code == 0, result.output
    assert verify_inputs == {
        'distance_matrix': bundle.artifact_path('distance_matrix'),
        'relations': bundle.artifact_path('relations'),
    }

@pytest.mark.unit
def test_verify_stage4_rejects_relations_from_outside_its_bundle(
    runner, verify_inputs, generated_bundle, tmp_path
):
    result = runner.invoke(
        tools_cli.app,
        [
            'verify-stage4',
            '--supervision-manifest',
            str(generated_bundle),
            '--relations',
            str(tmp_path / 'naics_relations.parquet'),
        ],
    )

    assert result.exit_code == 1
    assert 'relations path does not belong' in result.output
    assert verify_inputs == {}
