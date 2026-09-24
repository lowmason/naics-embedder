import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from naics_embedder.cli.commands import data as data_cli
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
from tests.fixtures.regressor_panel import (
    CODEBOOK,
    HELDOUT_GROUPS,
    SETTINGS,
    coordinate_table,
    text_only_table,
)

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

def test_data_regressor_groups_draws_with_the_regressor_config(monkeypatch, runner, tmp_path):
    calls = []

    def fake_generate(cfg, codebook_path, force):
        calls.append((cfg, codebook_path, force))
        return tmp_path / 'regressor_heldout_groups.csv'

    monkeypatch.setattr(data_cli, 'generate_regressor_group_table', fake_generate)

    result = runner.invoke(
        data_cli.app, ['regressor-groups', '--codebook', '/bundle/naics_codebook.parquet']
    )

    assert result.exit_code == 0, result.output
    [(cfg, codebook_path, force)] = calls
    assert cfg.seed == 20260924
    assert cfg.branch_record is not None
    assert codebook_path == Path('/bundle/naics_codebook.parquet')
    assert force is False
    assert 'regressor_heldout_groups.csv' in result.output

def test_data_regressor_groups_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(cfg, codebook_path, force):
        raise FileExistsError('the table exists; pass --force only to redraw it deliberately')

    monkeypatch.setattr(data_cli, 'generate_regressor_group_table', refuse)

    result = runner.invoke(data_cli.app, ['regressor-groups', '--codebook', 'codebook.parquet'])

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

# -------------------------------------------------------------------------------------------------
# Outcome panel: lexical baseline
# -------------------------------------------------------------------------------------------------

BASELINE_ROLES = [
    (0, '111110', 'Soybean farming', 'examples'),
    (1, '111110', 'Edamame farming', 'validation'),
    (2, '111120', 'Canola farming', 'examples'),
    (3, '111120', 'Sunflower farming', 'validation'),
    (4, '111120', 'Rapeseed farming', 'test'),
]

def _baseline_inputs(tmp_path, examples):
    roles = tmp_path / 'naics_index_roles.parquet'
    descriptions = tmp_path / 'naics_descriptions.parquet'
    pl.DataFrame(
        BASELINE_ROLES,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    ).write_parquet(roles)
    pl.DataFrame(
        {
            'code': ['111110', '111120', '112130'],
            'title': ['Soybean Farming', 'Oilseed Farming', 'Dual-Purpose Cattle Ranching'],
            'description': ['Grows soybeans.', 'Grows oilseeds.', 'Raises cattle.'],
            'examples': examples,
            'excluded': [None, None, None],
        },
        schema_overrides={
            'excluded': pl.Utf8
        },
    ).write_parquet(descriptions)
    return [
        '--index-roles',
        str(roles),
        '--descriptions',
        str(descriptions),
        '--log',
        str(tmp_path / 'selection_log.jsonl'),
    ]

@pytest.mark.unit
def test_outcome_baseline_scores_validation_and_logs_one_read(runner, tmp_path):
    arguments = _baseline_inputs(tmp_path, ['Soybean farming', 'Canola farming', None])
    output = tmp_path / 'baseline.json'

    result = runner.invoke(tools_cli.app, ['outcome-baseline', *arguments, '--output', str(output)])

    assert result.exit_code == 0, result.output
    assert 'mrr' in result.output
    records = SelectionLog(tmp_path / 'selection_log.jsonl').records()
    assert [(r['event'], r['split'], r['n_queries'])
            for r in records] == [('read', 'validation', 2)]
    summary = json.loads(output.read_text())['summary']
    assert summary['n_queries'] == 2
    assert summary['n_candidates'] == 3

@pytest.mark.unit
def test_outcome_baseline_refuses_descriptions_that_hold_every_entry(runner, tmp_path):
    stale = [
        'Soybean farming; Edamame farming',
        'Canola farming; Sunflower farming; Rapeseed farming',
        None,
    ]

    result = runner.invoke(tools_cli.app, ['outcome-baseline', *_baseline_inputs(tmp_path, stale)])

    assert result.exit_code == 1
    assert 'Outcome baseline failed' in result.output
    assert SelectionLog(tmp_path / 'selection_log.jsonl').records() == []

# -------------------------------------------------------------------------------------------------
# Regressor panel
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_text_only_table_embeds_with_the_regressor_configs_backbone(monkeypatch, runner, tmp_path):
    calls = []

    def fake_build(descriptions, output, *, backbone, max_length, batch_size):
        calls.append((descriptions, output, backbone, max_length, batch_size))
        return output

    monkeypatch.setattr(tools_cli, 'build_text_only_table', fake_build)
    output = tmp_path / 'text_only.parquet'

    result = runner.invoke(
        tools_cli.app,
        ['text-only-table', '--descriptions', 'descriptions.parquet', '--output',
         str(output)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (Path('descriptions.parquet'), output, 'sentence-transformers/all-MiniLM-L6-v2', 512, 32)
    ]
    assert 'text_only_provenance.json' in result.output

def _regressor_arguments(tmp_path):
    coordinates = tmp_path / 'arm.parquet'
    text_only = tmp_path / 'text_only.parquet'
    coordinate_table(CODEBOOK).write_parquet(coordinates)
    text_only_table(CODEBOOK).write_parquet(text_only)
    return [
        'regressor-panel',
        '--coordinates',
        str(coordinates),
        '--text-only',
        str(text_only),
        '--codebook',
        'naics_codebook.parquet',
    ]

@pytest.fixture
def fixture_panel(monkeypatch, tmp_path, regressor_rows):
    '''The fixture panel in place of the real one: every opening below is a fixture's.'''

    log = SelectionLog(tmp_path / 'selection_log.jsonl')
    loaded = []

    def fake_load(cfg, codebook, *, log_path, levels):
        loaded.append((codebook, log_path, tuple(levels)))
        return RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    monkeypatch.setattr(tools_cli, 'load_regressor_panel', fake_load)
    return log, loaded

@pytest.mark.unit
def test_regressor_panel_needs_an_open_purpose_for_the_test_split(runner, tmp_path, fixture_panel):
    log, loaded = fixture_panel

    result = runner.invoke(tools_cli.app, [*_regressor_arguments(tmp_path), '--split', 'test'])

    assert result.exit_code == 1
    assert '--open-purpose' in result.output
    assert loaded == []
    assert log.records() == []

@pytest.mark.unit
def test_regressor_panel_scores_validation_and_reports_undefined_cells(
    runner, tmp_path, fixture_panel
):
    log, loaded = fixture_panel
    output = tmp_path / 'predictions.parquet'
    levels = ['--level', '6', '--level', '3', '--level', '2']

    result = runner.invoke(
        tools_cli.app, [*_regressor_arguments(tmp_path), *levels, '--output',
                        str(output)]
    )

    assert result.exit_code == 0, result.output
    assert loaded == [('naics_codebook.parquet', None, (2, 3, 6))]
    assert 'seen, level 2: undefined' in result.output
    assert 'heldout, level 3: undefined' in result.output
    assert [(r['event'], r['panel'], r['split'], r['detail']['level']) for r in log.records()] == [
        ('read', 'regressor_seen', 'validation', 3),
        ('read', 'regressor_seen', 'validation', 6),
        ('read', 'regressor_heldout', 'validation', 6),
    ]
    assert set(pl.read_parquet(output).get_column('split')) == {'validation'}

@pytest.mark.unit
def test_regressor_panel_opens_each_regime_once_before_its_test_read(
    runner, tmp_path, fixture_panel
):
    log, _ = fixture_panel
    arguments = [
        *_regressor_arguments(tmp_path), '--split', 'test', '--open-purpose', 'fixture opening'
    ]

    first = runner.invoke(tools_cli.app, arguments)
    second = runner.invoke(tools_cli.app, arguments)

    assert first.exit_code == 0, first.output
    assert [(r['event'], r['panel'], r['split']) for r in log.records()] == [
        ('open', 'regressor_seen', 'test'),
        ('read', 'regressor_seen', 'test'),
        ('open', 'regressor_heldout', 'test'),
        ('read', 'regressor_heldout', 'test'),
    ]
    assert second.exit_code == 1
    assert 'reopen_reason' in second.output
