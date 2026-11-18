import json
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

import click
import numpy as np
import polars as pl
import pytest
import torch
import typer
import yaml
from typer.testing import CliRunner

from naics_embedder.cli.commands import data as data_cli
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.decision.candidates import (
    CandidateEntry,
    CandidateList,
    candidate_table_path,
    economic_spec,
)
from naics_embedder.decision.economic import (
    GROWTH_PANELS,
    check_economic_arm,
    read_deterministic_arm,
)
from naics_embedder.decision.records import (
    ArmRecord,
    DecisionRecord,
    EconomicArmRecord,
    EconomicDecisionRecord,
    MarginRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.store import ArtifactStore, provenance_fields
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import Regime, RegressorPanel, reference_comparators
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import SummariesPin
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model.builders import build_candidate
from naics_embedder.utils.config import Config
from tests.fixtures.decision import spec, synthetic_arm, write_text_only
from tests.fixtures.economic import (
    candidate_spec,
    fixture_list,
    scaled_arm,
    text_only_pins,
    trained_source,
    write_candidate,
    write_candidate_list,
    write_codebook,
)
from tests.fixtures.regressor_panel import (
    CODEBOOK,
    HELDOUT_GROUPS,
    POPULATION,
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
    # The machine path goes in the gitignored override, never the tracked config (audit row 1)
    assert 'config.supervision.manifest_path in conf/local.yaml' in result.output
    assert configs[0].contract_version == 'stage3-supervision-v2'
    assert configs[0].relation_id['cross_sector'] == 99

@pytest.mark.parametrize('command', ['relations', 'distances', 'triplets'])
def test_legacy_stage_commands_build_nothing(monkeypatch, runner, command):
    calls = []
    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', calls.append)

    result = runner.invoke(data_cli.app, [command])

    # A script running the old three-step sequence stops at its first step
    assert result.exit_code == 1
    assert 'naics-embedder data supervision' in result.output.replace('\n', ' ')
    assert calls == []

def test_the_data_module_docstring_says_the_legacy_stages_build_nothing():
    '''The CLI API page renders this docstring, so it describes the stubs as they behave.'''

    doc = ' '.join(data_cli.__doc__.split())
    legacy = doc[doc.index('relations, distances, triplets:'):doc.index('all:')]

    assert 'builds nothing' in legacy
    assert 'supervision bundle' not in legacy

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

def test_data_summaries_builds_for_the_training_configs_backbone(monkeypatch, runner):
    calls = []

    def fake_generate(descriptions, output, *, backbone, force):
        calls.append((descriptions, output, backbone, force))
        return SummariesPin(path=str(output), sha256='a' * 64, window=128)

    monkeypatch.setattr(data_cli, 'generate_window_summaries', fake_generate)

    result = runner.invoke(data_cli.app, ['summaries'])

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            Path('./data/naics_descriptions.parquet'),
            Path('conf/data/window_summaries.csv'),
            'sentence-transformers/all-MiniLM-L6-v2',
            False,
        )
    ]
    assert f"sha256='{'a' * 64}'" in result.output.replace('\n', '')

def test_data_summaries_refuses_to_rebuild_without_force(monkeypatch, runner):

    def refuse(descriptions, output, *, backbone, force):
        raise FileExistsError('the summaries exist; pass --force only to rebuild them')

    monkeypatch.setattr(data_cli, 'generate_window_summaries', refuse)

    result = runner.invoke(data_cli.app, ['summaries', '--backbone', 'other/backbone'])

    assert result.exit_code == 1
    assert '--force' in result.output.replace('\n', '')

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

    result = runner.invoke(tools_cli.app, ['visualize', '--summary', 'fixture-summary.jsonl'])

    assert result.exit_code == 1
    assert 'Error' in result.output

def test_tools_investigate_is_gone(runner):
    '''Spec 4.4: Req 6's statistics come only from tools diagnostics, so investigate is retired.'''

    # Checked before invoking, so the command never runs while it still exists
    assert 'investigate' not in typer.main.get_command(tools_cli.app).commands
    assert not hasattr(tools_cli, 'investigate_hierarchy')

    result = runner.invoke(tools_cli.app, ['investigate'])

    # Click's usage error: no such command
    assert result.exit_code == 2
    assert "No such command 'investigate'" in click.unstyle(result.output).replace('\n', '')

# -------------------------------------------------------------------------------------------------
# Decisions (Req 5) and diagnostics (Req 6)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def decision_inputs(tmp_path):
    '''A synthetic reference arm's record, written, and the store it references.'''

    store = ArtifactStore(tmp_path / 'store')
    reference = write_record(
        synthetic_arm(store, tmp_path, spec('reference'), {}), tmp_path / 'reference.json'
    )
    return store, reference

def _margins(runner, tmp_path, store, reference):
    return runner.invoke(
        tools_cli.app,
        [
            'margins',
            '--reference',
            str(reference),
            '--multiple',
            '2',
            '--name',
            'fixture margins',
            '--store',
            str(store.root),
            '--output',
            str(tmp_path / 'margins.json'),
        ],
    )

def _decide(runner, tmp_path, store, arms, candidates=None):
    arguments = ['decide']
    for path in arms:
        arguments += ['--arm', str(path)]
    arguments += [
        '--margins',
        str(tmp_path / 'margins.json'),
        '--name',
        'fixture decision',
        '--question',
        'which arm?',
        '--store',
        str(store.root),
        '--output',
        str(tmp_path / 'decision.json'),
    ]
    if candidates is not None:
        arguments += ['--candidates', str(candidates)]
    return runner.invoke(tools_cli.app, arguments)

@pytest.mark.unit
def test_margins_writes_each_panels_margin(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs

    result = _margins(runner, tmp_path, store, reference)

    assert result.exit_code == 0, result.output
    record = read_record(tmp_path / 'margins.json', MarginRecord)
    panels = [entry.panel for entry in record.margins]
    assert panels == ['outcome', 'regressor_seen', 'regressor_heldout']
    assert 'regressor_heldout: δ' in result.output.replace('\n', '')

@pytest.mark.unit
def test_margins_reports_a_missing_reference(runner, tmp_path, decision_inputs):
    store, _ = decision_inputs

    result = _margins(runner, tmp_path, store, tmp_path / 'missing.json')

    assert result.exit_code == 1
    assert 'Margins failed' in result.output
    assert not (tmp_path / 'margins.json').exists()

@pytest.mark.unit
def test_margins_refuses_an_existing_record_before_any_work(
    runner, tmp_path, decision_inputs, monkeypatch
):
    store, reference = decision_inputs
    (tmp_path / 'margins.json').write_text('an earlier record')
    calls = []
    monkeypatch.setattr(tools_cli, 'fix_margins', lambda *args, **kwargs: calls.append(args))

    result = _margins(runner, tmp_path, store, reference)

    assert calls == []
    assert result.exit_code == 1
    assert 'exists; a record is written once' in ' '.join(result.output.split())
    assert (tmp_path / 'margins.json').read_text() == 'an earlier record'

@pytest.mark.unit
def test_decide_refuses_an_existing_record_before_any_work(
    runner, tmp_path, decision_inputs, monkeypatch
):
    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    (tmp_path / 'decision.json').write_text('an earlier record')
    calls = []
    monkeypatch.setattr(tools_cli, 'decide', lambda *args, **kwargs: calls.append(args))

    result = _decide(runner, tmp_path, store, [reference, reference])

    assert calls == []
    assert result.exit_code == 1
    assert 'exists; a record is written once' in ' '.join(result.output.split())
    assert (tmp_path / 'decision.json').read_text() == 'an earlier record'

@pytest.mark.unit
def test_decide_adopts_the_better_arm_and_writes_the_record(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    better = write_record(
        synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5}),
        tmp_path / 'better.json',
    )

    result = _decide(runner, tmp_path, store, [better, reference])

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    assert 'better over reference: adopted' in output
    # Three panels still read 98⅓ % (Req 24); Rich may wrap the line at any space
    assert '98.33 % [' in ' '.join(result.output.split())
    assert 'Chosen: better' in output
    assert read_record(tmp_path / 'decision.json', DecisionRecord).chosen == 'better'

@pytest.mark.unit
def test_decide_refuses_candidates_with_d8_margins(runner, tmp_path, decision_inputs):
    '''--candidates names the economic decision's frozen list, which a D8 decision never reads.'''

    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    better = write_record(
        synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5}),
        tmp_path / 'better.json',
    )
    listed = tmp_path / 'candidates.yaml'
    listed.write_text('candidates: []\n')

    result = _decide(runner, tmp_path, store, [better, reference], candidates=listed)

    assert result.exit_code == 1
    output = ' '.join(click.unstyle(result.output).split())
    assert 'applies only to the economic decision' in output
    assert not (tmp_path / 'decision.json').exists()

@pytest.mark.unit
def test_decide_reports_a_tie_it_cannot_break(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    twin = write_record(synthetic_arm(store, tmp_path, spec('twin'), {}), tmp_path / 'twin.json')

    result = _decide(runner, tmp_path, store, [twin, reference])

    assert result.exit_code == 1
    assert 'Decision failed' in result.output
    assert 'tie on components' in ' '.join(result.output.split())
    assert not (tmp_path / 'decision.json').exists()

@pytest.fixture
def growth_panel(monkeypatch, tmp_path, regressor_growth_rows):
    '''The fixture growth panel in place of the real one, with the outcome each load asked for.'''

    log = SelectionLog(tmp_path / 'selection_log.jsonl')
    panel = RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )
    loaded = []

    def fake_load(cfg, codebook, *, log_path, levels):
        loaded.append((cfg.outcome, codebook, log_path, tuple(levels)))
        return panel

    monkeypatch.setattr(tools_cli, 'load_regressor_panel', fake_load)
    return panel, loaded

def _growth_margins(runner, tmp_path, store):
    lock = tmp_path / 'uv.lock'
    lock.write_text('a fixture lock\n')
    return runner.invoke(
        tools_cli.app,
        [
            'growth-margins',
            '--codebook',
            'naics_codebook.parquet',
            '--name',
            'growth margins',
            '--store',
            str(store.root),
            '--lock',
            str(lock),
            '--output',
            str(tmp_path / 'margins.json'),
        ],
    )

@pytest.mark.unit
def test_growth_margins_reads_the_reference_alone_and_writes_the_margins(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel

    result = _growth_margins(runner, tmp_path, ArtifactStore(tmp_path / 'store'))

    assert result.exit_code == 0, result.output
    # The committed config reads level mode; the margin read reads growth in memory
    assert loaded == [('growth', 'naics_codebook.parquet', None, (6, ))]
    record = read_record(tmp_path / 'margins.json', RelativeMarginRecord)
    assert record.r == 0.05
    assert [entry.panel for entry in record.margins] == ['growth_seen', 'growth_heldout']
    assert [read['detail']['comparators'] for read in panel.log.records()] == [
        list(reference_comparators(regime, 6)) for regime in Regime
    ]
    assert {read['purpose'] for read in panel.log.records()} == {'economic margin read'}
    output = result.output.replace('\n', '')
    assert 'growth_heldout: δ' in output
    assert record.read.environment.lock_sha256 in output

@pytest.mark.unit
def test_growth_margins_refuses_an_existing_record_before_anything_is_read(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel
    (tmp_path / 'margins.json').write_text('an earlier record')

    result = _growth_margins(runner, tmp_path, ArtifactStore(tmp_path / 'store'))

    assert result.exit_code == 1
    assert 'exists; a record is written once' in ' '.join(result.output.split())
    assert (loaded, panel.log.records()) == ([], [])
    assert (tmp_path / 'margins.json').read_text() == 'an earlier record'

def _economic_candidate(tmp_path, panel, store):
    '''The fixture tree candidate read after the growth margins, and its text-only table.'''

    table = write_candidate(tmp_path / 'candidates', 'tree', coordinate_table(CODEBOOK))
    text_only = write_text_only(tmp_path / 'text', codes=CODEBOOK)
    candidate = read_deterministic_arm(
        candidate_spec('tree'),
        table,
        regressor_panel=panel,
        text_only_table=text_only,
        margins=read_record(tmp_path / 'margins.json', RelativeMarginRecord),
        store=store,
        purpose='fixture candidate read',
        lock=tmp_path / 'uv.lock',
    )
    return candidate, text_only

@pytest.mark.unit
def test_decide_reads_an_economic_decision_from_relative_margins(runner, tmp_path, growth_panel):
    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')
    assert _growth_margins(runner, tmp_path, store).exit_code == 0
    candidate, text_only = _economic_candidate(tmp_path, panel, store)
    listed = write_candidate_list(
        tmp_path / 'candidates.yaml', [candidate], store, **text_only_pins(text_only)
    )

    result = _decide(
        runner,
        tmp_path,
        store, [write_record(candidate, tmp_path / 'tree.json')],
        candidates=listed
    )

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    assert 'tree over covariates+ancestors: not adopted' in output
    assert '97.5 % [' in ' '.join(result.output.split())
    assert 'Chosen: none; the reference covariates+ancestors stands' in output
    assert read_record(tmp_path / 'decision.json', EconomicDecisionRecord).chosen is None

@pytest.mark.unit
def test_decide_reads_the_committed_candidate_list_by_default(
    runner, tmp_path, growth_panel, monkeypatch
):
    '''With no --candidates, the economic decision reads conf/data/economic_candidates.yaml.'''

    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')
    assert _growth_margins(runner, tmp_path, store).exit_code == 0
    candidate, _ = _economic_candidate(tmp_path, panel, store)
    arm = write_record(candidate, tmp_path / 'tree.json')
    # The committed list is relative to the repository root, where the commands run
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    result = _decide(runner, tmp_path, store, [arm])

    assert result.exit_code == 1
    output = ' '.join(click.unstyle(result.output).split())
    assert "'tree' is not on the frozen candidate list" in output
    assert not (tmp_path / 'decision.json').exists()

def _built_candidate(tmp_path, panel, store, kind, dimension, factor, codebook, text_only):
    '''
    A candidate built from the fixture codebook, read after the growth margins, its errors
    ``factor`` times the reference's on both growth panels. It reads a backbone unless it is the
    tree candidate (P7).
    '''

    name = f'{kind}-d{dimension}'
    reads_text = kind != 'tree'
    table = build_candidate(
        kind,
        dimension,
        codebook,
        tmp_path / 'built' / f'{name}.parquet',
        text_only=text_only if reads_text else None,
    )
    if reads_text:
        arm_spec = spec(name, dimension=dimension, geometry='euclidean')
    else:
        arm_spec = candidate_spec(name, dimension=dimension)
    arm = read_deterministic_arm(
        arm_spec,
        table,
        regressor_panel=panel,
        text_only_table=text_only,
        margins=read_record(tmp_path / 'margins.json', RelativeMarginRecord),
        store=store,
        purpose='fixture candidate read',
        lock=tmp_path / 'uv.lock',
    )
    return scaled_arm(arm, dict.fromkeys(GROWTH_PANELS, factor), store)

def _decide_built(runner, tmp_path, panel, store, candidates):
    '''
    ``tools decide`` over built candidates, ``(kind, dimension, factor)`` in the order of their
    frozen list, which is the order of their ``--arm`` options.
    '''

    assert _growth_margins(runner, tmp_path, store).exit_code == 0
    codebook = write_codebook(tmp_path / 'codebook' / 'naics_codebook.parquet')
    text_only = write_text_only(tmp_path / 'text', codes=CODEBOOK)
    arms = [
        _built_candidate(tmp_path, panel, store, kind, dimension, factor, codebook, text_only)
        for kind, dimension, factor in candidates
    ]
    listed = write_candidate_list(
        tmp_path / 'candidates.yaml',
        arms,
        store,
        **text_only_pins(text_only),
        codebook_sha256=sha256_file(codebook),
    )
    paths = [write_record(arm, tmp_path / f'{arm.spec.name}.json') for arm in arms]
    return _decide(runner, tmp_path, store, paths, candidates=listed)

@pytest.mark.unit
def test_decide_prints_the_chosen_candidate_when_one_is_adopted(runner, tmp_path, growth_panel):
    '''Deferred :650: the adopted path of the economic decision, which no test reached.'''

    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')

    result = _decide_built(runner, tmp_path, panel, store, [('tree', 3, 0.5)])

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output).replace('\n', '')
    assert 'tree-d3 over covariates+ancestors: adopted' in output
    assert 'Adopted over the reference: tree-d3' in output
    assert 'Chosen: tree-d3' in output
    assert 'stands' not in output
    # A list with no smoothed candidate has no propagation comparison to print
    assert 'Propagation:' not in output
    assert read_record(tmp_path / 'decision.json', EconomicDecisionRecord).chosen == 'tree-d3'

@pytest.mark.unit
@pytest.mark.parametrize(
    'smoothed, twin, verdict', [(0.5, 1.0, 'adopted'), (1.0, 0.5, 'not adopted')],
    ids=['adopted', 'not-adopted']
)
def test_decide_prints_each_propagation_comparison(
    runner, tmp_path, growth_panel, smoothed, twin, verdict
):
    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')

    result = _decide_built(
        runner, tmp_path, panel, store, [('smoothed', 3, smoothed), ('text-only', 3, twin)]
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output).replace('\n', '')
    assert f'Propagation: smoothed-d3 over text-only-d3: {verdict}' in output
    assert output.count('Propagation:') == 1
    # After the comparisons, before what the decision adopted over the reference
    assert output.rindex('•') < output.index('Propagation:') < output.index('Adopted over')
    record = read_record(tmp_path / 'decision.json', EconomicDecisionRecord)
    assert [c.adopted for c in record.propagation] == [verdict == 'adopted']

@pytest.mark.unit
def test_decide_prints_one_propagation_line_for_each_pair_in_the_lists_order(
    runner, tmp_path, growth_panel
):
    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')

    result = _decide_built(
        runner,
        tmp_path,
        panel,
        store,
        [
            ('text-only', 2, 1.0),
            ('smoothed', 3, 0.5),
            ('text-only', 3, 1.0),
            ('smoothed', 2, 1.0),
        ],
    )

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output).replace('\n', '')
    first = 'Propagation: smoothed-d3 over text-only-d3: adopted'
    second = 'Propagation: smoothed-d2 over text-only-d2: not adopted'
    assert output.count('Propagation:') == 2
    assert 0 <= output.index(first) < output.index(second)

@pytest.mark.unit
def test_a_d8_decision_has_no_propagation_comparison_to_print(runner, tmp_path, decision_inputs):
    '''Only the economic decision reports propagation: a D8 decision record has none.'''

    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    better = write_record(
        synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5}),
        tmp_path / 'better.json',
    )

    result = _decide(runner, tmp_path, store, [better, reference])

    assert result.exit_code == 0, result.output
    output = click.unstyle(result.output).replace('\n', '')
    assert 'Chosen: better' in output
    assert 'Propagation' not in output
    assert not hasattr(read_record(tmp_path / 'decision.json', DecisionRecord), 'propagation')

@pytest.mark.unit
@pytest.mark.parametrize(
    'content, kind',
    [({
        'kind': 'economic_arm'
    }, "'economic_arm'"), ({}, 'None'), ([], 'None')],
    ids=['another-kind', 'no-kind', 'not-an-object'],
)
def test_decide_refuses_a_file_that_is_not_a_margin_record(
    runner, tmp_path, decision_inputs, content, kind
):
    '''Deferred :650: --margins must be a margin record of either kind, before any arm is read.'''

    store, reference = decision_inputs
    (tmp_path / 'margins.json').write_text(json.dumps(content))

    result = _decide(runner, tmp_path, store, [reference])

    assert result.exit_code == 1
    output = click.unstyle(result.output).replace('\n', '')
    assert 'Decision failed' in output
    assert 'is not a margin record' in output
    assert f'its kind is {kind}' in output
    assert not (tmp_path / 'decision.json').exists()

@pytest.mark.unit
def test_diagnostics_reports_every_statistic_and_writes_json(runner, tmp_path):
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK, dimension=4).write_parquet(table)
    arguments = ['diagnostics', '--table', str(table), '--codebook', str(codebook)]

    result = runner.invoke(
        tools_cli.app,
        [*arguments, '--geometry', 'hyperbolic', '--output',
         str(tmp_path / 'report.json')],
    )
    unknown = runner.invoke(tools_cli.app, [*arguments, '--geometry', 'poincare'])

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    for statistic in (
        'sector separation AUC', 'within-sector rank correlation', 'MAP over ancestors', 'NDCG',
        'distance Pearson', 'parent retrieval'
    ):
        assert statistic in output
    report = json.loads((tmp_path / 'report.json').read_text())
    assert set(report) == set(DiagnosticsReport.model_fields)
    assert unknown.exit_code == 1

@pytest.mark.unit
def test_diagnostics_refuses_a_table_that_misses_a_codebook_code(runner, tmp_path):
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK[1:], dimension=4).write_parquet(table)

    result = runner.invoke(
        tools_cli.app,
        [
            'diagnostics', '--table',
            str(table), '--codebook',
            str(codebook), '--geometry', 'euclidean'
        ],
    )

    assert result.exit_code == 1
    output = click.unstyle(result.output).replace('\n', '')
    assert 'Diagnostics failed' in output
    assert 'first 11' in output

@pytest.mark.unit
def test_diagnostics_reads_a_codebook_whose_codes_are_not_strings(runner, tmp_path):
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': [int(code) for code in CODEBOOK]}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK, dimension=4).write_parquet(table)

    result = runner.invoke(
        tools_cli.app,
        [
            'diagnostics', '--table',
            str(table), '--codebook',
            str(codebook), '--geometry', 'euclidean'
        ],
    )

    assert result.exit_code == 0, result.output
    assert 'parent retrieval: @1 ' in click.unstyle(result.output).replace('\n', '')

def _diagnose(runner, tmp_path, geometry, provenance):
    '''
    ``tools diagnostics`` on a stub arm's table, with ``provenance`` (JSON text) written beside
    it, and the report written to ``tmp_path/report.json``.
    '''

    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK, dimension=4).write_parquet(table)
    provenance_path(table).write_text(provenance)
    return runner.invoke(
        tools_cli.app,
        [
            'diagnostics', '--table',
            str(table), '--codebook',
            str(codebook), '--geometry', geometry, '--output',
            str(tmp_path / 'report.json')
        ],
    )

@pytest.mark.unit
@pytest.mark.parametrize(
    ('exported', 'geometry'),
    [('euclidean', 'spherical'), ('spherical', 'hyperbolic')],
    ids=['euclidean-as-spherical', 'spherical-as-hyperbolic'],
)
def test_diagnostics_refuses_a_table_exported_from_another_geometry(
    runner, tmp_path, monkeypatch, exported, geometry
):
    '''P9: the export names its arm's geometry, and Req 6's numbers for a table read under
    another arm's distance look plausible, so the table is refused before anything is read.'''

    reports = []
    report = tools_cli.diagnostics_report

    def spy(*args, **kwargs):
        reports.append(args)
        return report(*args, **kwargs)

    monkeypatch.setattr(tools_cli, 'diagnostics_report', spy)

    result = _diagnose(runner, tmp_path, geometry, json.dumps({'geometry': exported}))

    assert result.exit_code == 1
    output = ' '.join(result.output.split())
    assert 'Diagnostics failed' in output
    assert f'--geometry {geometry}' in output
    assert f'names {exported}' in output
    assert reports == []
    assert not (tmp_path / 'report.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize(
    ('geometry', 'named'),
    [
        ('euclidean', 'euclidean'),
        ('spherical', 'spherical'),
        ('hyperbolic', 'hyperbolic'),
        ('hyperbolic', None),
    ],
    ids=['euclidean', 'spherical', 'hyperbolic', 'no-geometry-key'],
)
def test_diagnostics_reads_a_table_whose_provenance_agrees_or_names_no_geometry(
    runner, tmp_path, geometry, named
):
    '''
    P9: only a provenance that names another geometry refuses the table. A table that names its
    own, or whose provenance has no geometry key (a text-only or HGCN table, a Stage 7 export),
    is read as before; a table with no provenance is covered by the first diagnostics test.
    '''

    provenance = {'dimension': 4}
    if named is not None:
        provenance['geometry'] = named

    result = _diagnose(runner, tmp_path, geometry, json.dumps(provenance))

    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / 'report.json').read_text())['geometry'] == geometry

@pytest.mark.unit
def test_diagnostics_refuses_a_table_whose_provenance_is_not_json(runner, tmp_path):
    '''An unreadable provenance cannot show the arm, so the table is refused, not read blind.'''

    result = _diagnose(runner, tmp_path, 'hyperbolic', '{"geometry": ')

    assert result.exit_code == 1
    output = ' '.join(result.output.split())
    assert 'Diagnostics failed' in output
    assert 'cannot be read as JSON' in output
    assert not (tmp_path / 'report.json').exists()

@pytest.mark.unit
def test_verify_stage4_is_gone(runner):
    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code != 0
    assert 'No such command' in result.output

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
        (Path('descriptions.parquet'), output, 'sentence-transformers/all-MiniLM-L6-v2', 128, 32)
    ]
    # Rich folds long paths at the terminal's width (80 columns on CI), wherever it falls
    assert 'text_only_provenance.json' in result.output.replace('\n', '')

@pytest.mark.unit
def test_text_only_table_refuses_a_backbone_without_a_recorded_window(runner, tmp_path):
    output = tmp_path / 'text_only.parquet'

    result = runner.invoke(
        tools_cli.app,
        [
            'text-only-table', '--descriptions', 'descriptions.parquet', '--output',
            str(output), '--backbone', 'some/other-backbone'
        ],
    )

    assert result.exit_code == 1
    # Rich may wrap the message at a space; compare with the whitespace collapsed
    assert 'no trained input window' in ' '.join(result.output.split())
    assert not output.exists()

def _regressor_arguments(tmp_path, codes=CODEBOOK):
    coordinates = tmp_path / 'arm.parquet'
    text_only = tmp_path / 'text_only.parquet'
    coordinate_table(codes).write_parquet(coordinates)
    text_only_table(codes).write_parquet(text_only)
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
def test_regressor_panel_refuses_a_growth_config_before_anything_is_read(
    runner, tmp_path, fixture_panel, monkeypatch
):
    # Every read scores the arm and the text-only table, and on growth no candidate's error may
    # exist before the economic margin is recorded (Req 24)
    log, loaded = fixture_panel
    config = tmp_path / 'regressor_panel.yaml'
    config.write_text('outcome: growth\n')
    monkeypatch.setattr(tools_cli, 'REGRESSOR_PANEL_CONFIG', str(config))

    result = runner.invoke(tools_cli.app, _regressor_arguments(tmp_path))

    assert result.exit_code == 1
    assert 'reads level mode only' in ' '.join(result.output.split())
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
    assert {r['purpose'] for r in log.records()} == {'regressor panel validation read'}
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
    # Without --purpose, a test read is not recorded as a validation read
    purposes = [r['purpose'] for r in log.records()]
    assert purposes == ['fixture opening', 'regressor panel test read'] * 2
    assert second.exit_code == 1
    assert 'reopen_reason' in second.output

@pytest.mark.unit
def test_regressor_panel_checks_every_opening_before_opening_any(runner, tmp_path, fixture_panel):
    # Opening the seen regime and then being refused the held-out one would use the first up
    log, _ = fixture_panel
    arguments = [
        *_regressor_arguments(tmp_path), '--split', 'test', '--open-purpose', 'fixture opening'
    ]

    earlier = runner.invoke(tools_cli.app, [*arguments, '--regime', 'heldout'])
    both = runner.invoke(tools_cli.app, arguments)

    assert earlier.exit_code == 0, earlier.output
    assert both.exit_code == 1
    assert 'reopen_reason' in both.output
    assert [(r['event'], r['panel']) for r in log.records()] == [
        ('open', 'regressor_heldout'),
        ('read', 'regressor_heldout'),
    ]

@pytest.mark.unit
def test_regressor_panel_reads_a_repeated_regime_once(runner, tmp_path, fixture_panel):
    log, _ = fixture_panel
    arguments = [
        *_regressor_arguments(tmp_path), '--split', 'test', '--open-purpose', 'fixture opening'
    ]

    result = runner.invoke(tools_cli.app, [*arguments, '--regime', 'seen', '--regime', 'seen'])

    assert result.exit_code == 0, result.output
    assert [(r['event'], r['panel']) for r in log.records()] == [
        ('open', 'regressor_seen'),
        ('read', 'regressor_seen'),
    ]

@pytest.mark.unit
def test_regressor_panel_checks_the_arm_and_the_output_before_opening(
    runner, tmp_path, fixture_panel
):
    # A test read that failed after its opening would use the opening up
    log, _ = fixture_panel
    test_split = ['--split', 'test', '--open-purpose', 'fixture opening']
    codes = [code for code in CODEBOOK if code != POPULATION[0]]
    (tmp_path / 'file').write_text('')
    unwritable = str(tmp_path / 'file' / 'predictions.parquet')

    missing = runner.invoke(tools_cli.app, [*_regressor_arguments(tmp_path, codes), *test_split])
    blocked = runner.invoke(
        tools_cli.app, [*_regressor_arguments(tmp_path), *test_split, '--output', unwritable]
    )
    directory = runner.invoke(
        tools_cli.app, [*_regressor_arguments(tmp_path), *test_split, '--output',
                        str(tmp_path)]
    )

    assert missing.exit_code == 1
    assert 'no coordinates' in missing.output
    for result in (blocked, directory):
        assert result.exit_code == 1
        assert 'Regressor panel failed' in result.output
    assert log.records() == []

# -------------------------------------------------------------------------------------------------
# Shared-encoder arms: export and the outcome read
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def default_config(monkeypatch):
    '''--config resolves to the default Config, whatever file it names.'''

    monkeypatch.setattr(Config, 'from_yaml', classmethod(lambda cls, path: Config()))

@pytest.mark.unit
def test_export_table_exports_under_the_configured_bundle_and_cache(
    monkeypatch, runner, tmp_path, default_config
):
    bundle = object()
    calls = []

    def fake_gate(cfg):
        calls.append(('gate', cfg.data_loader.streaming.max_length))
        return bundle

    def fake_export(checkpoint, chosen, token_config, output, *, device):
        calls.append(('export', checkpoint, chosen, token_config.max_length, output, device))
        return output

    monkeypatch.setattr(tools_cli, 'require_valid_supervision_bundle', fake_gate)
    monkeypatch.setattr(tools_cli, 'export_code_table', fake_export)
    monkeypatch.setattr(tools_cli, 'pick_device', lambda *_args: 'cpu')
    output = tmp_path / 'arm.parquet'

    result = runner.invoke(
        tools_cli.app,
        [
            'export-table', '--checkpoint', 'arm.ckpt', '--output',
            str(output), 'data_loader.streaming.max_length=64'
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [('gate', 64), ('export', 'arm.ckpt', bundle, 64, output, 'cpu')]
    # Rich folds long paths at the terminal's width (80 columns on CI), wherever it falls
    assert 'arm_provenance.json' in result.output.replace('\n', '')

@pytest.mark.unit
def test_export_table_refuses_the_removed_supervision_mode(
    monkeypatch, runner, tmp_path, default_config
):

    def never(*_args, **_kwargs):
        raise AssertionError('a supervision mode reached the export')

    monkeypatch.setattr(tools_cli, 'export_code_table', never)

    result = runner.invoke(
        tools_cli.app,
        [
            'export-table', '--checkpoint', 'arm.ckpt', '--output',
            str(tmp_path / 'arm.parquet'), 'supervision.mode=legacy_containment'
        ],
    )

    assert result.exit_code == 1
    # D2: training is always repaired, so the config refuses the key as an extra input
    message = ' '.join(click.unstyle(result.output).split())
    assert 'supervision.mode Extra inputs are not permitted' in message

@pytest.mark.unit
def test_export_table_refuses_an_override_without_a_value(runner, tmp_path, default_config):
    result = runner.invoke(
        tools_cli.app,
        [
            'export-table', '--checkpoint', 'arm.ckpt', '--output',
            str(tmp_path / 'arm.parquet'), 'model.dimension'
        ],
    )

    assert result.exit_code == 1
    assert 'key=value' in ' '.join(result.output.split())

class FakeArm:
    '''Stands in for ArmEncoder: points on the hyperboloid's spatial axes, and the logged names.'''

    distance = 'lorentz'
    table_fingerprint = 't' * 64
    checkpoint_sha256 = 'c' * 64

    def encode_queries(self, texts):
        return torch.zeros(len(texts), 3, dtype=torch.float64)

    def encode_codes(self, codes):
        return torch.tensor(
            [[0.0, float(row), 0.0] for row in range(len(codes))], dtype=torch.float64
        )

@pytest.mark.unit
def test_outcome_panel_reads_validation_and_logs_the_table(
    monkeypatch, runner, tmp_path, default_config, validated_bundle
):
    calls = []

    def fake_from_files(checkpoint, table, bundle, token_config, *, device):
        calls.append((checkpoint, table, bundle, token_config.max_length, device))
        return FakeArm()

    monkeypatch.setattr(tools_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle)
    monkeypatch.setattr(tools_cli.ArmEncoder, 'from_files', staticmethod(fake_from_files))
    monkeypatch.setattr(tools_cli, 'pick_device', lambda *_args: 'cpu')
    log = tmp_path / 'selection_log.jsonl'
    output = tmp_path / 'read.json'

    result = runner.invoke(
        tools_cli.app,
        [
            'outcome-panel', '--checkpoint', 'arm.ckpt', '--table', 'arm.parquet', '--purpose',
            'first live read', '--log',
            str(log), '--output',
            str(output)
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [('arm.ckpt', 'arm.parquet', validated_bundle, 128, 'cpu')]
    [record] = SelectionLog(log).records()
    assert (record['split'], record['purpose']) == ('validation', 'first live read')
    assert record['detail'] == {
        'encoder': 'FakeArm',
        'distance': 'lorentz',
        'table': 't' * 64,
        'checkpoint': 'c' * 64,
    }
    payload = json.loads(output.read_text())
    assert (payload['table'], payload['checkpoint']) == ('t' * 64, 'c' * 64)
    assert payload['summary']['n_queries'] == 1

@pytest.mark.unit
def test_outcome_panel_needs_a_purpose(runner, tmp_path, default_config):
    log = tmp_path / 'selection_log.jsonl'

    result = runner.invoke(
        tools_cli.app,
        ['outcome-panel', '--checkpoint', 'arm.ckpt', '--table', 'arm.parquet', '--log',
         str(log)],
    )

    # Click's usage error: the option is required
    assert result.exit_code == 2
    assert "Missing option '--purpose'" in click.unstyle(result.output).replace('\n', '')
    assert SelectionLog(log).records() == []

@pytest.mark.unit
def test_outcome_panel_refuses_a_text_only_table_without_a_traceback(
    monkeypatch, runner, tmp_path, default_config, validated_bundle, text_only_comparator_table
):
    '''The comparator's table sits beside an arm's in data/plan8/: a read can name it by mistake.'''

    monkeypatch.setattr(tools_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle)
    monkeypatch.setattr(tools_cli, 'pick_device', lambda *_args: 'cpu')
    checkpoint = tmp_path / 'arm.ckpt'
    checkpoint.write_bytes(b'never read: the table is refused first')
    log = tmp_path / 'selection_log.jsonl'

    result = runner.invoke(
        tools_cli.app,
        [
            'outcome-panel', '--checkpoint',
            str(checkpoint), '--table',
            str(text_only_comparator_table), '--purpose', 'a mixed-up read', '--log',
            str(log)
        ],
    )

    # The runner exits 1 for an uncaught KeyError too, and keeps its traceback out of the output:
    # a refusal the command handles is the one that raises SystemExit
    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    # Rich folds at spaces, so rejoin the words; the phrases asserted hold no path
    message = ' '.join(click.unstyle(result.output).split())
    assert 'Outcome panel failed' in message
    assert 'is not an exported arm table: its provenance names no checkpoint' in message
    assert SelectionLog(log).records() == []

# -------------------------------------------------------------------------------------------------
# Economic candidates: the build and the read (spec economic-candidates, 4.2 and 4.6)
# -------------------------------------------------------------------------------------------------

# The growth margins' read: the fixture panel's only load before a candidate's
MARGINS_LOAD = ('growth', 'naics_codebook.parquet', None, (6, ))
DETERMINISTIC = ('text-only-d3', 'tree-d3', 'smoothed-d3')
COMPONENTS = {'text-only': 1, 'tree': 1, 'smoothed': 2}
ROOT = Path(__file__).resolve().parents[2]

class EconomicInputs(NamedTuple):
    '''
    What the economic commands read, each option's file (None leaves the option out), and the
    frozen list as ``candidates`` holds it.
    '''

    root: Path
    codebook: Path
    text_only: Optional[Path]
    candidates: Optional[Path]
    frozen: CandidateList
    store: ArtifactStore
    runs_root: Optional[Path]
    margins: Path
    lock: Optional[Path]

def _write_list(path: Path, frozen: CandidateList) -> Path:
    path.write_text(yaml.safe_dump(frozen.model_dump(mode='json'), sort_keys=False))
    return path

def _economic_inputs(tmp_path, store):
    '''
    The fixture codebook and text-only table, and a frozen list of ``text-only-d3``, ``tree-d3``,
    ``smoothed-d3`` and the trained ``euclidean-d3`` over the store (``trained_source``). The
    three deterministic tables are built where the list pins them, and then the growth margins
    are read: the fixture panel's one load so far.
    '''

    codebook = write_codebook(tmp_path / 'codebook' / 'naics_codebook.parquet')
    text_only = write_text_only(tmp_path / 'text', codes=CODEBOOK)
    trained, runs_root = trained_source(
        store, tmp_path / 'trained', 'euclidean-d3', text_only=text_only
    )
    entries = []
    for name in DETERMINISTIC:
        kind = name[:-len('-d3')]
        table = build_candidate(
            kind,
            3,
            codebook,
            candidate_table_path(store.root, name),
            text_only=None if kind == 'tree' else text_only,
        )
        entries.append(
            CandidateEntry(
                name=name,
                kind=kind,
                dimension=3,
                components=COMPONENTS[kind],
                geometry='euclidean',
                table_sha256=sha256_file(table),
            )
        )
    frozen = fixture_list(
        [*entries, trained],
        **text_only_pins(text_only),
        codebook_sha256=sha256_file(codebook),
    )
    candidates = _write_list(tmp_path / 'candidates.yaml', frozen)
    assert _growth_margins(CliRunner(), tmp_path, store).exit_code == 0
    return EconomicInputs(
        root=tmp_path,
        codebook=codebook,
        text_only=text_only,
        candidates=candidates,
        frozen=frozen,
        store=store,
        runs_root=runs_root,
        margins=tmp_path / 'margins.json',
        lock=tmp_path / 'uv.lock',
    )

def _relisted(inputs, path, names=DETERMINISTIC, **update):
    '''The inputs with a frozen list whose rows of ``names`` take ``update``, written to ``path``.'''

    rows = [
        entry.model_copy(update=update) if entry.name in names else entry
        for entry in inputs.frozen.candidates
    ]
    frozen = inputs.frozen.model_copy(update={'candidates': rows})
    return inputs._replace(candidates=_write_list(path, frozen), frozen=frozen)

def _economic_read(runner, inputs, name, *extra, output=None):
    '''
    ``tools economic-read`` of ``name`` with every input option the inputs name, and ``--output``
    ``read.json`` beside them unless ``output`` is given.
    '''

    options = {
        '--candidates': inputs.candidates,
        '--codebook': inputs.codebook,
        '--margins': inputs.margins,
        '--store': inputs.store.root,
        '--text-only': inputs.text_only,
        '--runs-root': inputs.runs_root,
        '--lock': inputs.lock,
    }
    arguments = ['economic-read', '--candidate', name]
    for option, value in options.items():
        if value is not None:
            arguments += [option, str(value)]
    arguments += ['--output', str(output or inputs.root / 'read.json'), *extra]
    return runner.invoke(tools_cli.app, arguments)

def _economic_build(runner, inputs, store, *extra):
    '''``tools economic-build`` of the inputs' codebook and text-only table into ``store``.'''

    arguments = [
        'economic-build',
        '--codebook',
        str(inputs.codebook),
        '--text-only',
        str(inputs.text_only),
        '--store',
        str(store.root),
    ]
    if inputs.candidates is not None:
        arguments += ['--candidates', str(inputs.candidates)]
    return runner.invoke(tools_cli.app, [*arguments, *extra])

def _output(result):
    '''
    The output as one line. The economic commands print their long lines unwrapped: Rich folding
    at 80 columns (CI) can drop the space it folds at.
    '''

    return click.unstyle(result.output).replace('\n', '')

def _built(store):
    '''Every file under the store's ``candidates`` directory, by name.'''

    directory = store.root / 'candidates'
    return sorted(path.name for path in directory.rglob('*') if path.is_file())

def _untouched(panel, loaded):
    '''No panel loaded beyond the margins' read, and the log holds only its two reads.'''

    assert loaded == [MARGINS_LOAD]
    assert len(panel.log.records()) == 2

def _refused_read(result, *phrases):
    assert result.exit_code == 1, result.output
    output = _output(result)
    assert 'Economic read failed:' in output
    for phrase in phrases:
        assert phrase in output
    return output

def _rewrite_table(path):
    '''The table rewritten with other coordinates: its bytes are no longer its pin's.'''

    pl.read_parquet(path).with_columns(pl.col('e0') * 2).write_parquet(path)

def _edit_provenance(table, **entries):
    path = provenance_path(table)
    path.write_text(json.dumps({**json.loads(path.read_text()), **entries}))

def _source_run(inputs, seed):
    '''Seed ``seed``'s run in the trained candidate's stored source record.'''

    entry = inputs.frozen.entry('euclidean-d3')
    source = read_record(inputs.store.root / entry.record, ArmRecord)
    (run, ) = [run for run in source.runs if run.seed == seed]
    return run

# Build ---------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('pinned', [True, False], ids=['pinned', 'unpinned'])
def test_economic_build_writes_each_deterministic_table_with_its_provenance(
    runner, tmp_path, growth_panel, pinned
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    pins = {name: inputs.frozen.entry(name).table_sha256 for name in DETERMINISTIC}
    if not pinned:
        inputs = _relisted(inputs, tmp_path / 'unpinned.yaml', table_sha256=None)
    fresh = ArtifactStore(tmp_path / 'fresh')

    result = _economic_build(runner, inputs, fresh)

    assert result.exit_code == 0, result.output
    output = _output(result)
    for name in DETERMINISTIC:
        table = candidate_table_path(fresh.root, name)
        # Two builds write the same bytes (spec 4.2), the bytes the pins name
        assert sha256_file(table) == pins[name]
        assert json.loads(provenance_path(table).read_text())['table_sha256'] == pins[name]
        assert f'{name}: {table} (sha256 {pins[name]})' in output
    # Every deterministic row, in the list's order; the trained row is not built
    positions = [output.index(f'{name}: ') for name in DETERMINISTIC]
    assert positions == sorted(positions)
    assert _built(fresh) == sorted(
        f'{name}{suffix}' for name in DETERMINISTIC for suffix in ('.parquet', '_provenance.json')
    )
    # Building reads no panel
    _untouched(panel, loaded)

@pytest.mark.unit
@pytest.mark.parametrize(
    'chosen', [[], ['--candidate', 'tree-d3']], ids=['every-row', 'the-tree-row-alone']
)
def test_economic_build_refuses_a_text_only_table_other_than_the_lists(
    runner, tmp_path, growth_panel, chosen
):
    '''Review Focus 3: a text-only table with a valid provenance, but not the list's.'''

    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    other = write_text_only(tmp_path / 'plan8', codes=CODEBOOK[::-1])
    fresh = ArtifactStore(tmp_path / 'fresh')

    result = _economic_build(runner, inputs._replace(text_only=other), fresh, *chosen)

    assert result.exit_code == 1
    output = _output(result)
    assert 'Economic build failed:' in output
    assert f'{other} is not the shared text-only table that the frozen list pins' in output
    assert not (fresh.root / 'candidates').exists()

@pytest.mark.unit
def test_economic_build_refuses_a_codebook_other_than_the_lists(runner, tmp_path, growth_panel):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    other = write_codebook(tmp_path / 'other' / 'naics_codebook.parquet', CODEBOOK[::-1])
    fresh = ArtifactStore(tmp_path / 'fresh')

    result = _economic_build(runner, inputs._replace(codebook=other), fresh)

    assert result.exit_code == 1
    output = _output(result)
    assert 'Economic build failed:' in output
    assert f'{other} is not the codebook that the frozen list pins' in output
    assert not (fresh.root / 'candidates').exists()

@pytest.mark.unit
def test_economic_build_builds_each_named_candidate_once(runner, tmp_path, growth_panel):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    fresh = ArtifactStore(tmp_path / 'fresh')
    named = ['--candidate', 'smoothed-d3', '--candidate', 'tree-d3', '--candidate', 'smoothed-d3']

    result = _economic_build(runner, inputs, fresh, *named)

    assert result.exit_code == 0, result.output
    assert _built(fresh) == [
        'smoothed-d3.parquet',
        'smoothed-d3_provenance.json',
        'tree-d3.parquet',
        'tree-d3_provenance.json',
    ]
    output = _output(result)
    assert (output.count('smoothed-d3: '), output.count('tree-d3: ')) == (1, 1)

@pytest.mark.unit
@pytest.mark.parametrize(
    'names, message',
    [
        (['euclidean-d3'], 'euclidean-d3: a trained candidate is imported by tools economic-read'),
        (
            ['tree-d3', 'euclidean-d3'],
            'euclidean-d3: a trained candidate is imported by tools economic-read',
        ),
        (['unknown-d3'], "'unknown-d3' is not on the frozen candidate list"),
        (['tree-d3', 'unknown-d3'], "'unknown-d3' is not on the frozen candidate list"),
    ],
    ids=[
        'a-trained-row',
        'a-trained-row-after-a-deterministic-one',
        'an-unknown-name',
        'an-unknown-name-after-a-known-one',
    ],
)
def test_economic_build_refuses_a_row_it_cannot_build_before_writing_anything(
    runner, tmp_path, growth_panel, names, message
):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    fresh = ArtifactStore(tmp_path / 'fresh')
    named = [argument for name in names for argument in ('--candidate', name)]

    result = _economic_build(runner, inputs, fresh, *named)

    assert result.exit_code == 1
    output = _output(result)
    assert 'Economic build failed:' in output
    assert message in output
    assert not (fresh.root / 'candidates').exists()

@pytest.mark.unit
def test_economic_build_refuses_a_table_that_differs_from_its_pin(runner, tmp_path, growth_panel):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    inputs = _relisted(inputs, tmp_path / 'repinned.yaml', ['tree-d3'], table_sha256='a' * 64)
    fresh = ArtifactStore(tmp_path / 'fresh')

    result = _economic_build(runner, inputs, fresh)

    assert result.exit_code == 1
    output = _output(result)
    assert 'Economic build failed:' in output
    assert f'tree-d3: the built table {candidate_table_path(fresh.root, "tree-d3")}' in output
    assert f'differs from its pin {"a" * 64}' in output
    # The refusal stops the build
    assert not candidate_table_path(fresh.root, 'smoothed-d3').exists()

@pytest.mark.unit
@pytest.mark.parametrize('existing', ['table', 'provenance'])
def test_economic_build_refuses_an_existing_table_before_writing_anything(
    runner, tmp_path, growth_panel, existing
):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    fresh = ArtifactStore(tmp_path / 'fresh')
    table = candidate_table_path(fresh.root, 'smoothed-d3')
    path = table if existing == 'table' else provenance_path(table)
    path.parent.mkdir(parents=True)
    path.write_text('an earlier build')

    result = _economic_build(runner, inputs, fresh)

    assert result.exit_code == 1
    output = _output(result)
    assert 'Economic build failed:' in output
    assert f'{path} exists; a candidate table is built once' in output
    assert _built(fresh) == [path.name]
    assert path.read_text() == 'an earlier build'

# Read ----------------------------------------------------------------------------------------

@pytest.mark.unit
def test_economic_read_dry_run_loads_no_panel_and_leaves_the_log_untouched(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    log = tmp_path / 'real' / 'selection_log.jsonl'
    for name in ('text-only-d3', 'tree-d3', 'smoothed-d3', 'euclidean-d3'):
        result = _economic_read(runner, inputs, name, '--dry-run', '--log', str(log))
        assert result.exit_code == 0, result.output
        output = click.unstyle(result.output).replace('\n', '')
        assert 'no panel was loaded and nothing was logged' in output
    assert loaded == [('growth', 'naics_codebook.parquet', None, (6, ))]  # the margins' only
    assert not log.exists()
    assert not (tmp_path / 'read.json').exists()

@pytest.mark.unit
def test_economic_read_dry_run_leaves_an_existing_log_as_it_was(runner, tmp_path, growth_panel):
    '''Review Focus 1 with the main checkout's log, which exists: its bytes stay the same.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    log = tmp_path / 'real' / 'selection_log.jsonl'
    log.parent.mkdir()
    log.write_text('{"event": "read"}\n')
    before = sha256_file(log)

    for name in ('text-only-d3', 'tree-d3', 'smoothed-d3', 'euclidean-d3'):
        result = _economic_read(runner, inputs, name, '--dry-run', '--log', str(log))
        assert result.exit_code == 0, result.output
        assert f'Dry run: every check passed for {name}; ' in _output(result)

    assert sha256_file(log) == before
    _untouched(panel, loaded)
    assert not (tmp_path / 'read.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize('name', DETERMINISTIC)
def test_economic_read_refuses_a_text_only_table_other_than_the_lists(
    runner, tmp_path, growth_panel, name
):
    '''Review Focus 3: a text-only table with a valid provenance, but not the list's.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    other = write_text_only(tmp_path / 'plan8', codes=CODEBOOK[::-1])

    result = _economic_read(runner, inputs._replace(text_only=other), name)

    _refused_read(result, f'{other} is not the shared text-only table that the frozen list pins')
    _untouched(panel, loaded)
    assert not (tmp_path / 'read.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize('name', DETERMINISTIC)
@pytest.mark.parametrize('extra', [[], ['--dry-run']], ids=['read', 'dry-run'])
def test_economic_read_refuses_a_text_only_provenance_other_than_the_lists(
    runner, tmp_path, growth_panel, name, extra
):
    '''Review Focus 1: the list's table bytes, beside a regenerated provenance.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    copy = tmp_path / 'copy' / 'text_only.parquet'
    copy.parent.mkdir()
    shutil.copyfile(inputs.text_only, copy)
    # The same content in other bytes: the table's pin and provenance_fields both pass
    content = json.loads(provenance_path(inputs.text_only).read_text())
    provenance_path(copy).write_text(json.dumps(content, indent=4) + '\n')

    result = _economic_read(runner, inputs._replace(text_only=copy), name, *extra)

    _refused_read(
        result,
        f"{provenance_path(copy)} is not the shared text-only table's provenance that the frozen "
        'list pins',
    )
    _untouched(panel, loaded)
    assert not (tmp_path / 'read.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize('extra', [[], ['--dry-run']], ids=['read', 'dry-run'])
def test_economic_read_refuses_a_trained_source_with_another_text_only_provenance(
    runner, tmp_path, growth_panel, extra
):
    '''Review Focus 2: the list pins another provenance than the one every source record stores.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    frozen = inputs.frozen.model_copy(update={'text_only_provenance_sha256': 'f' * 64})
    repinned = inputs._replace(
        candidates=_write_list(tmp_path / 'repinned.yaml', frozen), frozen=frozen
    )

    result = _economic_read(runner, repinned, 'euclidean-d3', *extra)

    assert result.exit_code == 1, result.output
    output = _output(result)
    assert "euclidean-d3: its source's text-only provenance" in output
    assert 'ffffffffffff that the frozen list pins' in output
    _untouched(panel, loaded)
    assert not (tmp_path / 'read.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize('extra', [[], ['--dry-run']], ids=['read', 'dry-run'])
def test_economic_read_refuses_a_table_that_differs_from_its_pin_before_loading_the_panel(
    runner, tmp_path, growth_panel, extra
):
    '''Review Focus 2: the stored table is edited after it was pinned.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    table = candidate_table_path(inputs.store.root, 'text-only-d3')
    _rewrite_table(table)

    result = _economic_read(runner, inputs, 'text-only-d3', *extra)

    pin = inputs.frozen.entry('text-only-d3').table_sha256
    _refused_read(result, f'has sha256 {sha256_file(table)}, which differs from its pin {pin}')
    _untouched(panel, loaded)

@pytest.mark.unit
@pytest.mark.parametrize('extra', [[], ['--dry-run']], ids=['read', 'dry-run'])
def test_economic_read_refuses_a_trained_candidate_whose_store_lacks_a_checkpoint(
    runner, tmp_path, growth_panel, extra
):
    '''Review Focus 4: a partial store copy, without one later seed's checkpoint object.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    run = _source_run(inputs, 3)
    (inputs.store.root / run.checkpoint.path).unlink()

    result = _economic_read(runner, inputs, 'euclidean-d3', *extra)

    _refused_read(result, run.checkpoint.path, 'not in the artifact store')
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_refuses_an_existing_record_before_any_check(
    runner, tmp_path, growth_panel, monkeypatch
):
    '''Review Focus 5: nothing is checked, so nothing is hashed, and nothing is read.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    (tmp_path / 'read.json').write_text('an earlier record')
    calls = []
    monkeypatch.setattr(tools_cli, 'load_candidate_list', lambda *args: calls.append(args))

    result = _economic_read(runner, inputs, 'tree-d3')

    _refused_read(result, 'exists; a record is written once')
    assert calls == []
    _untouched(panel, loaded)
    assert (tmp_path / 'read.json').read_text() == 'an earlier record'

@pytest.mark.unit
def test_economic_read_refuses_d8_margins(runner, tmp_path, growth_panel, decision_inputs):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    store, reference = decision_inputs
    assert _margins(runner, tmp_path / 'd8', store, reference).exit_code == 0

    result = _economic_read(
        runner, inputs._replace(margins=tmp_path / 'd8' / 'margins.json'), 'tree-d3'
    )

    _refused_read(result, 'reads relative margins', 'is a D8 margin record')
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_reads_and_decide_record_a_decision_over_both_kinds(
    runner, tmp_path, growth_panel
):
    '''§6 "Decision": the command reads a deterministic and a trained candidate.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    arms = []
    for name, runs in (('tree-d3', 1), ('euclidean-d3', 5)):
        before = len(panel.log.records())
        result = _economic_read(runner, inputs, name, output=tmp_path / f'{name}.json')
        assert result.exit_code == 0, result.output
        added = panel.log.records()[before:]
        assert sorted(record['panel'] for record in added) == sorted(GROWTH_PANELS * runs)
        assert {(r['event'], r['split'], r['purpose'])
                for r in added} == {('read', 'validation', 'economic candidate read')}
        record = read_record(tmp_path / f'{name}.json', EconomicArmRecord)
        output = _output(result)
        for growth in GROWTH_PANELS:
            mean = np.mean([run.statistics[growth] for run in record.runs])
            assert f'{growth}: covariates+embedding MSE {mean:.6g}, ' in output
        arms.append(tmp_path / f'{name}.json')
    # The log gains 2 + 10 growth validation reads
    assert len(panel.log.records()) == 2 + 2 + 10
    assert loaded[1:] == [('growth', str(inputs.codebook), None, (6, ))] * 2
    both = inputs.frozen.model_copy(
        update={'candidates': [inputs.frozen.entry(name) for name in ('tree-d3', 'euclidean-d3')]}
    )

    result = _decide(
        runner, tmp_path, inputs.store, arms, candidates=_write_list(tmp_path / 'two.yaml', both)
    )

    assert result.exit_code == 0, result.output
    decision = read_record(tmp_path / 'decision.json', EconomicDecisionRecord)
    assert [(arm.spec.name, len(arm.runs)) for arm in decision.arms] == [
        ('tree-d3', 1),
        ('euclidean-d3', 5),
    ]
    assert 'Chosen:' in _output(result)

@pytest.mark.unit
def test_decide_refuses_a_record_that_stores_another_text_only_provenance(
    runner, tmp_path, growth_panel
):
    '''Review Focus 3: the same table bytes under another provenance, in a hand-built record.'''

    panel, _ = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    names = (*DETERMINISTIC, 'euclidean-d3')
    for name in names:
        result = _economic_read(runner, inputs, name, output=tmp_path / f'{name}.json')
        assert result.exit_code == 0, result.output
    other = inputs.store.put_text_only(
        write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK)
    )
    tree = read_record(tmp_path / 'tree-d3.json', EconomicArmRecord)
    assert other.table.sha256 == tree.text_only.table.sha256
    assert other.provenance.sha256 != tree.text_only.provenance.sha256
    edited = write_record(tree.model_copy(update={'text_only': other}), tmp_path / 'edited.json')
    arms = [edited if name == 'tree-d3' else tmp_path / f'{name}.json' for name in names]

    result = _decide(runner, tmp_path, inputs.store, arms, candidates=inputs.candidates)

    assert result.exit_code == 1, result.output
    output = _output(result)
    assert 'Decision failed:' in output
    assert 'tree-d3: its text-only provenance' in output
    assert not (tmp_path / 'decision.json').exists()

@pytest.mark.unit
@pytest.mark.parametrize('name', DETERMINISTIC)
def test_economic_read_writes_a_deterministic_record_with_the_text_only_tables_d9_fields(
    runner, tmp_path, growth_panel, name
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    log = tmp_path / 'real' / 'selection_log.jsonl'

    result = _economic_read(runner, inputs, name, '--purpose', 'a fixture read', '--log', str(log))

    assert result.exit_code == 0, result.output
    record = read_record(tmp_path / 'read.json', EconomicArmRecord)
    entry = inputs.frozen.entry(name)
    content = json.loads(provenance_path(inputs.text_only).read_text())
    fields = provenance_fields(
        content, sha256_file(inputs.text_only), content['matrix_fingerprint'], 'text-only'
    )
    assert record.spec == economic_spec(entry, text_only=fields)
    (run, ) = record.runs
    assert run.table.sha256 == entry.table_sha256
    check_economic_arm(record, inputs.store, inputs.frozen)
    # The growth panel, loaded as tools growth-margins loads it, with --codebook and --log
    assert loaded[1:] == [('growth', str(inputs.codebook), str(log), (6, ))]
    reads = panel.log.records()[2:]
    assert [(r['panel'], r['purpose'])
            for r in reads] == [(growth, 'a fixture read') for growth in GROWTH_PANELS]
    output = _output(result)
    for growth in GROWTH_PANELS:
        assert f'{growth}: covariates+embedding MSE {run.statistics[growth]:.6g}, ' in output
    assert f'Economic arm record: {tmp_path / "read.json"}' in output
    assert f'Reads logged to {panel.log.path}' in output

@pytest.mark.unit
def test_economic_read_refuses_a_row_that_pins_no_table(runner, tmp_path, growth_panel):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    inputs = _relisted(inputs, tmp_path / 'unpinned.yaml', ['tree-d3'], table_sha256=None)

    result = _economic_read(runner, inputs, 'tree-d3')

    _refused_read(result, 'tree-d3: its frozen-list row pins no table')
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_refuses_a_provenance_outside_text_and_tree(runner, tmp_path, growth_panel):
    '''The read itself checks only that a provenance describes its table: the rule runs first.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    table = candidate_table_path(inputs.store.root, 'smoothed-d3')
    _edit_provenance(table, qcew={'path': 'qcew.parquet', 'sha256': 'q' * 64})

    result = _economic_read(runner, inputs, 'smoothed-d3')

    _refused_read(result, "smoothed-d3: entries: its provenance has the entries ['qcew']", 'Req 29')
    _untouched(panel, loaded)

@pytest.mark.unit
@pytest.mark.parametrize(
    'field, message',
    [
        ('table_sha256', 'describes another file than the table'),
        ('matrix_fingerprint', 'names another matrix_fingerprint'),
    ],
    ids=['another-table-sha256', 'another-matrix-fingerprint'],
)
def test_economic_read_refuses_a_provenance_that_describes_another_table(
    runner, tmp_path, growth_panel, field, message
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    table = candidate_table_path(inputs.store.root, 'tree-d3')
    _edit_provenance(table, **{field: 'f' * 64})

    result = _economic_read(runner, inputs, 'tree-d3')

    _refused_read(result, f'{provenance_path(table)} {message}')
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_refuses_a_text_only_table_whose_provenance_does_not_describe_it(
    runner, tmp_path, growth_panel
):
    '''
    The list's table, whose provenance names other bytes: even the tree read refuses it, at the
    list's pin of the provenance file, before the provenance is parsed.
    '''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    copy = tmp_path / 'copy' / 'text_only.parquet'
    copy.parent.mkdir()
    shutil.copyfile(inputs.text_only, copy)
    provenance = json.loads(provenance_path(inputs.text_only).read_text())
    provenance_path(copy).write_text(json.dumps({**provenance, 'table_sha256': 'f' * 64}))

    result = _economic_read(runner, inputs._replace(text_only=copy), 'tree-d3')

    _refused_read(
        result,
        f"{provenance_path(copy)} is not the shared text-only table's provenance that the frozen "
        'list pins',
    )
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_checks_the_margin_record_in_a_dry_run(runner, tmp_path, growth_panel):
    '''Spec :258: the margin record's own checks run before the first read, and in a dry run.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    content = json.loads(inputs.margins.read_text())
    content['margins'][0]['margin'] *= 2
    tampered = tmp_path / 'tampered.json'
    tampered.write_text(json.dumps(content))

    result = _economic_read(runner, inputs._replace(margins=tampered), 'tree-d3', '--dry-run')

    output = _refused_read(result, "the growth_seen δ is not r × the reference's MSE")
    assert 'Dry run' not in output
    _untouched(panel, loaded)

@pytest.mark.unit
@pytest.mark.parametrize(
    'name, field, option',
    [('tree-d3', 'text_only', '--text-only'), ('euclidean-d3', 'runs_root', '--runs-root')],
    ids=['deterministic', 'trained'],
)
def test_economic_read_needs_the_option_its_candidates_kind_reads(
    runner, tmp_path, growth_panel, name, field, option
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))

    result = _economic_read(runner, inputs._replace(**{field: None}), name)

    _refused_read(result, f'{name}: ', f'so it needs {option}')
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_checks_a_deterministic_candidate_in_the_briefs_order(
    runner, tmp_path, growth_panel
):
    '''
    Each read keeps the input the read before it was refused for, and breaks one later check's,
    so each refusal names the earlier check (brief steps 1-5).
    '''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    (tmp_path / 'read.json').write_text('an earlier record')
    d8 = tmp_path / 'd8.json'
    d8.write_text(json.dumps({'kind': 'margins'}))
    table = candidate_table_path(inputs.store.root, 'text-only-d3')
    original = table.read_bytes()
    _rewrite_table(table)
    broken = inputs._replace(
        margins=d8,
        text_only=write_text_only(tmp_path / 'plan8', codes=CODEBOOK[::-1]),
        lock=tmp_path / 'missing.lock',
    )

    def refused(name, chosen, *extra):
        return _refused_read(_economic_read(runner, chosen, name, *extra))

    assert 'exists; a record is written once' in refused('unknown-d3', broken)
    (tmp_path / 'read.json').unlink()
    assert "'unknown-d3' is not on the frozen candidate list" in refused('unknown-d3', broken)
    assert 'reads relative margins' in refused('text-only-d3', broken)
    broken = broken._replace(margins=inputs.margins)
    assert 'differs from its pin' in refused('text-only-d3', broken)
    table.write_bytes(original)
    provenance = provenance_path(table).read_text()
    _edit_provenance(table, qcew={'path': 'qcew.parquet', 'sha256': 'q' * 64})
    assert 'Req 29' in refused('text-only-d3', broken)
    provenance_path(table).write_text(provenance)
    assert 'is not the shared text-only table' in refused('text-only-d3', broken)
    broken = broken._replace(text_only=inputs.text_only)
    assert 'no uv.lock to name the read by' in refused('text-only-d3', broken, '--dry-run')

    result = _economic_read(runner, inputs, 'text-only-d3', '--dry-run')

    assert result.exit_code == 0, result.output
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_checks_a_trained_candidate_in_the_briefs_order(
    runner, tmp_path, growth_panel
):
    '''As for a deterministic candidate: the margins, then the import, then the lock.'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    d8 = tmp_path / 'd8.json'
    d8.write_text(json.dumps({'kind': 'margins'}))
    run = _source_run(inputs, 3)
    checkpoint = inputs.store.root / run.checkpoint.path
    aside = tmp_path / 'aside.ckpt'
    checkpoint.rename(aside)
    broken = inputs._replace(margins=d8, lock=tmp_path / 'missing.lock')

    def refused(chosen, *extra):
        return _refused_read(_economic_read(runner, chosen, 'euclidean-d3', *extra))

    assert 'reads relative margins' in refused(broken)
    broken = broken._replace(margins=inputs.margins)
    assert run.checkpoint.path in refused(broken)
    aside.rename(checkpoint)
    assert 'no uv.lock to name the read by' in refused(broken, '--dry-run')

    result = _economic_read(runner, inputs, 'euclidean-d3', '--dry-run')

    assert result.exit_code == 0, result.output
    _untouched(panel, loaded)

@pytest.mark.unit
def test_economic_read_names_the_working_directorys_uv_lock_by_default(
    runner, tmp_path, growth_panel, monkeypatch
):
    '''``--lock`` is ``uv.lock`` in the directory the command runs from (spec 4.6).'''

    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))._replace(lock=None)

    monkeypatch.chdir(tmp_path)
    found = _economic_read(runner, inputs, 'tree-d3', '--dry-run')
    monkeypatch.chdir(tmp_path / 'codebook')
    missing = _economic_read(runner, inputs, 'tree-d3', '--dry-run')

    assert found.exit_code == 0, found.output
    _refused_read(missing, 'uv.lock: no uv.lock to name the read by')
    _untouched(panel, loaded)

# Both commands -------------------------------------------------------------------------------

def _economic_command(runner, inputs, command, store):
    if command == 'economic-build':
        return _economic_build(runner, inputs, store, '--candidate', 'tree-d3')
    return _economic_read(runner, inputs, 'tree-d3')

@pytest.mark.unit
@pytest.mark.parametrize('command', ['economic-build', 'economic-read'])
def test_the_economic_commands_read_the_committed_candidate_list_by_default(
    runner, tmp_path, growth_panel, monkeypatch, command
):
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))._replace(candidates=None)
    fresh = ArtifactStore(tmp_path / 'fresh')
    # The committed list is relative to the repository root, where the commands run
    monkeypatch.chdir(ROOT)

    result = _economic_command(runner, inputs, command, fresh)

    assert result.exit_code == 1
    output = _output(result)
    assert f'Economic {command.split("-")[1]} failed:' in output
    assert "'tree-d3' is not on the frozen candidate list" in output
    assert not (fresh.root / 'candidates').exists()

@pytest.mark.unit
@pytest.mark.parametrize('command', ['economic-build', 'economic-read'])
def test_the_economic_commands_refuse_a_candidate_list_that_is_not_yaml(
    runner, tmp_path, growth_panel, command
):
    '''The handlers catch it: load_candidate_list refuses broken YAML as a ValueError.'''

    panel, loaded = growth_panel
    broken = tmp_path / 'broken.yaml'
    broken.write_text('codebook_sha256: x: y\n')
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))._replace(
        candidates=broken
    )
    fresh = ArtifactStore(tmp_path / 'fresh')

    result = _economic_command(runner, inputs, command, fresh)

    # A handled refusal raises SystemExit; an uncaught YAMLError would exit 1 too
    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    output = _output(result)
    assert f'Economic {command.split("-")[1]} failed: {broken} is not safe YAML' in output
    assert not (fresh.root / 'candidates').exists()
    _untouched(panel, loaded)
