'''
``conf/local.yaml``: the gitignored override for this machine's supervision manifests.

It sets at most the text and graph manifests, each repo-relative under ``data/``, between the
tracked YAML and the command line. The suite runs with it off (``tests/conftest.py``); these tests
turn it on for a temporary checkout.
'''

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.graph_model import hgcn
from naics_embedder.graph_model.curriculum.preprocess_curriculum import resolve_graph_config
from naics_embedder.utils.config import (
    LOCAL_CONFIG_ENV,
    Config,
    GraphConfig,
    apply_local_config,
    read_local_config,
)

REPO = Path(__file__).resolve().parents[2]
TEXT_KEY = 'config.supervision.manifest_path'
GRAPH_KEY = 'graph.supervision_manifest_path'

def local_yaml(text=None, graph=None) -> str:
    '''An override setting the text manifest, the graph manifest, or both.'''

    data = {}
    if text is not None:
        data['config'] = {'supervision': {'manifest_path': text}}
    if graph is not None:
        data['graph'] = {'supervision_manifest_path': graph}
    return yaml.safe_dump(data)

@pytest.fixture
def checkout(tmp_path, monkeypatch):
    '''A checkout root with an empty ``conf/``, as the working directory, with the override on.'''

    root = tmp_path / 'checkout'
    (root / 'conf').mkdir(parents=True)
    monkeypatch.chdir(root)
    monkeypatch.delenv(LOCAL_CONFIG_ENV)
    return root

# -------------------------------------------------------------------------------------------------
# Off: the suite and the tracked files
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_suite_runs_with_the_override_off():
    assert os.environ[LOCAL_CONFIG_ENV] == 'off'

@pytest.mark.unit
def test_with_the_override_off_the_tracked_yamls_name_no_manifest():
    '''Whatever this checkout's own conf/local.yaml says, the tracked files pin nothing.'''

    text = apply_local_config(Config.from_yaml(str(REPO / 'conf/config.yaml')), REPO)
    graph = apply_local_config(GraphConfig.from_yaml(str(REPO / 'conf/graph.yaml')), REPO)

    assert text.supervision.manifest_path is None
    assert graph.supervision_manifest_path is None

@pytest.mark.unit
def test_off_reads_nothing(checkout, monkeypatch):
    (checkout / 'conf/local.yaml').write_text('config: [not valid')
    monkeypatch.setenv(LOCAL_CONFIG_ENV, 'off')

    local = read_local_config(checkout)

    assert (local.source, local.values) == (None, {})

@pytest.mark.unit
@pytest.mark.parametrize('content', [None, '', '{}\n'])
def test_an_absent_or_empty_override_sets_nothing(checkout, content):
    if content is not None:
        (checkout / 'conf/local.yaml').write_text(content)
    cfg = Config()

    assert read_local_config(checkout).values == {}
    assert apply_local_config(cfg, checkout) == cfg

@pytest.mark.unit
def test_the_override_is_gitignored_and_its_example_is_not():

    def ignored(name: str) -> bool:
        result = subprocess.run(['git', '-C', str(REPO), 'check-ignore', '-q', name], check=False)
        assert result.returncode in (0, 1), name
        return result.returncode == 0

    assert ignored('conf/local.yaml')
    assert not ignored('conf/local.example.yaml')
    assert not ignored('conf/config.yaml')

@pytest.mark.unit
def test_the_example_names_only_the_allowed_keys():
    example = yaml.safe_load((REPO / 'conf/local.example.yaml').read_text())

    assert example['config']['supervision']['manifest_path'].startswith('data/supervision/')
    assert example['graph']['supervision_manifest_path'].startswith('data/supervision/')
    assert list(example) == ['config', 'graph']

# -------------------------------------------------------------------------------------------------
# Precedence: tracked YAML < conf/local.yaml < the command line
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_override_sits_between_the_tracked_yaml_and_the_command_line(checkout):
    tracked = {'supervision': {'manifest_path': 'data/tracked/manifest.json'}}
    (checkout / 'conf/config.yaml').write_text(yaml.safe_dump(tracked))

    def manifest(*overrides: str) -> str:
        return tools_cli._run_config('conf/config.yaml', list(overrides)).supervision.manifest_path

    assert manifest() == 'data/tracked/manifest.json'
    (checkout / 'conf/local.yaml').write_text(local_yaml(text='data/local/manifest.json'))
    assert manifest() == 'data/local/manifest.json'
    assert manifest('supervision.manifest_path=data/cli/manifest.json') == 'data/cli/manifest.json'

@pytest.mark.unit
def test_the_override_changes_only_the_manifest(checkout):
    tracked = Config.from_yaml(str(REPO / 'conf/config.yaml'))
    (checkout / 'conf/local.yaml').write_text(local_yaml(text='data/local/manifest.json'))

    local = apply_local_config(tracked, checkout)

    assert local.supervision.manifest_path == 'data/local/manifest.json'
    assert local.model_copy(update={'supervision': tracked.supervision}) == tracked

@pytest.mark.unit
def test_the_text_override_leaves_the_graph_config_alone_and_the_reverse(checkout):
    (checkout / 'conf/local.yaml').write_text(local_yaml(text='data/text/manifest.json'))
    assert apply_local_config(GraphConfig(), checkout) == GraphConfig()

    (checkout / 'conf/local.yaml').write_text(local_yaml(graph='data/graph/manifest.json'))
    assert apply_local_config(Config(), checkout) == Config()

# -------------------------------------------------------------------------------------------------
# The graph config: only the manifest becomes explicitly set
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_graph_override_marks_only_its_manifest_as_set(checkout):
    shutil.copyfile(REPO / 'conf/graph.yaml', checkout / 'conf/graph.yaml')
    (checkout / 'conf/local.yaml').write_text(local_yaml(graph='data/local/manifest.json'))

    cfg = apply_local_config(GraphConfig.from_yaml('conf/graph.yaml'), checkout)

    tracked_keys = set(yaml.safe_load((REPO / 'conf/graph.yaml').read_text()))
    assert cfg.supervision_manifest_path == 'data/local/manifest.json'
    assert cfg.model_fields_set == tracked_keys | {'supervision_manifest_path'}

@pytest.mark.unit
def test_the_graph_override_resolves_its_bundle(checkout, reference_manifest):
    '''Unset legacy paths are replaced; rebuilt from a dump, they would all count as explicit.'''

    bundle = checkout / 'data/bundles' / reference_manifest.parent.name
    shutil.copytree(reference_manifest.parent, bundle)
    shutil.copyfile(REPO / 'conf/graph.yaml', checkout / 'conf/graph.yaml')
    manifest = (bundle / 'manifest.json').relative_to(checkout).as_posix()
    (checkout / 'conf/local.yaml').write_text(local_yaml(graph=manifest))

    cfg = apply_local_config(GraphConfig.from_yaml('conf/graph.yaml'), checkout)
    resolved = resolve_graph_config(cfg)

    assert Path(resolved.relations_parquet).is_relative_to(bundle.resolve())
    with pytest.raises(ValueError, match='does not belong to supervision bundle'):
        resolve_graph_config(GraphConfig(**cfg.model_dump()))

@pytest.mark.unit
def test_hgcn_main_reads_the_override(checkout, monkeypatch):
    shutil.copyfile(REPO / 'conf/graph.yaml', checkout / 'conf/graph.yaml')
    (checkout / 'conf/local.yaml').write_text(local_yaml(graph='data/local/manifest.json'))
    seen = []

    class Resolved(Exception):
        pass

    def resolve(cfg):
        seen.append(cfg)
        raise Resolved

    monkeypatch.setattr(hgcn, 'resolve_graph_config', resolve)

    with pytest.raises(Resolved):
        hgcn.main('conf/graph.yaml')
    assert seen[0].supervision_manifest_path == 'data/local/manifest.json'

# -------------------------------------------------------------------------------------------------
# Refusals: any other key, and a manifest outside data/
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'content',
    [
        'seed: 1\n',
        'qcew_dir: data/QCEW\n',
        'config:\n  seed: 1\n',
        'config:\n  supervision:\n    contract_version: stage3-supervision-v2\n',
        'config:\n  supervision: null\n',
        'config: data/b/manifest.json\n',
        'graph:\n  relations_parquet: data/relations.parquet\n',
        '- data/b/manifest.json\n',
    ],
)
def test_any_other_key_is_refused(checkout, content):
    (checkout / 'conf/local.yaml').write_text(content)

    with pytest.raises(ValueError, match='may set only'):
        read_local_config(checkout)

@pytest.mark.unit
def test_malformed_yaml_is_a_value_error_naming_the_file(checkout):
    (checkout / 'conf/local.yaml').write_text('config: [not valid')

    with pytest.raises(ValueError, match='conf/local.yaml is not valid YAML'):
        read_local_config(checkout)

@pytest.mark.unit
@pytest.mark.parametrize(
    'value',
    [
        '/Users/someone/data/b/manifest.json',
        '~/data/b/manifest.json',
        '../data/b/manifest.json',
        'data/../outputs/manifest.json',
        'outputs/b/manifest.json',
        'data\\b\\manifest.json',
        './data/b/manifest.json',
        'data//b/manifest.json',
        'data/./b/manifest.json',
        'data',
        '',
        1,
        None,
        ['data/b/manifest.json'],
    ],
)
@pytest.mark.parametrize('section', ['config', 'graph'])
def test_a_manifest_outside_data_is_refused(checkout, section, value):
    data = {
        'config': {
            'config': {
                'supervision': {
                    'manifest_path': value
                }
            }
        },
        'graph': {
            'graph': {
                'supervision_manifest_path': value
            }
        },
    }[section]
    (checkout / 'conf/local.yaml').write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match='under data/'):
        read_local_config(checkout)

@pytest.mark.unit
@pytest.mark.parametrize('value', ['data/b/manifest.json', 'data/supervision/v2/id/manifest.json'])
def test_a_manifest_under_data_is_kept_as_written(checkout, value):
    (checkout / 'conf/local.yaml').write_text(local_yaml(text=value, graph=value))

    local = read_local_config(checkout)

    assert local.values == {TEXT_KEY: value, GRAPH_KEY: value}
    assert local.source == 'conf/local.yaml'
    assert apply_local_config(Config(), checkout).supervision.manifest_path == value

# -------------------------------------------------------------------------------------------------
# Where the override is read: the root, or the path NAICS_EMBEDDER_LOCAL_CONFIG names
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_override_is_read_at_the_root_not_the_cwd(checkout, tmp_path, monkeypatch):
    elsewhere = tmp_path / 'elsewhere'
    (elsewhere / 'conf').mkdir(parents=True)
    (elsewhere / 'conf/local.yaml').write_text(local_yaml(text='data/cwd/manifest.json'))
    (checkout / 'conf/local.yaml').write_text(local_yaml(text='data/root/manifest.json'))
    monkeypatch.chdir(elsewhere)

    assert read_local_config(checkout).values == {TEXT_KEY: 'data/root/manifest.json'}

@pytest.mark.unit
@pytest.mark.parametrize('absolute', [False, True])
def test_the_env_var_relocates_the_override(checkout, tmp_path, monkeypatch, absolute):
    (checkout / 'conf/local.yaml').write_text(local_yaml(graph='data/default/manifest.json'))
    (checkout / 'alt.yaml').write_text(local_yaml(graph='data/alt/manifest.json'))
    named = str(checkout / 'alt.yaml') if absolute else 'alt.yaml'
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv(LOCAL_CONFIG_ENV, named)

    local = read_local_config(checkout)

    assert local.values == {GRAPH_KEY: 'data/alt/manifest.json'}
    assert local.source == named

@pytest.mark.unit
def test_a_relocated_override_must_exist(checkout, monkeypatch):
    monkeypatch.setenv(LOCAL_CONFIG_ENV, 'missing.yaml')

    with pytest.raises(FileNotFoundError, match='missing.yaml'):
        read_local_config(checkout)
