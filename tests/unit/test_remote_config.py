from pathlib import Path

import pytest
from pydantic import ValidationError

from naics_embedder.remote.config import effective_config, load_remote_config, relative_path
from naics_embedder.utils.config import RemoteConfig

def test_defaults():
    cfg = load_remote_config(Path('conf/remote.yaml'))
    assert cfg == RemoteConfig()
    assert cfg.repo_dir == '~/naics-embedder'
    assert (cfg.sync_interval_seconds, cfg.in_flight_seconds, cfg.untracked_cap_bytes) == (
        600, 120, 10000000
    )
    assert cfg.pulled_directories == ['checkpoints', 'outputs', 'logs', '.remote/segments']
    assert cfg.instance_scan_ignore == [
        '__pycache__/', '*.pyc', '.pytest_cache/', '.ipynb_checkpoints/'
    ]
    assert cfg.rsync_path is None

@pytest.mark.parametrize(
    'name', ['/tmp/escape', '../escape', 'a/../../escape', 'a\\escape', 'a\nfile']
)
def test_repo_relative_paths_refuse_escape(tmp_path, name):
    with pytest.raises(ValueError, match='repo-relative'):
        relative_path(tmp_path, name)

def test_symlink_escape(tmp_path):
    (tmp_path / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match='repo-relative'):
        relative_path(tmp_path, 'outside/file')

def test_missing_and_unknown_remote_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_remote_config(tmp_path / 'missing')
    path = tmp_path / 'remote.yaml'
    path.write_text('unknown: true\n')
    with pytest.raises(ValidationError):
        load_remote_config(path)

@pytest.mark.parametrize(
    'directories', [
        ['logs'], ['checkpoints', 'outputs', 'logs', '/tmp'],
        ['checkpoints', 'outputs', 'logs', '.remote/segments', 'logs']
    ]
)
def test_pull_roots_are_contractual(directories):
    with pytest.raises(ValidationError):
        RemoteConfig(pulled_directories=directories)

@pytest.mark.parametrize('experiment', ['', '.', '..', 'a/b', 'a\\b', 'a\nb', 'a\x7fb', 'a\x85b'])
def test_experiment_is_single_component(tmp_path, experiment):
    (tmp_path / 'config.yaml').write_text('{}')
    with pytest.raises(ValueError, match='experiment'):
        effective_config(tmp_path, 'config.yaml', [f'experiment_name={experiment}'])

def test_effective_config_last_override_wins(tmp_path):
    (tmp_path / 'config.yaml').write_text('seed: 1\n')
    cfg = effective_config(
        tmp_path, 'config.yaml', ['seed=2', 'seed=3', 'training.learning_rate=1e-4']
    )
    assert cfg.seed == 3
    assert cfg.training.learning_rate == 1e-4
    with pytest.raises(ValueError, match='override'):
        effective_config(tmp_path, 'config.yaml', ['seed'])
    with pytest.raises(ValueError, match='repo-relative'):
        effective_config(tmp_path, '../config.yaml', [])
    with pytest.raises(FileNotFoundError):
        effective_config(tmp_path, 'missing.yaml', [])

def test_remote_repo_has_relative_canonical_inputs(remote_repo):
    cfg = effective_config(remote_repo.root, 'conf/config.yaml', [])
    assert cfg.supervision.manifest_path == remote_repo.manifest.relative_to(remote_repo.root
                                                                             ).as_posix()
    assert (remote_repo.root / cfg.data_loader.streaming.descriptions_parquet).is_file()
    assert (remote_repo.root / 'AGENTS.md').read_text() == 'Fixture repository\n'
