'''Isolated checkout and recorded transport fixtures for the remote workflow.'''

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from naics_embedder.utils.config import Config
from tests.fixtures.supervision import build_reference_bundle

# -------------------------------------------------------------------------------------------------
# Fixture checkout and fake transport
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteRepo:
    root: Path
    config: Config
    manifest: Path

@dataclass
class RecordedTransport:
    calls: list[tuple] = field(default_factory=list)
    replies: list[dict[str, object]] = field(default_factory=list)

    def probe(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(('probe', operation, payload))
        return self.replies.pop(0) if self.replies else {}

    def push(self, source: Path, destination: str, files: tuple[str, ...]) -> None:
        self.calls.append(('push', source, destination, files))

    def pull(self, source: str, destination: Path, files: tuple[str, ...]) -> None:
        self.calls.append(('pull', source, destination, files))

    def remove_code(self, paths: tuple[str, ...]) -> None:
        self.calls.append(('remove_code', paths))

    def checksum(self, mapping: object) -> tuple[str, ...]:
        self.calls.append(('checksum', mapping))
        return ()

    def launch(self, script: str, segment_id: str) -> None:
        self.calls.append(('launch', script, segment_id))

    def interrupt_training(self, segment_id: str) -> None:
        self.calls.append(('interrupt_training', segment_id))

@pytest.fixture
def recorded_transport() -> RecordedTransport:
    return RecordedTransport()

@pytest.fixture
def remote_repo(tmp_path: Path) -> RemoteRepo:
    '''A temp Git checkout with fixture canonical inputs and a relative training config.'''
    root = tmp_path / 'repo'
    (root / 'conf').mkdir(parents=True)
    (root / 'src').mkdir()
    (root / 'src/tiny.py').write_text('VALUE = 1\n')
    (root / 'CLAUDE.md').write_text('Fixture repository\n')
    (root / 'AGENTS.md').symlink_to('CLAUDE.md')
    shutil.copyfile(Path(__file__).parents[2] / 'conf/remote.yaml', root / 'conf/remote.yaml')
    original = build_reference_bundle(tmp_path / 'inputs')
    target = root / 'data/bundles' / original.parent.name
    shutil.copytree(original.parent, target)
    descriptions = root / 'data/naics_descriptions.parquet'
    shutil.copyfile(tmp_path / 'inputs/reference_descriptions.parquet', descriptions)
    manifest = target / 'manifest.json'
    cfg = Config().override(
        {
            'supervision.manifest_path': manifest.relative_to(root).as_posix(),
            'data_loader.streaming.descriptions_parquet': descriptions.relative_to(root).as_posix(),
        }
    )
    cfg.to_yaml(str(root / 'conf/config.yaml'))
    (root / '.gitignore').write_text('data/\n.remote/\noutputs/remote/\n')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(
        [
            'git',
            '-C',
            str(root),
            '-c',
            'user.name=Fixture',
            '-c',
            'user.email=fixture@example.test',
            'commit',
            '-qm',
            'fixture',
        ],
        check=True
    )
    return RemoteRepo(root, cfg, manifest)
