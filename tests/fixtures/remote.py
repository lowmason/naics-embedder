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

@dataclass(frozen=True)
class RunnerCall:
    args: list[str]
    kwargs: dict[str, object]

@dataclass
class RecordedTransportRunner:
    calls: list[RunnerCall] = field(default_factory=list)
    replies: list[subprocess.CompletedProcess] = field(default_factory=list)

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(RunnerCall(args, kwargs))
        if self.replies:
            return self.replies.pop(0)
        stdout = b'rsync  version 3.5.1  protocol version 32\n' if '--version' in args else b'{}'
        return subprocess.CompletedProcess(args, 0, stdout, b'')

@pytest.fixture
def recorded_transport_runner() -> RecordedTransportRunner:
    return RecordedTransportRunner()

@dataclass(frozen=True)
class RemoteResumeFixture:
    root: Path
    cfg: Config
    inputs: object
    directory: Path
    remote_directory: str
    transport: RecordedTransport

@pytest.fixture
def remote_resume_fixture(remote_repo, trained_seeds):
    from naics_embedder.remote.canonical import canonical_inputs

    root = remote_repo.root
    manifest = root / 'data/bundles' / trained_seeds.bundle.root.name / 'manifest.json'
    shutil.copytree(trained_seeds.bundle.root, manifest.parent)
    shutil.copyfile(
        trained_seeds.cfg.data_loader.streaming.descriptions_parquet,
        root / 'data/naics_descriptions.parquet'
    )
    original = trained_seeds.directory(1)
    directory = root / 'checkpoints/remote-fixture'
    shutil.copytree(original, directory)
    cfg = trained_seeds.cfg.override(
        {
            'seed': 1,
            'experiment_name': 'remote-fixture',
            'dirs.checkpoint_dir': 'checkpoints',
            'supervision.manifest_path': str(manifest.relative_to(root)),
            'data_loader.streaming.descriptions_parquet': 'data/naics_descriptions.parquet',
            'training.trainer.accelerator': 'cpu',
            'training.trainer.precision': '32',
        }
    )
    # The original real callback directory is the simulated instance identity. Never rewrite it.
    return RemoteResumeFixture(
        root, cfg, canonical_inputs(root, cfg), directory, str(original.resolve()),
        RecordedTransport()
    )

@dataclass
class WorkflowTransport(RecordedTransport):
    root: Path = field(default_factory=Path)
    running: bool = False
    bootstrap_error: bool = False
    canonical_error: bool = False
    push_error: bool = False
    fail_after: str | None = None

    def probe(self, operation, payload):
        from naics_embedder.remote.worker import run_probe
        self.calls.append(('probe', operation, payload))
        if operation == 'training':
            return {'running': self.running}
        if operation == 'transport_prerequisites':
            return {'qualified': True}
        if operation == 'bootstrap':
            if self.bootstrap_error:
                raise RuntimeError('bootstrap failed')
            return dict(
                repo=str(self.root),
                checkpoint_base=str(self.root / 'checkpoints'),
                uv='/usr/bin/uv',
                python='/usr/bin/python',
                ntp=True,
                accelerator='cpu',
                gpu='fixture'
            )
        result = run_probe(operation, payload, self.root)
        if operation == 'canonical' and self.canonical_error:
            result['hashes'] = {}
        return result

    def push(self, source, destination, files):
        self.calls.append(('push', source, destination, files))
        target = Path(destination)
        for name in files:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            original = source / name
            if path.is_symlink():
                path.unlink()
            if original.is_symlink():
                path.symlink_to(original.readlink())
            else:
                shutil.copy2(original, path)
            if self.push_error and (self.fail_after is None or self.fail_after == name):
                raise RuntimeError('interrupted transfer')

@pytest.fixture
def remote_workflow_fixture(remote_repo, tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from naics_embedder.remote.workflow import RemoteWorkflow
    from naics_embedder.utils.config import RemoteConfig

    instance = tmp_path / 'instance'
    instance.mkdir()
    transport = WorkflowTransport(root=instance)
    now = [datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)]
    import naics_embedder.remote.push as push_module

    class FixtureDateTime(datetime):

        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(push_module, 'datetime', FixtureDateTime)
    cfg = RemoteConfig(repo_dir=str(instance))
    workflow = RemoteWorkflow(remote_repo.root, cfg, lambda host, cfg: transport, lambda: now[0])
    return SimpleNamespace(
        root=remote_repo.root,
        repo=remote_repo,
        instance=instance,
        transport=transport,
        workflow=workflow,
        now=now
    )
