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
        if operation == 'training' and payload.get('action', 'status') == 'status':
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

@dataclass
class SyncTransport(RecordedTransport):
    root: Path = field(default_factory=Path)
    fail_next_pull: bool = False
    after_pull: object = None
    differences: tuple[str, ...] = ()

    def probe(self, operation, payload):
        from naics_embedder.remote.worker import run_probe
        return run_probe(operation, payload, self.root)

    def pull(self, source, destination, files):
        self.calls.append(('pull', source, destination, files))
        for name in files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source) / name, target)
            if self.fail_next_pull:
                self.fail_next_pull = False
                raise OSError('network loss')
        if self.after_pull:
            self.after_pull(Path(source))

    def checksum(self, mapping):
        return self.differences

@pytest.fixture
def remote_sync_fixture(tmp_path):
    import json
    import os
    import time
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import torch

    from naics_embedder.remote.session import RemoteInfo, RemoteState, write_state
    from naics_embedder.utils.config import RemoteConfig

    root, instance = tmp_path / 'mac', tmp_path / 'instance'
    root.mkdir()
    instance.mkdir()
    run = instance / 'checkpoints/run'
    run.mkdir(parents=True)
    saved = dict(epoch=0, training_run='tiny-run', state_dict={}, hyper_parameters={'seed': 1})
    torch.save(saved, run / 'last.ckpt')
    shutil.copy2(run / 'last.ckpt', run / 'epoch=0.ckpt')
    read = dict(
        event='read',
        panel='outcome',
        split='validation',
        fingerprint='fixture',
        detail=dict(epoch=0, seed=1, training_run='tiny-run')
    )
    (run / 'monitor_reads.jsonl').write_text(json.dumps(dict(mrr=0.5, read=read)) + '\n')
    (run / 'epoch_summary.jsonl').write_text(json.dumps(dict(epoch=0, mrr=0.5)) + '\n')
    for directory, name in [
        ('outputs', 'tensor'), ('logs', 'selection_log.jsonl'),
        ('.remote/segments/segment', 'exit_code')
    ]:
        path = instance / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    now = time.time()
    for path in instance.rglob('*'):
        os.utime(path, (now - 600, now - 600))
    info = RemoteInfo(
        str(instance), str(instance / 'checkpoints'), '/uv', '/python', True, 'cuda', 'fixture'
    )
    state = RemoteState(
        host='fixture',
        session_id='session',
        started_utc=datetime.now(timezone.utc),
        status='ready',
        remote_info=info
    )
    write_state(root, state)
    cfg = RemoteConfig(repo_dir=str(instance), sync_interval_seconds=3, rsync_path='/custom/rsync')
    (root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=state.session_id, remote_config=cfg.model_dump()))
    )
    return SimpleNamespace(
        root=root,
        instance=instance,
        run=run,
        state=state,
        cfg=cfg,
        now=now,
        transport=SyncTransport(root=instance),
        run_files=tuple(run.iterdir())
    )

@pytest.fixture
def remote_launch_fixture(remote_workflow_fixture, remote_resume_fixture, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace

    import torch

    from naics_embedder.remote.canonical import canonical_inputs
    from naics_embedder.remote.session import read_state
    from naics_embedder.utils.training import read_checkpoint

    env = remote_workflow_fixture
    source = remote_resume_fixture
    cfg = source.cfg.override({'dirs.output_dir': 'outputs'})
    cfg.to_yaml(str(env.root / 'conf/config.yaml'))
    (env.root / '.gitignore').write_text(
        'data/\n!conf/data/\n!conf/data/outcome_panel.yaml\n.remote/\ncheckpoints/\nlogs/\noutputs/\n'
    )
    (env.root / 'conf/data').mkdir()
    shutil.copyfile(
        Path(__file__).parents[2] / 'conf/data/outcome_panel.yaml',
        env.root / 'conf/data/outcome_panel.yaml'
    )
    (env.root / 'uv.lock').write_text('fixture lock\n')
    state = env.workflow.up('fixture', 'conf/config.yaml', [])
    info = state.remote_info
    directory = env.root / 'checkpoints' / cfg.experiment_name
    # Fixture-only construction: make the tiny saved callback name this simulated instance.
    remote_directory = info.checkpoint_base + '/' + cfg.experiment_name
    for path in directory.glob('*.ckpt'):
        saved = read_checkpoint(path)
        for callback in saved['callbacks'].values():
            if isinstance(callback, dict) and 'dirpath' in callback:
                callback['dirpath'] = remote_directory
        torch.save(saved, path)
    base_probe = env.transport.probe
    ntp = [True]
    gpu = [
        dict(
            logical_index=0,
            name='fixture GPU',
            compute_capability=[8, 0],
            total_memory_bytes=1,
            native_bf16=True,
            cuda_visible_devices=None
        )
    ]

    def probe(operation, payload):
        if operation in {'gpu', 'clock'}:
            env.transport.calls.append(('probe', operation, payload))
            if operation == 'clock':
                return {'ntp': ntp[0]}
            if isinstance(gpu[0], Exception):
                raise gpu[0]
            return gpu[0]
        return base_probe(operation, payload)

    monkeypatch.setattr(env.transport, 'probe', probe)

    class LaunchDateTime(datetime):

        @classmethod
        def now(cls, tz=None):
            return env.now[0]

    monkeypatch.setattr('naics_embedder.remote.launch.datetime', LaunchDateTime)
    loops = []
    monkeypatch.setattr('naics_embedder.remote.loop.ensure_loop', lambda *args: loops.append(args))

    def set_finished_budget():
        pass  # The unmodified actual tiny run exhausted its saved three-epoch budget.

    def set_unfinished():
        for path in directory.glob('*.ckpt'):
            saved = read_checkpoint(path)
            if path.name != 'last.ckpt' and saved['epoch'] > 1:
                path.unlink()
                continue
            saved['epoch'] = min(saved['epoch'], 1)
            torch.save(saved, path)
        for name in ('monitor_reads.jsonl', 'epoch_summary.jsonl'):
            path = directory / name
            path.write_text('\n'.join(path.read_text().splitlines()[:2]) + '\n')
        return cfg

    env.transport.calls.clear()
    return SimpleNamespace(
        **vars(env),
        cfg=cfg,
        inputs=canonical_inputs(env.root, cfg),
        info=info,
        state=read_state(env.root),
        directory=directory,
        remote_directory=remote_directory,
        overrides=[],
        loops=loops,
        ntp=ntp,
        gpu=gpu,
        set_finished_budget=set_finished_budget,
        set_unfinished=set_unfinished
    )

@pytest.fixture
def remote_finish_fixture(remote_sync_fixture, monkeypatch):
    import json
    from dataclasses import asdict
    from datetime import datetime, timezone

    import torch

    from naics_embedder.remote.code_manifest import file_entry
    from naics_embedder.remote.session import RunRecord, write_state
    from naics_embedder.remote.sync import sync_once
    from naics_embedder.remote.workflow import RemoteWorkflow
    from naics_embedder.utils.training import outcome_checkpoint, read_checkpoint

    env = remote_sync_fixture
    env.running = False
    env.interrupted = []
    env.checks = []
    env.stops = []
    source = env.instance / 'src/tiny.py'
    source.parent.mkdir()
    source.write_text('VALUE = 1\n')
    entry = file_entry(env.instance, 'src/tiny.py')
    push = env.root / '.remote/pushes/push'
    push.mkdir(parents=True)
    (push / 'files.json').write_text(json.dumps([asdict(entry)]))
    (push / 'provenance.json').write_text(json.dumps(dict(head_sha='a' * 40, dirty=False)))
    env.state.push_id = 'push'
    (env.instance / '.remote/pushes').mkdir(parents=True)
    (env.instance / '.remote/pushes/session.json').write_text(
        json.dumps(dict(session_id='session', push_id='push'))
    )
    env.state.active_segment_id = 'segment'
    saved = read_checkpoint(env.run / 'last.ckpt')
    saved['hyper_parameters']['run_settings'] = {}
    saved['stage3_supervision'] = dict(bundle_id='bundle', codebook_fingerprint='codebook')
    callback = outcome_checkpoint(env.root / 'checkpoints/run')
    saved['callbacks'] = {callback.state_key: {'dirpath': str(env.run)}}
    for path in env.run.glob('*.ckpt'):
        torch.save(saved, path)
    record = RunRecord(
        experiment='run',
        remote_directory=str(env.run),
        bundle_id='bundle',
        codebook_fingerprint='codebook',
        description_fingerprint='description',
        seed=1,
        settings={},
        constructor_controls={},
        session_id='session',
        segment_id='segment'
    )
    (env.root / '.remote/runs').mkdir()
    (env.root / '.remote/runs/run.json').write_text(record.model_dump_json())
    segment = dict(
        session_id='session',
        segment_id='segment',
        experiment_name='run',
        remote_directory=str(env.run)
    )
    (env.instance / '.remote/segments/segment/segment.json').write_text(json.dumps(segment))
    write_state(env.root, env.state)
    original = env.transport.probe

    def probe(operation, payload):
        env.transport.calls.append(('probe', operation, payload))
        if operation == 'training':
            return dict(
                running=env.running,
                sessions=['naics-train'] if env.running else [],
                process_running=env.running,
                exit_code=0 if not env.running else None
            )
        if operation == 'gpu':
            return dict(utilization=12, memory_used=34)
        return original(operation, payload)

    def interrupt(segment):
        env.interrupted.append(segment)
        env.running = False

    def checksum(mapping):
        env.checks.append(mapping)
        return env.transport.differences

    monkeypatch.setattr(env.transport, 'probe', probe)
    monkeypatch.setattr(env.transport, 'interrupt_training', interrupt)
    monkeypatch.setattr(env.transport, 'checksum', checksum)
    monkeypatch.setattr(
        'naics_embedder.remote.loop.stop_loop', lambda *args: env.stops.append(args)
    )
    monkeypatch.setattr(
        'naics_embedder.remote.loop.loop_status', lambda *args: dict(
            alive=False, pid=None, session_id=None
        )
    )
    env.workflow = RemoteWorkflow(
        env.root, env.cfg, lambda host, cfg: env.transport, lambda: datetime.now(timezone.utc)
    )
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    env.last = env.root / 'checkpoints/run/last.ckpt'
    env.transport.calls.clear()
    return env
