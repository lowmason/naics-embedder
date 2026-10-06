'''Session preparation with ordered gates and durable recovery state.'''

import hashlib
import json
import os
import posixpath
import stat
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from naics_embedder.remote.canonical import canonical_inputs
from naics_embedder.remote.code_manifest import is_credential_path
from naics_embedder.remote.config import effective_config
from naics_embedder.remote.launch import LaunchResult
from naics_embedder.remote.push import push_code, scan_code, upload_inputs
from naics_embedder.remote.session import (
    FileEntry,
    GpuEvidence,
    RemoteInfo,
    RemoteState,
    RunRecord,
    new_id,
    pull_mappings,
    read_state,
    state_lock,
    write_state,
)
from naics_embedder.utils.config import Config, RemoteConfig

# -------------------------------------------------------------------------------------------------
# Durable local orchestration records
# -------------------------------------------------------------------------------------------------

def _atomic_record(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def _stable_runs(root: Path, repo: str, checkpoint_base: str, cfg: Config) -> None:
    if (
        not PurePosixPath(repo).is_absolute()
        or not PurePosixPath(checkpoint_base).is_absolute() or '..' in PurePosixPath(
            checkpoint_base
        ).parts or not PurePosixPath(checkpoint_base).is_relative_to(PurePosixPath(repo))
    ):
        raise ValueError('instance checkpoint base must resolve under its absolute repo root')
    proposed = posixpath.normpath(
        str(PurePosixPath(repo) / cfg.dirs.checkpoint_dir / cfg.experiment_name)
    )
    if not PurePosixPath(proposed).is_relative_to(PurePosixPath(repo)):
        raise ValueError('configured checkpoint location escapes remote root')
    directory = root / '.remote/runs'
    if directory.exists():
        for path in directory.glob('*.json'):
            record = RunRecord.model_validate_json(path.read_text())
            expected = proposed if record.experiment == cfg.experiment_name else str(
                PurePosixPath(checkpoint_base) / record.experiment
            )
            if record.remote_directory != expected:
                raise ValueError(
                    f'persisted run absolute checkpoint location would change: {record.experiment}'
                )
    if not PurePosixPath(repo).is_absolute() or not PurePosixPath(checkpoint_base).is_absolute():
        raise ValueError('instance identity must report absolute directories')

# -------------------------------------------------------------------------------------------------
# Finish evidence
# -------------------------------------------------------------------------------------------------

TRAINING_STOP_TIMEOUT_SECONDS = 60.0
TRAINING_STOP_POLL_SECONDS = 0.2

class _TrainingStopTimeout(RuntimeError):
    '''A reachable instance has not stopped training within the interruption bound.'''

@dataclass(frozen=True)
class FinishResult:
    safe: bool
    abandoned: bool
    latest_checkpoint: str | None
    local_sha256: str | None
    remote_sha256: str | None

def _training(transport: object, state: RemoteState) -> dict:
    result = transport.probe(
        'training', {
            'action': 'status',
            'segment_id': state.active_segment_id
        }
    )
    if not isinstance(result.get('running'), bool):
        raise ValueError('training status could not establish process exit')
    return result

def _running(result: dict) -> bool:
    return result['running'] or bool(result.get('process_running'))

def _edit_snapshot(expected: dict, actual: dict) -> tuple[list[str], dict]:
    changed = sorted(
        name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)
    )
    snapshot = {
        'expected': [asdict(expected[name]) for name in changed if name in expected],
        'actual': [asdict(actual[name]) for name in changed if name in actual],
        'tombstones': [name for name in changed if name not in actual]
    }
    return changed, snapshot

def _verify_rescue(directory: Path, snapshot: dict) -> None:
    from naics_embedder.remote.worker import (
        _code_item_at,
        _parent_descriptor,
        _root_descriptor,
        _same_metadata,
    )
    if not snapshot['actual']:
        return
    root = _root_descriptor(directory / 'files')
    try:
        for row in snapshot['actual']:
            entry = FileEntry(**row)
            parent = _parent_descriptor(root, entry.path)
            try:
                leaf = entry.path.split('/')[-1]
                before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode):
                    # A rescue may omit an unchanged target. Hash link text without following it.
                    target = os.readlink(leaf, dir_fd=parent)
                    content = os.fsencode(target)
                    actual = FileEntry(
                        entry.path,
                        hashlib.sha256(content).hexdigest(), len(content),
                        stat.S_IMODE(before.st_mode), 'symlink', target
                    )
                else:
                    actual = FileEntry(**_code_item_at(root, parent, entry.path))
                after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if actual != entry or not _same_metadata(before, after):
                    raise ValueError('rescued instance bytes or file type changed: ' + entry.path)
            finally:
                os.close(parent)
    finally:
        os.close(root)

def _handle_edits(
    root: Path, state: RemoteState, transport: object, cfg: RemoteConfig, pull_edits: bool
) -> None:
    from naics_embedder.remote.push import read_push_record
    from naics_embedder.remote.sync import _promotion_parent
    from naics_embedder.remote.worker import _root_descriptor
    record = read_push_record(root, state.push_id)
    if record is None:
        raise ValueError('finish requires a successful push snapshot')
    expected = {item.path: item for item in record.entries}
    actual = scan_code(transport, record.entries, cfg)
    changed, snapshot = _edit_snapshot(expected, actual)
    if not changed:
        return
    parent = root / '.remote/instance-edits' / state.session_id
    try:
        descriptor = _root_descriptor(parent)
    except FileNotFoundError:
        pass
    else:
        os.close(descriptor)
    encoded = json.dumps(snapshot, sort_keys=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if parent.exists():
        for manifest in parent.glob('*/manifest.json'):
            handled = manifest.parent / 'handled.json'
            if not handled.is_file():
                continue
            stored = json.loads(manifest.read_text())
            if stored.get('snapshot') == snapshot and json.loads(handled.read_text()) == {
                'snapshot_sha256': digest
            }:
                _verify_rescue(manifest.parent, snapshot)
                return
    if not pull_edits:
        raise ValueError('instance code edits: ' + ', '.join(changed))
    directory = parent / uuid.uuid4().hex
    descriptor = _root_descriptor(root)
    rescue_parent = None
    try:
        rescue_parent = _promotion_parent(
            descriptor, (parent / 'manifest.json').relative_to(root).as_posix()
        )
        os.mkdir(directory.name, dir_fd=rescue_parent)
    finally:
        if rescue_parent is not None:
            os.close(rescue_parent)
        os.close(descriptor)
    files = tuple(row['path'] for row in snapshot['actual'])
    if any(row['kind'] not in ('file', 'symlink') for row in snapshot['actual']):
        raise ValueError('unsupported instance edit type')
    if files:
        transport.pull(state.remote_info.repo, directory / 'files', files)
    _verify_rescue(directory, snapshot)
    _, rechecked = _edit_snapshot(expected, scan_code(transport, record.entries, cfg))
    if rechecked != snapshot:
        raise ValueError('instance edits changed during rescue; new rescue required')
    # Exclusive evidence files preserve partial rescues and every prior accepted generation.
    with (directory / 'manifest.json').open('x') as stream:
        json.dump(dict(snapshot=snapshot, **snapshot), stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    with (directory / 'handled.json').open('x') as stream:
        json.dump({'snapshot_sha256': digest}, stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def _latest_checkpoint(root: Path, state: RemoteState, transport: object) -> FinishResult:
    from naics_embedder.remote.sync import _local_hash, _manifest, _source_inventory
    from naics_embedder.utils.training import read_checkpoint
    if state.active_segment_id is None:
        return FinishResult(True, False, None, None, None)
    from naics_embedder.remote.transport import safe_files
    from naics_embedder.remote.worker import _probe_text
    safe_files((state.active_segment_id, ))
    if '/' in state.active_segment_id:
        raise ValueError('invalid active segment component')
    segment_names = [
        '.remote/segments/' + state.active_segment_id + '/segment.json', 'outputs/remote/'
        + state.session_id + '/segments/' + state.active_segment_id + '/segment.json'
    ]
    content = next(
        (value for name in segment_names if (value := _probe_text(root, name)) is not None), None
    )
    if content is None:
        raise ValueError('missing latest segment provenance')
    segment = json.loads(content)
    experiment = segment.get('experiment_name')
    if not isinstance(experiment, str) or not experiment or '/' in experiment:
        raise ValueError('latest segment names no valid experiment')
    safe_files((experiment, ))
    record_content = _probe_text(root, '.remote/runs/' + experiment + '.json')
    if record_content is None:
        raise ValueError('latest segment has no persistent run record')
    record = RunRecord.model_validate_json(record_content)
    expected_directory = state.remote_info.checkpoint_base + '/' + record.experiment
    if (
        record.experiment != experiment or record.remote_directory
        != expected_directory or segment.get('session_id') != state.session_id or segment.get(
            'segment_id'
        ) != state.active_segment_id or segment.get('remote_directory') != record.remote_directory
    ):
        raise ValueError('latest segment differs from persistent run identity')
    last = root / 'checkpoints' / record.experiment / 'last.ckpt'
    remote = _source_inventory(transport, record.remote_directory).get('last.ckpt')
    local = _local_hash(root, last.relative_to(root).as_posix())
    if local is None and remote is None:
        return FinishResult(True, False, None, None, None)
    if (
        remote is None or local != remote['sha256'] or _manifest(root, state).get(
            last.relative_to(root).as_posix()
        ) != local
    ):
        raise ValueError('latest checkpoint local/remote successful hashes differ')
    if record.training_run is None or read_checkpoint(last).get(
        'training_run'
    ) != record.training_run:
        raise ValueError('latest checkpoint has no verified bound training_run')
    return FinishResult(True, False, str(last), local, remote['sha256'])

# -------------------------------------------------------------------------------------------------
# Workflow
# -------------------------------------------------------------------------------------------------

class RemoteWorkflow:
    '''Prepare code and canonical inputs through an injected synchronous transport.'''

    def __init__(
        self, root: Path, remote_cfg: RemoteConfig, transport_factory: Callable, clock: Callable
    ):
        self.root = root.resolve()
        self.remote_cfg = remote_cfg
        self.transport_factory = transport_factory
        self.clock = clock

    def up(
        self, host: str, config_path: str, overrides: list[str], force: bool = False
    ) -> RemoteState:
        with state_lock(self.root):
            # Transport construction qualifies local tools before any SSH or instance mutation.
            transport = self.transport_factory(host, self.remote_cfg)
            cfg = effective_config(self.root, config_path, overrides)
            try:
                inputs = canonical_inputs(self.root, cfg)
            except ValueError as error:
                raise ValueError(
                    f'supervision.manifest_path canonical validation: {error}'
                ) from error
            for name in inputs.paths:
                if is_credential_path(name):
                    raise ValueError(f'credential path forbidden in canonical inputs: {name}')
            old = read_state(self.root)
            if old is not None and old.last_sync_manifest is not None:
                from naics_embedder.remote.sync import bind_verified_runs
                bind_verified_runs(self.root, old)
            new_session = old is None or old.host != host or old.status in ('finished', 'abandoned')
            loss = None
            if old is not None and old.host != host and old.status not in ('finished', 'abandoned'):
                if not force:
                    raise ValueError(
                        f'unfinished session {old.session_id}; last sync {old.last_sync_utc}; '
                        'use --force to record loss risk'
                    )
                loss = {
                    'session_id': old.session_id,
                    'host': old.host,
                    'last_sync_utc': old.last_sync_utc.isoformat() if old.last_sync_utc else None,
                    'risk': 'Anything after the last successful sync may be lost.'
                }
            if old is not None and not new_session and old.remote_info is not None:
                transport.repo = old.remote_info.repo
                transport.python = old.remote_info.python or None
            identity = transport.probe('identity', {})
            repo = str(identity['repo'])
            base = str(identity['checkpoint_base'])
            _stable_runs(self.root, repo, base, cfg)
            if old is not None and not new_session and old.status == 'ready' and (
                identity.get('session_id') != old.session_id
                or identity.get('push_id') != old.push_id
            ):
                if not force:
                    raise ValueError(
                        'ready host session marker is missing or different; '
                        'use --force for a new session'
                    )
                new_session = True
                loss = {
                    'session_id': old.session_id,
                    'host': old.host,
                    'risk': 'Reused host has no matching session marker; unsynced results may be lost.'
                }
            if transport.probe('training', {'action': 'status'})['running']:
                raise ValueError('training tmux session naics-train is running')
            now = self.clock()
            if new_session:
                known = set()
                if old is not None:
                    known.add(old.session_id)
                session = new_id('session', now, known)
                state = RemoteState(host=host, session_id=session, started_utc=now)
                if old is not None and old.host == host:
                    state.push_id = old.push_id
                    state.pending_push_id = old.pending_push_id
            else:
                state = old.model_copy(deep=True)
                state.status = 'preparing'
            transport.repo = repo
            state.remote_info = RemoteInfo(repo, base, '', '', False, '', '')
            write_state(self.root, state)
            _atomic_record(
                self.root / '.remote/session-config.json', {
                    'session_id': state.session_id,
                    'remote_config': self.remote_cfg.model_dump()
                }
            )
            if loss is not None:
                _atomic_record(self.root / '.remote' / ('loss-' + state.session_id + '.json'), loss)
            record = push_code(self.root, state, transport, self.remote_cfg, force)
            bootstrap = transport.probe('bootstrap', {})
            if bootstrap.get('gpu_evidence') is not None:
                evidence = dict(bootstrap['gpu_evidence'])
                evidence['compute_capability'] = tuple(evidence['compute_capability'])
                bootstrap['gpu_evidence'] = GpuEvidence(**evidence)
            info = RemoteInfo(**bootstrap)
            if not info.ntp or info.repo != repo or info.checkpoint_base != base:
                raise ValueError('bootstrap identity or NTP verification failed')
            state.remote_info = info
            transport.repo = info.repo
            transport.python = info.python
            write_state(self.root, state)
            upload_inputs(inputs, transport)
            remote = transport.probe('canonical', {'config': cfg.model_dump(mode='json')})
            if remote.get('hashes') != inputs.hashes or set(remote.get('paths', [])) != set(
                inputs.paths
            ):
                raise ValueError('remote canonical validation hashes differ')
            # Revalidate after transfer so a concurrent Mac input edit cannot become ready.
            if canonical_inputs(self.root, cfg).hashes != inputs.hashes:
                raise ValueError('local canonical inputs changed during transfer')
            actual = scan_code(transport, record.entries, self.remote_cfg)
            expected = {item.path: item for item in record.entries}
            changed = sorted(
                name for name in actual.keys() | expected.keys()
                if actual.get(name) != expected.get(name)
            )
            if changed:
                raise ValueError('remote code changed before ready: ' + ', '.join(changed))
            transport.probe(
                'write_record', {
                    'path': '.remote/pushes/session.json',
                    'record': {
                        'session_id': state.session_id,
                        'push_id': state.push_id
                    }
                }
            )
            marker = transport.probe('identity', {})
            if marker.get('session_id') != state.session_id or marker.get(
                'push_id'
            ) != state.push_id:
                raise ValueError('remote session marker verification failed')
            _atomic_record(
                self.root / '.remote/session-inputs.json', {
                    'session_id': state.session_id,
                    'push_id': state.push_id,
                    'paths': list(inputs.paths),
                    'hashes': inputs.hashes
                }
            )
            state.status = 'ready'
            write_state(self.root, state)
            return state

    def train(self, resume: bool, config_path: str, overrides: list[str]) -> LaunchResult:
        '''Resolve current inputs and launch while owning the workflow transaction lock.'''
        from naics_embedder.remote.launch import launch_training
        from naics_embedder.remote.loop import _session_config
        with state_lock(self.root):
            state = read_state(self.root)
            if state is None or state.status != 'ready' or state.remote_info is None:
                raise ValueError('remote train requires a ready session; run remote up')
            remote_cfg = _session_config(self.root, state.session_id)
            transport = self.transport_factory(state.host, remote_cfg)
            transport.repo = state.remote_info.repo
            transport.python = state.remote_info.python
            cfg = effective_config(self.root, config_path, overrides)
            inputs = canonical_inputs(self.root, cfg)
            return launch_training(
                self.root, state, transport, cfg, config_path, overrides, inputs, resume
            )

    def finish(
        self, stop_training: bool = False, pull_edits: bool = False, abandon: bool = False
    ) -> FinishResult:
        '''Quiesce the owned worker and verify final results before closing a session.'''
        from naics_embedder.remote.loop import _record, _session_config, stop_loop
        from naics_embedder.remote.sync import (
            recover_pending_promotion,
            sync_once_locked,
            verify_final_mappings,
            verify_local_sync_manifest,
        )
        if abandon and (stop_training or pull_edits):
            raise ValueError('abandon cannot combine with stop-training or pull-edits')
        initial = read_state(self.root)
        if initial is None:
            raise ValueError('finish requires a recorded session')
        stop_loop(self.root, initial.session_id)
        with state_lock(self.root):
            state = read_state(self.root)
            if state is None or state.session_id != initial.session_id:
                raise ValueError('session changed during worker shutdown')
            ownership = _record(self.root)
            if (self.root / '.remote/sync.pid').exists() and (
                ownership is None or ownership.get('session_id') == state.session_id
            ):
                raise ValueError('sync worker ownership changed or was not quiesced')
            if abandon:
                if state.status == 'finished':
                    raise ValueError('cannot abandon a finished session')
                _atomic_record(
                    self.root / '.remote' / ('abandon-' + state.session_id + '.json'),
                    dict(
                        session_id=state.session_id,
                        host=state.host,
                        utc=self.clock().isoformat(),
                        last_sync_utc=state.last_sync_utc.isoformat()
                        if state.last_sync_utc else None,
                        risk='Anything after the last successful sync may be lost.'
                    )
                )
                state.status = 'abandoned'
                write_state(self.root, state)
                return FinishResult(False, True, None, None, None)
            if state.status not in ('ready', 'finished') or state.remote_info is None:
                raise ValueError('finish requires a ready session')
            cfg = _session_config(self.root, state.session_id)
            transport = self.transport_factory(state.host, cfg)
            transport.repo = state.remote_info.repo
            transport.python = state.remote_info.python
            try:
                recover_pending_promotion(self.root, state)
                try:
                    verify_local_sync_manifest(self.root, state)
                except ValueError as error:
                    raise ValueError('Mac copy changed: ' + str(error)) from error
                identity = transport.probe('identity', {})
                if (
                    identity.get('repo') != state.remote_info.repo or identity.get(
                        'checkpoint_base'
                    ) != state.remote_info.checkpoint_base or identity.get('session_id')
                    != state.session_id or identity.get('push_id') != state.push_id
                ):
                    raise ValueError('finish instance identity differs')
                training = _training(transport, state)
                if _running(training):
                    if not stop_training:
                        raise ValueError('training is running; use stop-training')
                    if state.active_segment_id is None:
                        raise ValueError('running training has no owned segment to interrupt')
                    transport.interrupt_training(state.active_segment_id)
                    deadline = time.monotonic() + TRAINING_STOP_TIMEOUT_SECONDS
                    while _running(_training(transport, state)):
                        if time.monotonic() >= deadline:
                            raise _TrainingStopTimeout(
                                'training stop timeout; no forced kill permitted'
                            )
                        time.sleep(
                            min(TRAINING_STOP_POLL_SECONDS, max(0, deadline - time.monotonic()))
                        )
                result = sync_once_locked(self.root, state, transport, cfg, final=True)
                if result.pending or (self.root / '.remote/pulls/pending.json').exists():
                    raise ValueError('final sync has pending files or promotion')
                _handle_edits(self.root, state, transport, cfg, pull_edits)
                latest = _latest_checkpoint(self.root, state, transport)
                verify_final_mappings(self.root, state, transport)
                _handle_edits(self.root, state, transport, cfg, False)
                verify_local_sync_manifest(self.root, state)
                if (self.root / '.remote/pulls/pending.json').exists():
                    raise ValueError('final promotion is pending')
                if _running(_training(transport, state)):
                    raise ValueError('training is running after final verification')
                state.status = 'finished'
                write_state(self.root, state)
                return latest
            except _TrainingStopTimeout:
                raise
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                if state.unreachable_since is None:
                    state.unreachable_since = self.clock()
                    write_state(self.root, state)
                raise

    def status(self) -> dict[str, object]:
        '''Observe local and remote evidence without locking, recovering or changing records.'''
        from naics_embedder.remote.loop import _session_config, loop_status
        errors = []
        try:
            state = read_state(self.root)
        except Exception as error:
            return dict(status='invalid state', errors=[str(error)])
        if state is None:
            return dict(status='no session', errors=[])
        result = dict(
            status=state.status,
            host=state.host,
            session_id=state.session_id,
            active_segment_id=state.active_segment_id,
            last_sync_utc=state.last_sync_utc.isoformat() if state.last_sync_utc else None,
            unreachable_since=state.unreachable_since.isoformat()
            if state.unreachable_since else None,
            unreachable=False,
            pending=None,
            training=None,
            gpu=None,
            loop=None,
            pending_promotion=(self.root / '.remote/pulls/pending.json').exists(),
            errors=errors
        )
        if result['pending_promotion']:
            errors.append('pending promotion requires recovery')
        try:
            result['loop'] = loop_status(self.root, state.session_id)
        except Exception as error:
            errors.append('sync loop: ' + str(error))
        try:
            cfg = _session_config(self.root, state.session_id)
            if state.remote_info is None:
                raise ValueError('missing remote identity')
            transport = self.transport_factory(state.host, cfg)
            transport.repo = state.remote_info.repo
            transport.python = state.remote_info.python
        except Exception as error:
            errors.append('configuration: ' + str(error))
            return result
        for name, operation, payload in [
            ('training', 'training', {
                'action': 'status',
                'segment_id': state.active_segment_id
            }), ('gpu', 'gpu', {
                'action': 'status'
            })
        ]:
            try:
                result[name] = transport.probe(operation, payload)
            except Exception as error:
                errors.append(name + ': ' + str(error))
                result['unreachable'] = True
        try:
            result['pending'] = sum(
                len(transport.checksum(mapping))
                for mapping in pull_mappings(self.root, state.session_id, state.remote_info)
            )
        except Exception as error:
            errors.append('pending dry run: ' + str(error))
            result['unreachable'] = True
        return result
