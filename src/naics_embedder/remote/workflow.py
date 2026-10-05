'''Session preparation with ordered gates and durable recovery state.'''

import json
import os
import posixpath
from pathlib import Path, PurePosixPath
from typing import Callable

from naics_embedder.remote.canonical import canonical_inputs
from naics_embedder.remote.code_manifest import is_credential_path
from naics_embedder.remote.config import effective_config
from naics_embedder.remote.push import push_code, scan_code, upload_inputs
from naics_embedder.remote.session import (
    GpuEvidence,
    RemoteInfo,
    RemoteState,
    RunRecord,
    new_id,
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
            state.status = 'ready'
            write_state(self.root, state)
            return state
