'''Owned detached Mac sync worker, verified by PID, start identity and command token.'''

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

from naics_embedder.remote.session import RemoteState, read_state, state_lock
from naics_embedder.remote.sync import _atomic_json, sync_once_locked
from naics_embedder.utils.config import RemoteConfig

logger = logging.getLogger(__name__)
_spawn = subprocess.Popen
_signal = os.kill

# -------------------------------------------------------------------------------------------------
# Configuration and process ownership
# -------------------------------------------------------------------------------------------------

def _session_config(root: Path, session_id: str) -> RemoteConfig:
    envelope = json.loads((root / '.remote/session-config.json').read_text())
    if envelope.get('session_id') != session_id:
        raise ValueError('effective remote configuration session mismatch')
    values = envelope.get('remote_config')
    if not isinstance(values, dict) or set(values) != set(RemoteConfig.model_fields):
        raise ValueError('missing or corrupt effective remote configuration')
    return RemoteConfig.model_validate(values)

def _process_identity(pid: int) -> dict[str, str] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'command='],
        capture_output=True,
        text=True,
        check=False,
        timeout=10
    )
    if result.returncode or not result.stdout.strip():
        return None
    row = result.stdout.strip()
    # BSD/Linux ps lstart occupies five whitespace-delimited fields.
    fields = row.split(None, 5)
    if len(fields) != 6:
        return None
    return dict(start=' '.join(fields[:5]), command=fields[5])

def _record(root: Path) -> dict | None:
    try:
        value = json.loads((root / '.remote/sync.pid').read_text())
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, ValueError):
        return None

def _owned(root: Path, session_id: str, record: dict | None) -> bool:
    if not record or record.get('session_id') != session_id:
        return False
    identity = _process_identity(record.get('pid'))
    if identity is None or identity != record.get('identity'):
        return False
    expected = record.get('argv')
    return isinstance(expected, list) and identity['command'] == ' '.join(expected) and all(
        value in expected for value in [
            'sync-loop', '--root',
            str(root), '--session-id', session_id, '--token',
            record.get('token')
        ]
    )

def loop_status(root: Path, session_id: str) -> dict[str, object]:
    record = _record(root)
    return dict(
        alive=_owned(root, session_id, record),
        pid=record.get('pid') if record else None,
        session_id=record.get('session_id') if record else None
    )

def ensure_loop(root: Path, state: RemoteState) -> None:
    '''Already-locked lifecycle entry; workflow caller owns state_lock.'''
    _session_config(root, state.session_id)
    if state.status != 'ready':
        raise ValueError('sync loop requires a ready session')
    if loop_status(root, state.session_id)['alive']:
        return
    token = uuid.uuid4().hex
    argv = [
        'caffeinate', '-i',
        str(Path(sys.executable).absolute()), '-m', 'naics_embedder.remote.worker', 'sync-loop',
        '--root',
        str(root), '--session-id', state.session_id, '--token', token
    ]
    with (root / '.remote/sync.log').open('ab') as stream:
        process = _spawn(
            argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True
        )
    identity = _process_identity(process.pid)
    if identity is None:
        raise RuntimeError('detached sync process could not be identified')
    _atomic_json(
        root / '.remote/sync.pid',
        dict(
            pid=process.pid, session_id=state.session_id, token=token, identity=identity, argv=argv
        )
    )
    if not _owned(root, state.session_id, _record(root)):
        raise RuntimeError('detached sync process command identity mismatch')

def stop_loop(root: Path, session_id: str) -> None:
    '''Own state_lock to quiesce passes; call before a workflow final transaction lock.'''
    with state_lock(root):
        _stop_loop_locked(root, session_id)

def _stop_loop_locked(root: Path, session_id: str) -> None:
    record = _record(root)
    if record and record.get('session_id') == session_id:
        # Under the pass lock this prevents every later child pass, even if caffeinate exits
        # without terminating its utility process. No wait for child registration is needed.
        _atomic_json(
            root / '.remote/sync-stop.json', dict(session_id=session_id, token=record.get('token'))
        )
        worker_pid = record.get('worker_pid')
        worker_identity = _process_identity(worker_pid) if worker_pid else None
        if worker_identity is not None and worker_identity == record.get('worker_identity'):
            expected = record.get('argv', [])[2:]
            if worker_identity['command'] == ' '.join(expected):
                try:
                    _signal(worker_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    if _owned(root, session_id, record):
        # Recheck immediately at the signal boundary rather than trusting status output.
        if _owned(root, session_id, record):
            try:
                _signal(record['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    if record and record.get('session_id') == session_id:
        (root / '.remote/sync.pid').unlink(missing_ok=True)

# -------------------------------------------------------------------------------------------------
# Local worker entry
# -------------------------------------------------------------------------------------------------

def _transport_factory(host: str, cfg: RemoteConfig) -> object:
    from naics_embedder.remote.transport import SshTransport
    return SshTransport(host, cfg.repo_dir, cfg.rsync_path)

def run_loop(
    root: Path,
    session_id: str,
    token: str,
    transport_factory: Callable = _transport_factory,
    wait: Callable = time.sleep
) -> None:
    '''Lock each pass, reload ownership/config, and retry failures at the recorded interval.'''
    logging.basicConfig(level=logging.INFO)
    registered = False
    while True:
        cfg = None
        try:
            with state_lock(root):
                state = read_state(root)
                if state is None or state.session_id != session_id or state.status in (
                    'finished', 'abandoned'
                ):
                    return
                try:
                    stopped = json.loads((root / '.remote/sync-stop.json').read_text())
                except FileNotFoundError:
                    stopped = {}
                if stopped == dict(session_id=session_id, token=token):
                    return
                record = _record(root)
                if record and (
                    record.get('session_id') != session_id or record.get('token') != token
                ):
                    return
                if record and not registered:
                    identity = _process_identity(os.getpid())
                    if identity is None or identity['command'] != ' '.join(record['argv'][2:]):
                        raise RuntimeError('local sync worker command identity mismatch')
                    record.update(worker_pid=os.getpid(), worker_identity=identity)
                    _atomic_json(root / '.remote/sync.pid', record)
                    registered = True
                cfg = _session_config(root, session_id)
                transport = transport_factory(state.host, cfg)
                if state.remote_info is not None:
                    transport.repo = state.remote_info.repo
                    transport.python = state.remote_info.python
                sync_once_locked(root, state, transport, cfg)
        except Exception:
            logger.exception('Remote sync pass failed; retrying at the next interval')
        if cfg is None:
            # Missing/corrupt config must not silently select default transport values.
            cfg = _session_config(root, session_id)
        wait(cfg.sync_interval_seconds)
