'''Verified code snapshots, conservative retries, and explicit canonical uploads.'''

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from naics_embedder.remote.canonical import InputSet
from naics_embedder.remote.code_manifest import code_entries, file_entry, git_bytes
from naics_embedder.remote.provenance import PushRecord, write_push_record
from naics_embedder.remote.session import FileEntry, RemoteState, new_id, write_state
from naics_embedder.utils.config import RemoteConfig

# -------------------------------------------------------------------------------------------------
# Snapshot and guard helpers
# -------------------------------------------------------------------------------------------------

def read_push_record(root: Path, push_id: str | None) -> PushRecord | None:
    '''Rehydrate the immutable local snapshot; absent IDs alone mean no baseline.'''
    if push_id is None:
        return None
    if Path(push_id).name != push_id or push_id in ('', '.', '..'):
        raise ValueError('unsafe push ID')
    directory = root / '.remote/pushes' / push_id
    if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe push record directory')
    entries = tuple(
        FileEntry(**item) for item in json.loads((directory / 'files.json').read_text())
    )
    provenance = json.loads((directory / 'provenance.json').read_text())
    return PushRecord(push_id, directory, entries, provenance['head_sha'], provenance['dirty'])

def scan_code(
    transport: object,
    entries: tuple[FileEntry, ...],
    cfg: RemoteConfig,
    pending: tuple[FileEntry, ...] = ()
) -> dict[str, FileEntry]:
    '''Use the stdlib pre-upload scan with precisely named previously controlled paths.'''
    response = transport.probe(
        'edits', {
            'controlled': True,
            'expected': sorted({item.path
                                for item in entries}),
            'ignore': cfg.instance_scan_ignore,
            'pending_entries': [asdict(item) for item in pending]
        }
    )
    actual = tuple(FileEntry(**item) for item in response['files'])
    if len({item.path for item in actual}) != len(actual):
        raise ValueError('duplicate instance code paths')
    return {item.path: item for item in actual}

def _edits(
    actual: dict[str, FileEntry], old: PushRecord | None, pending: PushRecord | None,
    desired: PushRecord
) -> tuple[str, ...]:
    snapshots = [
        {
            item.path: item
            for item in record.entries
        } for record in (old, pending) if record is not None
    ]
    current = {item.path: item for item in desired.entries}
    if old is None:
        snapshots.append({})
        if pending is None:
            snapshots.append(current)
    names = set(actual) | set().union(*(set(snapshot) for snapshot in snapshots))
    return tuple(
        sorted(
            name for name in names
            if not any(actual.get(name) == snapshot.get(name) for snapshot in snapshots)
        )
    )

def _delete(
    transport: object,
    actual: dict[str, FileEntry],
    names: tuple[str, ...],
    classification: str,
    cfg: RemoteConfig,
    previous: tuple[FileEntry, ...] = (),
    current: tuple[FileEntry, ...] = ()
) -> None:
    present = tuple(name for name in names if name in actual)
    if present:
        transport.probe(
            'edits', {
                'remove': list(present),
                'authorized': [asdict(actual[name]) for name in present],
                'classification': classification,
                'ignore': cfg.instance_scan_ignore,
                'previous': [asdict(item) for item in previous],
                'current_paths': [item.path for item in current]
            }
        )

def _verify(actual: dict[str, FileEntry], entries: tuple[FileEntry, ...]) -> None:
    for entry in entries:
        if actual.get(entry.path) != entry:
            raise ValueError(f'remote code hash/type/mode mismatch: {entry.path}')

# -------------------------------------------------------------------------------------------------
# Transfers and publication
# -------------------------------------------------------------------------------------------------

def push_code(
    root: Path, state: RemoteState, transport: object, cfg: RemoteConfig, force: bool
) -> PushRecord:
    '''Publish only a verified snapshot, preserving the old baseline on partial transfer.'''
    old = read_push_record(root, state.push_id)
    pending = read_push_record(root, state.pending_push_id)
    current = code_entries(root)
    head = git_bytes(root, 'rev-parse', 'HEAD').decode().strip()
    reusable = pending or old
    if reusable is not None and reusable.entries == current and reusable.head_sha == head:
        record = reusable
    else:
        parent = root / '.remote/pushes'
        existing = {path.name for path in parent.iterdir()} if parent.exists() else set()
        push_id = new_id('push', datetime.now(timezone.utc), existing, head)
        record = write_push_record(root, state.host, push_id, cfg.untracked_cap_bytes)
    controlled = tuple(
        {
            item.path: item
            for snapshot in (old, pending, record) if snapshot is not None
            for item in snapshot.entries
        }.values()
    )
    actual = scan_code(transport, controlled, cfg, pending.entries if pending is not None else ())
    edits = _edits(actual, old, pending, record)
    if edits and not force:
        raise ValueError('instance code edits: ' + ', '.join(edits))
    for name in edits:
        if name in actual and actual[name].kind not in ('file', 'symlink'):
            raise ValueError(f'unsupported instance code type: {name} ({actual[name].kind})')
    intended = {item.path for item in record.entries}
    previous = {
        item.path
        for snapshot in (old, pending) if snapshot is not None for item in snapshot.entries
    }
    deleted = tuple(sorted(previous - intended))
    new_deleted = tuple(sorted(set(edits) - previous - intended))
    journal = {
        'push_id': record.push_id,
        'previous_push_id': state.push_id,
        'prior_pending_push_id': state.pending_push_id,
        'previous_code_deletions': list(deleted),
        'force_new_code_deletions': list(new_deleted),
        'modified_or_deleted': list(edits),
        'qualified_code': [asdict(actual[name]) for name in edits if name in actual],
    }
    if edits:
        directory = root / '.remote/instance-edits' / state.session_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ('discard-' + record.push_id + '-' + uuid4().hex + '.json')
        with path.open('x') as stream:
            json.dump(journal, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    if transport.probe('transport_prerequisites', {}).get('qualified') is not True:
        raise ValueError('remote transport prerequisites are not qualified')
    state.pending_push_id = record.push_id
    write_state(root, state)
    transport.push(root, state.remote_info.repo, tuple(item.path for item in record.entries))
    previous_entries = tuple(
        item for snapshot in (old, pending) if snapshot is not None for item in snapshot.entries
    )
    _delete(transport, actual, deleted, 'previous', cfg, previous_entries, record.entries)
    if force:
        _delete(transport, actual, new_deleted, 'new', cfg)
    verified = scan_code(transport, controlled, cfg)
    _verify(verified, record.entries)
    unexpected = set(verified) - intended
    if unexpected:
        raise ValueError(
            'unexpected instance code after transfer: ' + ', '.join(sorted(unexpected))
        )
    if any(name in verified for name in (*deleted, *new_deleted)):
        raise ValueError('remote code deletion verification failed')
    if code_entries(root) != record.entries or git_bytes(root, 'rev-parse', 'HEAD'
                                                         ).decode().strip() != record.head_sha:
        raise ValueError('local code changed during transfer')
    names = tuple(sorted(path.name for path in record.directory.iterdir()))
    destination = state.remote_info.repo + '/.remote/pushes/' + record.push_id
    transport.push(record.directory, destination, names)
    # Explicit paths use the controlled scan so unowned runtime/credentials remain unopened.
    record_entries = tuple(
        file_entry(root, '.remote/pushes/' + record.push_id + '/' + name) for name in names
    )
    _verify(scan_code(transport, record_entries, cfg), record_entries)
    if code_entries(root) != record.entries or git_bytes(root, 'rev-parse', 'HEAD'
                                                         ).decode().strip() != record.head_sha:
        raise ValueError('local code changed during record transfer')
    final = scan_code(transport, controlled, cfg)
    _verify(final, record.entries)
    unexpected = set(final) - intended
    if unexpected:
        raise ValueError(
            'unexpected instance code after record transfer: ' + ', '.join(sorted(unexpected))
        )
    state.push_id = record.push_id
    state.pending_push_id = None
    write_state(root, state)
    return record

def upload_inputs(inputs: InputSet, transport: object) -> None:
    '''Copy validated input bytes to the same repo-relative locations and compare every hash.'''
    candidates = []
    for name in inputs.paths:
        if inputs.manifest.as_posix().endswith('/' + name):
            candidate = inputs.manifest
            for _ in Path(name).parts:
                candidate = candidate.parent
            if any(candidate / relative == inputs.descriptions for relative in inputs.paths):
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError('canonical input paths do not identify a unique checkout root')
    root = candidates[0]
    recorded_repo = getattr(transport, 'repo', None)
    repo = str(transport.probe('identity', {})['repo'])
    if recorded_repo is not None and repo != recorded_repo:
        raise ValueError('remote canonical input root differs from recorded repo')
    transport.push(root, repo, inputs.paths)
    response = transport.probe(
        'edits', {
            'controlled': True,
            'expected': list(inputs.paths),
            'ignore': ['__pycache__/', '*.pyc']
        }
    )
    actual = {item['path']: item for item in response['files']}
    for name, digest in inputs.hashes.items():
        item = actual.get(name)
        if item is None or item['kind'] != 'file' or item['sha256'] != digest:
            raise ValueError(f'remote canonical input hash mismatch: {name}')
