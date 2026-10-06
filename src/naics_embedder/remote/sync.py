'''Coherent staged result pulls with recoverable promotion and cumulative Mac integrity.'''

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from naics_embedder.remote.session import (
    RemoteState,
    RunRecord,
    pull_mappings,
    read_state,
    state_lock,
    write_state,
)
from naics_embedder.remote.transport import safe_files
from naics_embedder.remote.worker import (
    _code_item_at,
    _code_item_from,
    _open_directory,
    _parent_descriptor,
    _root_descriptor,
)
from naics_embedder.utils.config import RemoteConfig

# -------------------------------------------------------------------------------------------------
# Durable generation records and integrity
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncResult:
    pulled: int
    pending: int
    last_sha256: dict[str, str]

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='record-')
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _local_hash(root: Path, name: str) -> str | None:
    safe_files((name, ))
    descriptor = _root_descriptor(root)
    try:
        item = _code_item_from(descriptor, name)
    finally:
        os.close(descriptor)
    if item is None:
        return None
    if item['kind'] != 'file':
        raise ValueError(f'unsafe Mac file: {name}')
    return item['sha256']

def _manifest(root: Path, state: RemoteState) -> dict[str, str]:
    if state.last_sync_manifest is None:
        return {}
    name = state.last_sync_manifest
    safe_files((name, ))
    if not name.startswith('.remote/pulls/') or not name.endswith('/manifest.json'):
        raise ValueError('invalid sync manifest reference')
    if _local_hash(root, name) is None:
        raise ValueError('missing successful sync manifest')
    record = json.loads((root / name).read_text())
    if record['session_id'] != state.session_id or not isinstance(record['files'], dict):
        raise ValueError('sync manifest session mismatch')
    return record['files']

def verify_local_sync_manifest(root: Path, state: RemoteState) -> None:
    '''Verify every cumulative successful Mac copy; never repair tampering silently.'''
    if (root / '.remote/pulls/pending.json').exists():
        raise ValueError('pending promotion requires locked recovery before integrity verification')
    for name, digest in _manifest(root, state).items():
        if _local_hash(root, name) != digest:
            raise ValueError(f'Mac file changed or missing since successful sync: {name}')

def _allowed_destination(root: Path, state: RemoteState, name: str) -> None:
    safe_files((name, ))
    destination = root / name
    if state.remote_info is None or not any(
        destination.is_relative_to(mapping.destination) and destination != mapping.destination
        for mapping in pull_mappings(root, state.session_id, state.remote_info)
    ):
        raise ValueError(f'promotion outside result mappings: {name}')

def _promotion_parent(root: int, name: str) -> int:
    descriptor = os.dup(root)
    try:
        for part in name.split('/')[:-1]:
            try:
                os.mkdir(part, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = _open_directory(part, descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def _verify_pending_destinations(root: Path, previous: dict, replacements: dict) -> None:
    for name, digest in previous.items():
        allowed = {digest}
        if name in replacements:
            allowed.add(replacements[name]['new'])
        if _local_hash(root, name) not in allowed:
            raise ValueError(f'Mac file changed outside pending promotion: {name}')
    for name, item in replacements.items():
        if _local_hash(root, name) not in {item['old'], item['new']}:
            raise ValueError(f'Mac file changed outside pending promotion: {name}')

def _promote_file(root: Path, name: str, item: dict) -> None:
    descriptor = _root_descriptor(root)
    source = destination = None
    try:
        source = _parent_descriptor(descriptor, item['staged'])
        destination = _promotion_parent(descriptor, name)
        staged = _code_item_at(descriptor, source, item['staged'])
        local = _code_item_at(descriptor, destination, name)
        if staged is None or staged['kind'] != 'file' or staged['sha256'] != item['new']:
            raise ValueError(f'pending staged bytes changed: {name}')
        digest = local['sha256'] if local is not None and local['kind'] == 'file' else None
        if (local is not None and local['kind'] != 'file') or digest not in {
            item['old'], item['new']
        }:
            raise ValueError(f'Mac file changed at promotion boundary: {name}')
        os.replace(
            item['staged'].split('/')[-1],
            name.split('/')[-1],
            src_dir_fd=source,
            dst_dir_fd=destination
        )
    finally:
        for opened in (source, destination, descriptor):
            if opened is not None:
                os.close(opened)

def recover_pending_promotion(root: Path, state: RemoteState) -> None:
    '''Already-locked recovery. Old/new hashes alone authorize crash replay.'''
    journal = root / '.remote/pulls/pending.json'
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    if record['session_id'] != state.session_id:
        raise ValueError('pending promotion belongs to another session')
    replacements = record['replacements']
    previous = _manifest(root, state)
    # Validate the entire transaction before completing even one replacement.
    _verify_pending_destinations(root, previous, replacements)
    for name, item in replacements.items():
        _allowed_destination(root, state, name)
        current = _local_hash(root, name)
        if current not in {item['old'], item['new']}:
            raise ValueError(f'Mac file changed outside pending promotion: {name}')
        staged = item['staged']
        safe_files((staged, ))
        if not staged.startswith(record['pass_directory'] + '/'):
            raise ValueError('staged file outside pending pass')
        if current != item['new'] and _local_hash(root, staged) != item['new']:
            raise ValueError(f'pending staged bytes missing or changed: {name}')
    for name, item in replacements.items():
        _verify_pending_destinations(root, previous, replacements)
        if _local_hash(root, name) != item['new']:
            _promote_file(root, name, item)
        if _local_hash(root, name) != item['new']:
            raise ValueError(f'Mac replacement changed during promotion: {name}')
    reference = record['pass_directory'] + '/manifest.json'
    _atomic_json(root / reference, dict(session_id=state.session_id, files=record['files']))
    state.last_sync_manifest = reference
    state.last_sync_utc = datetime.fromisoformat(record['utc'])
    state.unreachable_since = None
    write_state(root, state)
    journal.unlink()
    bind_verified_runs(root, state)

def bind_verified_runs(root: Path, state: RemoteState) -> None:
    '''Bind run IDs only from the session's published successful pull and unchanged Mac bytes.'''
    from naics_embedder.utils.training import read_checkpoint, refuse_a_resume_from_another_directory
    verify_local_sync_manifest(root, state)
    files = _manifest(root, state)
    directory = root / '.remote/runs'
    if not directory.exists():
        return
    for path in directory.glob('*.json'):
        record = RunRecord.model_validate_json(path.read_text())
        last = root / 'checkpoints' / record.experiment / 'last.ckpt'
        name = last.relative_to(root).as_posix()
        if name not in files:
            continue
        if _local_hash(root, name) != files[name]:
            raise ValueError('last checkpoint differs from successful pull evidence')
        saved = read_checkpoint(last)
        _validate_run(last.parent)
        hparams = saved.get('hyper_parameters', {})
        contract = saved.get('stage3_supervision', {})
        if (
            hparams.get('seed') != record.seed
            or hparams.get('run_settings') != record.settings or any(
                hparams.get(key) != value for key, value in record.constructor_controls.items()
            ) or contract.get('bundle_id') != record.bundle_id or contract.get(
                'codebook_fingerprint'
            ) != record.codebook_fingerprint
        ):
            raise ValueError('verified pulled checkpoint differs from persistent run identity')
        refuse_a_resume_from_another_directory(
            saved, last.parent, resolved_dirpath=record.remote_directory
        )
        run = saved['training_run']
        if record.training_run is not None and record.training_run != run:
            raise ValueError('verified pull names another training_run')
        if record.training_run is None:
            record.training_run = run
            if _local_hash(root, name) != files[name]:
                raise ValueError('last checkpoint changed during run binding')
            _atomic_json(path, record.model_dump(mode='json'))

# -------------------------------------------------------------------------------------------------
# Stable source and run qualification
# -------------------------------------------------------------------------------------------------

def _source_inventory(transport: object, source: str) -> dict[str, dict]:
    rows = transport.probe('inventory', {'path': source})['files']
    result = {}
    for item in rows:
        name = item['path']
        safe_files((name, ))
        if name in result or item['kind'] != 'file':
            raise ValueError(f'unsupported or duplicate result file: {name}')
        result[name] = item
    return result

def _validate_run(directory: Path) -> None:
    from naics_embedder.remote.canonical import _epoch, _histories
    from naics_embedder.utils.training import read_checkpoint
    try:
        saved = read_checkpoint(directory / 'last.ckpt')
        epoch = _epoch(saved.get('epoch'))
        run = saved.get('training_run')
        seed = saved.get('hyper_parameters', {}).get('seed')
        if not isinstance(run, str) or not run.strip() or isinstance(seed, bool) or not isinstance(
            seed, int
        ):
            raise ValueError('checkpoint names no run or integer seed')
        for path in directory.rglob('*.ckpt'):
            checkpoint = read_checkpoint(path)
            if (
                not isinstance(checkpoint.get('state_dict'), dict) or checkpoint.get(
                    'training_run'
                ) != run or checkpoint.get('hyper_parameters', {}).get('seed') != seed or _epoch(
                    checkpoint.get('epoch')
                ) > epoch
            ):
                raise ValueError('kept checkpoint names another run, seed or later epoch')
        _histories(directory, epoch, run, seed)
    except Exception as error:
        raise ValueError(f'incoherent checkpoint/history run: {directory}: {error}') from error

def sync_once(
    root: Path, state: RemoteState, transport: object, cfg: RemoteConfig, final: bool = False
) -> SyncResult:
    '''Own the checkout lock for the entire recovery, transfer and promotion transaction.'''
    from naics_embedder.remote.loop import _session_config
    with state_lock(root):
        current = read_state(root)
        if current is None or current.session_id != state.session_id:
            raise ValueError('sync session changed or missing')
        if current.status != 'ready':
            raise ValueError('remote sync requires a ready session; run remote up')
        if _session_config(root, current.session_id) != cfg:
            raise ValueError('effective remote configuration changed before sync')
        # Public preflight must share the mutation lock; finish/loop own their locked checks.
        return sync_once_locked(root, state, transport, cfg, final)

def sync_once_locked(
    root: Path, state: RemoteState, transport: object, cfg: RemoteConfig, final: bool = False
) -> SyncResult:
    '''Caller MUST hold state_lock. Finish/loop can share a lock without nested flock.'''
    current = read_state(root)
    if current is None or current.session_id != state.session_id:
        raise ValueError('sync session changed or missing')
    # Use durable metadata when a caller still holds a pre-pass state object.
    state.__dict__.update(current.__dict__)
    try:
        recover_pending_promotion(root, state)
        verify_local_sync_manifest(root, state)
        bind_verified_runs(root, state)
        if state.remote_info is None:
            raise ValueError('sync requires recorded remote identity')
        previous = _manifest(root, state)
        cumulative = dict(previous)
        pass_directory = '.remote/pulls/' + uuid.uuid4().hex
        replacements = {}
        inventories = {}
        pulled = pending = 0
        last_hashes = {}
        now = datetime.now(timezone.utc)
        for index, mapping in enumerate(pull_mappings(root, state.session_id, state.remote_info)):
            before = _source_inventory(transport, mapping.source)
            inventories[mapping.source] = before
            selected = set(before)
            busy = {
                name
                for name, item in before.items()
                if not final and now.timestamp() - item['mtime'] < cfg.in_flight_seconds
            }
            if index == 0:
                groups = {}
                for name in before:
                    groups.setdefault(name.split('/')[0], set()).add(name)
                for run, names in groups.items():
                    required = {
                        run + '/' + leaf
                        for leaf in ('last.ckpt', 'monitor_reads.jsonl', 'epoch_summary.jsonl')
                    }
                    if not required.issubset(names) or names & busy:
                        selected -= names
            else:
                selected -= busy
            pending += len(before) - len(selected)
            if not selected:
                continue
            stage = root / pass_directory / str(index)
            stage.mkdir(parents=True)
            transport.pull(mapping.source, stage, tuple(sorted(selected)))
            after = _source_inventory(transport, mapping.source)
            if before != after:
                raise ValueError(f'unstable source inventory during pull: {mapping.source}')
            for name in selected:
                staged_name = (stage / name).relative_to(root).as_posix()
                if _local_hash(root, staged_name) != before[name]['sha256']:
                    raise ValueError(f'unstable or truncated staged copy: {name}')
            if index == 0:
                for run in {name.split('/')[0] for name in selected}:
                    _validate_run(stage / run)
            for name in sorted(selected):
                local = (mapping.destination / name).relative_to(root).as_posix()
                digest = before[name]['sha256']
                cumulative[local] = digest
                replacements[local] = dict(
                    old=_local_hash(root, local),
                    new=digest,
                    staged=(stage / name).relative_to(root).as_posix()
                )
                if index == 0 and name.endswith('/last.ckpt'):
                    last_hashes[name.rsplit('/', 1)[0]] = digest
            pulled += len(selected)
        for source, before in inventories.items():
            if _source_inventory(transport, source) != before:
                raise ValueError(f'unstable source inventory before promotion: {source}')
        verify_local_sync_manifest(root, state)
        _atomic_json(
            root / '.remote/pulls/pending.json',
            dict(
                session_id=state.session_id,
                pass_directory=pass_directory,
                replacements=replacements,
                files=cumulative,
                utc=now.isoformat()
            )
        )
        recover_pending_promotion(root, state)
        return SyncResult(pulled, pending, last_hashes)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        if state.unreachable_since is None:
            state.unreachable_since = datetime.now(timezone.utc)
            write_state(root, state)
        raise

def verify_final_mappings(root: Path, state: RemoteState, transport: object) -> None:
    '''Require no content/type differences across the four final result mappings.'''
    if state.remote_info is None:
        raise ValueError('final checksum requires remote identity')
    for mapping in pull_mappings(root, state.session_id, state.remote_info):
        for line in transport.checksum(mapping):
            raise ValueError(f'final checksum difference: {line}')
