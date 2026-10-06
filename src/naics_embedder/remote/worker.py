'''Internal fixed JSON probes; no evaluation or data-generation entry points.'''

import argparse
import fnmatch
import hashlib
import json
import os
import posixpath
import shlex
import stat
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from naics_embedder.remote.session import GpuEvidence, RemoteInfo

TMUX_SESSION = 'naics-train'

# -------------------------------------------------------------------------------------------------
# Capability and filesystem probes
# -------------------------------------------------------------------------------------------------

def gpu_evidence(torch_module: object | None = None) -> GpuEvidence:
    '''Qualify native BF16 on logical CUDA device zero, without precision fallback.'''
    try:
        if torch_module is None:
            import torch as torch_module
        cuda = torch_module.cuda
        if not cuda.is_available():
            raise ValueError('CUDA unavailable')
        with cuda.device(0):
            if not cuda.is_bf16_supported(including_emulation=False):
                raise ValueError('selected device has no native BF16 support')
            properties = cuda.get_device_properties(0)
            return GpuEvidence(
                0, properties.name, (properties.major, properties.minor), properties.total_memory,
                True, os.environ.get('CUDA_VISIBLE_DEVICES')
            )
    except Exception as error:
        raise RuntimeError(
            'CUDA native BF16 qualification failed; inspect the driver, locked '
            f'PyTorch environment and CUDA_VISIBLE_DEVICES: {error}'
        ) from error

def _inside(root: Path, value: str) -> Path:
    root = root.resolve()
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'path escapes remote root: {value}')
    return path

def _result_names(root: int, prefix: str = '') -> set[str]:
    names = set()
    for leaf in os.listdir(root):
        name = prefix + leaf
        if '.rsync-partial' in Path(name).parts or _credential_name(name):
            continue
        metadata = os.stat(leaf, dir_fd=root, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_directory(leaf, root)
            try:
                if not _same_metadata(metadata, os.fstat(child)):
                    raise ValueError(f'result directory changed during inventory: {name}')
                names.update(_result_names(child, name + '/'))
            finally:
                os.close(child)
        else:
            names.add(name)
    return names

def _inventory(root: Path, value: str) -> dict[str, object]:
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to(root)
    if '..' in path.parts or _credential_name(str(path)):
        raise ValueError('unsafe result inventory root')
    try:
        descriptor = _root_descriptor(root / path)
    except FileNotFoundError:
        return {'files': []}
    try:
        entries = []
        for name in sorted(_result_names(descriptor)):
            parent = _parent_descriptor(descriptor, name)
            try:
                metadata = os.stat(name.split('/')[-1], dir_fd=parent, follow_symlinks=False)
                item = _code_item_at(descriptor, parent, name)
                after = os.stat(name.split('/')[-1], dir_fd=parent, follow_symlinks=False)
                if not _same_metadata(metadata, after):
                    raise ValueError(f'result changed during inventory: {name}')
                item['mtime'] = metadata.st_mtime
                entries.append(item)
            finally:
                os.close(parent)
        return {'files': entries}
    finally:
        os.close(descriptor)

def _code_name(name: str) -> None:
    path = Path(name)
    if not name or '\0' in name or path.is_absolute() or any(
        part in ('', '.', '..') for part in name.split('/')
    ):
        raise ValueError(f'unsafe code path: {name!r}')

def _credential_name(name: str) -> bool:
    return any(
        part in {'.git', '.ssh', '.aws', '.env'} or part.startswith('.env.') or part.lower() in {
            'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519', 'identity'
        } or part.lower().endswith(('.pem', '.key', '.p12', '.pfx', '.ppk'))
        for part in Path(name).parts
    )

def _generated_name(name: str, ignore: tuple[str, ...]) -> bool:
    parts = Path(name).parts
    if parts[0] in {'.git', '.venv', 'data', 'checkpoints', 'logs', 'outputs', '.remote'}:
        return True
    # The locked editable build emits this project-specific directory (already Git-ignored).
    if parts[:2] == ('src', 'naics_embedder.egg-info'):
        return True
    if '.rsync-partial' in parts:
        return True
    if _credential_name(name):
        return True
    for pattern in ignore:
        if pattern.endswith('/'):
            if pattern.rstrip('/') in parts or fnmatch.fnmatchcase(name + '/', pattern + '*'):
                return True
        elif fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(parts[-1], pattern):
            return True
    return False

def _same_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    stable = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    return all(getattr(left, key) == getattr(right, key) for key in stable)

def _open_directory(name: str, parent: int | None = None) -> int:
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)

def _root_descriptor(root: Path) -> int:
    # Anchor every absolute component, including the repository's ancestors.
    descriptor = _open_directory('/')
    try:
        for part in root.absolute().parts[1:]:
            child = _open_directory(part, descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def _parent_descriptor(root: int, name: str) -> int:
    descriptor = os.dup(root)
    try:
        for part in name.split('/')[:-1]:
            child = _open_directory(part, descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def _safe_link(root: int, name: str, target: str) -> bool:
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
    if posixpath.isabs(target):
        return False
    for _ in range(40):
        try:
            _code_name(candidate)
            if _credential_name(candidate):
                return False
            parts = candidate.split('/')
            descriptor = os.dup(root)
            replacement = None
            try:
                for index, part in enumerate(parts):
                    metadata = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        link = os.readlink(part, dir_fd=descriptor)
                        after = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                        if posixpath.isabs(link) or not _same_metadata(metadata, after):
                            return False
                        replacement = posixpath.normpath(
                            posixpath.join(*parts[:index], link, *parts[index + 1:])
                        )
                        break
                    if index < len(parts) - 1:
                        child = _open_directory(part, descriptor)
                        if not _same_metadata(metadata, os.fstat(child)):
                            os.close(child)
                            return False
                        os.close(descriptor)
                        descriptor = child
                    elif not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                        return False
            finally:
                os.close(descriptor)
            if replacement is None:
                return True
            candidate = replacement
        except (OSError, ValueError):
            return False
    return False

def _code_item_at(root: int, parent: int, name: str) -> dict[str, object] | None:
    leaf = name.split('/')[-1]
    try:
        metadata = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    mode = metadata.st_mode
    target = None
    digest = hashlib.sha256()
    if stat.S_ISREG(mode):
        # NONBLOCK prevents a regular-to-FIFO race from hanging before fstat qualification.
        descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            if not _same_metadata(metadata, os.fstat(descriptor)):
                raise ValueError(f'code changed during inventory: {name}')
            for block in iter(lambda: os.read(descriptor, 1024 * 1024), b''):
                digest.update(block)
            after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not _same_metadata(metadata, after) or not _same_metadata(
                metadata, os.fstat(descriptor)
            ):
                raise ValueError(f'code changed during inventory: {name}')
        finally:
            os.close(descriptor)
        kind = 'file'
    elif stat.S_ISLNK(mode):
        target = os.readlink(leaf, dir_fd=parent)
        if not _same_metadata(metadata, os.stat(leaf, dir_fd=parent, follow_symlinks=False)):
            raise ValueError(f'code changed during inventory: {name}')
        digest.update(os.fsencode(target))
        kind = 'symlink' if _safe_link(root, name, target) else 'unsafe_symlink'
    else:
        kind = 'directory' if stat.S_ISDIR(mode) else 'special'
    return dict(
        path=name,
        sha256=digest.hexdigest(),
        size=len(os.fsencode(target)) if target is not None else metadata.st_size,
        mode=stat.S_IMODE(mode),
        kind=kind,
        target=target
    )

def _code_item_from(root: int, name: str) -> dict[str, object] | None:
    _code_name(name)
    try:
        parent = _parent_descriptor(root, name)
    except FileNotFoundError:
        return None
    except OSError:
        return dict(path=name, sha256='', size=0, mode=0, kind='unsafe_ancestor', target=None)
    try:
        return _code_item_at(root, parent, name)
    finally:
        os.close(parent)

def _code_item(root: Path, name: str) -> dict[str, object] | None:
    descriptor = _root_descriptor(root)
    try:
        return _code_item_from(descriptor, name)
    finally:
        os.close(descriptor)

def _code_names(root: int, ignore: tuple[str, ...], prefix: str = '') -> set[str]:
    names = set()
    for leaf in os.listdir(root):
        name = prefix + leaf
        if _generated_name(name, ignore):
            continue
        metadata = os.stat(leaf, dir_fd=root, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_directory(leaf, root)
            try:
                if not _same_metadata(metadata, os.fstat(child)):
                    raise ValueError(f'code directory changed during inventory: {name}')
                names.update(_code_names(child, ignore, name + '/'))
            finally:
                os.close(child)
        else:
            names.add(name)
    return names

def _pending_link(item: dict[str, object], pending: dict[str, dict]) -> bool:
    expected = pending.get(item['path'])
    if expected is None or expected.get('kind') != 'symlink':
        return False
    if any(
        item.get(key) != expected.get(key) for key in ('path', 'sha256', 'size', 'mode', 'target')
    ):
        return False
    candidate = item['path']
    for _ in range(len(pending) + 1):
        record = pending.get(candidate)
        if record is not None and record.get('kind') == 'symlink':
            target = record.get('target')
            if not isinstance(target, str) or Path(target).is_absolute():
                return False
            candidate = posixpath.normpath(str(Path(candidate).parent / target))
            try:
                _code_name(candidate)
            except ValueError:
                return False
            if _credential_name(candidate):
                return False
            continue
        if record is not None and record.get('kind') == 'file':
            return True
        parts = Path(candidate).parts
        replaced = False
        for length in range(1, len(parts)):
            prefix = '/'.join(parts[:length])
            record = pending.get(prefix)
            if record is not None and record.get('kind') == 'symlink':
                target = record.get('target')
                if not isinstance(target, str) or Path(target).is_absolute():
                    return False
                candidate = posixpath.normpath(
                    str(Path(prefix).parent / target / Path(*parts[length:]))
                )
                try:
                    _code_name(candidate)
                except ValueError:
                    return False
                if _credential_name(candidate):
                    return False
                replaced = True
                break
        if not replaced:
            return any(name.startswith(candidate + '/') for name in pending)
    return False

def _controlled_inventory(root: Path, payload: dict[str, object]) -> dict[str, object]:
    expected = payload.get('expected', [])
    ignore = payload.get('ignore', [])
    if not isinstance(expected, list) or not all(isinstance(name, str) for name in expected):
        raise ValueError('expected must contain code paths')
    if not isinstance(ignore, list) or not all(isinstance(pattern, str) for pattern in ignore):
        raise ValueError('ignore must contain patterns')
    for name in expected:
        _code_name(name)
        if _credential_name(name):
            raise ValueError('credential path cannot be controlled code')
    names = set(expected)
    pending_entries = payload.get('pending_entries', [])
    if not isinstance(pending_entries, list) or not all(
        isinstance(item, dict) for item in pending_entries
    ):
        raise ValueError('invalid pending code manifest')
    pending = {item['path']: item for item in pending_entries}
    if len(pending) != len(pending_entries):
        raise ValueError('duplicate pending code paths')
    for name in pending:
        _code_name(name)
        if _credential_name(name):
            raise ValueError('credential path cannot be pending code')
    try:
        descriptor = _root_descriptor(root)
    except FileNotFoundError:
        return {'files': []}
    try:
        names.update(_code_names(descriptor, tuple(ignore)))
        entries = [_code_item_from(descriptor, name) for name in sorted(names)]
    finally:
        os.close(descriptor)
    for item in entries:
        if item is not None and item['kind'] == 'unsafe_symlink' and _pending_link(item, pending):
            item['kind'] = 'symlink'
    return {'files': [item for item in entries if item is not None]}

def _owned_remove(root: Path, payload: dict[str, object]) -> dict[str, object]:
    names = payload['remove']
    authorized = payload.get('authorized')
    classification = payload.get('classification')
    if not isinstance(names, list) or not isinstance(authorized, list):
        raise ValueError('owned deletion requires an explicit manifest')
    if classification not in ('previous', 'new'):
        raise ValueError('unknown deletion classification')
    records = {item['path']: item for item in authorized}
    if len(records) != len(authorized) or set(names) != set(records):
        raise ValueError('deletion names must exactly match authorized manifest')
    previous_records = {}
    if classification == 'previous':
        previous = payload.get('previous')
        current = payload.get('current_paths')
        if not isinstance(previous, list) or not isinstance(current, list):
            raise ValueError('deletion requires previous manifest and current paths')
        previous_names = set()
        for item in previous:
            if not isinstance(item, dict) or not isinstance(item.get('path'), str):
                raise ValueError('invalid previous manifest')
            _code_name(item['path'])
            previous_names.add(item['path'])
            previous_records[item['path']] = item
        for name in current:
            _code_name(name)
        if not set(names).issubset(previous_names - set(current)):
            raise ValueError('deletion is outside previous manifest minus current paths')
    descriptor = _root_descriptor(root)
    try:
        for name in names:
            _code_name(name)
            if _credential_name(name) or (
                classification == 'new' and _generated_name(name, tuple(payload.get('ignore', [])))
            ):
                raise ValueError(f'protected code deletion: {name}')
            try:
                parent = _parent_descriptor(descriptor, name)
            except FileNotFoundError:
                continue
            try:
                leaf = name.split('/')[-1]
                actual = _code_item_at(descriptor, parent, name)
                if actual is None:
                    continue
                if actual['kind'] == 'unsafe_symlink' and _pending_link(actual, previous_records):
                    actual['kind'] = 'symlink'
                if actual['kind'] not in ('file', 'symlink') or actual != records[name]:
                    raise ValueError(f'deletion type or bytes changed: {name}')
                # Recheck this exact entry immediately before its anchored mutation.
                before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                checked = _code_item_at(descriptor, parent, name)
                if checked is not None and checked['kind'] == 'unsafe_symlink' and _pending_link(
                    checked, previous_records
                ):
                    checked['kind'] = 'symlink'
                if checked != actual or not _same_metadata(
                    before, os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                ):
                    raise ValueError(f'deletion type or bytes changed: {name}')
                os.unlink(leaf, dir_fd=parent)
            finally:
                os.close(parent)
    finally:
        os.close(descriptor)
    return {'removed': names}

def _launch_path(root: Path, value: str) -> Path:
    '''Refuse redirected write paths before training can create a result or token cache.'''
    path = _inside(root, value)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('links are forbidden in launch write paths')
    ancestor = next(parent for parent in path.parents if parent.exists())
    descriptor = _root_descriptor(ancestor)
    os.close(descriptor)
    return path

def _training_status(payload: dict[str, object]) -> dict[str, object]:
    try:
        result = subprocess.run(
            ['tmux', 'list-sessions', '-F', '#{session_name}'],
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
    except FileNotFoundError:
        return {'running': False, 'sessions': []}
    diagnostic = result.stderr.strip()
    missing_socket = (
        diagnostic.startswith('error connecting to /') and diagnostic.endswith(
            ' (No such file or directory)'
        ) and '\n' not in diagnostic
    )
    stopped = result.returncode == 1 and (
        missing_socket or any(
            message in diagnostic for message in ('no server running', 'no sessions')
        )
    )
    if result.returncode and not stopped:
        raise RuntimeError('unable to inspect tmux training sessions: ' + result.stderr)
    sessions = [name for name in result.stdout.splitlines() if name == TMUX_SESSION]
    return {'running': bool(sessions), 'sessions': sessions}

def _probe_text(root: Path, name: str) -> str | None:
    descriptor = _root_descriptor(root)
    parent = opened = None
    try:
        try:
            parent = _parent_descriptor(descriptor, name)
            opened = os.open(name.split('/')[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(os.fstat(opened).st_mode):
            raise ValueError('unsafe probe evidence: ' + name)
        with os.fdopen(opened, 'r') as stream:
            opened = None
            return stream.read(1024 * 1024)
    finally:
        for value in (opened, parent, descriptor):
            if value is not None:
                os.close(value)

def _training_observation(root: Path, payload: dict[str, object]) -> dict[str, object]:
    result = _training_status(payload)
    processes = subprocess.run(
        ['ps', '-ww', '-axo', 'pid=,command='],
        capture_output=True,
        text=True,
        check=True,
        timeout=10
    )
    pids = []
    for row in processes.stdout.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2:
            continue
        # ps flattens argv without shell escaping; quotes in arguments remain literal.
        # Whitespace matching conservatively retains possible training processes.
        command = fields[1].split()
        # uv and its Python entry-point child can outlive a tmux server. Both must exit.
        if any(
            Path(item).name == 'naics-embedder' and index
            + 1 < len(command) and command[index + 1] == 'train'
            for index, item in enumerate(command)
        ):
            pids.append(int(fields[0]))
    result.update(
        process_running=bool(pids),
        process_pids=pids,
        running=result['running'] or bool(pids),
        exit_code=None
    )
    segment = payload.get('segment_id')
    if segment is not None:
        if not isinstance(segment, str) or not segment or any(
            char not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'
            for char in segment
        ):
            raise ValueError('invalid segment_id')
        code = _probe_text(root, '.remote/segments/' + segment + '/exit_code')
        if code is not None:
            result['exit_code'] = int(code.strip())
    return result

def _interrupt_training(root: Path, payload: dict[str, object]) -> dict[str, object]:
    segment = payload.get('segment_id')
    if not isinstance(segment, str) or not segment or any(
        char not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'
        for char in segment
    ):
        raise ValueError('invalid segment_id')
    record = json.loads(_probe_text(root, '.remote/segments/' + segment + '/segment.json') or '{}')
    marker = json.loads(_probe_text(root, '.remote/pushes/session.json') or '{}')
    if (
        record.get('segment_id') != segment or not record.get('session_id') or record['session_id']
        != marker.get('session_id') or record.get('push_id') != marker.get('push_id')
    ):
        raise ValueError('interrupt requires the owned session segment')
    panes = subprocess.run(
        ['tmux', 'list-panes', '-t', TMUX_SESSION, '-F', '#{pane_start_command}'],
        capture_output=True,
        text=True,
        check=True,
        timeout=10
    )
    expected = 'bash ' + shlex.quote(str(root / '.remote/segments' / segment / 'launch.sh'))
    if panes.stdout.strip() != expected + ' < /dev/null':
        raise ValueError('training tmux is not the recorded owned wrapper')
    subprocess.run(['tmux', 'send-keys', '-t', TMUX_SESSION, 'C-c'], check=True, timeout=10)
    return {'interrupted': True}

def _gpu_status() -> dict[str, object]:
    result = subprocess.run(
        [
            'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10
    )
    devices = []
    for row in result.stdout.splitlines():
        index, utilization, used, total = (int(value.strip()) for value in row.split(','))
        devices.append(
            dict(
                index=index, utilization=utilization, memory_used_mib=used, memory_total_mib=total
            )
        )
    return {'devices': devices}

def _remove_code(root: Path, names: tuple[str, ...]) -> dict[str, object]:
    for name in names:
        path = _inside(root, name)
        path.unlink(missing_ok=True)
    return {'removed': list(names)}

def _clock() -> dict[str, object]:
    result = subprocess.run(
        ['timedatectl', 'show', '--property=NTPSynchronized', '--value'],
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )
    return {'ntp': result.stdout.strip() == 'yes', 'utc': datetime.now(timezone.utc).isoformat()}

def bootstrap_info(root: Path, uv: str) -> dict[str, object]:
    evidence = gpu_evidence()
    ntp = _clock()['ntp']
    if not ntp:
        raise RuntimeError('NTP synchronization required; timedatectl must report yes')
    return asdict(
        RemoteInfo(
            str(root.resolve()), str((root / 'checkpoints').resolve()), uv, sys.executable, True,
            'cuda', evidence.name, evidence
        )
    )

# -------------------------------------------------------------------------------------------------
# Fixed worker operations
# -------------------------------------------------------------------------------------------------

def run_probe(operation: str, payload: dict[str, object], root: Path) -> dict[str, object]:
    '''Dispatch only named operations with constrained path and record destinations.'''
    from naics_embedder.remote.transport import OPERATIONS, _qualified_gpu, safe_files
    if operation not in OPERATIONS:
        raise ValueError(f'unknown remote operation: {operation}')
    root = root.resolve()
    if operation == 'identity':
        marker = root / '.remote/pushes/session.json'
        session = None
        push_id = None
        if marker.exists():
            _inside(root, str(marker))
            if marker.is_symlink() or not marker.is_file():
                raise ValueError('unsafe session marker')
            marker_record = json.loads(marker.read_text())
            session = marker_record.get('session_id')
            push_id = marker_record.get('push_id')
        return {
            'repo': str(root),
            'checkpoint_base': str((root / 'checkpoints').resolve()),
            'session_id': session,
            'push_id': push_id
        }
    if operation == 'gpu':
        if payload.get('action') == 'status':
            return _gpu_status()
        visibility = payload.get('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'))
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            if visibility is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            elif isinstance(visibility, str):
                os.environ['CUDA_VISIBLE_DEVICES'] = visibility
            else:
                raise ValueError('CUDA visibility must be a string or null')
            return asdict(gpu_evidence())
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous
    if operation == 'clock':
        return _clock()
    if operation == 'bootstrap':
        return bootstrap_info(root, str(payload['uv']))
    if operation == 'inventory':
        return _inventory(root, str(payload.get('path', '.')))
    if operation == 'write_record':
        path = _inside(root, str(payload['path']))
        relative = path.relative_to(root)
        if relative.parts[:2] not in {('.remote', 'segments'), ('.remote', 'pushes')}:
            raise ValueError('record must be under .remote/segments or .remote/pushes')
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload.get('immutable'):
            with path.open('x') as stream:
                stream.write(json.dumps(payload['record'], indent=2) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return {'written': str(relative)}
        temporary = path.with_name(path.name + '.tmp')
        _inside(root, str(temporary))
        temporary.write_text(json.dumps(payload['record'], sort_keys=True) + '\n')
        temporary.replace(path)
        return {'written': str(relative)}
    if operation == 'edits':
        if 'remove' in payload:
            names = tuple(payload['remove'])
            if 'authorized' in payload:
                return _owned_remove(root, payload)
            safe_files(names, code=True)
            return _remove_code(root, names)
        if payload.get('controlled'):
            return _controlled_inventory(root, payload)
        entries = _inventory(root, '.')['files']
        return {'files': entries}
    if operation == 'launch_lock':
        segment = str(payload['segment_id'])
        safe_files((segment, ))
        if '/' in segment or segment in {'.', '..'}:
            raise ValueError('segment_id must be a component')
        lock = _inside(root, '.remote/launch.lock')
        lock.parent.mkdir(parents=True, exist_ok=True)
        action = payload.get('action', 'acquire')
        if action == 'acquire':
            with lock.open('x') as stream:
                stream.write(segment)
        elif action == 'release':
            if lock.read_text() != segment:
                raise ValueError('launch lock belongs to another segment')
            lock.unlink()
        else:
            raise ValueError('unknown launch lock action')
        return {'locked': action == 'acquire'}
    if operation == 'training':
        action = payload.get('action', 'status')
        if action == 'status':
            return _training_observation(root, payload)
        if action == 'preflight':
            from naics_embedder.remote.canonical import canonical_inputs, resume_plan
            from naics_embedder.remote.launch import _mapped_path
            from naics_embedder.utils.config import Config, OutcomePanelConfig, load_config
            from naics_embedder.utils.training import refuse_a_fresh_start_into_a_used_directory
            cfg = Config.model_validate(payload['config'])
            directory = _launch_path(root, str(payload['remote_directory']))
            panel_path = root / 'conf/data/outcome_panel.yaml'
            if not panel_path.is_file():
                raise ValueError('missing pushed outcome monitor configuration')
            panel = load_config(OutcomePanelConfig, panel_path)
            for value, mapping in (
                (cfg.dirs.output_dir, 'outputs'), (cfg.dirs.log_dir, 'logs'), (
                    panel.selection_log, 'logs'
                ), (cfg.data_loader.tokenization.output_path, 'data')
            ):
                _mapped_path(str(root), value, mapping)
                _launch_path(root, value)
            if str(directory) != str(Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name):
                raise ValueError('preflight run directory differs')
            inputs = canonical_inputs(root, cfg)
            if payload['resume']:
                plan = resume_plan(root, cfg, inputs, str(directory))
                if plan.finished:
                    raise ValueError('remote run is finished; do not relaunch')
            else:
                refuse_a_fresh_start_into_a_used_directory(directory)
            return {'qualified': True}
        segment = str(payload['segment_id'])
        if not segment or any(
            char not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'
            for char in segment
        ):
            raise ValueError('invalid segment_id')
        session = TMUX_SESSION
        if action == 'launch':
            script = _inside(root, str(payload['script']))
            expected = root / '.remote/segments' / segment
            if not script.resolve().is_relative_to(expected.resolve()) or not script.is_file():
                raise ValueError('launch script must exist under its owned segment directory')
            if not _clock()['ntp']:
                raise RuntimeError('NTP synchronization required immediately before launch')
            record = json.loads((expected / 'segment.json').read_text())
            if (root / '.remote/launch.lock').read_text() != segment:
                raise ValueError('launch requires the owned instance lock')
            evidence = run_probe(
                'gpu', {'cuda_visible_devices': record['cuda_visible_devices']}, root
            )
            if not _qualified_gpu(evidence) or json.loads(json.dumps(evidence)
                                                          ) != record['gpu_evidence']:
                raise ValueError('launch native BF16 evidence or CUDA visibility changed')
            from naics_embedder.remote.launch import _wrapper
            info = RemoteInfo(
                str(root), str(root / 'checkpoints'), record['argv'][0], sys.executable, True,
                'cuda', evidence['name']
            )
            if script.read_text() != _wrapper(
                info, tuple(record['argv']), str(expected / 'exit_code'),
                record['cuda_visible_devices']
            ):
                raise ValueError('launch wrapper differs from recorded command or CUDA visibility')
            if _training_status({})['running']:
                raise ValueError('training tmux already running')
            subprocess.run(
                [
                    'tmux', 'new-session', '-d', '-s', session,
                    'bash ' + shlex.quote(str(script)) + ' < /dev/null'
                ],
                stdin=subprocess.DEVNULL,
                cwd=root,
                check=True,
                timeout=30
            )
            return {'running': True}
        if action == 'interrupt':
            return _interrupt_training(root, payload)
        raise ValueError('unknown training operation')
    if operation == 'canonical':
        from naics_embedder.remote.canonical import canonical_inputs
        from naics_embedder.utils.config import Config
        inputs = canonical_inputs(root, Config.model_validate(payload['config']))
        return {'paths': list(inputs.paths), 'hashes': inputs.hashes}
    raise ValueError('transport_prerequisites uses the pre-upload system probe')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('operation')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--uv')
    parser.add_argument('--session-id')
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.operation == 'sync-loop':
        if not args.session_id or not args.token:
            parser.error('sync-loop requires session-id and token')
        from naics_embedder.remote.loop import run_loop
        run_loop(args.root, args.session_id, args.token)
        return
    try:
        payload = json.load(sys.stdin)
        if args.uv is not None:
            payload['uv'] = args.uv
        print(json.dumps(run_probe(args.operation, payload, args.root)))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

if __name__ == '__main__':
    main()
