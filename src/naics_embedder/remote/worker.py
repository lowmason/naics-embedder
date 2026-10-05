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

def _inventory(root: Path, value: str) -> dict[str, object]:
    directory = _inside(root, value)
    entries = []
    if directory.exists():
        for path in sorted(directory.rglob('*')):
            if path.is_dir() and not path.is_symlink():
                continue
            _inside(root, str(path))
            metadata = path.lstat()
            link = path.is_symlink()
            digest = hashlib.sha256()
            if link:
                digest.update(os.readlink(path).encode())
            else:
                hash_block_bytes = 1024 * 1024
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(hash_block_bytes), b''):
                        digest.update(block)
            entries.append(
                {
                    'path': path.relative_to(directory).as_posix(),
                    'sha256': digest.hexdigest(),
                    'size': metadata.st_size,
                    'mode': metadata.st_mode & 0o777,
                    'kind': 'symlink' if link else 'file',
                    'target': os.readlink(path) if link else None,
                    'mtime': metadata.st_mtime
                }
            )
    return {'files': entries}

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
    if _credential_name(name):
        return True
    for pattern in ignore:
        if pattern.endswith('/'):
            if pattern.rstrip('/') in parts or fnmatch.fnmatchcase(name + '/', pattern + '*'):
                return True
        elif fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(parts[-1], pattern):
            return True
    return False

def _code_item(root: Path, name: str) -> dict[str, object] | None:
    _code_name(name)
    path = root / name
    # A replaced ancestor must never redirect a read outside the controlled tree.
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            return dict(path=name, sha256='', size=0, mode=0, kind='unsafe_ancestor', target=None)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    mode = metadata.st_mode
    target = None
    digest = hashlib.sha256()
    if stat.S_ISREG(mode):
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        kind = 'file'
        after = path.lstat()
        stable = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(after, key) != getattr(metadata, key) for key in stable):
            raise ValueError(f'code changed during inventory: {name}')
    elif stat.S_ISLNK(mode):
        target = os.readlink(path)
        digest.update(os.fsencode(target))
        try:
            resolved = path.resolve()
            safe = (
                not Path(target).is_absolute() and resolved.is_relative_to(root) and
                resolved.exists() and not _credential_name(resolved.relative_to(root).as_posix())
            )
        except (OSError, RuntimeError):
            safe = False
        kind = 'symlink' if safe else 'unsafe_symlink'
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
    root = root.resolve()
    names = set(expected)
    if root.exists():
        for directory, children, files in os.walk(root, followlinks=False):
            relative = Path(directory).relative_to(root)
            for name in children[:]:
                path = Path(directory) / name
                value = (relative / name).as_posix()
                if _generated_name(value, tuple(ignore)):
                    children.remove(name)
                elif path.is_symlink():
                    children.remove(name)
                    names.add(value)
            for name in files:
                value = (relative / name).as_posix()
                if not _generated_name(value, tuple(ignore)):
                    names.add(value)
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
    entries = [_code_item(root, name) for name in sorted(names)]
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
    root = root.resolve()
    for name in names:
        _code_name(name)
        if _credential_name(name) or (
            classification == 'new' and _generated_name(name, tuple(payload.get('ignore', [])))
        ):
            raise ValueError(f'protected code deletion: {name}')
        actual = _code_item(root, name)
        if actual is not None:
            if actual['kind'] == 'unsafe_symlink' and _pending_link(actual, previous_records):
                actual['kind'] = 'symlink'
            if actual['kind'] not in ('file', 'symlink') or actual != records[name]:
                raise ValueError(f'deletion type or bytes changed: {name}')
    for name in names:
        (root / name).unlink(missing_ok=True)
    return {'removed': names}

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
    from naics_embedder.remote.transport import OPERATIONS, safe_files
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
        return asdict(gpu_evidence())
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
            return _training_status(payload)
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
            gpu_evidence()
            subprocess.run(
                [
                    'tmux', 'new-session', '-d', '-s', session,
                    'bash ' + shlex.quote(str(script)) + ' < /dev/null'
                ],
                stdin=subprocess.DEVNULL,
                check=True,
                timeout=30
            )
            return {'running': True}
        if action == 'interrupt':
            subprocess.run(['tmux', 'send-keys', '-t', session, 'C-c'], check=True, timeout=30)
            return {'interrupted': True}
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
    args = parser.parse_args()
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
