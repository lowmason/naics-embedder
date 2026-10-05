'''Internal fixed JSON probes; no evaluation or data-generation entry points.'''

import argparse
import hashlib
import json
import os
import shlex
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
        return {'repo': str(root), 'checkpoint_base': str((root / 'checkpoints').resolve())}
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
            safe_files(names, code=True)
            return _remove_code(root, names)
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
