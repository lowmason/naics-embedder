'''Bounded SSH/rsync transport with fixed remote operations and explicit file lists.'''

import inspect
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

from naics_embedder.remote.session import GpuEvidence, PullMapping
from naics_embedder.remote.worker import _credential_name, _generated_name

# -------------------------------------------------------------------------------------------------
# Tool and path boundaries
# -------------------------------------------------------------------------------------------------

OPERATIONS = frozenset(
    {
        'identity', 'transport_prerequisites', 'bootstrap', 'canonical', 'inventory', 'training',
        'gpu', 'write_record', 'launch_lock', 'edits', 'clock'
    }
)
SSH_OPTIONS = [
    '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15'
]
COMMAND_TIMEOUT = 120
TRANSFER_TIMEOUT = 1800

# These probes precede the package upload and deliberately use only the system Python.
IDENTITY_CODE = """import json,sys; from pathlib import Path
p=json.load(sys.stdin); r=Path(p['repo']).expanduser().resolve()
b=Path(p.get('checkpoint_base',str(r/'checkpoints'))).expanduser()
if not b.is_absolute(): b=r/b
m=r/'.remote/pushes/session.json'
if m.is_symlink() or not m.resolve().is_relative_to(r): raise ValueError('unsafe session marker')
s=json.loads(m.read_text()) if m.exists() else {}
print(json.dumps({'repo':str(r),'checkpoint_base':str(b.resolve()),'session_id':s.get('session_id'),'push_id':s.get('push_id')}))"""
PREREQUISITE_CODE = """import json,re,shutil,subprocess,sys
json.load(sys.stdin)
def qualified():
    if not shutil.which('rsync'): return False
    p=subprocess.run(['rsync','--version'],capture_output=True,text=True,timeout=30)
    m=re.search(r'^rsync\\s+version (\\d+)\\.(\\d+)\\.(\\d+)',p.stdout)
    return bool(p.returncode==0 and m and tuple(map(int,m.groups())) >= (3,2,0))
if not qualified():
    p=subprocess.run(['sudo','-n','apt-get','update'],stdout=sys.stderr,stderr=sys.stderr,timeout=300)
    if p.returncode: raise SystemExit('rsync prerequisite: noninteractive sudo/apt update failed')
    p=subprocess.run(['sudo','-n','apt-get','install','-y','rsync'],stdout=sys.stderr,stderr=sys.stderr,timeout=300)
    if p.returncode: raise SystemExit('rsync prerequisite: distro installation failed')
if not qualified(): raise SystemExit('rsync prerequisite: distro GNU rsync must be >=3.2')
print(json.dumps({'rsync':shutil.which('rsync'),'qualified':True}))"""

def gnu_rsync_version(banner: str) -> tuple[int, int, int]:
    '''Require GNU rsync 3.2+; compatibility banners do not qualify.'''
    match = re.search(r'^rsync\s+version (\d+)\.(\d+)\.(\d+)', banner, re.MULTILINE)
    if 'openrsync' in banner.lower() or match is None:
        raise ValueError('GNU rsync >=3.2 required; brew install rsync')
    version = tuple(int(part) for part in match.groups())
    if version < (3, 2, 0):
        raise ValueError('GNU rsync >=3.2 required; brew install rsync')
    return version

def safe_files(files: tuple[str, ...], code: bool = False) -> bytes:
    '''Validate relative file names and send them as NUL-delimited bytes.'''
    for name in files:
        path = PurePosixPath(name)
        if not name or not path.parts or path.is_absolute() or '..' in path.parts or '\0' in name:
            raise ValueError(f'unsafe relative path: {name!r}')
        if any(part in {'.ssh', '.aws', '.git'} for part in path.parts):
            raise ValueError(f'credential or git path refused: {name}')
        if code and _credential_name(name):
            raise ValueError(f'credential code deletion refused: {name}')
        if code and _generated_name(name, ()):
            raise ValueError(f'protected code deletion: {name}')
    return b''.join(name.encode() + b'\0' for name in files)

def _contained_files(root: Path, files: tuple[str, ...]) -> None:
    for name in files:
        if not (root / name).resolve().is_relative_to(root.resolve()):
            raise ValueError(f'path escapes transfer root: {name}')

def _system_probe_code(operation: str) -> str:
    from naics_embedder.remote.worker import (
        TMUX_SESSION,
        _code_item,
        _code_item_at,
        _code_item_from,
        _code_name,
        _code_names,
        _controlled_inventory,
        _credential_name,
        _generated_name,
        _inside,
        _inventory,
        _open_directory,
        _owned_remove,
        _parent_descriptor,
        _pending_link,
        _remove_code,
        _root_descriptor,
        _safe_link,
        _same_metadata,
        _training_status,
    )
    preamble = (
        'import fnmatch,hashlib,json,os,posixpath,stat,subprocess,sys\n'
        'from pathlib import Path, PurePosixPath\n'
    )
    preamble += f'TMUX_SESSION = {TMUX_SESSION!r}\n'
    functions = {
        'inventory': [_inside, _inventory],
        'training': [_training_status],
        'edits': [
            safe_files, _inside, _remove_code, _inventory, _code_name, _credential_name,
            _generated_name, _same_metadata, _open_directory, _root_descriptor, _parent_descriptor,
            _safe_link, _code_item_at, _code_item_from, _code_item, _code_names, _pending_link,
            _controlled_inventory, _owned_remove
        ]
    }[operation]
    code = preamble + '\n'.join(inspect.getsource(function) for function in functions)
    code += '\np=json.load(sys.stdin)\n'
    if operation == 'inventory':
        code += "print(json.dumps(_inventory(Path(p['repo']),p.get('path','.'))))"
    elif operation == 'edits':
        code += "r=Path(p['repo'])\n"
        code += "if 'remove' in p:\n"
        code += " if 'authorized' in p: result=_owned_remove(r,p)\n"
        code += " else:\n  safe_files(tuple(p['remove']),code=True)\n"
        code += "  result=_remove_code(r,tuple(p['remove']))\n"
        code += "else: result=_controlled_inventory(r,p) if p.get('controlled') else _inventory(r,'.')\n"
        code += 'print(json.dumps(result))'
    else:
        code += 'print(json.dumps(_training_status(p)))'
    return code

def _qualified_gpu(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        evidence = GpuEvidence(**value)
        return (
            evidence.logical_index == 0 and evidence.native_bf16 is True and isinstance(
                evidence.name, str
            ) and bool(evidence.name) and len(evidence.compute_capability) == 2 and all(
                isinstance(part, int) for part in evidence.compute_capability
            ) and isinstance(evidence.total_memory_bytes, int) and evidence.total_memory_bytes > 0
            and (
                evidence.cuda_visible_devices is None or isinstance(
                    evidence.cuda_visible_devices, str
                )
            )
        )
    except (TypeError, ValueError):
        return False

def _text(value: str | bytes | None) -> str:
    return value.decode() if isinstance(value, bytes) else value or ''

# -------------------------------------------------------------------------------------------------
# Real instance transport
# -------------------------------------------------------------------------------------------------

class SshTransport:
    '''Only fixed remote probes and explicit rsync roots cross this boundary.'''

    def __init__(
        self, host: str, repo: str, rsync_path: str | None, runner: Callable = subprocess.run
    ):
        if not host or host.startswith('-') or any(char.isspace() for char in host):
            raise ValueError('invalid SSH host')
        self.host = host
        self.repo = repo
        self.runner = runner
        self.python: str | None = None
        self.rsync = rsync_path or shutil.which('rsync') or 'rsync'
        gnu_rsync_version(_text(self._run([self.rsync, '--version']).stdout))

    def _run(
        self, args: list[str], input: bytes | None = None, timeout: int = COMMAND_TIMEOUT
    ) -> subprocess.CompletedProcess:
        result = self.runner(
            args,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=timeout,
            check=False
        )
        if result.returncode:
            diagnostic = _text(result.stderr)
            if 'REMOTE HOST IDENTIFICATION HAS CHANGED' in diagnostic:
                host = self.host.rsplit('@', 1)[-1]
                raise RuntimeError(
                    f'changed SSH host key; verify it manually, then ssh-keygen -R {host}'
                )
            raise RuntimeError(f'command failed ({result.returncode}): {diagnostic}')
        return result

    def _ssh(self, argv: list[str], payload: dict[str, object],
             timeout: int = COMMAND_TIMEOUT) -> dict[str, object]:
        result = self._run(
            ['ssh', *SSH_OPTIONS, self.host, shlex.join(argv)],
            json.dumps(payload).encode(), timeout
        )
        return json.loads(_text(result.stdout))

    def probe(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        if operation not in OPERATIONS:
            raise ValueError(f'unknown remote operation: {operation}')
        payload = dict(payload, repo=self.repo)
        if operation == 'identity':
            result = self._ssh(['python3', '-c', IDENTITY_CODE], payload)
            self.repo = str(result['repo'])
            return result
        if operation == 'transport_prerequisites':
            return self._ssh(['python3', '-c', PREREQUISITE_CODE], payload, 700)
        if operation == 'bootstrap':
            result = self._ssh(
                ['bash', self.repo + '/src/naics_embedder/remote/bootstrap.sh', self.repo], payload,
                1800
            )
            evidence = result.get('gpu_evidence')
            if result.get('accelerator') != 'cuda' or not _qualified_gpu(evidence):
                raise RuntimeError('bootstrap must qualify CUDA native BF16 evidence')
            self.python = str(result['python'])
            return result
        if operation == 'edits':
            return self._ssh(
                [self.python or 'python3', '-c',
                 _system_probe_code(operation)], payload
            )
        if self.python is None and operation in {'inventory', 'training'}:
            if operation == 'training' and payload.get('action', 'status') != 'status':
                raise RuntimeError('bootstrap required before training mutations')
            return self._ssh(['python3', '-c', _system_probe_code(operation)], payload)
        if self.python is None:
            raise RuntimeError('bootstrap required before package probes')
        return self._ssh(
            [self.python, '-m', 'naics_embedder.remote.worker', operation, '--root', self.repo],
            payload
        )

    def _transfer(
        self,
        source: str,
        destination: str,
        files: tuple[str, ...] | None,
        checksum: bool,
        dry: bool = False
    ) -> subprocess.CompletedProcess:
        args = [self.rsync, '-rlpt', '--protect-args', '--partial-dir=.rsync-partial']
        data = None
        if files is not None:
            data = safe_files(files)
            args += ['--from0', '--files-from=-']
        if checksum:
            args.append('--checksum')
        if dry:
            args += ['--dry-run', '--itemize-changes', '--out-format=%i|%n']
        args += [
            '-e',
            shlex.join(['ssh', *SSH_OPTIONS]), '--',
            source.rstrip('/') + '/',
            destination.rstrip('/') + '/'
        ]
        return self._run(args, data, TRANSFER_TIMEOUT)

    def push(self, source: Path, destination: str, files: tuple[str, ...]) -> None:
        safe_files(files)
        _contained_files(source, files)
        self._transfer(str(source), self.host + ':' + destination, files, True)

    def pull(self, source: str, destination: Path, files: tuple[str, ...]) -> None:
        safe_files(files)
        _contained_files(destination, files)
        self._transfer(self.host + ':' + source, str(destination), files, True)

    def remove_code(self, paths: tuple[str, ...]) -> None:
        safe_files(paths, code=True)
        self.probe('edits', {'remove': list(paths)})

    def checksum(self, mapping: PullMapping) -> tuple[str, ...]:
        result = self._transfer(
            self.host + ':' + mapping.source, str(mapping.destination), None, True, dry=True
        )
        return tuple(
            line.split('|', 1)[1] for line in _text(result.stdout).splitlines()
            if re.match(r'^[<>ch.*][^|]{10}\|', line)
        )

    def launch(self, script: str, segment_id: str) -> None:
        self.probe('training', {'action': 'launch', 'script': script, 'segment_id': segment_id})

    def interrupt_training(self, segment_id: str) -> None:
        self.probe('training', {'action': 'interrupt', 'segment_id': segment_id})

# -------------------------------------------------------------------------------------------------
# Isolated filesystem transport
# -------------------------------------------------------------------------------------------------

class LocalTransport:
    '''Use real GNU rsync locally, with all process/GPU behavior injected.'''

    def __init__(self, root: Path, process: object, rsync_path: str):
        self.root = root.resolve()
        self.process = process
        self.rsync = rsync_path
        gnu_rsync_version(
            _text(
                subprocess.run(
                    [rsync_path, '--version'], capture_output=True, check=True, timeout=30
                ).stdout
            )
        )

    def _path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            try:
                path.relative_to(self.root)
            except ValueError:
                path = self.root / str(path).lstrip('/')
        else:
            path = self.root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('destination escapes local instance root')
        return resolved

    def probe(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        from naics_embedder.remote.worker import run_probe
        if operation not in OPERATIONS:
            raise ValueError(f'unknown remote operation: {operation}')
        if operation in {'bootstrap', 'training', 'gpu', 'clock', 'transport_prerequisites'}:
            return self.process.probe(operation, payload)
        if operation == 'identity':
            return {'repo': str(self.root), 'checkpoint_base': str(self.root / 'checkpoints')}
        return run_probe(operation, payload, self.root)

    def _transfer(
        self, source: Path, destination: Path, files: tuple[str, ...] | None, dry: bool = False
    ) -> subprocess.CompletedProcess:
        args = [self.rsync, '-rlpt', '--protect-args', '--checksum', '--partial-dir=.rsync-partial']
        data = None
        if files is not None:
            data = safe_files(files)
            for file in files:
                if not (source / file).resolve().is_relative_to(source.resolve()):
                    raise ValueError('source escapes transfer root')
                if not (destination / file).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('destination escapes transfer root')
            args += ['--from0', '--files-from=-']
        if dry:
            args += ['--dry-run', '--itemize-changes', '--out-format=%i|%n']
        else:
            destination.mkdir(parents=True, exist_ok=True)
        args += ['--', str(source) + '/', str(destination) + '/']
        return subprocess.run(
            args,
            input=data,
            capture_output=True,
            check=True,
            shell=False,
            timeout=TRANSFER_TIMEOUT
        )

    def push(self, source: Path, destination: str, files: tuple[str, ...]) -> None:
        self._transfer(source, self._path(destination), files)

    def pull(self, source: str, destination: Path, files: tuple[str, ...]) -> None:
        self._transfer(self._path(source), destination, files)

    def remove_code(self, paths: tuple[str, ...]) -> None:
        self.probe('edits', {'remove': list(paths)})

    def checksum(self, mapping: PullMapping) -> tuple[str, ...]:
        result = self._transfer(self._path(mapping.source), mapping.destination, None, dry=True)
        return tuple(
            line.split('|', 1)[1] for line in _text(result.stdout).splitlines()
            if re.match(r'^[<>ch.*][^|]{10}\|', line)
        )

    def launch(self, script: str, segment_id: str) -> None:
        self.process.launch(script, segment_id)

    def interrupt_training(self, segment_id: str) -> None:
        self.process.interrupt_training(segment_id)
