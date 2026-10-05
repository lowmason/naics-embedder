'''Git-defined code bytes and conservative instance-edit classification.'''

import fnmatch
import hashlib
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath

from naics_embedder.remote.session import FileEntry

HASH_BLOCK_BYTES = 1024 * 1024
RUNTIME_ROOTS = frozenset({'.git', '.venv', 'data', 'checkpoints', 'logs', 'outputs', '.remote'})
CREDENTIAL_ROOTS = frozenset({'.git', '.ssh', '.aws'})
PRIVATE_KEY_NAMES = frozenset({'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519', 'identity'})
PRIVATE_KEY_SUFFIXES = ('.pem', '.key', '.p12', '.pfx', '.ppk')

# -------------------------------------------------------------------------------------------------
# Git enumeration and safe file entries
# -------------------------------------------------------------------------------------------------

def git_bytes(root: Path, *args: str) -> bytes:
    '''Run a fixed Git operation without shell parsing, preserving NUL-separated names.'''
    return subprocess.check_output(
        ['git', '-c', 'core.fsmonitor=false', '-C',
         str(root), *args], stderr=subprocess.PIPE
    )

def git_paths(root: Path, *args: str) -> tuple[str, ...]:
    return tuple(
        sorted({os.fsdecode(name)
                for name in git_bytes(root, *args).split(b'\0') if name})
    )

def validate_code_path(name: str) -> None:
    '''Require a normalized repo-relative path before filesystem access or deletion.'''
    path = PurePosixPath(name)
    if not name or '\0' in name or path.is_absolute() or any(
        part in ('', '.', '..') for part in name.split('/')
    ):
        raise ValueError(f'unsafe code path: {name!r}')

def is_credential_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(
        part in CREDENTIAL_ROOTS or part == '.env' or part.startswith('.env.')
        or part.lower() in PRIVATE_KEY_NAMES or part.lower().endswith(PRIVATE_KEY_SUFFIXES)
        for part in parts
    )

def _safe_link(root: Path, name: str) -> str:
    path = root / name
    target = os.readlink(path)
    try:
        resolved = path.resolve()
    except (RuntimeError, OSError) as error:
        raise ValueError(f'unsafe symlink: {name!r}') from error
    if Path(target).is_absolute() or not resolved.is_relative_to(root) or not resolved.exists():
        raise ValueError(f'unsafe or broken symlink: {name!r}')
    if is_credential_path(resolved.relative_to(root).as_posix()):
        raise ValueError(f'credential symlink target: {name!r}')
    return target

def file_entry(root: Path, name: str) -> FileEntry:
    '''Hash a regular file or a safe existing relative symlink using lstat semantics.'''
    validate_code_path(name)
    path = root / name
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError(f'code path escapes root: {name!r}')
    metadata = path.lstat()
    target = None
    if stat.S_ISLNK(metadata.st_mode):
        target = _safe_link(root, name)
        content = os.fsencode(target)
        digest = hashlib.sha256(content).hexdigest()
        size = len(content)
        kind = 'symlink'
    elif stat.S_ISREG(metadata.st_mode):
        digest_state = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b''):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        size = metadata.st_size
        kind = 'file'
    else:
        raise ValueError(f'unsupported code file type: {name!r}')
    return FileEntry(name, digest, size, stat.S_IMODE(metadata.st_mode), kind, target)

def _refuse_special_files(root: Path) -> None:
    for directory, children, files in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        directory_names = [(relative / name).as_posix() for name in children if name != '.git']
        ignored = set()
        if directory_names:
            result = subprocess.run(
                [
                    'git', '-c', 'core.fsmonitor=false', '-C',
                    str(root), 'check-ignore', '-z', '--stdin'
                ],
                input=b'\0'.join(os.fsencode(name) for name in directory_names) + b'\0',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode not in (0, 1):
                raise ValueError('cannot inspect Git ignore rules')
            ignored = {os.fsdecode(name) for name in result.stdout.split(b'\0') if name}
        children[:] = [
            name for name in children
            if (relative / name).as_posix() not in ignored and name != '.git'
        ]
        for name in files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) and not stat.S_ISLNK(mode):
                relative_name = path.relative_to(root).as_posix()
                result = subprocess.run(
                    [
                        'git', '-c', 'core.fsmonitor=false', '-C',
                        str(root), 'check-ignore', '--', relative_name
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                if result.returncode == 1:
                    raise ValueError(f'unsupported code file type: {relative_name!r}')
                if result.returncode != 0:
                    raise ValueError('cannot inspect Git ignore rules')

def code_entries(root: Path) -> tuple[FileEntry, ...]:
    '''Describe all cached and nonignored untracked files, refusing credential names first.'''
    root = root.resolve()
    names = git_paths(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    head_names = git_paths(root, 'ls-tree', '-r', '--name-only', '-z', 'HEAD')
    for name in set(names) | set(head_names):
        validate_code_path(name)
        if is_credential_path(name):
            raise ValueError(f'credential path must be committed safely or ignored: {name!r}')
    present_names = {name for name in names if os.path.lexists(root / name)}
    for name in names:
        if (root / name).is_symlink():
            _safe_link(root, name)
            target = (root / name).resolve().relative_to(root).as_posix()
            if target != '.' and target not in present_names and not any(
                candidate.startswith(target + '/') for candidate in present_names
            ):
                raise ValueError(f'symlink target is outside pushed file set: {name!r}')
    _refuse_special_files(root)
    for line in git_bytes(root, 'ls-files', '-z', '--stage').split(b'\0'):
        if line:
            metadata, name = line.split(b'\t', 1)
            mode, _, stage = metadata.split()
            if mode == b'160000' or stage != b'0':
                raise ValueError(
                    f'unsupported submodule or unresolved index: {os.fsdecode(name)!r}'
                )
    return tuple(file_entry(root, name) for name in names if os.path.lexists(root / name))

# -------------------------------------------------------------------------------------------------
# Changes and instance code boundary
# -------------------------------------------------------------------------------------------------

def deletion_set(old: tuple[FileEntry, ...], new: tuple[FileEntry, ...]) -> tuple[str, ...]:
    '''Return only previously pushed paths absent from the new manifest.'''
    return tuple(sorted({item.path for item in old} - {item.path for item in new}))

def is_runtime_path(name: str, ignore: tuple[str, ...]) -> bool:
    '''New generated and credential files never become force-discard candidates.'''
    parts = PurePosixPath(name).parts
    if parts[0] in RUNTIME_ROOTS or is_credential_path(name):
        return True
    for pattern in ignore:
        if pattern.endswith('/'):
            directory = pattern.rstrip('/')
            if directory in parts or fnmatch.fnmatchcase(name + '/', pattern + '*'):
                return True
        elif fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(parts[-1], pattern):
            return True
    return False

def instance_edits(
    expected: tuple[FileEntry, ...], actual: tuple[FileEntry, ...], ignore: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    '''Inspect every pushed path, ignoring runtime patterns only for new instance paths.'''
    for item in (*expected, *actual):
        validate_code_path(item.path)
    previous = {item.path: item for item in expected}
    current = {item.path: item for item in actual}
    return {
        'modified': tuple(
            sorted(
                name for name in previous.keys() & current.keys() if previous[name] != current[name]
            )
        ),
        'deleted': tuple(sorted(previous.keys() - current.keys())),
        'new': tuple(
            sorted(
                name for name in current.keys() - previous.keys()
                if not is_runtime_path(name, ignore)
            )
        ),
    }
