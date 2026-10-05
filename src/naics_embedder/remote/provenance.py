'''Reconstructible immutable code records published only after a stable snapshot.'''

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from naics_embedder.remote.code_manifest import code_entries, git_bytes, git_paths
from naics_embedder.remote.session import FileEntry

# -------------------------------------------------------------------------------------------------
# Push record and archive construction
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PushRecord:
    push_id: str
    directory: Path
    entries: tuple[FileEntry, ...]
    head_sha: str
    dirty: bool

def _write_json(path: Path, value: object) -> None:
    with path.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def _write_archive(root: Path, path: Path, entries: tuple[FileEntry, ...]) -> None:
    '''Archive validated untracked bytes without traversing links or adding directories.'''
    with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            member = tarfile.TarInfo(entry.path)
            member.mode = entry.mode
            if entry.kind == 'symlink':
                member.type = tarfile.SYMTYPE
                member.linkname = entry.target
                archive.addfile(member)
            else:
                content = (root / entry.path).read_bytes()
                if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
                    raise ValueError(f'code changed while recording: {entry.path!r}')
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

def write_push_record(root: Path, host: str, push_id: str, cap_bytes: int) -> PushRecord:
    '''Stage a complete record, revalidate its source, and publish once without overwriting.'''
    if not push_id or push_id in ('.', '..') or any(
        char in '/\\' or ord(char) < 32 or ord(char) == 127 for char in push_id
    ):
        raise ValueError('push_id must be a single nonempty component')
    if cap_bytes < 0:
        raise ValueError('untracked cap must be nonnegative')
    root = root.resolve()
    parent = root / '.remote/pushes'
    if not parent.resolve().is_relative_to(root):
        raise ValueError('push record directory escapes checkout')
    directory = parent / push_id
    if os.path.lexists(directory):
        raise ValueError(f'push record already exists: {push_id}')
    entries = code_entries(root)
    head = git_bytes(root, 'rev-parse', 'HEAD').decode().strip()
    branch = git_bytes(root, 'rev-parse', '--abbrev-ref', 'HEAD').decode().strip()
    patch = git_bytes(root, 'diff', '--binary', 'HEAD', '--')
    status = git_bytes(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
    untracked_names = git_paths(root, 'ls-files', '-z', '--others', '--exclude-standard')
    untracked = tuple(entry for entry in entries if entry.path in set(untracked_names))
    if sum(entry.size for entry in untracked) > cap_bytes:
        names = ', '.join(repr(entry.path) for entry in untracked)
        raise ValueError(f'untracked files exceed {cap_bytes} bytes: {names}')
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.pending-', dir=parent))
    try:
        (staging / 'uncommitted.patch').write_bytes(patch)
        _write_archive(root, staging / 'untracked.tar', untracked)
        _write_json(staging / 'hashes.json', {entry.path: entry.sha256 for entry in entries})
        _write_json(staging / 'files.json', [asdict(entry) for entry in entries])
        _write_json(
            staging / 'provenance.json', {
                'push_id': push_id,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'host': host,
                'head_sha': head,
                'branch': branch,
                'dirty': bool(status),
                'untracked': [
                    {
                        'path': entry.path,
                        'sha256': entry.sha256,
                        'size': entry.size
                    } for entry in untracked
                ],
                'file_count': len(entries),
            }
        )
        if (
            code_entries(root) != entries or git_bytes(root, 'rev-parse', 'HEAD').decode().strip()
            != head or git_bytes(root, 'rev-parse', '--abbrev-ref',
                                 'HEAD').decode().strip() != branch or git_bytes(
                                     root, 'diff', '--binary', 'HEAD', '--'
                                 ) != patch or git_bytes(
                                     root, 'status', '--porcelain=v1', '-z', '--untracked-files=all'
                                 ) != status or git_paths(
                                     root, 'ls-files', '-z', '--others', '--exclude-standard'
                                 ) != untracked_names
        ):
            raise ValueError('code changed while recording push')
        for path in staging.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        if os.path.lexists(directory):
            raise ValueError(f'push record already exists: {push_id}')
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return PushRecord(push_id, directory, entries, head, bool(status))
