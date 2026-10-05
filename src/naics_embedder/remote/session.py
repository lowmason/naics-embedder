'''Typed durable session state, stable identifiers and protected result destinations.'''

import fcntl
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, field_validator

# -------------------------------------------------------------------------------------------------
# Shared records
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    size: int
    mode: int
    kind: str
    target: str | None

@dataclass(frozen=True)
class GpuEvidence:
    logical_index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    native_bf16: bool
    cuda_visible_devices: str | None

@dataclass(frozen=True)
class RemoteInfo:
    repo: str
    checkpoint_base: str
    uv: str
    python: str
    ntp: bool
    accelerator: str
    gpu: str
    gpu_evidence: GpuEvidence | None = None

@dataclass(frozen=True)
class PullMapping:
    source: str
    destination: Path

class RemoteState(BaseModel):
    '''One instance session; run identities persist in separate RunRecord files.'''

    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1] = 1
    status: Literal['preparing', 'ready', 'finished', 'abandoned'] = 'preparing'
    host: str
    session_id: str
    started_utc: datetime
    remote_info: RemoteInfo | None = None
    push_id: str | None = None
    pending_push_id: str | None = None
    active_segment_id: str | None = None
    last_sync_utc: datetime | None = None
    unreachable_since: datetime | None = None
    last_sync_manifest: str | None = None

    @field_validator('started_utc', 'last_sync_utc', 'unreachable_since')
    @classmethod
    def utc_timestamp(cls, value: datetime | None) -> datetime | None:
        '''Require timezone-aware timestamps and normalize them to UTC.'''
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('timestamp must be timezone-aware')
        return value.astimezone(timezone.utc)

class RunRecord(BaseModel):
    '''Persistent identity of one experiment across replacement instance sessions.'''

    model_config = ConfigDict(extra='forbid')

    experiment: str
    remote_directory: str
    bundle_id: str
    codebook_fingerprint: str
    description_fingerprint: str
    seed: int
    settings: dict[str, object]
    constructor_controls: dict[str, object]
    training_run: str | None = None
    session_id: str
    segment_id: str | None = None

class StateBusyError(RuntimeError):
    '''Another transaction holds the checkout's remote state lock.'''

# -------------------------------------------------------------------------------------------------
# Identifiers and persistence
# -------------------------------------------------------------------------------------------------

def new_id(kind: str, now: datetime, existing: set[str], head: str = '') -> str:
    '''Allocate a UTC timestamp ID, refusing collisions rather than changing its meaning.'''
    if kind not in ('session', 'segment', 'push'):
        raise ValueError(f'unknown ID kind: {kind}')
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('ID timestamp must be timezone-aware')
    identifier = now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if kind == 'push':
        if len(head) < 7 or any(char not in '0123456789abcdefABCDEF' for char in head):
            raise ValueError('push ID requires a commit SHA')
        identifier += '-' + head[:7]
    if identifier in existing:
        raise ValueError(f'{kind} ID collision: {identifier}')
    return identifier

def read_state(root: Path) -> RemoteState | None:
    '''Read the session record, returning None only when it is absent.'''
    path = root / '.remote/state.json'
    try:
        content = path.read_text()
    except FileNotFoundError:
        return None
    return RemoteState.model_validate_json(content)

def write_state(root: Path, state: RemoteState) -> None:
    '''Atomically save the state using a flushed sibling temporary file.'''
    directory = root / '.remote'
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='state-', suffix='.tmp', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(state.model_dump_json(indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / 'state.json')
    finally:
        temporary.unlink(missing_ok=True)

@contextmanager
def state_lock(root: Path) -> Iterator[None]:
    '''Hold a nonblocking advisory lock for a complete state transaction.'''
    directory = root / '.remote'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'state.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StateBusyError(
                'remote state is busy; another transaction holds the lock'
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def pull_mappings(root: Path, session_id: str, info: RemoteInfo) -> tuple[PullMapping, ...]:
    '''Map the four result roots without sharing the Mac decision-log destination.'''
    if not session_id or session_id in ('.', '..') or any(
        char in '/\\' or ord(char) < 32 or 127 <= ord(char) <= 159 for char in session_id
    ):
        raise ValueError('session_id must be a single nonempty component')
    output = root / 'outputs/remote' / session_id
    return (
        PullMapping(info.checkpoint_base, root / 'checkpoints'),
        PullMapping(info.repo + '/outputs', output),
        PullMapping(info.repo + '/logs', root / 'logs/remote' / session_id),
        PullMapping(info.repo + '/.remote/segments', output / 'segments'),
    )
