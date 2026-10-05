from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from naics_embedder.remote import session
from naics_embedder.remote.session import (
    RemoteInfo,
    RemoteState,
    RunRecord,
    StateBusyError,
    new_id,
    pull_mappings,
    read_state,
    state_lock,
    write_state,
)

INFO = RemoteInfo(
    '/home/ubuntu/naics-embedder', '/home/ubuntu/naics-embedder/checkpoints',
    '/home/ubuntu/.local/bin/uv', '/usr/bin/python3', True, 'cuda', 'fixture'
)
NOW = datetime(2026, 10, 5, 21, tzinfo=timezone.utc)

def test_pull_logs_cannot_replace_the_mac_selection_log(tmp_path):
    mappings = pull_mappings(tmp_path, '20261005T210000Z', INFO)
    assert [
        (m.source.rsplit('/', 1)[-1], m.destination.relative_to(tmp_path).as_posix())
        for m in mappings
    ] == [
        ('checkpoints', 'checkpoints'),
        ('outputs', 'outputs/remote/20261005T210000Z'),
        ('logs', 'logs/remote/20261005T210000Z'),
        ('segments', 'outputs/remote/20261005T210000Z/segments'),
    ]

def test_id_timestamp_collision_and_push_suffix():
    assert new_id('session', NOW, set()) == '20261005T210000Z'
    assert new_id('push', NOW, set(), 'abcdef0123456789') == '20261005T210000Z-abcdef0'
    with pytest.raises(ValueError, match='collision'):
        new_id('segment', NOW, {'20261005T210000Z'})
    with pytest.raises(ValueError):
        new_id('session', NOW.replace(tzinfo=None), set())

def make_state():
    return RemoteState(
        host='ubuntu@fixture', session_id='20261005T210000Z', started_utc=NOW, remote_info=INFO
    )

def test_atomic_state_survives_failed_replace(tmp_path, monkeypatch):
    state = make_state()
    write_state(tmp_path, state)
    original = (tmp_path / '.remote/state.json').read_bytes()

    def fail_replace(*args):
        raise OSError('injected failure')

    monkeypatch.setattr(session.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='injected failure'):
        write_state(tmp_path, state.model_copy(update={'status': 'ready'}))
    assert (tmp_path / '.remote/state.json').read_bytes() == original
    assert list((tmp_path / '.remote').glob('*.tmp')) == []

def test_lock_is_busy_then_reusable(tmp_path):
    with state_lock(tmp_path):
        with pytest.raises(StateBusyError, match='busy'):
            with state_lock(tmp_path):
                pass
    with state_lock(tmp_path):
        assert (tmp_path / '.remote/state.lock').is_file()

def test_reopening_preserves_mapping_and_separate_runs(tmp_path):
    assert read_state(tmp_path) is None
    state = make_state()
    runs = tmp_path / '.remote/runs'
    runs.mkdir(parents=True)
    run = RunRecord(
        experiment='reference',
        remote_directory=INFO.checkpoint_base + '/reference',
        bundle_id='fixture',
        codebook_fingerprint='a',
        description_fingerprint='b',
        seed=1,
        settings={'max_epochs': 2},
        constructor_controls={'lora_rank': 4},
        session_id=state.session_id,
        segment_id=None
    )
    path = runs / 'reference.json'
    path.write_text(run.model_dump_json())
    write_state(tmp_path, state)
    reopened = read_state(tmp_path)
    assert reopened == state
    assert pull_mappings(tmp_path, reopened.session_id,
                         reopened.remote_info) == pull_mappings(tmp_path, state.session_id, INFO)
    write_state(tmp_path, state.model_copy(update={'status': 'finished'}))
    assert RunRecord.model_validate_json(path.read_text()) == run

def test_state_rejects_unknown_keys_and_naive_time():
    with pytest.raises(ValidationError):
        RemoteState(**make_state().model_dump(), unknown=True)
    with pytest.raises(ValidationError):
        RemoteState(host='x', session_id='x', started_utc=NOW.replace(tzinfo=None))
