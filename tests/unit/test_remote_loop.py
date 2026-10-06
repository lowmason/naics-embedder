import json
from types import SimpleNamespace

import pytest

import naics_embedder.remote.loop as loop
from naics_embedder.remote.session import write_state

def install_processes(monkeypatch):
    calls = []
    identities = {}

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        identities[123] = {'start': 'today', 'command': ' '.join(argv)}
        import os
        identities[os.getpid()] = {'start': 'today', 'command': ' '.join(argv[2:])}
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(loop, '_spawn', spawn)
    monkeypatch.setattr(loop, '_process_identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(loop, '_signal', lambda pid, sig: calls.append((pid, sig)))
    return calls, identities

def test_loop_owned_lifecycle_and_launch_argv(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    argv, kwargs = calls[0]
    assert argv[:2] == ['caffeinate', '-i']
    assert argv[2].startswith('/')
    assert kwargs['start_new_session'] is True
    assert '--token' in argv and 'sync-loop' in argv
    assert loop.loop_status(env.root, env.state.session_id)['alive']
    loop.ensure_loop(env.root, env.state)
    assert len(calls) == 1
    loop.stop_loop(env.root, env.state.session_id)
    assert calls[-1][0] == 123

@pytest.mark.parametrize('identity', [None, {'start': 'reused', 'command': 'unowned'}])
def test_stale_or_reused_pid_never_killed_and_restarts(remote_sync_fixture, monkeypatch, identity):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    identities[123] = identity
    assert not loop.loop_status(env.root, env.state.session_id)['alive']
    loop.stop_loop(env.root, env.state.session_id)
    assert len(calls) == 1
    loop.ensure_loop(env.root, env.state)
    assert len(calls) == 2

@pytest.mark.parametrize('mutation', ['missing', 'wrong_session', 'corrupt'])
def test_loop_requires_persisted_effective_configuration(
    remote_sync_fixture, monkeypatch, mutation
):
    env = remote_sync_fixture
    calls, _ = install_processes(monkeypatch)
    path = env.root / '.remote/session-config.json'
    if mutation == 'missing':
        path.unlink()
    else:
        path.write_text(
            '{}' if mutation == 'corrupt' else json.dumps(
                dict(session_id='other', remote_config=env.cfg.model_dump())
            )
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        loop.ensure_loop(env.root, env.state)
    assert not calls

@pytest.mark.parametrize('status', ['finished', 'abandoned', 'different'])
def test_worker_exits_on_closed_or_changed_session(remote_sync_fixture, monkeypatch, status):
    env = remote_sync_fixture
    env.state.status = status if status != 'different' else 'ready'
    if status == 'different':
        env.state.session_id = 'different'
    write_state(env.root, env.state)
    calls = []
    loop.run_loop(
        env.root,
        'session',
        'token',
        transport_factory=lambda *args: calls.append(args),
        wait=lambda seconds: calls.append(seconds)
    )
    assert not calls

def test_worker_retries_failure_and_uses_custom_config(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    env.transport.fail_next_pull = True
    seen = []

    def factory(host, cfg):
        seen.append(cfg)
        return env.transport

    waits = []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            env.state.status = 'finished'
            write_state(env.root, env.state)

    loop.run_loop(env.root, 'session', 'token', transport_factory=factory, wait=wait)
    assert waits == [3, 3]
    assert all(cfg.rsync_path == '/custom/rsync' for cfg in seen)
    assert (env.root / 'checkpoints/run/last.ckpt').exists()

def test_cooperative_stop_prevents_wrapper_child_from_another_pass(
    remote_sync_fixture, monkeypatch
):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record = json.loads((env.root / '.remote/sync.pid').read_text())
    passes = []

    def factory(host, cfg):
        passes.append('pass')
        return env.transport

    def wait(seconds):
        if len(passes) > 1:
            raise AssertionError('worker ran after cooperative stop')
        loop.stop_loop(env.root, env.state.session_id)

    loop.run_loop(
        env.root, env.state.session_id, record['token'], transport_factory=factory, wait=wait
    )
    assert passes == ['pass']
    assert (env.root / '.remote/sync-stop.json').exists()

def test_loop_supports_literal_spaces_in_checkout_path(remote_sync_fixture, monkeypatch):
    import shutil
    env = remote_sync_fixture
    moved = env.root.with_name('mac with spaces')
    shutil.move(env.root, moved)
    calls, _ = install_processes(monkeypatch)
    loop.ensure_loop(moved, env.state)
    assert loop.loop_status(moved, env.state.session_id)['alive']
    loop.stop_loop(moved, env.state.session_id)
    assert calls[-1][0] == 123

def test_loop_rejects_incomplete_config_snapshot(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    install_processes(monkeypatch)
    (env.root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=env.state.session_id, remote_config={}))
    )
    with pytest.raises(ValueError):
        loop.ensure_loop(env.root, env.state)

def test_reused_python_worker_pid_never_signaled(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    path = env.root / '.remote/sync.pid'
    record = json.loads(path.read_text())
    record.update(worker_pid=456, worker_identity={'start': 'old', 'command': 'owned'})
    path.write_text(json.dumps(record))
    identities[456] = {'start': 'new', 'command': 'unowned'}
    loop.stop_loop(env.root, env.state.session_id)
    assert all(call[0] != 456 for call in calls)
