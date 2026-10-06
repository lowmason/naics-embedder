import json
from types import SimpleNamespace

import pytest

import naics_embedder.remote.loop as loop
from naics_embedder.remote.session import write_state

REAL_WORKER_DISCOVERY = loop._worker_processes

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

    def signal(pid, sig):
        calls.append((pid, sig))
        identities.pop(pid, None)

    monkeypatch.setattr(loop, '_signal', signal)

    def workers(record):
        import os
        commands = {' '.join(record['argv']), ' '.join(record['argv'][2:])}
        return {
            pid: identity
            for pid, identity in identities.items()
            if pid not in (record['pid'],
                           os.getpid()) and identity and identity['command'] in commands
        }

    monkeypatch.setattr(loop, '_worker_processes', workers)
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

def install_shutdown_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(loop.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(
        loop.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    return clock

@pytest.mark.parametrize('registered', [False, True])
def test_stop_verifies_delayed_worker_exit_and_retains_evidence(
    remote_sync_fixture, monkeypatch, registered
):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    clock = install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    path = env.root / '.remote/sync.pid'
    record = json.loads(path.read_text())
    child = {'start': 'child start', 'command': ' '.join(record['argv'][2:])}
    identities[456] = child
    if registered:
        record.update(worker_pid=456, worker_identity=child)
        path.write_text(json.dumps(record))
    monkeypatch.setattr(
        loop,
        '_worker_processes',
        lambda record: {456: child} if identities.get(456) else {},
        raising=False
    )

    def signal(pid, sig):
        assert path.exists()
        calls.append((pid, sig))
        if pid == 123:
            identities.pop(123, None)

    monkeypatch.setattr(loop, '_signal', signal)

    def sleep(seconds):
        assert path.exists()
        clock[0] += seconds
        if clock[0] >= 0.3:
            identities.pop(456, None)

    monkeypatch.setattr(loop.time, 'sleep', sleep)
    loop.stop_loop(env.root, env.state.session_id)
    assert clock[0] >= 0.3
    assert not path.exists() and not identities.get(456)
    assert any(call[0] == 456 for call in calls)

@pytest.mark.parametrize('registered', [False, True])
def test_stop_refuses_never_exiting_child_and_keeps_identity(
    remote_sync_fixture, monkeypatch, registered
):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    path = env.root / '.remote/sync.pid'
    record = json.loads(path.read_text())
    child = {'start': 'child start', 'command': ' '.join(record['argv'][2:])}
    identities[456] = child
    if registered:
        record.update(worker_pid=456, worker_identity=child)
        path.write_text(json.dumps(record))
    monkeypatch.setattr(loop, '_worker_processes', lambda record: {456: child}, raising=False)

    def signal(pid, sig):
        calls.append((pid, sig))
        if pid == 123:
            identities.pop(123, None)

    monkeypatch.setattr(loop, '_signal', signal)
    with pytest.raises(RuntimeError, match='exit|terminate|stop'):
        loop.stop_loop(env.root, env.state.session_id)
    assert path.exists()
    assert json.loads(path.read_text())['token'] == record['token']
    assert (env.root / '.remote/sync-stop.json').exists()

def test_stop_detects_child_that_execs_after_wrapper_exit(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    clock = install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record = json.loads((env.root / '.remote/sync.pid').read_text())
    child = {'start': 'forked child', 'command': ' '.join(record['argv'])}
    identities[456] = child
    monkeypatch.setattr(
        loop,
        '_worker_processes',
        lambda record: {456: identities[456]} if identities.get(456) else {},
        raising=False
    )

    def signal(pid, sig):
        calls.append((pid, sig))
        if pid == 123:
            identities.pop(123, None)

    monkeypatch.setattr(loop, '_signal', signal)

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 0.2:
            identities.pop(456, None)

    monkeypatch.setattr(loop.time, 'sleep', sleep)
    loop.stop_loop(env.root, env.state.session_id)
    assert clock[0] >= 0.2
    assert any(call[0] == 456 for call in calls)

def test_worker_discovery_matches_only_owned_commands_and_start_identity(
    remote_sync_fixture, monkeypatch
):
    import subprocess
    env = remote_sync_fixture
    install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record = json.loads((env.root / '.remote/sync.pid').read_text())
    worker_command = ' '.join(record['argv'][2:])
    wrapper_command = ' '.join(record['argv'])
    response = (
        '123 Mon Oct 5 12:00:00 2026 ' + wrapper_command + '\n'
        '456 Mon Oct 5 12:00:01 2026 ' + worker_command + '\n'
        '789 Mon Oct 5 12:00:02 2026 ' + worker_command.replace(record['token'], 'other') + '\n'
        '654 Mon Oct 5 12:00:03 2026 ' + wrapper_command + '\n'
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, response, '')

    monkeypatch.setattr(loop.subprocess, 'run', runner)
    # Exercise the real parser after injecting the fixture helper's discovery seam.
    # The function definition is retained separately before install_processes replaces it.
    found = REAL_WORKER_DISCOVERY(record)
    assert set(found) == {456, 654}
    assert found[456] == {'start': 'Mon Oct 5 12:00:01 2026', 'command': worker_command}
    assert calls[0][0] == ['ps', '-ww', '-axo', 'pid=,lstart=,command=']
    assert calls[0][1]['timeout'] > 0

def test_process_inspection_failure_retains_stop_evidence(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    install_processes(monkeypatch)
    loop.ensure_loop(env.root, env.state)

    def failure(record):
        raise RuntimeError('process inspection unavailable')

    monkeypatch.setattr(loop, '_worker_processes', failure)
    with pytest.raises(RuntimeError, match='inspection'):
        loop.stop_loop(env.root, env.state.session_id)
    assert (env.root / '.remote/sync.pid').exists()
    assert (env.root / '.remote/sync-stop.json').exists()

def test_discovered_pid_reuse_at_signal_boundary_never_signaled(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record = json.loads((env.root / '.remote/sync.pid').read_text())
    before = {'start': 'old', 'command': ' '.join(record['argv'][2:])}
    identities[456] = {'start': 'new', 'command': 'unowned'}
    monkeypatch.setattr(loop, '_worker_processes', lambda record: {456: before})
    with pytest.raises(RuntimeError, match='timeout'):
        loop.stop_loop(env.root, env.state.session_id)
    assert all(call[0] != 456 for call in calls)
    assert (env.root / '.remote/sync.pid').exists()

def test_registered_pid_reused_with_same_command_is_not_owned(remote_sync_fixture, monkeypatch):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    path = env.root / '.remote/sync.pid'
    record = json.loads(path.read_text())
    command = ' '.join(record['argv'][2:])
    record.update(worker_pid=456, worker_identity={'start': 'old', 'command': command})
    path.write_text(json.dumps(record))
    identities[456] = {'start': 'reused', 'command': command}
    monkeypatch.setattr(
        loop, '_worker_processes', lambda record: {456: identities[456]}
        if identities.get(456) else {}
    )
    loop.stop_loop(env.root, env.state.session_id)
    assert all(call[0] != 456 for call in calls)

def test_unregistered_pid_reuse_between_polls_is_not_signaled_again(
    remote_sync_fixture, monkeypatch
):
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    clock = install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record = json.loads((env.root / '.remote/sync.pid').read_text())
    command = ' '.join(record['argv'][2:])
    identities[456] = {'start': 'old', 'command': command}
    monkeypatch.setattr(loop, '_worker_processes', lambda record: {456: identities[456]})
    signals = []

    def signal(pid, sig):
        signals.append((pid, identities[pid]['start']))
        if pid == 123:
            identities.pop(123)

    monkeypatch.setattr(loop, '_signal', signal)

    def sleep(seconds):
        clock[0] += seconds
        identities[456] = {'start': 'reused', 'command': command}
        if clock[0] >= 0.2:
            raise AssertionError('stop did not recognize original worker exit')

    monkeypatch.setattr(loop.time, 'sleep', sleep)
    loop.stop_loop(env.root, env.state.session_id)
    assert signals.count((456, 'old')) == 1
    assert (456, 'reused') not in signals
