'''Process lifecycle ownership and exact tmux interrupt boundaries.'''
import io
import json
import shlex
import subprocess
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from naics_embedder.remote import loop, worker

@pytest.mark.parametrize(
    'change', [
        'exec', 'reused_pid', 'foreign_token', 'wrong_session', 'reverse_exec', 'malformed_identity'
    ]
)
def test_spawned_pid_transition_retains_identity_guards(tmp_path, monkeypatch, change):
    session = 'local-session'
    argv = [
        'caffeinate', '-i', '/python', '-m', 'naics_embedder.remote.worker', 'sync-loop', '--root',
        str(tmp_path), '--session-id', session, '--token', 'unique-token'
    ]
    previous = {'start': 'original-start', 'command': ' '.join(argv)}
    current = {'start': 'original-start', 'command': ' '.join(argv[2:])}
    record = dict(pid=123, session_id=session, token='unique-token', argv=argv, identity=previous)
    if change == 'reused_pid':
        current['start'] = 'different-start'
    elif change == 'foreign_token':
        current['command'] = current['command'].replace('unique-token', 'foreign-token')
    elif change == 'wrong_session':
        record['session_id'] = 'other-session'
    elif change == 'reverse_exec':
        previous['command'], current['command'] = current['command'], previous['command']
    elif change == 'malformed_identity':
        record['identity'] = None
    monkeypatch.setattr(loop, '_process_identity', lambda pid: current)
    assert loop._owned(tmp_path, session, record) is (change == 'exec')

@pytest.mark.parametrize(
    'response', [
        'unquoted', 'quoted', 'foreign_script', 'wrong_stdin', 'extra_command', 'second_pane',
        'malformed_quote', 'empty', 'empty_before', 'empty_after', 'whitespace_before',
        'whitespace_after'
    ]
)
@pytest.mark.parametrize('rendered', [False, True])
def test_tmux_quoting_keeps_exact_command_boundary(tmp_path, monkeypatch, response, rendered):
    root = tmp_path / "directory with spaces and ' quotes"
    segment = 'owned-segment'
    directory = root / '.remote/segments' / segment
    directory.mkdir(parents=True)
    (directory / 'segment.json').write_text(
        json.dumps(dict(segment_id=segment, session_id='session', push_id='push'))
    )
    marker = root / '.remote/pushes/session.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(dict(session_id='session', push_id='push')))
    expected = 'bash ' + shlex.quote(str(directory / 'launch.sh')) + ' < /dev/null'
    actual = {
        'unquoted': expected,
        'quoted': json.dumps(expected),
        'foreign_script': json.dumps(expected.replace('/owned-segment/', '/foreign-segment/')),
        'wrong_stdin': json.dumps(expected.replace('/dev/null', '/tmp/other-input')),
        'extra_command': json.dumps(expected + '; another-command'),
        'second_pane': json.dumps(expected) + '\n' + json.dumps(expected),
        'malformed_quote': '"unterminated',
        'empty': '',
        'empty_before': '\n' + json.dumps(expected),
        'empty_after': json.dumps(expected) + '\n',
        'whitespace_before': ' \n' + json.dumps(expected),
        'whitespace_after': json.dumps(expected) + '\n ',
    }[response]
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        output = actual + '\n' if 'list-panes' in argv else ''
        return subprocess.CompletedProcess(argv, 0, output, '')

    monkeypatch.setattr(worker.subprocess, 'run', run)

    def interrupt():
        if not rendered:
            return worker._interrupt_training(root, {'segment_id': segment})
        from naics_embedder.remote.transport import _system_probe_code
        payload = dict(repo=str(root), action='interrupt', segment_id=segment)
        output = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(payload))):
            with redirect_stdout(output):
                exec(_system_probe_code('training'), {})
        return json.loads(output.getvalue())

    if response in ('unquoted', 'quoted'):
        assert interrupt() == {'interrupted': True}
        assert calls[-1] == ['tmux', 'send-keys', '-t', 'naics-train', 'C-c']
    else:
        with pytest.raises(ValueError, match='recorded owned wrapper'):
            interrupt()
        assert len(calls) == 1 and 'list-panes' in calls[0]

def test_spawned_pid_execs_between_capture_and_launch_check(remote_sync_fixture, monkeypatch):
    from tests.unit.test_remote_loop import install_processes
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    original_lookup = loop._process_identity
    reads = 0

    def identity(pid):
        nonlocal reads
        if pid == 123:
            reads += 1
            if reads == 2:
                identities[pid] = dict(start='today', command=' '.join(calls[0][0][2:]))
        return original_lookup(pid)

    monkeypatch.setattr(loop, '_process_identity', identity)
    loop.ensure_loop(env.root, env.state)
    assert loop.loop_status(env.root, env.state.session_id)['alive'] is True
    assert len(calls) == 1
    loop.stop_loop(env.root, env.state.session_id)
    assert calls[-1][0] == 123
    assert not (env.root / '.remote/sync.pid').exists()

def test_spawned_pid_execs_at_shutdown_signal_boundary(remote_sync_fixture, monkeypatch):
    from tests.unit.test_remote_loop import install_processes, install_shutdown_clock
    env = remote_sync_fixture
    calls, identities = install_processes(monkeypatch)
    install_shutdown_clock(monkeypatch)
    loop.ensure_loop(env.root, env.state)
    record_path = env.root / '.remote/sync.pid'
    record = json.loads(record_path.read_text())
    utility = dict(start='today', command=' '.join(record['argv'][2:]))
    reads = 0

    def identity(pid):
        nonlocal reads
        if pid == 123:
            reads += 1
            if reads == 3:
                identities[pid] = utility
        return identities.get(pid)

    signaled = []

    def terminate(pid, sig):
        assert identities[pid] == utility
        assert json.loads(record_path.read_text())['observed_workers'][str(pid)] == utility
        signaled.append(pid)
        identities.pop(pid)

    monkeypatch.setattr(loop, '_process_identity', identity)
    monkeypatch.setattr(loop, '_signal', terminate)
    loop.stop_loop(env.root, env.state.session_id)
    assert signaled == [123]
    assert not record_path.exists()
