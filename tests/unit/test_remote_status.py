'''Status observes failures and pending state without updating records.'''
import pytest

def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}

def test_status_reports_readonly_process_gpu_sync_pending(remote_finish_fixture):
    env = remote_finish_fixture
    env.transport.differences = ('pending', )
    before = snapshot(env.root)
    result = env.workflow.status()
    assert result['host'] == 'fixture' and result['session_id'] == 'session'
    assert result['training']['exit_code'] == 0
    assert result['gpu']['utilization'] == 12
    assert result['pending'] == 4
    assert result['last_sync_utc']
    assert not result['loop']['alive']
    assert snapshot(env.root) == before

@pytest.mark.parametrize('problem', ['unreachable', 'config', 'pending'])
def test_status_errors_never_mutate(remote_finish_fixture, monkeypatch, problem):
    env = remote_finish_fixture
    if problem == 'config':
        (env.root / '.remote/session-config.json').write_text('{}')
    elif problem == 'pending':
        (env.root / '.remote/pulls/pending.json').write_text('{}')
    else:

        def offline(*args):
            raise OSError('unreachable')

        monkeypatch.setattr(env.transport, 'probe', offline)
    before = snapshot(env.root)
    result = env.workflow.status()
    assert result['errors']
    assert snapshot(env.root) == before

def test_status_missing_session_creates_nothing(tmp_path):
    from naics_embedder.remote.workflow import RemoteWorkflow
    from naics_embedder.utils.config import RemoteConfig
    workflow = RemoteWorkflow(tmp_path, RemoteConfig(), lambda *args: None, lambda: None)
    assert workflow.status()['status'] == 'no session'
    assert list(tmp_path.iterdir()) == []

def test_worker_status_reports_orphan_training_and_exit(tmp_path, monkeypatch):
    import subprocess

    from naics_embedder.remote.worker import run_probe
    segment = tmp_path / '.remote/segments/s'
    segment.mkdir(parents=True)
    (segment / 'exit_code').write_text('7\n')
    replies = iter(
        [
            subprocess.CompletedProcess([], 1, '', 'no server running'),
            subprocess.CompletedProcess(
                [], 0, '42 /uv run --locked naics-embedder train seed=1\n', ''
            ),
        ]
    )
    monkeypatch.setattr(
        'naics_embedder.remote.worker.subprocess.run', lambda *args, **kwargs: next(replies)
    )
    result = run_probe('training', {'action': 'status', 'segment_id': 's'}, tmp_path)
    assert result['process_running'] and result['running']
    assert result['exit_code'] == 7

def test_trusted_training_status_source_is_rendered_locally(recorded_transport_runner):
    from naics_embedder.remote.transport import SshTransport
    transport = SshTransport('fixture', '/repo', '/rsync', runner=recorded_transport_runner)
    transport.python = '/python'
    transport.probe('training', {'action': 'status'})
    assert '-c' in recorded_transport_runner.calls[-1].args[-1]

def test_worker_gpu_status_is_readonly(tmp_path, monkeypatch):
    import subprocess

    from naics_embedder.remote.worker import run_probe
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '0, 12, 34, 100\n', '')

    monkeypatch.setattr('naics_embedder.remote.worker.subprocess.run', run)
    result = run_probe('gpu', {'action': 'status'}, tmp_path)
    assert result['devices'][0]['utilization'] == 12
    assert calls[0][0] == 'nvidia-smi'

def test_worker_interrupt_rejects_foreign_segment_without_signal(tmp_path, monkeypatch):
    from naics_embedder.remote.worker import run_probe

    def forbidden(*args, **kwargs):
        raise AssertionError('unowned interruption must not run a command')

    monkeypatch.setattr('naics_embedder.remote.worker.subprocess.run', forbidden)
    with pytest.raises(ValueError, match='owned session'):
        run_probe('training', {'action': 'interrupt', 'segment_id': 'foreign'}, tmp_path)

def test_worker_interrupt_exact_owned_wrapper(tmp_path, monkeypatch):
    import json
    import shlex
    import subprocess

    from naics_embedder.remote.worker import run_probe
    segment = tmp_path / '.remote/segments/s'
    segment.mkdir(parents=True)
    marker = tmp_path / '.remote/pushes/session.json'
    marker.parent.mkdir()
    marker.write_text(json.dumps(dict(session_id='session', push_id='push')))
    (segment / 'segment.json').write_text(
        json.dumps(dict(session_id='session', push_id='push', segment_id='s'))
    )
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, 'bash ' + shlex.quote(str(segment / 'launch.sh')) + ' < /dev/null\n', ''
        )

    monkeypatch.setattr('naics_embedder.remote.worker.subprocess.run', run)
    assert run_probe('training', {
        'action': 'interrupt',
        'segment_id': 's'
    }, tmp_path)['interrupted']
    assert calls[-1] == ['tmux', 'send-keys', '-t', 'naics-train', 'C-c']

def test_status_reports_stale_loop_without_signal_or_record_repair(
    remote_finish_fixture, monkeypatch
):
    env = remote_finish_fixture
    from naics_embedder.remote import loop
    (env.root / '.remote/sync.pid').write_text('{"pid":123,"session_id":"session"}')
    monkeypatch.setattr(
        loop, 'loop_status', lambda *args: dict(alive=False, pid=123, session_id='session')
    )
    before = snapshot(env.root)
    result = env.workflow.status()
    assert result['loop'] == dict(alive=False, pid=123, session_id='session')
    assert snapshot(env.root) == before

def test_status_invalid_state_remains_readonly(remote_finish_fixture):
    env = remote_finish_fixture
    (env.root / '.remote/state.json').write_text('{invalid')
    before = snapshot(env.root)
    assert env.workflow.status()['status'] == 'invalid state'
    assert snapshot(env.root) == before

@pytest.mark.parametrize('rendered', [False, True])
@pytest.mark.parametrize('include_training', [False, True])
def test_raw_process_apostrophe_is_literal_and_training_stays_visible(
    tmp_path, monkeypatch, rendered, include_training
):
    import io
    import json
    import subprocess
    from contextlib import redirect_stdout

    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe

    output = "41 /usr/bin/python3 /tmp/worker.py --label O'Brien\n"
    if include_training:
        output += "42 /uv run --locked naics-embedder train --label O'Brien\n"
        output += "43 /venv/python /tmp/O'Brien/naics-embedder train seed=1\n"
    assert "O'Brien" in output and "O\\'Brien" not in output
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        if args[0] == 'tmux':
            return subprocess.CompletedProcess(args, 1, '', 'no server running')
        assert args == ['ps', '-ww', '-axo', 'pid=,command=']
        return subprocess.CompletedProcess(args, 0, output, '')

    monkeypatch.setattr('naics_embedder.remote.worker.subprocess.run', runner)
    before = snapshot(tmp_path)
    if rendered:
        stream = io.StringIO()
        payload = dict(repo=str(tmp_path), action='status')
        monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(payload)))
        with redirect_stdout(stream):
            exec(_system_probe_code('training'), {})
        result = json.loads(stream.getvalue())
    else:
        result = run_probe('training', {'action': 'status'}, tmp_path)
    assert result['running'] is include_training
    assert result['process_running'] is include_training
    assert result['process_pids'] == ([42, 43] if include_training else [])
    assert result['sessions'] == []
    assert snapshot(tmp_path) == before
    assert len(calls) == 2
