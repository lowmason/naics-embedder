'''Transport argv and fixed operation boundary tests; no network.'''

import json
import shlex
import subprocess

import pytest

from naics_embedder.remote.session import PullMapping
from naics_embedder.remote.transport import SshTransport, gnu_rsync_version
from naics_embedder.remote.worker import run_probe

@pytest.mark.parametrize(
    'banner', [
        'openrsync: protocol version 29\nrsync version 2.6.9 compatible',
        'rsync  version 3.1.3  protocol version 31'
    ]
)
def test_old_rsync_is_refused(banner):
    with pytest.raises(ValueError, match='brew install rsync'):
        gnu_rsync_version(banner)

@pytest.mark.parametrize('version', ['3.2.0', '3.5.1'])
def test_gnu_rsync_passes(version):
    assert gnu_rsync_version(f'rsync  version {version}  protocol version 31') == tuple(
        int(part) for part in version.split('.')
    )

def test_pull_uses_partial_files_without_deleting(recorded_transport_runner, tmp_path):
    runner = recorded_transport_runner
    transport = SshTransport('ubuntu@192.0.2.1', '/home/ubuntu/naics embedder', 'rsync', runner)
    transport.pull('/home/ubuntu/checkpoints', tmp_path, ('run/last.ckpt', 'run/a ;$.ckpt'))
    call = runner.calls[-1]
    assert '--from0' in call.args and '--files-from=-' in call.args
    assert '--partial-dir=.rsync-partial' in call.args and '--checksum' in call.args
    assert '-rlpt' in call.args and '--protect-args' in call.args
    assert not any(arg.startswith(('--delete', '--inplace', '--append')) for arg in call.args)
    assert call.kwargs['input'] == b'run/last.ckpt\0run/a ;$.ckpt\0'
    assert call.kwargs['shell'] is False and call.kwargs['timeout'] > 0

@pytest.mark.parametrize('path', ['../escape', '/absolute', 'a/../../b', 'a\x00b', '.ssh/id_rsa'])
def test_push_rejects_escaping_and_credentials(path, recorded_transport_runner, tmp_path):
    transport = SshTransport('ubuntu@192.0.2.1', '/repo', 'rsync', recorded_transport_runner)
    with pytest.raises(ValueError):
        transport.push(tmp_path, '/repo', (path, ))

def test_probe_quotes_literal_paths_and_uses_stdin(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('ubuntu@192.0.2.1', '/repo ;$(literal)', 'rsync', runner)
    runner.replies.append(
        subprocess.CompletedProcess([], 0,
                                    json.dumps({
                                        'repo': '/repo ;$(literal)'
                                    }).encode(), b'')
    )
    transport.probe('identity', {})
    call = runner.calls[-1]
    assert 'BatchMode=yes' in call.args and 'StrictHostKeyChecking=accept-new' in call.args
    assert 'ConnectTimeout=15' in call.args
    command = shlex.split(call.args[-1])
    assert command[:2] == ['python3', '-c']
    assert json.loads(call.kwargs['input'])['repo'] == '/repo ;$(literal)'
    with pytest.raises(ValueError, match='operation'):
        transport.probe('shell', {'command': 'anything'})

def test_changed_host_key_gives_manual_instruction(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('ubuntu@192.0.2.1', '/repo', 'rsync', runner)
    runner.replies.append(
        subprocess.CompletedProcess(
            [], 255, b'', b'WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!'
        )
    )
    with pytest.raises(RuntimeError, match='ssh-keygen -R 192.0.2.1'):
        transport.probe('identity', {})

def test_checksum_parses_only_itemized_changes(recorded_transport_runner, tmp_path):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/repo', 'rsync', runner)
    runner.replies.append(
        subprocess.CompletedProcess([], 0, b'>fc........|run/a b.ckpt\n.d..t......|run/\n', b'')
    )
    assert transport.checksum(PullMapping('/repo/checkpoints', tmp_path)) == (
        'run/a b.ckpt', 'run/'
    )

def test_explicit_deletion_preserves_protected_paths(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/gone.py').write_text('old')
    assert run_probe('edits', {'remove': ['src/gone.py']}, tmp_path) == {'removed': ['src/gone.py']}
    assert not (tmp_path / 'src/gone.py').exists()
    with pytest.raises(ValueError):
        run_probe('edits', {'remove': ['data/keep.parquet']}, tmp_path)

def test_worker_inventory_and_records(tmp_path):
    run_probe(
        'write_record', {
            'path': '.remote/segments/one/segment.json',
            'record': {
                'id': 'one'
            }
        }, tmp_path
    )
    inventory = run_probe('inventory', {'path': '.remote/segments'}, tmp_path)
    assert inventory['files'][0]['path'] == 'one/segment.json'
    assert len(inventory['files'][0]['sha256']) == 64
    with pytest.raises(ValueError):
        run_probe('write_record', {'path': '../bad', 'record': {}}, tmp_path)

def test_prerequisite_is_system_python_before_upload(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/not-uploaded', 'rsync', runner)
    transport.probe('transport_prerequisites', {})
    argv = shlex.split(runner.calls[-1].args[-1])
    assert argv[:2] == ['python3', '-c'] and 'naics_embedder' not in argv[2]
    assert runner.calls[-1].kwargs['timeout'] == 700

@pytest.mark.parametrize(
    'initial,installed,sudo_exit,success', [
        ('missing', '3.5.1', 0, True),
        ('3.1.3', '3.2.7', 0, True),
        ('missing', '3.5.1', 1, False),
        ('missing', '3.1.3', 0, False),
    ]
)
def test_prerequisite_injected_executables(tmp_path, initial, installed, sudo_exit, success):
    import os
    import sys

    from naics_embedder.remote.transport import PREREQUISITE_CODE
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'calls'
    rsync = bin_dir / 'rsync'
    if initial != 'missing':
        rsync.write_text(f'#!/bin/sh\necho "rsync  version {initial}  protocol version 31"\n')
        rsync.chmod(0o755)
    sudo = bin_dir / 'sudo'
    sudo.write_text(
        '#!/bin/sh\necho "$*" >> "$TEST_LOG"\n'
        'if [ "$TEST_SUDO_EXIT" != 0 ]; then exit "$TEST_SUDO_EXIT"; fi\n'
        'case "$*" in *install*) printf "#!/bin/sh\\necho \'rsync  version ' + installed
        + '  protocol version 31\'\\n" > "$TEST_RSYNC"; '
        '/bin/chmod +x "$TEST_RSYNC";; esac\n'
    )
    sudo.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_dir),
        TEST_LOG=str(log),
        TEST_RSYNC=str(rsync),
        TEST_SUDO_EXIT=str(sudo_exit)
    )
    result = subprocess.run(
        [sys.executable, '-c', PREREQUISITE_CODE], input=b'{}', env=env, capture_output=True
    )
    assert (result.returncode == 0) == success, result.stderr.decode()
    assert 'apt-get' in log.read_text()
    if success:
        rerun = subprocess.run(
            [sys.executable, '-c', PREREQUISITE_CODE], input=b'{}', env=env, capture_output=True
        )
        assert rerun.returncode == 0
        assert log.read_text().count('install') == 1

def test_worker_refuses_symlink_escapes(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / '.remote').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        run_probe('write_record', {'path': '.remote/segments/one/a.json', 'record': {}}, root)

def test_local_transport_refuses_destination_symlink(tmp_path):
    from naics_embedder.remote.transport import LocalTransport
    transport = object.__new__(LocalTransport)
    transport.root = tmp_path / 'instance'
    transport.root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (transport.root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        transport._path('escape/file')

def test_pull_rejects_destination_symlink(recorded_transport_runner, tmp_path):
    root = tmp_path / 'results'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'run').symlink_to(outside, target_is_directory=True)
    transport = SshTransport('u@host', '/repo', 'rsync', recorded_transport_runner)
    with pytest.raises(ValueError, match='escapes'):
        transport.pull('/repo/checkpoints', root, ('run/last.ckpt', ))

def test_push_rejects_source_escape(recorded_transport_runner, tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (root / 'secret').symlink_to(outside)
    transport = SshTransport('u@host', '/repo', 'rsync', recorded_transport_runner)
    with pytest.raises(ValueError, match='escapes'):
        transport.push(root, '/repo', ('secret', ))

@pytest.mark.parametrize('operation', ['inventory', 'training'])
def test_initial_gates_do_not_require_uploaded_package(recorded_transport_runner, operation):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/repo', 'rsync', runner)
    transport.probe(operation, {'path': '.', 'segment_id': 'fixture'})
    argv = shlex.split(runner.calls[-1].args[-1])
    assert argv[:2] == ['python3', '-c'] and 'naics_embedder' not in argv[2]

@pytest.mark.parametrize('path', ['.', './', '', 'data'])
def test_deletion_root_and_protected_paths_refused(tmp_path, path):
    with pytest.raises(ValueError):
        run_probe('edits', {'remove': [path]}, tmp_path)

def test_initial_deletions_use_shared_system_probe(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/repo', 'rsync', runner)
    transport.remove_code(('src/old.py', ))
    argv = shlex.split(runner.calls[-1].args[-1])
    assert argv[:2] == ['python3', '-c'] and 'naics_embedder' not in argv[2]

def test_uninspectable_tmux_status_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, '', 'permission denied'
        )
    )
    with pytest.raises(RuntimeError, match='unable to inspect'):
        run_probe('training', {}, tmp_path)

@pytest.mark.parametrize('evidence', [None, {'native_bf16': False}, {'native_bf16': True}])
def test_real_bootstrap_requires_complete_native_gpu_evidence(recorded_transport_runner, evidence):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/repo', 'rsync', runner)
    reply = {'accelerator': 'cuda', 'gpu_evidence': evidence, 'python': '/repo/.venv/bin/python'}
    runner.replies.append(subprocess.CompletedProcess([], 0, json.dumps(reply).encode(), b''))
    with pytest.raises(RuntimeError, match='native BF16'):
        transport.probe('bootstrap', {})

@pytest.mark.parametrize('operation', ['inventory', 'edits'])
def test_rendered_system_probe_executes_without_package(tmp_path, operation):
    import sys

    from naics_embedder.remote.transport import _system_probe_code
    (tmp_path / 'tiny.py').write_text('fixture')
    payload = {'repo': str(tmp_path), 'path': '.'}
    if operation == 'edits':
        payload['remove'] = ['tiny.py']
    result = subprocess.run(
        [sys.executable, '-I', '-c', _system_probe_code(operation)],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=True
    )
    reply = json.loads(result.stdout)
    if operation == 'inventory':
        assert reply['files'][0]['path'] == 'tiny.py'
        assert reply['files'][0]['mtime'] > 0
    else:
        assert reply == {'removed': ['tiny.py']}
        assert not (tmp_path / 'tiny.py').exists()

def test_launch_refuses_unowned_script_before_any_process(tmp_path, monkeypatch):
    (tmp_path / 'unowned.sh').write_text('echo unowned')

    def forbidden(*args, **kwargs):
        raise AssertionError('no process may run for an invalid launch path')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='owned segment'):
        run_probe(
            'training', {
                'action': 'launch',
                'segment_id': 'one',
                'script': 'unowned.sh'
            }, tmp_path
        )

def test_launch_gpu_failure_prevents_tmux(tmp_path, monkeypatch):
    from naics_embedder.remote import worker
    directory = tmp_path / '.remote/segments/one'
    directory.mkdir(parents=True)
    script = directory / 'launch.sh'
    script.write_text('exit 0')
    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})

    def refused():
        raise RuntimeError('native BF16 qualification failed')

    monkeypatch.setattr(worker, 'gpu_evidence', refused)

    def forbidden(*args, **kwargs):
        raise AssertionError('tmux must not run after GPU refusal')

    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(RuntimeError, match='native BF16'):
        run_probe(
            'training', {
                'action': 'launch',
                'segment_id': 'one',
                'script': str(script)
            }, tmp_path
        )

@pytest.mark.parametrize('rendered', [False, True])
@pytest.mark.parametrize(
    'diagnostic,stopped', [
        ('error connecting to /tmp/tmux-1000/default (No such file or directory)', True),
        ('error connecting to /tmp/tmux-1000/default (Permission denied)', False),
        ('error connecting to /tmp/tmux-1000/default (Connection refused)', False),
        ('No such file or directory: unrelated configuration file', False),
    ]
)
def test_missing_server_socket_and_real_inspection_failure(
    tmp_path, monkeypatch, rendered, diagnostic, stopped
):
    import os
    import sys

    from naics_embedder.remote.transport import _system_probe_code
    if rendered:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        executable = bin_dir / 'tmux'
        executable.write_text(
            '#!/bin/sh\nprintf "%s\\n" ' + shlex.quote(diagnostic) + ' >&2\nexit 1\n'
        )
        executable.chmod(0o755)
        result = subprocess.run(
            [sys.executable, '-I', '-c', _system_probe_code('training')],
            input=b'{}',
            env=dict(os.environ, PATH=str(bin_dir)),
            capture_output=True
        )
        if stopped:
            assert result.returncode == 0, result.stderr.decode()
            assert json.loads(result.stdout) == {'running': False, 'sessions': []}
        else:
            assert result.returncode != 0 and b'unable to inspect' in result.stderr
    else:
        monkeypatch.setattr(
            subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 1, '', diagnostic
            )
        )
        if stopped:
            assert run_probe('training', {}, tmp_path) == {'running': False, 'sessions': []}
        else:
            with pytest.raises(RuntimeError, match='unable to inspect'):
                run_probe('training', {}, tmp_path)

@pytest.mark.parametrize('rendered', [False, True])
def test_status_uses_fixed_session_even_with_segment_metadata(tmp_path, monkeypatch, rendered):
    import os
    import sys

    from naics_embedder.remote.transport import _system_probe_code
    output = 'naics-train\nnaics-other-segment\nunrelated\n'
    payload = {'segment_id': 'record-20261005'}
    if rendered:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        executable = bin_dir / 'tmux'
        executable.write_text('#!/bin/sh\nprintf "%s" ' + shlex.quote(output) + '\n')
        executable.chmod(0o755)
        result = subprocess.run(
            [sys.executable, '-I', '-c', _system_probe_code('training')],
            input=json.dumps(payload).encode(),
            env=dict(os.environ, PATH=str(bin_dir)),
            capture_output=True,
            check=True
        )
        reply = json.loads(result.stdout)
    else:
        monkeypatch.setattr(
            subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(
                [], 0, output, ''
            )
        )
        reply = run_probe('training', payload, tmp_path)
    assert reply == {'running': True, 'sessions': ['naics-train']}

def test_launch_and_interrupt_use_one_exact_tmux_target(tmp_path, monkeypatch):
    from naics_embedder.remote import worker
    directory = tmp_path / '.remote/segments/record-one'
    directory.mkdir(parents=True)
    script = directory / 'launch.sh'
    script.write_text('exit 0')
    calls = []

    def recorded(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})
    monkeypatch.setattr(worker, 'gpu_evidence', lambda: object())
    monkeypatch.setattr(subprocess, 'run', recorded)
    run_probe(
        'training', {
            'action': 'launch',
            'segment_id': 'record-one',
            'script': str(script)
        }, tmp_path
    )
    run_probe('training', {'action': 'interrupt', 'segment_id': 'record-two'}, tmp_path)
    assert calls[0][0][:6] == [
        'tmux', 'new-session', '-d', '-s', 'naics-train',
        'bash ' + shlex.quote(str(script)) + ' < /dev/null'
    ]
    assert calls[0][1]['stdin'] == subprocess.DEVNULL
    assert calls[1][0] == ['tmux', 'send-keys', '-t', 'naics-train', 'C-c']
