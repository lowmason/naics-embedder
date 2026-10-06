'''Transport argv and fixed operation boundary tests; no network.'''

import ast
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
    assert argv[:2] == ['python3', '-c']
    imports = []
    for node in ast.walk(ast.parse(argv[2])):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any(name.startswith('naics_embedder') for name in imports)
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
    assert argv[:2] == ['python3', '-c']
    imports = []
    for node in ast.walk(ast.parse(argv[2])):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any(name.startswith('naics_embedder') for name in imports)

@pytest.mark.parametrize('path', ['.', './', '', 'data'])
def test_deletion_root_and_protected_paths_refused(tmp_path, path):
    with pytest.raises(ValueError):
        run_probe('edits', {'remove': [path]}, tmp_path)

def test_initial_deletions_use_shared_system_probe(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('u@host', '/repo', 'rsync', runner)
    transport.remove_code(('src/old.py', ))
    argv = shlex.split(runner.calls[-1].args[-1])
    assert argv[:2] == ['python3', '-c']
    imports = []
    for node in ast.walk(ast.parse(argv[2])):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert not any(name.startswith('naics_embedder') for name in imports)

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

def _owned_launch_files(root, segment):
    from dataclasses import asdict

    from naics_embedder.remote.launch import _wrapper
    from naics_embedder.remote.session import GpuEvidence, RemoteInfo
    directory = root / '.remote/segments' / segment
    directory.mkdir(parents=True)
    script = directory / 'launch.sh'
    evidence = GpuEvidence(0, 'fixture', (8, 0), 1, True, None)
    argv = ('/uv', 'run', '--locked', 'naics-embedder', 'train')
    info = RemoteInfo(
        str(root), str(root / 'checkpoints'), '/uv', '/python', True, 'cuda', 'fixture'
    )
    script.write_text(_wrapper(info, argv, str(directory / 'exit_code'), None))
    (directory / 'segment.json').write_text(
        json.dumps(
            {
                'argv': list(argv),
                'gpu_evidence': asdict(evidence),
                'cuda_visible_devices': None
            }
        )
    )
    (root / '.remote/launch.lock').write_text(segment)
    return script, evidence

def test_launch_gpu_failure_prevents_tmux(tmp_path, monkeypatch):
    from naics_embedder.remote import worker
    script, _ = _owned_launch_files(tmp_path, 'one')
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
    script, evidence = _owned_launch_files(tmp_path, 'record-one')
    calls = []

    def recorded(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})
    monkeypatch.setattr(worker, 'gpu_evidence', lambda: evidence)
    monkeypatch.setattr(worker, '_training_status', lambda payload: {'running': False})
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

@pytest.mark.parametrize('rendered', [False, True])
def test_controlled_scan_prunes_contents_and_reports_prior_replacements(
    tmp_path, rendered, monkeypatch
):
    import json
    import os
    import subprocess
    from pathlib import Path

    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe
    root = tmp_path / 'instance'
    root.mkdir()
    for name in ['outputs/generated', '.aws/credentials', '.env', '__pycache__/new.pyc']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(path)
    (root / 'outputs/previous.py').mkdir()
    os.mkfifo(root / 'special.py')
    (root / 'source.py').write_text('source')
    (root / 'outside.py').symlink_to(tmp_path / 'outside')
    if not rendered:
        original_open = Path.open

        def refuse_fifo(path, *args, **kwargs):
            if path.name in {'generated', 'credentials', '.env', 'new.pyc', 'special.py'}:
                raise AssertionError('probe opened pruned/special content')
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', refuse_fifo)
    payload = dict(
        repo=str(root),
        controlled=True,
        expected=['outputs/previous.py'],
        ignore=['__pycache__/', '*.pyc']
    )
    if rendered:
        result = subprocess.run(
            ['python3', '-c', _system_probe_code('edits')],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        result = json.loads(result.stdout)
    else:
        result = run_probe('edits', payload, root)
    files = {item['path']: item for item in result['files']}
    assert set(files) == {'outputs/previous.py', 'special.py', 'source.py', 'outside.py'}
    assert files['outputs/previous.py']['kind'] == 'directory'
    assert files['special.py']['kind'] == 'special'
    assert files['outside.py']['kind'] == 'unsafe_symlink'

@pytest.mark.parametrize('rendered', [False, True])
def test_owned_runtime_deletion_preserves_siblings(tmp_path, rendered):
    import json
    import subprocess
    from dataclasses import asdict

    from naics_embedder.remote.code_manifest import file_entry
    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe
    root = tmp_path / 'instance'
    (root / 'outputs').mkdir(parents=True)
    (root / 'outputs/tracked').write_text('owned')
    (root / 'outputs/runtime').write_text('runtime')
    item = asdict(file_entry(root.resolve(), 'outputs/tracked'))
    payload = dict(
        repo=str(root),
        remove=['outputs/tracked'],
        authorized=[item],
        classification='previous',
        previous=[item],
        current_paths=[]
    )
    if rendered:
        result = subprocess.run(
            ['python3', '-c', _system_probe_code('edits')],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        assert json.loads(result.stdout)['removed'] == ['outputs/tracked']
    else:
        run_probe('edits', payload, root)
    assert not (root / 'outputs/tracked').exists()
    assert (root / 'outputs/runtime').read_text() == 'runtime'

def test_runtime_delete_requires_previous_manifest_membership(tmp_path):
    from dataclasses import asdict

    from naics_embedder.remote.code_manifest import file_entry
    from naics_embedder.remote.worker import run_probe
    (tmp_path / 'outputs').mkdir()
    path = tmp_path / 'outputs/generated'
    path.write_text('generated')
    item = asdict(file_entry(tmp_path.resolve(), 'outputs/generated'))
    with pytest.raises(ValueError, match='previous manifest'):
        run_probe(
            'edits',
            dict(
                remove=['outputs/generated'],
                authorized=[item],
                classification='previous',
                previous=[],
                current_paths=[]
            ), tmp_path
        )
    assert path.read_text() == 'generated'

@pytest.mark.parametrize('name', ['with spaces.py', 'line\nbreak.py', 'back\\slash.py'])
@pytest.mark.parametrize('rendered', [False, True])
def test_controlled_scan_preserves_literal_posix_filenames(tmp_path, name, rendered):
    import json
    import subprocess

    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe
    (tmp_path / name).write_text('code')
    payload = dict(repo=str(tmp_path), controlled=True, expected=[name], ignore=[])
    if rendered:
        result = subprocess.run(
            ['python3', '-c', _system_probe_code('edits')],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        response = json.loads(result.stdout)
    else:
        response = run_probe('edits', payload, tmp_path)
    assert [item['path'] for item in response['files']] == [name]

def test_controlled_hash_ignores_access_time_update(tmp_path):
    import os

    from naics_embedder.remote.worker import run_probe
    path = tmp_path / 'code.py'
    path.write_text('constant bytes')
    before = path.stat()
    os.utime(path, ns=(0, before.st_mtime_ns))
    before = path.stat()
    response = run_probe('edits', dict(controlled=True, expected=['code.py'], ignore=[]), tmp_path)
    after = path.stat()
    assert response['files'][0]['size'] == len('constant bytes')
    assert after.st_atime_ns > before.st_atime_ns
    assert (after.st_mtime_ns, after.st_ctime_ns) == (before.st_mtime_ns, before.st_ctime_ns)

def test_controlled_hash_refuses_content_mutation(tmp_path, monkeypatch):

    from naics_embedder.remote.worker import run_probe
    path = tmp_path / 'code.py'
    path.write_text('constant bytes')
    import os
    original = os.read
    changed = False

    def mutate(descriptor, count):
        nonlocal changed
        if not changed and os.fstat(descriptor).st_ino == path.stat().st_ino:
            changed = True
            with path.open('a') as stream:
                stream.write(' changed')
        return original(descriptor, count)

    monkeypatch.setattr(os, 'read', mutate)
    with pytest.raises(ValueError, match='changed during inventory'):
        run_probe('edits', dict(controlled=True, expected=['code.py'], ignore=[]), tmp_path)

@pytest.mark.parametrize('rendered', [False, True])
def test_exact_pending_link_allowed_only_in_pretransfer_scan(tmp_path, rendered):
    import hashlib
    import json
    import subprocess

    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe
    (tmp_path / 'AGENTS.md').symlink_to('CLAUDE.md')
    pending = [
        dict(
            path='AGENTS.md',
            sha256=hashlib.sha256(b'CLAUDE.md').hexdigest(),
            size=9,
            mode=(tmp_path / 'AGENTS.md').lstat().st_mode & 0o777,
            kind='symlink',
            target='CLAUDE.md'
        ),
        dict(path='CLAUDE.md', sha256='expected', size=1, mode=0o644, kind='file', target=None)
    ]

    def probe(entries):
        payload = dict(
            repo=str(tmp_path),
            controlled=True,
            expected=['AGENTS.md', 'CLAUDE.md'],
            ignore=[],
            pending_entries=entries
        )
        if rendered:
            result = subprocess.run(
                ['python3', '-c', _system_probe_code('edits')],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            return json.loads(result.stdout)
        return run_probe('edits', payload, tmp_path)

    assert probe(pending)['files'][0]['kind'] == 'symlink'
    assert probe([])['files'][0]['kind'] == 'unsafe_symlink'
    pending[0]['target'] = 'other.md'
    assert probe(pending)['files'][0]['kind'] == 'unsafe_symlink'

@pytest.mark.parametrize(
    'name', ['.env', '.env.production', 'private.pem', 'nested/id_ed25519', 'nested/private.pfx']
)
def test_generic_code_delete_refuses_credential_filenames(name):
    from naics_embedder.remote.transport import safe_files
    with pytest.raises(ValueError, match='credential'):
        safe_files((name, ), code=True)

@pytest.mark.parametrize('rendered', [False, True])
def test_editable_metadata_pruned_except_previously_pushed_path(tmp_path, rendered):
    import json
    import subprocess

    from naics_embedder.remote.transport import _system_probe_code
    from naics_embedder.remote.worker import run_probe
    directory = tmp_path / 'src/naics_embedder.egg-info'
    directory.mkdir(parents=True)
    (directory / 'PKG-INFO').write_text('previous code')
    (directory / 'SOURCES.txt').write_text('new metadata')
    other = tmp_path / 'src/other.egg-info'
    other.mkdir()
    (other / 'code.py').write_text('qualified unknown code')
    payload = dict(
        repo=str(tmp_path),
        controlled=True,
        expected=['src/naics_embedder.egg-info/PKG-INFO'],
        ignore=[]
    )
    if rendered:
        result = subprocess.run(
            ['python3', '-c', _system_probe_code('edits')],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        response = json.loads(result.stdout)
    else:
        response = run_probe('edits', payload, tmp_path)
    assert {item['path']
            for item in response['files']} == {
                'src/naics_embedder.egg-info/PKG-INFO', 'src/other.egg-info/code.py'
            }

def test_generic_deletion_of_editable_metadata_is_protected():
    from naics_embedder.remote.transport import safe_files
    with pytest.raises(ValueError, match='protected'):
        safe_files(('src/naics_embedder.egg-info/PKG-INFO', ), code=True)

def integrity_functions(rendered):
    from naics_embedder.remote import worker
    from naics_embedder.remote.transport import _system_probe_code
    if not rendered:
        return worker._controlled_inventory, worker._owned_remove
    namespace = {}
    exec(_system_probe_code('edits').split('\np=json.load(sys.stdin)')[0], namespace)
    return namespace['_controlled_inventory'], namespace['_owned_remove']

@pytest.mark.parametrize('rendered', [False, True])
@pytest.mark.parametrize('ancestor', [False, True])
def test_controlled_open_never_follows_swapped_link(tmp_path, monkeypatch, rendered, ancestor):
    import io
    import os
    from pathlib import Path
    scan, _ = integrity_functions(rendered)
    root = tmp_path / 'repo'
    (root / 'owned').mkdir(parents=True)
    victim = root / 'owned/code.py'
    victim.write_text('authorized')
    external = tmp_path / 'external'
    external.mkdir()
    (external / '.env').write_text('fake credential bytes')
    (external / 'code.py').symlink_to('.env')
    original_open, original_io = os.open, io.open
    swapped = False
    external_reads = []

    def swap():
        nonlocal swapped
        if not swapped:
            swapped = True
            if ancestor:
                (root / 'owned').rename(root / 'detached')
                (root / 'owned').symlink_to(external, target_is_directory=True)
            else:
                victim.unlink()
                victim.symlink_to(external / '.env')

    def opening(path, flags, *args, **kwargs):
        if str(path).endswith('code.py'):
            swap()
        descriptor = original_open(path, flags, *args, **kwargs)
        if os.fstat(descriptor).st_ino == (external / 'code.py').stat().st_ino:
            external_reads.append(str(path))
        return descriptor

    def io_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and str(path).endswith('code.py'):
            swap()
            if Path(path).resolve() == external / '.env':
                external_reads.append(str(path))
        return original_io(path, *args, **kwargs)

    monkeypatch.setattr(os, 'open', opening)
    monkeypatch.setattr(io, 'open', io_open)
    try:
        scan(root, dict(expected=['owned/code.py'], ignore=[]))
    except (ValueError, OSError):
        pass
    assert swapped
    assert external_reads == []

@pytest.mark.parametrize('rendered', [False, True])
@pytest.mark.parametrize('change', ['ancestor', 'next_entry'])
def test_owned_unlink_stays_anchored_and_rechecks_each_entry(
    tmp_path, monkeypatch, rendered, change
):
    import os
    scan, remove = integrity_functions(rendered)
    root = tmp_path / 'repo'
    (root / 'owned').mkdir(parents=True)
    for name in ['a.py', 'b.py']:
        (root / 'owned' / name).write_text('authorized')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'a.py').write_text('external bytes')
    records = scan(root, dict(expected=[], ignore=[]))['files']
    original = os.unlink
    changed = False

    def unlink(path, *args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            if change == 'ancestor':
                (root / 'owned').rename(root / 'detached')
                (root / 'owned').symlink_to(external, target_is_directory=True)
            else:
                (root / 'owned/b.py').write_text('unrelated edit')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, 'unlink', unlink)
    try:
        remove(
            root,
            dict(
                remove=['owned/a.py', 'owned/b.py'],
                authorized=records,
                classification='previous',
                previous=records,
                current_paths=[]
            )
        )
    except (ValueError, OSError):
        pass
    assert changed
    assert (external / 'a.py').read_text() == 'external bytes'
    if change == 'next_entry':
        assert (root / 'owned/b.py').read_text() == 'unrelated edit'

@pytest.mark.parametrize('rendered', [False, True])
def test_partial_namespace_pruned_but_previous_members_checked(tmp_path, rendered):
    scan, _ = integrity_functions(rendered)
    directory = tmp_path / 'src/.rsync-partial'
    directory.mkdir(parents=True)
    (directory / 'tiny.py').write_text('generated partial')
    (directory / 'previous.py').write_text('previous code')
    response = scan(tmp_path, dict(expected=['src/.rsync-partial/previous.py'], ignore=[]))
    assert [entry['path'] for entry in response['files']] == ['src/.rsync-partial/previous.py']

def test_hydrated_integrity_probe_uses_fixed_local_source(recorded_transport_runner):
    runner = recorded_transport_runner
    transport = SshTransport('ubuntu@192.0.2.1', '/repo', 'rsync', runner)
    transport.python = '/repo/.venv/bin/python'
    transport.probe('edits', dict(controlled=True, expected=[], ignore=[]))
    command = shlex.split(runner.calls[-1].args[-1])
    assert command[:2] == [transport.python, '-c']
    assert 'naics_embedder.remote.worker' not in command

@pytest.mark.parametrize('rendered', [False, True])
def test_initial_controlled_scan_accepts_absent_repository(tmp_path, rendered):
    scan, _ = integrity_functions(rendered)
    assert scan(tmp_path / 'absent', dict(expected=[], ignore=[])) == {'files': []}
