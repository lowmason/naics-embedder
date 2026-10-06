import pytest

from naics_embedder.remote.session import read_state
from tests.unit.test_remote_up import up

def test_remote_edit_refuses_before_overwrite(remote_workflow_fixture):
    env = remote_workflow_fixture
    up(env)
    (env.instance / 'src/tiny.py').write_text('REMOTE = 2\n')
    count = len(env.transport.calls)
    with pytest.raises(ValueError, match='src/tiny.py'):
        up(env)
    assert not any(c[0] == 'push' for c in env.transport.calls[count:])
    assert (env.instance / 'src/tiny.py').read_text() == 'REMOTE = 2\n'
    up(env, force=True)
    journals = list((env.root / '.remote').rglob('discard*.json'))
    assert journals and 'src/tiny.py' in journals[0].read_text()

def test_initial_unrecorded_code_requires_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    (env.instance / 'surprise.py').write_text('SURPRISE = 1\n')
    with pytest.raises(ValueError, match='surprise.py'):
        up(env)
    up(env, force=True)
    assert not (env.instance / 'surprise.py').exists()

def test_interrupted_retry_preserves_baseline_and_blocks_unrelated(remote_workflow_fixture):
    env = remote_workflow_fixture
    initial = up(env)
    (env.root / 'src/tiny.py').write_text('VALUE = 2\n')
    env.transport.push_error = True
    with pytest.raises(RuntimeError, match='interrupted'):
        up(env)
    pending = read_state(env.root)
    assert pending.push_id == initial.push_id and pending.pending_push_id
    env.transport.push_error = False
    (env.instance / 'CLAUDE.md').write_text('unrelated edit\n')
    with pytest.raises(ValueError, match='CLAUDE.md'):
        up(env)
    (env.instance / 'CLAUDE.md').write_text('Fixture repository\n')
    assert up(env).pending_push_id is None

def test_previous_tracked_runtime_file_deleted_only_exact_name(remote_workflow_fixture):
    import subprocess
    env = remote_workflow_fixture
    (env.root / 'outputs').mkdir()
    (env.root / 'outputs/tracked.txt').write_text('tracked\n')
    subprocess.run(['git', '-C', str(env.root), 'add', 'outputs/tracked.txt'], check=True)
    up(env)
    (env.instance / 'outputs/generated.txt').write_text('generated\n')
    (env.root / 'outputs/tracked.txt').unlink()
    up(env)
    assert not (env.instance / 'outputs/tracked.txt').exists()
    assert (env.instance / 'outputs/generated.txt').read_text() == 'generated\n'

def test_remote_type_replacement_refuses_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    up(env)
    path = env.instance / 'src/tiny.py'
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match='type|directory'):
        up(env, force=True)
    assert path.is_dir()

def test_pending_new_bytes_allowed_but_other_new_file_blocked(remote_workflow_fixture):
    env = remote_workflow_fixture
    up(env)
    (env.root / 'AAA.py').write_text('new code\n')
    env.transport.fail_after = 'AAA.py'
    env.transport.push_error = True
    with pytest.raises(RuntimeError):
        up(env)
    assert (env.instance / 'AAA.py').exists()
    env.transport.push_error = False
    (env.instance / 'unrelated.py').write_text('unrelated\n')
    with pytest.raises(ValueError, match='unrelated.py'):
        up(env)
    (env.instance / 'unrelated.py').unlink()
    up(env)

def test_force_journal_never_names_generated_credentials(remote_workflow_fixture):
    env = remote_workflow_fixture
    up(env)
    (env.instance / '.aws').mkdir()
    (env.instance / '.aws/credentials').write_text('sensitive')
    assert (env.instance / 'outputs').is_dir()
    (env.instance / 'outputs/result').write_text('result')
    (env.instance / 'new.py').write_text('new code')
    up(env, force=True)
    journal = next((env.root / '.remote/instance-edits').rglob('discard*.json')).read_text()
    assert 'new.py' in journal
    assert all(name not in journal for name in ('credentials', 'sensitive', 'outputs/result'))
    assert (env.instance / '.aws/credentials').read_text() == 'sensitive'

def test_deleted_instance_path_requires_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    up(env)
    (env.instance / 'src/tiny.py').unlink()
    with pytest.raises(ValueError, match='src/tiny.py'):
        up(env)
    up(env, force=True)
    assert (env.instance / 'src/tiny.py').exists()

def test_custom_manifest_filename_uploads_same_relative_paths(remote_workflow_fixture):
    env = remote_workflow_fixture
    custom = env.repo.manifest.with_name('custom.json')
    custom.write_bytes(env.repo.manifest.read_bytes())
    env.repo.manifest.unlink()
    state = env.workflow.up(
        'ubuntu@192.0.2.1', 'conf/config.yaml', [
            'supervision.manifest_path=' + str(custom.relative_to(env.root))
        ]
    )
    assert state.status == 'ready'
    assert (env.instance / custom.relative_to(env.root)).read_bytes() == custom.read_bytes()

def test_bad_prerequisite_response_refuses_transfer(remote_workflow_fixture):
    env = remote_workflow_fixture
    original = env.transport.probe

    def probe(operation, payload):
        if operation == 'transport_prerequisites':
            return {'qualified': False}
        return original(operation, payload)

    env.transport.probe = probe
    with pytest.raises(ValueError, match='prerequisite'):
        up(env)
    assert not any(call[0] == 'push' for call in env.transport.calls)

def test_record_verification_does_not_open_unowned_credentials(
    remote_workflow_fixture, monkeypatch
):
    from pathlib import Path
    env = remote_workflow_fixture
    original_push = env.transport.push
    original_open = Path.open

    def push(source, destination, files):
        original_push(source, destination, files)
        if '/.remote/pushes/' in destination:
            path = Path(destination) / '.aws/credentials'
            path.parent.mkdir()
            path.write_text('private')

    def guard(path, *args, **kwargs):
        if path.name == 'credentials' and args and args[0] == 'rb':
            raise AssertionError('opened unowned credential content')
        return original_open(path, *args, **kwargs)

    env.transport.push = push
    monkeypatch.setattr(Path, 'open', guard)
    assert up(env).status == 'ready'

@pytest.mark.parametrize('phase', ['code', 'record'])
def test_new_remote_code_during_transfer_blocks_success(remote_workflow_fixture, phase):
    env = remote_workflow_fixture
    original = env.transport.push

    def push(source, destination, files):
        original(source, destination, files)
        is_record = '/.remote/pushes/' in destination
        if (phase == 'record') == is_record and not any(name.startswith('data/') for name in files):
            (env.instance / 'concurrent.py').write_text('remote edit')

    env.transport.push = push
    with pytest.raises(ValueError, match='concurrent.py'):
        up(env)
    assert read_state(env.root).push_id is None

@pytest.mark.parametrize('kind', ['finished', 'missing_marker'])
def test_same_host_new_session_still_inspects_previous_runtime_code(remote_workflow_fixture, kind):
    import subprocess

    from naics_embedder.remote.session import write_state
    env = remote_workflow_fixture
    (env.root / 'outputs').mkdir()
    file = env.root / 'outputs/previous.py'
    file.write_text('tracked')
    subprocess.run(['git', '-C', str(env.root), 'add', 'outputs/previous.py'], check=True)
    initial = up(env)
    file.unlink()
    if kind == 'finished':
        initial.status = 'finished'
        write_state(env.root, initial)
    else:
        (env.instance / '.remote/pushes/session.json').unlink()
    up(env, force=kind == 'missing_marker')
    assert not (env.instance / 'outputs/previous.py').exists()

def test_initial_partial_symlink_transfer_recovers_without_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    env.transport.fail_after = 'AGENTS.md'
    env.transport.push_error = True
    with pytest.raises(RuntimeError, match='interrupted'):
        up(env)
    assert (env.instance / 'AGENTS.md').is_symlink()
    assert not (env.instance / 'CLAUDE.md').exists()
    env.transport.push_error = False
    assert up(env).status == 'ready'
    assert (env.instance / 'AGENTS.md').read_text() == 'Fixture repository\n'

def test_partial_symlink_retry_refuses_unrelated_target(remote_workflow_fixture):
    env = remote_workflow_fixture
    env.transport.fail_after = 'AGENTS.md'
    env.transport.push_error = True
    with pytest.raises(RuntimeError):
        up(env)
    (env.instance / 'AGENTS.md').unlink()
    (env.instance / 'AGENTS.md').symlink_to('unrelated.md')
    env.transport.push_error = False
    with pytest.raises(ValueError, match='AGENTS.md'):
        up(env)

def test_upload_inputs_uses_transport_interface_without_repo_attribute(remote_workflow_fixture):
    from naics_embedder.remote.canonical import canonical_inputs
    from naics_embedder.remote.push import upload_inputs
    env = remote_workflow_fixture
    upload_inputs(canonical_inputs(env.root, env.repo.config), env.transport)
    assert (env.instance / env.repo.manifest.relative_to(env.root)).exists()

def test_removing_pending_link_propagates_exact_deletion_without_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    env.transport.fail_after = 'AGENTS.md'
    env.transport.push_error = True
    with pytest.raises(RuntimeError):
        up(env)
    (env.root / 'AGENTS.md').unlink()
    (env.root / 'CLAUDE.md').unlink()
    env.transport.push_error = False
    assert up(env).status == 'ready'
    assert not (env.instance / 'AGENTS.md').is_symlink()
