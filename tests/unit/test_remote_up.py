from datetime import timedelta

import pytest

from naics_embedder.remote.session import read_state, write_state

HOST = 'ubuntu@192.0.2.1'

def up(env, **kwargs):
    env.now[0] += timedelta(seconds=1)
    return env.workflow.up(HOST, 'conf/config.yaml', [], **kwargs)

def test_invalid_canonical_inputs_upload_nothing(remote_workflow_fixture):
    env = remote_workflow_fixture
    with pytest.raises(ValueError, match='manifest_path'):
        env.workflow.up(HOST, 'conf/config.yaml', ['supervision.manifest_path=null'])
    assert not any(
        call[0] in {'push', 'remove_code'} or call[:2] == ('probe', 'bootstrap')
        for call in env.transport.calls
    )

@pytest.mark.parametrize('failure', ['bootstrap_error', 'canonical_error'])
def test_failed_up_retains_preparing_state(remote_workflow_fixture, failure):
    env = remote_workflow_fixture
    setattr(env.transport, failure, True)
    with pytest.raises((RuntimeError, ValueError)):
        up(env)
    assert read_state(env.root).status == 'preparing'
    assert not (env.instance / '.remote/pushes/session.json').exists()
    setattr(env.transport, failure, False)
    assert up(env).status == 'ready'

def test_ready_host_reuses_session_and_finished_starts_new(remote_workflow_fixture):
    env = remote_workflow_fixture
    first = up(env)
    second = up(env)
    assert second.session_id == first.session_id
    second.status = 'finished'
    write_state(env.root, second)
    assert up(env).session_id != first.session_id

def test_unfinished_host_requires_force_and_journals_loss(remote_workflow_fixture):
    env = remote_workflow_fixture
    state = up(env)
    with pytest.raises(ValueError, match='last sync'):
        env.workflow.up('ubuntu@192.0.2.2', 'conf/config.yaml', [])
    env.now[0] += timedelta(seconds=1)
    state = env.workflow.up('ubuntu@192.0.2.2', 'conf/config.yaml', [], force=True)
    assert state.status == 'ready'
    assert list((env.root / '.remote').rglob('*loss*.json'))

def test_running_refuses_even_force_without_mutation(remote_workflow_fixture):
    env = remote_workflow_fixture
    env.transport.running = True
    with pytest.raises(ValueError, match='running'):
        up(env, force=True)
    assert not any(c[0] == 'push' for c in env.transport.calls)
    assert read_state(env.root) is None

def test_missing_marker_requires_force(remote_workflow_fixture):
    env = remote_workflow_fixture
    first = up(env)
    (env.instance / '.remote/pushes/session.json').unlink()
    with pytest.raises(ValueError, match='marker'):
        up(env)
    assert up(env, force=True).session_id != first.session_id

def test_reconstructed_transport_hydrates_recorded_paths(remote_workflow_fixture):
    env = remote_workflow_fixture
    state = up(env)
    env.transport.repo = 'wrong'
    env.transport.python = None
    up(env)
    assert env.transport.repo == state.remote_info.repo
    assert env.transport.python == state.remote_info.python

def test_wrong_push_marker_requires_force(remote_workflow_fixture):
    import json
    env = remote_workflow_fixture
    state = up(env)
    path = env.instance / '.remote/pushes/session.json'
    path.write_text(json.dumps({'session_id': state.session_id, 'push_id': 'wrong'}))
    with pytest.raises(ValueError, match='marker'):
        up(env)

def test_custom_remote_config_persisted_for_detached_loop(remote_workflow_fixture):
    import json
    env = remote_workflow_fixture
    env.workflow.remote_cfg.sync_interval_seconds = 37
    state = up(env)
    envelope = json.loads((env.root / '.remote/session-config.json').read_text())
    assert envelope == {
        'session_id': state.session_id,
        'remote_config': env.workflow.remote_cfg.model_dump()
    }

def test_preparing_retry_hydrates_none_for_unbootstrapped_python(remote_workflow_fixture):
    env = remote_workflow_fixture
    env.transport.push_error = True
    with pytest.raises(RuntimeError):
        up(env)
    env.transport.push_error = False
    original = env.transport.probe

    def inspect_python(operation, payload):
        if operation == 'training':
            assert env.transport.python is None
        return original(operation, payload)

    env.transport.probe = inspect_python
    up(env)

def test_persisted_run_location_guard_before_mutation(remote_workflow_fixture):
    import json
    env = remote_workflow_fixture
    path = env.root / '.remote/runs/example.json'
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                experiment='example',
                remote_directory='/different/checkpoints/example',
                bundle_id='bundle',
                codebook_fingerprint='code',
                description_fingerprint='desc',
                seed=1,
                settings={},
                constructor_controls={},
                session_id='old'
            )
        )
    )
    with pytest.raises(ValueError, match='checkpoint location'):
        up(env, force=True)
    assert not any(c[0] == 'push' for c in env.transport.calls)

@pytest.mark.parametrize('failure', ['input_hash', 'record_hash', 'local_after_record', 'marker'])
def test_verification_failure_never_ready(remote_workflow_fixture, failure):
    env = remote_workflow_fixture
    original_probe = env.transport.probe
    original_push = env.transport.push

    def probe(operation, payload):
        response = original_probe(operation, payload)
        if failure == 'input_hash' and operation == 'edits' and any(
            name.startswith('data/') for name in payload.get('expected', [])
        ):
            next(item for item in response['files']
                 if item['path'].startswith('data/'))['sha256'] = 'bad'
        if failure == 'record_hash' and operation == 'edits' and any(
            name.startswith('.remote/pushes/') for name in payload.get('expected', [])
        ):
            next(item for item in response['files']
                 if item['path'].startswith('.remote/pushes/'))['sha256'] = 'bad'
        if failure == 'marker' and operation == 'identity' and response.get('session_id'):
            response['push_id'] = 'bad'
        return response

    def push(source, destination, files):
        original_push(source, destination, files)
        if failure == 'local_after_record' and '/.remote/pushes/' in destination:
            (env.root / 'src/tiny.py').write_text('CHANGED = 1\n')

    env.transport.probe = probe
    env.transport.push = push
    with pytest.raises(ValueError):
        up(env)
    assert read_state(env.root).status == 'preparing'

def test_canonical_credential_member_is_refused_before_upload(remote_workflow_fixture):
    env = remote_workflow_fixture
    (env.repo.manifest.parent / '.env').write_text('fixture credential content')
    with pytest.raises(ValueError, match='credential'):
        up(env)
    assert not any(call[0] == 'push' for call in env.transport.calls)

def test_checkpoint_config_change_refused_for_persisted_run(remote_workflow_fixture):
    import json
    env = remote_workflow_fixture
    experiment = env.repo.config.experiment_name
    path = env.root / '.remote/runs' / (experiment + '.json')
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                experiment=experiment,
                remote_directory=str(env.instance / 'checkpoints' / experiment),
                bundle_id='bundle',
                codebook_fingerprint='code',
                description_fingerprint='desc',
                seed=1,
                settings={},
                constructor_controls={},
                session_id='old'
            )
        )
    )
    with pytest.raises(ValueError, match='checkpoint location'):
        env.workflow.up('ubuntu@192.0.2.1', 'conf/config.yaml', ['dirs.checkpoint_dir=alternate'])
    assert not any(call[0] == 'push' for call in env.transport.calls)

def test_checkpoint_base_outside_repo_refused_before_mutation(remote_workflow_fixture):
    env = remote_workflow_fixture
    original = env.transport.probe

    def probe(operation, payload):
        response = original(operation, payload)
        if operation == 'identity':
            response['checkpoint_base'] = '/outside/checkpoints'
        return response

    env.transport.probe = probe
    with pytest.raises(ValueError, match='checkpoint|root'):
        up(env)
    assert not any(call[0] == 'push' for call in env.transport.calls)

def test_bootstrap_gpu_evidence_rehydrates_tuple_contract(remote_workflow_fixture):
    env = remote_workflow_fixture
    original = env.transport.probe

    def probe(operation, payload):
        reply = original(operation, payload)
        if operation == 'bootstrap':
            reply['gpu_evidence'] = dict(
                logical_index=0,
                name='fixture',
                compute_capability=[8, 0],
                total_memory_bytes=1024,
                native_bf16=True,
                cuda_visible_devices=None
            )
        return reply

    env.transport.probe = probe
    state = up(env)
    assert state.remote_info.gpu_evidence.compute_capability == (8, 0)
