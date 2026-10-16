'''Pinned offline cache transfer and fresh/exact launch environment contracts.'''

import json
import os
import subprocess

import pytest

from naics_embedder.remote.launch import _wrapper
from naics_embedder.remote.session import RemoteInfo, read_state
from naics_embedder.utils.config import Config

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

@pytest.fixture
def cached_snapshot(tmp_path):
    hub = tmp_path / 'hub'
    model = hub / 'models--sentence-transformers--all-MiniLM-L6-v2'
    snapshot = model / 'snapshots' / REVISION
    snapshot.mkdir(parents=True)
    (model / 'refs').mkdir()
    (model / 'refs/main').write_text(REVISION)
    for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']:
        (snapshot / name).write_text(json.dumps({'fixture': name}))
    root = tmp_path / 'cache-repo'
    root.mkdir()
    return root, hub, snapshot

@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('visibility', [None, '2,0'])
def test_recorded_wrapper_enforces_offline_cache_in_actual_child(tmp_path, resume, visibility):
    root = tmp_path / 'instance with spaces'
    root.mkdir()
    executable = root / 'uv'
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n%s\\n%s\\n" "$HF_HUB_OFFLINE" "$HF_HUB_CACHE" '
        '"${CUDA_VISIBLE_DEVICES-unset}" > observed\nexit 0\n'
    )
    executable.chmod(0o755)
    info = RemoteInfo(
        str(root), str(root / 'checkpoints'), str(executable), '/python', True, 'cuda', 'fixture'
    )
    argv = (str(executable), 'run', '--locked', 'naics-embedder', 'train')
    if resume:
        argv += ('--ckpt-path', 'last', '--checkpoint-load-mode', 'exact')
    script = root / 'launch.sh'
    script.write_text(_wrapper(info, argv, str(root / 'exit_code'), visibility))
    result = subprocess.run(
        ['bash', str(script)],
        env=dict(
            os.environ, HF_HUB_OFFLINE='0', HF_HUB_CACHE='/wrong', CUDA_VISIBLE_DEVICES='wrong'
        ),
        capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert (root / 'observed').read_text().splitlines() == [
        '1', str(root / '.remote/model-cache/hub'), visibility or 'unset'
    ]
    assert (root / 'exit_code').read_text() == '0\n'

def test_staging_materializes_blob_links_and_verifies_complete_inventory(cached_snapshot):
    from naics_embedder.remote.model_cache import stage_model_cache, verify_model_cache
    root, hub, snapshot = cached_snapshot
    blobs = snapshot.parents[1] / 'blobs'
    blobs.mkdir()
    source = blobs / 'weights'
    source.write_text('cached weights')
    (snapshot / 'model.safetensors').unlink()
    (snapshot / 'model.safetensors').symlink_to(source)
    record = stage_model_cache(root, Config(), hub=hub)
    assert record['revisions'] == {MINILM: REVISION}
    assert verify_model_cache(root, record) == record
    target = root / '.remote/model-cache/hub/models--sentence-transformers--all-MiniLM-L6-v2'
    assert (target / 'refs/main').read_text() == REVISION
    assert (target / 'snapshots' / REVISION / 'model.safetensors').read_text() == 'cached weights'
    assert not any(path.is_symlink() for path in target.rglob('*'))
    assert stage_model_cache(root, Config(), hub=hub) == record

@pytest.mark.parametrize('failure', ['missing', 'wrong-ref', 'escaping-link'])
def test_staging_refuses_unqualified_local_snapshot(cached_snapshot, failure):
    from naics_embedder.remote.model_cache import stage_model_cache
    root, hub, snapshot = cached_snapshot
    if failure == 'missing':
        (snapshot / 'model.safetensors').unlink()
    elif failure == 'wrong-ref':
        (snapshot.parents[1] / 'refs/main').write_text('f' * 40)
    else:
        outside = hub.parent / 'outside'
        outside.write_text('foreign weights')
        (snapshot / 'model.safetensors').unlink()
        (snapshot / 'model.safetensors').symlink_to(outside)
    with pytest.raises(ValueError, match='cache|snapshot|revision'):
        stage_model_cache(root, Config(), hub=hub)
    assert not (root / '.remote/model-cache/hub').exists()

@pytest.mark.parametrize('failure', ['changed', 'extra', 'missing', 'link'])
def test_verified_cache_refuses_tampering_without_repair(cached_snapshot, failure):
    from naics_embedder.remote.model_cache import stage_model_cache, verify_model_cache
    root, hub, _ = cached_snapshot
    record = stage_model_cache(root, Config(), hub=hub)
    target = root / next(name for name in record['hashes'] if name.endswith('model.safetensors'))
    if failure == 'changed':
        target.write_text('changed')
    elif failure == 'extra':
        (target.parent / 'unexpected').write_text('extra')
    elif failure == 'missing':
        target.unlink()
    else:
        target.unlink()
        target.symlink_to(hub / 'missing')
    with pytest.raises(ValueError, match='cache'):
        verify_model_cache(root, record)
    with pytest.raises(ValueError, match='cache'):
        stage_model_cache(root, Config(), hub=hub)

def test_cuda_up_transfers_cache_before_ready(
    remote_workflow_fixture, cached_snapshot, monkeypatch
):
    import naics_embedder.remote.model_cache as cache
    root, hub, _ = cached_snapshot
    env = remote_workflow_fixture
    monkeypatch.setattr(cache, 'HF_HUB_CACHE', str(hub))
    original = env.transport.probe

    def probe(operation, payload):
        result = original(operation, payload)
        if operation == 'bootstrap':
            result['accelerator'] = 'cuda'
        return result

    env.transport.probe = probe
    state = env.workflow.up('fixture', 'conf/config.yaml', [])
    envelope = json.loads((env.root / '.remote/session-model-cache.json').read_text())
    assert envelope['session_id'] == state.session_id and envelope['push_id'] == state.push_id
    record = envelope['cache']
    assert cache.verify_model_cache(env.instance, record) == record
    assert cache.verify_model_cache(env.root, record) == record
    assert state.status == 'ready'

@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('side', ['local', 'remote'])
def test_launch_refuses_cache_tampering_before_continuation_upload(
    remote_launch_fixture, cached_snapshot, resume, side
):
    from naics_embedder.remote.model_cache import stage_model_cache
    env = remote_launch_fixture
    _, hub, _ = cached_snapshot
    if resume:
        env.set_unfinished()
    else:
        import shutil
        shutil.rmtree(env.directory)
    record = stage_model_cache(env.root, env.cfg, hub=hub)
    env.transport.push(env.root, str(env.instance), tuple(record['hashes']))
    state = read_state(env.root)
    (env.root / '.remote/session-model-cache.json').write_text(
        json.dumps({
            'session_id': state.session_id,
            'push_id': state.push_id,
            'cache': record
        })
    )
    target = (env.root if side == 'local' else env.instance) / next(
        name for name in record['hashes'] if name.endswith('model.safetensors')
    )
    target.write_text('tampered')
    env.transport.calls.clear()
    with pytest.raises(ValueError, match='cache'):
        env.workflow.train(resume, 'conf/config.yaml', env.overrides)
    assert not any(call[0] in {'push', 'launch'} for call in env.transport.calls)

@pytest.mark.parametrize('failure', ['model-link', 'refs-link', 'credential'])
def test_cache_source_refuses_linked_roots_and_credential_files(cached_snapshot, failure):
    from naics_embedder.remote.model_cache import stage_model_cache
    root, hub, snapshot = cached_snapshot
    model = snapshot.parents[1]
    if failure == 'model-link':
        moved = hub.parent / 'foreign-model'
        model.rename(moved)
        model.symlink_to(moved, target_is_directory=True)
    elif failure == 'refs-link':
        moved = hub.parent / 'foreign-refs'
        (model / 'refs').rename(moved)
        (model / 'refs').symlink_to(moved, target_is_directory=True)
    else:
        (snapshot / '.env').write_text('fixture, not a credential')
    with pytest.raises(ValueError, match='cache'):
        stage_model_cache(root, Config(), hub=hub)
    assert not (root / '.remote/model-cache/hub').exists()

@pytest.mark.parametrize('failure', ['missing', 'changed'])
def test_worker_refuses_missing_or_changed_cache_before_tmux(tmp_path, monkeypatch, failure):
    from dataclasses import asdict

    import naics_embedder.remote.worker as worker
    from naics_embedder.remote.session import GpuEvidence
    from tests.fixtures.remote import write_offline_cache
    root = tmp_path / 'instance'
    segment = root / '.remote/segments/segment'
    segment.mkdir(parents=True)
    record = write_offline_cache(root)
    evidence = GpuEvidence(0, 'fixture', (8, 0), 1, True, None)
    payload = {'argv': ['/uv'], 'cuda_visible_devices': None, 'gpu_evidence': asdict(evidence)}
    if failure == 'changed':
        payload['model_cache'] = record
        (root / next(iter(record['hashes']))).write_text('altered')
    (segment / 'segment.json').write_text(json.dumps(payload))
    (segment / 'launch.sh').write_text('fixture')
    (root / '.remote/launch.lock').write_text('segment')
    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})
    monkeypatch.setattr(worker, 'gpu_evidence', lambda: evidence)

    def forbidden(*args, **kwargs):
        pytest.fail('tmux reached before cache refusal')

    monkeypatch.setattr(worker.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='cache'):
        worker.run_probe(
            'training', {
                'action': 'launch',
                'segment_id': 'segment',
                'script': str(segment / 'launch.sh')
            }, root
        )

@pytest.mark.parametrize('failure', ['changed', 'missing', 'empty'])
def test_repeated_cuda_up_refuses_existing_cache_without_repair(
    remote_workflow_fixture, cached_snapshot, monkeypatch, failure
):
    import shutil
    from datetime import timedelta

    import naics_embedder.remote.model_cache as cache
    env = remote_workflow_fixture
    _, hub, _ = cached_snapshot
    monkeypatch.setattr(cache, 'HF_HUB_CACHE', str(hub))
    original = env.transport.probe

    def probe(operation, payload):
        result = original(operation, payload)
        if operation == 'bootstrap':
            result['accelerator'] = 'cuda'
        return result

    env.transport.probe = probe
    env.workflow.up('fixture', 'conf/config.yaml', [])
    target = next((env.instance / cache.CACHE_PATH).rglob('model.safetensors'))
    if failure == 'changed':
        target.write_text('altered')
    elif failure == 'missing':
        target.unlink()
    else:
        shutil.rmtree(env.instance / cache.CACHE_PATH)
        (env.instance / cache.CACHE_PATH).mkdir()
    env.now[0] += timedelta(seconds=2)
    env.transport.calls.clear()
    with pytest.raises(ValueError, match='cache'):
        env.workflow.up('fixture', 'conf/config.yaml', [])
    assert not any(
        call[0] == 'push' and any(name.startswith(cache.CACHE_PATH) for name in call[3])
        for call in env.transport.calls
    )
    if failure == 'changed':
        assert target.read_text() == 'altered'
    else:
        assert not target.exists()

@pytest.mark.parametrize('index', ['model.safetensors.index.json', 'pytorch_model.bin.index.json'])
def test_staging_refuses_indexed_weights_until_completeness_is_supported(cached_snapshot, index):
    from naics_embedder.remote.model_cache import stage_model_cache
    root, hub, snapshot = cached_snapshot
    (snapshot / 'model.safetensors').unlink()
    (snapshot / index).write_text(json.dumps({'weight_map': {'weight': 'absent-shard'}}))
    with pytest.raises(ValueError, match='weights'):
        stage_model_cache(root, Config(), hub=hub)
    assert not (root / '.remote/model-cache/hub').exists()

@pytest.mark.parametrize('present', [False, True])
def test_system_inventory_distinguishes_absent_cache_from_empty_cache(tmp_path, present):
    import sys

    from naics_embedder.remote.model_cache import CACHE_PATH
    from naics_embedder.remote.transport import _system_probe_code
    if present:
        (tmp_path / CACHE_PATH).mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, '-I', '-c', _system_probe_code('inventory')],
        input=json.dumps({
            'repo': str(tmp_path),
            'path': CACHE_PATH
        }),
        text=True,
        capture_output=True,
        check=True
    )
    assert json.loads(result.stdout) == {'exists': present, 'files': []}

def test_staging_refuses_safetensors_index_even_with_complete_pytorch_weights(cached_snapshot):
    from naics_embedder.remote.model_cache import stage_model_cache
    root, hub, snapshot = cached_snapshot
    (snapshot / 'model.safetensors').unlink()
    (snapshot / 'pytorch_model.bin').write_text('complete fixture weights')
    (snapshot / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {
            'weight': 'absent-shard'
        }})
    )
    with pytest.raises(ValueError, match='weights'):
        stage_model_cache(root, Config(), hub=hub)
    assert not (root / '.remote/model-cache/hub').exists()
