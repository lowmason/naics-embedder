'''Task 18 fixture evidence; never claims an instance or campaign qualification.'''

import pytest

from naics_embedder.remote.transport import _qualified_gpu, gnu_rsync_version

@pytest.mark.parametrize(
    'field,value', [
        ('native_bf16', False), ('logical_index', 1), ('cuda_visible_devices', 'different')
    ]
)
def test_launch_rechecks_native_device_zero_and_visibility(local_workflow, field, value):
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    env.process.gpu = {**env.process.gpu, field: value}
    with pytest.raises(ValueError, match='BF16|visibility'):
        env.workflow.train(True, 'conf/config.yaml', [])
    assert not any(c[0] == 'launch' for c in env.process.calls)

def test_gpu_probe_error_refuses_launch(local_workflow):
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    env.process.gpu = RuntimeError('uninspectable GPU')
    with pytest.raises(RuntimeError, match='uninspectable'):
        env.workflow.train(True, 'conf/config.yaml', [])
    assert not any(c[0] == 'launch' for c in env.process.calls)

def test_ntp_launch_recheck_refuses(local_workflow):
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    env.process.ntp = False
    with pytest.raises(ValueError, match='NTP'):
        env.workflow.train(True, 'conf/config.yaml', [])
    assert not any(c[0] == 'launch' for c in env.process.calls)

def test_one_device_contract(local_workflow):
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    with pytest.raises(ValueError, match='one device'):
        env.workflow.train(True, 'conf/config.yaml', ['training.trainer.devices=2'])

def test_gnu_discovery_rejects_system_openrsync():
    with pytest.raises(ValueError, match='GNU'):
        gnu_rsync_version('openrsync: protocol version 29')
    assert not _qualified_gpu(dict(logical_index=0, native_bf16=False))

@pytest.mark.parametrize('local_workflow', ['finished'], indirect=True)
def test_actual_finished_trainer_has_no_upload_records_loop_or_gpu(local_workflow, monkeypatch):
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    before = list((env.root / '.remote').rglob('segment.json'))

    def refuse(*args, **kwargs):
        pytest.fail('finished run uploaded files')

    monkeypatch.setattr(env.transport, 'push', refuse)
    env.process.calls.clear()
    result = env.workflow.train(True, 'conf/config.yaml', [])
    assert result.skipped and result.segment_id is None
    assert list((env.root / '.remote').rglob('segment.json')) == before
    assert not env.loops
    assert not any(c[0] == 'launch' or c[:2] == ('probe', 'gpu') for c in env.process.calls)

@pytest.mark.parametrize('local_workflow', ['moe'], indirect=True)
@pytest.mark.parametrize(
    'key', [
        'lora_r', 'lora_alpha', 'lora_dropout', 'num_experts', 'top_k', 'moe_hidden_dim',
        'load_balancing_coef'
    ]
)
def test_missing_active_constructor_control_fails_closed(local_workflow, key):
    from copy import deepcopy

    from naics_embedder.utils.training import read_checkpoint, refuse_other_constructor_settings
    env = local_workflow
    saved = deepcopy(read_checkpoint(env.directory / 'last.ckpt'))
    saved['hyper_parameters'].pop(key)
    with pytest.raises(ValueError, match='constructor'):
        refuse_other_constructor_settings(saved, env.cfg)
    from naics_embedder.supervision.artifacts import sha256_file
    assert env.plan.hashes['checkpoints/qualification/last.ckpt'] == sha256_file(
        env.directory / 'last.ckpt'
    )

def test_current_contract_omits_dropped_precheck_keys(local_workflow):
    from naics_embedder.utils.training import read_checkpoint
    saved = read_checkpoint(local_workflow.directory / 'last.ckpt')
    for key in [
        'supervision_mode', 'structural_preference_loss_version', 'mining_contract_version'
    ]:
        assert key not in saved['stage3_supervision']
    assert len(saved['hyper_parameters']['run_settings']) == 21

def test_required_gnu_mode_fails_missing_tool(request, monkeypatch):
    monkeypatch.setenv('REMOTE_REQUIRE_GNU_RSYNC', '1')
    monkeypatch.setenv('REMOTE_RSYNC', '/nonexistent/gnu-rsync')
    with pytest.raises(pytest.fail.Exception, match='GNU rsync >=3.2'):
        request.getfixturevalue('gnu_rsync')

@pytest.mark.parametrize('field,value', [('ntp', False), ('ntp', None)])
def test_bootstrap_clock_refuses_unavailable_or_unsynchronized(local_workflow, field, value):
    env = local_workflow
    setattr(env.process, field, value)
    with pytest.raises(ValueError, match='NTP'):
        env.workflow.up('a', 'conf/config.yaml', [])

def test_fresh_overrides_own_name_and_devnull(local_workflow):
    import json
    import shutil
    env = local_workflow
    env.workflow.up('a', 'conf/config.yaml', [])
    shutil.rmtree(env.directory)
    result = env.workflow.train(
        False, 'conf/config.yaml', ['experiment_name=fresh-fixture', 'seed=9']
    )
    segment = env.root / '.remote/segments' / result.segment_id
    record = json.loads((segment / 'segment.json').read_text())
    assert record['experiment_name'] == 'fresh-fixture'
    assert 'experiment_name=fresh-fixture' in record['argv'] and 'seed=9' in record['argv']
    assert record['remote_directory'] == str(env.root / 'checkpoints/fresh-fixture')
    assert '< /dev/null' in (segment / 'launch.sh').read_text()
