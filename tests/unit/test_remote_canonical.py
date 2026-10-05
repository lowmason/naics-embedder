'''Canonical input and continuation gates over real tiny trained checkpoints.'''

import json

import pytest
import torch

from naics_embedder.remote.canonical import canonical_inputs, finished_run, resume_plan
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import constructor_settings, read_checkpoint, run_settings

def plan(env):
    return resume_plan(env.root, env.cfg, env.inputs, env.remote_directory)

def mutate_checkpoint(env, change):
    path = env.directory / 'last.ckpt'
    saved = read_checkpoint(path)
    change(saved)
    torch.save(saved, path)

def test_canonical_inputs_include_every_member(remote_repo):
    inputs = canonical_inputs(remote_repo.root, remote_repo.config)
    assert inputs.manifest == remote_repo.manifest
    for name in inputs.bundle.manifest.artifacts:
        for path in inputs.bundle.member_paths(name):
            assert path.relative_to(remote_repo.root).as_posix() in inputs.paths
    assert set(inputs.paths) == set(inputs.hashes)

def test_missing_manifest_has_build_instruction(remote_repo):
    cfg = remote_repo.config.override({'supervision.manifest_path': None})
    with pytest.raises(ValueError, match='data supervision'):
        canonical_inputs(remote_repo.root, cfg)

def test_changed_description_bytes_refused(remote_repo):
    path = remote_repo.root / 'data/naics_descriptions.parquet'
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='description'):
        canonical_inputs(remote_repo.root, remote_repo.config)

@pytest.mark.parametrize('value', ['../outside', '/tmp/outside', 'conf/config.yaml'])
def test_canonical_paths_cannot_escape_data(remote_repo, value):
    cfg = remote_repo.config.override({'supervision.manifest_path': value})
    with pytest.raises(ValueError):
        canonical_inputs(remote_repo.root, cfg)

def test_real_finished_run_and_complete_inventory(remote_resume_fixture):
    env = remote_resume_fixture
    result = plan(env)
    assert result.finished and result.finish_reason == 'epoch budget exhausted'
    assert result.epoch == 2
    assert result.last.name == 'last.ckpt'
    assert set(result.files) == set(result.hashes)
    assert env.transport.calls == []

@pytest.mark.parametrize('name', ['last.ckpt', 'monitor_reads.jsonl', 'epoch_summary.jsonl'])
def test_missing_continuation_refuses_before_transport(remote_resume_fixture, name):
    env = remote_resume_fixture
    (env.directory / name).unlink()
    with pytest.raises(ValueError, match=name):
        plan(env)
    assert env.transport.calls == []

SETTING_KEYS = list(run_settings(Config(), accelerator='cpu', precision='32-true'))

@pytest.mark.parametrize('key', SETTING_KEYS)
@pytest.mark.parametrize('missing', [False, True])
def test_each_setting_is_guarded(remote_resume_fixture, key, missing):
    env = remote_resume_fixture

    def change(saved):
        settings = saved['hyper_parameters']['run_settings']
        if missing:
            del settings[key]
        else:
            settings[key] = 'changed'

    mutate_checkpoint(env, change)
    with pytest.raises(ValueError, match='settings'):
        plan(env)

@pytest.mark.parametrize('key', ['lora_r', 'lora_alpha', 'lora_dropout'])
@pytest.mark.parametrize('missing', [False, True])
def test_lora_controls_are_guarded(remote_resume_fixture, key, missing):
    env = remote_resume_fixture

    def change(saved):
        if missing:
            del saved['hyper_parameters'][key]
        else:
            saved['hyper_parameters'][key] = -1

    mutate_checkpoint(env, change)
    with pytest.raises(ValueError, match='constructor'):
        plan(env)

@pytest.mark.parametrize(
    'field,value', [
        ('epoch', True), ('epoch', -1), ('epoch', 1.5), ('training_run', ''), (
            'training_run', None
        )
    ]
)
def test_bad_checkpoint_identity_refused(remote_resume_fixture, field, value):
    env = remote_resume_fixture
    mutate_checkpoint(env, lambda saved: saved.update({field: value}))
    with pytest.raises(ValueError):
        plan(env)

@pytest.mark.parametrize(
    'field,value', [('panel', 'regressor'), ('split', 'test'), ('event', 'open')]
)
def test_wrong_monitor_read_identity_refused(remote_resume_fixture, field, value):
    env = remote_resume_fixture
    path = env.directory / 'monitor_reads.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['read'][field] = value
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='monitor'):
        plan(env)

@pytest.mark.parametrize('corruption', ['gap', 'duplicate', 'seed', 'run', 'mrr', 'summary'])
def test_incoherent_history_refused(remote_resume_fixture, corruption):
    env = remote_resume_fixture
    path = env.directory / 'monitor_reads.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == 'gap':
        rows.pop(1)
    elif corruption == 'duplicate':
        rows.append(rows[0])
    elif corruption == 'seed':
        rows[0]['read']['detail']['seed'] = 99
    elif corruption == 'run':
        rows[0]['read']['detail']['training_run'] = 'other'
    elif corruption == 'mrr':
        rows[0]['mrr'] = float('nan')
    else:
        rows[0]['mrr'] = rows[0]['mrr'] / 2
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        plan(env)

def test_valid_interrupted_later_rows_remain_byte_identical(remote_resume_fixture):
    env = remote_resume_fixture
    for filename in ['monitor_reads.jsonl', 'epoch_summary.jsonl']:
        path = env.directory / filename
        row = json.loads(path.read_text().splitlines()[-1])
        if filename.startswith('monitor'):
            row['read']['detail']['epoch'] = 3
        else:
            row['epoch'] = 3
        with path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    before = {p.name: p.read_bytes() for p in env.directory.glob('*.jsonl')}
    plan(env)
    assert before == {p.name: p.read_bytes() for p in env.directory.glob('*.jsonl')}

def test_other_directory_and_absent_callback_refused(remote_resume_fixture):
    env = remote_resume_fixture
    with pytest.raises(ValueError, match='another checkpoint directory'):
        resume_plan(env.root, env.cfg, env.inputs, '/home/ubuntu/other')
    mutate_checkpoint(env, lambda saved: saved.update({'callbacks': {}}))
    with pytest.raises(ValueError, match='ModelCheckpoint'):
        plan(env)

def test_early_stop_is_finished_and_corruption_refuses(remote_resume_fixture):
    from naics_embedder.utils.training import outcome_early_stopping
    env = remote_resume_fixture
    saved = read_checkpoint(env.directory / 'last.ckpt')
    key = outcome_early_stopping(env.cfg.training.early_stopping_patience).state_key
    saved['callbacks'][key]['stopped_epoch'] = 2
    assert finished_run(saved, env.cfg.training.early_stopping_patience, 3)[0]
    saved['callbacks'][key]['stopped_epoch'] = 'corrupt'
    with pytest.raises(ValueError):
        finished_run(saved, env.cfg.training.early_stopping_patience, 3)

@pytest.mark.parametrize(
    'field,value', [
        ('bundle_id', 'other'), ('objective', 'pre-req11'), ('summaries', '0' * 64),
        (
            'encoder', {
                'layout': 'shared',
                'fusion': 'masked_mean',
                'dimension': 16,
                'backbone': 'other'
            }
        )
    ]
)
def test_wrong_contract_is_refused(remote_resume_fixture, field, value):
    env = remote_resume_fixture

    def change(saved):
        saved['stage3_supervision'][field] = value

    mutate_checkpoint(env, change)
    with pytest.raises(ValueError, match='contract|objective|architecture'):
        plan(env)

def test_seed_is_guarded(remote_resume_fixture):
    env = remote_resume_fixture
    mutate_checkpoint(env, lambda saved: saved['hyper_parameters'].update({'seed': 99}))
    with pytest.raises(ValueError, match='seed'):
        plan(env)

def test_inactive_moe_controls_are_ignored(remote_resume_fixture):
    env = remote_resume_fixture
    mutate_checkpoint(
        env, lambda saved: saved['hyper_parameters'].update(
            {
                'num_experts': 999,
                'top_k': 999,
                'moe_hidden_dim': 999,
                'load_balancing_coef': 999
            }
        )
    )
    assert plan(env).finished

@pytest.mark.parametrize('key', ['num_experts', 'top_k', 'moe_hidden_dim', 'load_balancing_coef'])
@pytest.mark.parametrize('missing', [False, True])
def test_active_moe_constructor_guards_use_real_callback_state(remote_resume_fixture, key, missing):
    env = remote_resume_fixture
    cfg = env.cfg.override({'model.fusion': 'moe'})

    def change(saved):
        # Test copies change both contracts solely to reach the constructor guard. This is
        # guard evidence, not evidence that the fixture trained a MoE model.
        saved['stage3_supervision']['encoder']['fusion'] = 'moe'
        saved['hyper_parameters']['run_settings']['fusion'] = 'moe'
        saved['hyper_parameters'].update(constructor_settings(cfg))
        if missing:
            del saved['hyper_parameters'][key]
        else:
            saved['hyper_parameters'][key] = -1

    mutate_checkpoint(env, change)
    with pytest.raises(ValueError, match='constructor'):
        resume_plan(env.root, cfg, env.inputs, env.remote_directory)

def test_cuda_preflight_uses_selected_accelerator_on_mac(remote_resume_fixture, monkeypatch):
    env = remote_resume_fixture
    cfg = env.cfg.override(
        {
            'training.trainer.accelerator': 'auto',
            'training.trainer.precision': 'bf16-mixed'
        }
    )

    def change(saved):
        saved['hyper_parameters']['run_settings'].update(
            {
                'accelerator': 'cuda',
                'precision': 'bf16-mixed'
            }
        )

    mutate_checkpoint(env, change)
    from naics_embedder.utils import training
    monkeypatch.setattr(training, 'detect_hardware', lambda *args: pytest.fail('host detection'))
    assert resume_plan(env.root, cfg, env.inputs, env.remote_directory).finished

@pytest.mark.parametrize('filename', ['last.ckpt', 'epoch_summary.jsonl', 'monitor_reads.jsonl'])
def test_continuation_links_are_refused(remote_resume_fixture, filename):
    env = remote_resume_fixture
    path = env.directory / filename
    target = path.with_name(filename + '.original')
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match='links'):
        plan(env)

def test_unstable_run_artifact_refused(remote_resume_fixture):
    env = remote_resume_fixture
    (env.directory / 'epoch_summary.jsonl.tmp').write_text('{}')
    with pytest.raises(ValueError, match='unstable'):
        plan(env)

@pytest.mark.parametrize('corruption', ['bytes', 'member_escape', 'member_link'])
def test_invalid_bundle_is_refused(remote_repo, corruption):
    manifest = remote_repo.manifest
    raw = json.loads(manifest.read_text())
    artifact = next(iter(raw['artifacts'].values()))
    member = artifact['files'][0]
    path = manifest.parent / member['path']
    if corruption == 'bytes':
        path.write_bytes(path.read_bytes() + b'changed')
    elif corruption == 'member_escape':
        member['path'] = '../outside.parquet'
        manifest.write_text(json.dumps(raw))
    else:
        target = path.with_name(path.name + '.original')
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError):
        canonical_inputs(remote_repo.root, remote_repo.config)

def test_unfinished_predicate_does_not_extend_saved_budget(remote_resume_fixture):
    env = remote_resume_fixture
    saved = read_checkpoint(env.directory / 'last.ckpt')
    saved['epoch'] = 1
    assert finished_run(saved, env.cfg.training.early_stopping_patience, 3) == (False, None)

@pytest.mark.parametrize(
    'field,value', [
        ('stopped_epoch', None), ('stopped_epoch', True), ('stopped_epoch', -1),
        ('stopped_epoch', 999), ('wait_count', None), ('wait_count', -1),
        ('best_score', float('nan'))
    ]
)
def test_corrupt_completion_state_refuses(remote_resume_fixture, field, value):
    from naics_embedder.utils.training import outcome_early_stopping
    env = remote_resume_fixture
    saved = read_checkpoint(env.directory / 'last.ckpt')
    key = outcome_early_stopping(env.cfg.training.early_stopping_patience).state_key
    saved['callbacks'][key][field] = value
    with pytest.raises(ValueError):
        finished_run(saved, env.cfg.training.early_stopping_patience, 3)

def test_retired_contract_fields_are_absent_and_unrequired(remote_resume_fixture):
    env = remote_resume_fixture
    saved = read_checkpoint(env.directory / 'last.ckpt')
    assert not {
        'supervision_mode', 'structural_preference_loss_version', 'mining_contract_version'
    } & saved['stage3_supervision'].keys()
    assert plan(env).finished

@pytest.mark.parametrize('filename', ['monitor_reads.jsonl', 'epoch_summary.jsonl'])
@pytest.mark.parametrize('corruption', ['json', 'epoch', 'gap', 'duplicate'])
def test_malformed_histories_refuse(remote_resume_fixture, filename, corruption):
    env = remote_resume_fixture
    path = env.directory / filename
    if corruption == 'json':
        path.write_text('{invalid')
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == 'epoch':
            detail = rows[0]['read']['detail'] if filename.startswith('monitor') else rows[0]
            detail['epoch'] = True
        elif corruption == 'gap':
            rows.pop(1)
        else:
            rows.append(rows[-1])
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        plan(env)

def test_checkpoint_changing_during_inventory_refuses(remote_resume_fixture, monkeypatch):
    from naics_embedder.remote import canonical
    env = remote_resume_fixture
    original = canonical.sha256_file

    def changed(path):
        digest = original(path)
        if path.name == 'last.ckpt':
            path.write_bytes(path.read_bytes() + b'x')
        return digest

    monkeypatch.setattr(canonical, 'sha256_file', changed)
    with pytest.raises(ValueError, match='unstable'):
        plan(env)

def test_missing_saved_seed_refuses(remote_resume_fixture):
    env = remote_resume_fixture
    mutate_checkpoint(env, lambda saved: saved['hyper_parameters'].pop('seed'))
    with pytest.raises(ValueError, match='seed'):
        plan(env)

def test_missing_saved_contract_refuses(remote_resume_fixture):
    env = remote_resume_fixture
    mutate_checkpoint(env, lambda saved: saved.pop('stage3_supervision'))
    with pytest.raises(ValueError, match='legacy'):
        plan(env)

@pytest.mark.parametrize('value', [True, 1.0])
def test_monitor_seed_requires_integer(remote_resume_fixture, value):
    env = remote_resume_fixture
    path = env.directory / 'monitor_reads.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['read']['detail']['seed'] = value
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='seed'):
        plan(env)

def test_malformed_kept_checkpoint_refused(remote_resume_fixture):
    env = remote_resume_fixture
    (env.directory / 'epoch=999.ckpt').write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='malformed checkpoint'):
        plan(env)

def test_new_continuation_file_during_validation_refuses(remote_resume_fixture, monkeypatch):
    from naics_embedder.remote import canonical
    env = remote_resume_fixture
    original = canonical.read_checkpoint

    def changed(path):
        saved = original(path)
        (env.directory / 'new.tmp').write_text('interrupted write')
        return saved

    monkeypatch.setattr(canonical, 'read_checkpoint', changed)
    with pytest.raises(ValueError, match='changed|unstable'):
        plan(env)

def test_member_changed_after_bundle_validation_refuses(remote_repo, monkeypatch):
    from naics_embedder.remote import canonical
    original = canonical.load_validated_bundle

    def changed(path):
        bundle = original(path)
        member = bundle.member_paths(next(iter(bundle.manifest.artifacts)))[0]
        member.write_bytes(member.read_bytes() + b'changed after validation')
        return bundle

    monkeypatch.setattr(canonical, 'load_validated_bundle', changed)
    with pytest.raises(ValueError, match='changed|hash'):
        canonical_inputs(remote_repo.root, remote_repo.config)
