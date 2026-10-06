'''GNU-rsync filesystem qualification of the production workflow; processes are injected.'''

import json
import subprocess

import pytest

from naics_embedder.remote.session import read_state
from naics_embedder.remote.transport import LocalTransport
from naics_embedder.supervision.artifacts import sha256_file

def up(env, host='a', force=False):
    return env.workflow.up(host, 'conf/config.yaml', [], force)

def train(env):
    return env.workflow.train(True, 'conf/config.yaml', [])

def hashes(directory):
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in directory.rglob('*') if p.is_file()
    }

def test_push_matches_git_and_deletes_only_previously_pushed_code(local_workflow):
    env = local_workflow
    up(env)
    assert (env.transport.root / 'src/tiny.py').read_bytes() == (env.root
                                                                 / 'src/tiny.py').read_bytes()
    (env.transport.root / 'outputs/unowned').parent.mkdir(exist_ok=True)
    (env.transport.root / 'outputs/unowned').write_text('preserve')
    subprocess.run(['git', '-C', str(env.root), 'rm', 'src/tiny.py'], check=True)
    up(env)
    assert not (env.transport.root / 'src/tiny.py').exists()
    assert (env.transport.root / 'outputs/unowned').read_text() == 'preserve'
    assert (env.transport.root / 'AGENTS.md').is_symlink()

def test_instance_edits_are_refused_and_rescued_without_working_tree_changes(local_workflow):
    env = local_workflow
    state = up(env)
    before = (env.root / 'src/tiny.py').read_bytes()
    (env.transport.root / 'src/tiny.py').write_text('instance edit\n')
    with pytest.raises(ValueError, match='edit'):
        env.workflow.finish()
    result = env.workflow.finish(pull_edits=True)
    assert result.safe
    rescued = env.root / '.remote/instance-edits' / state.session_id
    assert any(p.read_bytes() == b'instance edit\n' for p in rescued.rglob('*') if p.is_file())
    assert (env.root / 'src/tiny.py').read_bytes() == before
    with pytest.raises(ValueError, match='edit|changed'):
        up(env)

def test_all_kept_checkpoints_and_both_histories_restore_on_instance_b(local_workflow):
    env = local_workflow
    up(env)
    before = hashes(env.directory)
    assert len([n for n in before if n.endswith('.ckpt')]) >= 2
    assert {'last.ckpt', 'monitor_reads.jsonl', 'epoch_summary.jsonl'} <= before.keys()
    train(env)
    env.workflow.sync(once=True)
    env.workflow.finish()
    physical = env.tmp_path / 'instance-b'
    physical.mkdir()
    for name in ['checkpoints', 'outputs', 'logs', '.remote/segments']:
        (physical / name).mkdir(parents=True, exist_ok=True)
    (physical / 'outputs/event').write_text('replacement output\n')
    (physical / 'logs/selection_log.jsonl').write_text('replacement remote log\n')
    replacement = LocalTransport(
        physical, env.process, env.gnu_rsync, logical_root=env.root, config=env.remote_cfg
    )
    replacement.repo = str(env.root)
    replacement.config = env.remote_cfg
    env.transports['b'] = replacement
    up(env, 'b')
    result = train(env)
    assert not result.skipped
    assert hashes(physical / 'checkpoints/qualification') == before
    segment = json.loads(
        (physical / '.remote/segments' / result.segment_id / 'segment.json').read_text()
    )
    assert segment['remote_directory'] == str(env.directory)
    assert segment['resumed_from']['sha256'] == before['last.ckpt']
    assert segment['continuation_hashes'] == env.plan.hashes
    assert hashes(env.directory) == before

def test_sessions_keep_logs_outputs_and_remote_selection_logs_separate(local_workflow):
    env = local_workflow
    state = up(env)
    for name in ['logs/selection_log.jsonl', 'outputs/event']:
        path = env.transport.root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('instance result\n')
    local_log = env.root / 'logs/selection_log.jsonl'
    local_log.write_text('Mac log\n')
    env.workflow.sync(once=True)
    assert local_log.read_text() == 'Mac log\n'
    assert (env.root / 'logs/remote' / state.session_id
            / 'selection_log.jsonl').read_text() == 'instance result\n'
    assert (env.root / 'outputs/remote' / state.session_id / 'event').exists()
    env.workflow.finish()
    next_state = up(env)
    assert next_state.session_id != state.session_id
    assert (env.root / 'logs/remote' / state.session_id / 'selection_log.jsonl').exists()

def test_background_defers_busy_run_and_finish_pulls_it(local_workflow):
    env = local_workflow
    up(env)
    train(env)
    env.remote_cfg.in_flight_seconds = 120
    # Persist the exact config consumed by public sync, without replacing other fields.
    state = read_state(env.root)
    (env.root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=state.session_id, remote_config=env.remote_cfg.model_dump()))
    )
    result = env.workflow.sync(once=True)
    assert result.pending >= len(env.plan.files)
    env.process.running = True
    result = env.workflow.finish(stop_training=True)
    assert result.safe
    assert any(c[0] == 'interrupt' for c in env.process.calls)
    assert hashes(env.directory) == hashes(env.transport.root / 'checkpoints/qualification')

@pytest.mark.parametrize('failure', ['partial', 'mutation'])
def test_partial_or_mutating_transfer_never_replaces_a_good_run(
    local_workflow, monkeypatch, failure
):
    env = local_workflow
    up(env)
    train(env)
    env.workflow.sync(once=True)
    before = hashes(env.directory)
    original = env.transport.pull

    def broken(source, destination, files):
        original(source, destination, files)
        if source.endswith('/checkpoints'):
            if failure == 'partial':
                raise OSError('injected transfer loss')
            (env.transport.root / 'checkpoints/qualification/epoch_summary.jsonl').write_text(
                'mutated\n'
            )

    monkeypatch.setattr(env.transport, 'pull', broken)
    with pytest.raises((OSError, ValueError)):
        env.workflow.sync(once=True)
    assert hashes(env.directory) == before

def test_clean_finish_has_zero_checksums_and_tamper_withholds_safe(local_workflow):
    env = local_workflow
    up(env)
    train(env)
    result = env.workflow.finish()
    assert result.safe and result.local_sha256 == result.remote_sha256
    assert env.workflow.status()['pending'] == 0
    (env.directory / 'last.ckpt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Mac'):
        env.workflow.finish()

@pytest.mark.parametrize('local_workflow', ['interrupted', 'finished'], indirect=True)
def test_resume_only_last_preserves_absolute_path_and_skips_finished(local_workflow, monkeypatch):
    env = local_workflow
    up(env)
    if env.plan.finished:

        def no_upload(*args, **kwargs):
            pytest.fail('finished run uploaded continuation')

        monkeypatch.setattr(env.transport, 'push', no_upload)
        env.process.calls.clear()
        assert train(env).skipped
        assert not env.loops
        assert not (env.root / '.remote/runs').exists()
        assert not any(c[0] == 'launch' or c[:2] == ('probe', 'gpu') for c in env.process.calls)
        return
    result = train(env)
    segment = json.loads(
        (env.root / '.remote/segments' / result.segment_id / 'segment.json').read_text()
    )
    argv = segment['argv']
    assert argv[argv.index('--ckpt-path') + 1] == 'last'
    assert argv[argv.index('--checkpoint-load-mode') + 1] == 'exact'
    assert '< /dev/null' in (env.root / '.remote/segments' / result.segment_id
                             / 'launch.sh').read_text()
    assert segment['remote_directory'] == str(env.directory)
    # Finished predicate is independently tested on actual completed Trainer fixtures.
    with pytest.raises(ValueError, match='settings|budget'):
        env.workflow.train(True, 'conf/config.yaml', ['training.trainer.max_epochs=2'])
    (env.transport.root / 'checkpoints/qualification/extra').write_text('newer')
    with pytest.raises(ValueError, match='sync first'):
        train(env)

@pytest.mark.parametrize('local_workflow', ['moe'], indirect=True)
@pytest.mark.parametrize(
    'control', [
        'model.lora.r', 'model.lora.alpha', 'model.lora.dropout', 'model.moe.num_experts',
        'model.moe.top_k', 'model.moe.hidden_dim', 'model.moe.load_balancing_coef'
    ]
)
def test_resume_preflight_honors_lora_and_active_moe(local_workflow, control):
    env = local_workflow
    up(env)
    value = 0.2 if control.endswith(('dropout', 'coef')) else 3
    if control.endswith('r'):
        value = 7
    with pytest.raises(ValueError, match='constructor|LoRA|lora'):
        env.workflow.train(True, 'conf/config.yaml', [f'{control}={value}'])
    assert not any(c[0] == 'launch' for c in env.process.calls)

def test_pending_promotion_recovery_blocks_launch_until_coherent(local_workflow, monkeypatch):
    import naics_embedder.remote.sync as sync
    env = local_workflow
    up(env)
    train(env)
    env.workflow.sync(once=True)
    (env.transport.root / 'outputs/tensor').write_text('new output')
    original = sync.os.replace

    def fail(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        if str(destination).endswith('tensor') and kwargs.get('src_dir_fd') is not None:
            raise OSError('promotion interrupted')
        return result

    monkeypatch.setattr(sync.os, 'replace', fail)
    with pytest.raises(OSError):
        env.workflow.sync(once=True)
    assert (env.root / '.remote/pulls/pending.json').exists()
    with pytest.raises(ValueError, match='pending pull'):
        train(env)
    monkeypatch.setattr(sync.os, 'replace', original)
    env.workflow.sync(once=True)
    assert not (env.root / '.remote/pulls/pending.json').exists()
    assert hashes(env.directory) == hashes(env.transport.root / 'checkpoints/qualification')
    assert not train(env).skipped

@pytest.mark.parametrize('kind', ['pushes', 'segments'])
@pytest.mark.parametrize('boundary', ['ssh', 'local'])
def test_production_ssh_first_metadata_upload_with_gnu(tmp_path, gnu_rsync, kind, boundary):
    import shlex
    import sys

    from naics_embedder.remote.transport import SshTransport

    source, repo = tmp_path / 'source', tmp_path / 'instance'
    source.mkdir()
    repo.mkdir()
    (source / 'record.json').write_text('first immutable record')
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'ssh':
            command = shlex.split(argv[-1])
            return subprocess.run(
                [sys.executable, '-I', *command[1:]],
                input=kwargs['input'],
                capture_output=True,
                timeout=kwargs['timeout']
            )
        if '--version' in argv:
            return subprocess.run(argv, capture_output=True, timeout=kwargs['timeout'])
        local = list(argv)
        index = local.index('-e')
        del local[index:index + 2]
        local[-1] = local[-1].split(':', 1)[1]
        return subprocess.run(
            local, input=kwargs['input'], capture_output=True, timeout=kwargs['timeout']
        )

    transport = (
        SshTransport('fixture', str(repo), gnu_rsync, runner)
        if boundary == 'ssh' else LocalTransport(repo, None, gnu_rsync)
    )
    destination = repo / '.remote' / kind / 'first'
    transport.push(source, str(destination), ('record.json', ))
    assert (destination / 'record.json').read_bytes() == (source / 'record.json').read_bytes()
    if boundary == 'ssh':
        assert any(argv[0] == 'ssh' for argv in calls)

def test_first_workflow_segment_upload_has_no_precreated_parent(local_workflow):
    import shutil

    env = local_workflow
    up(env)
    shutil.rmtree(env.transport.root / '.remote/segments', ignore_errors=True)
    result = train(env)
    assert not result.skipped
    assert (env.transport.root / '.remote/segments' / result.segment_id / 'segment.json').is_file()

def test_sparse_up_finish_qualifies_empty_owned_roots(local_workflow, monkeypatch):
    import shutil

    env = local_workflow
    for name in ('checkpoints', 'outputs', 'logs', '.remote/segments'):
        shutil.rmtree(env.transport.root / name, ignore_errors=True)
    up(env)
    assert not list((env.transport.root / '.remote/segments').glob('*/segment.json'))
    checks = []
    checksum = env.transport.checksum

    def qualified_checksum(mapping):
        checks.append(mapping.source)
        return checksum(mapping)

    monkeypatch.setattr(env.transport, 'checksum', qualified_checksum)
    result = env.workflow.finish()
    assert len(checks) == 4 and len(set(checks)) == 4
    assert result.safe and result.latest_checkpoint is None
    assert result.local_sha256 is None and result.remote_sha256 is None
    for name in ('checkpoints', 'outputs', 'logs', '.remote/segments'):
        assert (env.transport.root / name).is_dir()

@pytest.mark.parametrize('boundary', ['once', 'finish', 'resume'])
def test_missing_authoritative_kept_epoch_refuses_before_acceptance(local_workflow, boundary):
    import shutil

    env = local_workflow
    state = up(env)
    kept = next(env.directory.glob('epoch=*.ckpt'))
    if boundary == 'resume':
        kept.unlink()
        before = hashes(env.directory)
        with pytest.raises(ValueError, match='kept|selected'):
            train(env)
        assert hashes(env.directory) == before
        assert not list((env.transport.root / '.remote/segments').glob('*/segment.json'))
        assert not any(
            call[:2] == ('probe', 'gpu') or call[0] == 'launch' for call in env.process.calls
        )
        assert not (env.transport.root / 'checkpoints/qualification/last.ckpt').exists()
    else:
        train(env)
        (env.transport.root / 'checkpoints/qualification' / kept.name).unlink()
        shutil.rmtree(env.directory)
        before = hashes(env.transport.root / 'checkpoints/qualification')
        with pytest.raises(ValueError, match='kept|selected'):
            env.workflow.sync(once=True) if boundary == 'once' else env.workflow.finish()
        assert not (env.directory / 'last.ckpt').exists()
        assert hashes(env.transport.root / 'checkpoints/qualification') == before
        assert read_state(env.root).status == 'ready'
    assert state.last_sync_manifest is None
