'''Launch gates and provenance using isolated canonical inputs and injected processes.'''

import json
import shlex
import shutil

import pytest

from naics_embedder.remote.launch import training_argv
from naics_embedder.remote.session import read_state
from naics_embedder.supervision.artifacts import sha256_file

def fresh(env):
    shutil.rmtree(env.directory)

def unfinished(env):
    env.cfg = env.set_unfinished()

def launch(env, resume=False):
    return env.workflow.train(resume, 'conf/config.yaml', env.overrides)

def test_finished_resume_never_uploads_or_launches(remote_launch_fixture):
    env = remote_launch_fixture
    env.set_finished_budget()
    result = launch(env, True)
    assert result.skipped and result.segment_id is None
    assert not any(
        c[0] in {'push', 'launch'} or c[:2] == ('probe', 'write_record')
        for c in env.transport.calls
    )
    assert not any(c[:2] == ('probe', 'gpu') for c in env.transport.calls)
    assert env.loops == []

def test_resume_command_can_only_name_last(remote_launch_fixture):
    env = remote_launch_fixture
    argv = training_argv(env.cfg, 'conf/config.yaml', [], env.inputs, env.info, True)
    assert argv[:5] == ('/usr/bin/uv', 'run', '--locked', 'naics-embedder', 'train')
    assert argv[argv.index('--ckpt-path') + 1] == 'last'
    assert argv[argv.index('--checkpoint-load-mode') + 1] == 'exact'
    assert 'training.trainer.devices=1' in argv

@pytest.mark.parametrize(
    'override', [
        '--ckpt-path=last', '--checkpoint-load-mode=exact', '--ckpt-path',
        'dirs.checkpoint_dir=elsewhere', 'supervision.manifest_path=data/other.json'
    ]
)
def test_reserved_overrides_refuse(remote_launch_fixture, override):
    env = remote_launch_fixture
    with pytest.raises(ValueError):
        training_argv(env.cfg, 'conf/config.yaml', [override], env.inputs, env.info, False)

@pytest.mark.parametrize('side', ['local', 'remote'])
def test_fresh_nonempty_directory_refuses(remote_launch_fixture, side):
    env = remote_launch_fixture
    if side == 'remote':
        shutil.copytree(env.directory, env.instance / 'checkpoints' / env.cfg.experiment_name)
        fresh(env)
    with pytest.raises(ValueError, match='not empty|sync first'):
        launch(env)
    assert not any(c[0] == 'launch' for c in env.transport.calls)

@pytest.mark.parametrize('absolute', [False, True])
def test_fresh_launch_records_command_and_stable_identity(remote_launch_fixture, absolute):
    env = remote_launch_fixture
    fresh(env)
    if absolute:
        env.overrides = ['dirs.checkpoint_dir=' + env.info.checkpoint_base]
    result = launch(env)
    assert not result.skipped
    record_dir = env.instance / '.remote/segments' / result.segment_id
    record = json.loads((record_dir / 'segment.json').read_text())
    script = (record_dir / 'launch.sh').read_text()
    assert record['remote_directory'] == env.remote_directory
    assert record['effective_config']['dirs']['checkpoint_dir'] == env.info.checkpoint_base
    assert '< /dev/null' in script and 'status=$?' in script and '.tmp' in script
    assert 'unset CUDA_VISIBLE_DEVICES' in script
    assert record['command'] == shlex.join(record['argv'])
    assert record['gpu_evidence']['logical_index'] == 0
    assert record['input_hashes'] == env.inputs.hashes
    assert record['lock_sha256'] == sha256_file(env.root / 'uv.lock')
    assert (record_dir / 'code/provenance.json').exists()
    run = json.loads((env.root / '.remote/runs' / (env.cfg.experiment_name + '.json')).read_text())
    assert run['training_run'] is None and len(run['settings']) == 21
    assert read_state(env.root).active_segment_id == result.segment_id
    assert env.loops and not (env.instance / '.remote/launch.lock').exists()

@pytest.mark.parametrize('absolute', [False, True])
def test_resume_restores_every_byte_and_histories(remote_launch_fixture, absolute):
    env = remote_launch_fixture
    unfinished(env)
    if absolute:
        env.overrides.append('dirs.checkpoint_dir=' + env.info.checkpoint_base)
    before = {
        p.relative_to(env.root).as_posix(): sha256_file(p)
        for p in env.directory.rglob('*') if p.is_file()
    }
    result = launch(env, True)
    assert not result.skipped
    assert all(sha256_file(env.instance / name) == digest for name, digest in before.items())
    record = json.loads(
        (env.instance / '.remote/segments' / result.segment_id / 'segment.json').read_text()
    )
    assert record['continuation_hashes'] == before
    assert record['resumed_from']['sha256'] == before[env.directory.relative_to(env.root).as_posix()
                                                      + '/last.ckpt']

@pytest.mark.parametrize('name', ['last.ckpt', 'monitor_reads.jsonl', 'epoch_summary.jsonl'])
def test_missing_resume_member_never_launches(remote_launch_fixture, name):
    env = remote_launch_fixture
    unfinished(env)
    (env.directory / name).unlink()
    with pytest.raises(ValueError):
        launch(env, True)
    assert not any(c[0] in {'launch', 'push'} for c in env.transport.calls)

@pytest.mark.parametrize('name', ['last.ckpt', 'monitor_reads.jsonl', 'epoch_summary.jsonl'])
def test_stopped_remote_different_generation_needs_sync_without_overwrites(
    remote_launch_fixture, name
):
    env = remote_launch_fixture
    unfinished(env)
    target = env.instance / 'checkpoints' / env.cfg.experiment_name
    shutil.copytree(env.directory, target)
    (target / name).write_bytes((target / name).read_bytes() + b'newer')
    before = (target / name).read_bytes()
    with pytest.raises(ValueError, match='sync first'):
        launch(env, True)
    assert (target / name).read_bytes() == before
    assert not any(c[0] in {'push', 'launch'} for c in env.transport.calls)

@pytest.mark.parametrize(
    'failure', [
        'pending_pull', 'pending_push', 'running', 'clock', 'gpu', 'gpu_exception', 'code',
        'remote_code', 'input', 'marker'
    ]
)
def test_launch_gate_failures(remote_launch_fixture, failure):
    env = remote_launch_fixture
    fresh(env)
    if failure == 'pending_pull':
        path = env.root / '.remote/pulls/pending.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    elif failure == 'pending_push':
        from naics_embedder.remote.session import write_state
        env.state.pending_push_id = 'pending'
        write_state(env.root, env.state)
    elif failure == 'running':
        env.transport.running = True
    elif failure == 'clock':
        env.ntp[0] = False
    elif failure == 'gpu':
        env.gpu[0]['native_bf16'] = False
    elif failure == 'gpu_exception':
        env.gpu[0] = RuntimeError('probe failed')
    elif failure == 'code':
        (env.root / 'src/tiny.py').write_text('changed')
    elif failure == 'remote_code':
        (env.instance / 'src/tiny.py').write_text('changed')
    elif failure == 'input':
        env.transport.canonical_error = True
    else:
        (env.instance / '.remote/pushes/session.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError)):
        launch(env)
    assert not any(c[0] == 'launch' for c in env.transport.calls)
    assert env.loops == []

@pytest.mark.parametrize(
    'key,value', [
        ('dirs.output_dir', '/outside'), ('dirs.log_dir', 'outside'),
        ('data.outcome_panel.selection_log', 'outputs/log.jsonl'),
        ('data_loader.tokenization.output_path', 'outputs/cache.pt'),
        ('training.trainer.devices', '2')
    ]
)
def test_unmapped_outputs_and_multiple_devices_refuse(remote_launch_fixture, key, value):
    env = remote_launch_fixture
    fresh(env)
    env.overrides = [key + '=' + value]
    with pytest.raises(ValueError):
        launch(env)
    assert not any(c[0] == 'launch' for c in env.transport.calls)

def test_upload_failure_leaves_failed_named_segment_and_releases_lock(remote_launch_fixture):
    env = remote_launch_fixture
    unfinished(env)
    env.transport.push_error = True
    with pytest.raises(RuntimeError):
        launch(env, True)
    assert not (env.instance / '.remote/launch.lock').exists()
    assert not any(c[0] == 'launch' for c in env.transport.calls)

def test_argv_shell_metacharacters_remain_literal(remote_launch_fixture):
    env = remote_launch_fixture
    argv = training_argv(
        env.cfg, 'conf/config.yaml', ['experiment_name=literal$(touch nope);x'], env.inputs,
        env.info, False
    )
    assert 'experiment_name=literal$(touch nope);x' in shlex.split(shlex.join(argv))

@pytest.mark.parametrize('finished', [True, False])
def test_verified_pull_binds_training_run_before_session_replacement(
    remote_launch_fixture, finished
):
    from datetime import timedelta

    from naics_embedder.remote.sync import sync_once
    from tests.fixtures.remote import SyncTransport

    env = remote_launch_fixture
    if not finished:
        unfinished(env)
    backup = env.root / '.remote/fixture-run'
    shutil.copytree(env.directory, backup)
    fresh(env)
    launch(env)
    shutil.copytree(backup, env.instance / 'checkpoints' / env.cfg.experiment_name)
    state = read_state(env.root)
    sync_once(
        env.root, state, SyncTransport(root=env.instance), env.workflow.remote_cfg, final=True
    )
    run_path = env.root / '.remote/runs' / (env.cfg.experiment_name + '.json')
    record = json.loads(run_path.read_text())
    from naics_embedder.utils.training import read_checkpoint
    assert record['training_run'] == read_checkpoint(env.directory / 'last.ckpt')['training_run']
    origin = record['session_id']
    env.now[0] += timedelta(seconds=2)
    state.status = 'finished'
    from naics_embedder.remote.session import write_state
    write_state(env.root, state)
    shutil.rmtree(env.instance / 'checkpoints' / env.cfg.experiment_name)
    replacement = env.workflow.up('replacement', 'conf/config.yaml', [])
    assert replacement.session_id != origin and replacement.last_sync_manifest is None
    assert json.loads(run_path.read_text())['training_run'] == record['training_run']
    snapshot = json.loads((env.root / '.remote/session-inputs.json').read_text())
    assert snapshot['session_id'] == replacement.session_id
    result = launch(env, True)
    assert result.skipped == finished
    if not finished:
        assert (env.instance / 'checkpoints' / env.cfg.experiment_name
                / 'last.ckpt').read_bytes() == (env.directory / 'last.ckpt').read_bytes()

@pytest.mark.parametrize('failure', ['no_evidence', 'failed', 'tampered', 'foreign', 'pending'])
def test_unverified_checkpoint_cannot_bind_run(remote_launch_fixture, failure):
    from naics_embedder.remote.sync import bind_verified_runs

    env = remote_launch_fixture
    backup = env.root / '.remote/fixture-run'
    shutil.copytree(env.directory, backup)
    fresh(env)
    launch(env)
    shutil.copytree(backup, env.directory)
    state = read_state(env.root)
    if failure != 'no_evidence':
        path = env.root / '.remote/pulls/pass/manifest.json'
        path.parent.mkdir(parents=True)
        name = (env.directory / 'last.ckpt').relative_to(env.root).as_posix()
        path.write_text(
            json.dumps(
                {
                    'session_id': state.session_id if failure != 'foreign' else 'other',
                    'files': {
                        name: sha256_file(env.directory / 'last.ckpt')
                    }
                }
            )
        )
        if failure != 'failed':
            state.last_sync_manifest = '.remote/pulls/pass/manifest.json'
        if failure == 'tampered':
            with (env.directory / 'last.ckpt').open('ab') as stream:
                stream.write(b'tampered')
        if failure == 'pending':
            (env.root / '.remote/pulls/pending.json').write_text('{}')
    if failure in {'no_evidence', 'failed'}:
        bind_verified_runs(env.root, state)
    else:
        with pytest.raises(ValueError):
            bind_verified_runs(env.root, state)
    run = json.loads((env.root / '.remote/runs' / (env.cfg.experiment_name + '.json')).read_text())
    assert run['training_run'] is None

@pytest.mark.parametrize('failure', ['gpu_exception', 'gpu_native_false', 'wrapper_visibility'])
def test_worker_rechecks_qualification_and_wrapper_before_tmux(tmp_path, monkeypatch, failure):
    from dataclasses import asdict

    import naics_embedder.remote.worker as worker
    from naics_embedder.remote.launch import _wrapper
    from naics_embedder.remote.session import GpuEvidence, RemoteInfo
    from naics_embedder.remote.worker import run_probe

    root = tmp_path / 'instance'
    directory = root / '.remote/segments/segment'
    directory.mkdir(parents=True)
    info = RemoteInfo(
        str(root), str(root / 'checkpoints'), '/uv', '/python', True, 'cuda', 'fixture'
    )
    argv = ('/uv', 'run', '--locked', 'naics-embedder', 'train')
    evidence = GpuEvidence(0, 'fixture', (8, 0), 1, True, '2')
    (directory / 'segment.json').write_text(
        json.dumps(
            {
                'argv': list(argv),
                'cuda_visible_devices': '2',
                'gpu_evidence': asdict(evidence)
            }
        )
    )
    script = _wrapper(info, argv, str(directory / 'exit_code'), '2')
    if failure == 'wrapper_visibility':
        script = script.replace('CUDA_VISIBLE_DEVICES=2', 'CUDA_VISIBLE_DEVICES=3')
    (directory / 'launch.sh').write_text(script)
    (root / '.remote/launch.lock').write_text('segment')
    calls = []
    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})
    monkeypatch.setattr(worker, '_training_status', lambda payload: {'running': False})

    def gpu():
        import os
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
        if failure == 'gpu_exception':
            raise RuntimeError('native BF16 unavailable after bootstrap')
        return GpuEvidence(0, 'fixture', (8, 0), 1, failure != 'gpu_native_false', '2')

    monkeypatch.setattr(worker, 'gpu_evidence', gpu)
    monkeypatch.setattr(
        worker.subprocess, 'run', lambda *args, **kwargs: calls.append((args, kwargs))
    )
    with pytest.raises((ValueError, RuntimeError)):
        run_probe(
            'training', {
                'action': 'launch',
                'segment_id': 'segment',
                'script': str(directory / 'launch.sh')
            }, root
        )
    assert calls == []

def test_fresh_directory_symlink_refuses(remote_launch_fixture):
    env = remote_launch_fixture
    fresh(env)
    target = env.root / '.remote/empty'
    target.mkdir()
    env.directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='link'):
        launch(env)
    assert not any(c[0] == 'launch' for c in env.transport.calls)

def test_instance_launch_lock_collision_refuses(remote_launch_fixture):
    env = remote_launch_fixture
    fresh(env)
    (env.instance / '.remote/launch.lock').write_text('other-segment')
    with pytest.raises(FileExistsError):
        launch(env)
    assert (env.instance / '.remote/launch.lock').read_text() == 'other-segment'
    assert not any(c[0] == 'launch' for c in env.transport.calls)

def test_real_newer_finished_remote_checkpoint_requires_sync_first(remote_launch_fixture):
    env = remote_launch_fixture
    target = env.instance / 'checkpoints' / env.cfg.experiment_name
    shutil.copytree(env.directory, target)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    unfinished(env)
    with pytest.raises(ValueError, match='sync first'):
        launch(env, True)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}
    assert not any(c[0] in {'push', 'launch'} for c in env.transport.calls)

@pytest.mark.parametrize('change', ['seed', 'settings', 'lora', 'path'])
def test_wrong_resume_identity_precedes_gpu_and_upload(remote_launch_fixture, change):
    import torch

    from naics_embedder.utils.training import read_checkpoint
    env = remote_launch_fixture
    unfinished(env)
    path = env.directory / 'last.ckpt'
    saved = read_checkpoint(path)
    if change == 'seed':
        saved['hyper_parameters']['seed'] = 77
    elif change == 'settings':
        saved['hyper_parameters']['run_settings']['max_epochs'] = 99
    elif change == 'lora':
        saved['hyper_parameters']['lora_r'] = 99
    else:
        for callback in saved['callbacks'].values():
            if isinstance(callback, dict) and 'dirpath' in callback:
                callback['dirpath'] = '/home/other/checkpoints/run'
    torch.save(saved, path)
    with pytest.raises(ValueError):
        launch(env, True)
    assert not any(
        c[0] in {'push', 'launch'} or c[:2] == ('probe', 'gpu') for c in env.transport.calls
    )

def test_native_cuda_argv_keeps_campaign_precision_and_one_device(remote_launch_fixture):
    from dataclasses import replace
    env = remote_launch_fixture
    cfg = env.cfg.override(
        {
            'training.trainer.accelerator': 'cuda',
            'training.trainer.precision': 'bf16-mixed'
        }
    )
    info = replace(env.info, accelerator='cuda')
    argv = training_argv(cfg, 'conf/config.yaml', [], env.inputs, info, False)
    assert 'training.trainer.accelerator=cuda' in argv
    assert 'training.trainer.precision=bf16-mixed' in argv
    assert 'training.trainer.devices=1' in argv
    with pytest.raises(ValueError, match='precision fallback'):
        training_argv(env.cfg, 'conf/config.yaml', [], env.inputs, info, False)

def test_immutable_remote_segment_cannot_be_replaced(tmp_path):
    from naics_embedder.remote.worker import run_probe
    root = tmp_path / 'instance'
    root.mkdir()
    payload = {
        'path': '.remote/segments/segment/segment.json',
        'record': {
            'first': True
        },
        'immutable': True
    }
    run_probe('write_record', payload, root)
    with pytest.raises(FileExistsError):
        run_probe('write_record', dict(payload, record={'first': False}), root)
    assert json.loads((root / payload['path']).read_text()) == {'first': True}

def test_worker_launch_uses_devnull_explicit_cwd_and_recorded_visibility(tmp_path, monkeypatch):
    import os
    import subprocess
    from dataclasses import asdict

    import naics_embedder.remote.worker as worker
    from naics_embedder.remote.launch import _wrapper
    from naics_embedder.remote.session import GpuEvidence, RemoteInfo

    root = tmp_path / 'instance'
    directory = root / '.remote/segments/segment'
    directory.mkdir(parents=True)
    info = RemoteInfo(
        str(root), str(root / 'checkpoints'), '/uv', '/python', True, 'cuda', 'fixture'
    )
    argv = ('/uv', 'run', '--locked', 'naics-embedder', 'train')
    evidence = GpuEvidence(0, 'fixture', (8, 0), 1, True, '2')
    (directory / 'segment.json').write_text(
        json.dumps(
            {
                'argv': list(argv),
                'cuda_visible_devices': '2',
                'gpu_evidence': asdict(evidence)
            }
        )
    )
    (directory / 'launch.sh').write_text(_wrapper(info, argv, str(directory / 'exit_code'), '2'))
    (root / '.remote/launch.lock').write_text('segment')
    calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'original')
    monkeypatch.setattr(worker, '_clock', lambda: {'ntp': True})
    monkeypatch.setattr(worker, '_training_status', lambda payload: {'running': False})

    def gpu():
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
        return evidence

    monkeypatch.setattr(worker, 'gpu_evidence', gpu)
    monkeypatch.setattr(
        worker.subprocess, 'run', lambda *args, **kwargs: calls.append((args, kwargs))
    )
    worker.run_probe(
        'training', {
            'action': 'launch',
            'segment_id': 'segment',
            'script': str(directory / 'launch.sh')
        }, root
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][:6] == [
        'tmux', 'new-session', '-d', '-s', 'naics-train', 'bash ' + shlex.quote(
            str(directory / 'launch.sh')
        ) + ' < /dev/null'
    ]
    assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['cwd'] == root
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'original'

def test_remote_output_symlink_refuses_before_gpu(remote_launch_fixture):
    env = remote_launch_fixture
    fresh(env)
    target = env.instance / '.remote/unpulled'
    target.mkdir()
    (env.instance / 'outputs').symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='link'):
        launch(env)
    assert not any(c[0] == 'launch' or c[:2] == ('probe', 'gpu') for c in env.transport.calls)

def test_matching_but_unpushed_canonical_edits_require_up(remote_launch_fixture):
    env = remote_launch_fixture
    fresh(env)
    name = env.inputs.manifest.parent.relative_to(env.root) / 'extra.txt'
    (env.root / name).write_text('changed canonical inventory')
    (env.instance / name).write_text('changed canonical inventory')
    with pytest.raises(ValueError, match='canonical.*changed|canonical.*up'):
        launch(env)
    assert not any(c[0] == 'launch' or c[:2] == ('probe', 'gpu') for c in env.transport.calls)

@pytest.mark.parametrize('failure', ['missing', 'session', 'push', 'hashes'])
def test_canonical_ready_snapshot_fails_closed(remote_launch_fixture, failure):
    env = remote_launch_fixture
    fresh(env)
    path = env.root / '.remote/session-inputs.json'
    if failure == 'missing':
        path.unlink(missing_ok=True)
    else:
        path.write_text(
            json.dumps(
                {
                    'session_id': 'foreign' if failure == 'session' else env.state.session_id,
                    'push_id': 'foreign' if failure == 'push' else env.state.push_id,
                    'hashes': {} if failure == 'hashes' else env.inputs.hashes,
                    'paths': list(env.inputs.paths)
                }
            )
        )
    with pytest.raises((ValueError, FileNotFoundError), match='canonical|session-inputs'):
        launch(env)
    assert not any(c[0] == 'launch' or c[:2] == ('probe', 'gpu') for c in env.transport.calls)

def test_up_revalidates_legitimate_changed_input_inventory(remote_launch_fixture):
    from datetime import timedelta
    env = remote_launch_fixture
    fresh(env)
    path = env.inputs.manifest.parent / 'extra.txt'
    path.write_text('new canonical inventory')
    env.now[0] += timedelta(seconds=2)
    env.workflow.up('fixture', 'conf/config.yaml', [])
    snapshot = json.loads((env.root / '.remote/session-inputs.json').read_text())
    assert path.relative_to(env.root).as_posix() in snapshot['hashes']
    assert not launch(env).skipped

def test_bound_run_identity_cannot_resume_another_training_run(remote_launch_fixture):
    env = remote_launch_fixture
    backup = env.root / '.remote/fixture-run'
    shutil.copytree(env.directory, backup)
    fresh(env)
    launch(env)
    shutil.copytree(backup, env.directory)
    unfinished(env)
    path = env.root / '.remote/runs' / (env.cfg.experiment_name + '.json')
    record = json.loads(path.read_text())
    record['training_run'] = 'different-known-run'
    path.write_text(json.dumps(record))
    from datetime import timedelta
    env.now[0] += timedelta(seconds=2)
    env.transport.calls.clear()
    with pytest.raises(ValueError, match='training_run'):
        launch(env, True)
    assert not any(
        c[0] in {'push', 'launch'} or c[:2] == ('probe', 'gpu') for c in env.transport.calls
    )

@pytest.mark.parametrize('change', ['code', 'inputs', 'fresh_collision'])
def test_provenance_upload_does_not_hide_late_remote_changes(
    remote_launch_fixture, monkeypatch, change
):
    env = remote_launch_fixture
    fresh(env)
    original_push = env.transport.push

    def push(source, destination, files):
        original_push(source, destination, files)
        if source.parent.name == 'segments':
            if change == 'code':
                (env.instance / 'src/tiny.py').write_text('late change')
            elif change == 'inputs':
                name = env.inputs.manifest.parent.relative_to(env.root) / 'late.txt'
                (env.instance / name).write_text('late input')
            else:
                run = env.instance / 'checkpoints' / env.cfg.experiment_name
                run.mkdir(parents=True)
                (run / 'late.txt').write_text('late collision')

    monkeypatch.setattr(env.transport, 'push', push)
    with pytest.raises(ValueError):
        launch(env)
    assert not any(c[0] == 'launch' for c in env.transport.calls)
