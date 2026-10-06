'''Finish requires verified generations, stopped training and immutable edit evidence.'''
import json
import shutil

import pytest

from naics_embedder.remote.session import read_state

def test_tampering_is_detected_before_finish_repairs_it(remote_finish_fixture):
    env = remote_finish_fixture
    good = env.last.read_bytes()
    env.last.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='Mac copy changed'):
        env.workflow.finish()
    assert env.last.read_bytes() != good
    assert read_state(env.root).status != 'finished'
    assert not any(c[0] == 'pull' for c in env.transport.calls)

def test_finish_matches_actual_latest_run(remote_finish_fixture):
    env = remote_finish_fixture
    result = env.workflow.finish()
    assert result.safe and not result.abandoned
    assert result.latest_checkpoint == str(env.last)
    assert result.local_sha256 == result.remote_sha256
    assert len(result.local_sha256) == 64
    assert len(env.checks) == 4
    assert read_state(env.root).status == 'finished'
    assert len(env.stops) == 1

def test_running_refuses_without_interrupt(remote_finish_fixture):
    env = remote_finish_fixture
    env.running = True
    with pytest.raises(ValueError, match='running'):
        env.workflow.finish()
    assert not env.interrupted
    assert not env.checks

def test_stop_training_waits_then_syncs(remote_finish_fixture):
    env = remote_finish_fixture
    env.running = True
    result = env.workflow.finish(stop_training=True)
    assert result.safe
    assert env.interrupted == ['segment']

def test_stop_timeout_is_not_safe(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    env.running = True
    monkeypatch.setattr(env.transport, 'interrupt_training', env.interrupted.append)
    monkeypatch.setattr('naics_embedder.remote.workflow.TRAINING_STOP_TIMEOUT_SECONDS', 0)
    with pytest.raises(RuntimeError, match='timeout'):
        env.workflow.finish(stop_training=True)
    assert env.interrupted == ['segment']
    assert read_state(env.root).status == 'ready'
    assert not env.checks

@pytest.mark.parametrize('mapping_index', range(4))
def test_each_mapping_difference_refuses(remote_finish_fixture, mapping_index, monkeypatch):
    env = remote_finish_fixture

    def checksum(mapping):
        env.checks.append(mapping)
        return ('changed', ) if len(env.checks) == mapping_index + 1 else ()

    monkeypatch.setattr(env.transport, 'checksum', checksum)
    with pytest.raises(ValueError, match='checksum'):
        env.workflow.finish()
    assert read_state(env.root).status == 'ready'

@pytest.mark.parametrize('change', ['modified', 'deleted', 'new'])
def test_unresolved_edits_refuse(remote_finish_fixture, change):
    env = remote_finish_fixture
    path = env.instance / 'src/tiny.py'
    if change == 'deleted':
        path.unlink()
    elif change == 'new':
        (env.instance / 'src/new.py').write_text('NEW = 1\n')
    else:
        path.write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='instance code edits'):
        env.workflow.finish()
    assert read_state(env.root).status == 'ready'

def test_rescue_is_immutable_and_never_changes_working_tree(remote_finish_fixture):
    env = remote_finish_fixture
    source = env.instance / 'src/tiny.py'
    source.write_text('VALUE = 2\n')
    (env.instance / 'src/new.py').write_text('NEW = 1\n')
    assert env.workflow.finish(pull_edits=True).safe
    rescues = list((env.root / '.remote/instance-edits/session').iterdir())
    assert len(rescues) == 1
    rescue = rescues[0]
    assert (rescue / 'files/src/tiny.py').read_text() == 'VALUE = 2\n'
    assert not (env.root / 'src').exists()
    first = (rescue / 'manifest.json').read_bytes()
    source.write_text('VALUE = 3\n')
    with pytest.raises(ValueError, match='instance code edits'):
        env.workflow.finish()
    assert env.workflow.finish(pull_edits=True).safe
    assert (rescue / 'manifest.json').read_bytes() == first
    assert len(list(rescue.parent.iterdir())) == 2

def test_deleted_rescue_has_tombstone(remote_finish_fixture):
    env = remote_finish_fixture
    (env.instance / 'src/tiny.py').unlink()
    assert env.workflow.finish(pull_edits=True).safe
    manifest = next((env.root / '.remote/instance-edits/session').glob('*/manifest.json'))
    assert json.loads(manifest.read_text())['tombstones'] == ['src/tiny.py']

def test_unreachable_refuses_and_abandon_journals_loss(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture

    def offline(*args):
        raise OSError('unreachable')

    monkeypatch.setattr(env.transport, 'probe', offline)
    with pytest.raises(OSError):
        env.workflow.finish()
    assert read_state(env.root).status == 'ready'
    result = env.workflow.finish(abandon=True)
    assert not result.safe and result.abandoned
    assert read_state(env.root).status == 'abandoned'
    loss = json.loads((env.root / '.remote/abandon-session.json').read_text())
    assert loss['last_sync_utc'] and 'lost' in loss['risk']

@pytest.mark.parametrize('options', [dict(stop_training=True), dict(pull_edits=True)])
def test_abandon_options_refuse(remote_finish_fixture, options):
    with pytest.raises(ValueError, match='abandon'):
        remote_finish_fixture.workflow.finish(abandon=True, **options)
    assert not remote_finish_fixture.stops

def test_no_run_returns_explicit_null_checkpoint(remote_finish_fixture):
    env = remote_finish_fixture
    shutil.rmtree(env.instance / 'checkpoints')
    shutil.rmtree(env.root / 'checkpoints')
    shutil.rmtree(env.root / '.remote/runs')
    env.state.active_segment_id = None
    env.state.last_sync_manifest = None
    from naics_embedder.remote.session import write_state
    write_state(env.root, env.state)
    result = env.workflow.finish()
    assert result.safe
    assert result.latest_checkpoint is result.local_sha256 is result.remote_sha256 is None

def test_stopped_recheck_catches_restart(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    original = env.transport.checksum

    def checksum(mapping):
        value = original(mapping)
        env.running = True
        return value

    monkeypatch.setattr(env.transport, 'checksum', checksum)
    with pytest.raises(ValueError, match='running'):
        env.workflow.finish()
    assert read_state(env.root).status == 'ready'

def test_finish_includes_busy_recent_files(remote_finish_fixture):
    env = remote_finish_fixture
    path = env.instance / 'outputs/recent'
    path.write_text('recent bytes')
    assert env.workflow.finish().safe
    assert (env.root / 'outputs/remote/session/recent').read_text() == 'recent bytes'

def test_incomplete_remote_run_remains_pending(remote_finish_fixture):
    env = remote_finish_fixture
    (env.run / 'monitor_reads.jsonl').unlink()
    with pytest.raises(ValueError, match='pending'):
        env.workflow.finish()
    assert not env.checks

def test_owned_worker_timeout_prevents_finish_transaction(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture

    def timeout(*args):
        raise RuntimeError('owned worker stop timeout')

    monkeypatch.setattr('naics_embedder.remote.loop.stop_loop', timeout)
    with pytest.raises(RuntimeError, match='timeout'):
        env.workflow.finish()
    assert not env.transport.calls
    assert read_state(env.root).status == 'ready'

def test_recovery_precedes_ordinary_tamper_check(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    from naics_embedder.remote import sync
    (env.instance / 'outputs/tensor').write_text('updated tensor')
    original = sync._promote_file

    def crash(root, name, item):
        original(root, name, item)
        raise OSError('simulated crash after replacement')

    with monkeypatch.context() as patch:
        patch.setattr(sync, '_promote_file', crash)
        with pytest.raises(OSError, match='simulated crash'):
            sync.sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert (env.root / '.remote/pulls/pending.json').exists()
    assert env.workflow.finish().safe
    assert not (env.root / '.remote/pulls/pending.json').exists()

def test_finish_reconstructs_complete_session_config(remote_finish_fixture):
    env = remote_finish_fixture
    supplied = []

    def factory(host, cfg):
        supplied.append(cfg.model_dump())
        return env.transport

    env.workflow.transport_factory = factory
    assert env.workflow.finish().safe
    assert supplied == [env.cfg.model_dump()]
    assert env.transport.repo == env.state.remote_info.repo
    assert env.transport.python == env.state.remote_info.python

@pytest.mark.parametrize('change', ['snapshot', 'bytes'])
def test_failed_rescue_never_marks_handled(remote_finish_fixture, change):
    env = remote_finish_fixture
    (env.instance / 'src/tiny.py').write_text('VALUE = 2\n')

    def mutate(source):
        if source == env.instance:
            if change == 'snapshot':
                (source / 'src/tiny.py').write_text('VALUE = 3\n')
            else:
                rescue = next((env.root / '.remote/instance-edits/session').glob('*/files'))
                (rescue / 'src/tiny.py').write_text('bad copy')

    env.transport.after_pull = mutate
    with pytest.raises(ValueError, match='rescue|rescued'):
        env.workflow.finish(pull_edits=True)
    assert not list((env.root / '.remote/instance-edits/session').glob('*/handled.json'))
    assert read_state(env.root).status == 'ready'

def test_result_mutation_during_edit_rescue_is_not_safe(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    (env.instance / 'src/tiny.py').write_text('VALUE = 2\n')

    def mutate(source):
        if source == env.instance:
            env.transport.differences = ('result changed during rescue', )

    env.transport.after_pull = mutate
    with pytest.raises(ValueError, match='checksum'):
        env.workflow.finish(pull_edits=True)
    assert read_state(env.root).status == 'ready'

def test_foreign_session_marker_refuses(remote_finish_fixture):
    env = remote_finish_fixture
    (env.instance / '.remote/pushes/session.json').write_text(
        json.dumps(dict(session_id='foreign', push_id='push'))
    )
    with pytest.raises(ValueError, match='identity'):
        env.workflow.finish()
    assert not env.checks

def test_latest_resume_matches_stable_originating_run(remote_finish_fixture):
    env = remote_finish_fixture
    path = env.root / '.remote/runs/run.json'
    record = json.loads(path.read_text())
    record.update(session_id='origin-session', segment_id='origin-segment')
    path.write_text(json.dumps(record))
    result = env.workflow.finish()
    assert result.safe and result.latest_checkpoint == str(env.last)
    assert result.local_sha256 == result.remote_sha256
    assert json.loads(path.read_text())['session_id'] == 'origin-session'

def test_finish_abandoned_session_is_not_safe(remote_finish_fixture):
    env = remote_finish_fixture
    assert env.workflow.finish(abandon=True).abandoned
    with pytest.raises(ValueError, match='ready'):
        env.workflow.finish()

def test_stale_owned_pid_is_never_signaled_by_finish(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    from naics_embedder.remote import loop
    from naics_embedder.remote.session import state_lock
    argv = [
        'caffeinate', '-i', '/python', '-m', 'naics_embedder.remote.worker', 'sync-loop', '--root',
        str(env.root), '--session-id', 'session', '--token', 'token'
    ]
    record = dict(
        pid=123,
        session_id='session',
        token='token',
        argv=argv,
        identity=dict(start='old', command=' '.join(argv))
    )
    (env.root / '.remote/sync.pid').write_text(json.dumps(record))
    monkeypatch.setattr(loop, '_process_identity', lambda pid: dict(start='new', command='other'))
    monkeypatch.setattr(loop, '_worker_processes', lambda record: {})

    def forbidden(*args):
        raise AssertionError('reused PID must never receive a signal')

    monkeypatch.setattr(loop, '_signal', forbidden)

    def stop(root, session):
        with state_lock(root):
            loop._stop_loop_locked(root, session)

    monkeypatch.setattr(loop, 'stop_loop', stop)
    assert env.workflow.finish().safe
    assert not (env.root / '.remote/sync.pid').exists()

def test_latest_checkpoint_hash_change_at_final_probe_refuses(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    original = env.transport.probe

    def probe(operation, payload):
        result = original(operation, payload)
        if operation == 'inventory' and payload.get('path') == str(env.run):
            for item in result['files']:
                if item['path'] == 'last.ckpt':
                    item['sha256'] = '0' * 64
        return result

    monkeypatch.setattr(env.transport, 'probe', probe)
    with pytest.raises(ValueError, match='hashes differ'):
        env.workflow.finish()
    assert read_state(env.root).status == 'ready'

def test_rescued_symlink_preserves_link_without_source_writes(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture
    original = env.transport.pull
    (env.instance / 'src/link.py').symlink_to('tiny.py')

    def pull(source, destination, files):
        if source == str(env.instance):
            for name in files:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to((env.instance / name).readlink())
        else:
            original(source, destination, files)

    monkeypatch.setattr(env.transport, 'pull', pull)
    assert env.workflow.finish(pull_edits=True).safe
    rescue_directory = next((env.root / '.remote/instance-edits/session').iterdir())
    rescued = rescue_directory / 'files/src/link.py'
    assert rescued.is_symlink() and str(rescued.readlink()) == 'tiny.py'
    assert not (env.root / 'src').exists()

def test_worker_reappearing_after_shutdown_blocks_finish(remote_finish_fixture, monkeypatch):
    env = remote_finish_fixture

    def restarted(root, session):
        (root / '.remote/sync.pid').write_text(json.dumps(dict(session_id=session, pid=123)))

    monkeypatch.setattr('naics_embedder.remote.loop.stop_loop', restarted)
    with pytest.raises(ValueError, match='worker'):
        env.workflow.finish()
    assert not env.transport.calls
    assert read_state(env.root).status == 'ready'

def test_rescue_destination_symlink_never_writes_source_tree(remote_finish_fixture):
    env = remote_finish_fixture
    (env.instance / 'src/tiny.py').write_text('VALUE = 2\n')
    source = env.root / 'src'
    source.mkdir()
    (source / 'original.py').write_text('ORIGINAL = 1\n')
    (env.root / '.remote/instance-edits').symlink_to(source, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        env.workflow.finish(pull_edits=True)
    assert list(source.iterdir()) == [source / 'original.py']
    assert read_state(env.root).status == 'ready'

def test_reachable_training_stop_timeout_does_not_claim_unreachable(
    remote_finish_fixture, monkeypatch
):
    env = remote_finish_fixture
    env.running = True
    monkeypatch.setattr(env.transport, 'interrupt_training', env.interrupted.append)
    monkeypatch.setattr('naics_embedder.remote.workflow.TRAINING_STOP_TIMEOUT_SECONDS', 0)
    with pytest.raises(RuntimeError, match='timeout'):
        env.workflow.finish(stop_training=True)
    assert read_state(env.root).unreachable_since is None
