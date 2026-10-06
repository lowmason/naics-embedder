import json
import os
import shutil

import pytest

from naics_embedder.remote.session import StateBusyError, read_state, state_lock
from naics_embedder.remote.sync import sync_once, verify_final_mappings, verify_local_sync_manifest

def test_failed_transfer_keeps_previous_generation(remote_sync_fixture):
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    paths = list((env.root / 'checkpoints/run').iterdir())
    before = {p: p.read_bytes() for p in paths}
    env.transport.fail_next_pull = True
    with pytest.raises(OSError):
        sync_once(env.root, env.state, env.transport, env.cfg)
    assert {p: p.read_bytes() for p in paths} == before
    assert read_state(env.root).unreachable_since is not None
    sync_once(env.root, env.state, env.transport, env.cfg)
    assert read_state(env.root).unreachable_since is None

def test_busy_history_defers_entire_run_and_final_has_no_skip(remote_sync_fixture):
    env = remote_sync_fixture
    os.utime(env.run / 'epoch_summary.jsonl', (env.now, env.now))
    result = sync_once(env.root, env.state, env.transport, env.cfg)
    assert result.pending == 4
    assert not (env.root / 'checkpoints/run/last.ckpt').exists()
    result = sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert result.pending == 0 and result.pulled == 7

@pytest.mark.parametrize('mutation', ['rewrite', 'truncate', 'remove'])
def test_source_changes_never_promote(remote_sync_fixture, mutation):
    env = remote_sync_fixture

    def change(source):
        if source.name != 'checkpoints':
            return
        path = env.run / 'epoch_summary.jsonl'
        before = path.stat()
        if mutation == 'remove':
            path.unlink()
        else:
            path.write_text(path.read_text().replace('0.5', '0.6') if mutation == 'rewrite' else '')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    env.transport.after_pull = change
    with pytest.raises(ValueError, match='unstable'):
        sync_once(env.root, env.state, env.transport, env.cfg)
    assert not (env.root / 'checkpoints/run/last.ckpt').exists()

@pytest.mark.parametrize('mutation', ['mrr', 'identity', 'checkpoint'])
def test_incoherent_run_refuses(remote_sync_fixture, mutation):
    env = remote_sync_fixture
    if mutation == 'checkpoint':
        (env.run / 'last.ckpt').write_bytes(b'bad checkpoint')
    else:
        path = env.run / ('epoch_summary.jsonl' if mutation == 'mrr' else 'monitor_reads.jsonl')
        path.write_text(path.read_text().replace('0.5', '0.6').replace('tiny-run', 'other'))
    with pytest.raises((ValueError, RuntimeError)):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert not (env.root / 'checkpoints/run/last.ckpt').exists()

def test_cumulative_manifest_keeps_deferred_run_tamper_coverage(remote_sync_fixture):
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    os.utime(env.run / 'epoch_summary.jsonl', (env.now, env.now))
    sync_once(env.root, env.state, env.transport, env.cfg)
    path = env.root / 'checkpoints/run/last.ckpt'
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Mac'):
        sync_once(env.root, env.state, env.transport, env.cfg)

@pytest.mark.parametrize('remove', [False, True])
def test_tampered_or_missing_previous_file_refuses(remote_sync_fixture, remove):
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    path = env.root / 'checkpoints/run/last.ckpt'
    path.unlink() if remove else path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Mac'):
        verify_local_sync_manifest(env.root, env.state)
    with pytest.raises(ValueError, match='Mac'):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)

def test_all_mappings_preserve_mac_only_and_isolate_logs(remote_sync_fixture):
    env = remote_sync_fixture
    path = env.root / 'checkpoints/run/older.ckpt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'older')
    sync_once(env.root, env.state, env.transport, env.cfg)
    assert path.read_bytes() == b'older'
    assert (env.root / 'logs/remote/session/selection_log.jsonl').exists()
    assert not (env.root / 'logs/selection_log.jsonl').exists()
    assert (env.root / 'outputs/remote/session/segments/segment/exit_code').exists()
    verify_local_sync_manifest(env.root, env.state)

def test_sparse_run_is_pending(remote_sync_fixture):
    env = remote_sync_fixture
    (env.run / 'last.ckpt').unlink()
    result = sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert result.pending == 3
    assert not (env.root / 'checkpoints/run').exists()

def test_concurrent_sync_refuses(remote_sync_fixture):
    env = remote_sync_fixture
    with state_lock(env.root), pytest.raises(StateBusyError):
        sync_once(env.root, env.state, env.transport, env.cfg)

def test_checksum_ignores_directory_noise_but_refuses_content(remote_sync_fixture):
    env = remote_sync_fixture
    env.transport.differences = ()
    verify_final_mappings(env.root, env.state, env.transport)
    env.transport.differences = ('>fc........ run/last.ckpt', )
    with pytest.raises(ValueError, match='checksum'):
        verify_final_mappings(env.root, env.state, env.transport)

def test_promotion_crash_replays_before_previous_hash_check(remote_sync_fixture, monkeypatch):
    import naics_embedder.remote.sync as module
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    (env.instance / 'outputs/tensor').write_text('next')
    original = module.os.replace
    count = [0]

    def fail(source, target, *args, **kwargs):
        if (kwargs.get('src_dir_fd') is not None
            or '/pulls/' in str(source)) and str(target).endswith('tensor'):
            original(source, target, *args, **kwargs)
            count[0] += 1
            raise OSError('crashed after replace')
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    monkeypatch.setattr(module.os, 'replace', original)
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert count[0] == 1
    verify_local_sync_manifest(env.root, env.state)

def test_inventory_prunes_partials_and_credentials(remote_sync_fixture):
    env = remote_sync_fixture
    for name in ['.rsync-partial/last.ckpt', '.env', 'keys/secret.pem']:
        path = env.instance / 'outputs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'do not open')
    result = env.transport.probe('inventory', {'path': str(env.instance / 'outputs')})
    assert [item['path'] for item in result['files']] == ['tensor']

def test_transport_checksum_drops_directory_metadata(
    remote_sync_fixture, recorded_transport_runner
):
    import subprocess

    from naics_embedder.remote.session import PullMapping
    from naics_embedder.remote.transport import SshTransport
    env = remote_sync_fixture
    transport = SshTransport('fixture', '/repo', '/rsync', recorded_transport_runner)
    (env.root / 'run').mkdir()
    (env.root / 'mode-only').write_text('same')
    recorded_transport_runner.replies.append(
        subprocess.CompletedProcess(
            [], 0, b'.d..t.p....|run/\n.f...p.....|mode-only\n>fc........|run/last.ckpt\n', b''
        )
    )
    assert transport.checksum(PullMapping('/repo/checkpoints', env.root)) == ('run/last.ckpt', )

def test_source_changed_after_its_transfer_before_promotion_refuses(remote_sync_fixture):
    env = remote_sync_fixture

    def change(source):
        if source.name == 'logs':
            (env.run / 'epoch_summary.jsonl').write_text('{"epoch": 0, "mrr": 0.6}\n')

    env.transport.after_pull = change
    with pytest.raises(ValueError, match='unstable'):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert not (env.root / 'checkpoints/run/last.ckpt').exists()

def test_runtime_transport_error_records_unreachable(remote_sync_fixture):
    env = remote_sync_fixture

    def fail(operation, payload):
        raise RuntimeError('SSH connection refused')

    env.transport.probe = fail
    with pytest.raises(RuntimeError):
        sync_once(env.root, env.state, env.transport, env.cfg)
    assert read_state(env.root).unreachable_since is not None

def test_result_inventory_uses_trusted_rendered_source(
    remote_sync_fixture, recorded_transport_runner
):
    from naics_embedder.remote.transport import SshTransport
    transport = SshTransport('fixture', '/repo', '/rsync', recorded_transport_runner)
    transport.python = '/python'
    transport.probe('inventory', {'path': '/repo/checkpoints'})
    command = recorded_transport_runner.calls[-1].args[-1]
    assert ' -c ' in command and '-m naics_embedder' not in command
    assert '_result_names' in command and 'O_NOFOLLOW' in command

def test_inventory_refuses_swapped_ancestor_without_opening_target(
    remote_sync_fixture, monkeypatch
):
    from naics_embedder.remote import worker
    env = remote_sync_fixture
    directory = env.instance / 'outputs/child'
    directory.mkdir()
    (directory / 'result').write_text('old')
    outside = env.root / 'credentials'
    outside.mkdir()
    (outside / '.env').write_text('do not read')
    original = worker._open_directory

    def swap(name, parent=None):
        if name == 'child':
            directory.rename(directory.with_name('old-child'))
            directory.symlink_to(outside, target_is_directory=True)
        return original(name, parent)

    monkeypatch.setattr(worker, '_open_directory', swap)
    with pytest.raises(OSError):
        env.transport.probe('inventory', {'path': str(env.instance / 'outputs')})

@pytest.mark.parametrize('kind', ['file', 'symlink', 'missing'])
def test_checksum_directory_over_other_type_is_material(
    remote_sync_fixture, recorded_transport_runner, kind
):
    import subprocess

    from naics_embedder.remote.session import PullMapping
    from naics_embedder.remote.transport import SshTransport
    env = remote_sync_fixture
    path = env.root / 'run'
    if kind == 'file':
        path.write_bytes(b'file')
    elif kind == 'symlink':
        path.symlink_to(env.instance, target_is_directory=True)
    transport = SshTransport('fixture', '/repo', '/rsync', recorded_transport_runner)
    recorded_transport_runner.replies.append(
        subprocess.CompletedProcess([], 0, b'.d..t......|run/\n', b'')
    )
    assert transport.checksum(PullMapping('/repo/checkpoints', env.root)) == ('run/', )

def test_final_checksum_prunes_transport_partials(remote_sync_fixture, recorded_transport_runner):
    from naics_embedder.remote.session import PullMapping
    from naics_embedder.remote.transport import SshTransport
    env = remote_sync_fixture
    transport = SshTransport('fixture', '/repo', '/rsync', recorded_transport_runner)
    transport.checksum(PullMapping('/repo/checkpoints', env.root))
    assert '--exclude=.rsync-partial/' in recorded_transport_runner.calls[-1].args

def test_local_change_between_promotions_refuses_instead_of_repair(
    remote_sync_fixture, monkeypatch
):
    import naics_embedder.remote.sync as module
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    with (env.run / 'epoch=000.ckpt').open('ab') as stream:
        stream.write(b'new bytes')
    original = module.os.replace

    def change(source, target, *args, **kwargs):
        result = original(source, target, *args, **kwargs)
        if (kwargs.get('src_dir_fd') is not None
            or '/pulls/' in str(source)) and str(target).endswith('epoch=000.ckpt'):
            (env.root / 'checkpoints/run/last.ckpt').write_bytes(b'evidence')
        return result

    monkeypatch.setattr(module.os, 'replace', change)
    with pytest.raises(ValueError, match='Mac'):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert (env.root / 'checkpoints/run/last.ckpt').read_bytes() == b'evidence'

def test_rendered_result_inventory_prunes_before_reads(remote_sync_fixture):
    import subprocess
    import sys

    from naics_embedder.remote.transport import _system_probe_code
    env = remote_sync_fixture
    (env.instance / 'outputs/.env').write_bytes(b'credential')
    part = env.instance / 'outputs/.rsync-partial/file'
    part.parent.mkdir()
    part.write_bytes(b'partial')
    result = subprocess.run(
        [sys.executable, '-c', _system_probe_code('inventory')],
        input=json.dumps(dict(repo=str(env.instance), path='outputs')),
        text=True,
        capture_output=True,
        check=True,
        timeout=10
    )
    assert [item['path'] for item in json.loads(result.stdout)['files']] == ['tensor']

def test_promotion_anchors_destination_against_swapped_ancestor(remote_sync_fixture, monkeypatch):
    import naics_embedder.remote.sync as module
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    (env.instance / 'outputs/tensor').write_text('new')
    outside = env.root / 'external'
    outside.mkdir()
    (outside / 'tensor').write_text('credential')
    directory = env.root / 'outputs/remote/session'
    original = module.os.replace
    swapped = [False]

    def swap(source, target, *args, **kwargs):
        if str(target).endswith('tensor') and not swapped[0]:
            swapped[0] = True
            directory.rename(directory.with_name('saved-session'))
            directory.symlink_to(outside, target_is_directory=True)
        return original(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, 'replace', swap)
    with pytest.raises(ValueError, match='Mac'):
        sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert (outside / 'tensor').read_text() == 'credential'

def test_final_checksum_never_interprets_difference_path_as_itemize(remote_sync_fixture):
    env = remote_sync_fixture
    env.transport.differences = ('.d..t...... filename', )
    with pytest.raises(ValueError, match='checksum'):
        verify_final_mappings(env.root, env.state, env.transport)

@pytest.mark.parametrize('replacement', [False, True])
def test_next_stable_epoch_promotes_whole_bundle_preserving_kept(remote_sync_fixture, replacement):
    import torch
    env = remote_sync_fixture
    first = sync_once(env.root, env.state, env.transport, env.cfg)
    if replacement:
        from naics_embedder.remote.session import write_state
        from naics_embedder.remote.sync import inherit_sync_manifest
        state = env.state.model_copy(deep=True, update={'session_id': 'replacement'})
        inherit_sync_manifest(env.root, env.state, state)
        env.state = state
        write_state(env.root, state)
        (env.root / '.remote/session-config.json').write_text(
            json.dumps(dict(session_id=state.session_id, remote_config=env.cfg.model_dump()))
        )
    saved = torch.load(env.run / 'last.ckpt', weights_only=False)
    saved['epoch'] = 1
    torch.save(saved, env.run / 'last.ckpt')
    shutil.copy2(env.run / 'last.ckpt', env.run / 'epoch=001.ckpt')
    for name in ['monitor_reads.jsonl', 'epoch_summary.jsonl']:
        path = env.run / name
        row = json.loads(path.read_text())
        if name == 'monitor_reads.jsonl':
            row['read']['detail']['epoch'] = 1
        else:
            row['epoch'] = 1
        with path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    second = sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    assert first.last_sha256['run'] != second.last_sha256['run']
    local = env.root / 'checkpoints/run'
    assert (local / 'epoch=000.ckpt').exists() and (local / 'epoch=001.ckpt').exists()
    assert torch.load(local / 'last.ckpt', weights_only=False)['epoch'] == 1
    assert len((local / 'epoch_summary.jsonl').read_text().splitlines()) == 2
    verify_local_sync_manifest(env.root, env.state)

def test_transport_timeout_retains_first_unreachable_and_last_good_then_recovers(
    remote_sync_fixture
):
    import subprocess
    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg)
    before = read_state(env.root)
    probe = env.transport.probe

    def timeout(operation, payload):
        raise subprocess.TimeoutExpired(['ssh'], 30)

    env.transport.probe = timeout
    with pytest.raises(subprocess.TimeoutExpired):
        sync_once(env.root, env.state, env.transport, env.cfg)
    first = read_state(env.root)
    assert first.unreachable_since is not None
    assert first.last_sync_manifest == before.last_sync_manifest
    assert first.last_sync_utc == before.last_sync_utc
    with pytest.raises(subprocess.TimeoutExpired):
        sync_once(env.root, env.state, env.transport, env.cfg)
    assert read_state(env.root).unreachable_since == first.unreachable_since
    env.transport.probe = probe
    sync_once(env.root, env.state, env.transport, env.cfg)
    assert read_state(env.root).unreachable_since is None
    verify_local_sync_manifest(env.root, env.state)

@pytest.mark.parametrize('failure', ['missing', 'tampered', 'foreign'])
def test_inherited_original_manifest_provenance_refuses(remote_sync_fixture, failure):
    from naics_embedder.remote.session import write_state
    from naics_embedder.remote.sync import inherit_sync_manifest

    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    original = env.root / env.state.last_sync_manifest
    replacement = env.state.model_copy(deep=True, update={'session_id': 'replacement'})
    inherit_sync_manifest(env.root, env.state, replacement)
    write_state(env.root, replacement)
    (env.root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=replacement.session_id, remote_config=env.cfg.model_dump()))
    )
    if failure == 'missing':
        original.unlink()
    else:
        record = json.loads(original.read_text())
        record['session_id'
               if failure == 'foreign' else 'files'] = 'foreign' if failure == 'foreign' else {}
        original.write_text(json.dumps(record))
    env.transport.calls.clear()
    with pytest.raises(ValueError, match='manifest'):
        sync_once(env.root, replacement, env.transport, env.cfg, final=True)
    assert not env.transport.calls

@pytest.mark.parametrize('crash', [False, True])
def test_inherited_bytes_allow_legitimate_updates_and_recovery(
    remote_sync_fixture, monkeypatch, crash
):
    import naics_embedder.remote.sync as module
    from naics_embedder.remote.session import write_state

    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    old_reference = env.state.last_sync_manifest
    original = (env.root / old_reference).read_bytes()
    replacement = env.state.model_copy(deep=True, update={'session_id': 'replacement'})
    module.inherit_sync_manifest(env.root, env.state, replacement)
    write_state(env.root, replacement)
    (env.root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=replacement.session_id, remote_config=env.cfg.model_dump()))
    )
    (env.instance / 'outputs/tensor').write_text('legitimate later output')
    promote = module._promote_file
    if crash:

        def interrupt(*args):
            promote(*args)
            raise OSError('promotion crash')

        monkeypatch.setattr(module, '_promote_file', interrupt)
        with pytest.raises(OSError, match='promotion crash'):
            sync_once(env.root, replacement, env.transport, env.cfg, final=True)
        (env.root / 'logs/remote/session/selection_log.jsonl').write_text('altered inherited log')
        with pytest.raises(ValueError, match='Mac file changed outside pending'):
            sync_once(env.root, replacement, env.transport, env.cfg, final=True)
        (env.root / 'logs/remote/session/selection_log.jsonl').write_text('fixture\n')
        monkeypatch.setattr(module, '_promote_file', promote)
    sync_once(env.root, replacement, env.transport, env.cfg, final=True)
    assert (env.root / 'outputs/remote/replacement/tensor').read_text() == 'legitimate later output'
    assert (env.root / old_reference).read_bytes() == original
    verify_local_sync_manifest(env.root, replacement)

def test_manifest_replacement_symlink_never_follows_external_bytes(
    remote_sync_fixture, tmp_path, monkeypatch
):
    import naics_embedder.remote.sync as module

    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    reference = env.state.last_sync_manifest
    path = env.root / reference
    outside = tmp_path / 'outside-manifest'
    outside.write_bytes(path.read_bytes())
    original = module._local_hash
    swapped = []

    def swap_after_hash(root, name):
        digest = original(root, name)
        if name == reference and not swapped:
            swapped.append(True)
            path.unlink()
            path.symlink_to(outside)
        return digest

    monkeypatch.setattr(module, '_local_hash', swap_after_hash)
    with pytest.raises((OSError, ValueError)):
        verify_local_sync_manifest(env.root, env.state)
    assert swapped and outside.read_bytes()

def test_inherited_manifest_publication_never_follows_swapped_parent(
    remote_sync_fixture, tmp_path, monkeypatch
):
    import naics_embedder.remote.sync as module

    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    replacement = env.state.model_copy(deep=True, update={'session_id': 'replacement'})
    outside = tmp_path / 'external'
    outside.mkdir()
    original = module._manifest
    calls = []

    def swap_after_verification(root, state):
        files = original(root, state)
        calls.append(True)
        if len(calls) == 2:
            parent = root / '.remote/pulls'
            parent.rename(parent.with_name('preserved-pulls'))
            parent.symlink_to(outside, target_is_directory=True)
        return files

    monkeypatch.setattr(module, '_manifest', swap_after_verification)
    with pytest.raises((OSError, ValueError)):
        module.inherit_sync_manifest(env.root, env.state, replacement)
    assert not list(outside.iterdir())
    assert read_state(env.root).session_id == env.state.session_id

@pytest.mark.parametrize('field', ['files', 'inherited'])
def test_pending_recovery_cannot_drop_inherited_baseline(remote_sync_fixture, monkeypatch, field):
    import naics_embedder.remote.sync as module
    from naics_embedder.remote.session import write_state

    env = remote_sync_fixture
    sync_once(env.root, env.state, env.transport, env.cfg, final=True)
    state = env.state.model_copy(deep=True, update={'session_id': 'replacement'})
    module.inherit_sync_manifest(env.root, env.state, state)
    write_state(env.root, state)
    (env.root / '.remote/session-config.json').write_text(
        json.dumps(dict(session_id=state.session_id, remote_config=env.cfg.model_dump()))
    )
    promote = module._promote_file

    def interrupt(*args):
        raise OSError('staged promotion interruption')

    monkeypatch.setattr(module, '_promote_file', interrupt)
    with pytest.raises(OSError):
        sync_once(env.root, state, env.transport, env.cfg, final=True)
    journal = env.root / '.remote/pulls/pending.json'
    record = json.loads(journal.read_text())
    record[field] = {} if field == 'files' else []
    journal.write_text(json.dumps(record))
    monkeypatch.setattr(module, '_promote_file', promote)
    with pytest.raises(ValueError, match='baseline'):
        module.recover_pending_promotion(env.root, state)
    assert journal.is_file()
