'''Five thin CLI adapters and the workflow sync locking/configuration boundary.'''

import json
from pathlib import Path

import pytest

from naics_embedder.cli import app
from naics_embedder.remote.launch import LaunchResult
from naics_embedder.remote.session import StateBusyError, read_state, state_lock, write_state
from naics_embedder.remote.workflow import FinishResult, RemoteWorkflow
from naics_embedder.utils.config import RemoteConfig

pytestmark = pytest.mark.unit

def words(value):
    return ' '.join(value.split())

@pytest.mark.parametrize('command', [None, 'up', 'train', 'sync', 'finish', 'status'])
def test_remote_help_is_lazy(cli_runner, tmp_path, monkeypatch, command):
    from naics_embedder.cli.commands import remote
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote, '_workflow', lambda *args: pytest.fail('help constructed workflow'))
    args = ['remote'] + ([command] if command else []) + ['--help']
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == []
    if command is None:
        assert all(name in result.output for name in ('up', 'train', 'sync', 'finish', 'status'))
    assert '--ckpt-path' not in result.output
    assert 'weights-only' not in result.output
    assert '--command' not in result.output

def test_up_preserves_config_overrides_and_flags(cli_runner, fake_workflow):
    overrides = ['seed=1', 'experiment_name=a b', 'training.learning_rate=0.01', 'seed=2']
    result = cli_runner.invoke(
        app, [
            'remote', '--remote-config', 'conf/gnu.yaml', 'up', '--host', 'ubuntu@fixture',
            '--force', '--config', 'conf/custom.yaml', *overrides
        ]
    )
    assert result.exit_code == 0, result.output
    assert fake_workflow.calls == [('up', 'ubuntu@fixture', 'conf/custom.yaml', overrides, True)]
    assert fake_workflow.config_paths == [Path('conf/gnu.yaml')]

def test_train_forwards_resume_once(cli_runner, fake_workflow):
    overrides = ['seed=1', 'seed=2', 'experiment_name=literal[red]']
    result = cli_runner.invoke(
        app, ['remote', 'train', '--resume', '--config', 'conf/custom.yaml', *overrides]
    )
    assert result.exit_code == 0, result.output
    assert fake_workflow.calls == [('train', True, 'conf/custom.yaml', overrides)]

def test_remote_finished_skip_is_success_without_a_launch(cli_runner, fake_workflow):
    fake_workflow.train_result = LaunchResult(None, True, 'saved epoch budget exhausted')
    result = cli_runner.invoke(app, ['remote', 'train', '--resume', 'seed=1'])
    assert result.exit_code == 0
    assert 'saved epoch budget exhausted' in words(result.output)
    assert 'Launched' not in result.output

@pytest.mark.parametrize('once', [False, True])
def test_sync_forwards_once(cli_runner, fake_workflow, once):
    result = cli_runner.invoke(app, ['remote', 'sync'] + (['--once'] if once else []))
    assert result.exit_code == 0
    assert fake_workflow.calls == [('sync', once)]
    assert ('pending: 2' if once else 'Sync loop') in words(result.output)

@pytest.mark.parametrize(
    'flags, expected', [
        ([], (False, False, False)), (['--stop-training', '--pull-edits'], (True, True, False)),
        (['--abandon'], (False, False, True))
    ]
)
def test_finish_forwards_flags(cli_runner, fake_workflow, flags, expected):
    fake_workflow.finish_result = FinishResult(False, True, None, None,
                                               None) if expected[2] else fake_workflow.finish_result
    result = cli_runner.invoke(app, ['remote', 'finish', *flags])
    assert result.exit_code == 0
    assert fake_workflow.calls == [('finish', *expected)]
    assert ('Safe to terminate' in words(result.output)) is not expected[2]
    if expected[2]:
        assert 'may be lost' in words(result.output)

def test_finish_without_checkpoint_is_explicit(cli_runner, fake_workflow):
    fake_workflow.finish_result = FinishResult(True, False, None, None, None)
    result = cli_runner.invoke(app, ['remote', 'finish'])
    assert result.exit_code == 0
    assert 'Safe to terminate' in words(result.output)
    assert 'no checkpoint' in words(result.output)
    assert 'None' not in result.output

def test_unsafe_finish_never_prints_safe(cli_runner, fake_workflow):
    fake_workflow.finish_result = FinishResult(False, False, None, None, None)
    result = cli_runner.invoke(app, ['remote', 'finish'])
    assert result.exit_code == 1
    assert 'Safe to terminate' not in words(result.output)

@pytest.mark.parametrize('command', ['up', 'train', 'sync', 'finish', 'status'])
def test_refusal_is_exit_one_with_literal_markup(cli_runner, fake_workflow, command):
    fake_workflow.error = ValueError('refused [red]literal[/red] input')
    args = ['remote', command] + (['--host', 'fixture'] if command == 'up' else [])
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 1
    assert '[red]literal[/red]' in words(result.output)
    assert 'Safe to terminate' not in words(result.output)

def test_status_renders_observations_and_errors(cli_runner, fake_workflow):
    fake_workflow.status_result = dict(
        status='ready',
        host='fixture',
        pending=4,
        training=dict(running=False, exit_code=1),
        gpu=dict(utilization=12),
        loop=dict(alive=False),
        unreachable_since='timestamp',
        errors=['stale [red]owner[/red]']
    )
    result = cli_runner.invoke(app, ['remote', 'status'])
    assert result.exit_code == 1
    assert fake_workflow.calls == [('status', )]
    for value in (
        'pending', '4', 'exit_code', 'utilization', 'unreachable_since', '[red]owner[/red]'
    ):
        assert value in words(result.output)

@pytest.mark.parametrize('command', ['train', 'sync', 'finish'])
def test_missing_session_refuses_before_transport(cli_runner, remote_repo, monkeypatch, command):
    from naics_embedder.cli.commands import remote
    monkeypatch.chdir(remote_repo.root)

    def forbidden(*args):
        pytest.fail('no transport before a recorded ready session')

    monkeypatch.setattr(remote, '_transport', forbidden)
    result = cli_runner.invoke(app, ['remote', command])
    assert result.exit_code == 1, result.output

def test_missing_host_refuses_without_subprocess(cli_runner, remote_repo, monkeypatch):
    monkeypatch.chdir(remote_repo.root)
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('missing host ran transport'))
    result = cli_runner.invoke(app, ['remote', 'up'])
    assert result.exit_code == 1, result.output

def test_missing_remote_config_refuses_before_transport(cli_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'subprocess.run', lambda *a, **k: pytest.fail('missing config ran transport')
    )
    result = cli_runner.invoke(
        app, ['remote', '--remote-config', 'missing.yaml', 'up', '--host', 'fixture']
    )
    assert result.exit_code == 1, result.output

@pytest.mark.parametrize('once', [False, True])
def test_workflow_sync_uses_persisted_custom_config_and_correct_lock(
    remote_sync_fixture, monkeypatch, once
):
    env = remote_sync_fixture
    seen = []

    def factory(host, cfg):
        assert host == env.state.host
        assert cfg == env.cfg
        seen.append(('transport', cfg.rsync_path, cfg.sync_interval_seconds))
        return env.transport

    workflow = RemoteWorkflow(env.root, RemoteConfig(), factory, lambda: None)
    if once:
        from naics_embedder.remote.sync import sync_once

        def public(*args):
            # Acquiring the real public lock must work: the controller cannot own it here.
            result = sync_once(*args)
            seen.append(('once', result))
            return result

        monkeypatch.setattr('naics_embedder.remote.sync.sync_once', public)
        result = workflow.sync(True)
        assert result.pulled > 0
        assert seen[-1][0] == 'once'
        assert env.transport.repo == env.state.remote_info.repo
        assert env.transport.python == env.state.remote_info.python
    else:

        def ensure(root, state):
            with pytest.raises(StateBusyError):
                with state_lock(root):
                    pass
            assert state.session_id == env.state.session_id
            seen.append(('loop', state.session_id))

        monkeypatch.setattr('naics_embedder.remote.loop.ensure_loop', ensure)
        assert workflow.sync(False) is None
        assert seen == [('loop', env.state.session_id)]

@pytest.mark.parametrize('once', [False, True])
@pytest.mark.parametrize('damage', ['missing', 'foreign', 'incomplete', 'closed'])
def test_workflow_sync_fails_closed_without_default_config(
    remote_sync_fixture, monkeypatch, once, damage
):
    env = remote_sync_fixture
    path = env.root / '.remote/session-config.json'
    if damage == 'missing':
        path.unlink()
    elif damage == 'foreign':
        content = json.loads(path.read_text())
        content['session_id'] = 'other'
        path.write_text(json.dumps(content))
    elif damage == 'incomplete':
        content = json.loads(path.read_text())
        del content['remote_config']['rsync_path']
        path.write_text(json.dumps(content))
    else:
        state = read_state(env.root)
        state.status = 'abandoned'
        write_state(env.root, state)

    def forbidden(*args):
        pytest.fail('invalid persisted session reached transport or loop')

    monkeypatch.setattr('naics_embedder.remote.loop.ensure_loop', forbidden)
    workflow = RemoteWorkflow(env.root, RemoteConfig(), forbidden, lambda: None)
    with pytest.raises((ValueError, OSError)):
        workflow.sync(once)

@pytest.mark.parametrize('command', ['up', 'train'])
def test_missing_training_config_has_no_transport_operations(
    cli_runner, remote_workflow_fixture, monkeypatch, command
):
    from naics_embedder.cli.commands import remote
    env = remote_workflow_fixture
    if command == 'train':
        env.workflow.up('fixture', 'conf/config.yaml', [])
        env.transport.calls.clear()
    monkeypatch.chdir(env.root)
    monkeypatch.setattr(remote, '_transport', lambda *args: env.transport)
    args = ['remote', command, '--config', 'conf/missing.yaml']
    if command == 'up':
        args += ['--host', 'fixture']
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 1
    assert 'missing.yaml' in words(result.output)
    assert env.transport.calls == []

@pytest.mark.parametrize(
    'args', [
        ['exec', 'echo'],
        ['train', '--ckpt-path', 'last'],
        ['train', '--checkpoint-load-mode', 'weights_only'],
        ['train', '--command', 'echo'],
        ['status', '--force'],
    ]
)
def test_unavailable_switches_cannot_reach_workflow(cli_runner, fake_workflow, args):
    result = cli_runner.invoke(app, ['remote', *args])
    assert result.exit_code != 0
    assert fake_workflow.calls == []

def test_defaults_are_forwarded_without_synthetic_overrides(cli_runner, fake_workflow):
    for args in [['up', '--host', 'fixture'], ['train'], ['status']]:
        assert cli_runner.invoke(app, ['remote', *args]).exit_code == 0
    assert fake_workflow.calls == [
        ('up', 'fixture', 'conf/config.yaml', [], False), ('train', False, 'conf/config.yaml', []),
        ('status', )
    ]
    assert fake_workflow.config_paths == [Path('conf/remote.yaml')] * 3

@pytest.mark.parametrize(
    'change', [
        'finished', 'abandoned', 'config_changed', 'config_foreign', 'config_missing',
        'config_incomplete'
    ]
)
def test_one_shot_sync_revalidates_inside_public_transaction(
    remote_sync_fixture, monkeypatch, change
):
    from contextlib import contextmanager

    import naics_embedder.remote.sync as sync_module

    env = remote_sync_fixture
    observed = []
    snapshot = {}
    acquisitions = []
    original_probe = env.transport.probe
    original_lock = sync_module.state_lock

    def files():
        return {
            path.relative_to(env.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in env.root.rglob('*')
        }

    def probe(operation, payload):
        observed.append((operation, payload))
        return original_probe(operation, payload)

    @contextmanager
    def change_before_acquiring(root):
        # Workflow preflight/transport construction already used the original ready/config.
        assert read_state(root).status == 'ready'
        if change in ('finished', 'abandoned'):
            current = read_state(root)
            current.status = change
            write_state(root, current)
        else:
            path = root / '.remote/session-config.json'
            envelope = json.loads(path.read_text())
            if change == 'config_changed':
                envelope['remote_config']['rsync_path'] = '/other/gnu-rsync'
                envelope['remote_config']['sync_interval_seconds'] = 97
            elif change == 'config_foreign':
                envelope['session_id'] = 'foreign'
            elif change == 'config_incomplete':
                del envelope['remote_config']['rsync_path']
            else:
                path.unlink()
            if change != 'config_missing':
                path.write_text(json.dumps(envelope))
        with original_lock(root):
            acquisitions.append(root)
            snapshot.update(files())
            yield

    def factory(host, cfg):
        assert host == env.state.host
        assert cfg == env.cfg
        return env.transport

    monkeypatch.setattr(env.transport, 'probe', probe)
    monkeypatch.setattr(sync_module, 'state_lock', change_before_acquiring)
    workflow = RemoteWorkflow(env.root, RemoteConfig(), factory, lambda: None)
    with pytest.raises((ValueError, OSError)):
        workflow.sync(True)
    assert acquisitions == [env.root]
    assert observed == []
    assert env.transport.calls == []
    assert files() == snapshot
    assert read_state(env.root).status == (
        change if change in ('finished', 'abandoned') else 'ready'
    )
