'''Exact training launch gates and immutable segment provenance.'''

import json
import os
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from naics_embedder.remote.canonical import InputSet, ResumePlan, canonical_inputs, resume_plan
from naics_embedder.remote.code_manifest import code_entries, git_bytes
from naics_embedder.remote.config import relative_path
from naics_embedder.remote.provenance import PushRecord
from naics_embedder.remote.push import read_push_record, scan_code
from naics_embedder.remote.session import RemoteInfo, RemoteState, RunRecord, new_id, write_state
from naics_embedder.remote.transport import _qualified_gpu
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import Config, OutcomePanelConfig, load_config
from naics_embedder.utils.training import (
    constructor_settings,
    effective_precision,
    refuse_a_fresh_start_into_a_used_directory,
    run_settings,
)

# -------------------------------------------------------------------------------------------------
# Command and path authority
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchResult:
    segment_id: str | None
    skipped: bool
    reason: str | None

def _checkpoint_base(cfg: Config, info: RemoteInfo) -> str:
    repo, base = PurePosixPath(info.repo), PurePosixPath(info.checkpoint_base)
    if not repo.is_absolute() or not base.is_absolute() or '..' in repo.parts + base.parts:
        raise ValueError('remote roots must be literal absolute paths')
    if base != repo / 'checkpoints':
        raise ValueError('checkpoint base must be the recorded repo checkpoints directory')
    if cfg.dirs.checkpoint_dir not in ('checkpoints', './checkpoints', str(base)):
        raise ValueError('configured checkpoint base differs from recorded remote base')
    return str(base)

def _mapped_path(root: str, value: str, mapping: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f'{mapping} path escapes remote repo: {value}') from error
    if '..' in path.parts or not path.parts or path.parts[0] != mapping or '\\' in value:
        raise ValueError(f'path must remain under repo {mapping}/: {value}')
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        raise ValueError('control characters forbidden in output paths')
    return str(path)

def _execution_config(cfg: Config, inputs: InputSet, info: RemoteInfo) -> Config:
    base = _checkpoint_base(cfg, info)
    if cfg.training.trainer.devices != 1:
        raise ValueError('remote text training requires exactly one device')
    if info.accelerator not in ('cuda', 'cpu'):
        raise ValueError('remote accelerator must be CUDA (CPU only for injected tests)')
    manifest = next(
        name for name in inputs.paths if inputs.manifest.as_posix().endswith('/' + name)
    )
    if cfg.supervision.manifest_path not in (manifest, './' + manifest, str(inputs.manifest)):
        raise ValueError('configured manifest differs from canonical inputs')
    for value, mapping in (
        (cfg.dirs.output_dir, 'outputs'), (cfg.dirs.log_dir, 'logs'), (
            cfg.data_loader.tokenization.output_path, 'data'
        )
    ):
        _mapped_path(info.repo, value, mapping)
    precision = effective_precision(cfg, info.accelerator)
    if info.accelerator == 'cuda' and precision != 'bf16-mixed':
        raise ValueError('CUDA training requires native BF16 without precision fallback')
    return cfg.override(
        {
            'dirs.checkpoint_dir': base,
            'supervision.manifest_path': manifest,
            'training.trainer.accelerator': info.accelerator,
            'training.trainer.precision': '32' if precision == '32-true' else precision,
            'training.trainer.devices': 1,
        }
    )

def training_argv(
    cfg: Config, config_path: str, overrides: list[str], inputs: InputSet, info: RemoteInfo,
    resume: bool
) -> tuple[str, ...]:
    '''Render one literal argument vector, with exact-only resume and canonical path authority.'''
    relative_path(Path('/'), config_path)
    effective = _execution_config(cfg, inputs, info)
    manifest = next(
        name for name in inputs.paths if inputs.manifest.as_posix().endswith('/' + name)
    )
    root = inputs.manifest
    for _ in Path(manifest).parts:
        root = root.parent
    panel_path = root / 'conf/data/outcome_panel.yaml'
    if not panel_path.is_file():
        raise ValueError('missing pushed outcome monitor configuration')
    _mapped_path(info.repo, load_config(OutcomePanelConfig, panel_path).selection_log, 'logs')
    if not PurePosixPath(info.uv).is_absolute():
        raise ValueError('recorded uv executable must be absolute')
    reserved = {
        'dirs.checkpoint_dir': {'checkpoints', './checkpoints', info.checkpoint_base},
        'supervision.manifest_path': {
            effective.supervision.manifest_path, './' + effective.supervision.manifest_path,
            str(inputs.manifest)
        },
    }
    for value in overrides:
        if value.startswith('-') or '=' not in value:
            raise ValueError(
                'remote overrides must be literal key=value; checkpoint flags forbidden'
            )
        key, content = value.split('=', 1)
        if key in reserved and content not in reserved[key]:
            raise ValueError('conflicting remote-reserved path override: ' + key)
    argv = [info.uv, 'run', '--locked', 'naics-embedder', 'train', '--config', config_path]
    if resume:
        argv.extend(['--ckpt-path', 'last', '--checkpoint-load-mode', 'exact'])
    argv.extend(overrides)
    argv.extend(
        [
            'supervision.manifest_path=' + effective.supervision.manifest_path,
            'dirs.checkpoint_dir=' + effective.dirs.checkpoint_dir,
            'training.trainer.accelerator=' + info.accelerator,
            'training.trainer.precision=' + effective.training.trainer.precision,
            'training.trainer.devices=1',
        ]
    )
    return tuple(argv)

# -------------------------------------------------------------------------------------------------
# Durable identity and launch transaction
# -------------------------------------------------------------------------------------------------

def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def _run_record(
    root: Path, state: RemoteState, cfg: Config, inputs: InputSet, remote_directory: str,
    segment: str
) -> None:
    manifest = inputs.bundle.manifest
    candidate = RunRecord(
        experiment=cfg.experiment_name,
        remote_directory=remote_directory,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        description_fingerprint=manifest.description_fingerprint,
        seed=cfg.seed,
        settings=run_settings(
            cfg,
            accelerator=state.remote_info.accelerator,
            precision=effective_precision(cfg, state.remote_info.accelerator)
        ),
        constructor_controls=constructor_settings(cfg),
        session_id=state.session_id,
        segment_id=segment,
    )
    path = root / '.remote/runs' / (cfg.experiment_name + '.json')
    if path.exists():
        old = RunRecord.model_validate_json(path.read_text())
        excluded = {'training_run', 'session_id', 'segment_id'}
        if old.model_dump(exclude=excluded) != candidate.model_dump(exclude=excluded):
            raise ValueError('persisted run path or identity differs')
    else:
        _write_new(path, candidate.model_dump(mode='json'))

def _ready_code(root: Path, state: RemoteState, transport: object) -> PushRecord:
    if state.status != 'ready' or state.remote_info is None or state.push_id is None:
        raise ValueError('remote train requires a ready session; rerun remote up')
    if state.pending_push_id is not None:
        raise ValueError('pending push; rerun remote up before launch')
    if os.path.lexists(root / '.remote/pulls/pending.json'):
        raise ValueError('pending pull promotion; sync/recover first before launch')
    from naics_embedder.remote.sync import verify_local_sync_manifest
    verify_local_sync_manifest(root, state)
    record = read_push_record(root, state.push_id)
    if code_entries(root) != record.entries or git_bytes(root, 'rev-parse', 'HEAD'
                                                         ).decode().strip() != record.head_sha:
        raise ValueError('local code changed; rerun remote up before launch')
    from naics_embedder.remote.loop import _session_config
    remote_cfg = _session_config(root, state.session_id)
    actual = scan_code(transport, record.entries, remote_cfg)
    if actual != {entry.path: entry for entry in record.entries}:
        raise ValueError('pushed remote code changed; rerun remote up')
    return record

def _identity(state: RemoteState, transport: object) -> None:
    info = state.remote_info
    actual = transport.probe('identity', {})
    expected = {
        'repo': info.repo,
        'checkpoint_base': info.checkpoint_base,
        'session_id': state.session_id,
        'push_id': state.push_id
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError('remote root/user/session identity differs; rerun remote up')

def _stopped(transport: object) -> None:
    if transport.probe('training', {'action': 'status'}).get('running') is not False:
        raise ValueError('training tmux must be stopped before launch')

def _remote_hashes(transport: object, directory: str) -> dict[str, str]:
    result = {}
    for item in transport.probe('inventory', {'path': directory})['files']:
        if item['kind'] != 'file' or item['path'] in result:
            raise ValueError('unsafe remote continuation inventory; sync first')
        result[item['path']] = item['sha256']
    return result

def _restore(root: Path, transport: object, plan: ResumePlan) -> None:
    local = {
        str(Path(name).relative_to(plan.directory.relative_to(root))): digest
        for name, digest in plan.hashes.items()
    }
    remote = _remote_hashes(transport, plan.remote_directory)
    # A stopped instance may have progressed after the Mac's last pull. Never overwrite it.
    if any(local.get(name) != digest for name, digest in remote.items()):
        raise ValueError('remote run differs or is newer; sync first, then re-plan resume')
    transport.push(root, str(PurePosixPath(plan.remote_directory).parent.parent), plan.files)
    if _remote_hashes(transport, plan.remote_directory) != local:
        raise ValueError('restored continuation SHA inventory mismatch')

def _wrapper(
    info: RemoteInfo, argv: tuple[str, ...], exit_path: str, visibility: str | None
) -> str:
    environment = (
        'unset CUDA_VISIBLE_DEVICES\n'
        if visibility is None else 'export CUDA_VISIBLE_DEVICES=' + shlex.quote(visibility) + '\n'
    )
    return (
        f'cd {shlex.quote(info.repo)} || exit 1\n' + environment + 'set +e\n'
        + f'{shlex.join(argv)} < /dev/null\nstatus=$?\n'
        + f'printf "%s\\n" "$status" > {shlex.quote(exit_path)}.tmp\n'
        + f'mv {shlex.quote(exit_path)}.tmp {shlex.quote(exit_path)}\nexit "$status"\n'
    )

def launch_training(
    root: Path, state: RemoteState, transport: object, cfg: Config, config_path: str,
    overrides: list[str], inputs: InputSet, resume: bool
) -> LaunchResult:
    '''Launch only a verified stopped generation; caller holds the workflow state lock.'''
    record = _ready_code(root, state, transport)
    info = state.remote_info
    effective = _execution_config(cfg, inputs, info)
    argv = training_argv(cfg, config_path, overrides, inputs, info, resume)
    remote_directory = info.checkpoint_base + '/' + cfg.experiment_name
    if relative_path(root, config_path) not in {entry.path for entry in record.entries}:
        raise ValueError('training config was not pushed; rerun remote up')
    snapshot = json.loads((root / '.remote/session-inputs.json').read_text())
    if (
        snapshot.get('session_id') != state.session_id or snapshot.get('push_id') != state.push_id
        or snapshot.get('hashes') != inputs.hashes or set(snapshot.get('paths', [])) != set(
            inputs.paths
        )
    ):
        raise ValueError('canonical inputs changed or snapshot differs; rerun remote up')
    _identity(state, transport)
    _stopped(transport)
    if canonical_inputs(root, cfg).hashes != inputs.hashes:
        raise ValueError('local canonical inputs changed')
    if transport.probe('canonical', {
        'config': cfg.model_dump(mode='json')
    }).get('hashes') != inputs.hashes:
        raise ValueError('remote canonical inputs differ')
    plan = resume_plan(root, cfg, inputs, remote_directory) if resume else None
    run_path = root / '.remote/runs' / (cfg.experiment_name + '.json')
    if plan is not None and run_path.exists():
        known = RunRecord.model_validate_json(run_path.read_text()).training_run
        if known is not None and known != plan.training_run:
            raise ValueError('checkpoint training_run differs from verified persistent run')
    if plan is not None and plan.finished:
        return LaunchResult(None, True, plan.finish_reason)
    if not resume:
        local_directory = root / 'checkpoints' / cfg.experiment_name
        if local_directory.is_symlink() or local_directory.parent.is_symlink():
            raise ValueError('links are forbidden in checkpoint directories')
        refuse_a_fresh_start_into_a_used_directory(local_directory)
    parent = root / '.remote/segments'
    existing = {path.name for path in parent.iterdir()} if parent.exists() else set()
    now = datetime.now(timezone.utc)
    segment = new_id('segment', now, existing)
    directory = parent / segment
    remote_segment = info.repo + '/.remote/segments/' + segment
    transport.probe('launch_lock', {'segment_id': segment, 'action': 'acquire'})
    try:
        _stopped(transport)
        _identity(state, transport)
        if transport.probe('clock', {}).get('ntp') is not True:
            raise ValueError('NTP synchronization required immediately before launch')
        if resume:
            _restore(root, transport, plan)
            if resume_plan(root, cfg, inputs, remote_directory).hashes != plan.hashes:
                raise ValueError('local continuation changed during restoration')
        transport.probe(
            'training', {
                'action': 'preflight',
                'config': effective.model_dump(mode='json'),
                'resume': resume,
                'remote_directory': remote_directory
            }
        )
        _ready_code(root, state, transport)
        if canonical_inputs(root, cfg).hashes != inputs.hashes:
            raise ValueError('canonical inputs changed during launch')
        visibility = info.gpu_evidence.cuda_visible_devices if info.gpu_evidence else None
        evidence = transport.probe('gpu', {'cuda_visible_devices': visibility})
        if not _qualified_gpu(evidence) or evidence['cuda_visible_devices'] != visibility:
            raise ValueError('native BF16 GPU qualification or CUDA visibility failed')
        _run_record(root, state, effective, inputs, remote_directory, segment)
        manifest = inputs.bundle.manifest
        segment_record = {
            'segment_id': segment,
            'session_id': state.session_id,
            'host': state.host,
            'started_utc': now.isoformat(),
            'push_id': record.push_id,
            'head_sha': record.head_sha,
            'dirty': record.dirty,
            'experiment_name': cfg.experiment_name,
            'bundle_id': manifest.bundle_id,
            'codebook_fingerprint': manifest.codebook_fingerprint,
            'description_fingerprint': manifest.description_fingerprint,
            'resume': resume,
            'resumed_from': {
                'path': remote_directory + '/last.ckpt',
                'sha256': plan.hashes[plan.last.relative_to(root).as_posix()]
            } if plan else None,
            'argv': list(argv),
            'command': shlex.join(argv),
            'effective_config': effective.model_dump(mode='json'),
            'input_hashes': inputs.hashes,
            'continuation_hashes': plan.hashes if plan else {},
            'remote_directory': remote_directory,
            'lock_sha256': sha256_file(root / 'uv.lock'),
            'outcome_panel_config': load_config(
                OutcomePanelConfig, root / 'conf/data/outcome_panel.yaml'
            ).model_dump(mode='json'),
            'gpu_evidence': evidence,
            'cuda_visible_devices': visibility,
        }
        _write_new(directory / 'segment.json', segment_record)
        shutil.copytree(record.directory, directory / 'code')
        script = _wrapper(info, argv, remote_segment + '/exit_code', visibility)
        (directory / 'launch.sh').write_text(script)
        files = tuple(
            path.relative_to(directory).as_posix() for path in directory.rglob('*')
            if path.is_file()
        )
        transport.push(
            directory, remote_segment, tuple(name for name in files if name != 'segment.json')
        )
        transport.probe(
            'write_record', {
                'path': '.remote/segments/' + segment + '/segment.json',
                'record': segment_record,
                'immutable': True
            }
        )
        remote_files = _remote_hashes(transport, remote_segment)
        if remote_files != {name: sha256_file(directory / name) for name in files}:
            raise ValueError('remote segment provenance SHA mismatch')
        # Provenance transfer must not open a stale-code/input or fresh-directory window.
        _ready_code(root, state, transport)
        _identity(state, transport)
        if canonical_inputs(root, cfg).hashes != inputs.hashes or transport.probe(
            'canonical', {
                'config': cfg.model_dump(mode='json')
            }
        ).get('hashes') != inputs.hashes:
            raise ValueError('canonical inputs changed before tmux launch')
        transport.probe(
            'training', {
                'action': 'preflight',
                'config': effective.model_dump(mode='json'),
                'resume': resume,
                'remote_directory': remote_directory
            }
        )
        _stopped(transport)
        transport.launch(remote_segment + '/launch.sh', segment)
        state.active_segment_id = segment
        write_state(root, state)
    except BaseException as error:
        _write_new(directory / 'failure.json', {'segment_id': segment, 'error': str(error)})
        raise
    finally:
        transport.probe('launch_lock', {'segment_id': segment, 'action': 'release'})
    from naics_embedder.remote.loop import ensure_loop
    ensure_loop(root, state)
    return LaunchResult(segment, False, None)
