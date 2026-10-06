'''Canonical bundle transport and exact continuation gates, without training or panel reads.'''

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch
import yaml

from naics_embedder.cli.commands.training import runtime_contract_for
from naics_embedder.remote.config import relative_path
from naics_embedder.supervision.artifacts import (
    ValidatedSupervisionBundle,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.checkpoints import validate_exact_resume
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, read_epoch_summary
from naics_embedder.text_model.monitor import MONITOR_RECORDS, read_monitor_records
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import (
    effective_precision,
    outcome_checkpoint,
    outcome_early_stopping,
    read_checkpoint,
    refuse_a_resume_from_another_directory,
    refuse_a_resume_of_a_stopped_run,
    refuse_a_resume_under_other_settings,
    refuse_other_constructor_settings,
    run_settings,
)

# -------------------------------------------------------------------------------------------------
# Immutable inventories
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InputSet:
    manifest: Path
    descriptions: Path
    bundle: ValidatedSupervisionBundle
    paths: tuple[str, ...]
    hashes: dict[str, str]

@dataclass(frozen=True)
class ResumePlan:
    directory: Path
    remote_directory: str
    last: Path
    epoch: int
    training_run: str
    files: tuple[str, ...]
    hashes: dict[str, str]
    finished: bool
    finish_reason: str | None

def _regular_file(root: Path, path: Path) -> None:
    relative_path(root, path.relative_to(root).as_posix())
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f'links are forbidden in canonical or continuation files: {path}')
    if not path.is_file():
        raise ValueError(f'missing regular file: {path}')

def _inventory(root: Path, paths: list[Path]) -> tuple[tuple[str, ...], dict[str, str]]:
    hashes = {}
    for path in sorted(set(paths)):
        _regular_file(root, path)
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f'unstable file changed during inventory: {path}')
        hashes[path.relative_to(root).as_posix()] = digest
    return tuple(hashes), hashes

def _data_path(root: Path, value: str) -> Path:
    relative = relative_path(root, value)
    if Path(relative).parts[0] != 'data':
        raise ValueError(f'canonical inputs must be under repo data/: {value}')
    return root / relative

def canonical_inputs(root: Path, cfg: Config) -> InputSet:
    '''Validate every canonical member and description bytes, then hash their complete upload.'''
    root = root.resolve()
    manifest_value = cfg.supervision.manifest_path
    if manifest_value is None:
        raise ValueError('set supervision.manifest_path after running data supervision on the Mac')
    manifest = _data_path(root, manifest_value)
    descriptions = _data_path(root, cfg.data_loader.streaming.descriptions_parquet)
    _regular_file(root, manifest)
    _regular_file(root, descriptions)
    # Check paths before the bundle loader can read any member outside the canonical directory.
    manifest_hash = sha256_file(manifest)
    raw = json.loads(manifest.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get('artifacts'), dict):
        raise ValueError(f'malformed manifest {manifest}: expected an artifacts object')
    for name, artifact in raw['artifacts'].items():
        if not isinstance(artifact, dict) or not isinstance(artifact.get('path'), str):
            raise ValueError(f'malformed manifest {manifest}: artifact {name} needs a path')
        if not isinstance(artifact.get('files'), list):
            raise ValueError(f'malformed manifest {manifest}: artifact {name} needs a files list')
        for member in artifact['files']:
            if not isinstance(member, dict) or not isinstance(member.get('path'), str):
                raise ValueError(
                    f'malformed manifest {manifest}: artifact {name} member needs a path'
                )
            relative_path(manifest.parent, member['path'])
            _regular_file(root, manifest.parent / member['path'])
        relative_path(manifest.parent, artifact['path'])
    bundle = load_validated_bundle(manifest)
    if sha256_file(descriptions) != bundle.manifest.description_fingerprint:
        raise ValueError('configured description bytes differ from bundle description_fingerprint')
    paths = [descriptions]
    for path in manifest.parent.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'links are forbidden in canonical inputs: {path}')
        if path.is_file():
            paths.append(path)
    names, hashes = _inventory(root, paths)
    expected = {manifest: manifest_hash, descriptions: bundle.manifest.description_fingerprint}
    for artifact in bundle.manifest.artifacts.values():
        for member in artifact.files:
            expected[bundle.root / member.path] = member.sha256
    if any(
        hashes[path.relative_to(root).as_posix()] != digest for path, digest in expected.items()
    ):
        raise ValueError('canonical input hash changed after bundle validation')
    return InputSet(manifest, descriptions, bundle, names, hashes)

# -------------------------------------------------------------------------------------------------
# Completion and histories
# -------------------------------------------------------------------------------------------------

def _epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('last.ckpt names no non-negative integer epoch')
    return value

def finished_run(saved: Mapping[str, Any], patience: int,
                 max_epochs: int) -> tuple[bool, str | None]:
    '''Recognize the actual saved early stop or an exhausted unchanged epoch budget.'''
    epoch = _epoch(saved.get('epoch'))
    state = (saved.get('callbacks') or {}).get(outcome_early_stopping(patience).state_key) or {}
    stopped = state.get('stopped_epoch')
    if stopped is None:
        refuse_a_resume_of_a_stopped_run(saved, patience)
    if isinstance(stopped, bool) or not isinstance(stopped, int) or stopped < 0 or stopped > epoch:
        raise ValueError('corrupt EarlyStopping stopped_epoch')
    wait = state.get('wait_count')
    score = state.get('best_score')
    if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0:
        raise ValueError('corrupt EarlyStopping wait_count')
    if isinstance(score, torch.Tensor):
        if score.numel() != 1:
            raise ValueError('corrupt EarlyStopping best_score')
        score = score.item()
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('corrupt EarlyStopping best_score')
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError('epoch budget must be a positive integer')
    # The shared guard validates the exact same callback and raises for a real stopped run.
    try:
        refuse_a_resume_of_a_stopped_run(saved, patience)
    except ValueError:
        if stopped > 0:
            return True, f'early stopping ended the run at epoch {stopped}'
        raise
    if epoch + 1 >= max_epochs:
        return True, 'epoch budget exhausted'
    return False, None

def _histories(directory: Path, epoch: int, training_run: str, seed: int) -> None:
    summaries = read_epoch_summary(directory / EPOCH_SUMMARY)
    monitors = read_monitor_records(directory / MONITOR_RECORDS)
    monitor_epochs = []
    fingerprint = None
    mrrs = {}
    for row in monitors:
        read = row['read']
        detail = read['detail']
        mrr = row['mrr']
        if not math.isfinite(mrr) or not 0 <= mrr <= 1:
            raise ValueError('monitor mrr must be finite and in [0, 1]')
        if (read.get('event'), read.get('panel'), read.get('split')) != (
            'read', 'outcome', 'validation'
        ):
            raise ValueError('monitor must read the outcome validation panel')
        if detail.get('training_run') != training_run or detail.get('seed') != seed:
            raise ValueError('monitor names another training run or seed')
        if isinstance(detail.get('seed'), bool) or not isinstance(detail.get('seed'), int):
            raise ValueError('monitor seed must be an integer')
        named = read.get('fingerprint')
        if not isinstance(named, str) or not named.strip():
            raise ValueError('monitor names no panel fingerprint')
        if fingerprint is None:
            fingerprint = named
        if named != fingerprint:
            raise ValueError('monitor panel fingerprint changed')
        monitor_epochs.append(detail['epoch'])
        mrrs[detail['epoch']] = mrr
    summary_epochs = [row['epoch'] for row in summaries]
    for name, epochs in ((MONITOR_RECORDS, monitor_epochs), (EPOCH_SUMMARY, summary_epochs)):
        if epochs != list(range(len(epochs))) or len(epochs) < epoch + 1:
            raise ValueError(f'{name} must cover epochs 0 through {epoch} exactly once in order')
    for row in summaries:
        if row['epoch'] <= epoch and row['mrr'] != mrrs.get(row['epoch']):
            raise ValueError('epoch summary MRR differs from monitor MRR')
        if row['mrr'] is None:
            raise ValueError('epoch summary needs a finite monitor MRR')

def validate_continuation_set(
    directory: Path, saved: Mapping[str, Any], remote_directory: str
) -> None:
    '''Require the authoritative kept set and earliest-best monitor epoch without rewriting bytes.'''
    refuse_a_resume_from_another_directory(saved, directory, resolved_dirpath=remote_directory)
    epoch = _epoch(saved.get('epoch'))
    run = saved.get('training_run')
    seed = saved.get('hyper_parameters', {}).get('seed')
    if (
        not isinstance(run, str) or not run.strip() or isinstance(seed, bool) or not isinstance(
            seed, int
        )
    ):
        raise ValueError('checkpoint names no run or integer seed')
    _histories(directory, epoch, run, seed)
    key = outcome_checkpoint(directory).state_key
    callback = saved['callbacks'][key]
    if (
        not isinstance(callback.get('best_model_path'), str) or not callback['best_model_path']
        or not isinstance(callback.get('best_k_models'), dict) or not callback['best_k_models']
    ):
        raise ValueError('checkpoint names no authoritative kept ModelCheckpoint set')
    required = set()
    for state in saved['callbacks'].values():
        if not isinstance(state, dict) or 'dirpath' not in state:
            continue
        if state['dirpath'] != remote_directory:
            raise ValueError('kept callback directory differs from literal run directory')
        references = [
            state.get(field)
            for field in ('best_model_path', 'kth_best_model_path', 'last_model_path')
        ]
        kept = state.get('best_k_models', {})
        if not isinstance(kept, dict):
            raise ValueError('malformed authoritative kept checkpoint mapping')
        references.extend(kept)
        for reference in references:
            if reference in ('', None):
                continue
            if not isinstance(reference, str):
                raise ValueError('kept checkpoint reference must be a literal path')
            path = PurePosixPath(reference)
            if '..' in path.parts or str(path) != reference:
                raise ValueError('kept checkpoint reference is not a normalized literal path')
            try:
                relative = path.relative_to(PurePosixPath(remote_directory))
            except ValueError as error:
                raise ValueError(
                    'kept checkpoint reference escapes literal run directory'
                ) from error
            if not relative.parts or path.suffix != '.ckpt':
                raise ValueError('kept checkpoint reference names no checkpoint file')
            required.add(directory / str(relative))
    monitors = [
        row for row in read_monitor_records(directory / MONITOR_RECORDS)
        if row['read']['detail']['epoch'] <= epoch
    ]
    best = max(row['mrr'] for row in monitors)
    selected_epoch = min(row['read']['detail']['epoch'] for row in monitors if row['mrr'] == best)
    selected = directory / f'epoch={selected_epoch:03d}.ckpt'
    required.add(selected)
    for path in sorted(required):
        if not path.is_file():
            raise ValueError(f'missing authoritative kept or selected checkpoint: {path}')
        _regular_file(directory, path)
    if list(directory.glob(f'epoch={selected_epoch:03d}-v*.ckpt')):
        raise ValueError('selected kept checkpoint has ambiguous version siblings')
    for path in directory.rglob('*.ckpt'):
        _regular_file(directory, path)
        checkpoint = read_checkpoint(path)
        kept_epoch = _epoch(checkpoint.get('epoch'))
        kept_seed = checkpoint.get('hyper_parameters', {}).get('seed')
        if (
            not isinstance(checkpoint.get('state_dict'), Mapping) or
            checkpoint.get('training_run') != run or isinstance(kept_seed, bool) or not isinstance(
                kept_seed, int
            ) or kept_seed != seed or kept_epoch > epoch
        ):
            raise ValueError('kept checkpoint names another run, seed or later epoch')
        refuse_a_resume_from_another_directory(
            checkpoint, directory, resolved_dirpath=remote_directory
        )
        named_epoch = re.fullmatch(r'epoch=(\d+)(?:-v\d+)?\.ckpt', path.name)
        if named_epoch and int(named_epoch[1]) != kept_epoch:
            raise ValueError('kept checkpoint filename and saved epoch differ')
        if path == selected:
            score = checkpoint['callbacks'][key].get('best_model_score')
            if isinstance(score, torch.Tensor) and score.numel() == 1:
                score = score.item()
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score != best or kept_epoch != selected_epoch
            ):
                raise ValueError(
                    'selected kept checkpoint epoch or best score differs from monitor'
                )

# -------------------------------------------------------------------------------------------------
# Exact resume plan
# -------------------------------------------------------------------------------------------------

def resume_plan(root: Path, cfg: Config, inputs: InputSet, remote_directory: str) -> ResumePlan:
    '''Resume only last.ckpt and retain all original continuation bytes for exact restoration.'''
    root = root.resolve()
    base = cfg.dirs.checkpoint_dir
    if Path(base).is_absolute():
        if str(Path(remote_directory).parent) != base:
            raise ValueError('configured absolute checkpoint base differs from remote run')
        base = 'checkpoints'
    directory = root / relative_path(root, base) / relative_path(root, cfg.experiment_name)
    last = directory / 'last.ckpt'
    paths = []
    if directory.is_symlink():
        raise ValueError('links are forbidden in checkpoint directories')
    for required in (last, directory / MONITOR_RECORDS, directory / EPOCH_SUMMARY):
        _regular_file(root, required)
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'links are forbidden in continuation files: {path}')
        if path.is_file():
            if path.suffix in ('.tmp', '.part'):
                raise ValueError(f'unstable continuation file: {path}')
            if path.suffix in ('.yaml', '.yml'):
                if not isinstance(yaml.safe_load(path.read_text()), dict):
                    raise ValueError(f'malformed run config: {path}')
            if path.suffix == '.json':
                if not isinstance(json.loads(path.read_text()), dict):
                    raise ValueError(f'malformed run summary: {path}')
            paths.append(path)
    files, hashes = _inventory(root, paths)
    saved = None
    for path in paths:
        if path.suffix != '.ckpt':
            continue
        try:
            checkpoint = read_checkpoint(path)
            if not isinstance(checkpoint, Mapping) or not isinstance(
                checkpoint.get('state_dict'), Mapping
            ):
                raise ValueError('checkpoint has no model state')
            _epoch(checkpoint.get('epoch'))
        except (ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            raise ValueError(f'malformed checkpoint: {path}: {error}') from error
        if path == last:
            saved = checkpoint
    assert saved is not None  # last.ckpt is a required inventory member
    validate_exact_resume(last, runtime_contract_for(cfg, inputs.bundle))
    refuse_a_resume_from_another_directory(saved, directory, resolved_dirpath=remote_directory)
    accelerator = 'cpu' if cfg.training.trainer.accelerator == 'cpu' else 'cuda'
    settings = run_settings(
        cfg, accelerator=accelerator, precision=effective_precision(cfg, accelerator)
    )
    refuse_a_resume_under_other_settings(saved, settings, seed=cfg.seed)
    refuse_other_constructor_settings(saved, cfg)
    epoch = _epoch(saved.get('epoch'))
    training_run = saved.get('training_run')
    if not isinstance(training_run, str) or not training_run.strip():
        raise ValueError('last.ckpt names no training run')
    validate_continuation_set(directory, saved, remote_directory)
    finished, reason = finished_run(
        saved, cfg.training.early_stopping_patience, cfg.training.trainer.max_epochs
    )
    # Validation must not race a producer updating checkpoint or history bytes.
    current = [path for path in directory.rglob('*') if path.is_file() or path.is_symlink()]
    if set(current) != set(paths) or _inventory(root, paths)[1] != hashes:
        raise ValueError('continuation files changed during resume validation')
    return ResumePlan(
        directory, remote_directory, last, epoch, training_run, files, hashes, finished, reason
    )
