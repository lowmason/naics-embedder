'''
Stage-3 checkpoint contracts: exact resume and explicit weights-only migration.

A repaired checkpoint records the supervision contract it was trained under. Exact resume restores
optimizer, epoch, curriculum, and sampler state, so it requires an identical contract. Legacy or
mismatched checkpoints can only contribute allowlisted encoder weights through an explicit
weights-only migration that leaves all training state freshly initialized.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
)

CHECKPOINT_KEY = 'stage3_supervision'
WEIGHTS_ONLY_ALLOWED_PREFIXES = ('encoder.', )
WEIGHTS_ONLY_EXCLUDED_PREFIXES = (
    'loss_fn.',
    'hierarchy_loss_fn.',
    'lambdarank_loss_fn.',
    'structural_preference_loss_fn.',
    'ground_truth_distances',
    'norm_adaptive_margin.',
)

# -------------------------------------------------------------------------------------------------
# Contract
# -------------------------------------------------------------------------------------------------

class CheckpointContract(BaseModel):
    '''The supervision identity a checkpoint was trained under.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    supervision_mode: str
    contract_version: str = CONTRACT_VERSION
    bundle_id: str
    codebook_fingerprint: str
    structural_preference_loss_version: str = STRUCTURAL_PREFERENCE_LOSS_VERSION
    mining_contract_version: str = MINING_CONTRACT_VERSION

@dataclass(frozen=True)
class MigrationReport:
    '''Parameter groups a weights-only migration loaded, skipped, or left freshly initialized.'''

    loaded: Tuple[str, ...]
    skipped: Tuple[str, ...]
    missing: Tuple[str, ...]
    unexpected: Tuple[str, ...]

def contract_for_bundle(manifest: Any, supervision_mode: str = 'repaired') -> CheckpointContract:
    '''The runtime contract for training against a validated bundle manifest.'''

    return CheckpointContract(
        supervision_mode=supervision_mode,
        contract_version=manifest.contract_version,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
    )

# -------------------------------------------------------------------------------------------------
# Exact resume
# -------------------------------------------------------------------------------------------------

def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    # Lightning checkpoints carry pickled hyperparameters and loop state; they are trusted
    # artifacts produced by this project's own training runs.
    return torch.load(Path(path), map_location='cpu', weights_only=False)

def validate_checkpoint_contract(
    raw: Optional[Dict[str, Any]],
    runtime: CheckpointContract,
) -> None:
    '''
    Require a saved checkpoint contract identical to the runtime contract.

    Raises:
        ValueError: If the checkpoint predates the contract (legacy) or any field differs.
    '''

    if raw is None:
        raise ValueError(
            'legacy checkpoint has no Stage-3 contract and cannot exact resume; '
            'use weights_only explicitly'
        )
    saved = CheckpointContract.model_validate(raw)
    if saved != runtime:
        differences = {
            name: (getattr(saved, name), getattr(runtime, name))
            for name in CheckpointContract.model_fields
            if getattr(saved, name) != getattr(runtime, name)
        }
        raise ValueError(f'exact resume contract mismatch (saved, runtime): {differences}')

def validate_exact_resume(path: str | Path, runtime: CheckpointContract) -> None:
    '''Require that the checkpoint at ``path`` was trained under the runtime contract.'''

    validate_checkpoint_contract(_load_checkpoint(path).get(CHECKPOINT_KEY), runtime)

# -------------------------------------------------------------------------------------------------
# Weights-only migration
# -------------------------------------------------------------------------------------------------

def load_weights_only(model: torch.nn.Module, path: str | Path) -> MigrationReport:
    '''
    Load only allowlisted encoder weights; never optimizer, epoch, curriculum, or sampler state.

    Loss buffers and legacy structural matrices are skipped; bundle-derived buffers stay as the
    runtime bundle built them.

    Raises:
        ValueError: If the checkpoint has no state dict, or carries parameters that are neither
            allowlisted nor known-excluded, or allowlisted parameters with mismatched shapes.
    '''

    checkpoint = _load_checkpoint(path)
    source = checkpoint.get('state_dict')
    if not isinstance(source, dict):
        raise ValueError('weights-only checkpoint has no state_dict')
    target = model.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    skipped = []
    unexpected = []
    for name, value in source.items():
        if name.startswith(WEIGHTS_ONLY_ALLOWED_PREFIXES):
            if name not in target or target[name].shape != value.shape:
                unexpected.append(name)
            else:
                loaded[name] = value
        elif name.startswith(WEIGHTS_ONLY_EXCLUDED_PREFIXES):
            skipped.append(name)
        else:
            unexpected.append(name)
    if unexpected:
        raise ValueError(
            f'weights-only checkpoint has unexpected parameter groups: {sorted(unexpected)}'
        )
    model.load_state_dict(loaded, strict=False)
    missing = tuple(
        sorted(
            name for name in target
            if name.startswith(WEIGHTS_ONLY_ALLOWED_PREFIXES) and name not in loaded
        )
    )
    return MigrationReport(
        loaded=tuple(sorted(loaded)),
        skipped=tuple(sorted(skipped)),
        missing=missing,
        unexpected=(),
    )
