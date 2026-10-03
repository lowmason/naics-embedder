'''
Stage-3 checkpoint contracts: exact resume and explicit weights-only migration.

A repaired checkpoint records the supervision contract it was trained under and the encoder
architecture its weights belong to. Exact resume restores optimizer, epoch, curriculum, and
sampler state, so it requires an identical contract. A checkpoint of the same architecture under
other supervision can only contribute allowlisted encoder weights, through an explicit
weights-only migration that leaves all training state freshly initialized.

Another architecture can do neither. A checkpoint saved before Stage 6 has no encoder record and
reads as the legacy four-copy layout, and nothing migrates it into the shared encoder (roadmap
D2).
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
)

CHECKPOINT_KEY = 'stage3_supervision'
LEGACY_CONTAINMENT_BUNDLE_ID = 'legacy-containment'
UNVERSIONED_CODEBOOK_FINGERPRINT = 'unversioned'
WEIGHTS_ONLY_ALLOWED_PREFIXES = ('encoder.', )
WEIGHTS_ONLY_EXCLUDED_PREFIXES = (
    'loss_fn.',
    'hierarchy_loss_fn.',
    'lambdarank_loss_fn.',
    'structural_preference_loss_fn.',
    'ground_truth_distances',
    'norm_adaptive_margin.',
)
D2_REFUSAL = (
    'a checkpoint of another encoder architecture cannot load, and nothing migrates it: '
    'four-copy checkpoints cannot load into the shared encoder (roadmap D2)'
)

# -------------------------------------------------------------------------------------------------
# Contract
# -------------------------------------------------------------------------------------------------

class EncoderArchitecture(BaseModel):
    '''
    The encoder architecture a checkpoint's weights belong to (spec 4.4).

    ``shared`` is Stage 6's one backbone, and names its fusion, dimension and backbone.
    ``four-copy`` is the legacy layout of every checkpoint saved before Stage 6, and names nothing
    else. A field added later defaults to the value every earlier checkpoint had.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')

    layout: Literal['shared', 'four-copy']
    fusion: Optional[str] = None
    dimension: Optional[int] = None
    backbone: Optional[str] = None

    @model_validator(mode='after')
    def check_fields_match_the_layout(self) -> 'EncoderArchitecture':
        '''A shared record names its fusion, dimension and backbone; a four-copy one, none.'''

        recorded = (self.fusion, self.dimension, self.backbone)
        if self.layout == 'shared' and None in recorded:
            raise ValueError('a shared encoder record names its fusion, dimension and backbone')
        if self.layout == 'four-copy' and recorded != (None, None, None):
            raise ValueError('a four-copy encoder record names no fusion, dimension or backbone')
        return self

LEGACY_ENCODER = EncoderArchitecture(layout='four-copy')

def shared_encoder_architecture(
    *,
    fusion: str,
    dimension: int,
    backbone: str,
) -> EncoderArchitecture:
    '''
    The record of a Stage-6 shared encoder.

    The model builds its record here from its hyperparameters, and training builds the config's
    here too, so the two cannot drift apart.
    '''

    return EncoderArchitecture(
        layout='shared', fusion=fusion, dimension=dimension, backbone=backbone
    )

class CheckpointContract(BaseModel):
    '''The supervision identity a checkpoint was trained under, and its encoder architecture.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    supervision_mode: str
    contract_version: str = CONTRACT_VERSION
    bundle_id: str
    codebook_fingerprint: str
    structural_preference_loss_version: str = STRUCTURAL_PREFERENCE_LOSS_VERSION
    mining_contract_version: str = MINING_CONTRACT_VERSION
    # Absent from every contract saved before Stage 6, which therefore reads as four-copy
    encoder: EncoderArchitecture = LEGACY_ENCODER

@dataclass(frozen=True)
class MigrationReport:
    '''Parameter groups a weights-only migration loaded, skipped, or left freshly initialized.'''

    loaded: Tuple[str, ...]
    skipped: Tuple[str, ...]
    missing: Tuple[str, ...]
    unexpected: Tuple[str, ...]

def contract_for_bundle(
    manifest: Any,
    supervision_mode: str = 'repaired',
    *,
    encoder: EncoderArchitecture,
) -> CheckpointContract:
    '''The runtime contract for training against a validated bundle manifest.'''

    return CheckpointContract(
        supervision_mode=supervision_mode,
        contract_version=manifest.contract_version,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        encoder=encoder,
    )

def containment_contract(*, encoder: EncoderArchitecture) -> CheckpointContract:
    '''
    The tag every legacy-containment checkpoint carries.

    It can never equal a repaired contract, so containment checkpoints cannot exact-resume into
    repaired training.
    '''

    return CheckpointContract(
        supervision_mode='legacy_containment',
        bundle_id=LEGACY_CONTAINMENT_BUNDLE_ID,
        codebook_fingerprint=UNVERSIONED_CODEBOOK_FINGERPRINT,
        encoder=encoder,
    )

def saved_encoder(raw: Optional[Dict[str, Any]]) -> EncoderArchitecture:
    '''A saved contract's encoder record. No contract, or no record, is the four-copy layout.'''

    if raw is None:
        return LEGACY_ENCODER
    return CheckpointContract.model_validate(raw).encoder

def _differences(saved: CheckpointContract,
                 expected: CheckpointContract) -> Dict[str, Tuple[Any, Any]]:
    return {
        name: (getattr(saved, name), getattr(expected, name))
        for name in CheckpointContract.model_fields
        if getattr(saved, name) != getattr(expected, name)
    }

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
        ValueError: If the checkpoint predates the contract (legacy) or any field differs. A
            checkpoint without a contract, or of another encoder architecture, carries the D2
            refusal.
    '''

    if raw is None:
        raise ValueError(
            f'legacy checkpoint has no Stage-3 contract and cannot exact resume; {D2_REFUSAL}'
        )
    saved = CheckpointContract.model_validate(raw)
    if saved != runtime:
        differences = _differences(saved, runtime)
        message = f'exact resume contract mismatch (saved, runtime): {differences}'
        if 'encoder' in differences:
            message = f'{message}; {D2_REFUSAL}'
        raise ValueError(message)

def validate_exact_resume(path: str | Path, runtime: CheckpointContract) -> None:
    '''Require that the checkpoint at ``path`` was trained under the runtime contract.'''

    validate_checkpoint_contract(_load_checkpoint(path).get(CHECKPOINT_KEY), runtime)

def validate_supervision_contract(
    raw: Optional[Dict[str, Any]],
    manifest: Any,
    supervision_mode: str = 'repaired',
) -> CheckpointContract:
    '''
    Require a saved contract whose supervision fields match the configured bundle's.

    Export and reads take the encoder record from the checkpoint (spec 4.4), so it is not compared
    here. ``load_from_checkpoint`` rebuilds the checkpoint's own architecture from its saved
    hyperparameters, and refuses a four-copy one.

    Returns:
        The saved contract.

    Raises:
        ValueError: If the checkpoint has no contract, or a supervision field differs.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    configured = contract_for_bundle(manifest, supervision_mode, encoder=saved.encoder)
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
            f'{_differences(saved, configured)}'
        )
    return saved

# -------------------------------------------------------------------------------------------------
# Weights-only migration
# -------------------------------------------------------------------------------------------------

def load_weights_only(
    model: torch.nn.Module,
    path: str | Path,
    *,
    encoder: EncoderArchitecture,
) -> MigrationReport:
    '''
    Load only allowlisted encoder weights; never optimizer, epoch, curriculum, or sampler state.

    The saved encoder record (an absent one counts as four-copy) must equal ``encoder`` before any
    parameter is read (roadmap D2). Loss buffers and legacy structural matrices are skipped;
    bundle-derived buffers stay as the runtime bundle built them.

    Args:
        model: The freshly built runtime model.
        path: The checkpoint to migrate from.
        encoder: The runtime model's encoder record.

    Raises:
        ValueError: If the checkpoint's encoder record differs from ``encoder``; if it has no
            state dict; if it carries parameters that are neither allowlisted nor known-excluded,
            or allowlisted parameters with mismatched shapes; or if it contributes no allowlisted
            parameter at all.
    '''

    checkpoint = _load_checkpoint(path)
    saved = saved_encoder(checkpoint.get(CHECKPOINT_KEY))
    if saved != encoder:
        raise ValueError(
            f'weights-only encoder mismatch (saved, runtime): {(saved, encoder)}; {D2_REFUSAL}'
        )
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
    if not loaded:
        raise ValueError(
            f'weights-only checkpoint {path} has no allowlisted encoder parameters to load'
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
