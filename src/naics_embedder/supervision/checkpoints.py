'''
Stage-3 checkpoint contracts: exact resume, and the checks of export and reads.

A checkpoint records the supervision contract it was trained under and the encoder architecture
its weights belong to. Exact resume restores optimizer, epoch, curriculum, and sampler state, so
it requires an identical contract. Nothing loads a checkpoint under any other contract: there is
no weights-only migration (roadmap D2).

A checkpoint saved before Stage 6 has no encoder record and reads as the legacy four-copy layout,
and nothing migrates it into the shared encoder (roadmap D2).
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

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
    # The sha256 of the window-fitting summaries the model read (panels/window_summaries.py).
    # Absent from every contract saved before Stage 6b, which therefore reads as null: those
    # checkpoints trained on truncated text
    summaries: Optional[str] = None

def contract_for_bundle(
    manifest: Any,
    *,
    encoder: EncoderArchitecture,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    The runtime contract for training against a validated bundle manifest.

    ``summaries`` is the sha256 of the window-fitting summaries the model reads, or None for a
    backbone with no pin (``panels.window_summaries.summaries_identity``).
    '''

    return CheckpointContract(
        # The one mode left: legacy containment is deleted (roadmap D2)
        supervision_mode='repaired',
        contract_version=manifest.contract_version,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        encoder=encoder,
        summaries=summaries,
    )

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
    *,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    Require a saved contract whose supervision fields and summaries match the configured ones.

    Export and reads take the encoder record from the checkpoint (spec 4.4), so it is not compared
    here. ``load_from_checkpoint`` rebuilds the checkpoint's own architecture from its saved
    hyperparameters, and refuses a four-copy one.

    Args:
        raw: The checkpoint's saved contract, or None.
        manifest: The configured bundle's manifest.
        summaries: The sha256 of the summaries the read applies; keyword-only with no default,
            so a caller cannot omit it.

    Returns:
        The saved contract.

    Raises:
        ValueError: If the checkpoint has no contract, or a supervision field or the summaries
            differ.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    configured = contract_for_bundle(manifest, encoder=saved.encoder, summaries=summaries)
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
            f'{_differences(saved, configured)}'
        )
    return saved
