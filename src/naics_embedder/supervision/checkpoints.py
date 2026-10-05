'''
Stage-3 checkpoint contracts: exact resume, and the checks of export and reads.

A checkpoint records the supervision contract it was trained under: the bundle's identity, the
objective (spec 4.5) and the encoder architecture its weights belong to. Exact resume restores
the optimizer, epoch and monitor state, so it requires an identical contract. Nothing loads a
checkpoint under any other contract: there is no weights-only migration (roadmap D2).

A checkpoint saved before Stage 7 names no objective and reads as ``pre-req11``. Exact resume,
the export, the reads and the HGCN feeder refuse it before any other check, and nothing migrates
it (D2). A checkpoint saved before Stage 6 has no encoder record either, and reads as the legacy
four-copy layout.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from naics_embedder.supervision.schema import CONTRACT_VERSION, LEGACY_OBJECTIVE, OBJECTIVE

CHECKPOINT_KEY = 'stage3_supervision'
D2_REFUSAL = (
    'a checkpoint of another encoder architecture cannot load, and nothing migrates it: '
    'four-copy checkpoints cannot load into the shared encoder (roadmap D2)'
)
# The fields of every contract saved before Stage 7, which the contract no longer has (spec 4.5):
# legacy containment's supervision mode, and the six-term objective's loss and mining versions
PRE_STAGE_7_FIELDS = (
    'supervision_mode',
    'structural_preference_loss_version',
    'mining_contract_version',
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
    '''
    The supervision identity a checkpoint was trained under, its objective and its encoder
    architecture.

    A contract saved before Stage 7 parses: the three fields Stage 7 dropped are ignored, and its
    absent objective reads as ``pre-req11``, which every load refuses (P23). Any other unknown
    field is refused.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')

    contract_version: str = CONTRACT_VERSION
    bundle_id: str
    codebook_fingerprint: str
    # Absent from every contract saved before Stage 7, which therefore reads as the legacy
    # objective (spec 4.5)
    objective: str = LEGACY_OBJECTIVE
    # Absent from every contract saved before Stage 6, which therefore reads as four-copy
    encoder: EncoderArchitecture = LEGACY_ENCODER
    # The sha256 of the window-fitting summaries the model read (panels/window_summaries.py).
    # Absent from every contract saved before Stage 6b, which therefore reads as null: those
    # checkpoints trained on truncated text
    summaries: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def ignore_the_fields_stage_7_dropped(cls, data: Any) -> Any:
        '''A contract saved before Stage 7 parses: its supervision mode and versions are ignored.'''

        if isinstance(data, Mapping):
            return {name: value for name, value in data.items() if name not in PRE_STAGE_7_FIELDS}
        return data

def contract_for_bundle(
    manifest: Any,
    *,
    encoder: EncoderArchitecture,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    The runtime contract for training against a validated bundle manifest, under Req 11's
    objective (spec 4.5).

    ``summaries`` is the sha256 of the window-fitting summaries the model reads, or None for a
    backbone with no pin (``panels.window_summaries.summaries_identity``).
    '''

    return CheckpointContract(
        contract_version=manifest.contract_version,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        objective=OBJECTIVE,
        encoder=encoder,
        summaries=summaries,
    )

def _refuse_another_objective(saved: CheckpointContract) -> None:
    '''
    Refuse a checkpoint trained under any objective but Req 11's, before any other check (P23).

    A checkpoint of another objective holds another model's weights, so no other field's
    comparison is meaningful, and nothing migrates it (D2).
    '''

    if saved.objective != OBJECTIVE:
        raise ValueError(
            f'the checkpoint was trained under the objective {saved.objective}, not {OBJECTIVE} '
            "(Req 11's three terms, the radial form and the bound): it cannot load, and nothing "
            'migrates (D2)'
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
        ValueError: If the checkpoint predates the contract (legacy), was trained under another
            objective than Req 11's (checked first), or any field differs. A checkpoint without a
            contract, of another objective or of another encoder architecture carries a D2
            refusal.
    '''

    if raw is None:
        raise ValueError(
            f'legacy checkpoint has no Stage-3 contract and cannot exact resume; {D2_REFUSAL}'
        )
    saved = CheckpointContract.model_validate(raw)
    _refuse_another_objective(saved)
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
    Require a saved contract of Req 11's objective whose supervision fields and summaries match
    the configured ones.

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
        ValueError: If the checkpoint has no contract, was trained under another objective
            (checked first; nothing migrates it, D2), or a supervision field or the summaries
            differ.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    _refuse_another_objective(saved)
    configured = contract_for_bundle(manifest, encoder=saved.encoder, summaries=summaries)
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
            f'{_differences(saved, configured)}'
        )
    return saved
