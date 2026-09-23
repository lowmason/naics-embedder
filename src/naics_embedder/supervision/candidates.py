'''
Immutable runtime candidate types for Stage-3 negative selection.

A ``NegativeCandidateBatch`` owns every field aligned on ``[batch, candidate]``. Miners propose
source indices; a ``NegativeSelection`` names the chosen source indices and their occurrence
UIDs; ``NegativeCandidateBatch.select`` is the only final gather and applies one index to every
field at once.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Optional

import torch

# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def _gather_aligned(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    '''Gather ``[batch, candidate, ...]`` values at ``[batch, selected]`` source indices.'''

    if value.ndim < 2:
        raise ValueError('aligned candidate tensors require [batch, candidate, ...] dimensions')
    suffix = value.shape[2:]
    gather_index = indices.view(*indices.shape, *([1] * len(suffix))).expand(
        *indices.shape, *suffix
    )
    return value.gather(1, gather_index)

# -------------------------------------------------------------------------------------------------
# Candidate entities (candidate-intrinsic fields only)
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateEntityBatch:
    '''
    Candidate-intrinsic fields that may travel across ranks.

    Pair-dependent supervision (structure, exclusion direction) is deliberately absent: it is
    recomputed relative to each local anchor after any distributed gather.
    '''

    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        batch_candidates = self.code_id.shape
        if self.code_id.ndim != 2:
            raise ValueError('code_id must have shape [batch, candidate]')
        if self.candidate_uid.shape != (*batch_candidates, 3):
            raise ValueError('candidate_uid must have shape [batch, candidate, 3]')
        if self.embedding.shape[:2] != batch_candidates:
            raise ValueError('embedding does not align with code_id')
        if self.valid_mask.shape != batch_candidates:
            raise ValueError('valid_mask does not align with code_id')
        if (
            self.router_gate_probs is not None
            and self.router_gate_probs.shape[:2] != batch_candidates
        ):
            raise ValueError('router_gate_probs does not align with code_id')
        if (self.valid_mask & self.code_id.lt(0)).any():
            raise ValueError('valid candidate entities require nonnegative code IDs')
        valid_uid = self.valid_mask.unsqueeze(-1).expand_as(self.candidate_uid)
        if (valid_uid & self.candidate_uid.lt(0)).any():
            raise ValueError('valid candidate entities require nonnegative UID components')

# -------------------------------------------------------------------------------------------------
# Proposals and selections
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateProposal:
    '''Ranked source indices proposed by one strategy (never gathered values).'''

    source_indices: torch.Tensor
    scores: torch.Tensor
    reason: int

    def __post_init__(self) -> None:
        if self.source_indices.shape != self.scores.shape:
            raise ValueError('proposal indices and scores must share shape')

@dataclass(frozen=True)
class NegativeSelection:
    '''The final chosen source indices, their occurrence UIDs, scores, and reasons.'''

    source_indices: torch.Tensor
    source_candidate_uid: torch.Tensor
    scores: torch.Tensor
    reasons: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.source_indices.shape
        if self.source_indices.ndim != 2:
            raise ValueError('selection indices must have shape [batch, selected]')
        if self.source_candidate_uid.shape != (*shape, 3):
            raise ValueError('selection UIDs must have shape [batch, selected, 3]')
        if self.scores.shape != shape or self.reasons.shape != shape:
            raise ValueError('selection scores and reasons must align with indices')

# -------------------------------------------------------------------------------------------------
# Candidate pools
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectedNegativeBatch:
    '''Every candidate field gathered by one checked selection; the only input to candidate losses.'''

    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor
    sampling_role_id: torch.Tensor
    sampling_provenance_id: torch.Tensor
    relation_margin: torch.Tensor
    distance_margin: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    selection_scores: torch.Tensor
    selection_reasons: torch.Tensor
    runtime_fields: Mapping[str, torch.Tensor]

@dataclass(frozen=True)
class NegativeCandidateBatch:
    '''
    Immutable candidate pool owning every field aligned on ``[batch, candidate]``.

    Validates common leading dimensions, nonnegative identities for valid candidates, and the
    exclusion derivation. ``select`` is the one final gather.
    '''

    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor
    sampling_role_id: torch.Tensor
    sampling_provenance_id: torch.Tensor
    relation_margin: torch.Tensor
    distance_margin: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    runtime_fields: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        shape = self.code_id.shape
        if self.code_id.ndim != 2:
            raise ValueError('candidate code IDs must have shape [batch, candidate]')
        if self.candidate_uid.shape != (*shape, 3):
            raise ValueError('candidate UID must have shape [batch, candidate, 3]')
        aligned_names = (
            'structural_distance',
            'structural_relation_id',
            'anchor_excludes_candidate',
            'candidate_excludes_anchor',
            'is_explicit_exclusion',
            'semantic_target_id',
            'semantic_source_id',
            'sampling_role_id',
            'sampling_provenance_id',
            'relation_margin',
            'distance_margin',
            'valid_mask',
        )
        for name in aligned_names:
            if getattr(self, name).shape != shape:
                raise ValueError(f'{name} does not align with candidate code IDs')
        if self.embedding.shape[:2] != shape:
            raise ValueError('embedding does not align with candidate code IDs')
        if self.router_gate_probs is not None and self.router_gate_probs.shape[:2] != shape:
            raise ValueError('router_gate_probs does not align with candidate code IDs')
        for name, value in self.runtime_fields.items():
            if value.shape[:2] != shape:
                raise ValueError(f'runtime field {name!r} does not align with candidates')
        if (self.valid_mask & self.code_id.lt(0)).any():
            raise ValueError('valid negative candidates require nonnegative code IDs')
        valid_uid = self.valid_mask.unsqueeze(-1).expand_as(self.candidate_uid)
        if (valid_uid & self.candidate_uid.lt(0)).any():
            raise ValueError('valid negative candidates require nonnegative UID components')
        if not torch.equal(
            self.is_explicit_exclusion,
            self.anchor_excludes_candidate | self.candidate_excludes_anchor,
        ):
            raise ValueError('is_explicit_exclusion must equal the directional OR')
        object.__setattr__(self, 'runtime_fields', MappingProxyType(dict(self.runtime_fields)))

    def select(self, selection: NegativeSelection) -> SelectedNegativeBatch:
        '''
        Gather every aligned field with one checked set of source indices.

        Raises:
            ValueError: On batch mismatch, out-of-bounds or invalid sources, or a UID that does
                not match this pool (a stale or foreign selection).
        '''

        if selection.source_indices.shape[0] != self.code_id.shape[0]:
            raise ValueError('selection batch dimension does not match candidate pool')
        if selection.source_indices.lt(0).any() or selection.source_indices.ge(
            self.code_id.shape[1]
        ).any():
            raise ValueError('selection contains an out-of-bounds source index')
        selected_valid = _gather_aligned(self.valid_mask, selection.source_indices)
        if not selected_valid.all():
            row, slot = torch.nonzero(~selected_valid, as_tuple=False)[0].tolist()
            raise ValueError(
                f'selection references invalid source candidate at row {row}, slot {slot}'
            )
        actual_uid = _gather_aligned(self.candidate_uid, selection.source_indices)
        mismatch = actual_uid.ne(selection.source_candidate_uid).any(dim=-1)
        if mismatch.any():
            row, slot = torch.nonzero(mismatch, as_tuple=False)[0].tolist()
            raise ValueError(f'selection UID mismatch at row {row}, slot {slot}')

        gathered = {}
        for item in fields(self):
            if item.name in {'runtime_fields', 'router_gate_probs'}:
                continue
            gathered[item.name] = _gather_aligned(getattr(self, item.name), selection.source_indices)
        router = None
        if self.router_gate_probs is not None:
            router = _gather_aligned(self.router_gate_probs, selection.source_indices)
        runtime = {
            name: _gather_aligned(value, selection.source_indices)
            for name, value in self.runtime_fields.items()
        }
        return SelectedNegativeBatch(
            **gathered,
            router_gate_probs=router,
            selection_scores=selection.scores,
            selection_reasons=selection.reasons,
            runtime_fields=MappingProxyType(runtime),
        )
