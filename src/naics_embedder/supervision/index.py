'''
Dense, codebook-backed supervision lookup for runtime anchor/candidate joins.

Structural distance and relation are symmetric lookups; exclusion is stored per direction so a
join can report, for each (anchor, candidate) view, which code published the exclusion.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Tuple

import polars as pl
import torch

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.schema import SemanticSource, SemanticTarget

# -------------------------------------------------------------------------------------------------
# Semantic ID encodings
# -------------------------------------------------------------------------------------------------

TARGET_TO_ID = {
    SemanticTarget.UNKNOWN: 0,
    SemanticTarget.RELATED: 1,
    SemanticTarget.UNRELATED: 2,
}
SOURCE_TO_ID = {
    SemanticSource.UNLABELED: 0,
    SemanticSource.TRAINING_POSITIVE: 1,
    SemanticSource.EXPLICIT_EXCLUSION: 2,
}

# -------------------------------------------------------------------------------------------------
# Joined supervision
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PairSupervision:
    '''Anchor-relative supervision for a ``[batch, candidate]`` grid of candidate codes.'''

    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor

@dataclass(frozen=True)
class SupervisionIndex:
    '''Dense structural and directional-exclusion lookup over the bundle codebook.'''

    code_to_id: Dict[str, int]
    id_to_code: Tuple[str, ...]
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    directed_exclusion: torch.Tensor

    @classmethod
    def from_bundle(cls, bundle: ValidatedSupervisionBundle) -> 'SupervisionIndex':
        codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
        pair_facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
        size = codebook.height

        def column(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(pair_facts.get_column(name).to_numpy(), dtype=dtype)

        code_i = column('code_i_id', torch.long)
        code_j = column('code_j_id', torch.long)
        distance = torch.zeros((size, size), dtype=torch.float32)
        relation = torch.zeros((size, size), dtype=torch.int16)
        excludes = torch.zeros((size, size), dtype=torch.bool)
        distance_values = column('structural_distance', torch.float32)
        relation_values = column('structural_relation_id', torch.int16)
        distance[code_i, code_j] = distance_values
        distance[code_j, code_i] = distance_values
        relation[code_i, code_j] = relation_values
        relation[code_j, code_i] = relation_values
        excludes[code_i, code_j] = column('code_i_excludes_code_j', torch.bool)
        excludes[code_j, code_i] = column('code_j_excludes_code_i', torch.bool)

        codes = tuple(codebook.get_column('code').to_list())
        return cls(
            code_to_id={code: code_id for code_id, code in enumerate(codes)},
            id_to_code=codes,
            structural_distance=distance,
            structural_relation_id=relation,
            directed_exclusion=excludes,
        )

    def exclusion_code_ids(self, anchor_code_id: int) -> Tuple[int, ...]:
        '''Code IDs sharing an explicit exclusion with the anchor in either direction, sorted.'''

        if not 0 <= anchor_code_id < len(self.id_to_code):
            raise ValueError(f'unknown anchor code ID {anchor_code_id}')
        symmetric = (
            self.directed_exclusion[anchor_code_id] | self.directed_exclusion[:, anchor_code_id]
        )
        return tuple(torch.where(symmetric)[0].tolist())

    def join(
        self,
        anchor_code_ids: torch.Tensor,
        candidate_code_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PairSupervision:
        '''
        Join supervision for each anchor row against its candidate codes.

        Args:
            anchor_code_ids: ``[batch]`` anchor code IDs.
            candidate_code_ids: ``[batch, candidate]`` candidate code IDs (padding may be invalid).
            valid_mask: ``[batch, candidate]`` validity; invalid entries are never looked up.

        Raises:
            ValueError: On shape mismatch or an unknown anchor/valid candidate code ID.
        '''

        if candidate_code_ids.shape != valid_mask.shape:
            raise ValueError('candidate code IDs and valid mask must share shape')
        if anchor_code_ids.shape != (candidate_code_ids.shape[0], ):
            raise ValueError('one anchor code ID is required per candidate row')
        invalid_anchor = anchor_code_ids.lt(0) | anchor_code_ids.ge(len(self.id_to_code))
        if invalid_anchor.any():
            row = torch.where(invalid_anchor)[0][0].item()
            raise ValueError(
                f'anchor row {row} has unknown anchor code ID '
                f'{anchor_code_ids[row].item()}'
            )
        invalid = valid_mask & (
            candidate_code_ids.lt(0) | candidate_code_ids.ge(len(self.id_to_code))
        )
        if invalid.any():
            row, column = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            bad_id = candidate_code_ids[row, column].item()
            raise ValueError(f'anchor row {row} has unknown candidate code ID {bad_id}')

        valid_cpu = valid_mask.cpu()
        safe_ids = candidate_code_ids.masked_fill(~valid_mask, 0).cpu().long()
        anchor = anchor_code_ids.cpu().long().unsqueeze(1).expand_as(safe_ids)
        anchor_excludes = self.directed_exclusion[anchor, safe_ids] & valid_cpu
        candidate_excludes = self.directed_exclusion[safe_ids, anchor] & valid_cpu
        explicit = anchor_excludes | candidate_excludes
        target = torch.where(
            explicit,
            torch.tensor(TARGET_TO_ID[SemanticTarget.UNRELATED], dtype=torch.int8),
            torch.tensor(TARGET_TO_ID[SemanticTarget.UNKNOWN], dtype=torch.int8),
        )
        source = torch.where(
            explicit,
            torch.tensor(SOURCE_TO_ID[SemanticSource.EXPLICIT_EXCLUSION], dtype=torch.int8),
            torch.tensor(SOURCE_TO_ID[SemanticSource.UNLABELED], dtype=torch.int8),
        )
        device = candidate_code_ids.device
        return PairSupervision(
            structural_distance=self.structural_distance[anchor, safe_ids].to(device),
            structural_relation_id=self.structural_relation_id[anchor, safe_ids].to(device),
            anchor_excludes_candidate=anchor_excludes.to(device),
            candidate_excludes_anchor=candidate_excludes.to(device),
            is_explicit_exclusion=explicit.to(device),
            semantic_target_id=target.to(device),
            semantic_source_id=source.to(device),
        )
