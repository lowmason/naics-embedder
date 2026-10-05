'''
The fixed inputs of Req 11's code–code and radial terms (spec 4.1(ii), (iii)), read once.

For every code, in codebook order: its level λ, the number of its digits; the tree metric D* to
every code (Req 7); and its unary partner (Req 9), for a five-digit code and its only six-digit
child each other. The code–code term's J_a is every code but the anchor and its partner, so a
unary pair is neither a positive nor a negative.

The pair facts give D* and the unary flag. No exclusion column is read, so no exclusion pair can
act as a code–code negative (Req 8(c)).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import polars as pl

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle

# The only pair-fact columns read: never an exclusion column
PAIR_FACT_COLUMNS = ('code_i_id', 'code_j_id', 'structural_distance', 'unary_pair')
NO_PARTNER = -1

# -------------------------------------------------------------------------------------------------
# The targets
# -------------------------------------------------------------------------------------------------

# The __eq__ below compares by value: a generated one would compare the arrays as booleans, which
# numpy refuses. eq=False keeps the class unhashable: with eq=True, a frozen dataclass keeps that
# __eq__ but adds a __hash__ over the fields, which fails on the arrays.
@dataclass(frozen=True, eq=False)
class CodeTargets:
    '''
    Each code's level, D* to every code and unary partner, in codebook order.

    Built by ``from_bundle``. The arrays are fresh and writable, so ``torch.from_numpy`` can share
    them; nothing here writes to them.

    Attributes:
        codes: The codes, in codebook order.
        levels: λ, each code's number of digits, (N,) int64.
        structural_distance: D*, (N, N) float32, symmetric and zero on the diagonal only.
        unary_partner: Each code's unary partner's id, or -1 for none, (N,) int64.
    '''

    codes: Tuple[str, ...]
    levels: np.ndarray
    structural_distance: np.ndarray
    unary_partner: np.ndarray

    @classmethod
    def from_bundle(cls, bundle: ValidatedSupervisionBundle) -> 'CodeTargets':
        '''
        Read the codebook and the pair facts' ``PAIR_FACT_COLUMNS`` from a validated bundle.

        The loader has checked that the code ids run from zero in codebook order, that the pair
        facts hold every pair of distinct codes once with D* as their distance, and that the
        unary flag marks exactly each five-digit code and its only six-digit child.
        '''

        codebook = pl.read_parquet(bundle.artifact_path('codebook'), columns=['code_id', 'code'])
        codes = tuple(codebook.sort('code_id').get_column('code').to_list())
        facts = pl.read_parquet(bundle.artifact_path('pair_facts'), columns=list(PAIR_FACT_COLUMNS))
        code_i = facts.get_column('code_i_id').to_numpy()
        code_j = facts.get_column('code_j_id').to_numpy()
        distance = facts.get_column('structural_distance').to_numpy()
        unary = facts.get_column('unary_pair').to_numpy()

        structural = np.zeros((len(codes), len(codes)), dtype=np.float32)
        structural[code_i, code_j] = distance
        structural[code_j, code_i] = distance
        partner = np.full(len(codes), NO_PARTNER, dtype=np.int64)
        partner[code_i[unary]] = code_j[unary]
        partner[code_j[unary]] = code_i[unary]
        return cls(
            codes=codes,
            levels=np.array([len(code) for code in codes], dtype=np.int64),
            structural_distance=structural,
            unary_partner=partner,
        )

    def keep(self, ids: np.ndarray) -> np.ndarray:
        '''
        J_a for each anchor: every code but the anchor itself and its unary partner.

        Args:
            ids: The anchors' code ids, (A,), an integer array (a CPU tensor converts).

        Returns:
            A fresh bool mask over the codes, (A, N).

        Raises:
            ValueError: If ``ids`` is not a one-dimensional integer array of code ids.
        '''

        ids = np.asarray(ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                'anchor ids must be a one-dimensional integer array, not '
                f'{ids.dtype} {ids.shape}'
            )
        outside = (ids < 0) | (ids >= len(self.codes))
        if outside.any():
            raise ValueError(
                f'anchor ids must be code ids in 0..{len(self.codes) - 1}, not '
                f'{sorted(set(ids[outside].tolist()))}'
            )
        rows = np.arange(len(ids))
        keep = np.ones((len(ids), len(self.codes)), dtype=bool)
        keep[rows, ids] = False
        partners = self.unary_partner[ids]
        paired = partners != NO_PARTNER
        keep[rows[paired], partners[paired]] = False
        return keep

    def __eq__(self, other: object) -> bool:
        '''Equal codes, and arrays of equal dtype and values.'''

        if not isinstance(other, CodeTargets):
            return NotImplemented
        arrays = (
            (self.levels, other.levels),
            (self.structural_distance, other.structural_distance),
            (self.unary_partner, other.unary_partner),
        )
        return self.codes == other.codes and all(
            mine.dtype == theirs.dtype and np.array_equal(mine, theirs) for mine, theirs in arrays
        )
