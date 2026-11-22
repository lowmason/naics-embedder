'''
The arms of the relatedness judging panel (spec ``relatedness-artifact``, 4.4 and 4.5).

An arm gives the similarity of every anchor to every candidate, read by six-digit code. Two adapters
cover the arms the panel reads:

- ``TreeArm`` wraps the tree metric. Its similarity is -D*, read by code, so the codebook's row
  order does not matter.
- ``VectorArm`` wraps a vector table. Its similarity is the float64 cosine. With ``center`` it first
  subtracts the mean of the table's six-digit rows from every row (plan 21, P7). That mean is fixed
  at construction, so a value does not depend on which codes a read asks for. A zero vector has no
  cosine, so a read of one is refused, never given a NaN.

A request names codes the arm does not hold, or a zero-norm row, with a ``ValueError`` that names the
first five and how many there are. No refusal drops a code.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from collections import Counter
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.text_only import text_only_matrix
from naics_embedder.supervision.code_targets import CodeTargets

# A refusal names this many codes, then says how many more there are
LISTED_CODES = 5

# The number of digits of a code that centers a table
SIX_DIGITS = 6

# -------------------------------------------------------------------------------------------------
# The protocol
# -------------------------------------------------------------------------------------------------

class RelatednessArm(Protocol):
    '''What the panel asks of an arm: the similarity of each anchor to each candidate.'''

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        '''
        The similarity of every anchor to every candidate.

        Args:
            anchors: The codes of the matrix's rows.
            candidates: The codes of its columns.

        Returns:
            A ``(len(anchors), len(candidates))`` float64 matrix.
        '''

        ...

# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def _listing(codes: Sequence[str]) -> str:
    '''The first ``LISTED_CODES`` codes, then how many more there are.'''

    listed = ', '.join(codes[:LISTED_CODES])
    more = len(codes) - LISTED_CODES
    return f'{listed} and {more} more' if more > 0 else listed

def _distinct(codes: Sequence[str]) -> List[str]:
    '''The codes without a repeat, in the order each first appears.'''

    return list(dict.fromkeys(codes))

def _require_distinct(codes: Sequence[str]) -> None:
    repeated = [code for code, count in Counter(codes).items() if count > 1]
    if repeated:
        raise ValueError(
            f'codes must be distinct, repeated ({len(repeated)}): {_listing(repeated)}'
        )

def _rows_of(
    row_of: Dict[str, int],
    anchors: Sequence[str],
    candidates: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    The table rows of the anchors and of the candidates.

    Raises:
        ValueError: If a code is not in the table. It names the first five missing codes, anchors
            before candidates and each once, and how many there are.
    '''

    missing = _distinct([code for code in (*anchors, *candidates) if code not in row_of])
    if missing:
        raise ValueError(f'requested codes not in the arm ({len(missing)}): {_listing(missing)}')
    return (
        np.array([row_of[code] for code in anchors], dtype=np.intp),
        np.array([row_of[code] for code in candidates], dtype=np.intp),
    )

# -------------------------------------------------------------------------------------------------
# The tree arm
# -------------------------------------------------------------------------------------------------

class TreeArm:
    '''
    The tree metric as an arm: the similarity of two codes is -D*.

    Args:
        codes: The codes of the matrix's rows and columns, each once.
        structural_distance: D* between every two of ``codes``, ``(N, N)`` for N codes. It is
            read as float64, whatever its type.

    Raises:
        ValueError: If a code is repeated or the matrix is not ``(N, N)``.
    '''

    def __init__(self, codes: Sequence[str], structural_distance: np.ndarray):
        codes = tuple(codes)
        _require_distinct(codes)
        distance = np.array(structural_distance, dtype=np.float64)
        if distance.shape != (len(codes), len(codes)):
            raise ValueError(
                f'the distance matrix must be {len(codes)} by {len(codes)} for {len(codes)} '
                f'codes, not {distance.shape}'
            )
        self._row_of = {code: row for row, code in enumerate(codes)}
        self._similarity = -distance

    @classmethod
    def from_targets(cls, targets: CodeTargets) -> 'TreeArm':
        '''The arm over the bundle's codes and its D*, as ``CodeTargets`` reads them.'''

        return cls(targets.codes, targets.structural_distance)

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        '''
        -D* of every anchor to every candidate.

        Args:
            anchors: The codes of the rows. An anchor may also be a candidate: its own cell is 0.
            candidates: The codes of the columns.

        Returns:
            A ``(len(anchors), len(candidates))`` float64 matrix.

        Raises:
            ValueError: If a code is not in the arm. It names the first five missing codes, anchors
                before candidates and each once, and how many there are.
        '''

        rows, columns = _rows_of(self._row_of, anchors, candidates)
        return self._similarity[np.ix_(rows, columns)]

# -------------------------------------------------------------------------------------------------
# The vector arm
# -------------------------------------------------------------------------------------------------

class VectorArm:
    '''
    Cosine over a vector table as an arm, plain or centered.

    The table is checked and, with ``center``, centered once, here. Each row becomes a unit vector,
    so a pair's cosine is a dot product that reads only the pair's two rows.

    Args:
        codes: The codes of the table's rows, each once.
        vectors: One row per code, ``(N, d)``.
        center: Subtract the mean of the rows whose code has six digits from every row first (P7).
            The mean is over the table, never over the codes a read asks for.

    Raises:
        ValueError: If a code is repeated, the table is not ``(N, d)``, a value is not finite (it
            names the codes of those rows), or ``center`` is set and no code has six digits.
    '''

    def __init__(self, codes: Sequence[str], vectors: np.ndarray, *, center: bool = False):
        codes = tuple(codes)
        _require_distinct(codes)
        matrix = np.array(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(codes):
            raise ValueError(
                f'the vectors must have one row for each of {len(codes)} codes, not {matrix.shape}'
            )
        unfit = [code for code, fits in zip(codes, np.isfinite(matrix).all(axis=1)) if not fits]
        if unfit:
            raise ValueError(
                f'the vectors must be finite, not finite for ({len(unfit)}): {_listing(unfit)}'
            )
        if center:
            six_digit = np.array([len(code) == SIX_DIGITS for code in codes], dtype=bool)
            if not six_digit.any():
                raise ValueError('centering needs at least one six-digit code in the table')
            matrix = matrix - matrix[six_digit].mean(axis=0)
        norms = np.linalg.norm(matrix, axis=1)
        self._row_of = {code: row for row, code in enumerate(codes)}
        self._is_zero = norms == 0.0
        # A zero row stays zero, and a read that names it is refused
        self._unit = matrix / np.where(self._is_zero, 1.0, norms)[:, None]

    @classmethod
    def from_text_only(cls, table: pl.DataFrame, *, center: bool = False) -> 'VectorArm':
        '''The arm over a text-only table: its ``code`` column and its ``t`` columns.'''

        codes, matrix = text_only_matrix(table)
        return cls(codes, matrix, center=center)

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        '''
        The cosine of every anchor to every candidate, in float64.

        Each cell is a sum over the vector's width that reads the pair's two rows alone, so a cell
        does not depend on what else the request names.

        Args:
            anchors: The codes of the rows.
            candidates: The codes of the columns.

        Returns:
            A ``(len(anchors), len(candidates))`` float64 matrix.

        Raises:
            ValueError: If a code is not in the arm, or a requested row has zero norm (after
                centering, with ``center``). Each names the first five codes, anchors before
                candidates and each once, and how many there are.
        '''

        rows, columns = _rows_of(self._row_of, anchors, candidates)
        zero = _distinct(
            [
                code for code, row in zip((*anchors, *candidates), (*rows, *columns))
                if self._is_zero[row]
            ]
        )
        if zero:
            raise ValueError(
                f'a zero vector has no cosine, requested ({len(zero)}): {_listing(zero)}'
            )
        unit_columns = self._unit[columns]
        similarity = np.empty((len(rows), len(columns)), dtype=np.float64)
        for position, row in enumerate(rows):
            similarity[position] = (unit_columns * self._unit[row]).sum(axis=1)
        return similarity
