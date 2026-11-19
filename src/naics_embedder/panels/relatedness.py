'''
The relatedness judging panel (spec ``relatedness-artifact``): Stage 10's co-ownership tables read
into the panel's gains.

Inputs
    The panel reads two of Stage 10's tables, ``ita_co_ownership`` and ``co_ownership_marginals``,
    and the bundle's six-digit codes. From the target rows it reads ``family``, ``measure``, the two
    unit types, the two units and ``n_unweighted``, which is J(a, b), the employers with
    establishments in both industries. From the marginal rows it reads ``node_id`` and
    ``n_multi_industry_employers``, which is m(a). Each node id has the form
    ``naics:2022:national_industry:<code>``. An input that disagrees with the spec is refused with a
    ``ValueError`` that names the table and the first offending row, never dropped. The checks run
    in a fixed order: a column that is missing or of another dtype, a null in a column read, a
    family, measure or unit type that is not accepted, a node id of another system, vintage or
    level, a code outside the codebook, a repeated marginal node, a target row whose industry has no
    marginal row, an industry paired with itself, a repeated ordered pair, a J below 1, a pair in
    one direction only, a pair whose two directions hold different J, and a J above m of either
    industry. A codebook entry that is not six digits is refused before any table is read. The gain
    of a candidate b for an anchor a is J(a, b) / m(b), and 0 when no row joins them.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import dataclasses
from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Mapping, Sequence, Tuple

import polars as pl

from naics_embedder.panels.ita import CODE_PATTERN, node_code

# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

# The columns of ``ita_co_ownership`` the panel reads, with Stage 10's dtypes. ``n_unweighted`` is J
TARGET_COLUMNS: Dict[str, pl.DataType] = {
    'family': pl.Utf8,
    'measure': pl.Utf8,
    'unit_a_type': pl.Utf8,
    'unit_a': pl.Utf8,
    'unit_b_type': pl.Utf8,
    'unit_b': pl.Utf8,
    'n_unweighted': pl.UInt64,
}

# The columns of ``co_ownership_marginals`` the panel reads. ``n_multi_industry_employers`` is m
MARGINAL_COLUMNS: Dict[str, pl.DataType] = {
    'node_id': pl.Utf8,
    'n_multi_industry_employers': pl.UInt64,
}

# The only family, measure and unit type a target row may hold
CO_OWNERSHIP_FAMILY = 'co_ownership'
CO_OWNERSHIP_MEASURE = 'employer_share'
INDUSTRY_UNIT = 'naics_national_industry'

# The two tables' names, which a refusal gives
TABLES = ('ita_co_ownership', 'co_ownership_marginals')
TARGETS_TABLE, MARGINALS_TABLE = TABLES

# A target column with one accepted value, and what a refusal calls it
_ACCEPTED = (
    ('family', 'family', CO_OWNERSHIP_FAMILY),
    ('measure', 'measure', CO_OWNERSHIP_MEASURE),
    ('unit_a_type', 'unit type', INDUSTRY_UNIT),
    ('unit_b_type', 'unit type', INDUSTRY_UNIT),
)

# The columns that name a target row and a marginal row in a refusal
_TARGET_ROW = ('unit_a', 'unit_b')
_MARGINAL_ROW = ('node_id', )

# One target row as the panel reads it: the codes of its two industries, then J
Pair = Tuple[str, str, int]

@dataclasses.dataclass(frozen=True)
class CoOwnership:
    '''
    Stage 10's co-ownership tables, read into the numbers a gain needs.

    Attributes:
        candidates: The six-digit codes that have a marginal row, sorted.
        multi: m for each candidate: its employers that also operate in another industry.
        shared: J for each ordered pair of codes with a target row. Both directions of a pair are
            keys, with one J.
    '''

    candidates: Tuple[str, ...]
    multi: Mapping[str, int]
    shared: Mapping[Tuple[str, str], int]

    @cached_property
    def _partner_index(self) -> Dict[str, Tuple[str, ...]]:
        partners: Dict[str, List[str]] = {}
        for code_a, code_b in self.shared:
            partners.setdefault(code_a, []).append(code_b)
        return {code: tuple(sorted(codes)) for code, codes in partners.items()}

    def partners(self, code: str) -> Tuple[str, ...]:
        '''
        The codes b that have a target row (``code``, b).

        Args:
            code: A six-digit code.

        Returns:
            The codes, sorted. The tuple is empty for a code with no row.
        '''

        return self._partner_index.get(code, ())

    def gain(self, anchor: str, candidate: str) -> float:
        '''
        The gain of a candidate for an anchor: of the candidate's multi-industry employers, the
        share that also operate in the anchor, J(anchor, candidate) / m(candidate).

        Args:
            anchor: The anchor's six-digit code.
            candidate: The candidate's six-digit code.

        Returns:
            The gain, in (0, 1] for a pair with a row and 0.0 for any other pair.
        '''

        shared = self.shared.get((anchor, candidate))
        return 0.0 if shared is None else shared / self.multi[candidate]

def _require_columns(table: pl.DataFrame, name: str, expected: Dict[str, pl.DataType]) -> None:
    for column, dtype in expected.items():
        if column not in table.columns:
            raise ValueError(f'{name} lacks column {column!r}')
        if table.schema[column] != dtype:
            raise ValueError(
                f'{name} column {column!r} has type {table.schema[column]}, expected {dtype}'
            )

def _row_text(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    return '(' + ', '.join(str(row[column]) for column in columns) + ')'

def _refuse_nulls(
    table: pl.DataFrame,
    name: str,
    columns: Sequence[str],
    row_columns: Sequence[str],
) -> None:
    '''Refuse the first row, in table order, with a null in one of the columns read.'''

    bad = table.filter(pl.any_horizontal([pl.col(column).is_null() for column in columns]))
    if not bad.is_empty():
        row = bad.row(0, named=True)
        column = next(column for column in columns if row[column] is None)
        raise ValueError(f'{name} has a null {column!r} in row {_row_text(row, row_columns)}')

def _refuse_unaccepted_values(targets: pl.DataFrame) -> None:
    '''Refuse the first row, in table order, whose family, measure or unit type is not accepted.'''

    bad = targets.filter(
        pl.any_horizontal([pl.col(column) != accepted for column, _, accepted in _ACCEPTED])
    )
    if not bad.is_empty():
        row = bad.row(0, named=True)
        column, label, accepted = next(
            (column, label, accepted) for column, label, accepted in _ACCEPTED
            if row[column] != accepted
        )
        raise ValueError(
            f'{TARGETS_TABLE} row {_row_text(row, _TARGET_ROW)} has {label} {row[column]!r} in '
            f'column {column!r}, which is not accepted: expected {accepted!r}'
        )

def _code(node_id: str, name: str, column: str, row: str = '') -> str:
    '''The code of a node id, or a refusal that names the table, the column and the row.'''

    try:
        return node_code(node_id)
    except ValueError as error:
        where = f'{name} row {row} column {column!r}' if row else f'{name} column {column!r}'
        raise ValueError(f'{where}: {error}') from None

def _pairs(targets: pl.DataFrame) -> List[Pair]:
    '''Each target row's two codes and J, in table order.'''

    pairs = []
    for unit_a, unit_b, j in targets.select('unit_a', 'unit_b', 'n_unweighted').iter_rows():
        row = f'({unit_a}, {unit_b})'
        pairs.append(
            (
                _code(unit_a, TARGETS_TABLE, 'unit_a', row),
                _code(unit_b, TARGETS_TABLE, 'unit_b', row),
                j,
            )
        )
    return pairs

def _nodes(marginals: pl.DataFrame) -> List[Tuple[str, int]]:
    '''Each marginal row's code and m, in table order.'''

    return [
        (_code(node_id, MARGINALS_TABLE, 'node_id'), m)
        for node_id, m in marginals.select('node_id', 'n_multi_industry_employers').iter_rows()
    ]

def _codebook(codebook_codes: Sequence[str]) -> FrozenSet[str]:
    for code in codebook_codes:
        if not CODE_PATTERN.fullmatch(code):
            raise ValueError(f'codebook_codes holds {code!r}, which is not a six-digit code')
    return frozenset(codebook_codes)

def _refuse_codes_outside(
    codebook: FrozenSet[str],
    pairs: Sequence[Pair],
    nodes: Sequence[Tuple[str, int]],
) -> None:
    for code_a, code_b, _ in pairs:
        for code in (code_a, code_b):
            if code not in codebook:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has {code}, '
                    'which is not a codebook code'
                )
    for code, _ in nodes:
        if code not in codebook:
            raise ValueError(f'{MARGINALS_TABLE} has {code}, which is not a codebook code')

def _marginal_counts(nodes: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    multi: Dict[str, int] = {}
    for code, m in nodes:
        if code in multi:
            raise ValueError(f'{MARGINALS_TABLE} repeats node {code}')
        multi[code] = m
    return multi

def _refuse_industries_without_marginals(pairs: Sequence[Pair], multi: Mapping[str, int]) -> None:
    for code_a, code_b, _ in pairs:
        for code in (code_a, code_b):
            if code not in multi:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has no marginal row for {code}'
                )

def _refuse_self_pairs(pairs: Sequence[Pair]) -> None:
    for code_a, code_b, _ in pairs:
        if code_a == code_b:
            raise ValueError(f'{TARGETS_TABLE} pairs {code_a} with itself')

def _shared_counts(pairs: Sequence[Pair]) -> Dict[Tuple[str, str], int]:
    shared: Dict[Tuple[str, str], int] = {}
    for code_a, code_b, j in pairs:
        if (code_a, code_b) in shared:
            raise ValueError(f'{TARGETS_TABLE} repeats the pair ({code_a}, {code_b})')
        shared[(code_a, code_b)] = j
    return shared

def _refuse_counts_below_one(shared: Mapping[Tuple[str, str], int]) -> None:
    for (code_a, code_b), j in shared.items():
        if j < 1:
            raise ValueError(f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, not at least 1')

def _refuse_one_way_pairs(shared: Mapping[Tuple[str, str], int]) -> None:
    for code_a, code_b in shared:
        if (code_b, code_a) not in shared:
            raise ValueError(
                f'{TARGETS_TABLE} has the pair ({code_a}, {code_b}) in one direction only: '
                f'no row ({code_b}, {code_a})'
            )

def _refuse_unequal_mirrors(shared: Mapping[Tuple[str, str], int]) -> None:
    for (code_a, code_b), j in shared.items():
        if shared[(code_b, code_a)] != j:
            raise ValueError(
                f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, but its mirror '
                f'({code_b}, {code_a}) has J {shared[(code_b, code_a)]}'
            )

def _refuse_counts_above_marginals(
    shared: Mapping[Tuple[str, str], int],
    multi: Mapping[str, int],
) -> None:
    for (code_a, code_b), j in shared.items():
        for code in (code_a, code_b):
            if j > multi[code]:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, which exceeds m of '
                    f'{code} ({multi[code]})'
                )

def co_ownership_inputs(
    targets: pl.DataFrame,
    marginals: pl.DataFrame,
    codebook_codes: Sequence[str],
) -> CoOwnership:
    '''
    Stage 10's co-ownership tables as the numbers a gain needs.

    Only the columns in ``TARGET_COLUMNS`` and ``MARGINAL_COLUMNS`` are read. Row order and other
    columns do not matter.

    Args:
        targets: ``ita_co_ownership``.
        marginals: ``co_ownership_marginals``.
        codebook_codes: The bundle's six-digit codes.

    Returns:
        The candidates (the codes with a marginal row), m for each, and J for each ordered pair.

    Raises:
        ValueError: When a codebook entry is not six digits. Then, in this order, when a column the
            panel reads is missing or of another dtype, a column it reads holds a null, a family,
            measure or unit type is not ``co_ownership``, ``employer_share`` or
            ``naics_national_industry``, a node id is of another system, vintage or level (see
            ``node_code``), a code is outside ``codebook_codes``, a marginal node repeats, a target
            row's industry has no marginal row, an industry is paired with itself, an ordered pair
            repeats, a J is below 1, a pair has a row in one direction only, a pair's two directions
            hold different J, or a J exceeds m of either industry. The message names the table and
            the first offending row.
    '''

    codebook = _codebook(codebook_codes)
    _require_columns(targets, TARGETS_TABLE, TARGET_COLUMNS)
    _require_columns(marginals, MARGINALS_TABLE, MARGINAL_COLUMNS)
    _refuse_nulls(targets, TARGETS_TABLE, list(TARGET_COLUMNS), _TARGET_ROW)
    _refuse_nulls(marginals, MARGINALS_TABLE, list(MARGINAL_COLUMNS), _MARGINAL_ROW)
    _refuse_unaccepted_values(targets)
    pairs = _pairs(targets)
    nodes = _nodes(marginals)
    _refuse_codes_outside(codebook, pairs, nodes)
    multi = _marginal_counts(nodes)
    _refuse_industries_without_marginals(pairs, multi)
    _refuse_self_pairs(pairs)
    shared = _shared_counts(pairs)
    _refuse_counts_below_one(shared)
    _refuse_one_way_pairs(shared)
    _refuse_unequal_mirrors(shared)
    _refuse_counts_above_marginals(shared, multi)
    return CoOwnership(candidates=tuple(sorted(multi)), multi=multi, shared=shared)
