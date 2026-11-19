'''
Stage 10's co-ownership tables read into gains (plan 21, Task 2; spec relatedness-artifact, 4.2 and
4.4).

The base tables give the fixture's gains, and each input that disagrees with the spec is refused with
a ``ValueError`` that names the table and the first offending row. The refusals come in a fixed order
(the brief's 13 checks), which the order tests pin with pairs of defects. Every table here is built by
``tests/fixtures/relatedness.py``, and no test reads a real table.
'''

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import polars as pl
import pytest

from naics_embedder.panels.relatedness import (
    CO_OWNERSHIP_FAMILY,
    CO_OWNERSHIP_MEASURE,
    INDUSTRY_UNIT,
    MARGINAL_COLUMNS,
    TABLES,
    TARGET_COLUMNS,
    CoOwnership,
    co_ownership_inputs,
)
from tests.fixtures.relatedness import (
    BASE_MULTI,
    BASE_PAIRS,
    RELATEDNESS_CODES,
    co_ownership_tables,
    node,
)

Tables = Tuple[pl.DataFrame, pl.DataFrame]

# The two tables' names in a refusal
TARGETS = 'ita_co_ownership'
MARGINALS = 'co_ownership_marginals'

# -------------------------------------------------------------------------------------------------
# Building a table with one defect
# -------------------------------------------------------------------------------------------------

def row_of(code_a: str, code_b: str) -> pl.Expr:
    '''The target row from one industry to another.'''

    return (pl.col('unit_a') == node(code_a)) & (pl.col('unit_b') == node(code_b))

def edit(table: pl.DataFrame, where: pl.Expr, **values) -> pl.DataFrame:
    '''The table with the named columns set on the rows ``where`` selects.'''

    return table.with_columns(
        [
            pl.when(where).then(pl.lit(value, dtype=table.schema[column])).otherwise(
                pl.col(column)
            ).alias(column) for column, value in values.items()
        ]
    )

def with_row(targets: pl.DataFrame, code_a: str, code_b: str, j: int) -> pl.DataFrame:
    '''One more target row, with the first row's metadata.'''

    row = edit(
        targets.head(1), pl.lit(True), unit_a=node(code_a), unit_b=node(code_b), n_unweighted=j
    )
    return pl.concat([targets, row])

def with_marginal(marginals: pl.DataFrame, node_id: str, m: int) -> pl.DataFrame:
    '''One more marginal row, with the first row's other columns.'''

    row = edit(marginals.head(1), pl.lit(True), node_id=node_id, n_multi_industry_employers=m)
    return pl.concat([marginals, row])

@dataclass
class Scenario:
    '''
    The base tables, changed. Pair-level changes go to ``pairs`` and ``multi``, which the fixture
    builds in both directions. Row-level changes are ``edits``, applied to the built tables in order.
    '''

    pairs: List[Tuple[str, str, int]] = field(default_factory=lambda: list(BASE_PAIRS))
    multi: Dict[str, int] = field(default_factory=lambda: dict(BASE_MULTI))
    codebook: Tuple[str, ...] = RELATEDNESS_CODES
    edits: List[Callable[[Tables], Tables]] = field(default_factory=list)

    def targets(self, change: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
        self.edits.append(lambda tables: (change(tables[0]), tables[1]))

    def marginals(self, change: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
        self.edits.append(lambda tables: (tables[0], change(tables[1])))

    def set_j(self, code_a: str, code_b: str, j: int) -> None:
        self.pairs = [(a, b, j if (a, b) == (code_a, code_b) else old) for a, b, old in self.pairs]

    def tables(self) -> Tables:
        tables = co_ownership_tables(self.pairs, self.multi)
        for change in self.edits:
            tables = change(tables)
        return tables

    def read(self) -> CoOwnership:
        targets, marginals = self.tables()
        return co_ownership_inputs(targets, marginals, self.codebook)

# -------------------------------------------------------------------------------------------------
# The brief's cases
# -------------------------------------------------------------------------------------------------

def missing_column(s: Scenario) -> None:
    s.targets(lambda t: t.drop('n_unweighted'))

def wrong_dtype(s: Scenario) -> None:
    s.targets(lambda t: t.cast({'n_unweighted': pl.Int64}))

def null(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311211', '311212'), unit_b=None))

def family(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311211', '311212'), family='co_location'))

def measure(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311211', '311212'), measure='employer_count'))

def unit_type(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311211', '311212'), unit_b_type='naics_industry'))

def node_level(s: Scenario) -> None:
    s.marginals(lambda m: with_marginal(m, 'naics:2022:industry:31121', 1))

def node_vintage(s: Scenario) -> None:
    s.targets(
        lambda t: edit(t, row_of('311211', '311212'), unit_a='naics:2017:national_industry:311211')
    )

def not_in_codebook(s: Scenario) -> None:
    s.pairs.append(('311211', '999999', 1))
    s.multi['999999'] = 1

def repeated_marginal(s: Scenario) -> None:
    s.marginals(lambda m: pl.concat([m, m.head(1)]))

def no_marginal(s: Scenario) -> None:
    s.pairs.append(('211120', '311211', 1))

def self_pair(s: Scenario) -> None:
    s.targets(lambda t: with_row(t, '311211', '311211', 1))

def repeated_pair(s: Scenario) -> None:
    s.targets(lambda t: pl.concat([t, t.head(1)]))

def zero_j(s: Scenario) -> None:
    s.set_j('311211', '311212', 0)

def one_way(s: Scenario) -> None:
    s.targets(lambda t: t.filter(~row_of('311212', '311211')))

def unequal_mirror(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311212', '311211'), n_unweighted=2))

def j_above_m(s: Scenario) -> None:
    s.set_j('311211', '311212', 4)

@dataclass(frozen=True)
class Case:
    '''
    A defect and its refusal.

    Attributes:
        id: The test id.
        defect: Changes a ``Scenario`` so that its tables disagree with the spec.
        match: A regular expression the message holds.
        named: Fragments the message holds: the table and the codes of the first offending row.
        absent: Fragments the message does not hold.
    '''

    id: str
    defect: Callable[[Scenario], None]
    match: str
    named: Tuple[str, ...] = ()
    absent: Tuple[str, ...] = ()

CASES = [
    Case('missing-column', missing_column, "lacks column 'n_unweighted'", (TARGETS, )),
    Case('wrong-dtype', wrong_dtype, 'has type', (TARGETS, "'n_unweighted'", 'Int64')),
    Case('null', null, 'null', (TARGETS, "'unit_b'", '311211')),
    Case('family', family, 'family', (TARGETS, '311211', '311212', 'co_location')),
    Case('measure', measure, 'measure', (TARGETS, '311211', '311212', 'employer_count')),
    Case('unit-type', unit_type, 'unit type', (TARGETS, '311211', '311212', 'naics_industry')),
    Case('node-level', node_level, 'node id', (MARGINALS, 'naics:2022:industry:31121')),
    Case(
        'node-vintage', node_vintage, 'node id', (
            TARGETS, "'unit_a'", 'naics:2017:national_industry:311211', '311212'
        )
    ),
    Case('not-in-codebook', not_in_codebook, 'not a codebook', (TARGETS, '311211', '999999')),
    Case('repeated-marginal', repeated_marginal, 'repeats', (MARGINALS, '111110')),
    Case('no-marginal', no_marginal, 'no marginal row', (TARGETS, '211120')),
    Case('self-pair', self_pair, 'itself', (TARGETS, '311211')),
    Case('repeated-pair', repeated_pair, 'repeats', (TARGETS, '111110', '111120')),
    Case('zero-j', zero_j, 'at least 1', (TARGETS, '(311211, 311212) has J 0')),
    Case('one-way', one_way, 'one direction', (TARGETS, '(311211, 311212) in one direction')),
    Case(
        'unequal-mirror', unequal_mirror, 'mirror', (
            TARGETS, '(311211, 311212) has J 3', '(311212, 311211) has J 2'
        )
    ),
    Case(
        'j-above-m', j_above_m, 'exceeds', (
            TARGETS, '(311211, 311212) has J 4', 'exceeds m of 311212 (3)'
        )
    ),
]

# -------------------------------------------------------------------------------------------------
# The base tables
# -------------------------------------------------------------------------------------------------

def test_the_base_tables_give_the_fixture_gains():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert co.candidates == tuple(sorted(BASE_MULTI))
    assert co.gain('311211', '311212') == 1.0  # J 3 / m(311212) 3
    assert co.gain('311212', '311211') == pytest.approx(0.6)  # J 3 / m(311211) 5
    assert co.gain('311211', '111120') == pytest.approx(1 / 3)
    assert co.gain('311211', '445131') == 0.0
    assert co.partners('445131') == ()
    assert set(co.shared) == {(a, b) for a, b, _ in BASE_PAIRS} | {(b, a) for a, b, _ in BASE_PAIRS}

def test_codebook_codes_without_a_marginal_row_are_not_candidates():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert '211120' not in co.candidates and '445131' in co.candidates

def test_row_order_and_extra_columns_do_not_matter():
    targets, marginals = co_ownership_tables()
    base = co_ownership_inputs(targets, marginals, RELATEDNESS_CODES)
    shuffled_targets = targets.sample(fraction=1.0, shuffle=True,
                                      seed=7).with_columns(pl.lit('x').alias('extra'))
    shuffled_marginals = marginals.sample(fraction=1.0, shuffle=True,
                                          seed=11).with_columns(pl.lit(1).alias('extra'))
    assert shuffled_targets.get_column('unit_a').to_list() != targets.get_column('unit_a').to_list()
    assert shuffled_marginals.get_column('node_id').to_list() != (
        marginals.get_column('node_id').to_list()
    )
    shuffled = co_ownership_inputs(shuffled_targets, shuffled_marginals, RELATEDNESS_CODES)
    assert shuffled == base
    assert shuffled.candidates == tuple(sorted(BASE_MULTI))
    for code in RELATEDNESS_CODES:
        assert shuffled.partners(code) == base.partners(code)

@pytest.mark.parametrize('case', CASES, ids=[case.id for case in CASES])
def test_an_input_that_disagrees_with_the_spec_is_refused(case):
    scenario = Scenario()
    case.defect(scenario)
    with pytest.raises(ValueError, match=case.match):
        scenario.read()

# -------------------------------------------------------------------------------------------------
# Added by the mutant step: input shapes the plan's tests above do not use
# -------------------------------------------------------------------------------------------------

def expected_shared() -> Dict[Tuple[str, str], int]:
    '''J for each ordered pair of the base fixture, from the pairs and not from the tables.'''

    return {**{(a, b): j for a, b, j in BASE_PAIRS}, **{(b, a): j for a, b, j in BASE_PAIRS}}

def test_the_module_names_the_spec_values():
    assert TABLES == ('ita_co_ownership', 'co_ownership_marginals')
    assert CO_OWNERSHIP_FAMILY == 'co_ownership'
    assert CO_OWNERSHIP_MEASURE == 'employer_share'
    assert INDUSTRY_UNIT == 'naics_national_industry'
    assert TARGET_COLUMNS == {
        'family': pl.Utf8,
        'measure': pl.Utf8,
        'unit_a_type': pl.Utf8,
        'unit_a': pl.Utf8,
        'unit_b_type': pl.Utf8,
        'unit_b': pl.Utf8,
        'n_unweighted': pl.UInt64,
    }
    assert MARGINAL_COLUMNS == {'node_id': pl.Utf8, 'n_multi_industry_employers': pl.UInt64}

def test_the_base_tables_give_every_ordered_pair_its_j_and_every_code_its_m():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert dict(co.shared) == expected_shared()
    assert dict(co.multi) == BASE_MULTI
    assert all(type(j) is int for j in co.shared.values())

def test_a_gain_is_j_over_the_candidates_m_never_the_anchors():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    for a, b, j in BASE_PAIRS:
        assert co.gain(a, b) == j / BASE_MULTI[b]
        assert co.gain(b, a) == j / BASE_MULTI[a]
        assert 0 < co.gain(a, b) <= 1 and 0 < co.gain(b, a) <= 1

def test_a_gain_is_zero_for_every_pair_without_a_row():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    shared = expected_shared()
    for a in RELATEDNESS_CODES:
        for b in RELATEDNESS_CODES:
            if (a, b) not in shared:
                gain = co.gain(a, b)
                assert gain == 0.0 and isinstance(gain, float), (a, b)

def test_partners_are_the_codes_with_a_row_from_the_anchor_sorted():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert co.partners('111110') == ('111120', '311111', '445110')
    for code in RELATEDNESS_CODES:
        expected = tuple(sorted(b for a, b in expected_shared() if a == code))
        assert co.partners(code) == expected and isinstance(co.partners(code), tuple)

def test_candidates_are_the_codes_with_a_marginal_row_sorted():
    targets, marginals = co_ownership_tables()
    shuffled = marginals.sample(fraction=1.0, shuffle=True, seed=3)
    co = co_ownership_inputs(targets, shuffled, RELATEDNESS_CODES)
    assert co.candidates == tuple(sorted(BASE_MULTI))
    assert list(co.candidates) == sorted(co.candidates) and len(co.candidates) == 14

def test_a_marginal_row_with_no_pair_is_a_candidate_with_no_partner():
    scenario = Scenario(pairs=[], multi={'311211': 2, '311212': 1})
    co = scenario.read()
    assert co.candidates == ('311211', '311212')
    assert dict(co.shared) == {} and co.partners('311211') == ()
    assert co.gain('311211', '311212') == 0.0

def test_an_empty_target_table_is_read_and_an_empty_marginal_table_has_no_candidates():
    targets, marginals = co_ownership_tables()
    co = co_ownership_inputs(targets.head(0), marginals, RELATEDNESS_CODES)
    assert dict(co.shared) == {} and co.candidates == tuple(sorted(BASE_MULTI))
    empty = co_ownership_inputs(targets.head(0), marginals.head(0), RELATEDNESS_CODES)
    assert empty.candidates == () and dict(empty.multi) == {}

def test_a_codebook_that_lacks_a_tables_code_refuses_it():
    codebook = tuple(code for code in RELATEDNESS_CODES if code != '621210')
    with pytest.raises(ValueError, match='not a codebook') as refusal:
        co_ownership_inputs(*co_ownership_tables(), codebook)
    assert '621210' in str(refusal.value)

@pytest.mark.parametrize(
    'bad', [
        '31121', '3112111', 'abcdef', '31121a', '', ' 311211', 'naics:2022:national_industry:311211'
    ]
)
def test_a_codebook_entry_that_is_not_six_digits_is_refused(bad):
    codebook = (*RELATEDNESS_CODES, bad)
    with pytest.raises(ValueError, match='not a six-digit code') as refusal:
        co_ownership_inputs(*co_ownership_tables(), codebook)
    assert repr(bad) in str(refusal.value)

def test_a_co_ownership_is_frozen():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        co.candidates = ()  # type: ignore[misc]

def test_a_null_in_a_column_the_panel_does_not_read_is_ignored():
    targets, marginals = co_ownership_tables()
    base = co_ownership_inputs(targets, marginals, RELATEDNESS_CODES)
    assert targets.get_column('se').null_count() == targets.height
    nulled = targets.with_columns(
        pl.lit(None, dtype=pl.Float64).alias('n_weighted'),
        pl.lit(None, dtype=pl.Utf8).alias('source')
    )
    marginals = marginals.with_columns(pl.lit(None, dtype=pl.UInt64).alias('n_employers'))
    assert co_ownership_inputs(nulled, marginals, RELATEDNESS_CODES) == base

# -------------------------------------------------------------------------------------------------
# Each refusal names its table and the first offending row
# -------------------------------------------------------------------------------------------------

def mirrored(code_a: str, code_b: str, **values) -> Callable[[Scenario], None]:
    '''A defect that sets columns on the one row from ``code_a`` to ``code_b``.'''

    return lambda s: s.targets(lambda t: edit(t, row_of(code_a, code_b), **values))

def null_marginal(column: str) -> Callable[[Scenario], None]:
    return lambda s: s.marginals(
        lambda m: edit(m,
                       pl.col('node_id') == node('311211'), **{column: None})
    )

def dropping(column: str, marginal: bool) -> Callable[[Scenario], None]:
    if marginal:
        return lambda s: s.marginals(lambda m: m.drop(column))
    return lambda s: s.targets(lambda t: t.drop(column))

def recast(column: str, dtype: pl.DataType, marginal: bool) -> Callable[[Scenario], None]:
    if marginal:
        return lambda s: s.marginals(lambda m: m.cast({column: dtype}))
    return lambda s: s.targets(lambda t: t.cast({column: dtype}))

def a_late_family_and_an_early_measure(s: Scenario) -> None:
    s.targets(
        lambda t: edit(
            edit(t, row_of('311211', '311212'), measure='employer_count'),
            row_of('312111', '312112'),
            family='co_location',
        )
    )

def an_early_null_count_and_a_late_null_unit(s: Scenario) -> None:
    s.targets(
        lambda t: edit(
            edit(t, row_of('311211', '311212'), n_unweighted=None),
            row_of('312111', '312112'),
            unit_a=None,
        )
    )

def a_repeated_pair_with_another_j(s: Scenario) -> None:
    s.targets(lambda t: pl.concat([t, edit(t.head(1), pl.lit(True), n_unweighted=3)]))

def a_marginal_with_another_m_repeated(s: Scenario) -> None:
    s.marginals(
        lambda m: pl.concat([m, edit(m.head(1), pl.lit(True), n_multi_industry_employers=9)])
    )

def an_unmarked_unit_b(s: Scenario) -> None:
    '''The row (311211, 211120) only: 211120 has no marginal row and is only a unit_b.'''

    s.targets(lambda t: with_row(t, '311211', '211120', 1))

def an_unmarked_unit_a(s: Scenario) -> None:
    s.targets(lambda t: with_row(t, '211120', '311211', 1))

def a_unit_b_outside_the_codebook(s: Scenario) -> None:
    s.targets(lambda t: with_row(t, '311211', '999999', 1))

def a_unit_a_outside_the_codebook(s: Scenario) -> None:
    s.targets(lambda t: with_row(t, '999999', '311211', 1))

def a_marginal_outside_the_codebook(s: Scenario) -> None:
    s.marginals(lambda m: with_marginal(m, node('999999'), 1))

def a_pair_in_one_direction_from_the_smaller_code(s: Scenario) -> None:
    s.targets(lambda t: t.filter(~row_of('311211', '311212')))

def a_j_of_zero_on_one_row(s: Scenario) -> None:
    s.targets(lambda t: edit(t, row_of('311212', '311211'), n_unweighted=0))

def a_j_above_the_smaller_codes_m(s: Scenario) -> None:
    s.set_j('111150', '311221', 3)  # m(111150) 2, m(311221) 3

def a_j_over_a_zero_m(s: Scenario) -> None:
    s.pairs.append(('311211', '445131', 1))  # m(445131) 0

def a_self_pair_of_a_zero_m_industry(s: Scenario) -> None:
    s.targets(lambda t: with_row(t, '445131', '445131', 1))

def a_bad_node_id_on_unit_b(s: Scenario) -> None:
    s.targets(
        lambda t: edit(t, row_of('311211', '311212'), unit_b='naics:2022:national_industry:31121')
    )

NODE_IDS = [
    'sic:2022:national_industry:311211',
    'naics:2017:national_industry:311211',
    'naics:2022:industry:311211',
    'naics:2022:subsector:311',
    'naics:2022:national_industry:31121',
    'naics:2022:national_industry:31121a',
    'naics:2022:national_industry:311211:extra',
    'naics:2022:national_industry:',
    '311211',
    '',
]

MORE_CASES = [
    # a column that is missing or of another dtype, in either table
    *[
        Case(
            f'missing-{table}-{column}',
            dropping(column, table == MARGINALS),
            f'lacks column {column!r}',
            (table, f"'{column}'"),
        ) for table, columns in ((TARGETS, TARGET_COLUMNS), (MARGINALS, MARGINAL_COLUMNS))
        for column in columns
    ],
    *[
        Case(
            f'dtype-{table}-{column}',
            recast(column, pl.Categorical if dtype == pl.Utf8 else pl.Int64, table == MARGINALS),
            'has type',
            (table, f"'{column}'"),
        ) for table, columns in ((TARGETS, TARGET_COLUMNS), (MARGINALS, MARGINAL_COLUMNS))
        for column, dtype in columns.items()
    ],
    # a null in any column it reads, in either table; the first row in table order is named
    *[
        Case(
            f'null-{column}',
            mirrored('311211', '311212', **{column: None}),
            'has a null',
            (TARGETS, f"'{column}'"),
        ) for column in TARGET_COLUMNS
    ],
    *[
        Case(
            f'null-marginal-{column}',
            null_marginal(column),
            'has a null',
            (MARGINALS, f"'{column}'"),
        ) for column in MARGINAL_COLUMNS
    ],
    Case(
        'null-first-row-in-table-order',
        an_early_null_count_and_a_late_null_unit,
        'has a null',
        (TARGETS, "'n_unweighted'", '311211', '311212'),
        ("'unit_a'", ),
    ),
    # a family, measure or unit type other than the accepted ones, in each column
    *[
        Case(
            f'{column}-{bad}',
            mirrored('311211', '311212', **{column: bad}),
            f'has {label} {bad!r}',
            (TARGETS, f"'{column}'", '311211', '311212'),
        ) for column, label, bad in (
            ('family', 'family', 'co_location'),
            ('family', 'family', 'Co_Ownership'),
            ('measure', 'measure', 'employer_count'),
            ('unit_a_type', 'unit type', 'naics_industry'),
            ('unit_b_type', 'unit type', 'naics_industry'),
        )
    ],
    Case(
        'first-bad-row-in-table-order',
        a_late_family_and_an_early_measure,
        'has measure',
        (TARGETS, '311211', '311212'),
        ('co_location', "'family'"),
    ),
    # a node id of another system, vintage or level, in each column that holds one
    *[
        Case(
            f'node-id-{column}-{bad}',
            mirrored('311211', '311212', **{column: bad}),
            'node id',
            (TARGETS, f"'{column}'", repr(bad), '311212' if column == 'unit_a' else '311211'),
        ) for column in ('unit_a', 'unit_b') for bad in NODE_IDS
    ],
    *[
        Case(
            f'node-id-marginal-{bad}',
            lambda s, bad=bad: s.marginals(lambda m: with_marginal(m, bad, 1)),
            'node id',
            (MARGINALS, repr(bad)),
        ) for bad in NODE_IDS
    ],
    # a code outside the codebook, in each place it can be found
    Case('codebook-unit-a', a_unit_a_outside_the_codebook, 'not a codebook', (TARGETS, '999999')),
    Case('codebook-unit-b', a_unit_b_outside_the_codebook, 'not a codebook', (TARGETS, '999999')),
    Case(
        'codebook-marginal', a_marginal_outside_the_codebook, 'not a codebook', (
            MARGINALS, '999999'
        )
    ),
    # a repeated marginal node, whatever its m
    Case(
        'repeated-marginal-with-another-m', a_marginal_with_another_m_repeated, 'repeats node', (
            MARGINALS, '111110'
        )
    ),
    # a target row whose industry has no marginal row, as unit_a and as unit_b
    Case('no-marginal-unit-a', an_unmarked_unit_a, 'no marginal row', (TARGETS, '211120')),
    Case('no-marginal-unit-b', an_unmarked_unit_b, 'no marginal row', (TARGETS, '211120')),
    # an industry paired with itself
    Case(
        'self-pair-of-an-industry-with-m-zero', a_self_pair_of_a_zero_m_industry, 'itself', (
            TARGETS, '445131'
        )
    ),
    # a repeated ordered pair is neither summed nor won by its last row
    Case(
        'repeated-pair-with-another-j', a_repeated_pair_with_another_j, 'repeats the pair', (
            TARGETS, '111110', '111120'
        )
    ),
    # a row with J below 1
    Case('zero-j-on-one-row', a_j_of_zero_on_one_row, 'at least 1', (TARGETS, '311211', '311212')),
    # a pair in one direction only, in each direction
    Case(
        'one-way-from-the-smaller-code', a_pair_in_one_direction_from_the_smaller_code,
        'one direction', (TARGETS, '311211', '311212')
    ),
    # a pair whose J exceeds m of either industry
    Case(
        'j-above-the-smaller-codes-m', a_j_above_the_smaller_codes_m, 'exceeds', (
            TARGETS, '(111150, 311221) has J 3', 'exceeds m of 111150 (2)'
        )
    ),
    Case(
        'j-over-a-zero-m', a_j_over_a_zero_m, 'exceeds', (
            TARGETS, '(311211, 445131) has J 1', 'exceeds m of 445131 (0)'
        )
    ),
]

@pytest.mark.parametrize('case', [*CASES, *MORE_CASES], ids=lambda case: case.id)
def test_a_refusal_names_the_table_and_the_first_offending_row(case):
    scenario = Scenario()
    case.defect(scenario)
    with pytest.raises(ValueError, match=case.match) as refusal:
        scenario.read()
    message = str(refusal.value)
    for fragment in case.named:
        assert fragment in message, (fragment, message)
    for fragment in case.absent:
        assert fragment not in message, (fragment, message)

# -------------------------------------------------------------------------------------------------
# The refusals come in the brief's order
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    '''One of the brief's 13 checks: a fragment only its message holds, and a defect that trips it.'''

    number: int
    fragment: str
    defect: Callable[[Scenario], None]

# Each defect works on its own rows, so any set of them applies together
CHECKS = [
    Check(1, r'lacks column|has type', dropping('n_unweighted', False)),
    Check(2, r'has a null', mirrored('111150', '111191', unit_b=None)),
    Check(3, r'which is not accepted', mirrored('111150', '311221', family='co_location')),
    Check(4, r'node id', lambda s: s.marginals(lambda m: with_marginal(m, 'naics:2022:x:1', 1))),
    Check(
        5, r'not a codebook', lambda s: (
            s.pairs.append(('311119', '999999', 1)), s.multi.update({'999999': 1})
        )
    ),
    Check(6, r'repeats node', repeated_marginal),
    Check(7, r'no marginal row', lambda s: s.pairs.append(('236220', '311119', 1))),
    Check(8, r'itself', lambda s: s.targets(lambda t: with_row(t, '311221', '311221', 1))),
    Check(9, r'repeats the pair', repeated_pair),
    Check(10, r'at least 1', lambda s: s.set_j('311211', '311221', 0)),
    Check(
        11, r'one direction', lambda s: s.targets(lambda t: t.filter(~row_of('312111', '445110')))
    ),
    Check(12, r'mirror', mirrored('312112', '312111', n_unweighted=4)),
    Check(13, r'exceeds', lambda s: s.set_j('111110', '445110', 3)),
]

def refusal_from(checks: List[Check]) -> str:
    '''The message for tables with each check's defect, applied last check first.'''

    scenario = Scenario()
    for check in sorted(checks, key=lambda check: -check.number):
        check.defect(scenario)
    with pytest.raises(ValueError) as refusal:
        scenario.read()
    return str(refusal.value)

def fragments_in(message: str) -> List[int]:
    return [check.number for check in CHECKS if re.search(check.fragment, message)]

@pytest.mark.parametrize('check', CHECKS, ids=lambda check: f'check-{check.number}')
def test_each_check_is_tripped_by_its_own_defect_alone(check):
    assert fragments_in(refusal_from([check])) == [check.number]

@pytest.mark.parametrize('check', CHECKS[:-1], ids=lambda check: f'check-{check.number}')
def test_a_check_fires_before_every_later_check(check):
    later = [other for other in CHECKS if other.number >= check.number]
    assert fragments_in(refusal_from(later)) == [check.number]

def test_the_base_tables_trip_no_check():
    Scenario().read()

def test_a_marginal_tables_missing_column_is_refused_before_a_target_null():
    scenario = Scenario()
    mirrored('311211', '311212', unit_b=None)(scenario)
    dropping('n_multi_industry_employers', True)(scenario)
    with pytest.raises(ValueError, match='lacks column') as refusal:
        scenario.read()
    assert MARGINALS in str(refusal.value)

def test_a_marginal_null_is_refused_before_a_target_family():
    scenario = Scenario()
    mirrored('311211', '311212', family='co_location')(scenario)
    null_marginal('n_multi_industry_employers')(scenario)
    with pytest.raises(ValueError, match='has a null') as refusal:
        scenario.read()
    assert MARGINALS in str(refusal.value)

def test_a_target_node_id_is_refused_before_a_marginal_outside_the_codebook():
    scenario = Scenario()
    a_bad_node_id_on_unit_b(scenario)
    a_marginal_outside_the_codebook(scenario)
    with pytest.raises(ValueError, match='node id') as refusal:
        scenario.read()
    assert TARGETS in str(refusal.value)
