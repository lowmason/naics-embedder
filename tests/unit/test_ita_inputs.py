'''
Stage 10's two ITA tables read into panel pairs (plan 17, Task 2; spec text-to-code-ita-panel, 4.1).

The panel keeps the headline pairs, takes each one's reported codes from its headline-basis rows,
and refuses an input that disagrees with the spec instead of dropping it. Each test builds its
tables from ``tests/fixtures/ita.py`` and changes one thing. A refusal test matches the refusal's
own wording, so a different refusal cannot pass for it.
'''

import re
from typing import Any, Dict

import polars as pl
import pytest

from naics_embedder.panels.ita import (
    NODE_PREFIX,
    PAIR_CODE_COLUMNS,
    PAIR_COLUMNS,
    PANEL_PAIR_SCHEMA,
    node_code,
    panel_pairs,
)
from tests.fixtures.ita import (
    ITA_CANDIDATES,
    ITA_PAIR_CODES_SCHEMA,
    ITA_PAIRS_SCHEMA,
    ITA_SETTINGS,
    ita_tables,
)

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

def _node(code: str) -> str:
    return f'naics:2022:national_industry:{code}'

def _reported(code: str, naics_year: int = 2022) -> Dict[str, Any]:
    '''A headline-basis row on a code, to build a pair-code row with ``ita_tables``.'''

    return {
        'naics_code': code,
        'naics_year': naics_year,
        'naics_2022_node_id': _node(code),
        'flag_headline_basis': True,
    }

def _set(frame: pl.DataFrame, pair_id: str, column: str, value: Any) -> pl.DataFrame:
    '''The frame with one column's value changed on the rows of one pair.'''

    dtype = frame.schema[column]
    return frame.with_columns(
        pl.when(pl.col('pair_id') == pair_id).then(pl.lit(value, dtype=dtype)).otherwise(
            pl.col(column)
        ).alias(column)
    )

def _retype(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    '''The frame with a column of another dtype: strings cannot cast to a number, so replace it.'''

    if frame.schema[column] == pl.Boolean:
        return frame.with_columns(pl.col(column).cast(pl.Utf8))
    return frame.with_columns(pl.lit(0).alias(column))

def _panel(pairs: pl.DataFrame, codes: pl.DataFrame) -> pl.DataFrame:
    return panel_pairs(pairs, codes, ITA_CANDIDATES, ITA_SETTINGS.salt)

def _base() -> pl.DataFrame:
    return _panel(*ita_tables())

# -------------------------------------------------------------------------------------------------
# The panel pairs
# -------------------------------------------------------------------------------------------------

def test_panel_pairs_are_the_headline_pairs_with_their_parts():
    frame = panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt)
    assert frame.get_column('pair_id').to_list() == [
        f'p{n:02d}' for n in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
    ]
    assert frame.filter(pl.col('part') == 'sealed').get_column('pair_id').to_list() == [
        'p08', 'p09', 'p10', 'p11'
    ]
    assert frame.filter(pl.col('pair_id') == 'p04').get_column('codes').to_list() == [
        ['111110', '311111']
    ]
    assert frame.filter(pl.col('copied')).get_column('pair_id').to_list() == ['p05', 'p11']

def test_a_node_repeated_across_years_is_one_reported_code():
    frame = panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt)
    row = frame.filter(pl.col('pair_id') == 'p01').row(0, named=True)
    assert row['codes'] == ['311111'] and row['conflict'] is False

def test_the_panel_pair_schema_is_the_planned_one():
    assert list(PANEL_PAIR_SCHEMA) == [
        'pair_id', 'description_key', 'query', 'codes', 'copied', 'conflict', 'mojibake',
        'company_fallback', 'part'
    ]
    assert dict(PANEL_PAIR_SCHEMA) == {
        'pair_id': pl.Utf8,
        'description_key': pl.Utf8,
        'query': pl.Utf8,
        'codes': pl.List(pl.Utf8),
        'copied': pl.Boolean,
        'conflict': pl.Boolean,
        'mojibake': pl.Boolean,
        'company_fallback': pl.Boolean,
        'part': pl.Utf8,
    }

def test_the_panel_pairs_come_in_the_panel_pair_schema():
    frame = _base()
    assert frame.columns == list(PANEL_PAIR_SCHEMA)
    assert dict(frame.schema) == dict(PANEL_PAIR_SCHEMA)

def test_the_columns_read_are_stage_10s_columns_with_its_types():
    assert list(PAIR_COLUMNS) == [
        'pair_id', 'description_key', 'representative_string', 'flag_headline',
        'flag_headline_codes_conflict', 'flag_copied_title_any', 'flag_copied_index_any',
        'flag_suspected_mojibake', 'flag_employer_company_fallback'
    ]
    assert list(PAIR_CODE_COLUMNS) == ['pair_id', 'naics_2022_node_id', 'flag_headline_basis']
    assert all(ITA_PAIRS_SCHEMA[name] == dtype for name, dtype in PAIR_COLUMNS.items())
    assert all(ITA_PAIR_CODES_SCHEMA[name] == dtype for name, dtype in PAIR_CODE_COLUMNS.items())

def test_the_query_is_the_representative_string_not_the_key():
    frame = _base()
    by_pair = {row['pair_id']: row for row in frame.iter_rows(named=True)}
    assert by_pair['p01']['query'] == 'We make dog food'
    assert by_pair['p02']['query'] == 'we make dog food'
    assert by_pair['p01']['description_key'] == by_pair['p02']['description_key'
                                                               ] == 'we make dog food'

def test_each_flag_comes_from_its_own_column():
    frame = _base()

    def pairs_where(column: str) -> list:
        return frame.filter(pl.col(column)).get_column('pair_id').to_list()

    assert pairs_where('copied') == ['p05', 'p11']
    assert pairs_where('conflict') == ['p04']
    assert pairs_where('mojibake') == ['p10']
    assert pairs_where('company_fallback') == ['p07']

def test_a_copied_title_or_a_copied_index_alone_marks_a_pair_copied():
    pairs, codes = ita_tables()
    only_title = _set(pairs, 'p05', 'flag_copied_title_any', True)
    only_index = _set(pairs, 'p05', 'flag_copied_title_any', False)
    only_index = _set(only_index, 'p05', 'flag_copied_index_any', True)
    for table in (only_title, only_index):
        frame = _panel(table, codes)
        assert frame.filter(pl.col('pair_id') == 'p05').get_column('copied').to_list() == [True]

def test_the_parts_come_from_the_salted_description_key():
    frame = _base()
    parts = dict(zip(frame.get_column('pair_id').to_list(), frame.get_column('part').to_list()))
    assert parts == {
        'p01': 'validation',
        'p02': 'validation',
        'p03': 'validation',
        'p04': 'validation',
        'p05': 'validation',
        'p07': 'validation',
        'p08': 'sealed',
        'p09': 'sealed',
        'p10': 'sealed',
        'p11': 'sealed',
    }

def test_the_salt_moves_the_parts():
    pairs, codes = ita_tables()
    base = _base()
    other = panel_pairs(pairs, codes, ITA_CANDIDATES, 'f' * 32)
    assert other.get_column('part').to_list() != base.get_column('part').to_list()

def test_row_order_does_not_change_the_panel_pairs():
    pairs, codes = ita_tables()
    assert _panel(pairs.reverse(), codes.reverse()).equals(_base())

def test_codes_are_sorted_and_distinct_whatever_the_row_order():
    extra_pair = {
        'pair_id': 'p12',
        'description_key': 'every code',
        'representative_string': 'Every code',
        'flag_headline': True,
        'flag_headline_codes_conflict': True,
    }
    twice = [
        {
            'pair_id': 'p12',
            **_reported(code, naics_year=year)
        } for code in reversed(ITA_CANDIDATES) for year in (2022, 2023)
    ]
    pairs, codes = ita_tables(extra_pairs=[extra_pair], extra_codes=twice)
    frame = _panel(pairs.reverse(), codes.reverse())
    row = frame.filter(pl.col('pair_id') == 'p12').row(0, named=True)
    assert row['codes'] == sorted(ITA_CANDIDATES)

def test_extra_columns_are_ignored():
    pairs, codes = ita_tables()
    slim = _panel(pairs.select(list(PAIR_COLUMNS)), codes.select(list(PAIR_CODE_COLUMNS)))
    assert slim.equals(_base())
    extra = _panel(
        pairs.with_columns(pl.lit(0).alias('zz_extra'),
                           pl.col('n_records').cast(pl.Utf8)),
        codes.with_columns(pl.lit('x').alias('zz_extra'),
                           pl.lit(0).alias('years')),
    )
    assert extra.equals(_base())

# -------------------------------------------------------------------------------------------------
# What is not read
# -------------------------------------------------------------------------------------------------

def test_a_pair_that_is_not_headline_is_not_read():
    pairs, codes = ita_tables(
        replace_codes={
            'p06': [
                {
                    'naics_code': '517112',
                    'naics_2022_node_id': 'naics:2022:industry:517112',
                    'flag_headline_basis': True,
                },
                {
                    'naics_code': '999999',
                    'naics_2022_node_id': _node('999999'),
                    'flag_headline_basis': True,
                },
                {
                    'naics_code': '311111',
                    'naics_2022_node_id': None,
                    'flag_headline_basis': True,
                },
                {
                    'naics_code': '311112',
                    'naics_2022_node_id': _node('311112'),
                    'flag_headline_basis': None,
                },
            ]
        }
    )
    pairs = _set(pairs, 'p06', 'representative_string', None)
    pairs = _set(pairs, 'p06', 'description_key', None)
    pairs = _set(pairs, 'p06', 'flag_copied_title_any', None)
    pairs = _set(pairs, 'p06', 'flag_headline_codes_conflict', None)
    assert _panel(pairs, codes).equals(_base())

def test_a_row_that_is_not_headline_basis_is_not_read():
    not_basis = [
        {
            'pair_id': 'p03',
            'naics_code': '517112',
            'naics_2022_node_id': 'naics:2022:industry:517112',
        },
        {
            'pair_id': 'p03',
            'naics_code': '311111',
            'naics_year': 2017,
            'naics_2022_node_id': 'naics:2017:national_industry:311111',
        },
        {
            'pair_id': 'p03',
            'naics_code': '999999',
            'naics_2022_node_id': _node('999999'),
        },
        {
            'pair_id': 'p03',
            'naics_code': '311112',
            'naics_2022_node_id': None,
        },
    ]
    pairs, codes = ita_tables(extra_codes=not_basis)
    assert _panel(pairs, codes).equals(_base())

# -------------------------------------------------------------------------------------------------
# Node ids
# -------------------------------------------------------------------------------------------------

def test_the_node_prefix_is_naics_2022_national_industry():
    assert NODE_PREFIX == ('naics', '2022', 'national_industry')

@pytest.mark.parametrize('code', ['311111', '111110', '000000', '999999'])
def test_node_code_returns_the_six_digit_code(code):
    assert node_code(_node(code)) == code

@pytest.mark.parametrize(
    'node_id', [
        'naics:2022:industry:517112',
        'naics:2017:national_industry:311111',
        'sic:2022:national_industry:311111',
        'naics:2022:national_industry:31111',
    ]
)
def test_node_code_refuses_another_system_vintage_level_or_width(node_id):
    with pytest.raises(ValueError, match=re.escape(node_id)):
        node_code(node_id)

@pytest.mark.parametrize(
    'node_id',
    [
        '',
        '311111',
        'naics:2022:311111',
        'naics:2022:national_industry',
        'naics:2022:national_industry:',
        'naics:2022:national_industry:311111:1',
        'naics:2022:national_industry:extra:311111',
        'naics:2022:national_industry:3111111',
        'naics:2022:national_industry:31111a',
        'naics:2022:national_industry:311 11',
        'naics:2022:national_industry:３１１１１１',
        'naics:2022:national_industry:311111\n',
        'NAICS:2022:national_industry:311111',
        'naics:2022:National_Industry:311111',
    ],
)
def test_node_code_refuses_another_part_count_or_a_code_that_is_not_six_digits(node_id):
    with pytest.raises(
        ValueError, match=re.escape('is not a naics:2022:national_industry:<six digits> node id')
    ):
        node_code(node_id)

def test_node_code_names_the_node_id_it_refuses():
    with pytest.raises(ValueError) as refusal:
        node_code('naics:2022:industry:517112')
    assert 'naics:2022:industry:517112' in str(refusal.value)

# -------------------------------------------------------------------------------------------------
# Refusals
# -------------------------------------------------------------------------------------------------

def test_a_code_outside_the_candidates_is_refused():
    pairs, codes = ita_tables(replace_codes={'p03': [_reported('999999')]})
    with pytest.raises(ValueError, match='pair p03 reports code 999999, which is not a candidate'):
        _panel(pairs, codes)

def test_the_candidates_are_the_ones_given():
    pairs, codes = ita_tables()
    candidates = tuple(code for code in ITA_CANDIDATES if code != '211120')
    with pytest.raises(ValueError, match='pair p03 reports code 211120, which is not a candidate'):
        panel_pairs(pairs, codes, candidates, ITA_SETTINGS.salt)

def test_a_reported_node_of_another_level_is_refused():
    node_id = 'naics:2022:industry:517112'
    reported = {**_reported('517112'), 'naics_2022_node_id': node_id}
    pairs, codes = ita_tables(replace_codes={'p03': [reported]})
    with pytest.raises(ValueError, match=re.escape(node_id)):
        _panel(pairs, codes)

def test_a_reported_node_of_another_vintage_is_refused():
    node_id = 'naics:2017:national_industry:311111'
    reported = {**_reported('311111'), 'naics_2022_node_id': node_id}
    pairs, codes = ita_tables(replace_codes={'p03': [reported]})
    with pytest.raises(ValueError, match=re.escape(node_id)):
        _panel(pairs, codes)

def test_a_reported_row_without_a_node_id_is_refused():
    reported = {**_reported('311111'), 'naics_2022_node_id': None}
    pairs, codes = ita_tables(replace_codes={'p03': [reported]})
    with pytest.raises(
        ValueError, match="pair p03 has a headline-basis row with a null 'naics_2022_node_id'"
    ):
        _panel(pairs, codes)

@pytest.mark.parametrize(
    'replacement',
    [
        [],
        [{
            **_reported('211120'), 'flag_headline_basis': False
        }],
    ],
    ids=['no code row', 'only a row that is not headline-basis'],
)
def test_a_headline_pair_without_a_reported_code_is_refused(replacement):
    pairs, codes = ita_tables(replace_codes={'p03': replacement})
    with pytest.raises(ValueError, match='headline pair p03 has no headline-basis row'):
        _panel(pairs, codes)

def test_a_null_representative_string_is_refused():
    pairs, codes = ita_tables()
    pairs = _set(pairs, 'p03', 'representative_string', None)
    with pytest.raises(ValueError, match="headline pair p03 has a null 'representative_string'"):
        _panel(pairs, codes)

@pytest.mark.parametrize(
    'column', [
        name for name in PAIR_COLUMNS
        if name not in ('pair_id', 'flag_headline', 'representative_string')
    ]
)
def test_a_null_in_a_column_read_for_a_headline_pair_is_refused(column):
    pairs, codes = ita_tables()
    pairs = _set(pairs, 'p03', column, None)
    with pytest.raises(ValueError, match=f"headline pair p03 has a null '{column}'"):
        _panel(pairs, codes)

@pytest.mark.parametrize('column', ['pair_id', 'flag_headline'])
def test_a_null_pair_id_or_headline_flag_is_refused(column):
    pairs, codes = ita_tables()
    pairs = _set(pairs, 'p06', column, None)
    with pytest.raises(ValueError, match=f"pair table has a null '{column}'"):
        _panel(pairs, codes)

def test_a_null_headline_basis_flag_on_a_headline_pair_is_refused():
    pairs, codes = ita_tables()
    codes = _set(codes, 'p03', 'flag_headline_basis', None)
    with pytest.raises(ValueError, match="headline pair p03 has a null 'flag_headline_basis'"):
        _panel(pairs, codes)

@pytest.mark.parametrize(
    ('pair_id', 'flag', 'reported'),
    [
        ('p04', False, r"\['111110', '311111'\]"),
        ('p03', True, r"\['211120'\]"),
        ('p01', True, r"\['311111'\]"),
    ],
    ids=['two codes, flag false', 'one code, flag true', 'one node over two years, flag true'],
)
def test_a_conflict_flag_that_disagrees_with_the_codes_is_refused(pair_id, flag, reported):
    pairs, codes = ita_tables()
    pairs = _set(pairs, pair_id, 'flag_headline_codes_conflict', flag)
    message = f'pair {pair_id} has flag_headline_codes_conflict {flag} but reports {reported}'
    with pytest.raises(ValueError, match=message):
        _panel(pairs, codes)

@pytest.mark.parametrize('column', list(PAIR_COLUMNS))
def test_a_missing_column_of_the_pair_table_is_refused(column):
    pairs, codes = ita_tables()
    with pytest.raises(ValueError, match=f"pair table lacks column '{column}'"):
        _panel(pairs.drop(column), codes)

@pytest.mark.parametrize('column', list(PAIR_CODE_COLUMNS))
def test_a_missing_column_of_the_pair_code_table_is_refused(column):
    pairs, codes = ita_tables()
    with pytest.raises(ValueError, match=f"pair-code table lacks column '{column}'"):
        _panel(pairs, codes.drop(column))

@pytest.mark.parametrize('column', list(PAIR_COLUMNS))
def test_a_retyped_column_of_the_pair_table_is_refused(column):
    pairs, codes = ita_tables()
    with pytest.raises(ValueError, match=f"pair table column '{column}' has type"):
        _panel(_retype(pairs, column), codes)

@pytest.mark.parametrize('column', list(PAIR_CODE_COLUMNS))
def test_a_retyped_column_of_the_pair_code_table_is_refused(column):
    pairs, codes = ita_tables()
    with pytest.raises(ValueError, match=f"pair-code table column '{column}' has type"):
        _panel(pairs, _retype(codes, column))

@pytest.mark.parametrize(
    ('repeated', 'headline'),
    [('p03', True), ('p06', False)],
    ids=['a headline pair', 'a pair that is not headline'],
)
def test_a_repeated_pair_id_is_refused(repeated, headline):
    twin = {
        'pair_id': repeated,
        'description_key': 'a twin',
        'representative_string': 'A twin',
        'flag_headline': headline,
    }
    pairs, codes = ita_tables(extra_pairs=[twin])
    with pytest.raises(ValueError, match=f"pair table repeats pair_id: \\['{repeated}'\\]"):
        _panel(pairs, codes)

def test_a_pair_id_repeated_in_the_pair_code_table_is_not_a_repeat():
    pairs, codes = ita_tables()
    assert codes.get_column('pair_id').is_duplicated().any()
    assert _panel(pairs, codes).equals(_base())
