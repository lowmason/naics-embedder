'''
The regressor panel's partition and its committed held-out draw (Req 2; Req 4; D7).
'''

import hashlib
from collections import Counter
from fractions import Fraction

import polars as pl
import pytest

from naics_embedder.panels.regressor_splits import (
    REMAINDER_FEATURE_YEARS,
    SEALED_FEATURE_YEAR,
    RegressorSplit,
    ancestor_at,
    assign_splits,
    check_partition,
    codes_fingerprint,
    draw_heldout_groups,
    group_of,
    group_table_fingerprint,
    heldout_tainted,
    read_codebook_codes,
    read_group_table,
    sector_quotas,
    split_counts,
    write_group_table,
)
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, POPULATION

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Hierarchy
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('code', 'level', 'ancestor'),
    [
        ('332111', 2, '31'),
        ('332111', 4, '3321'),
        ('452210', 2, '44'),
        ('493110', 2, '48'),
        ('493110', 6, '493110'),
        ('321', 2, '31'),
    ],
)
def test_ancestors_follow_the_naics_tree(code, level, ancestor):
    assert ancestor_at(code, level) == ancestor

@pytest.mark.parametrize(('code', 'level'), [('332111', 1), ('3321', 5)])
def test_there_is_no_ancestor_outside_the_codes_levels(code, level):
    with pytest.raises(ValueError, match='no ancestor'):
        ancestor_at(code, level)

def test_a_rows_group_is_its_four_digit_ancestor_or_the_code_itself():
    assert group_of('332111') == '3321'
    assert group_of('33211') == '3321'
    assert group_of('3321') == '3321'
    assert group_of('332') == '332'
    assert group_of('31') == '31'

def test_the_seen_outer_set_is_the_last_feature_year():
    assert SEALED_FEATURE_YEAR == 2024
    assert REMAINDER_FEATURE_YEARS == (2022, 2023)

# -------------------------------------------------------------------------------------------------
# The held-out draw
# -------------------------------------------------------------------------------------------------

def test_quotas_are_largest_remainder_shares_of_each_sector():
    # 3.8, 1.0, 0.6 and 0.2 floor to 4 groups; a fifth of 28, rounded, is 6
    quotas = sector_quotas({'11': 19, '21': 5, '22': 3, '55': 1}, Fraction(1, 5), 20260924)

    assert quotas == {'11': 4, '21': 1, '22': 1, '55': 0}

def test_remainder_ties_are_broken_by_the_seed():
    counts = {sector: 1 for sector in ('11', '21', '22', '23', '31')}

    draws = {seed: sector_quotas(counts, Fraction(1, 5), seed) for seed in range(20)}

    assert all(sum(quotas.values()) == 1 for quotas in draws.values())
    assert draws[3] == sector_quotas(counts, Fraction(1, 5), 3)
    assert len({tuple(sorted(quotas.items())) for quotas in draws.values()}) > 1

@pytest.mark.parametrize('fraction', [Fraction(0), Fraction(1), Fraction(3, 2)])
def test_the_fraction_must_lie_strictly_between_zero_and_one(fraction):
    with pytest.raises(ValueError, match='fraction'):
        sector_quotas({'11': 5}, fraction, 1)

def _codes(groups_by_sector):
    return [f'{group}11' for groups in groups_by_sector.values() for group in groups]

SECTOR_GROUPS = {
    '11': [f'11{number:02d}' for number in range(11, 21)],
    '31': ['3111', '3112', '3211', '3212', '3321'],
    '52': ['5211', '5221', '5222', '5231', '5232'],
}

def test_the_draw_is_seeded_and_stratified_by_sector():
    drawn = draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)

    assert drawn == draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)
    assert drawn == tuple(sorted(drawn))
    by_sector = Counter(ancestor_at(group, 2) for group in drawn)
    assert by_sector == Counter({'11': 2, '31': 1, '52': 1})
    assert draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 1) != drawn

def test_a_sectors_draw_does_not_depend_on_the_other_sectors():
    both = draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)
    alone = draw_heldout_groups(_codes({'31': SECTOR_GROUPS['31']}), Fraction(1, 5), 20260924)

    assert [group for group in both if group.startswith(('31', '32', '33'))] == list(alone)

# -------------------------------------------------------------------------------------------------
# The committed table and the codebook
# -------------------------------------------------------------------------------------------------

def test_the_fingerprint_is_the_committed_files_sha256(tmp_path):
    path = tmp_path / 'groups.csv'

    fingerprint = write_group_table(['5231', '1113', '3211'], path)

    assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fingerprint == group_table_fingerprint(HELDOUT_GROUPS)
    assert read_group_table(path) == HELDOUT_GROUPS
    assert path.read_text() == 'group\n1113\n3211\n5231\n'

@pytest.mark.parametrize('body', ['group\n1113\n1113\n', 'group\n111\n', 'group\n11131\n'])
def test_a_table_of_anything_but_distinct_four_digit_groups_is_refused(tmp_path, body):
    path = tmp_path / 'groups.csv'
    path.write_text(body)

    with pytest.raises(ValueError, match='distinct four-digit'):
        read_group_table(path)

def test_the_codebook_is_read_under_its_codes_fingerprint(tmp_path):
    path = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code_id': range(len(CODEBOOK)), 'code': list(CODEBOOK)}).write_parquet(path)
    expected = hashlib.sha256(''.join(f'{code}\n' for code in CODEBOOK).encode()).hexdigest()

    assert codes_fingerprint(reversed(CODEBOOK)) == expected
    assert read_codebook_codes(path, expected) == CODEBOOK
    with pytest.raises(ValueError, match='does not match'):
        read_codebook_codes(path, '0' * 64)
    repeated = tmp_path / 'repeated.parquet'
    pl.DataFrame({'code': ['11', '11']}).write_parquet(repeated)
    with pytest.raises(ValueError, match='repeats a code'):
        read_codebook_codes(repeated, expected)

# -------------------------------------------------------------------------------------------------
# The partition
# -------------------------------------------------------------------------------------------------

def test_the_partition_is_disjoint_and_covers_every_row(regressor_rows):
    rows = assign_splits(regressor_rows[6], HELDOUT_GROUPS)

    check_partition(rows, POPULATION)
    held = rows.get_column('group').is_in(list(HELDOUT_GROUPS))
    year = rows.get_column('feature_year')
    split = rows.get_column('split')
    assert (split.filter(held) == RegressorSplit.HELDOUT_OUTER.value).all()
    assert (split.filter(~held & (year == 2024)) == RegressorSplit.SEEN_OUTER.value).all()
    assert (split.filter(~held & (year < 2024)) == RegressorSplit.REMAINDER.value).all()
    assert sum(split_counts(rows).values()) == rows.height

def test_a_held_out_group_seals_every_code_it_rolls_into():
    assert heldout_tainted('311', ['3111'])
    assert heldout_tainted('31', ['3211'])
    assert heldout_tainted('311111', ['3111'])
    assert heldout_tainted('31111', ['3111'])
    assert not heldout_tainted('312', ['3111'])
    assert not heldout_tainted('311211', ['3111'])

def test_aggregate_codes_join_the_held_out_set_in_every_year(regressor_rows):
    rows = assign_splits(regressor_rows[3], HELDOUT_GROUPS)

    tainted = rows.filter(pl.col('code').is_in(['111', '321', '523']))
    assert set(tainted.get_column('split')) == {RegressorSplit.HELDOUT_OUTER.value}
    assert tainted.height == 9
    remainder = rows.filter(pl.col('split') == RegressorSplit.REMAINDER.value)
    assert sorted(remainder.get_column('code').unique()) == ['112', '238', '311', '332', '522']

def test_split_counts_name_every_split():
    rows = pl.DataFrame({'split': [RegressorSplit.REMAINDER.value]})

    assert split_counts(rows) == {'remainder': 1, 'seen_outer': 0, 'heldout_outer': 0}

def test_a_missing_or_repeated_code_year_breaks_the_partition(regressor_rows):
    rows = assign_splits(regressor_rows[6], HELDOUT_GROUPS)

    with pytest.raises(ValueError, match='code-years'):
        check_partition(rows.slice(1), POPULATION)
    # As many rows as code-years, but one code-year twice and another missing
    with pytest.raises(ValueError, match='code-years'):
        check_partition(pl.concat([rows.slice(1), rows.slice(1, 1)]), POPULATION)
    with pytest.raises(ValueError, match='no split'):
        check_partition(rows.with_columns(split=pl.lit(None, dtype=pl.Utf8)), POPULATION)
