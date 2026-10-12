'''
QCEW national rows for the regressor panel (Req 2; Req 21; roadmap Stage 3; D1; D7).

Every slice here is synthetic (``tests/fixtures/regressor_panel.py``): the real slices stay outside
the repo, under the sha256 values Stage 1's finding records.
'''

import math

import polars as pl
import pytest

from naics_embedder.panels.qcew_rows import (
    FEATURE_YEARS,
    ROW_COLUMNS,
    WINDOW_YEARS,
    RegressorOutcome,
    find_split_codes,
    level_cells,
    load_national_cells,
    panel_rows,
    population,
    read_national_slice,
    slice_name,
)
from tests.fixtures.regressor_panel import (
    CODEBOOK,
    POPULATION,
    SPLIT_CODE,
    SUPPRESSED_CODE,
    write_qcew_slices,
)

pytestmark = pytest.mark.unit

def test_the_window_gives_three_feature_years_before_their_outcome_years():
    assert WINDOW_YEARS == (2022, 2023, 2024, 2025)
    assert FEATURE_YEARS == (2022, 2023, 2024)
    assert slice_name(2024) == '2024_US000_annual.csv'

def test_a_slice_keeps_only_national_private_all_size_annual_rows(tmp_path, regressor_cells):
    write_qcew_slices(tmp_path, regressor_cells)

    frame = read_national_slice(tmp_path / slice_name(2022))

    assert frame.columns == [
        'industry_code', 'agglvl_code', 'year', 'disclosure_code', 'estabs', 'emp', 'wages'
    ]
    assert frame.height == regressor_cells.filter(pl.col('year') == 2022).height
    assert frame.get_column('year').dtype == pl.Int32
    assert frame.get_column('emp').dtype == pl.Int64
    # The reader drops every row the fixture adds with one employee
    assert frame.get_column('emp').min() > 1

def test_every_slice_is_read_under_its_pinned_sha256(tmp_path, regressor_cells):
    pins = write_qcew_slices(tmp_path, regressor_cells)

    cells = load_national_cells(tmp_path, pins)

    assert sorted(cells.get_column('year').unique().to_list()) == list(WINDOW_YEARS)
    with pytest.raises(ValueError, match='does not match'):
        load_national_cells(tmp_path, {**pins, slice_name(2023): '0' * 64})
    missing = {name: digest for name, digest in pins.items() if name != slice_name(2025)}
    with pytest.raises(ValueError, match='no pinned sha256'):
        load_national_cells(tmp_path, missing)

def test_combined_sectors_are_keyed_as_in_the_codebook(regressor_cells):
    sectors = level_cells(regressor_cells, CODEBOOK, 2)

    assert sorted(sectors.get_column('code').unique().to_list()) == ['11', '23', '31', '52']

def test_a_split_six_digit_code_is_read_from_its_five_digit_parent(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    five = level_cells(regressor_cells, CODEBOOK, 5)

    split = six.filter(pl.col('code') == SPLIT_CODE)
    parent = five.filter(pl.col('code') == SPLIT_CODE[:5])
    assert split.get_column('source').unique().to_list() == ['five_digit_parent']
    assert split.select('year', 'emp', 'estabs',
                        'wages').equals(parent.select('year', 'emp', 'estabs', 'wages'))
    assert set(six.filter(pl.col('code') != SPLIT_CODE).get_column('source')) == {'published'}

def test_a_split_code_must_be_an_only_child():
    published = ['238111', '238112']

    assert find_split_codes(published, ['238110']) == ('238110', )
    with pytest.raises(ValueError, match='not an only child'):
        find_split_codes(published, ['238110', '238115'])

def test_levels_outside_two_to_six_are_refused(regressor_cells):
    with pytest.raises(ValueError, match='level must be one of'):
        level_cells(regressor_cells, CODEBOOK, 7)

def test_the_population_needs_a_usable_cell_in_every_window_year(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    assert population(six) == POPULATION
    assert SUPPRESSED_CODE not in population(six)
    zero_wages = six.with_columns(
        wages=pl.when((pl.col('code') == POPULATION[0])
                      & (pl.col('year') == 2025)).then(0).otherwise(pl.col('wages'))
    )
    assert POPULATION[0] not in population(zero_wages)

def test_every_rows_features_are_dated_before_its_outcome(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    rows = panel_rows(six, POPULATION)

    assert rows.columns == list(ROW_COLUMNS)
    assert rows.height == len(POPULATION) * len(FEATURE_YEARS)
    assert (rows.get_column('outcome_year') == rows.get_column('feature_year') + 1).all()
    assert sorted(rows.get_column('feature_year').unique().to_list()) == list(FEATURE_YEARS)
    row = rows.filter((pl.col('code') == POPULATION[0])
                      & (pl.col('feature_year') == 2023)).row(0, named=True)
    cell = {
        year: six.filter((pl.col('code') == POPULATION[0])
                         & (pl.col('year') == year)).row(0, named=True)
        for year in (2023, 2024)
    }
    assert row['log_estabs'] == pytest.approx(math.log(cell[2023]['estabs']))
    assert row['log_wages'] == pytest.approx(math.log(cell[2023]['wages']))
    assert row['outcome'] == pytest.approx(math.log(cell[2024]['emp']))

@pytest.mark.parametrize('outcome', list(RegressorOutcome))
def test_panel_rows_refuse_a_code_without_every_window_year(regressor_cells, outcome):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    with pytest.raises(ValueError, match='panel rows for'):
        panel_rows(six, [*POPULATION, SUPPRESSED_CODE], outcome)

# -------------------------------------------------------------------------------------------------
# The growth outcome (Req 21)
# -------------------------------------------------------------------------------------------------

def test_a_growth_rows_outcome_is_the_log_change_in_employment_from_the_same_cells(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    cell = {(row['code'], row['year']): row for row in six.iter_rows(named=True)}

    growth = panel_rows(six, POPULATION, RegressorOutcome.GROWTH)

    assert growth.columns == list(ROW_COLUMNS)
    assert growth.height == len(POPULATION) * len(FEATURE_YEARS)
    for row in growth.iter_rows(named=True):
        now, later = cell[row['code'], row['feature_year']], cell[row['code'], row['outcome_year']]
        change = math.log(later['emp']) - math.log(now['emp'])
        # Features from year t; the outcome from t to t + 1
        assert row['outcome_year'] == row['feature_year'] + 1
        assert row['log_estabs'] == pytest.approx(math.log(now['estabs']), abs=1e-12)
        assert row['log_wages'] == pytest.approx(math.log(now['wages']), abs=1e-12)
        assert row['outcome'] == pytest.approx(change, abs=1e-12)

def test_a_growth_row_differs_from_its_level_row_in_the_outcome_alone(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    level = panel_rows(six, POPULATION)
    earlier = level.select(
        'code', feature_year=pl.col('feature_year') + 1, earlier=pl.col('outcome')
    )

    growth = panel_rows(six, POPULATION, RegressorOutcome.GROWTH)

    assert growth.drop('outcome').equals(level.drop('outcome'))
    # From 2023 on, log E(t) is the outcome of the code's level row a year earlier, bit for bit
    paired = growth.join(
        level.select('code', 'feature_year', level=pl.col('outcome')), on=['code', 'feature_year']
    ).join(earlier, on=['code', 'feature_year'])
    assert paired.height == 2 * len(POPULATION)
    assert (
        paired.get_column('outcome') == paired.get_column('level') - paired.get_column('earlier')
    ).all()

def test_an_outcome_is_level_or_growth(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    by_name = panel_rows(six, POPULATION, 'growth')

    assert [outcome.value for outcome in RegressorOutcome] == ['level', 'growth']
    assert by_name.equals(panel_rows(six, POPULATION, RegressorOutcome.GROWTH))
    with pytest.raises(ValueError, match='not a valid RegressorOutcome'):
        panel_rows(six, POPULATION, 'rate')
