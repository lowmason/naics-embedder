import csv
import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'employment_statistics_coverage.py'
_SPEC = importlib.util.spec_from_file_location('employment_statistics_coverage', _SCRIPT)
esc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = esc
_SPEC.loader.exec_module(esc)

WINDOW = (2022, 2023, 2024, 2025)
CSV_COLUMNS = (
    *esc.KEY_COLUMNS,
    'annual_avg_estabs',
    'annual_avg_emplvl',
    'total_annual_wages',
    'avg_annual_pay',
)
SIX_DIGIT = ('111110', '112130', '238110', '238120', '541511', '541512', '921110')
HIGHER = (
    '11', '111', '1111', '11111', '112', '1121', '11213', '23', '238', '2381', '23811', '23812',
    '54', '541', '5415', '54151', '92', '921', '9211', '92111'
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

def _record(year, area, own, industry, agglvl, disclosure, estabs, emp, wages, size='0'):
    return {
        'area_fips': area,
        'own_code': own,
        'industry_code': industry,
        'agglvl_code': agglvl,
        'size_code': size,
        'year': str(year),
        'qtr': 'A',
        'disclosure_code': disclosure,
        'annual_avg_estabs': str(estabs),
        'annual_avg_emplvl': str(emp),
        'total_annual_wages': str(wages),
        'avg_annual_pay': '0',
    }

def _annual_rows(year):
    connecticut = '09001' if year <= 2023 else '09110'
    rows = [
        ('US000', '5', '10', '11', '', 100, 1000, 100000),
        ('US000', '5', '111110', '18', '', 10, 100, 10000),
        ('US000', '5', '541511', '18', 'N', 3, 0, 0),
        ('US000', '5', '541512', '18', '-', 0, 0, 0),
        ('US000', '5', '238111', '18', '', 4, 40, 4000),
        ('US000', '5', '238112', '18', 'N', 1, 0, 0),
        ('US000', '5', '238121', '18', '', 2, 20, 2000),
        ('US000', '5', '238122', '18', 'N', 2, 0, 0),
        ('US000', '5', '999999', '18', '', 1, 5, 500),
        ('US000', '1', '921110', '18', '', 1, 50, 5000),
        ('US000', '5', '11111', '17', '', 10, 100, 10000),
        ('US000', '5', '23811', '17', '', 5, 45, 4500),
        ('US000', '5', '23812', '17', 'N', 4, 0, 0),
        ('01000', '5', '111110', '58', '', 6, 60, 6000),
        ('09000', '5', '111110', '58', '', 3, 30, 3000),
        ('72000', '5', '111110', '58', '', 1, 5, 500),
        ('01000', '5', '541511', '58', 'N', 2, 0, 0),
        ('01000', '5', '23811', '57', '', 3, 25, 2500),
        ('01001', '5', '111110', '78', '', 4, 35, 3500),
        ('01003', '5', '111110', '78', 'N', 1, 0, 0),
        ('01999', '5', '111110', '78', '', 1, 5, 500),
        (connecticut, '5', '111110', '78', '', 2, 20, 2000),
    ]
    if year <= 2024:
        rows.append(('C1010', '5', '111110', '48', '', 3, 30, 3000))
    return [_record(year, *row) for row in rows]

def _csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()

def _write_codebook(directory):
    path = directory / 'naics_codebook.parquet'
    codes = sorted([*HIGHER, *SIX_DIGIT])
    pl.DataFrame(
        {
            'code_id': list(range(len(codes))),
            'code': codes
        },
        schema={
            'code_id': pl.Int32,
            'code': pl.String
        }
    ).write_parquet(path)
    return path

def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def universe(tmp_path):
    path = _write_codebook(tmp_path)
    return esc.load_universe(path, _digest(path))

@pytest.fixture
def frames():
    return {year: esc.read_annual_csv(_csv_bytes(_annual_rows(year))) for year in WINDOW}

def _cells(frames, universe, grain):
    return pl.concat(
        [esc.grain_cells(frames[year], universe, ('238110', '238120'), grain) for year in WINDOW]
    )

# -------------------------------------------------------------------------------------------------
# Universe and reading
# -------------------------------------------------------------------------------------------------

def test_load_universe_checks_the_hash_and_finds_only_children(tmp_path):
    path = _write_codebook(tmp_path)
    universe = esc.load_universe(path, _digest(path))
    assert universe.six_digit == SIX_DIGIT
    assert universe.only_children == {'111110', '112130', '238110', '238120', '921110'}
    with pytest.raises(ValueError, match='sha256'):
        esc.load_universe(path, '0' * 64)

def test_resolve_estabs_column_accepts_one_spelling():
    assert esc.resolve_estabs_column(['annual_avg_estabs']) == 'annual_avg_estabs'
    assert esc.resolve_estabs_column(['annual_avg_estabs_count']) == 'annual_avg_estabs_count'
    with pytest.raises(ValueError, match='exactly one'):
        esc.resolve_estabs_column(['annual_avg_estabs', 'annual_avg_estabs_count'])
    with pytest.raises(ValueError, match='exactly one'):
        esc.resolve_estabs_column(['annual_avg_emplvl'])

def test_read_annual_csv_keeps_codes_as_strings():
    rows = _annual_rows(2024) + [_record(2024, 'US000', '5', '111110', '28', '', 1, 1, 1, '1')]
    frame = esc.read_annual_csv(_csv_bytes(rows))
    assert frame.schema['area_fips'] == pl.String
    assert frame.schema['estabs'] == pl.Int64
    assert frame.schema['year'] == pl.Int32
    assert {'01001', '01999', '09110', 'C1010'} <= set(frame.get_column('area_fips').to_list())
    assert '28' not in frame.get_column('agglvl_code').to_list()
    assert frame.filter(pl.col('industry_code') == '111110').get_column('disclosure_code')[0] == ''

def test_read_annual_csv_rejects_missing_columns():
    data = _csv_bytes(_annual_rows(2024)).replace(b'total_annual_wages', b'tot_wages')
    with pytest.raises(ValueError, match='lacks columns'):
        esc.read_annual_csv(data)

def test_read_singlefile_zip_reads_the_only_csv(tmp_path):
    path = tmp_path / '2024_annual_singlefile.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('2024.annual.singlefile.csv', _csv_bytes(_annual_rows(2024)))
    assert esc.read_singlefile_zip(path).height == len(_annual_rows(2024))

# -------------------------------------------------------------------------------------------------
# Cells
# -------------------------------------------------------------------------------------------------

def test_find_split_codes_detects_bls_children(universe, frames):
    published = esc.national_six_digit_codes(frames[2024])
    assert '921110' in published  # government-only codes count as published
    assert esc.find_split_codes(published, universe) == ('238110', '238120')

def test_find_split_codes_rejects_a_split_code_with_siblings(tmp_path):
    path = tmp_path / 'codebook.parquet'
    pl.DataFrame({'code': ['238110', '238113']}).write_parquet(path)
    universe = esc.load_universe(path, _digest(path))
    with pytest.raises(ValueError, match='only child'):
        esc.find_split_codes(['238111', '238113'], universe)

def test_grain_cells_never_read_suppression_as_zero(universe, frames):
    cells = esc.grain_cells(frames[2024], universe, ('238110', '238120'), 'national')
    private = {row['code']: row for row in cells.filter(pl.col('own_code') == '5').to_dicts()}
    assert set(private) == {'111110', '238110', '238120', '541511', '541512'}
    assert private['541511']['status'] == esc.SUPPRESSED
    assert private['541511']['estabs_status'] == esc.DISCLOSED
    assert private['541512']['status'] == esc.OTHER
    assert private['238110']['source'] == 'five_digit_parent'
    assert private['238110']['status'] == esc.DISCLOSED
    assert private['238120']['status'] == esc.SUPPRESSED
    assert private['111110']['source'] == 'six_digit'

def test_grain_cells_drop_unknown_counties(universe, frames):
    cells = esc.grain_cells(frames[2024], universe, ('238110', '238120'), 'county')
    assert set(cells.get_column('area_fips').to_list()) == {'01001', '01003', '09110'}
