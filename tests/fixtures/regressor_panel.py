'''
A miniature regressor panel: a synthetic NAICS tree, its QCEW national slices and stub arm tables.

Four sectors, one of them combined (31-33), and seventeen four-digit groups. Every group but 2381
has one five-digit code with two six-digit children. 238110 is its five-digit parent's only child,
and BLS publishes it only as 238111 and 238112, so its cell comes from 23811's row. 523211 is
suppressed in 2023, so it leaves the population, as the finding's excluded codes do. Values
follow a noisy size model, so the covariates carry signal.
'''

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.qcew_rows import (
    SLICE_COLUMNS,
    WINDOW_YEARS,
    level_cells,
    panel_rows,
    population,
    slice_name,
)
from naics_embedder.panels.regressor import ArmTables, FitSettings

GROUPS = (
    '1111',
    '1112',
    '1113',
    '1114',
    '1121',
    '1122',
    '2381',
    '3111',
    '3112',
    '3211',
    '3212',
    '3321',
    '3322',
    '5221',
    '5222',
    '5231',
    '5232',
)
SPLIT_CODE = '238110'
SUPPRESSED_CODE = '523211'
# One held-out group per sector with more than one group: 111, 321 and 523 are tainted at level 3
HELDOUT_GROUPS = ('1113', '3211', '5231')
QCEW_SECTORS = {'31': '31-33'}
AGGLVL = {2: '14', 3: '15', 4: '16', 5: '17', 6: '18'}

def _six_digit(group: str) -> List[str]:
    return [SPLIT_CODE] if group == '2381' else [f'{group}11', f'{group}12']

SIX_DIGIT = tuple(sorted(code for group in GROUPS for code in _six_digit(group)))
FIVE_DIGIT = tuple(sorted({code[:5] for code in SIX_DIGIT}))
SUBSECTORS = tuple(sorted({group[:3] for group in GROUPS}))
SECTORS = ('11', '23', '31', '52')
CODEBOOK = tuple(sorted((*SECTORS, *SUBSECTORS, *GROUPS, *FIVE_DIGIT, *SIX_DIGIT)))
POPULATION = tuple(code for code in SIX_DIGIT if code != SUPPRESSED_CODE)

BRANCH_RECORD = {
    'branch': 'A',
    'source': 'QCEW annual averages',
    'reference_years': list(WINDOW_YEARS),
    'ownership': '5',
    'grain': 'national',
    'population_seen': len(POPULATION),
    'population_heldout': len(POPULATION),
    'time_respecting_outcome': True,
    'seen_regime': True,
    'excluded_codes': [SUPPRESSED_CODE],
}
SETTINGS = FitSettings(
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0), folds=2, repeats=2, inner_folds=2, min_groups=4
)

# -------------------------------------------------------------------------------------------------
# QCEW cells and slices
# -------------------------------------------------------------------------------------------------

def _published(code: str) -> List[Tuple[str, str]]:
    '''The (industry_code, agglvl_code) rows QCEW publishes for a codebook code.'''

    if code == SPLIT_CODE:
        return [(f'{code[:5]}1', AGGLVL[6]), (f'{code[:5]}2', AGGLVL[6])]
    return [(QCEW_SECTORS.get(code, code), AGGLVL[len(code)])]

def synthetic_cells() -> pl.DataFrame:
    '''The private national annual rows ``read_national_slice`` returns, every window year.'''

    rng = np.random.default_rng(20260924)
    rows = []
    for code in CODEBOOK:
        for industry_code, agglvl_code in _published(code):
            size = rng.normal(7.0, 1.5)
            growth = rng.normal(0.02, 0.05)
            for year in WINDOW_YEARS:
                log_emp = size + growth * (year - WINDOW_YEARS[0]) + rng.normal(0.0, 0.05)
                estabs = int(round(np.exp(size - 2.3 + rng.normal(0.0, 0.1))))
                wages = int(round(np.exp(size + 10.8 + rng.normal(0.0, 0.1))))
                cell = ('', estabs, int(round(np.exp(log_emp))), wages)
                if code == SUPPRESSED_CODE and year == 2023:
                    cell = ('N', 0, 0, 0)
                rows.append((industry_code, agglvl_code, year, *cell))
    return pl.DataFrame(
        rows,
        schema={
            'industry_code': pl.Utf8,
            'agglvl_code': pl.Utf8,
            'year': pl.Int32,
            'disclosure_code': pl.Utf8,
            'estabs': pl.Int64,
            'emp': pl.Int64,
            'wages': pl.Int64,
        },
        orient='row',
    )

def _slice_rows(cells: pl.DataFrame, year: int) -> List[Dict[str, str]]:
    rows = []
    for cell in cells.filter(pl.col('year') == year).iter_rows(named=True):
        base = {
            'area_fips': 'US000',
            'own_code': '5',
            'industry_code': cell['industry_code'],
            'agglvl_code': cell['agglvl_code'],
            'size_code': '0',
            'year': str(year),
            'qtr': 'A',
            'disclosure_code': cell['disclosure_code'],
            'annual_avg_estabs': str(cell['estabs']),
            'annual_avg_emplvl': str(cell['emp']),
            'total_annual_wages': str(cell['wages']),
        }
        rows.append(base)
        # Rows the reader must drop: all ownerships, a state, a quarter
        rows.append({**base, 'own_code': '0', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'area_fips': '01000', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'qtr': '1', 'annual_avg_emplvl': '1'})
    return rows

def write_qcew_slices(directory: Path, cells: pl.DataFrame) -> Dict[str, str]:
    '''Write each window year's national slice; return their sha256 pins by file name.'''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = {name: pl.Utf8 for name in SLICE_COLUMNS}
    pins = {}
    for year in WINDOW_YEARS:
        frame = pl.DataFrame(_slice_rows(cells, year), schema=schema)
        path = directory / slice_name(year)
        frame.write_csv(path)
        pins[slice_name(year)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pins

def synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6)) -> Dict[int, pl.DataFrame]:
    '''Panel rows per level, as ``RegressorPanel.from_sources`` builds them.'''

    rows = {}
    for level in levels:
        cells_at_level = level_cells(cells, CODEBOOK, level)
        rows[level] = panel_rows(cells_at_level, population(cells_at_level))
    return rows

# -------------------------------------------------------------------------------------------------
# Stub arm tables
# -------------------------------------------------------------------------------------------------

def coordinate_table(codes, dimension: int = 3, seed: int = 7) -> pl.DataFrame:
    '''``code`` plus ``e0`` … ``e{dimension-1}``: a stub arm in the export form.'''

    values = np.random.default_rng(seed).normal(size=(len(codes), dimension))
    schema = {f'e{index}': pl.Float64 for index in range(dimension)}
    frame = pl.DataFrame(values, schema=schema, orient='row')
    return pl.DataFrame({'code': list(codes)}, schema={'code': pl.Utf8}).hstack(frame)

def text_only_table(codes, width: int = 5, seed: int = 11) -> pl.DataFrame:
    '''``code`` plus ``t0`` … ``t{width-1}``: a stub text-only table.'''

    values = np.random.default_rng(seed).normal(size=(len(codes), width))
    schema = {f't{index}': pl.Float64 for index in range(width)}
    frame = pl.DataFrame(values, schema=schema, orient='row')
    return pl.DataFrame({'code': list(codes)}, schema={'code': pl.Utf8}).hstack(frame)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def regressor_cells() -> pl.DataFrame:
    return synthetic_cells()

@pytest.fixture(scope='session')
def regressor_rows(regressor_cells) -> Dict[int, pl.DataFrame]:
    return synthetic_rows(regressor_cells)

@pytest.fixture(scope='session')
def regressor_arm() -> ArmTables:
    return ArmTables.from_tables(coordinate_table(CODEBOOK), text_only_table(CODEBOOK))
