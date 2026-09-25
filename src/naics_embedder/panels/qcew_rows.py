'''
QCEW national rows for the regressor panel (roadmap Stage 3; Req 2; D1; D7).

Stage 1's finding (``specs/findings/employment-statistics-coverage.md``) fixes the source: QCEW
annual averages for 2022–2025, national grain, private ownership (``own_code`` 5). The panel
reads the Open Data Access national slices (``{year}_US000_annual.csv``) under the sha256 values
the finding records; they agree with the annual single files on every national row.

- **Cells.** One cell per code and year. Sectors 31-33, 44-45 and 48-49 are keyed 31, 44 and 48,
  as in the codebook. Each of the 19 six-digit NAICS 238 codes that BLS publishes only as
  residential (``…1``) and nonresidential (``…2``) codes is its five-digit parent's only child, so
  its cell is read from the parent's row.
- **Usable.** A cell is usable when it is disclosed and its employment, establishments and wages
  are all positive. A suppressed cell is never read as zero.
- **Population.** The codes with a usable cell in every window year.
- **Rows (D7).** One row per population code and feature year t in 2022–2024: the covariates are
  log establishments and log total annual wages from year t (D1), and the outcome is log
  annual-average employment in year t + 1.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from collections import Counter
from pathlib import Path
from typing import Collection, Mapping, Sequence, Tuple

import polars as pl

from naics_embedder.supervision.artifacts import sha256_file

WINDOW_YEARS = (2022, 2023, 2024, 2025)
# D7: features from year t, outcome in t + 1, so the last window year is an outcome year only
FEATURE_YEARS = WINDOW_YEARS[:-1]
PRIVATE = '5'
NATIONAL_AREA = 'US000'
AGGLVL_BY_LEVEL = {2: '14', 3: '15', 4: '16', 5: '17', 6: '18'}
QCEW_SECTOR_CODES = {'31-33': '31', '44-45': '44', '48-49': '48'}

SLICE_COLUMNS = (
    'area_fips',
    'own_code',
    'industry_code',
    'agglvl_code',
    'size_code',
    'year',
    'qtr',
    'disclosure_code',
    'annual_avg_estabs',
    'annual_avg_emplvl',
    'total_annual_wages',
)
CELL_COLUMNS = ('code', 'year', 'disclosure_code', 'estabs', 'emp', 'wages', 'source')
ROW_COLUMNS = ('code', 'feature_year', 'outcome_year', 'log_estabs', 'log_wages', 'outcome')

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def slice_name(year: int) -> str:
    '''File name of one year's national slice.'''

    return f'{year}_US000_annual.csv'

def read_national_slice(path: Path) -> pl.DataFrame:
    '''
    The private, all-size annual rows of one national slice.

    Returns:
        ``industry_code``, ``agglvl_code``, ``year`` (Int32), ``disclosure_code`` (blank reads as
        ``''``), and ``estabs``, ``emp``, ``wages`` as Int64.
    '''

    frame = pl.read_csv(Path(path), columns=list(SLICE_COLUMNS), infer_schema=False)
    keys = [name for name in SLICE_COLUMNS if not name.startswith(('annual_', 'total_'))]
    # yapf: disable
    return (
        frame
        .with_columns(pl.col(*keys).fill_null('').str.strip_chars())
        .filter(
            (pl.col('area_fips') == NATIONAL_AREA)
            & (pl.col('own_code') == PRIVATE)
            & (pl.col('size_code') == '0')
            & (pl.col('qtr') == 'A')
        )
        .select(
            'industry_code',
            'agglvl_code',
            pl.col('year').cast(pl.Int32),
            'disclosure_code',
            estabs=pl.col('annual_avg_estabs').str.strip_chars().cast(pl.Int64),
            emp=pl.col('annual_avg_emplvl').str.strip_chars().cast(pl.Int64),
            wages=pl.col('total_annual_wages').str.strip_chars().cast(pl.Int64),
        )
    )
    # yapf: enable

def load_national_cells(qcew_dir: Path, expected_sha256: Mapping[str, str]) -> pl.DataFrame:
    '''
    Read every window year's national slice after checking its sha256.

    Raises:
        ValueError: If a slice has no pinned sha256 or its sha256 differs from the pinned one.
    '''

    frames = []
    for year in WINDOW_YEARS:
        name = slice_name(year)
        if name not in expected_sha256:
            raise ValueError(f'no pinned sha256 for {name}')
        path = Path(qcew_dir).expanduser() / name
        digest = sha256_file(path)
        if digest != expected_sha256[name]:
            raise ValueError(f'{path}: sha256 {digest} does not match {expected_sha256[name]}')
        frames.append(read_national_slice(path))
    return pl.concat(frames)

# -------------------------------------------------------------------------------------------------
# Cells, population and rows
# -------------------------------------------------------------------------------------------------

def find_split_codes(published: Collection[str], codebook_six: Sequence[str]) -> Tuple[str, ...]:
    '''
    Codebook six-digit codes that QCEW publishes only as BLS residential and nonresidential codes.

    Raises:
        ValueError: If such a code is not its five-digit parent's only child, so the parent's row
            would not carry the code's own value.
    '''

    published = set(published)
    codebook = set(codebook_six)
    siblings = Counter(code[:5] for code in codebook)
    split = []
    for code in sorted(codebook):
        bls_codes = {code[:5] + '1', code[:5] + '2'} - codebook
        if code in published or not bls_codes & published:
            continue
        if siblings[code[:5]] != 1:
            raise ValueError(f'{code} is split into BLS codes but is not an only child')
        split.append(code)
    return tuple(split)

def level_cells(cells: pl.DataFrame, codebook_codes: Sequence[str], level: int) -> pl.DataFrame:
    '''
    One cell per codebook code of the given level and year.

    Returns:
        ``code``, ``year``, ``disclosure_code``, ``estabs``, ``emp``, ``wages`` and ``source``
        (``published``, or ``five_digit_parent`` for a split six-digit code).
    '''

    if level not in AGGLVL_BY_LEVEL:
        raise ValueError(f'level must be one of {sorted(AGGLVL_BY_LEVEL)}, got {level}')
    level_codes = sorted(code for code in codebook_codes if len(code) == level)
    rows = cells.filter(pl.col('agglvl_code') == AGGLVL_BY_LEVEL[level]).with_columns(
        code=pl.col('industry_code').replace(QCEW_SECTOR_CODES)
    )
    frames = [
        rows.filter(pl.col('code').is_in(level_codes)).with_columns(source=pl.lit('published'))
    ]
    if level == 6:
        split = find_split_codes(rows.get_column('industry_code').unique(), level_codes)
        parents = pl.DataFrame(
            {
                'industry_code': [code[:5] for code in split],
                'split_code': list(split)
            },
            schema={
                'industry_code': pl.Utf8,
                'split_code': pl.Utf8
            },
        )
        five_digit = cells.filter(pl.col('agglvl_code') == AGGLVL_BY_LEVEL[5])
        frames.append(
            five_digit.join(parents, on='industry_code').with_columns(
                code=pl.col('split_code'), source=pl.lit('five_digit_parent')
            )
        )
    level_frame = pl.concat([frame.select(CELL_COLUMNS) for frame in frames])
    if level_frame.select('code', 'year').is_duplicated().any():
        raise ValueError(f'level {level}: more than one national private cell for a code and year')
    return level_frame.sort('code', 'year')

def usable_cell() -> pl.Expr:
    '''Disclosed, with positive employment, establishments and wages.'''

    return (
        (pl.col('disclosure_code') == '') & (pl.col('emp') > 0) & (pl.col('estabs') > 0)
        & (pl.col('wages') > 0)
    )

def population(cells: pl.DataFrame) -> Tuple[str, ...]:
    '''Codes with a usable cell in every window year, sorted.'''

    years = cells.filter(usable_cell() & pl.col('year').is_in(WINDOW_YEARS))
    complete = years.group_by('code').agg(pl.col('year').n_unique().alias('years'))
    codes = complete.filter(pl.col('years') == len(WINDOW_YEARS)).get_column('code')
    return tuple(sorted(codes.to_list()))

def panel_rows(cells: pl.DataFrame, codes: Collection[str]) -> pl.DataFrame:
    '''
    One row per code and feature year (D7): year-t covariates and the year-(t + 1) outcome.

    Raises:
        ValueError: If a code lacks a usable cell in a window year.
    '''

    usable = cells.filter(usable_cell() & pl.col('code').is_in(list(codes)))
    features = usable.filter(pl.col('year').is_in(FEATURE_YEARS)).select(
        'code',
        feature_year=pl.col('year'),
        outcome_year=pl.col('year') + 1,
        log_estabs=pl.col('estabs').cast(pl.Float64).log(),
        log_wages=pl.col('wages').cast(pl.Float64).log(),
    )
    outcomes = usable.select(
        'code', outcome_year=pl.col('year'), outcome=pl.col('emp').cast(pl.Float64).log()
    )
    rows = features.join(outcomes, on=['code', 'outcome_year'], how='inner')
    expected = len(set(codes)) * len(FEATURE_YEARS)
    if rows.height != expected:
        raise ValueError(f'{rows.height:,} panel rows for {expected:,} code-years')
    return rows.select(ROW_COLUMNS).sort('code', 'feature_year')
