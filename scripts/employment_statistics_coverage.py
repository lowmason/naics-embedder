# -------------------------------------------------------------------------------------------------
# Employment-statistics coverage (roadmap Stage 1)
# -------------------------------------------------------------------------------------------------
'''
Measure how much QCEW disclosure suppression removes from the NAICS 2022 six-digit universe.

Reads the QCEW annual-average single files, the Open Data Access national slices and the
supervision bundle's codebook, then writes the tables and the pre-registered Req 2 decision that
specs/findings/employment-statistics-coverage.md records. A suppressed cell is never read as zero.

    uv run python scripts/employment_statistics_coverage.py manifest --qcew-dir DIR
    uv run python scripts/employment_statistics_coverage.py run --qcew-dir DIR --codebook PATH \
        --final-years 2022 2023 2024 2025 --out-dir DIR
'''

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

CODEBOOK_SHA256 = '5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b'
WINDOW = (2022, 2023, 2024, 2025)
VINTAGE_CHECK_YEAR = 2021
PRIVATE = '5'
OWNERSHIPS = ('1', '2', '3', '5')

# Six-digit and five-digit aggregation levels per grain, checked against agglevel_titles.csv.
GRAINS = {
    'national': ('18', '17'),
    'state': ('58', '57'),
    'county': ('78', '77'),
    'msa': ('48', '47'),
}
# MSA is reported but never a candidate: no six-digit rows from 2025, and its codes break
# between 2023 (OMB 13-01) and 2024 (OMB 23-01).
CANDIDATE_GRAINS = ('national', 'state', 'county')
NATIONAL_TOTAL_AGGLVL = '11'
TOTAL_INDUSTRY = '10'
NATIONAL_AREA = 'US000'
UNKNOWN_COUNTY_SUFFIX = '999'
CONNECTICUT_LEGACY = ('09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015')
USED_AGGLVLS = (NATIONAL_TOTAL_AGGLVL, *(level for pair in GRAINS.values() for level in pair))

# Pre-registered in plan 3. Never tune these after reading the data.
SURVIVAL_FLOOR = 506  # 50 % of the codebook's 1,012 six-digit codes
ASK_BAND = (405, 607)  # 40 % to 60 %: a deciding count in here goes to the user
MIN_WINDOW_YEARS = 3

DISCLOSED = 'disclosed'
SUPPRESSED = 'suppressed'
OTHER = 'other'
ABSENT = 'absent'

ESTABS_COLUMNS = ('annual_avg_estabs', 'annual_avg_estabs_count')
KEY_COLUMNS = (
    'area_fips',
    'own_code',
    'industry_code',
    'agglvl_code',
    'size_code',
    'year',
    'qtr',
    'disclosure_code',
)
CELL_COLUMNS = (
    'year',
    'area_fips',
    'own_code',
    'code',
    'source',
    'disclosure_code',
    'status',
    'estabs_status',
    'estabs',
    'emp',
    'wages',
)
SERIES = (('employment', 'status'), ('wages', 'status'), ('establishments', 'estabs_status'))

BLS = 'https://data.bls.gov/cew'
SOURCES = {
    **{
        f'{year}_annual_singlefile.zip': f'{BLS}/data/files/{year}/csv/{year}_annual_singlefile.zip'
        for year in WINDOW
    },
    **{
        f'{year}_US000_annual.csv': f'{BLS}/data/api/{year}/a/area/US000.csv'
        for year in (VINTAGE_CHECK_YEAR, *WINDOW)
    },
    'industry_titles.csv': f'{BLS}/doc/titles/industry/industry_titles.csv',
    'agglevel_titles.csv': f'{BLS}/doc/titles/agglevel/agglevel_titles.csv',
    'area_titles.csv': f'{BLS}/doc/titles/area/area_titles.csv',
    'ownership_titles.csv': f'{BLS}/doc/titles/ownership/ownership_titles.csv',
}

# -------------------------------------------------------------------------------------------------
# Code universe
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    '''The codebook's six-digit codes, and those that are their five-digit parent's only child.'''

    six_digit: tuple[str, ...]
    only_children: frozenset[str]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def load_universe(path: Path, expected_sha256: str = CODEBOOK_SHA256) -> Universe:
    '''Load a supervision-bundle codebook's six-digit codes after checking the file's hash.'''
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f'{path}: sha256 {digest} does not match {expected_sha256}')
    codes = pl.read_parquet(path).get_column('code').to_list()
    six_digit = tuple(sorted(code for code in codes if len(code) == 6))
    siblings: dict[str, int] = {}
    for code in six_digit:
        siblings[code[:5]] = siblings.get(code[:5], 0) + 1
    only_children = frozenset(code for code in six_digit if siblings[code[:5]] == 1)
    return Universe(six_digit=six_digit, only_children=only_children)

# -------------------------------------------------------------------------------------------------
# Reading QCEW files
# -------------------------------------------------------------------------------------------------

def resolve_estabs_column(header: Sequence[str]) -> str:
    '''Name of the annual-average establishment column; BLS documents two spellings.'''
    present = [name for name in ESTABS_COLUMNS if name in header]
    if len(present) != 1:
        raise ValueError(f'expected exactly one of {ESTABS_COLUMNS} in the header, found {present}')
    return present[0]

def read_annual_csv(data: bytes, agglvls: Collection[str] = USED_AGGLVLS) -> pl.DataFrame:
    '''Read a QCEW annual-average CSV with every code column kept as a string.

    Returns the key columns plus estabs, emp and wages as Int64, for annual all-size rows at the
    given aggregation levels. A blank disclosure code reads as ''.
    '''
    header = pl.read_csv(data, n_rows=0, infer_schema=False).columns
    estabs = resolve_estabs_column(header)
    needed = [*KEY_COLUMNS, estabs, 'annual_avg_emplvl', 'total_annual_wages']
    missing = [name for name in needed if name not in header]
    if missing:
        raise ValueError(f'QCEW annual CSV lacks columns {missing}')
    renamed = pl.read_csv(data, columns=needed, infer_schema=False).rename(
        {
            estabs: 'estabs',
            'annual_avg_emplvl': 'emp',
            'total_annual_wages': 'wages',
        }
    )
    # yapf: disable
    return (
        renamed
        .with_columns(pl.col(*KEY_COLUMNS).fill_null('').str.strip_chars())
        .with_columns(pl.col('estabs', 'emp', 'wages').str.strip_chars().cast(pl.Int64))
        .filter(
            pl.col('agglvl_code').is_in(list(agglvls))
            & (pl.col('qtr') == 'A')
            & (pl.col('size_code') == '0')
        )
        .with_columns(pl.col('year').cast(pl.Int32))
    )
    # yapf: enable

def read_singlefile_zip(path: Path) -> pl.DataFrame:
    '''Read the one CSV inside a QCEW annual single-file zip.'''
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(members) != 1:
            raise ValueError(f'{path.name}: expected one CSV member, found {members}')
        data = archive.read(members[0])
    return read_annual_csv(data)

# -------------------------------------------------------------------------------------------------
# Cells
# -------------------------------------------------------------------------------------------------

def national_six_digit_codes(frame: pl.DataFrame) -> set[str]:
    '''Industry codes on a frame's national six-digit rows, in any ownership.'''
    rows = frame.filter(pl.col('agglvl_code') == GRAINS['national'][0])
    return set(rows.get_column('industry_code').to_list())

def find_split_codes(published: Collection[str], universe: Universe) -> tuple[str, ...]:
    '''Codebook codes QCEW replaces with BLS residential (xxxxx1) and nonresidential (xxxxx2) codes.

    Each such code must be its five-digit parent's only child: the parent's QCEW row then carries
    the code's own value, under the parent's disclosure status.
    '''
    published = set(published)
    codebook = set(universe.six_digit)
    split = []
    for code in universe.six_digit:
        bls_children = {code[:5] + '1', code[:5] + '2'} - codebook
        if code in published or not bls_children & published:
            continue
        if code not in universe.only_children:
            raise ValueError(f'{code} is split into BLS codes but is not an only child')
        split.append(code)
    return tuple(split)

def disclosure_status() -> pl.Expr:
    '''Employment and wage status: one disclosure flag governs both series.'''
    flag = pl.col('disclosure_code')
    # yapf: disable
    return (
        pl.when(flag == '').then(pl.lit(DISCLOSED))
        .when(flag == 'N').then(pl.lit(SUPPRESSED))
        .otherwise(pl.lit(OTHER))
    )
    # yapf: enable

def estabs_status() -> pl.Expr:
    '''Establishment status: on an N row the count stands unless suppression zero-filled it.'''
    flag = pl.col('disclosure_code')
    # yapf: disable
    return (
        pl.when(flag == '').then(pl.lit(DISCLOSED))
        .when((flag == 'N') & (pl.col('estabs') > 0)).then(pl.lit(DISCLOSED))
        .when(flag == 'N').then(pl.lit(SUPPRESSED))
        .otherwise(pl.lit(OTHER))
    )
    # yapf: enable

def grain_cells(
    frame: pl.DataFrame, universe: Universe, split: Collection[str], grain: str
) -> pl.DataFrame:
    '''Every published cell of a codebook six-digit code at one grain, with its status.

    Directly published codes come from the grain's six-digit level; each split code comes from
    its five-digit parent's row. A code with no row is absent: it appears only in code-level
    counts. County rows for unknown or undefined locations (county part 999) are not areas.
    '''
    six_level, five_level = GRAINS[grain]
    direct_codes = sorted(set(universe.six_digit) - set(split))
    direct = frame.filter(
        (pl.col('agglvl_code') == six_level) & pl.col('industry_code').is_in(direct_codes)
    ).with_columns(code=pl.col('industry_code'), source=pl.lit('six_digit'))
    parents = pl.DataFrame(
        {
            'industry_code': [code[:5] for code in split],
            'code': list(split),
        },
        schema={
            'industry_code': pl.String,
            'code': pl.String,
        },
    )
    five_digit = frame.filter(pl.col('agglvl_code') == five_level)
    recovered = five_digit.join(parents, on='industry_code').with_columns(
        source=pl.lit('five_digit_parent')
    )
    cells = pl.concat([direct, recovered], how='diagonal')
    if grain == 'county':
        cells = cells.filter(~pl.col('area_fips').str.ends_with(UNKNOWN_COUNTY_SUFFIX))
    labelled = cells.with_columns(status=disclosure_status(), estabs_status=estabs_status())
    return labelled.select(CELL_COLUMNS)
