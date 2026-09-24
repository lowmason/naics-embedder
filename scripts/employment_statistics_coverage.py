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

import argparse
import hashlib
import json
import logging
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Mapping, Sequence

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

# -------------------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------------------

def code_status_counts(cells: pl.DataFrame, universe: Universe, year: int,
                       own: str) -> list[dict[str, object]]:
    '''National grain: how many codebook codes are disclosed, suppressed, other or absent.'''
    subset = cells.filter((pl.col('year') == year) & (pl.col('own_code') == own))
    if not subset.get_column('code').is_unique().all():
        raise ValueError(f'{year} own {own}: more than one national cell for a code')
    recovered = subset.filter(pl.col('source') == 'five_digit_parent').height
    rows = []
    for series, column in SERIES:
        counts = dict(subset.group_by(column).len().iter_rows())
        rows.append(
            {
                'year': year,
                'own_code': own,
                'series': series,
                DISCLOSED: counts.get(DISCLOSED, 0),
                SUPPRESSED: counts.get(SUPPRESSED, 0),
                OTHER: counts.get(OTHER, 0),
                ABSENT: len(universe.six_digit) - subset.height,
                'recovered_via_parent': recovered,
                'suppressed_share': counts.get(SUPPRESSED, 0) / len(universe.six_digit),
            }
        )
    return rows

def area_coverage(cells: pl.DataFrame, universe: Universe, grain: str,
                  year: int) -> dict[str, object]:
    '''Private cells at one grain and year: suppressed shares and code-level survival.

    Cell shares divide by the published private cells; code shares divide by the codebook's
    six-digit codes, so an absent code counts as having no usable cell.
    '''
    subset = cells.filter((pl.col('year') == year) & (pl.col('own_code') == PRIVATE))
    usable = subset.filter(pl.col('status') == DISCLOSED)
    per_code = usable.group_by('code').agg(pl.col('area_fips').n_unique().alias('areas'))
    published = subset.height
    suppressed = subset.filter(pl.col('status') == SUPPRESSED).height
    estabs_suppressed = subset.filter(pl.col('estabs_status') == SUPPRESSED).height
    codes_published = subset.get_column('code').n_unique()
    without_usable = len(universe.six_digit) - per_code.height
    return {
        'grain': grain,
        'year': year,
        'areas': subset.get_column('area_fips').n_unique(),
        'published_cells': published,
        'suppressed_cells': suppressed,
        'other_cells': subset.filter(pl.col('status') == OTHER).height,
        'suppressed_share': suppressed / published if published else None,
        'estabs_suppressed_cells': estabs_suppressed,
        'estabs_suppressed_share': estabs_suppressed / published if published else None,
        'codes_usable': per_code.height,
        'codes_usable_2plus_areas': per_code.filter(pl.col('areas') >= 2).height,
        'codes_published_never_usable': codes_published - per_code.height,
        'codes_absent': len(universe.six_digit) - codes_published,
        'codes_without_usable': without_usable,
        'share_without_usable': without_usable / len(universe.six_digit),
        'median_usable_areas': per_code.get_column('areas').median() if per_code.height else None,
    }

def size_by_status(cells: pl.DataFrame, grain: str, year: int) -> list[dict[str, object]]:
    '''Establishment counts of private disclosed and suppressed cells: is suppression selective?'''
    subset = cells.filter(
        (pl.col('year') == year) & (pl.col('own_code') == PRIVATE)
        & pl.col('status').is_in([DISCLOSED, SUPPRESSED])
    )
    summary = subset.group_by('status').agg(
        pl.len().alias('cells'),
        pl.col('estabs').median().alias('median_estabs'),
        pl.col('estabs').quantile(0.9).alias('p90_estabs'),
    ).sort('status')
    return [{'grain': grain, 'year': year, **row} for row in summary.to_dicts()]

def vintage_report(
    published_by_year: Mapping[int, Collection[str]], universe: Universe, split: Collection[str]
) -> list[dict[str, object]]:
    '''Per year, national six-digit codes outside the codebook and codebook codes unpublished.

    BLS residential and nonresidential children of the split codes and 999999 (unclassified) are
    expected outside the codebook; anything else there means a different NAICS vintage.
    '''
    codebook = set(universe.six_digit)
    expected_extra = {code[:5] + digit for code in split for digit in '12'} | {'999999'}
    rows = []
    for year in sorted(published_by_year):
        published = set(published_by_year[year])
        outside = sorted(published - codebook - expected_extra)
        unpublished = sorted(codebook - published - set(split))
        rows.append(
            {
                'year': year,
                'published_six_digit': len(published),
                'outside_codebook': len(outside),
                'outside_examples': outside[:12],
                'codebook_unpublished': len(unpublished),
                'unpublished_examples': unpublished[:12],
            }
        )
    return rows

def private_gaps(cells: pl.DataFrame, universe: Universe, year: int) -> list[dict[str, object]]:
    '''National codes with no private cell in a year, and the ownerships that do have one.'''
    subset = cells.filter(pl.col('year') == year)
    private = set(subset.filter(pl.col('own_code') == PRIVATE).get_column('code').to_list())
    owners = dict(
        subset.group_by('code').agg(pl.col('own_code').unique().sort().alias('owners')).iter_rows()
    )
    return [
        {
            'code': code,
            'ownerships_with_cells': owners.get(code, [])
        } for code in universe.six_digit if code not in private
    ]

def excluded_codes(cells: pl.DataFrame, universe: Universe,
                   window: Sequence[int]) -> list[dict[str, object]]:
    '''Codes with no usable private cell anywhere in the window at one grain, with the reason.'''
    private = cells.filter((pl.col('own_code') == PRIVATE) & pl.col('year').is_in(list(window)))
    published = set(private.get_column('code').to_list())
    usable = set(private.filter(pl.col('status') == DISCLOSED).get_column('code').to_list())
    return [
        {
            'code': code,
            'reason': 'private cells never usable' if code in published else 'no private cell',
        } for code in universe.six_digit if code not in usable
    ]

def connecticut_areas(county_cells: pl.DataFrame) -> list[dict[str, object]]:
    '''Connecticut county-equivalents per year: legacy counties or planning regions (from 2024).'''
    rows = []
    for year in sorted(set(county_cells.get_column('year').to_list())):
        areas = set(
            county_cells.filter(
                (pl.col('year') == year)
                & pl.col('area_fips').str.starts_with('09')
            ).get_column('area_fips').to_list()
        )
        regions = {area for area in areas if '09110' <= area <= '09190'}
        rows.append(
            {
                'year': year,
                'legacy_counties': len(areas & set(CONNECTICUT_LEGACY)),
                'planning_regions': len(regions),
            }
        )
    return rows

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def _disclosed_private(frame: pl.DataFrame, level: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col('agglvl_code') == level) & (pl.col('own_code') == PRIVATE)
        & (pl.col('disclosure_code') == '')
    )

def _nested_excess(detail: pl.DataFrame, parents: pl.DataFrame, year: int, child: str,
                   parent: str) -> list[str]:
    '''Detail cells may not sum past their parent cell (employment allows annual-average rounding).'''
    sums = detail.group_by('parent', 'industry_code').agg(
        pl.col('emp').sum(),
        pl.col('wages').sum(),
        pl.len().alias('cells'),
    )
    joined = sums.join(
        parents.select('parent', 'industry_code', 'emp', 'wages'),
        on=['parent', 'industry_code'],
        suffix='_parent',
    )
    bad = joined.filter(
        (pl.col('emp') > pl.col('emp_parent') + pl.col('cells') / 2 + 1)
        | (pl.col('wages') > pl.col('wages_parent'))
    )
    failures = [
        f'{year}: {child} cells exceed their {parent} cell for {row["parent"]} '
        f'{row["industry_code"]}' for row in bad.head(20).to_dicts()
    ]
    if bad.height > 20:
        failures.append(f'{year}: {bad.height} {child}-over-{parent} excesses in all')
    return failures

def check_invariants(frame: pl.DataFrame, year: int) -> list[str]:
    '''Describe every failed invariant of one year's single file (empty when all hold).

    Disclosed private detail never sums past a published total: six-digit cells against the
    national private total, states (50 plus DC) against the nation per code, and counties
    (unknown locations included) against their state per code.
    '''
    total = frame.filter(
        (pl.col('agglvl_code') == NATIONAL_TOTAL_AGGLVL)
        & (pl.col('own_code') == PRIVATE)
        & (pl.col('industry_code') == TOTAL_INDUSTRY)
        & (pl.col('area_fips') == NATIONAL_AREA)
    )
    if total.height != 1:
        return [f'{year}: expected one national private total row, found {total.height}']
    failures = []
    national = _disclosed_private(frame, GRAINS['national'][0])
    for column in ('emp', 'wages'):
        detail, whole = national.get_column(column).sum(), total.get_column(column).item()
        if detail > whole:
            failures.append(f'{year}: disclosed six-digit {column} {detail} exceeds total {whole}')
    states = _disclosed_private(frame, GRAINS['state'][0])
    failures += _nested_excess(
        states.filter(pl.col('area_fips').str.slice(0, 2).cast(pl.Int32) <= 56).with_columns(
            parent=pl.lit(NATIONAL_AREA)
        ),
        national.rename({'area_fips': 'parent'}),
        year,
        'state',
        'national',
    )
    counties = _disclosed_private(frame, GRAINS['county'][0])
    failures += _nested_excess(
        counties.with_columns(parent=pl.col('area_fips').str.slice(0, 2) + '000'),
        states.rename({'area_fips': 'parent'}),
        year,
        'county',
        'state',
    )
    return failures

def singlefile_header(path: Path) -> list[str]:
    '''Column names on the first line of a QCEW annual single-file zip.'''
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith('.csv'))
        with archive.open(member) as handle:
            first = handle.readline().decode()
    return [name.strip().strip('"') for name in first.strip().split(',')]

def file_conventions(frame: pl.DataFrame, year: int) -> dict[str, object]:
    '''What one annual file's six-digit rows show about the conventions the tables rely on.'''
    six = frame.filter(pl.col('agglvl_code').is_in([six for six, _ in GRAINS.values()]))
    codes = dict(six.group_by('disclosure_code').len().sort('disclosure_code').iter_rows())
    suppressed = six.filter(pl.col('disclosure_code') == 'N')
    shows_values = (pl.col('emp') != 0) | (pl.col('wages') != 0)
    return {
        'year': year,
        'disclosure_codes': {
            code or 'blank': count
            for code, count in codes.items()
        },
        'own_code_0_rows': six.filter(pl.col('own_code') == '0').height,
        'suppressed_rows': suppressed.height,
        'suppressed_rows_with_emp_or_wages': suppressed.filter(shows_values).height,
        'suppressed_rows_with_estabs': suppressed.filter(pl.col('estabs') > 0).height,
    }

def compare_national_slices(frame: pl.DataFrame, national_slice: pl.DataFrame,
                            year: int) -> list[str]:
    '''The single file and the Open Data Access US000 slice must agree on every national row.'''
    keys = ['own_code', 'industry_code', 'agglvl_code']
    values = ['disclosure_code', 'estabs', 'emp', 'wages']
    levels = [NATIONAL_TOTAL_AGGLVL, *GRAINS['national']]
    left = frame.filter(
        (pl.col('area_fips') == NATIONAL_AREA)
        & pl.col('agglvl_code').is_in(levels)
    ).select(*keys, *values)
    right = national_slice.filter(pl.col('agglvl_code').is_in(levels)).select(*keys, *values)
    joined = left.join(right, on=keys, how='full', suffix='_slice', coalesce=True)
    differs = pl.any_horizontal(
        [pl.col(name).ne_missing(pl.col(f'{name}_slice')) for name in values]
    )
    mismatched = joined.filter(differs).height
    if mismatched:
        return [f'{year}: {mismatched} national rows differ between the single file and the slice']
    return []

# -------------------------------------------------------------------------------------------------
# Decision rule (pre-registered in plan 3)
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GrainSummary:
    '''What the decision rule reads about one grain's private cells over the window.'''

    grain: str
    complete: bool  # six-digit rows in every window year
    mean_suppressed_share: float  # suppressed / published cells, averaged over the years
    seen_by_year: int  # codes usable in the last year and in an earlier one
    seen_by_area: int  # codes usable in two or more areas in the last year (0 when national)
    time_eligible: int  # codes with a same-area usable pair in the last pair and an earlier one
    heldout_population: int  # codes usable at least once in the window

    @property
    def seen(self) -> int:
        return max(self.seen_by_year, self.seen_by_area)

@dataclass(frozen=True)
class Decision:
    branch: str  # 'A' time-respecting, 'B' cross-sectional, 'C' held-out codes only
    grain: str | None
    time_respecting: bool
    seen_regime: bool
    needs_user: bool  # a deciding count fell inside the ask band: stop and ask
    reasons: tuple[str, ...]

def summarize_grain(cells: pl.DataFrame, grain: str, window: Sequence[int]) -> GrainSummary:
    '''Reduce one grain's cells to the counts the decision rule reads.'''
    private = cells.filter(pl.col('own_code') == PRIVATE)
    disclosed = private.filter(pl.col('status') == DISCLOSED)
    usable = disclosed.select('year', 'area_fips', 'code').unique()
    years = sorted(window)
    shares = []
    for year in years:
        subset = private.filter(pl.col('year') == year)
        if subset.height:
            shares.append(subset.filter(pl.col('status') == SUPPRESSED).height / subset.height)

    def codes(frame: pl.DataFrame) -> set[str]:
        return set(frame.get_column('code').to_list())

    def paired(start: int) -> set[str]:
        first = usable.filter(pl.col('year') == start).select('area_fips', 'code')
        second = usable.filter(pl.col('year') == start + 1).select('area_fips', 'code')
        return codes(first.join(second, on=['area_fips', 'code']))

    last = years[-1]
    in_last = codes(usable.filter(pl.col('year') == last))
    seen_by_year = len(in_last & codes(usable.filter(pl.col('year').is_in(years[:-1]))))
    seen_by_area = 0
    if grain != 'national':
        areas = usable.filter(pl.col('year') == last).group_by('code').agg(
            pl.col('area_fips').n_unique().alias('areas')
        )
        seen_by_area = areas.filter(pl.col('areas') >= 2).height
    starts = [year for year in years[:-1] if year + 1 in years]
    time_eligible = 0
    if len(starts) >= 2:
        earlier = set().union(*(paired(start) for start in starts[:-1]))
        time_eligible = len(paired(starts[-1]) & earlier)
    return GrainSummary(
        grain=grain,
        complete=len(shares) == len(years),
        mean_suppressed_share=sum(shares) / len(shares) if shares else 1.0,
        seen_by_year=seen_by_year,
        seen_by_area=seen_by_area,
        time_eligible=time_eligible,
        heldout_population=usable.get_column('code').n_unique(),
    )

def decide(
    summaries: Sequence[GrainSummary],
    window: Sequence[int],
    final_years: Collection[int],
    floor: int = SURVIVAL_FLOOR,
    band: tuple[int, int] = ASK_BAND,
) -> Decision:
    '''Apply plan 3's pre-registered rule.

    A candidate is a CANDIDATE_GRAINS grain with six-digit rows in every window year; it
    survives when at least `floor` codes can run the seen-code regime. The chosen grain is the
    surviving candidate with the lowest mean suppressed share (ties: national, state, county).
    The outcome is time-respecting when the window is at least MIN_WINDOW_YEARS consecutive
    final years and at least `floor` codes are time-eligible at the chosen grain. A deciding
    count inside `band` sets needs_user.
    '''

    def in_band(count: int) -> bool:
        return band[0] <= count <= band[1]

    order = {grain: rank for rank, grain in enumerate(CANDIDATE_GRAINS)}
    candidates = sorted(
        (summary for summary in summaries if summary.grain in order and summary.complete),
        key=lambda summary: (summary.mean_suppressed_share, order[summary.grain]),
    )
    reasons = [
        f'{s.grain}: {s.seen} codes can run the seen-code regime (floor {floor}); '
        f'mean suppressed share {s.mean_suppressed_share:.4f}' for s in candidates
    ]
    reasons += [
        f'{s.grain}: not a candidate (not a candidate grain, or no six-digit rows in a window year)'
        for s in summaries if s not in candidates
    ]
    surviving = [summary for summary in candidates if summary.seen >= floor]
    if not surviving:
        best = max((summary.seen for summary in candidates), default=0)
        reasons.append('no grain below the code survives suppression')
        return Decision('C', None, False, False, in_band(best), tuple(reasons))
    chosen = surviving[0]
    needs_user = in_band(chosen.seen)
    for summary in candidates[:candidates.index(chosen)]:
        if in_band(summary.seen):
            needs_user = True
            reasons.append(
                f'{summary.grain} ranks ahead of {chosen.grain} with {summary.seen} '
                f'codes, inside the ask band {band}'
            )
    years = sorted(window)
    consecutive = years == list(range(years[0], years[0] + len(years)))
    final = all(year in final_years for year in years)
    window_ok = len(years) >= MIN_WINDOW_YEARS and consecutive and final
    if not window_ok:
        reasons.append(
            f'window {years} is not {MIN_WINDOW_YEARS} or more consecutive final years '
            f'(final: {sorted(final_years)})'
        )
    elif in_band(chosen.time_eligible):
        needs_user = True
    reasons.append(
        f'{chosen.grain}: {chosen.time_eligible} codes are time-eligible (floor {floor})'
    )
    time_respecting = window_ok and chosen.time_eligible >= floor
    return Decision(
        'A' if time_respecting else 'B', chosen.grain, time_respecting, True, needs_user,
        tuple(reasons)
    )

BRANCH_TEXT = {
    'A': 'The verified window supports a time-respecting outcome (the outcome dated after the '
    'features, with splits by time), so the panel includes one.',
    'B': 'The verified window does not support a time-respecting outcome, so the panel is '
    'cross-sectional.',
    'C': 'No grain below the code survives suppression, so the panel is held-out-codes only and '
    'the one-hot comparison could not run.',
}
ROW_GRAIN = {
    'national': 'a six-digit code in a reference year (national, private ownership)',
    'state': 'a six-digit code in a state in a reference year (private ownership)',
    'county': 'a six-digit code in a county in a reference year (private ownership)',
}

def render_decision(
    decision: Decision, summaries: Sequence[GrainSummary], window: Sequence[int]
) -> str:
    '''The finding's "Decision for Stage 3" block, in fixed wording.'''
    chosen = next((summary for summary in summaries if summary.grain == decision.grain), None)
    lines = [
        '<!-- decision:begin -->',
        f'- **Branch:** {decision.branch}. {BRANCH_TEXT[decision.branch]}',
        f'- **Source:** QCEW annual averages, reference years {", ".join(map(str, window))}, '
        'private ownership (own_code 5).',
    ]
    if chosen is None:
        lines.append('- **Row grain:** one row per code; no grain below the code survives.')
    else:
        lines += [
            f'- **Row grain:** {ROW_GRAIN[chosen.grain]}.',
            f'- **Population:** {chosen.seen} codes for the seen-code regime and '
            f'{chosen.heldout_population} for the held-out-code regime, of the 1,012 six-digit '
            'codes in the codebook.',
        ]
    lines += [
        f'- **Time-respecting outcome:** {"yes" if decision.time_respecting else "no"}.',
        f'- **Seen-code regime:** {"yes" if decision.seen_regime else "no"}.',
        f'- **Rule:** plan 3, survival floor {SURVIVAL_FLOOR} codes, ask band {ASK_BAND[0]} to '
        f'{ASK_BAND[1]}; user review {"required" if decision.needs_user else "not required"}.',
        '- **Reasons:**',
        *[f'  - {reason}' for reason in decision.reasons],
        '<!-- decision:end -->',
    ]
    return '\n'.join(lines) + '\n'

# -------------------------------------------------------------------------------------------------
# Provenance
# -------------------------------------------------------------------------------------------------

def parse_headers(text: str) -> dict[str, str]:
    '''Parse a `curl -D` header dump; after redirects, the last response wins.'''
    blocks = [block for block in text.replace('\r\n', '\n').split('\n\n') if block.strip()]
    status_line, *field_lines = blocks[-1].splitlines()
    fields = {'status': status_line.split()[1]}
    for line in field_lines:
        name, _, value = line.partition(':')
        fields[name.strip().lower()] = value.strip()
    return fields

def build_manifest(qcew_dir: Path, sources: Mapping[str, str] = SOURCES) -> list[dict[str, object]]:
    '''Provenance of each source file: URL, bytes, sha256, Last-Modified and download time.'''
    entries = []
    for name, url in sources.items():
        path = qcew_dir / name
        headers_path = qcew_dir / 'headers' / f'{name}.headers'
        headers = parse_headers(headers_path.read_text())
        size = path.stat().st_size
        if headers['status'] != '200':
            raise ValueError(f'{name}: HTTP status {headers["status"]}')
        if 'content-length' in headers and int(headers['content-length']) != size:
            raise ValueError(
                f'{name}: {size} bytes on disk, Content-Length '
                f'{headers["content-length"]}'
            )
        downloaded = datetime.fromtimestamp(headers_path.stat().st_mtime, tz=timezone.utc)
        entries.append(
            {
                'file': name,
                'url': url,
                'bytes': size,
                'sha256': sha256_file(path),
                'last_modified': headers.get('last-modified'),
                'downloaded_at': downloaded.isoformat(timespec='seconds'),
            }
        )
    return entries

# -------------------------------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------------------------------

def _cell(value: object) -> str:
    if isinstance(value, float):
        return f'{value:.4f}'
    if isinstance(value, (list, tuple)):
        return ', '.join(map(str, value)) or '-'
    return '-' if value is None else str(value)

def markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return '_none_\n'
    columns = list(rows[0])
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join(_cell(row[column]) for column in columns) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'

def render_tables(report: Mapping[str, object]) -> str:
    failures = [{'failure': failure} for failure in report['failures']]
    split = [{'code': code} for code in report['split_codes']]
    sections = [
        ('Invariant failures', failures),
        ('File conventions (six-digit rows)', report['conventions']),
        ('Vintage check (national six-digit codes)', report['vintage']),
        ('Split codes recovered from their five-digit parent', split),
        ('National grain: codebook codes by status', report['national_status']),
        ('Private cells by grain and year', report['area_coverage']),
        ('Establishments of disclosed and suppressed private cells', report['size_by_status']),
        ('Codes with no private national cell (last year)', report['private_gaps']),
        ('Connecticut county-equivalents', report['connecticut']),
        ('MSA six-digit rows per year', report['msa_rows']),
        ('Decision inputs', report['summaries']),
        ('Codes excluded at the chosen grain', report['excluded']),
    ]
    return '\n'.join(f'### {title}\n\n{markdown_table(rows)}' for title, rows in sections)

# -------------------------------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------------------------------

def run(
    qcew_dir: Path,
    codebook: Path,
    final_years: Collection[int],
    out_dir: Path,
    window: Sequence[int] = WINDOW,
    codebook_sha256: str = CODEBOOK_SHA256,
    floor: int = SURVIVAL_FLOOR,
    band: tuple[int, int] = ASK_BAND,
) -> tuple[Decision, list[str]]:
    '''Compute every table and the decision; write coverage.json, tables.md and decision.md.'''
    universe = load_universe(codebook, codebook_sha256)
    slices = {
        year: read_annual_csv((qcew_dir / f'{year}_US000_annual.csv').read_bytes())
        for year in (VINTAGE_CHECK_YEAR, *window)
    }
    published = {year: national_six_digit_codes(frame) for year, frame in slices.items()}
    split = find_split_codes(published[max(window)], universe)
    failures = [
        f'{year}: split codes {codes} differ from {max(window)}' for year in window
        if (codes := find_split_codes(published[year], universe)) != split
    ]
    parts: dict[str, list[pl.DataFrame]] = {grain: [] for grain in GRAINS}
    conventions = []
    for year in window:
        logger.info('reading %s', year)
        path = qcew_dir / f'{year}_annual_singlefile.zip'
        frame = read_singlefile_zip(path)
        estabs_column = resolve_estabs_column(singlefile_header(path))
        conventions.append({**file_conventions(frame, year), 'estabs_column': estabs_column})
        failures += check_invariants(frame, year)
        failures += compare_national_slices(frame, slices[year], year)
        for grain in GRAINS:
            parts[grain].append(grain_cells(frame, universe, split, grain))
    cells = {grain: pl.concat(frames) for grain, frames in parts.items()}
    summaries = [summarize_grain(cells[grain], grain, window) for grain in GRAINS]
    decision = decide(summaries, window, final_years, floor, band)
    chosen = decision.grain or 'national'
    msa_counts = {year: cells['msa'].filter(pl.col('year') == year).height for year in window}
    msa_rows = [{'year': year, 'rows': count} for year, count in msa_counts.items()]
    summary_rows = [{**asdict(summary), 'seen': summary.seen} for summary in summaries]
    report = {
        'window': list(window),
        'final_years': sorted(final_years),
        'codebook_sha256': codebook_sha256,
        'failures': failures,
        'conventions': conventions,
        'split_codes': list(split),
        'vintage': vintage_report(published, universe, split),
        'national_status': [
            row for year in window for own in OWNERSHIPS
            for row in code_status_counts(cells['national'], universe, year, own)
        ],
        'area_coverage': [
            area_coverage(cells[grain], universe, grain, year) for grain in GRAINS
            for year in window
        ],
        'size_by_status': [
            row for grain in GRAINS for year in window
            for row in size_by_status(cells[grain], grain, year)
        ],
        'private_gaps': private_gaps(cells['national'], universe, max(window)),
        'connecticut': connecticut_areas(cells['county']),
        'msa_rows': msa_rows,
        'summaries': summary_rows,
        'decision': asdict(decision),
        'excluded': excluded_codes(cells[chosen], universe, window),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'coverage.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out_dir / 'tables.md').write_text(render_tables(report))
    (out_dir / 'decision.md').write_text(render_decision(decision, summaries, window))
    for failure in failures:
        logger.warning('invariant failed: %s', failure)
    return decision, failures

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='QCEW six-digit coverage for roadmap Stage 1.')
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest', help='record provenance of the downloaded files')
    manifest.add_argument('--qcew-dir', type=Path, required=True)
    coverage = commands.add_parser('run', help='compute the tables and the Req 2 decision')
    coverage.add_argument('--qcew-dir', type=Path, required=True)
    coverage.add_argument('--codebook', type=Path, required=True)
    coverage.add_argument('--final-years', type=int, nargs='+', required=True)
    coverage.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.command == 'manifest':
        path = args.qcew_dir / 'MANIFEST.json'
        path.write_text(json.dumps(build_manifest(args.qcew_dir), indent=2) + '\n')
        logger.info('wrote %s', path)
        return 0
    decision, failures = run(args.qcew_dir, args.codebook, args.final_years, args.out_dir)
    logger.info('branch %s at grain %s', decision.branch, decision.grain)
    if failures or decision.needs_user:
        review = 'required' if decision.needs_user else 'not required'
        logger.warning('stop and ask: %d invariant failures; user review %s', len(failures), review)
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
