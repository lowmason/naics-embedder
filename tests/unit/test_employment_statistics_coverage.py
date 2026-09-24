import csv
import hashlib
import importlib.util
import io
import json
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

# -------------------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------------------

def test_code_status_counts_use_the_whole_universe(universe, frames):
    cells = _cells(frames, universe, 'national')
    rows = {row['series']: row for row in esc.code_status_counts(cells, universe, 2024, '5')}
    employment = rows['employment']
    assert (employment['disclosed'], employment['suppressed'], employment['other']) == (2, 2, 1)
    assert employment['absent'] == 2
    assert employment['recovered_via_parent'] == 2
    establishments = rows['establishments']
    assert (establishments['disclosed'], establishments['suppressed']) == (4, 0)

def test_area_coverage_counts_cells_and_codes(universe, frames):
    row = esc.area_coverage(_cells(frames, universe, 'state'), universe, 'state', 2024)
    assert (row['areas'], row['published_cells'], row['suppressed_cells']) == (3, 5, 1)
    assert (row['codes_usable'], row['codes_usable_2plus_areas']) == (2, 1)
    assert (row['codes_published_never_usable'], row['codes_absent']) == (1, 4)
    assert (row['codes_without_usable'], row['share_without_usable']) == (5, pytest.approx(5 / 7))
    assert (row['estabs_suppressed_cells'], row['estabs_suppressed_share']) == (0, 0.0)
    assert row['median_usable_areas'] == 2.0

def test_size_by_status_compares_establishment_counts(universe, frames):
    rows = esc.size_by_status(_cells(frames, universe, 'state'), 'state', 2024)
    by_status = {row['status']: row for row in rows}
    assert by_status[esc.DISCLOSED]['cells'] == 4
    assert by_status[esc.DISCLOSED]['median_estabs'] == 3.0
    assert by_status[esc.SUPPRESSED]['median_estabs'] == 2.0

def test_vintage_report_flags_codes_outside_the_codebook(universe, frames):
    published = {
        2021: {'111110', '454110', '238111', '238112'},
        2024: set(frames[2024].filter(pl.col('agglvl_code') == '18')['industry_code'].to_list()),
    }
    rows = {row['year']: row for row in esc.vintage_report(published, universe, ('238110', ))}
    assert rows[2021]['outside_codebook'] == 1
    assert rows[2021]['outside_examples'] == ['454110']
    assert rows[2024]['outside_codebook'] == 2  # 238121 and 238122: 238120 not passed as split
    assert rows[2024]['unpublished_examples'] == ['112130', '238120']

def test_private_gaps_and_exclusions(universe, frames):
    national = _cells(frames, universe, 'national')
    gaps = {
        row['code']: row['ownerships_with_cells']
        for row in esc.private_gaps(national, universe, 2025)
    }
    assert gaps == {'112130': [], '921110': ['1']}
    excluded = {
        row['code']: row['reason']
        for row in esc.excluded_codes(national, universe, WINDOW)
    }
    assert excluded == {
        '112130': 'no private cell',
        '238120': 'private cells never usable',
        '541511': 'private cells never usable',
        '541512': 'private cells never usable',
        '921110': 'no private cell',
    }

def test_connecticut_areas_switch_in_2024(universe, frames):
    rows = {row['year']: row for row in esc.connecticut_areas(_cells(frames, universe, 'county'))}
    assert (rows[2023]['legacy_counties'], rows[2023]['planning_regions']) == (1, 0)
    assert (rows[2024]['legacy_counties'], rows[2024]['planning_regions']) == (0, 1)

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def test_check_invariants_pass_on_consistent_files(frames):
    assert esc.check_invariants(frames[2024], 2024) == []

def test_check_invariants_catch_detail_above_its_total():
    rows = _annual_rows(2024)
    for row in rows:
        if (row['area_fips'], row['industry_code'], row['agglvl_code']) == (
            '01000', '111110', '58'
        ):
            row['annual_avg_emplvl'], row['total_annual_wages'] = '80', '8000'
    failures = esc.check_invariants(esc.read_annual_csv(_csv_bytes(rows)), 2024)
    assert any('state cells exceed their national cell' in failure for failure in failures)
    assert not any('county cells exceed' in failure for failure in failures)

def test_singlefile_header_reads_the_first_line(tmp_path):
    path = tmp_path / '2024_annual_singlefile.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('2024.annual.singlefile.csv', _csv_bytes(_annual_rows(2024)))
    header = esc.singlefile_header(path)
    assert header == list(CSV_COLUMNS)
    assert esc.resolve_estabs_column(header) == 'annual_avg_estabs'

def test_file_conventions_describe_six_digit_rows(frames):
    row = esc.file_conventions(frames[2024], 2024)
    assert row['disclosure_codes'] == {'blank': 12, '-': 1, 'N': 5}
    assert row['own_code_0_rows'] == 0
    assert row['suppressed_rows'] == 5
    assert row['suppressed_rows_with_emp_or_wages'] == 0
    assert row['suppressed_rows_with_estabs'] == 5

def test_compare_national_slices_finds_differences(frames):
    national = frames[2024].filter(pl.col('area_fips') == 'US000')
    assert esc.compare_national_slices(frames[2024], national, 2024) == []
    changed = national.with_columns(
        emp=pl.when(pl.col('industry_code') == '111110').then(99).otherwise(pl.col('emp'))
    )
    assert esc.compare_national_slices(frames[2024], changed, 2024) == [
        '2024: 1 national rows differ between the single file and the slice'
    ]

# -------------------------------------------------------------------------------------------------
# Decision rule
# -------------------------------------------------------------------------------------------------

def _summary(grain, share, seen, time, complete=True, by_area=0):
    return esc.GrainSummary(grain, complete, share, seen, by_area, time, seen)

def test_summarize_grain_counts_what_the_rule_reads(universe, frames):
    national = esc.summarize_grain(_cells(frames, universe, 'national'), 'national', WINDOW)
    assert national.complete
    assert national.mean_suppressed_share == pytest.approx(0.4)
    assert (national.seen_by_year, national.seen_by_area, national.time_eligible) == (2, 0, 2)
    assert national.heldout_population == 2
    county = esc.summarize_grain(_cells(frames, universe, 'county'), 'county', WINDOW)
    assert (county.seen_by_year, county.seen_by_area, county.time_eligible) == (1, 1, 1)
    msa = esc.summarize_grain(_cells(frames, universe, 'msa'), 'msa', WINDOW)
    assert not msa.complete

def test_decide_prefers_the_least_suppressed_surviving_grain():
    summaries = [
        _summary('national', 0.01, 990, 980),
        _summary('state', 0.30, 900, 850),
        _summary('county', 0.55, 700, 600),
        _summary('msa', 0.001, 999, 999, complete=False),
    ]
    decision = esc.decide(summaries, WINDOW, WINDOW)
    assert (decision.branch, decision.grain) == ('A', 'national')
    assert decision.time_respecting and decision.seen_regime and not decision.needs_user

def test_decide_is_cross_sectional_without_three_consecutive_final_years():
    summaries = [_summary('national', 0.01, 990, 980)]
    assert esc.decide(summaries, (2024, 2025), (2024, 2025)).branch == 'B'
    assert esc.decide(summaries, WINDOW, (2022, 2023, 2024)).branch == 'B'
    assert esc.decide(summaries, (2022, 2023, 2025), (2022, 2023, 2025)).branch == 'B'

def test_decide_is_held_out_only_when_no_grain_survives():
    summaries = [_summary('national', 0.2, 300, 300), _summary('state', 0.6, 200, 100)]
    decision = esc.decide(summaries, WINDOW, WINDOW)
    assert (decision.branch, decision.grain, decision.seen_regime) == ('C', None, False)
    assert not decision.needs_user

def test_decide_asks_when_a_deciding_count_is_in_the_band():
    ahead_in_band = [_summary('national', 0.01, 500, 500), _summary('state', 0.3, 800, 700)]
    decision = esc.decide(ahead_in_band, WINDOW, WINDOW)
    assert (decision.branch, decision.grain, decision.needs_user) == ('A', 'state', True)
    time_in_band = [_summary('national', 0.01, 990, 450)]
    decision = esc.decide(time_in_band, WINDOW, WINDOW)
    assert (decision.branch, decision.needs_user) == ('B', True)
    assert esc.decide([_summary('national', 0.2, 450, 0)], WINDOW, WINDOW).needs_user

def test_render_decision_uses_fixed_wording():
    summaries = [_summary('national', 0.01, 990, 980)]
    text = esc.render_decision(esc.decide(summaries, WINDOW, WINDOW), summaries, WINDOW)
    assert text.startswith('<!-- decision:begin -->\n- **Branch:** A. The verified window')
    assert '- **Row grain:** a six-digit code in a reference year (national, private' in text
    assert text.rstrip().endswith('<!-- decision:end -->')

# -------------------------------------------------------------------------------------------------
# Provenance and run
# -------------------------------------------------------------------------------------------------

def test_parse_headers_keeps_the_last_response():
    text = (
        'HTTP/2 301\r\nlocation: https://x\r\n\r\n'
        'HTTP/2 200\r\ncontent-length: 5\r\nlast-modified: Tue, 02 Sep 2025 11:20:46 GMT\r\n\r\n'
    )
    headers = esc.parse_headers(text)
    assert headers['status'] == '200'
    assert headers['content-length'] == '5'
    assert headers['last-modified'] == 'Tue, 02 Sep 2025 11:20:46 GMT'

def test_build_manifest_records_provenance(tmp_path):
    (tmp_path / 'headers').mkdir()
    (tmp_path / 'a.csv').write_bytes(b'hello')
    (tmp_path / 'headers' / 'a.csv.headers').write_text(
        'HTTP/2 200\r\ncontent-length: 5\r\nlast-modified: Tue, 02 Sep 2025 11:20:46 GMT\r\n\r\n'
    )
    [entry] = esc.build_manifest(tmp_path, {'a.csv': 'https://example.test/a.csv'})
    assert entry['bytes'] == 5
    assert entry['sha256'] == hashlib.sha256(b'hello').hexdigest()
    assert entry['last_modified'] == 'Tue, 02 Sep 2025 11:20:46 GMT'
    (tmp_path / 'a.csv').write_bytes(b'hello!')
    with pytest.raises(ValueError, match='Content-Length'):
        esc.build_manifest(tmp_path, {'a.csv': 'https://example.test/a.csv'})

def _vintage_2017_rows():
    rows = [
        ('US000', '5', '10', '11', '', 100, 1000, 100000),
        ('US000', '5', '111110', '18', '', 10, 100, 10000),
        ('US000', '5', '454110', '18', '', 7, 70, 7000),
        ('US000', '5', '238111', '18', '', 4, 40, 4000),
        ('US000', '5', '238112', '18', 'N', 1, 0, 0),
    ]
    return [_record(2021, *row) for row in rows]

def _write_qcew_dir(directory):
    directory.mkdir()
    for year in WINDOW:
        rows = _annual_rows(year)
        with zipfile.ZipFile(directory / f'{year}_annual_singlefile.zip', 'w') as archive:
            archive.writestr(f'{year}.annual.singlefile.csv', _csv_bytes(rows))
        national = [row for row in rows if row['area_fips'] == 'US000']
        (directory / f'{year}_US000_annual.csv').write_bytes(_csv_bytes(national))
    (directory / '2021_US000_annual.csv').write_bytes(_csv_bytes(_vintage_2017_rows()))
    return directory

def test_run_writes_tables_and_decision(tmp_path):
    qcew_dir = _write_qcew_dir(tmp_path / 'qcew')
    codebook = _write_codebook(tmp_path)
    out_dir = tmp_path / 'out'
    decision, failures = esc.run(
        qcew_dir,
        codebook,
        WINDOW,
        out_dir,
        codebook_sha256=_digest(codebook),
        floor=2,
        band=(0, 0)
    )
    assert failures == []
    assert (decision.branch, decision.grain) == ('A', 'state')
    report = json.loads((out_dir / 'coverage.json').read_text())
    assert report['split_codes'] == ['238110', '238120']
    assert {row['estabs_column'] for row in report['conventions']} == {'annual_avg_estabs'}
    assert report['msa_rows'][-1] == {'year': 2025, 'rows': 0}
    vintage = {row['year']: row['outside_codebook'] for row in report['vintage']}
    assert vintage == {2021: 1, 2022: 0, 2023: 0, 2024: 0, 2025: 0}
    assert '<!-- decision:begin -->' in (out_dir / 'decision.md').read_text()
    assert '### Decision inputs' in (out_dir / 'tables.md').read_text()

def test_main_exit_code_signals_stop_and_ask(tmp_path, monkeypatch):
    argv = [
        'run', '--qcew-dir',
        str(tmp_path), '--codebook',
        str(tmp_path / 'codebook.parquet'), '--final-years', '2022', '--out-dir',
        str(tmp_path / 'out')
    ]
    clean = esc.Decision('A', 'national', True, True, False, ())
    asking = esc.Decision('A', 'national', True, True, True, ())
    monkeypatch.setattr(esc, 'run', lambda *args: (clean, []))
    assert esc.main(argv) == 0
    monkeypatch.setattr(esc, 'run', lambda *args: (clean, ['2024: an invariant failed']))
    assert esc.main(argv) == 2
    monkeypatch.setattr(esc, 'run', lambda *args: (asking, []))
    assert esc.main(argv) == 2
