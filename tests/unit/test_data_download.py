import hashlib
import json
from io import BytesIO
from typing import Dict, List, Set, cast

import polars as pl
import pytest

from naics_embedder.data import download_data
from naics_embedder.utils.config import DownloadConfig
from tests.fixtures.naics_sources import TITLES

ENTRY_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'text': pl.Utf8}

@pytest.mark.unit
def test_get_titles_normalizes_index_and_level():
    titles_df = pl.DataFrame(
        {
            'index': [1, 2],
            'code': ['31', '311'],
            'title': ['Manufacturing', 'Food Manufacturing'],
        }
    )

    titles, codes = download_data._get_titles(titles_df)

    assert titles.shape == (2, 4)
    assert titles.get_column('index').to_list() == [0, 1]
    assert titles.get_column('level').to_list() == [2, 3]
    assert codes == {'31', '311'}

@pytest.mark.unit
def test_get_descriptions_filters_cross_references():
    descriptions_df = pl.DataFrame(
        {
            'code': ['31111'],
            'description': [
                'Primary line\r\nCross-References.\r\nThe Sector as a Whole\r\nFinal line',
            ],
        }
    )

    _, descriptions_clean = download_data._get_descriptions_1(descriptions_df)

    cleaned = descriptions_clean.get_column('description').to_list()
    assert cleaned == ['Primary line', 'Final line']
    ids = descriptions_clean.get_column('description_id').to_list()
    assert ids == sorted(ids)

@pytest.mark.unit
@pytest.mark.parametrize('eol', [pytest.param('\r\n', id='crlf'), pytest.param('\n', id='lf')])
def test_get_descriptions_splits_lines_for_any_line_ending(eol: str):
    # The xlsx stores CRLF line endings; whether they reach the pipeline as CRLF or LF depends on
    # the Excel reader and its version
    description = eol.join(
        [
            'The Sector as a Whole',
            '',
            'This industry comprises establishments growing grain.',
            '',
            'Illustrative Examples:',
            '',
            'Barley farming',
            'Rye farming',
            '',
            '',
            'Cross-References. Establishments growing wheat are classified in Industry 111140.',
        ]
    )
    descriptions_df = pl.DataFrame({'code': ['111199'], 'description': [description]})

    _, descriptions_clean = download_data._get_descriptions_1(descriptions_df)

    assert descriptions_clean.get_column('description').to_list() == [
        'This industry comprises establishments growing grain.',
        'Illustrative Examples:',
        'Barley farming',
        'Rye farming',
    ]

@pytest.mark.unit
def test_read_xlsx_bytes_renames_columns(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_read_excel(buffer, sheet_name, columns, schema_overrides):
        captured['sheet'] = sheet_name
        assert isinstance(buffer, BytesIO)
        assert columns == ['A']
        assert schema_overrides == {'A': pl.Utf8}
        return pl.DataFrame({'A': ['value']})

    monkeypatch.setattr(download_data.pl, 'read_excel', fake_read_excel)

    schema = cast(Dict[str, pl.DataType], {'A': pl.Utf8})
    result = download_data._read_xlsx_bytes(
        data=b'noop',
        sheet='Sheet1',
        schema=schema,
        cols={'A': 'code'},
    )

    assert captured['sheet'] == 'Sheet1'
    assert result.columns == ['code']
    assert result.get_column('code').to_list() == ['value']

@pytest.mark.unit
def test_get_examples_joins_examples_role_entries_in_index_order():
    examples_entries = pl.DataFrame(
        [(5, '111', 'Second entry'), (2, '111', 'First entry')], schema=ENTRY_SCHEMA, orient='row'
    )
    descriptions_2 = pl.DataFrame(
        {
            'code': ['111', '111', '111'],
            'description_id': [1, 2, 3],
            'description': ['Intro', 'Illustrative Examples:', 'Example bullet'],
        }
    )
    descriptions_3 = descriptions_2.clone()

    examples, descriptions_examples = download_data._get_examples(
        {'111'}, examples_entries, descriptions_2, descriptions_3
    )

    assert examples.rows() == [('111', 'First entry; Second entry')]
    # The examples section, and so the description cutoff, starts at the marker itself
    assert descriptions_examples.height == 1
    assert descriptions_examples.row(0, named=True)['description_id_min'] == 2

@pytest.mark.unit
def test_get_examples_rejects_entries_of_codes_without_index_entries():
    examples_entries = pl.DataFrame([(0, '222', 'Stray entry')], schema=ENTRY_SCHEMA, orient='row')
    descriptions = pl.DataFrame({'code': ['111'], 'description_id': [1], 'description': ['Intro']})

    with pytest.raises(ValueError, match='without index entries'):
        download_data._get_examples({'111'}, examples_entries, descriptions, descriptions)

@pytest.mark.unit
def test_get_exclusions_combines_crossrefs_and_descriptions():
    exclusions_df = pl.DataFrame({
        'code': ['111'],
        'excluded': ['See 222 and 333'],
    })
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111', '111'],
            'description_id': pl.Series('description_id', [1, 2], dtype=pl.UInt32),
            'description': ['Some text', 'Excluded 333'],
        }
    )
    codes = {'111', '222', '333'}

    exclusions, descriptions_exclusions = download_data._get_exclusions(
        exclusions_df, descriptions_3, codes
    )

    assert descriptions_exclusions.height == 1
    assert descriptions_exclusions.row(0, named=True)['description_id'] == 2

    assert exclusions.height == 1
    row = exclusions.row(0, named=True)
    assert set(row['excluded_codes']) == {'222', '333'}

@pytest.mark.unit
def test_get_descriptions_2_removes_flagged_sections():
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111', '111', '111'],
            'description_id': [1, 2, 3],
            'description': ['Keep me', 'Drop exclusion', 'Drop example'],
        }
    )
    descriptions_exclusions = pl.DataFrame({'code': ['111'], 'description_id': [2]})
    descriptions_examples = pl.DataFrame({'code': ['111'], 'description_id_min': [3]})

    cleaned = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )

    assert cleaned.height == 1
    text = cleaned.row(0, named=True)['description']
    assert 'Drop exclusion' not in text
    assert 'Drop example' not in text

@pytest.mark.unit
@pytest.mark.parametrize(
    'index_codes, entry_texts, expected_examples',
    [
        pytest.param({'111199'}, ['Grain farming, mixed'], ['Grain farming, mixed'], id='index'),
        pytest.param(set(), [], ['Barley farming; Rye farming'], id='description-text'),
        # Every entry of the code is a query: its examples channel stays empty, never the bullets
        pytest.param({'111199'}, [], [], id='index-without-examples-role'),
    ],
)
def test_description_drops_whole_illustrative_examples_section(
    index_codes: Set[str], entry_texts: List[str], expected_examples: List[str]
):
    # A real description split one block per line, the structure _get_examples relies on. The
    # marker and its bullets leave the description whichever source fills the examples column:
    # the examples-role index entries of a code with entries, or else the bullets themselves.
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111199'] * 4,
            'description_id': pl.Series([1, 2, 3, 4], dtype=pl.UInt32),
            'description': [
                'This industry comprises establishments growing grain.',
                'Illustrative Examples:',
                'Barley farming',
                'Rye farming',
            ],
        }
    )
    examples_entries = pl.DataFrame(
        [(entry_id, '111199', text) for entry_id, text in enumerate(entry_texts)],
        schema=ENTRY_SCHEMA,
        orient='row',
    )
    descriptions_exclusions = pl.DataFrame(schema={'code': pl.Utf8, 'description_id': pl.UInt32})

    examples, descriptions_examples = download_data._get_examples(
        index_codes, examples_entries, descriptions_3, descriptions_3
    )
    descriptions = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )

    assert descriptions.get_column('description').to_list() == [
        'This industry comprises establishments growing grain.'
    ]
    assert examples.get_column('examples').to_list() == expected_examples

# -------------------------------------------------------------------------------------------------
# Local sources and the pinned index file
# -------------------------------------------------------------------------------------------------

CODES_URL = 'https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx'

@pytest.fixture
def captured_bytes(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_read_xlsx_bytes(data, sheet, schema, cols):
        captured['data'] = data
        return pl.DataFrame({'code': ['111110']})

    def no_download(*_args, **_kwargs):
        raise AssertionError('a local source must not be downloaded')

    monkeypatch.setattr(download_data, '_read_xlsx_bytes', fake_read_xlsx_bytes)
    monkeypatch.setattr(download_data, '_download_with_retry', no_download)
    return captured

@pytest.mark.unit
def test_read_xlsx_reads_the_local_copy_named_by_the_url(tmp_path, captured_bytes):
    (tmp_path / '2-6 digit_2022_Codes.xlsx').write_bytes(b'workbook')

    download_data._read_xlsx(
        CODES_URL,
        'Sheet1',
        {},
        {},
        source_dir=str(tmp_path),
        expected_sha256=hashlib.sha256(b'workbook').hexdigest(),
    )

    assert captured_bytes['data'] == b'workbook'

@pytest.mark.unit
def test_read_xlsx_rejects_a_file_other_than_the_pinned_one(tmp_path, captured_bytes):
    (tmp_path / '2-6 digit_2022_Codes.xlsx').write_bytes(b'another workbook')

    with pytest.raises(ValueError, match='pinned'):
        download_data._read_xlsx(
            CODES_URL,
            'Sheet1',
            {},
            {},
            source_dir=str(tmp_path),
            expected_sha256=hashlib.sha256(b'workbook').hexdigest(),
        )
    assert captured_bytes == {}

@pytest.mark.unit
def test_read_xlsx_fails_when_the_local_copy_is_missing(tmp_path, captured_bytes):
    with pytest.raises(FileNotFoundError, match='no local copy'):
        download_data._read_xlsx(CODES_URL, 'Sheet1', {}, {}, source_dir=str(tmp_path))

# -------------------------------------------------------------------------------------------------
# Index entries and the examples channel
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_index_entries_are_sheet_rows_of_six_digit_codes(naics_sources):
    entries = download_data.naics_index_entries(naics_sources)

    # Sheet row 8 is a "see" row (code ******); entry 3 had surrounding whitespace
    assert entries.get_column('entry_id').to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    assert entries.row(3) == (3, '111110', 'Soybean seed production')
    assert entries.schema == pl.Schema(ENTRY_SCHEMA)

@pytest.mark.unit
def test_build_descriptions_keeps_queries_out_of_the_examples_channel(naics_sources):
    entries = download_data.naics_index_entries(naics_sources)
    examples_entries = entries.filter(pl.col('entry_id').is_in([2, 0, 9]))

    descriptions = download_data.build_descriptions(naics_sources, examples_entries)
    examples = dict(descriptions.select('code', 'examples').iter_rows())

    assert descriptions.get_column('code').to_list() == [code for code, _ in TITLES]
    assert examples['111110'] == 'Soybean farming, field and seed production; Soybeans, organic'
    assert examples['111120'] == 'Rapeseed farming'
    assert examples['11119'] == 'Barley farming; Rye farming'
    assert examples['111191'] is None
    assert 'Illustrative' not in descriptions.filter(pl.col('code') == '11119')['description'][0]

# -------------------------------------------------------------------------------------------------
# The descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def pinned_output(tmp_path):
    output = tmp_path / 'naics_descriptions.parquet'
    output.write_bytes(b'descriptions')
    manifest = tmp_path / 'bundle' / 'manifest.json'
    manifest.parent.mkdir()
    fingerprint = hashlib.sha256(b'descriptions').hexdigest()
    manifest.write_text(json.dumps({'description_fingerprint': fingerprint}))
    config = tmp_path / 'config.yaml'
    config.write_text(f'supervision:\n  manifest_path: {manifest}\n')
    graph = tmp_path / 'graph.yaml'
    graph.write_text(f'supervision_manifest_path: {manifest}\n')
    return output, config, graph

@pytest.mark.unit
@pytest.mark.parametrize('pinned_by', ['config', 'graph'])
def test_refuses_to_overwrite_the_pinned_descriptions(pinned_output, pinned_by):
    output, config, graph = pinned_output
    pins = {
        'config': ((config, ('supervision', 'manifest_path')), ),
        'graph': ((graph, ('supervision_manifest_path', )), ),
    }[pinned_by]

    with pytest.raises(FileExistsError, match='pins'):
        download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)
    download_data.refuse_pinned_overwrite(output, force=True, pinning_configs=pins)

@pytest.mark.unit
def test_other_descriptions_files_are_not_pinned(pinned_output, tmp_path):
    output, config, _ = pinned_output
    pins = ((config, ('supervision', 'manifest_path')), )
    output.write_bytes(b'rebuilt descriptions')

    download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)
    download_data.refuse_pinned_overwrite(
        tmp_path / 'absent.parquet', force=False, pinning_configs=pins
    )

@pytest.mark.unit
def test_unset_or_missing_manifests_pin_nothing(tmp_path, pinned_output):
    output, _, _ = pinned_output
    unset = tmp_path / 'unset.yaml'
    unset.write_text('supervision:\n  manifest_path: null\n')
    missing = tmp_path / 'missing.yaml'
    missing.write_text('supervision_manifest_path: nowhere/manifest.json\n')
    pins = (
        (unset, ('supervision', 'manifest_path')),
        (missing, ('supervision_manifest_path', )),
        (tmp_path / 'no_such_config.yaml', ('supervision', 'manifest_path')),
    )

    assert download_data.pinned_description_fingerprints(pins) == {}
    download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)

# -------------------------------------------------------------------------------------------------
# Preprocessing with the frozen role table
# -------------------------------------------------------------------------------------------------

ROLE_TABLE = [
    (0, '111110', 'examples'),
    (1, '111110', 'validation'),
    (2, '111110', 'training'),
    (3, '111110', 'test'),
    (4, '111120', 'examples'),
    (5, '111120', 'examples'),
    (6, '111120', 'validation'),
    (7, '111120', 'training'),
    (9, '111120', 'test'),
    (10, '111120', 'training'),
]

@pytest.fixture
def preprocess_cfg(tmp_path, monkeypatch, naics_sources):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download_data, 'load_naics_sources', lambda cfg: naics_sources)
    roles_csv = tmp_path / 'conf' / 'index_roles.csv'
    roles_csv.parent.mkdir()
    roles_csv.write_text(
        'entry_id,code,role\n' + ''.join(f'{i},{c},{r}\n' for i, c, r in ROLE_TABLE)
    )
    return DownloadConfig(
        output_parquet=str(tmp_path / 'data' / 'naics_descriptions.parquet'),
        index_roles_parquet=str(tmp_path / 'data' / 'naics_index_roles.parquet'),
        index_roles_csv=str(roles_csv),
    )

@pytest.mark.unit
def test_preprocess_builds_examples_from_the_role_table(preprocess_cfg):
    descriptions = download_data.download_preprocess_data(preprocess_cfg)

    written = pl.read_parquet(preprocess_cfg.output_parquet)
    roles = pl.read_parquet(preprocess_cfg.index_roles_parquet)
    examples = dict(written.select('code', 'examples').iter_rows())
    assert written.equals(descriptions)
    assert examples['111110'] == 'Soybean farming, field and seed production'
    assert examples['111120'] == 'Canola farming; Flaxseed farming'
    assert roles.columns == ['entry_id', 'code', 'text', 'role']
    assert roles.get_column('entry_id').to_list() == [i for i, _, _ in ROLE_TABLE]
    assert roles.row(3) == (3, '111110', 'Soybean seed production', 'test')

@pytest.mark.unit
def test_preprocess_refuses_a_held_out_query_that_matches_training_text(preprocess_cfg):
    # Entry 10 repeats 111120's title; the table makes it a test query
    leaky = [(i, c, 'test' if i == 10 else ('training' if i == 9 else r)) for i, c, r in ROLE_TABLE]
    with open(preprocess_cfg.index_roles_csv, 'w') as handle:
        handle.write('entry_id,code,role\n' + ''.join(f'{i},{c},{r}\n' for i, c, r in leaky))

    with pytest.raises(ValueError, match='held-out queries match training text'):
        download_data.download_preprocess_data(preprocess_cfg)

@pytest.mark.unit
def test_preprocess_needs_the_role_table(preprocess_cfg, tmp_path):
    missing = preprocess_cfg.model_copy(update={'index_roles_csv': str(tmp_path / 'none.csv')})

    with pytest.raises(FileNotFoundError, match='data roles'):
        download_data.download_preprocess_data(missing)
