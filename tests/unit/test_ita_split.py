'''
The ITA panel's settings, its salted split of ``description_key`` and the fixture tables (plan 17,
Task 1; spec text-to-code-ita-panel, 4.2).

The split is exercised on golden keys whose parts are pinned, on its encoding, on the pinned rule
fingerprint and on the committed settings file. The fixture module's tables are checked against
Stage 10's schemas, copied verbatim, and against the brief's table of base pairs.
'''

import hashlib
import json
import math
from pathlib import Path

import polars as pl
import pytest
import torch
from pydantic import ValidationError

from naics_embedder.panels.ita_split import (
    ITA_PANEL,
    ItaPanelSettings,
    ItaPart,
    load_ita_settings,
    split_fingerprint,
    split_part,
    split_rule,
)
from tests.fixtures.ita import (
    ASSIGNED,
    ITA_CANDIDATES,
    ITA_PAIR_CODES_SCHEMA,
    ITA_PAIRS_SCHEMA,
    ITA_SETTINGS,
    AssignedStubEncoder,
    ita_tables,
    write_ita_tables,
)

ROOT = Path(__file__).resolve().parents[2]  # the repository, whatever the working directory
SALT = ITA_SETTINGS.salt

GOLDEN = {
    'we make dog food': 'validation',
    'soybean farm': 'sealed',
    'oil and gas wells': 'validation',
    'canola growers': 'sealed',
    'sawmill': 'sealed',
    'dental office': 'sealed',
    'general contractor': 'validation',
    'pet food maker': 'validation',
}

# Pinned from a hand computation with hashlib, so each is also a key whose encoding matters: UTF-8
# bytes, not UTF-16, UTF-32 or Latin-1 ones
ENCODING_KEYS = {'über cleaners': 'sealed', 'señor taco': 'sealed', '寿司店': 'validation'}

# Stage 10's ``ita_text_pairs.parquet`` and ``ita_text_pair_codes.parquet`` (embedding-data spec at
# ec6fb29), every column in its order
STAGE_10_PAIRS = [
    ('pair_id', pl.Utf8),
    ('pair_kind', pl.Utf8),
    ('employer_key', pl.Utf8),
    ('employer_basis', pl.Utf8),
    ('description_key', pl.Utf8),
    ('original_strings', pl.List(pl.Utf8)),
    ('representative_string', pl.Utf8),
    ('n_records', pl.UInt64),
    ('n_headline_records', pl.UInt64),
    ('first_year', pl.Int64),
    ('last_year', pl.Int64),
    ('n_years', pl.UInt64),
    ('flag_headline', pl.Boolean),
    ('flag_headline_codes_conflict', pl.Boolean),
    ('flag_empty', pl.Boolean),
    ('flag_placeholder', pl.Boolean),
    ('flag_copied_title_any', pl.Boolean),
    ('flag_copied_index_any', pl.Boolean),
    ('flag_suspected_mojibake', pl.Boolean),
    ('flag_employer_unknown', pl.Boolean),
    ('flag_employer_company_fallback', pl.Boolean),
    ('n_records_ein_padded', pl.UInt64),
]
STAGE_10_CODES = [
    ('pair_id', pl.Utf8),
    ('naics_code', pl.Utf8),
    ('naics_year', pl.Int64),
    ('naics_mapping_status', pl.Utf8),
    ('naics_2022_candidates', pl.List(pl.Utf8)),
    ('naics_2022_candidate_node_ids', pl.List(pl.Utf8)),
    ('naics_2022_node_id', pl.Utf8),
    ('naics_deepest_common_node_id', pl.Utf8),
    ('flag_naics_crosswalked', pl.Boolean),
    ('flag_naics_ambiguous', pl.Boolean),
    ('flag_naics_unresolved', pl.Boolean),
    ('flag_headline_basis', pl.Boolean),
    ('flag_copied_title_native', pl.Boolean),
    ('flag_copied_index_native', pl.Boolean),
    ('n_records', pl.UInt64),
    ('n_records_year_from_dictionary', pl.UInt64),
    ('years', pl.List(pl.Int64)),
]

# The brief's base fixture: each pair's part, then its reported codes (flag_headline_basis rows)
BASE_PARTS = {
    'p01': 'validation',
    'p02': 'validation',
    'p03': 'validation',
    'p04': 'validation',
    'p05': 'validation',
    'p06': 'validation',
    'p07': 'validation',
    'p08': 'sealed',
    'p09': 'sealed',
    'p10': 'sealed',
    'p11': 'sealed',
}
BASE_CODES = {
    'p01': ['311111'],
    'p02': ['311111'],
    'p03': ['211120'],
    'p04': ['111110', '311111'],
    'p05': ['513210'],
    'p06': ['111110'],
    'p07': ['236220'],
    'p08': ['111110'],
    'p09': ['111120'],
    'p10': ['321113'],
    'p11': ['621210'],
}
# The flags that are true on each base pair: every other flag is false
BASE_FLAGS = {
    'p01': {'flag_headline'},
    'p02': {'flag_headline'},
    'p03': {'flag_headline'},
    'p04': {'flag_headline', 'flag_headline_codes_conflict'},
    'p05': {'flag_headline', 'flag_copied_title_any'},
    'p06': set(),
    'p07': {'flag_headline', 'flag_employer_company_fallback'},
    'p08': {'flag_headline'},
    'p09': {'flag_headline'},
    'p10': {'flag_headline', 'flag_suspected_mojibake'},
    'p11': {'flag_headline', 'flag_copied_index_any'},
}

# -------------------------------------------------------------------------------------------------
# The split
# -------------------------------------------------------------------------------------------------

def test_split_part_matches_the_golden_parts():
    for key, part in GOLDEN.items():
        assert split_part(key, ITA_SETTINGS.salt) is ItaPart(part)

def test_split_fingerprint_is_pinned():
    assert split_fingerprint(ITA_SETTINGS) == (
        '8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c'
    )

def test_committed_settings_hold_the_planned_values():
    # ROOT = Path(__file__).resolve().parents[2], as test_economic_candidates.py resolves it
    settings = load_ita_settings(ROOT / 'conf/data/ita_panel.yaml')
    assert settings == ITA_SETTINGS

def test_each_description_key_falls_in_one_part():
    pairs, _ = ita_tables()
    parts = pairs.with_columns(
        part=pl.col('description_key').map_elements(
            lambda key: split_part(key, ITA_SETTINGS.salt).value, return_dtype=pl.Utf8
        )
    )
    assert parts.group_by('description_key').agg(pl.col('part').n_unique()
                                                 ).get_column('part').max() == 1
    assert set(parts.get_column('part')) == {'validation', 'sealed'}

def test_a_rebuilt_table_keeps_every_old_key_in_its_old_part():
    before, _ = ita_tables()
    added = {
        'pair_id': 'p12',
        'description_key': 'pet food maker',
        'representative_string': 'Pet food maker'
    }
    after, _ = ita_tables(extra_pairs=[added])
    old = {key: split_part(key, SALT) for key in before.get_column('description_key')}
    assert after.height == before.height + 1
    for key, part in old.items():
        assert after.filter(pl.col('description_key') == key).height >= 1
        assert split_part(key, SALT) is part
    # the new key lands in a part of its own, with no table consulted
    assert 'pet food maker' not in old
    assert split_part('pet food maker', SALT) is ItaPart.VALIDATION
    assert split_fingerprint(ITA_SETTINGS) == (
        '8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c'
    )

def test_settings_refuse_a_bad_salt_or_margin():
    good = dict(panel='ita_text', rule_version=1, salt=SALT, relative_margin=0.05)
    assert ItaPanelSettings(**good) == ITA_SETTINGS
    for bad in (
        {
            'salt': SALT[:31]
        },
        {
            'salt': SALT.upper()
        },
        {
            'relative_margin': 0
        },
        {
            'relative_margin': math.inf
        },
        {
            'unknown_key': 1
        },
    ):
        with pytest.raises(ValidationError):
            ItaPanelSettings(**{**good, **bad})

def test_fixture_tables_carry_every_stage_10_column():
    pairs, codes = ita_tables()
    assert dict(pairs.schema) == ITA_PAIRS_SCHEMA and dict(codes.schema) == ITA_PAIR_CODES_SCHEMA

# Added by the mutant step: input shapes the plan's tests above do not use

@pytest.mark.parametrize('key, part', sorted(ENCODING_KEYS.items()))
def test_split_part_hashes_a_keys_utf8_bytes(key, part):
    assert split_part(key, SALT) is ItaPart(part)

def test_split_part_reads_an_empty_key():
    assert split_part('', SALT) is ItaPart.VALIDATION

def test_the_salt_is_separated_from_the_key_by_a_null_byte():
    # 'aa' + 'b' + key and 'aab' + key would hash the same bytes with no separator
    keys = [f'key {number}' for number in range(40)]
    joined = [split_part('b' + key, 'aa') for key in keys]
    separate = [split_part(key, 'aab') for key in keys]
    assert joined != separate

def test_another_salt_moves_some_keys_to_the_other_part():
    other = 'f' * 32
    assert {split_part(key, other) for key in GOLDEN} == set(ItaPart)
    assert [split_part(key, other) for key in GOLDEN] != [split_part(key, SALT) for key in GOLDEN]

def test_split_part_returns_the_two_part_names_for_zero_and_one():
    # 0 is validation, 1 is sealed (spec 4.2); the golden keys cover both
    assert ItaPart.VALIDATION.value == 'validation' and ItaPart.SEALED.value == 'sealed'
    assert [part.value for part in ItaPart] == ['validation', 'sealed']
    assert issubclass(ItaPart, str) and ITA_PANEL == 'ita_text'

def test_split_rule_is_exactly_the_planned_rule():
    assert split_rule(ITA_SETTINGS) == {
        'salt': SALT,
        'rule_version': 1,
        'hash': 'sha256',
        'prefix_bytes': 8,
        'byteorder': 'big',
        'modulus': 2,
        'parts': ['validation', 'sealed'],
    }

def test_split_fingerprint_is_the_sha256_of_the_rules_canonical_json():
    canonical = json.dumps(split_rule(ITA_SETTINGS), sort_keys=True, separators=(',', ':'))
    assert split_fingerprint(ITA_SETTINGS) == hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def test_split_fingerprint_names_the_split_not_the_margin():
    wider = ITA_SETTINGS.model_copy(update={'relative_margin': 0.1})
    resalted = ITA_SETTINGS.model_copy(update={'salt': 'f' * 32})
    assert split_fingerprint(wider) == split_fingerprint(ITA_SETTINGS)
    assert split_fingerprint(resalted) != split_fingerprint(ITA_SETTINGS)

@pytest.mark.parametrize(
    'bad', [
        {
            'salt': SALT + '0'
        },
        {
            'salt': SALT + '\n'
        },
        {
            'salt': SALT[:-1] + 'g'
        },
        {
            'salt': ''
        },
        {
            'relative_margin': -0.05
        },
        {
            'relative_margin': -math.inf
        },
        {
            'relative_margin': math.nan
        },
        {
            'panel': 'outcome'
        },
        {
            'rule_version': 2
        },
    ]
)
def test_settings_refuse_other_bad_values(bad):
    good = dict(panel='ita_text', rule_version=1, salt=SALT, relative_margin=0.05)
    with pytest.raises(ValidationError):
        ItaPanelSettings(**{**good, **bad})

@pytest.mark.parametrize('missing', ['panel', 'rule_version', 'salt', 'relative_margin'])
def test_settings_refuse_a_missing_key(missing):
    good = dict(panel='ita_text', rule_version=1, salt=SALT, relative_margin=0.05)
    del good[missing]
    with pytest.raises(ValidationError):
        ItaPanelSettings(**good)

def test_settings_are_frozen():
    with pytest.raises(ValidationError):
        ITA_SETTINGS.salt = 'f' * 32

def test_load_ita_settings_reads_a_path_or_a_string(tmp_path):
    path = tmp_path / 'ita_panel.yaml'
    path.write_text(f"panel: ita_text\nrule_version: 1\nsalt: '{SALT}'\nrelative_margin: 0.05\n")
    assert load_ita_settings(path) == ITA_SETTINGS
    assert load_ita_settings(str(path)) == ITA_SETTINGS

@pytest.mark.parametrize(
    'text', [
        f"panel: ita_text\nrule_version: 1\nsalt: '{SALT}'\nrelative_margin: 0.05\nextra: 1\n",
        f"panel: ita_text\nrule_version: 1\nsalt: '{SALT}'\n",
        f"panel: ita_text\nrule_version: 1\nsalt: '{SALT[:-1]}'\nrelative_margin: 0.05\n",
        '',
        '- not a mapping\n',
    ]
)
def test_load_ita_settings_refuses_a_file_that_breaks_the_rules(tmp_path, text):
    path = tmp_path / 'ita_panel.yaml'
    path.write_text(text)
    with pytest.raises(ValidationError):
        load_ita_settings(path)

def test_the_committed_file_names_no_machine_path():
    assert '/Users/' not in (ROOT / 'conf/data/ita_panel.yaml').read_text()

# -------------------------------------------------------------------------------------------------
# The fixture tables
# -------------------------------------------------------------------------------------------------

def test_fixture_schemas_are_stage_10s_in_its_order_and_types():
    assert list(ITA_PAIRS_SCHEMA.items()) == STAGE_10_PAIRS
    assert list(ITA_PAIR_CODES_SCHEMA.items()) == STAGE_10_CODES
    pairs, codes = ita_tables()
    assert list(pairs.schema.items()) == STAGE_10_PAIRS
    assert list(codes.schema.items()) == STAGE_10_CODES

def test_the_fixture_pairs_fall_in_the_planned_parts():
    pairs, _ = ita_tables()
    assert pairs.get_column('pair_id').to_list() == list(BASE_PARTS)
    for row in pairs.iter_rows(named=True):
        assert split_part(row['description_key'], SALT).value == BASE_PARTS[row['pair_id']]

def test_several_employers_share_one_description():
    pairs, _ = ita_tables()
    dog = pairs.filter(pl.col('description_key') == 'we make dog food')
    assert dog.get_column('pair_id').to_list() == ['p01', 'p02']
    assert dog.get_column('employer_key').n_unique() == 2
    assert dog.get_column('representative_string').to_list() == [
        'We make dog food', 'we make dog food'
    ]

def test_the_fixture_flags_are_the_planned_ones():
    pairs, _ = ita_tables()
    flags = [name for name in ITA_PAIRS_SCHEMA if name.startswith('flag_')]
    for row in pairs.iter_rows(named=True):
        assert {name for name in flags if row[name]} == BASE_FLAGS[row['pair_id']], row['pair_id']

def test_a_company_fallback_pair_has_a_company_basis():
    pairs, _ = ita_tables()
    bases = dict(pairs.select('pair_id', 'employer_basis').iter_rows())
    assert bases['p07'] == 'company'
    assert {basis for pair_id, basis in bases.items() if pair_id != 'p07'} == {'ein'}

def test_the_fixture_reported_codes_are_the_planned_ones():
    _, codes = ita_tables()
    basis = codes.filter(pl.col('flag_headline_basis'))
    reported = basis.group_by('pair_id').agg(pl.col('naics_code').unique().sort())
    assert dict(reported.iter_rows()) == BASE_CODES

def test_every_basis_row_carries_its_national_industry_node():
    _, codes = ita_tables()
    for row in codes.iter_rows(named=True):
        if row['flag_headline_basis']:
            assert row['naics_2022_node_id'] == f'naics:2022:national_industry:{row["naics_code"]}'
            assert row['naics_2022_candidates'] == [row['naics_code']]
            assert row['naics_2022_candidate_node_ids'] == [row['naics_2022_node_id']]

def test_p01_has_two_rows_for_one_node_and_an_unasserted_2017_crosswalk():
    _, codes = ita_tables()
    rows = codes.filter(pl.col('pair_id') == 'p01').to_dicts()
    basis = [row for row in rows if row['flag_headline_basis']]
    assert len(rows) == 3 and len(basis) == 2
    assert {row['naics_2022_node_id'] for row in basis} == {'naics:2022:national_industry:311111'}
    assert sorted(row['years'] for row in basis) == [[2023], [2024]]
    assert len({row['naics_year'] for row in basis}) == 2
    (crosswalk, ) = [row for row in rows if not row['flag_headline_basis']]
    assert crosswalk['naics_year'] == 2017 and crosswalk['flag_naics_crosswalked'] is True
    assert crosswalk['naics_2022_node_id'] is None
    assert crosswalk['naics_code'] not in {row['naics_code'] for row in basis}

def test_a_native_copied_flag_is_null_exactly_when_the_year_is():
    _, codes = ita_tables()
    for row in codes.iter_rows(named=True):
        for name in ('flag_copied_title_native', 'flag_copied_index_native'):
            assert (row[name] is None) == (row['naics_year'] is None)

def test_the_codes_table_is_sorted_by_pair_code_and_year_with_nulls_first():
    _, codes = ita_tables()
    keys = [
        (row['pair_id'], row['naics_code'], row['naics_year'])
        for row in codes.iter_rows(named=True)
    ]
    ordered = sorted(
        keys, key=lambda key: (key[0], key[1], -math.inf if key[2] is None else key[2])
    )
    assert keys == ordered
    assert [key[2] for key in keys if key[0] == 'p01' and key[1] == '311111'] == [None, 2022]

def test_extra_pairs_and_codes_are_filled_with_neutral_values():
    pairs, codes = ita_tables(
        extra_pairs=[{
            'pair_id': 'p00',
            'description_key': 'x',
            'representative_string': 'X'
        }],
        extra_codes=[{
            'pair_id': 'p00',
            'naics_code': '111110'
        }],
    )
    pair = pairs.row(0, named=True)
    assert pair['pair_id'] == 'p00'
    assert pair['n_records'] == 0 and pair['first_year'] is None and pair['n_years'] == 0
    assert pair['original_strings'] == ['X'] and pair['pair_kind'] == 'employer'
    assert not any(pair[name] for name in ITA_PAIRS_SCHEMA if name.startswith('flag_'))
    code = codes.row(0, named=True)
    assert code['pair_id'] == 'p00' and code['naics_code'] == '111110'
    assert code['n_records'] == 0 and code['years'] == [] and code['naics_2022_candidates'] == []
    assert not any(code[name] for name in ITA_PAIR_CODES_SCHEMA if name.startswith('flag_'))
    assert dict(pairs.schema) == ITA_PAIRS_SCHEMA and dict(codes.schema) == ITA_PAIR_CODES_SCHEMA

def test_an_extra_row_may_repeat_a_pair_id():
    pairs, _ = ita_tables(
        extra_pairs=[{
            'pair_id': 'p01',
            'description_key': 'x',
            'representative_string': 'X'
        }]
    )
    assert pairs.get_column('pair_id').to_list().count('p01') == 2

def test_replace_codes_swaps_one_pairs_rows_and_leaves_the_rest():
    _, base = ita_tables()
    node = 'naics:2022:national_industry:999999'
    _, codes = ita_tables(
        replace_codes={
            'p03': [
                {
                    'naics_code': '999999',
                    'naics_2022_node_id': node,
                    'flag_headline_basis': True
                }
            ]
        }
    )
    assert codes.filter(pl.col('pair_id') == 'p03').get_column('naics_2022_node_id').to_list() == [
        node
    ]
    assert codes.filter(pl.col('pair_id') != 'p03').equals(base.filter(pl.col('pair_id') != 'p03'))
    assert dict(codes.schema) == ITA_PAIR_CODES_SCHEMA

def test_replace_codes_names_the_pair_that_does_not_exist():
    with pytest.raises(ValueError, match='p99'):
        ita_tables(replace_codes={'p99': []})

def test_a_column_stage_10_does_not_have_is_refused():
    with pytest.raises(ValueError, match='not_a_column'):
        ita_tables(
            extra_pairs=[
                {
                    'pair_id': 'p12',
                    'description_key': 'x',
                    'representative_string': 'X',
                    'not_a_column': 1
                }
            ]
        )
    with pytest.raises(ValueError, match='not_a_column'):
        ita_tables(extra_codes=[{'pair_id': 'p01', 'naics_code': '1', 'not_a_column': 1}])

def test_write_ita_tables_writes_both_parquet_files(tmp_path):
    pairs, codes = ita_tables()
    pairs_path, codes_path = write_ita_tables(tmp_path, pairs, codes)
    assert (pairs_path.name, codes_path.name) == (
        'ita_text_pairs.parquet', 'ita_text_pair_codes.parquet'
    )
    assert pairs_path.parent == tmp_path and codes_path.parent == tmp_path
    assert pl.read_parquet(pairs_path).equals(pairs)
    assert pl.read_parquet(codes_path).equals(codes)

# -------------------------------------------------------------------------------------------------
# The assigned stub encoder
# -------------------------------------------------------------------------------------------------

def test_the_candidates_are_the_planned_eight_codes():
    assert ITA_CANDIDATES == (
        '111110', '111120', '211120', '236220', '311111', '321113', '513210', '621210'
    )

def test_the_stub_embeds_each_code_on_its_own_axis():
    codes = AssignedStubEncoder().encode_codes(['311111', '111110', '621210'])
    assert codes.shape == (3, 8) and codes.dtype == torch.float32
    assert codes.argmax(dim=1).tolist() == [4, 0, 7]
    assert codes.sum(dim=1).tolist() == [1.0, 1.0, 1.0]
    assert AssignedStubEncoder().encode_codes(list(ITA_CANDIDATES)).equal(torch.eye(8))

def test_the_stub_embeds_each_query_on_its_assigned_codes_axis():
    stub = AssignedStubEncoder()
    queries = ['Oil and gas wells', 'We make dog food', 'Sawmill']
    points = stub.encode_queries(queries)
    assert points.shape == (3, 8) and points.dtype == torch.float32
    assert points.argmax(dim=1).tolist() == [0, 4, 5]
    assert points.sum(dim=1).tolist() == [1.0, 1.0, 1.0]
    assert stub.encode_queries([]).shape == (0, 8)

def test_the_stub_has_no_distance_but_takes_one():
    stub = AssignedStubEncoder()
    assert not hasattr(stub, 'distance') and not hasattr(AssignedStubEncoder, 'distance')
    stub.distance = 'euclidean'
    assert stub.distance == 'euclidean'

def test_assigned_covers_the_headline_pairs_and_not_the_other():
    pairs, _ = ita_tables()
    rows = pairs.iter_rows(named=True)
    queries = {row['representative_string'] for row in rows if row['flag_headline']}
    assert set(ASSIGNED) == queries
    assert 'Corn farming' not in ASSIGNED
    assert set(ASSIGNED.values()) <= set(ITA_CANDIDATES)
    # p03's query is assigned a code other than its reported one
    assert ASSIGNED['Oil and gas wells'] == '111110' and BASE_CODES['p03'] == ['211120']
    assert ASSIGNED['Cat food plant'] in BASE_CODES['p04']
    reported = {
        row['representative_string']: BASE_CODES[row['pair_id']]
        for row in pairs.iter_rows(named=True) if row['flag_headline']
    }
    others = {query for query, code in ASSIGNED.items() if code not in reported[query]}
    assert others == {'Oil and gas wells'}

def test_a_missed_query_ties_every_other_candidate_with_its_reported_code():
    stub = AssignedStubEncoder()
    codes = stub.encode_codes(ITA_CANDIDATES)
    queries = stub.encode_queries(['Oil and gas wells', 'Cat food plant'])
    cosine = queries @ codes.T  # unit rows: the larger, the nearer
    assert cosine.max(dim=1).values.tolist() == [1.0, 1.0]
    assert (cosine == 0).sum(dim=1).tolist() == [7, 7]  # every other candidate ties at distance 1
