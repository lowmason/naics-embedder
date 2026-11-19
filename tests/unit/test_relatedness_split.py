'''
The relatedness panel's settings, its salted split of the six-digit codes and the fixture tables
(plan 21, Task 1; spec relatedness-artifact, 4.3 and 6).

The split is exercised on golden codes whose parts are pinned, on the pinned rule fingerprint and
on the committed settings file. The fixture module's tables are checked against Stage 10's
schemas, copied from embedding-data's ``e6e12b2``, and against the brief's table of base codes.
'''

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from naics_embedder.panels import ita_split
from naics_embedder.panels.relatedness_split import (
    RELATEDNESS_PANEL,
    RelatednessPanelSettings,
    RelatednessPart,
    load_relatedness_settings,
    split_fingerprint,
    split_part,
    split_rule,
)
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix
from tests.fixtures.ita import ITA_SETTINGS
from tests.fixtures.relatedness import (
    BASE_MULTI,
    BASE_PAIRS,
    CO_OWNERSHIP_MARGINAL_SCHEMA,
    CO_OWNERSHIP_TARGET_SCHEMA,
    MARGINALS_FILE,
    RELATEDNESS_CODES,
    RELATEDNESS_SETTINGS,
    SECOND_FAMILY,
    TARGETS_FILE,
    co_ownership_tables,
    node,
    tree_distance,
    write_co_ownership_tables,
)

ROOT = Path(__file__).resolve().parents[2]  # the repository, whatever the working directory
SALT = RELATEDNESS_SETTINGS.salt

GOLDEN = {
    '111110': 'validation',
    '111120': 'validation',
    '111150': 'sealed',
    '111191': 'validation',
    '211120': 'sealed',
    '236220': 'validation',
    '311111': 'sealed',
    '311119': 'sealed',
    '311211': 'validation',
    '311212': 'sealed',
    '311221': 'validation',
    '312111': 'validation',
    '312112': 'sealed',
    '445110': 'validation',
    '445131': 'validation',
    '621210': 'sealed',
}

# -------------------------------------------------------------------------------------------------
# The split
# -------------------------------------------------------------------------------------------------

def test_split_part_matches_the_golden_parts():
    for code, part in GOLDEN.items():
        assert split_part(code, RELATEDNESS_SETTINGS.salt) is RelatednessPart(part)

def test_the_rule_is_the_ita_rule():
    for code in GOLDEN:
        assert split_part(code, SALT).value == ita_split.split_part(code, SALT).value
    assert set(split_rule(RELATEDNESS_SETTINGS)) == set(ita_split.split_rule(ITA_SETTINGS))

def test_split_fingerprint_is_pinned():
    assert split_fingerprint(RELATEDNESS_SETTINGS) == (
        'abb6c1e6b7ed16e73e081bfe682dadb3364041b95ee417ecb4de6712e38906b1'
    )

def test_a_changed_salt_changes_the_fingerprint():
    other = RELATEDNESS_SETTINGS.model_copy(update={'salt': '4bfd5195916df66d0e1c619c6507f2ae'})
    assert split_fingerprint(other) == (
        'bc1fb6c5e38bca1479cf5e0464163be95932713a0f2356b2ee95e852ffff3cc1'
    )

def test_committed_settings_hold_the_planned_values():
    assert load_relatedness_settings(
        ROOT / 'conf/data/relatedness_panel.yaml'
    ) == RELATEDNESS_SETTINGS

def test_settings_refuse_bad_values():
    good = RELATEDNESS_SETTINGS.model_dump()
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
            'k': 20
        },
        {
            'panel': 'ita_text'
        },
        {
            'unknown_key': 1
        },
    ):
        with pytest.raises(ValidationError):
            RelatednessPanelSettings(**{**good, **bad})

def test_fixture_tables_carry_every_stage_10_column_in_order():
    targets, marginals = co_ownership_tables()
    assert list(targets.schema.items()) == list(CO_OWNERSHIP_TARGET_SCHEMA.items())
    assert list(marginals.schema.items()) == list(CO_OWNERSHIP_MARGINAL_SCHEMA.items())
    assert targets.height == 24 and marginals.height == 14

def test_fixture_gains_are_asymmetric():
    for a, b, j in BASE_PAIRS:
        assert j / BASE_MULTI[a] != j / BASE_MULTI[b]

# -------------------------------------------------------------------------------------------------
# Added by the mutant step: input shapes the plan's tests above do not use
# -------------------------------------------------------------------------------------------------

# A second salt, one character off the committed one. Its parts are pinned from a hand computation
# with hashlib, apart from the module under test: some codes change part, and some keep theirs
OTHER_SALT = '4bfd5195916df66d0e1c619c6507f2ae'
OTHER_SALT_PARTS = {
    '111110': 'validation',
    '111120': 'validation',
    '111150': 'validation',
    '111191': 'sealed',
    '211120': 'validation',
    '236220': 'sealed',
    '311111': 'validation',
    '311119': 'sealed',
    '311211': 'sealed',
    '311212': 'sealed',
    '311221': 'validation',
    '312111': 'sealed',
    '312112': 'validation',
    '445110': 'validation',
    '445131': 'validation',
    '621210': 'validation',
}

def test_the_split_rule_is_exactly_the_ita_rule_with_the_panels_salt():
    assert split_rule(RELATEDNESS_SETTINGS) == {
        'salt': '4bfd5195916df66d0e1c619c6507f2ad',
        'rule_version': 1,
        'hash': 'sha256',
        'prefix_bytes': 8,
        'byteorder': 'big',
        'modulus': 2,
        'parts': ['validation', 'sealed'],
    }
    assert split_rule(RELATEDNESS_SETTINGS) == ita_split.split_rule(ITA_SETTINGS) | {'salt': SALT}

def test_the_panel_names_its_parts_and_shares_no_class_with_ita():
    assert RELATEDNESS_PANEL == 'relatedness_co_ownership'
    assert RELATEDNESS_SETTINGS.panel == RELATEDNESS_PANEL
    assert [part.value for part in RelatednessPart] == ['validation', 'sealed']
    assert issubclass(RelatednessPart, str)
    assert RelatednessPart is not ita_split.ItaPart
    assert not isinstance(split_part('111110', SALT), ita_split.ItaPart)

def test_split_part_reads_the_salt():
    assert OTHER_SALT_PARTS != GOLDEN
    for code, part in OTHER_SALT_PARTS.items():
        assert split_part(code, OTHER_SALT) is RelatednessPart(part)

def test_a_code_falls_in_one_part_whatever_was_read_before_it():
    first = {code: split_part(code, SALT) for code in GOLDEN}
    for salt in (OTHER_SALT, SALT, OTHER_SALT):
        for code in reversed(list(GOLDEN)):
            split_part(code, salt)
    assert {code: split_part(code, SALT) for code in GOLDEN} == first
    assert {code: split_part(code, OTHER_SALT).value for code in GOLDEN} == OTHER_SALT_PARTS

def _anchor_codes(targets: pl.DataFrame) -> list:
    return sorted({unit.rsplit(':', 1)[1] for unit in targets.get_column('unit_a')})

def test_a_rebuilt_fixture_with_an_added_industry_keeps_every_old_industry_in_its_old_part():
    before, _ = co_ownership_tables()
    after, _ = co_ownership_tables(pairs=(*BASE_PAIRS, ('236220', '311111', 1)))
    old = {code: split_part(code, SALT) for code in _anchor_codes(before)}
    assert '236220' not in old
    assert _anchor_codes(after) == sorted([*old, '236220'])
    for code, part in old.items():
        assert split_part(code, SALT) is part
        assert part is RelatednessPart(GOLDEN[code])
    # the new industry lands in a part of its own, with no table consulted
    assert split_part('236220', SALT) is RelatednessPart.VALIDATION
    assert split_fingerprint(RELATEDNESS_SETTINGS) == (
        'abb6c1e6b7ed16e73e081bfe682dadb3364041b95ee417ecb4de6712e38906b1'
    )

def test_each_part_holds_codes_of_the_base_fixture():
    parts = {code: split_part(code, SALT) for code in RELATEDNESS_CODES}
    assert set(parts.values()) == set(RelatednessPart)
    assert sum(part is RelatednessPart.SEALED for part in parts.values()) == 7

def test_settings_are_frozen():
    with pytest.raises(ValidationError):
        RELATEDNESS_SETTINGS.salt = OTHER_SALT

def test_settings_refuse_more_bad_values():
    good = RELATEDNESS_SETTINGS.model_dump()
    assert RelatednessPanelSettings(**good) == RELATEDNESS_SETTINGS
    for bad in (
        {
            'salt': SALT + '0'
        },
        {
            'salt': 'g' * 32
        },
        {
            'salt': ''
        },
        {
            'relative_margin': -0.05
        },
        {
            'relative_margin': math.nan
        },
        {
            'k': 5
        },
        {
            'rule_version': 2
        },
        {
            'panel': 'relatedness_co_ownership_2'
        },
    ):
        with pytest.raises(ValidationError):
            RelatednessPanelSettings(**{**good, **bad})
    for missing in good:
        with pytest.raises(ValidationError):
            RelatednessPanelSettings(
                **{
                    key: value
                    for key, value in good.items() if key != missing
                }
            )

def test_load_settings_reads_a_file_and_refuses_a_bad_one(tmp_path):
    good = RELATEDNESS_SETTINGS.model_dump()
    path = tmp_path / 'panel.yaml'
    path.write_text(yaml.safe_dump(good), encoding='utf-8')
    assert load_relatedness_settings(path) == RELATEDNESS_SETTINGS
    assert load_relatedness_settings(str(path)) == RELATEDNESS_SETTINGS
    path.write_text(yaml.safe_dump({**good, 'k': 20}), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_relatedness_settings(path)
    path.write_text(yaml.safe_dump({**good, 'unknown_key': 1}), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_relatedness_settings(path)
    path.write_text(yaml.safe_dump({'panel': good['panel']}), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_relatedness_settings(path)

def test_the_committed_settings_file_names_each_key_once():
    text = (ROOT / 'conf/data/relatedness_panel.yaml').read_text(encoding='utf-8')
    assert yaml.safe_load(text) == RELATEDNESS_SETTINGS.model_dump()
    assert text.count('salt:') == 1 and text.count('4bfd5195916df66d0e1c619c6507f2ad') == 1

# -------------------------------------------------------------------------------------------------
# The fixture tables
# -------------------------------------------------------------------------------------------------

# The brief's base fixture: each code's part, its m and its partners (J)
BRIEF_TABLE = {
    '111110': ('validation', 4, {
        '311111': 2,
        '445110': 1,
        '111120': 2
    }),
    '111120': ('validation', 3, {
        '311211': 1,
        '111110': 2
    }),
    '111150': ('sealed', 2, {
        '111191': 1,
        '311221': 1
    }),
    '111191': ('validation', 1, {
        '111150': 1
    }),
    '311111': ('sealed', 5, {
        '311119': 4,
        '111110': 2
    }),
    '311119': ('sealed', 4, {
        '311111': 4
    }),
    '311211': ('validation', 5, {
        '311212': 3,
        '111120': 1,
        '311221': 2
    }),
    '311212': ('sealed', 3, {
        '311211': 3
    }),
    '311221': ('validation', 3, {
        '311211': 2,
        '111150': 1
    }),
    '312111': ('validation', 6, {
        '312112': 5,
        '445110': 1
    }),
    '312112': ('sealed', 5, {
        '312111': 5
    }),
    '445110': ('validation', 2, {
        '312111': 1,
        '111110': 1,
        '621210': 1
    }),
    '445131': ('validation', 0, {}),
    '621210': ('sealed', 1, {
        '445110': 1
    }),
    '211120': ('sealed', None, {}),
    '236220': ('validation', None, {}),
}

def _targets_by_anchor(targets: pl.DataFrame) -> dict:
    anchors: dict = {}
    for row in targets.iter_rows(named=True):
        code_a, code_b = (row[key].rsplit(':', 1)[1] for key in ('unit_a', 'unit_b'))
        anchors.setdefault(code_a, {})[code_b] = row
    return anchors

def test_the_brief_table_holds_in_the_fixture():
    targets, marginals = co_ownership_tables()
    anchors = _targets_by_anchor(targets)
    assert sorted(BRIEF_TABLE) == sorted(GOLDEN) == list(RELATEDNESS_CODES)
    multi = {
        node_id.rsplit(':', 1)[1]: m
        for node_id, m in zip(
            marginals.get_column('node_id'), marginals.get_column('n_multi_industry_employers')
        )
    }
    for code, (part, m, partners) in BRIEF_TABLE.items():
        assert split_part(code, SALT) is RelatednessPart(part)
        assert multi.get(code) == m
        assert BASE_MULTI.get(code) == m
        assert {b: row['n_unweighted'] for b, row in anchors.get(code, {}).items()} == partners
    assert dict(BASE_MULTI) == {code: m for code, (_, m, _) in BRIEF_TABLE.items() if m is not None}

def test_base_pairs_are_the_twelve_unordered_pairs_with_a_below_b():
    assert len(BASE_PAIRS) == 12 and len({(a, b) for a, b, _ in BASE_PAIRS}) == 12
    assert all(a < b and j >= 1 for a, b, j in BASE_PAIRS)
    assert all(
        a in BASE_MULTI and b in BASE_MULTI and j <= min(BASE_MULTI[a], BASE_MULTI[b])
        for a, b, j in BASE_PAIRS
    )

def test_each_pair_is_emitted_in_both_directions_with_one_j():
    targets, _ = co_ownership_tables()
    rows = {(row['unit_a'], row['unit_b']): row for row in targets.iter_rows(named=True)}
    assert len(rows) == 24
    for a, b, j in BASE_PAIRS:
        assert rows[(node(a), node(b))]['n_unweighted'] == j
        assert rows[(node(b), node(a))]['n_unweighted'] == j
        assert rows[(node(a), node(b))]['n_weighted'] == float(j)
        assert rows[(node(a), node(b))]['value'] == j / (BASE_MULTI[a] + 3)
        assert rows[(node(b), node(a))]['value'] == j / (BASE_MULTI[b] + 3)

def test_target_rows_carry_stage_10s_metadata():
    targets, _ = co_ownership_tables()
    constant = {
        'source': 'osha_ita',
        'family': 'co_ownership',
        'unit_a_type': 'naics_national_industry',
        'unit_b_type': 'naics_national_industry',
        'measure': 'employer_share',
        'se_method': 'none',
        'sample_set': 'ita_300a_2023_2025',
        'vintage': 'naics=2022',
        'universe': 'fixture',
    }
    for column, value in constant.items():
        assert set(targets.get_column(column)) == {value}
    assert targets.get_column('se').null_count() == targets.height
    assert set(map(tuple, targets.get_column('flags').to_list())) == {('size_gate_pending', )}
    assert node('311111') == 'naics:2022:national_industry:311111'
    assert SECOND_FAMILY == 'relatedness_second_family'

def test_targets_sort_by_the_key_columns_and_marginals_by_node_id():
    targets, marginals = co_ownership_tables()
    key = ['unit_a', 'unit_b']
    assert targets.equals(targets.sort(key))
    assert targets.select(key).row(0) == (node('111110'), node('111120'))
    assert marginals.get_column('node_id').to_list() == sorted(node(code) for code in BASE_MULTI)
    # a different input order gives the same tables
    shuffled, shuffled_marginals = co_ownership_tables(
        pairs=tuple(reversed(BASE_PAIRS)), multi=dict(reversed(list(BASE_MULTI.items())))
    )
    assert shuffled.equals(targets) and shuffled_marginals.equals(marginals)

def test_marginals_follow_the_m_plus_three_rule():
    _, marginals = co_ownership_tables()
    for row in marginals.iter_rows(named=True):
        m = BASE_MULTI[row['node_id'].rsplit(':', 1)[1]]
        assert row['n_multi_industry_employers'] == m
        assert row['n_employers'] == m + 3
        assert row['n_records'] == 2 * (m + 3)
        assert row['n_employers_including_padded'] == m + 3
        assert row['n_employers_partial'] == 0

def test_a_code_outside_multi_has_no_marginal_row_and_m_of_zero():
    pairs = (*BASE_PAIRS, ('236220', '311111', 2))
    targets, marginals = co_ownership_tables(pairs=pairs)
    rows = {(row['unit_a'], row['unit_b']): row for row in targets.iter_rows(named=True)}
    assert targets.height == 26 and marginals.height == 14
    assert node('236220') not in set(marginals.get_column('node_id'))
    assert rows[(node('236220'), node('311111'))]['value'] == 2 / 3
    assert rows[(node('311111'), node('236220'))]['value'] == 2 / (5 + 3)

def test_a_pair_may_exceed_a_marginal_count():
    targets, marginals = co_ownership_tables(
        pairs=(('111110', '311111', 9), ), multi={
            '111110': 4,
            '311111': 5
        }
    )
    assert targets.get_column('n_unweighted').to_list() == [9, 9]
    assert targets.get_column('value').to_list() == [9 / 7, 9 / 8]
    assert marginals.height == 2

def test_tables_round_trip_through_parquet(tmp_path):
    targets, marginals = co_ownership_tables()
    targets_path, marginals_path = write_co_ownership_tables(tmp_path, targets, marginals)
    assert (targets_path, marginals_path) == (tmp_path / TARGETS_FILE, tmp_path / MARGINALS_FILE)
    assert TARGETS_FILE == 'ita_co_ownership.parquet'
    assert MARGINALS_FILE == 'co_ownership_marginals.parquet'
    assert sorted(path.name
                  for path in tmp_path.iterdir()) == sorted([TARGETS_FILE, MARGINALS_FILE])
    assert pl.read_parquet(targets_path).equals(targets)
    assert pl.read_parquet(marginals_path).equals(marginals)
    assert dict(pl.read_parquet(targets_path).schema) == CO_OWNERSHIP_TARGET_SCHEMA

def test_tree_distance_is_twice_the_levels_below_the_common_ancestor():
    distance = tree_distance(['311111', '311119', '311211', '312111', '111110', '445110', '445131'])
    assert distance.shape == (7, 7) and distance.dtype == np.float64
    assert np.array_equal(distance, distance.T) and not distance.diagonal().any()
    # one level below a shared five-digit code, a shared industry group, a shared sector, none
    assert distance[0, 1] == 2 and distance[0, 2] == 6 and distance[0, 3] == 8
    assert distance[0, 4] == 10 and distance[5, 6] == 4

def test_tree_distance_agrees_with_the_codebooks_tree_metric_on_six_digit_codes():
    assert np.array_equal(
        tree_distance(RELATEDNESS_CODES),
        tree_distance_matrix(RELATEDNESS_CODES).astype(float)
    )
    assert tree_distance([]).shape == (0, 0)
