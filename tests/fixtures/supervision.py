'''
Hand-written supervision fixtures shared across the Stage-3 supervision test suites.

Expected values in these fixtures are written by hand; tests must never use production derivation
code as their oracle.
'''

import polars as pl
import pytest

from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames


@pytest.fixture
def descriptions_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'index': [0, 1, 2, 3, 4],
            'code': ['111111', '111112', '111113', '222222', '333333'],
            'excluded_codes': [['111113'], None, None, ['111112'], None],
        }
    )


@pytest.fixture
def structural_frames_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    pair_columns = {
        'idx_i': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        'idx_j': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
        'code_i': [
            '111111',
            '111111',
            '111111',
            '111111',
            '111112',
            '111112',
            '111112',
            '111113',
            '111113',
            '222222',
        ],
        'code_j': [
            '111112',
            '111113',
            '222222',
            '333333',
            '111113',
            '222222',
            '333333',
            '222222',
            '333333',
            '333333',
        ],
    }
    distances = pl.DataFrame(
        pair_columns
        | {
            'structural_distance': [
                0.5,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ]
        }
    )
    relations = pl.DataFrame(
        pair_columns
        | {
            'structural_relation_id': [1, 2, 99, 99, 3, 99, 99, 99, 99, 99],
            'structural_relation_name': [
                'child',
                'sibling',
                'cross_sector',
                'cross_sector',
                'grandchild',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
            ],
        }
    )
    return distances, relations


@pytest.fixture
def pair_facts_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
            'code_j_id': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
            'code_i': [
                '111111',
                '111111',
                '111111',
                '111111',
                '111112',
                '111112',
                '111112',
                '111113',
                '111113',
                '222222',
            ],
            'code_j': [
                '111112',
                '111113',
                '222222',
                '333333',
                '111113',
                '222222',
                '333333',
                '222222',
                '333333',
                '333333',
            ],
            'structural_distance': [
                0.5,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ],
            'structural_relation_id': [1, 2, 99, 99, 3, 99, 99, 99, 99, 99],
            'structural_relation_name': [
                'child',
                'sibling',
                'cross_sector',
                'cross_sector',
                'grandchild',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
            ],
            'code_i_excludes_code_j': [
                False, True, False, False, False, False, False, False, False, False
            ],
            'code_j_excludes_code_i': [
                False, False, False, False, False, True, False, False, False, False
            ],
            'is_explicit_exclusion': [
                False, True, False, False, False, True, False, False, False, False
            ],
        }
    )


@pytest.fixture
def generated_bundle(tmp_path, descriptions_fixture, pair_facts_fixture):
    return generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )


# -------------------------------------------------------------------------------------------------
# Production-shaped hierarchy
#
# Unlike the five-code fixtures above, this hierarchy mirrors the real descriptions artifact:
# `index` follows lexicographic code order (a depth-first walk), level equals code length, and the
# 31-33 manufacturing sectors share one tree rooted at '31'. It therefore contains a canonical
# (shallower-first) pair whose code IDs are descending ('3112' before '31111'), a same-level
# cross-prefix pair inside a merged sector ('311'/'321'), and exclusions on both a merged-sector
# pair and a cross-sector pair.
# -------------------------------------------------------------------------------------------------

HIERARCHY_CODES = (
    '31',
    '311',
    '3111',
    '31111',
    '311111',
    '3112',
    '31121',
    '311211',
    '321',
    '3211',
    '32111',
    '321111',
    '44',
    '441',
    '4411',
    '44111',
    '441111',
)


@pytest.fixture
def hierarchy_descriptions() -> pl.DataFrame:
    excluded_codes = {
        '311111': ['321111'],
        '441111': ['311211'],
    }
    return pl.DataFrame(
        {
            'index': list(range(len(HIERARCHY_CODES))),
            'level': [len(code) for code in HIERARCHY_CODES],
            'code': list(HIERARCHY_CODES),
            'title': [f'Industry {code}' for code in HIERARCHY_CODES],
            'excluded_codes': [excluded_codes.get(code) for code in HIERARCHY_CODES],
        },
        schema_overrides={
            'index': pl.UInt32,
            'level': pl.UInt8,
            'excluded_codes': pl.List(pl.Utf8),
        },
    )


@pytest.fixture
def hierarchy_descriptions_parquet(tmp_path, hierarchy_descriptions) -> str:
    path = tmp_path / 'hierarchy_descriptions.parquet'
    hierarchy_descriptions.write_parquet(path)
    return str(path)
