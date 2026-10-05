'''
Hand-written supervision fixtures shared across the Stage-3 supervision test suites.

Expected values in these fixtures are written by hand; tests must never use production derivation
code as their oracle.
'''

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl
import pytest

from naics_embedder.data.redirections import REDIRECTIONS_SCHEMA
from naics_embedder.data.supervision_bundle import (
    generate_supervision_bundle,
    generate_supervision_bundle_from_frames,
)
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, load_validated_bundle
from naics_embedder.supervision.schema import InputWindowRecord
from naics_embedder.utils.config import SupervisionBuildConfig

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
        | {'structural_distance': [
            2.0,
            2.0,
            10.0,
            10.0,
            2.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]}
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
                2.0,
                2.0,
                10.0,
                10.0,
                2.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
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
            # No five-digit code, so no unary pair
            'unary_pair': [False] * 10,
        }
    )

# -------------------------------------------------------------------------------------------------
# The five-code bundle, with the index-roles and redirections members every bundle carries
# -------------------------------------------------------------------------------------------------

INDEX_ROLE_ROWS = [
    (0, '111111', 'Soybean farming', 'examples'),
    (1, '111111', 'Edamame farming', 'validation'),
    (2, '111112', 'Canola farming', 'examples'),
    (3, '111112', 'Sunflower farming', 'test'),
    (4, '222222', 'Coal mining', 'examples'),
    (5, '222222', 'Lignite mining', 'training'),
]
INDEX_ROLE_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'text': pl.Utf8, 'role': pl.Utf8}

@pytest.fixture
def index_roles_fixture() -> pl.DataFrame:
    return pl.DataFrame(INDEX_ROLE_ROWS, schema=INDEX_ROLE_SCHEMA, orient='row')

# One cross-reference row per exclusion of the pair facts: '111111' sends peanut growing to
# '111113', and '222222' sends canola crushing to '111112'
REDIRECTION_ROWS = [
    (
        0,
        'cross_reference',
        '111111',
        'Growing peanuts--are classified in Industry 111113.',
        'Growing peanuts',
        ['111113'],
        [],
        False,
    ),
    (
        1,
        'cross_reference',
        '222222',
        'Canola crushing--are classified in Industry 111112.',
        'Canola crushing',
        ['111112'],
        [],
        False,
    ),
]

# The five codes' texts are short, so none exceeds the window
FIVE_CODE_INPUT_WINDOW = {
    'backbone': 'sentence-transformers/all-MiniLM-L6-v2',
    'window': 128,
    'channels': {
        'title': {
            'present': 5,
            'over': 0,
            'share': 0.0
        },
        'description': {
            'present': 5,
            'over': 0,
            'share': 0.0
        },
        'examples': {
            'present': 3,
            'over': 0,
            'share': 0.0
        },
        'excluded': {
            'present': 2,
            'over': 0,
            'share': 0.0
        },
    },
}

@pytest.fixture
def redirections_fixture() -> pl.DataFrame:
    return pl.DataFrame(REDIRECTION_ROWS, schema=REDIRECTIONS_SCHEMA, orient='row')

@pytest.fixture
def text_descriptions_fixture(descriptions_fixture) -> pl.DataFrame:
    '''
    The five-code descriptions with text channels.

    Examples hold examples-role entries only, and each exclusion channel is its code's one
    redirection row.
    '''

    examples = {'111111': 'Soybean farming', '111112': 'Canola farming', '222222': 'Coal mining'}
    excluded = {row[2]: row[3] for row in REDIRECTION_ROWS}
    return descriptions_fixture.with_columns(
        title=pl.concat_str(pl.lit('Industry '), pl.col('code')),
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.col('code').replace_strict(excluded, default=None),
    )

@pytest.fixture
def build_bundle(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture,
    redirections_fixture
):
    '''
    Build the five-code bundle with ``generate_supervision_bundle_from_frames``.

    The bundle is ``bundle-a`` under ``tmp_path``; keyword arguments override single inputs.
    '''

    def build(**overrides) -> Path:
        inputs = {
            'output_root': tmp_path,
            'bundle_id': 'bundle-a',
            'generator_revision': 'revision-a',
            'naics_vintage': 2022,
            'descriptions': text_descriptions_fixture,
            'pair_facts': pair_facts_fixture,
            'index_roles': index_roles_fixture,
            'redirections': redirections_fixture,
            'input_window': InputWindowRecord.model_validate(FIVE_CODE_INPUT_WINDOW),
        }
        return generate_supervision_bundle_from_frames(**{**inputs, **overrides})

    return build

@pytest.fixture
def generated_bundle(build_bundle):
    return build_bundle()

@pytest.fixture
def validated_bundle(generated_bundle):
    return load_validated_bundle(
        generated_bundle,
        expected_contract='stage3-supervision-v2',
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

# '311111' sends sawmilling to '321111' in a cross-reference row, and an "Excluded" paragraph of
# '441111' names '311211': the hierarchy's two exclusions
HIERARCHY_REDIRECTION_ROWS = [
    (
        0,
        'cross_reference',
        '311111',
        'Sawmilling--are classified in Industry 321111.',
        'Sawmilling',
        ['321111'],
        [],
        False,
    ),
    (
        1,
        'description',
        '441111',
        'Flour milling is classified in Industry 311211.',
        None,
        ['311211'],
        [],
        False,
    ),
]

@pytest.fixture
def hierarchy_descriptions() -> pl.DataFrame:
    excluded_codes = {
        '311111': ['321111'],
        '441111': ['311211'],
    }
    excluded = {row[2]: row[3] for row in HIERARCHY_REDIRECTION_ROWS}
    return pl.DataFrame(
        {
            'index': list(range(len(HIERARCHY_CODES))),
            'level': [len(code) for code in HIERARCHY_CODES],
            'code': list(HIERARCHY_CODES),
            'title': [f'Industry {code}' for code in HIERARCHY_CODES],
            'description': ['This industry comprises establishments.'] * len(HIERARCHY_CODES),
            'examples': [None] * len(HIERARCHY_CODES),
            'excluded': [excluded.get(code) for code in HIERARCHY_CODES],
            'excluded_codes': [excluded_codes.get(code) for code in HIERARCHY_CODES],
        },
        schema_overrides={
            'index': pl.UInt32,
            'level': pl.UInt8,
            'examples': pl.Utf8,
            'excluded_codes': pl.List(pl.Utf8),
        },
    )

@pytest.fixture
def hierarchy_descriptions_parquet(tmp_path, hierarchy_descriptions) -> str:
    path = tmp_path / 'hierarchy_descriptions.parquet'
    hierarchy_descriptions.write_parquet(path)
    return str(path)

@pytest.fixture
def hierarchy_redirections() -> pl.DataFrame:
    return pl.DataFrame(HIERARCHY_REDIRECTION_ROWS, schema=REDIRECTIONS_SCHEMA, orient='row')

def _count_words(texts: List[str]) -> List[int]:
    return [len(text.split()) + 2 for text in texts]

@pytest.fixture
def count_words():
    '''Token counts for tests: one token per word, plus two for [CLS] and [SEP].'''

    return _count_words

@pytest.fixture
def hierarchy_build_config(
    tmp_path, hierarchy_descriptions_parquet, hierarchy_redirections
) -> SupervisionBuildConfig:
    '''
    The build configuration of the hierarchy's bundle, its inputs written under ``tmp_path``.

    The hierarchy has no index entries, so its role table is empty.
    '''

    roles_path = tmp_path / 'hierarchy_index_roles.parquet'
    redirections_path = tmp_path / 'hierarchy_redirections.parquet'
    pl.DataFrame(schema=INDEX_ROLE_SCHEMA).write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    return SupervisionBuildConfig(
        descriptions_parquet=hierarchy_descriptions_parquet,
        index_roles_parquet=str(roles_path),
        redirections_parquet=str(redirections_path),
        output_root=str(tmp_path / 'bundles'),
    )

@pytest.fixture
def hierarchy_manifest(hierarchy_build_config) -> Path:
    '''The manifest of the bundle that the production path builds from the hierarchy.'''

    return generate_supervision_bundle(hierarchy_build_config, count_tokens=_count_words)

# -------------------------------------------------------------------------------------------------
# The reference bundle (Stage 7): the hierarchy's codes with every kind of task query
#
# The production path builds it from the hierarchy's 17 codes (levels 2-6, the sectors '31' and
# '44', four unary pairs). Its rows give, by hand: training entries, one of whose texts is also a
# level-6 phrase (the two merge) and one a level-5 phrase (the two stay apart); a validation entry
# for each of the four six-digit codes, so an outcome read ranks each against the others; a
# two-target phrase; a phrase with destinations at two levels; a phrase at each of levels 2, 3
# and 5; a lineal-only row; a withheld row; and an "Excluded" paragraph.
# -------------------------------------------------------------------------------------------------

IndexRoleRow = Tuple[int, str, str, str]
RedirectionRow = Tuple[int, str, str, str, Optional[str], List[str], List[str], bool]

REFERENCE_INDEX_ROLE_ROWS: Tuple[IndexRoleRow, ...] = (
    (0, '311111', 'Dog food manufacturing', 'examples'),
    (1, '311111', 'Cat food manufacturing', 'training'),
    (2, '311111', 'Pet food canning', 'validation'),
    (3, '311211', 'Wheat flour milling', 'examples'),
    (4, '311211', 'Corn meal milling', 'training'),
    (5, '311211', 'Rice flour milling', 'validation'),
    (6, '311211', 'Oat bran milling', 'test'),
    (7, '321111', 'Lumber sawing', 'examples'),
    # Also the phrase of redirection row 0, at level 6
    (8, '321111', 'Sawmilling', 'training'),
    (9, '321111', 'Timber resawing mills', 'validation'),
    (10, '441111', 'Automobile dealerships', 'examples'),
    (11, '441111', 'New car dealers', 'training'),
    # Also the phrase of redirection row 7, whose destination is this code's parent, at level 5
    (12, '441111', 'Dealing in new cars', 'training'),
    (13, '441111', 'Selling new automobiles', 'validation'),
    (14, '441111', 'Car showroom operation', 'test'),
)

REFERENCE_REDIRECTION_ROWS: Tuple[RedirectionRow, ...] = (
    # A level-6 phrase that is also a training entry's text
    (
        0,
        'cross_reference',
        '311111',
        'Sawmilling--are classified in Industry 321111.',
        'Sawmilling',
        ['321111'],
        [],
        False,
    ),
    # Two destinations at one level: one query with two targets
    (
        1,
        'cross_reference',
        '441111',
        'Food manufacturing--are classified in Industry 311111 and Industry 311211.',
        'Food manufacturing',
        ['311111', '311211'],
        [],
        False,
    ),
    # Its one named code is its code's ancestor, so the row stays text only (Req 8)
    (
        2,
        'cross_reference',
        '311111',
        'Mixed food making--are classified in Industry Group 3111.',
        'Mixed food making',
        ['3111'],
        ['3111'],
        False,
    ),
    # The validation entry 'Rice flour milling' leaks into it, so it is withheld and carries no
    # phrase; its named code still counts as an exclusion
    (
        3,
        'cross_reference',
        '321111',
        'Rice flour milling--are classified in Industry 311211.',
        None,
        ['311211'],
        [],
        True,
    ),
    # Destinations at two levels: a query at each
    (
        4,
        'cross_reference',
        '321111',
        'Wood flour grinding--are classified in Industry Group 3112 and Industry 311211.',
        'Wood flour grinding',
        ['3112', '311211'],
        [],
        False,
    ),
    # A phrase at each of levels 2, 3 and 5, each sent away from a six-digit code
    (
        5,
        'cross_reference',
        '311211',
        'Retailing new cars--are classified in Sector 44.',
        'Retailing new cars',
        ['44'],
        [],
        False,
    ),
    (
        6,
        'cross_reference',
        '441111',
        'Wood products manufacturing--are classified in Subsector 321.',
        'Wood products manufacturing',
        ['321'],
        [],
        False,
    ),
    (
        7,
        'cross_reference',
        '311111',
        'Dealing in new cars--are classified in Industry 44111.',
        'Dealing in new cars',
        ['44111'],
        [],
        False,
    ),
    # An "Excluded" paragraph names a code but carries no phrase
    (
        8,
        'description',
        '441111',
        'Flour milling is classified in Industry 311211.',
        None,
        ['311211'],
        [],
        False,
    ),
)

def _reference_descriptions(
    index_role_rows: Sequence[IndexRoleRow],
    redirection_rows: Sequence[RedirectionRow],
) -> pl.DataFrame:
    '''
    The hierarchy's descriptions with the examples and exclusion channels the rows give.

    A code's examples channel is its examples-role entries, joined by '; ' in entry order. Its
    exclusion channel comes from its redirection rows: the texts of those not withheld, joined by
    one space in table order, and every code they name, withheld rows included, once each in order
    of first appearance.
    '''

    examples: Dict[str, List[str]] = {}
    for _, code, text, role in sorted(index_role_rows):
        if role == 'examples':
            examples.setdefault(code, []).append(text)
    excluded: Dict[str, List[str]] = {}
    excluded_codes: Dict[str, List[str]] = {}
    for _, _, code, text, _, named, _, withheld in redirection_rows:
        if not withheld:
            excluded.setdefault(code, []).append(text)
        codes = excluded_codes.setdefault(code, [])
        for other in named:
            if other not in codes:
                codes.append(other)
    return pl.DataFrame(
        {
            'index': list(range(len(HIERARCHY_CODES))),
            'level': [len(code) for code in HIERARCHY_CODES],
            'code': list(HIERARCHY_CODES),
            'title': [f'Industry {code}' for code in HIERARCHY_CODES],
            'description': ['This industry comprises establishments.'] * len(HIERARCHY_CODES),
            'examples': ['; '.join(examples.get(code, [])) or None for code in HIERARCHY_CODES],
            'excluded': [' '.join(excluded.get(code, [])) or None for code in HIERARCHY_CODES],
            'excluded_codes': [excluded_codes.get(code) or None for code in HIERARCHY_CODES],
        },
        schema_overrides={
            'index': pl.UInt32,
            'level': pl.UInt8,
            'examples': pl.Utf8,
            'excluded': pl.Utf8,
            'excluded_codes': pl.List(pl.Utf8),
        },
    )

def build_reference_bundle(
    root: Path,
    *,
    index_role_rows: Sequence[IndexRoleRow] = REFERENCE_INDEX_ROLE_ROWS,
    redirection_rows: Sequence[RedirectionRow] = REFERENCE_REDIRECTION_ROWS,
) -> Path:
    '''
    Build a reference bundle under ``root`` through the production path; return its manifest.

    The inputs go to ``reference_descriptions.parquet``, ``reference_index_roles.parquet`` and
    ``reference_redirections.parquet`` under ``root`` (the manifest's ``generation_parameters``
    name them), and the bundle to ``<root>/bundles/<bundle_id>``. The build derives the pair facts'
    exclusion flags from the descriptions' excluded codes, so the rows given here fix the
    redirection table, the exclusion channel and the pair facts' exclusions together. A plain
    function rather than a fixture, so that fixtures of any scope can build one.
    '''

    root.mkdir(parents=True, exist_ok=True)
    descriptions_path = root / 'reference_descriptions.parquet'
    roles_path = root / 'reference_index_roles.parquet'
    redirections_path = root / 'reference_redirections.parquet'
    _reference_descriptions(index_role_rows, redirection_rows).write_parquet(descriptions_path)
    roles = pl.DataFrame(list(index_role_rows), schema=INDEX_ROLE_SCHEMA, orient='row')
    roles.write_parquet(roles_path)
    redirections = pl.DataFrame(list(redirection_rows), schema=REDIRECTIONS_SCHEMA, orient='row')
    redirections.write_parquet(redirections_path)
    build_config = SupervisionBuildConfig(
        descriptions_parquet=str(descriptions_path),
        index_roles_parquet=str(roles_path),
        redirections_parquet=str(redirections_path),
        output_root=str(root / 'bundles'),
    )
    return generate_supervision_bundle(build_config, count_tokens=_count_words)

@pytest.fixture
def reference_manifest(tmp_path) -> Path:
    '''The manifest of the reference bundle, built under ``tmp_path``.'''

    return build_reference_bundle(tmp_path / 'reference')

@pytest.fixture
def reference_bundle(reference_manifest) -> ValidatedSupervisionBundle:
    '''The reference bundle, as the loader validates it.'''

    return load_validated_bundle(reference_manifest)
