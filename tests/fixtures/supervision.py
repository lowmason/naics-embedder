'''
Hand-written supervision fixtures shared across the Stage-3 supervision test suites.

Expected values in these fixtures are written by hand; tests must never use production derivation
code as their oracle.
'''

from pathlib import Path
from typing import List

import polars as pl
import pytest
import torch

from naics_embedder.data.redirections import REDIRECTIONS_SCHEMA
from naics_embedder.data.supervision_bundle import (
    generate_supervision_bundle,
    generate_supervision_bundle_from_frames,
)
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.candidates import NegativeCandidateBatch
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
# Candidate batches: every aligned field carries a distinguishable per-slot ordinal (1, 2, 3, ...)
# -------------------------------------------------------------------------------------------------

def _negative_candidate_batch(
    code_ids: list[int],
    explicit_exclusions: list[bool],
) -> NegativeCandidateBatch:
    count = len(code_ids)
    shape = (1, count)
    slots = torch.arange(count, dtype=torch.long)
    candidate_uid = torch.stack([torch.zeros_like(slots),
                                 torch.zeros_like(slots), slots], dim=-1).unsqueeze(0)
    ordinal = torch.arange(1, count + 1, dtype=torch.float64).unsqueeze(0)
    anchor_excludes = torch.tensor([explicit_exclusions], dtype=torch.bool)
    candidate_excludes = torch.zeros(shape, dtype=torch.bool)
    explicit = anchor_excludes | candidate_excludes
    return NegativeCandidateBatch(
        candidate_uid=candidate_uid,
        code_id=torch.tensor([code_ids], dtype=torch.long),
        embedding=torch.stack([ordinal, ordinal + 0.5], dim=-1),
        structural_distance=ordinal.clone(),
        structural_relation_id=ordinal.to(torch.int16),
        anchor_excludes_candidate=anchor_excludes,
        candidate_excludes_anchor=candidate_excludes,
        is_explicit_exclusion=explicit,
        semantic_target_id=torch.where(explicit, 2, 0).to(torch.int8),
        semantic_source_id=torch.where(explicit, 2, 0).to(torch.int8),
        sampling_role_id=torch.full(shape, 2, dtype=torch.int8),
        sampling_provenance_id=torch.full(shape, 2, dtype=torch.int8),
        relation_margin=ordinal.clone(),
        distance_margin=ordinal.clone(),
        router_gate_probs=torch.stack([ordinal / 10.0, 1.0 - ordinal / 10.0], dim=-1),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        runtime_fields={'difficulty': ordinal * 10.0},
    )

@pytest.fixture
def candidate_batch() -> NegativeCandidateBatch:
    return _negative_candidate_batch([101, 102, 103], [False, False, False])

@pytest.fixture
def candidate_batch_with_exclusions() -> NegativeCandidateBatch:
    return _negative_candidate_batch(
        [20, 21, 22, 30, 31, 32],
        [True, True, True, False, False, False],
    )

@pytest.fixture
def candidate_batch_with_duplicate_code() -> NegativeCandidateBatch:
    return _negative_candidate_batch([101, 101, 102], [False, False, False])

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
