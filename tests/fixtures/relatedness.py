'''
Stage 10's two co-ownership tables on a miniature panel, and D* for the fixture's codes (plan 21).

``co_ownership_tables`` builds the target table and the marginal table with the exact column names,
order and dtypes of embedding-data at ``e6e12b2``: ``TARGET_SCHEMA`` (``target_contract.py:12``) is
copied into ``CO_OWNERSHIP_TARGET_SCHEMA``, and ``METADATA`` and ``MARGINAL_SCHEMA``
(``ita_co_ownership.py:38-52``) into the constant columns and ``CO_OWNERSHIP_MARGINAL_SCHEMA``. The
base fixture has 12 unordered pairs over the 14 codes of ``BASE_MULTI``, each emitted in both
directions with one J, so its target table has 24 rows. Its marginal table has 14 rows. 445131 has
a marginal row and no partner: a candidate, never an anchor. 211120 and 236220 are in
``RELATEDNESS_CODES`` and have no marginal row.

A row's ``value`` is J / (m(a) + 3), so J / m(a) differs from J / m(b) on every base pair. A code
outside ``multi`` has m of 0. Nothing here reads embedding-data. ``write_co_ownership_tables`` writes
under a directory the test gives, ``tmp_path``.
'''

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.decoding import lca_level
from naics_embedder.panels.relatedness_split import RelatednessPanelSettings

# Stage 10's ita_co_ownership.parquet: every targets-v1 column, in its order, with its dtype
CO_OWNERSHIP_TARGET_SCHEMA: Dict[str, pl.DataType] = {
    'source': pl.Utf8,
    'family': pl.Utf8,
    'unit_a_type': pl.Utf8,
    'unit_a': pl.Utf8,
    'unit_b_type': pl.Utf8,
    'unit_b': pl.Utf8,
    'measure': pl.Utf8,
    'value': pl.Float64,
    'se': pl.Float64,
    'se_method': pl.Utf8,
    'n_unweighted': pl.UInt64,
    'n_weighted': pl.Float64,
    'sample_set': pl.Utf8,
    'vintage': pl.Utf8,
    'universe': pl.Utf8,
    'flags': pl.List(pl.Utf8),
}

# Stage 10's co_ownership_marginals.parquet
CO_OWNERSHIP_MARGINAL_SCHEMA: Dict[str, pl.DataType] = {
    'node_id': pl.Utf8,
    'n_employers': pl.UInt64,
    'n_multi_industry_employers': pl.UInt64,
    'n_records': pl.UInt64,
    'n_employers_including_padded': pl.UInt64,
    'n_employers_partial': pl.UInt64,
}

TARGETS_FILE = 'ita_co_ownership.parquet'
MARGINALS_FILE = 'co_ownership_marginals.parquet'

# Stage 10's METADATA, with a fixture universe
_METADATA = {
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
_SIZE_LABEL = 'size_gate_pending'

# The key a target table is sorted by: every column but the measured ones
_TARGET_KEY_COLUMNS = [
    column for column in CO_OWNERSHIP_TARGET_SCHEMA
    if column not in {'value', 'se', 'n_unweighted', 'n_weighted', 'flags'}
]

# The sixteen codes of the base fixture: the bundle's six-digit codes, sorted. 211120 and 236220
# have no marginal row
RELATEDNESS_CODES = (
    '111110',
    '111120',
    '111150',
    '111191',
    '211120',
    '236220',
    '311111',
    '311119',
    '311211',
    '311212',
    '311221',
    '312111',
    '312112',
    '445110',
    '445131',
    '621210',
)

# m, the employers in each of the 14 industries with a marginal row that also operate in another
BASE_MULTI: Dict[str, int] = {
    '111110': 4,
    '111120': 3,
    '111150': 2,
    '111191': 1,
    '311111': 5,
    '311119': 4,
    '311211': 5,
    '311212': 3,
    '311221': 3,
    '312111': 6,
    '312112': 5,
    '445110': 2,
    '445131': 0,
    '621210': 1,
}

# The 12 unordered pairs (a, b, J), a < b
BASE_PAIRS: Tuple[Tuple[str, str, int], ...] = (
    ('111110', '311111', 2),
    ('111110', '445110', 1),
    ('111110', '111120', 2),
    ('111120', '311211', 1),
    ('111150', '111191', 1),
    ('111150', '311221', 1),
    ('311111', '311119', 4),
    ('311211', '311212', 3),
    ('311211', '311221', 2),
    ('312111', '312112', 5),
    ('312111', '445110', 1),
    ('445110', '621210', 1),
)

# The committed salt: the fixture's parts are those of conf/data/relatedness_panel.yaml
RELATEDNESS_SETTINGS = RelatednessPanelSettings(
    panel='relatedness_co_ownership',
    rule_version=1,
    salt='4bfd5195916df66d0e1c619c6507f2ad',
    relative_margin=0.05,
    k=10,
)

# A second judging family's name, for the tests that add one (P4)
SECOND_FAMILY = 'relatedness_second_family'

# -------------------------------------------------------------------------------------------------
# The tables
# -------------------------------------------------------------------------------------------------

def node(code: str) -> str:
    '''The node id of a six-digit national industry in NAICS 2022.'''

    return f'naics:2022:national_industry:{code}'

def co_ownership_tables(
    pairs: Sequence[Tuple[str, str, int]] = BASE_PAIRS,
    multi: Mapping[str, int] = BASE_MULTI,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    The target table and the marginal table, with Stage 10's columns.

    Args:
        pairs: Unordered pairs (a, b, J). Each is emitted in both directions with one J.
        multi: m for each industry with a marginal row. A code outside it has no marginal row, and
            its m is 0 in a target row's value.

    Returns:
        The target table sorted by every column but ``value``, ``se``, ``n_unweighted``,
        ``n_weighted`` and ``flags``, and the marginal table sorted by ``node_id``, each cast to
        its schema. A row's ``value`` is J / (m(a) + 3), its ``se`` is null, its ``n_unweighted``
        is J and its ``n_weighted`` is J as a float. A marginal row has ``n_employers`` of m + 3.
    '''

    rows = []
    for code_a, code_b, j in pairs:
        for source, target in ((code_a, code_b), (code_b, code_a)):
            rows.append(
                {
                    **_METADATA,
                    'unit_a': node(source),
                    'unit_b': node(target),
                    'value': j / (multi.get(source, 0) + 3),
                    'se': None,
                    'n_unweighted': j,
                    'n_weighted': float(j),
                    'flags': [_SIZE_LABEL],
                }
            )
    targets = pl.DataFrame(
        {
            name: [row[name] for row in rows]
            for name in CO_OWNERSHIP_TARGET_SCHEMA
        },
        schema=CO_OWNERSHIP_TARGET_SCHEMA,
    ).sort(_TARGET_KEY_COLUMNS)
    marginals = pl.DataFrame(
        {
            'node_id': [node(code) for code in multi],
            'n_employers': [m + 3 for m in multi.values()],
            'n_multi_industry_employers': list(multi.values()),
            'n_records': [2 * (m + 3) for m in multi.values()],
            'n_employers_including_padded': [m + 3 for m in multi.values()],
            'n_employers_partial': [0 for _ in multi],
        },
        schema=CO_OWNERSHIP_MARGINAL_SCHEMA,
    ).sort('node_id')
    return targets, marginals

def write_co_ownership_tables(
    directory: Path,
    targets: pl.DataFrame,
    marginals: pl.DataFrame,
) -> Tuple[Path, Path]:
    '''
    Write the two tables under their Stage 10 file names.

    Args:
        directory: An existing directory, ``tmp_path`` in a test.
        targets: The target table.
        marginals: The marginal table.

    Returns:
        The target table's path and the marginal table's path.
    '''

    targets_path = directory / TARGETS_FILE
    marginals_path = directory / MARGINALS_FILE
    targets.write_parquet(targets_path)
    marginals.write_parquet(marginals_path)
    return targets_path, marginals_path

# -------------------------------------------------------------------------------------------------
# The tree metric
# -------------------------------------------------------------------------------------------------

def tree_distance(codes: Sequence[str]) -> np.ndarray:
    '''
    D* between six-digit codes: twice the levels below their lowest common ancestor.

    This is for fixtures only. The panel's tree arm reads the bundle's D*.

    Args:
        codes: Six-digit codes.

    Returns:
        A ``(len(codes), len(codes))`` float64 matrix, 2 × (6 − ``lca_level``) off the diagonal and 0
        on it.
    '''

    distance = np.zeros((len(codes), len(codes)), dtype=np.float64)
    for row, code_a in enumerate(codes):
        for column, code_b in enumerate(codes):
            if row != column:
                distance[row, column] = 2 * (6 - lca_level(code_a, code_b))
    return distance
