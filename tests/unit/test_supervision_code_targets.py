'''
The code–code and radial terms' fixed inputs (spec 4.1(ii), (iii); section 6, "Code–code term").

D*, the unary partners and the levels are written by hand from the hierarchy's 17 codes, which the
reference bundle (``tests/fixtures/supervision.py``) is built on. D* is the tree path through a
virtual root: depth_i + depth_j - 2 depth_LCA, a sector at depth 1 (Req 7).
'''

import dataclasses
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.code_targets import PAIR_FACT_COLUMNS, CodeTargets
from naics_embedder.text_model.loss import code_code_loss
from tests.fixtures.supervision import HIERARCHY_CODES, build_reference_bundle

LEVELS = [2, 3, 4, 5, 6, 4, 5, 6, 3, 4, 5, 6, 2, 3, 4, 5, 6]

# D* from three codes to every code, in codebook order
D_STAR_ROWS = {
    '31': [0, 1, 2, 3, 4, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 6],
    '321': [1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 3, 3, 4, 5, 6, 7],
    '311111': [4, 3, 2, 1, 0, 4, 5, 6, 5, 6, 7, 8, 6, 7, 8, 9, 10],
}

# Each five-digit code of the hierarchy has one six-digit child
UNARY_PARTNERS = {
    '31111': '311111',
    '311111': '31111',
    '31121': '311211',
    '311211': '31121',
    '32111': '321111',
    '321111': '32111',
    '44111': '441111',
    '441111': '44111',
}

EXCLUSION_COLUMNS = ('code_i_excludes_code_j', 'code_j_excludes_code_i', 'is_explicit_exclusion')

# Exclusions on other pairs than the reference bundle's: '441111' sends sawmilling to '321111', and
# '321111' sends car retailing to '441111'
OTHER_EXCLUSION_ROWS = (
    (
        0,
        'cross_reference',
        '441111',
        'Sawmilling--are classified in Industry 321111.',
        'Sawmilling',
        ['321111'],
        [],
        False,
    ),
    (
        1,
        'cross_reference',
        '321111',
        'Retailing new cars--are classified in Industry 441111.',
        'Retailing new cars',
        ['441111'],
        [],
        False,
    ),
)

@pytest.fixture
def code_targets(reference_bundle) -> CodeTargets:
    return CodeTargets.from_bundle(reference_bundle)

def _exclusion_pairs(bundle):
    facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
    return set(facts.filter('is_explicit_exclusion').select('code_i', 'code_j').rows())

# -------------------------------------------------------------------------------------------------
# Codes, levels, D* and the unary partners
# -------------------------------------------------------------------------------------------------

def test_the_codes_follow_the_codebook_and_each_level_is_its_digit_count(code_targets):
    assert code_targets.codes == HIERARCHY_CODES
    assert code_targets.levels.dtype == np.int64
    assert code_targets.levels.tolist() == LEVELS

def test_d_star_is_the_tree_metric_between_every_two_codes(code_targets):
    d_star = code_targets.structural_distance

    assert d_star.dtype == np.float32
    assert d_star.shape == (17, 17)
    for code, row in D_STAR_ROWS.items():
        assert d_star[HIERARCHY_CODES.index(code)].tolist() == row
    assert np.array_equal(d_star, d_star.T)
    assert np.array_equal(d_star == 0, np.eye(17, dtype=bool))

def test_each_unary_partner_is_the_other_code_of_its_pair(code_targets):
    partners = code_targets.unary_partner

    assert partners.dtype == np.int64
    assert {
        code: HIERARCHY_CODES[partner]
        for code, partner in zip(HIERARCHY_CODES, partners.tolist()) if partner != -1
    } == UNARY_PARTNERS
    assert partners.tolist().count(-1) == 17 - len(UNARY_PARTNERS)

def test_the_arrays_are_writable_so_torch_can_share_them(code_targets):
    # torch.from_numpy warns on a read-only array, such as a zero-copy view of a polars column
    for array in (
        code_targets.levels,
        code_targets.structural_distance,
        code_targets.unary_partner,
    ):
        assert array.flags.writeable

# -------------------------------------------------------------------------------------------------
# J_a: every code but the anchor and its unary partner (Req 9)
# -------------------------------------------------------------------------------------------------

def test_j_a_is_every_code_but_the_anchor_and_its_unary_partner(code_targets):
    # '311111' and its partner '31111'; the sector '31', which has none; '44111' and '441111'; and
    # '311111' again
    keep = code_targets.keep(np.array([4, 0, 15, 4]))

    assert keep.dtype == bool
    assert keep.shape == (4, 17)
    assert [np.flatnonzero(~row).tolist() for row in keep] == [[3, 4], [0], [15, 16], [3, 4]]

@pytest.mark.parametrize(
    'ids, message',
    [
        (np.array([17]), 'anchor ids must be code ids in 0..16, not [17]'),
        (np.array([0, -1]), 'anchor ids must be code ids in 0..16, not [-1]'),
        (np.array([[0]]), 'anchor ids must be a one-dimensional integer array, not int64 (1, 1)'),
        (np.array([0.0]), 'anchor ids must be a one-dimensional integer array, not float64 (1,)'),
    ],
    ids=['past-the-codebook', 'negative', 'two-dimensional', 'not-integer'],
)
def test_anchor_ids_that_are_not_code_ids_are_refused(code_targets, ids, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        code_targets.keep(ids)

# -------------------------------------------------------------------------------------------------
# Equality, and no exclusion data (Req 8(c), Verification "Exclusions")
# -------------------------------------------------------------------------------------------------

def test_code_targets_compare_by_value(code_targets, reference_bundle):
    assert CodeTargets.from_bundle(reference_bundle) == code_targets
    for field, changed in (
        ('codes', code_targets.codes[::-1]),
        ('levels', code_targets.levels + 1),
        ('structural_distance', code_targets.structural_distance * 2),
        ('unary_partner', np.full(17, -1, dtype=np.int64)),
        ('levels', code_targets.levels.astype(np.int32)),
    ):
        assert dataclasses.replace(code_targets, **{field: changed}) != code_targets

def test_other_exclusions_leave_the_targets_and_the_code_code_loss_bit_identical(
    tmp_path, reference_bundle, code_targets
):
    other = load_validated_bundle(
        build_reference_bundle(tmp_path / 'other', redirection_rows=OTHER_EXCLUSION_ROWS)
    )
    excluded = _exclusion_pairs(reference_bundle)
    # The two bundles' exclusions share no pair
    assert excluded and _exclusion_pairs(other)
    assert not excluded & _exclusion_pairs(other)

    other_targets = CodeTargets.from_bundle(other)
    ids = np.arange(17)
    distances = 10.0 * torch.rand((17, 17), generator=torch.Generator().manual_seed(0))
    scale = torch.tensor(1.5)

    def loss(targets: CodeTargets, keep: np.ndarray) -> torch.Tensor:
        structural = torch.from_numpy(targets.structural_distance[ids])
        return code_code_loss(distances, scale, structural, torch.from_numpy(keep), 1.0)

    assert other_targets == code_targets
    assert torch.equal(
        loss(other_targets, other_targets.keep(ids)),
        loss(code_targets, code_targets.keep(ids)),
    )
    # Had J_a dropped the reference bundle's exclusion pairs, these distances would show it
    without_exclusions = code_targets.keep(ids)
    for code_i, code_j in excluded:
        i, j = HIERARCHY_CODES.index(code_i), HIERARCHY_CODES.index(code_j)
        without_exclusions[i, j] = without_exclusions[j, i] = False
    assert not torch.equal(
        loss(code_targets, without_exclusions),
        loss(code_targets, code_targets.keep(ids)),
    )

def test_the_targets_read_no_exclusion_column(reference_bundle, code_targets):
    # Past the loader, which would refuse the rewrite: a read of a dropped column would fail
    path = reference_bundle.artifact_path('pair_facts')
    pl.read_parquet(path).drop(EXCLUSION_COLUMNS).write_parquet(path)

    assert CodeTargets.from_bundle(reference_bundle) == code_targets

def test_the_targets_read_only_the_four_pair_fact_columns(reference_bundle, monkeypatch):
    read_parquet = pl.read_parquet
    reads = []

    def spy(source, *args, **kwargs):
        reads.append((Path(source).name, kwargs.get('columns')))
        return read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(pl, 'read_parquet', spy)
    CodeTargets.from_bundle(reference_bundle)

    pair_facts = reference_bundle.artifact_path('pair_facts').name
    assert [columns for name, columns in reads if name == pair_facts] == [list(PAIR_FACT_COLUMNS)]
