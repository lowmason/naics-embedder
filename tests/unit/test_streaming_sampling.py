'''
Unit tests for streaming dataset Phase 1 sampling with explicit exclusions.
'''

from typing import Any

import pytest
import torch

from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.selection import NegativeSelectionCoordinator
from naics_embedder.text_model.dataloader.streaming_dataset import (
    _compute_phase1_weights,
    _sample_negatives_phase1,
    _sample_negatives_sans_static,
    build_candidate_pool,
)
from naics_embedder.utils.config import SansStaticConfig


def _index(size: int, anchor_code_id: int, exclusion_code_ids: tuple[int, ...]) -> SupervisionIndex:
    directed = torch.zeros((size, size), dtype=torch.bool)
    for code_id in exclusion_code_ids:
        directed[anchor_code_id, code_id] = True
    return SupervisionIndex(
        code_to_id={str(code_id): code_id for code_id in range(size)},
        id_to_code=tuple(str(code_id) for code_id in range(size)),
        structural_distance=torch.full((size, size), 99.0),
        structural_relation_id=torch.full((size, size), 99, dtype=torch.int16),
        directed_exclusion=directed,
    )


@pytest.fixture
def pool_builder():
    def build(
        *,
        anchor_code_id: int,
        positive_code_id: int,
        raw_candidate_code_ids: list[int],
        exclusion_code_ids: tuple[int, ...],
        n_candidates: int,
        epoch: int,
    ) -> list[dict[str, Any]]:
        size = max(
            anchor_code_id,
            positive_code_id,
            *raw_candidate_code_ids,
            *exclusion_code_ids,
        ) + 4
        index = _index(size, anchor_code_id, exclusion_code_ids)
        raw = [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_structural_distance': 99.0,
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in raw_candidate_code_ids
        ]
        return build_candidate_pool(
            anchor_code_id=anchor_code_id,
            positive_code_id=positive_code_id,
            raw_candidates=raw,
            supervision_index=index,
            n_candidates=n_candidates,
            final_k=n_candidates,
            epoch=epoch,
            seed=7,
        )

    return build


def test_candidate_pool_contains_every_exclusion_and_unique_ordinary_codes(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 12, 13, 14],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=2,
    )

    assert {20, 21, 22} <= {item['negative_code_id'] for item in pool}
    assert len({item['negative_code_id'] for item in pool}) == len(pool)


def test_candidate_pool_backfills_to_final_selection_capacity(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12],
        exclusion_code_ids=(),
        n_candidates=3,
        epoch=0,
    )

    assert len({item['negative_code_id'] for item in pool}) >= 3


# -------------------------------------------------------------------------------------------------
# Candidate-pool capacity under the one-slot exclusion quota
# -------------------------------------------------------------------------------------------------

def _selection_capacity(pool: list[dict[str, Any]]) -> int:
    exclusions = sum(item['negative_is_explicit_exclusion'] for item in pool)
    ordinary = len(pool) - exclusions
    return ordinary + min(exclusions, 1)


def test_pool_supports_k_selections_when_several_exclusions_exist(pool_builder):
    # Three exclusions and K = 4: selection takes exactly one exclusion, so the pool needs at least
    # three ordinary codes; sizing by max(n, K, E) alone would leave only one.
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 13],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=0,
    )

    assert _selection_capacity(pool) >= 4


def test_pool_feeds_the_coordinator_for_k_selections(pool_builder):
    from naics_embedder.supervision.candidates import NegativeCandidateBatch

    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 13, 14],
        exclusion_code_ids=(20, 21),
        n_candidates=4,
        epoch=1,
    )
    count = len(pool)
    explicit = torch.tensor([[item['negative_is_explicit_exclusion'] for item in pool]])
    batch = NegativeCandidateBatch(
        candidate_uid=torch.stack(
            [torch.zeros(count, dtype=torch.long)] * 2 + [torch.arange(count)], dim=-1
        ).unsqueeze(0),
        code_id=torch.tensor([[item['negative_code_id'] for item in pool]]),
        embedding=torch.zeros((1, count, 2)),
        structural_distance=torch.full((1, count), 99.0),
        structural_relation_id=torch.full((1, count), 99, dtype=torch.int16),
        anchor_excludes_candidate=explicit,
        candidate_excludes_anchor=torch.zeros_like(explicit),
        is_explicit_exclusion=explicit,
        semantic_target_id=torch.zeros((1, count), dtype=torch.int8),
        semantic_source_id=torch.zeros((1, count), dtype=torch.int8),
        sampling_role_id=torch.full((1, count), 2, dtype=torch.int8),
        sampling_provenance_id=torch.full((1, count), 1, dtype=torch.int8),
        relation_margin=torch.zeros((1, count)),
        distance_margin=torch.zeros((1, count)),
        router_gate_probs=None,
        valid_mask=torch.ones((1, count), dtype=torch.bool),
        runtime_fields={},
    )

    selection = NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([10]),
        positive_code_ids=torch.tensor([11]),
        k=4,
        epoch=1,
        global_seed=7,
        proposals=(),
    )

    assert batch.select(selection).is_explicit_exclusion.sum().item() == 1


def test_pool_never_contains_anchor_or_positive(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[10, 11, 12],
        exclusion_code_ids=(11, 20),
        n_candidates=3,
        epoch=0,
    )

    codes = {item['negative_code_id'] for item in pool}
    assert not codes & {10, 11}
    assert 20 in codes


def test_pool_marks_exclusions_and_backfill_provenance(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12],
        exclusion_code_ids=(20,),
        n_candidates=3,
        epoch=0,
    )
    by_code = {item['negative_code_id']: item for item in pool}

    assert by_code[20]['negative_is_explicit_exclusion'] is True
    assert by_code[12]['negative_is_explicit_exclusion'] is False
    assert by_code[12]['sampling_provenance_id'] == 2
    assert all(
        item['sampling_provenance_id'] == 5
        for code, item in by_code.items()
        if code not in (12,)
    )


def test_pool_is_reproducible_for_one_epoch(pool_builder):
    kwargs = {
        'anchor_code_id': 10,
        'positive_code_id': 11,
        'raw_candidate_code_ids': list(range(12, 40)),
        'exclusion_code_ids': (),
        'n_candidates': 5,
    }

    first = pool_builder(**kwargs, epoch=3)
    second = pool_builder(**kwargs, epoch=3)

    assert [item['negative_code_id'] for item in first] == [
        item['negative_code_id'] for item in second
    ]


def test_pool_fails_when_the_universe_cannot_supply_k(pool_builder):
    with pytest.raises(ValueError, match='anchor code ID 10.*requires 30'):
        pool_builder(
            anchor_code_id=10,
            positive_code_id=11,
            raw_candidate_code_ids=[12],
            exclusion_code_ids=(),
            n_candidates=30,
            epoch=0,
        )


# -------------------------------------------------------------------------------------------------
# Structural eligibility on a production-shaped hierarchy
#
# For anchor '311111' (4) with its grandparent '3111' (2; grandchild relation, distance 1.5) as the
# positive, only the parent '31111' (3; child relation, distance 0.5) is structurally closer. Every
# other non-forbidden code is farther: ancestors '31'/'311' (0, 1), the collateral codes 5-10, the
# exclusion '321111' (11), and the cross-sector '44' family (12-16).
# -------------------------------------------------------------------------------------------------

HIERARCHY_ELIGIBLE_ORDINARY = {0, 1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16}


@pytest.fixture
def hierarchy_index(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    return SupervisionIndex.from_bundle(load_validated_bundle(manifest))


def _raw(index, code_ids):
    return [
        {
            'negative_code_id': code_id,
            'negative_code': index.id_to_code[code_id],
            'negative_structural_distance': float(index.structural_distance[4, code_id]),
            'sampling_role_id': 2,
            'sampling_provenance_id': 2,
        }
        for code_id in code_ids
    ]


def test_pool_rejects_a_raw_candidate_structurally_closer_than_the_positive(hierarchy_index):
    with pytest.raises(ValueError, match='raw candidate code ID 3 .* not structurally farther'):
        build_candidate_pool(
            anchor_code_id=4,
            positive_code_id=2,
            raw_candidates=_raw(hierarchy_index, [12, 3, 13]),
            supervision_index=hierarchy_index,
            n_candidates=4,
            final_k=3,
            epoch=0,
            seed=0,
        )


def test_pool_backfills_only_structurally_eligible_codes(hierarchy_index):
    pool = build_candidate_pool(
        anchor_code_id=4,
        positive_code_id=2,
        raw_candidates=_raw(hierarchy_index, [12]),
        supervision_index=hierarchy_index,
        n_candidates=17,
        final_k=5,
        epoch=0,
        seed=0,
    )

    codes = [candidate['negative_code_id'] for candidate in pool]
    ordinary = {
        candidate['negative_code_id']
        for candidate in pool
        if not candidate['negative_is_explicit_exclusion']
    }
    # The universe is exhausted, yet the parent is never backfilled.
    assert 3 not in codes
    assert ordinary == HIERARCHY_ELIGIBLE_ORDINARY
    assert codes[0] == 11


def test_pool_capacity_counts_only_structurally_eligible_codes(hierarchy_index):
    with pytest.raises(ValueError, match='requires 15 .* structurally farther'):
        build_candidate_pool(
            anchor_code_id=4,
            positive_code_id=2,
            raw_candidates=[],
            supervision_index=hierarchy_index,
            n_candidates=15,
            final_k=15,
            epoch=0,
            seed=0,
        )


def test_phase1_weights_ignore_exclusions_without_a_weight():
    candidates = [
        {'negative_code': '222222', 'negative_idx': 1},
        {'negative_code': '333333', 'negative_idx': 2},
    ]
    weights, excluded = _compute_phase1_weights(
        anchor_code='111111',
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup={('111111', '222222'): 4.0, ('111111', '333333'): 4.0},
        excluded_map={'111111': {'222222'}},
        code_to_idx={},
        alpha=1.0,
        exclusion_weight=None,
    )

    assert weights.tolist() == [0.25, 0.25]
    assert not excluded.any()

def test_excluded_negatives_get_high_weight():
    '''Excluded negatives should be prioritized and flagged.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 4
        },
        {
            'negative_code': '333',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6
        },
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 6.0,
    }
    excluded_map = {anchor: {'333'}}
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    weights, mask = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=1.5,
        exclusion_weight=100.0,
    )

    assert mask.tolist() == [False, True]
    assert weights[1] > weights[0]

def test_sample_marks_explicit_exclusions(monkeypatch):
    '''Sampled negatives should carry explicit_exclusion flag.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 4
        },
        {
            'negative_code': '333',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6
        },
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 6.0,
    }
    excluded_map = {anchor: {'333'}}
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    # Fix RNG for deterministic sampling
    monkeypatch.setenv('PYTHONHASHSEED', '0')

    sampled = _sample_negatives_phase1(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=1.5,
        exclusion_weight=100.0,
        seed=0,
    )

    assert len(sampled) == 2
    # Ensure the excluded negative is present and flagged
    excluded_flags = [neg['explicit_exclusion'] for neg in sampled]
    assert any(excluded_flags)

def test_sans_static_sampling_prefers_near_bucket():
    '''SANS static sampling should bias probabilities toward near negatives.'''

    anchor = '111'
    candidates = [
        {
            'negative_code': '1111',
            'negative_idx': 0,
            'relation_margin': 0,
            'distance_margin': 1,
        },
        {
            'negative_code': '2111',
            'negative_idx': 1,
            'relation_margin': 0,
            'distance_margin': 6,
        },
        {
            'negative_code': '3111',
            'negative_idx': 2,
            'relation_margin': 0,
            'distance_margin': 7,
        },
    ]
    distance_lookup = {
        (anchor, '1111'): 1.0,
        (anchor, '2111'): 6.0,
        (anchor, '3111'): 7.0,
    }

    sans_cfg = SansStaticConfig(
        near_distance_threshold=4.0,
        near_bucket_weight=0.8,
        far_bucket_weight=0.2,
    )

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=0,
    )

    assert len(sampled) == 2
    assert metadata['candidates_near'] == 1
    assert metadata['candidates_far'] == 2
    assert metadata['effective_near_weight'] > metadata['effective_far_weight']
    assert metadata['sampled_near'] + metadata['sampled_far'] == len(sampled)
