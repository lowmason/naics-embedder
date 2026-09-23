'''
Unit tests for router-guided negative mining utilities.
'''

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from naics_embedder.supervision.candidates import CandidateEntityBatch
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.supervision.selection import NegativeSelectionCoordinator
from naics_embedder.text_model.hard_negative_mining import (
    LorentzianHardNegativeMiner,
    RouterGuidedNegativeMiner,
)
from naics_embedder.text_model.mixins.curriculum import (
    CurriculumMixin,
    _proposal_from_local_uids,
)
from naics_embedder.text_model.mixins.distributed import DistributedMixin

def test_geometric_miner_returns_source_indices_not_embeddings(candidate_batch):
    anchor = candidate_batch.embedding[:, 0]
    proposal = LorentzianHardNegativeMiner().propose(anchor, candidate_batch, k=2)

    assert proposal.source_indices.shape == (1, 2)
    assert proposal.reason == SelectionReason.GEOMETRIC
    assert not hasattr(proposal, 'embedding')

def test_router_miner_indices_recover_matching_gate_rows(candidate_batch):
    anchor_gate = torch.tensor([[0.9, 0.1]])
    proposal = RouterGuidedNegativeMiner().propose(
        anchor_gate_probs=anchor_gate,
        candidates=candidate_batch,
        k=2,
    )
    selected_gates = candidate_batch.router_gate_probs.gather(
        1,
        proposal.source_indices.unsqueeze(-1).expand(-1, -1, 2),
    )

    assert selected_gates.shape == (1, 2, 2)
    assert proposal.reason == SelectionReason.ROUTER

def test_gathered_entity_is_rejoined_for_each_local_anchor(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)
    gathered = CandidateEntityBatch(
        candidate_uid=torch.tensor([[[1, 0, 0]]]),
        code_id=torch.tensor([[2]]),
        embedding=torch.tensor([[[0.25, 0.75]]]),
        router_gate_probs=torch.tensor([[[0.4, 0.6]]]),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    active_code_ids = gathered.code_id.expand(2, -1)
    active_valid = gathered.valid_mask.expand(2, -1)

    joined = index.join(
        anchor_code_ids=torch.tensor([0, 1]),
        candidate_code_ids=active_code_ids,
        valid_mask=active_valid,
    )

    assert joined.structural_distance.tolist() == [[2.0], [3.0]]
    assert joined.is_explicit_exclusion.tolist() == [[True], [False]]

# -------------------------------------------------------------------------------------------------
# Proposal masking and ranking
# -------------------------------------------------------------------------------------------------

def _lorentz(spatial: list[float]) -> torch.Tensor:
    point = torch.tensor(spatial, dtype=torch.float32)
    return torch.cat([torch.sqrt(1.0 + point.square().sum()).unsqueeze(0), point])

def test_geometric_proposals_rank_the_closest_eligible_candidates_first(
    candidate_batch_with_exclusions,
):
    batch = candidate_batch_with_exclusions
    positions = [0.0, 0.1, 0.2, 0.3, 0.9, 0.5]
    embedding = torch.stack([_lorentz([value, 0.0]) for value in positions]).unsqueeze(0)
    batch = replace(
        batch,
        embedding=embedding,
        valid_mask=torch.tensor([[True, True, True, True, True, False]]),
    )

    proposal = LorentzianHardNegativeMiner().propose(_lorentz([0.0, 0.0]).unsqueeze(0), batch, k=6)

    finite = torch.isfinite(proposal.scores[0])
    # Exclusions (slots 0-2) and the invalid slot 5 are never eligible.
    assert proposal.source_indices[0][finite].tolist() == [3, 4]

def test_router_proposals_never_include_exclusions_or_invalid_rows(
    candidate_batch_with_exclusions,
):
    batch = replace(
        candidate_batch_with_exclusions,
        valid_mask=torch.tensor([[True, True, True, True, False, True]]),
    )

    proposal = RouterGuidedNegativeMiner().propose(
        anchor_gate_probs=torch.tensor([[0.2, 0.8]]), candidates=batch, k=6
    )

    finite = torch.isfinite(proposal.scores[0])
    assert set(proposal.source_indices[0][finite].tolist()) == {3, 5}

# -------------------------------------------------------------------------------------------------
# Canonical selection boundary (curriculum mixin)
# -------------------------------------------------------------------------------------------------

def test_local_difficulty_proposals_translate_by_occurrence_uid(candidate_batch):
    # The active pool is the local pool in a different order: UIDs, not positions, carry identity.
    order = torch.tensor([2, 0, 1])
    active = replace(
        candidate_batch,
        **{
            name: getattr(candidate_batch, name)[:, order]
            for name in (
                'candidate_uid',
                'code_id',
                'embedding',
                'structural_distance',
                'structural_relation_id',
                'anchor_excludes_candidate',
                'candidate_excludes_anchor',
                'is_explicit_exclusion',
                'semantic_target_id',
                'semantic_source_id',
                'sampling_role_id',
                'sampling_provenance_id',
                'relation_margin',
                'distance_margin',
                'router_gate_probs',
                'valid_mask',
            )
        },
        runtime_fields={},
    )

    proposal = _proposal_from_local_uids(
        local_candidate_uid=candidate_batch.candidate_uid,
        active_candidates=active,
        local_source_indices=torch.tensor([[0, 2, -1]]),
    )

    assert proposal.source_indices.tolist() == [[1, 0, -1]]
    assert proposal.reason == SelectionReason.DIFFICULTY
    assert proposal.scores[0, 0] > proposal.scores[0, 1]
    assert proposal.scores[0, 2] == float('-inf')

class _SelectionHost(DistributedMixin, CurriculumMixin):

    def __init__(self, index: SupervisionIndex, flags: dict):
        self.supervision_index = index
        self.current_curriculum_flags = flags
        self.current_schedule_scalars = {}
        self.current_epoch = 0
        self.hparams = SimpleNamespace(selection_seed=7)
        self.hard_negative_miner = LorentzianHardNegativeMiner()
        self.router_guided_miner = RouterGuidedNegativeMiner()
        self.selection_coordinator = NegativeSelectionCoordinator()
        self.health = []

    def _log_selection_health(self, candidates, selected, batch_size, *, entity_valid_mask):
        self.health.append((candidates, selected, entity_valid_mask))

def _host_batch(pools: list[list[int]], anchor: int = 0, positive: int = 1) -> dict:
    width = max(len(pool) for pool in pools)
    return {
        'batch_size': len(pools),
        'k_candidates': width,
        'selection_k': 2,
        'anchor_code_id': torch.tensor([anchor] * len(pools)),
        'positive_code_id': torch.tensor([positive] * len(pools)),
        'positive_structural_distance': torch.tensor([0.5] * len(pools)),
        'positive_structural_relation_id': torch.tensor([1] * len(pools), dtype=torch.int16),
        'candidate_code_id': torch.tensor([pool + [-1] * (width - len(pool)) for pool in pools]),
        'candidate_valid_mask': torch.tensor(
            [[True] * len(pool) + [False] * (width - len(pool)) for pool in pools]
        ),
        'candidate_source_slot': torch.tensor(
            [list(range(len(pool))) + [-1] * (width - len(pool)) for pool in pools]
        ),
        'candidate_sampling_role_id': torch.full((len(pools), width), 2, dtype=torch.int8),
        'candidate_sampling_provenance_id': torch.full((len(pools), width), 1, dtype=torch.int8),
        'difficulty_proposal_indices': torch.tensor(
            [list(range(len(pool))) + [-1] * (width - len(pool)) for pool in pools]
        ),
    }

def _candidate_output(batch: dict) -> dict:
    code_ids = batch['candidate_code_id'].clamp_min(0).to(torch.float32).reshape(-1)
    spatial = torch.stack([code_ids / 10.0, torch.zeros_like(code_ids)], dim=1)
    embedding = torch.cat([torch.sqrt(1.0 + spatial.square().sum(1, keepdim=True)), spatial], 1)
    gate = torch.stack([code_ids / 10.0, 1.0 - code_ids / 10.0], dim=1)
    return {'embedding': embedding, 'gate_probs': gate}

def _local_uid(batch: dict) -> torch.Tensor:
    slots = batch['candidate_source_slot']
    rows = torch.arange(slots.shape[0]).unsqueeze(1).expand_as(slots)
    return torch.stack([torch.zeros_like(slots), rows, slots], dim=-1)

@pytest.mark.parametrize('mining', [False, True])
def test_select_negative_batch_keeps_every_field_on_one_identity(validated_bundle, mining):
    # Anchor 0 ('111111') excludes code 2; pool [2, 3, 4] with padding in the second row.
    index = SupervisionIndex.from_bundle(validated_bundle)
    host = _SelectionHost(
        index,
        {
            'enable_hard_negative_mining': mining,
            'enable_router_guided_sampling': mining
        },
    )
    batch = _host_batch([[2, 3, 4], [3, 4]])
    candidate_output = _candidate_output(batch)
    anchor_output = {
        'embedding': candidate_output['embedding'][:2].detach().clone(),
        'gate_probs': torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    }

    selected = host._select_negative_batch(
        batch=batch,
        anchor_output=anchor_output,
        candidate_output=candidate_output,
        candidate_uid=_local_uid(batch),
        batch_idx=0,
    )

    codes = selected.code_id.tolist()
    assert selected.is_explicit_exclusion.tolist()[0].count(True) == 1
    assert 2 in codes[0]
    assert all(code >= 0 for row in codes for code in row)
    assert torch.equal(selected.embedding[..., 1], selected.code_id.to(torch.float32) / 10.0)
    assert torch.equal(
        selected.router_gate_probs[..., 0],
        selected.code_id.to(torch.float32) / 10.0
    )
    expected_distance = index.structural_distance[0][selected.code_id]
    assert torch.equal(selected.structural_distance, expected_distance)
    # Positive: child at distance 0.5. Code 2 is the anchor's sibling exclusion (relation 2,
    # distance 2.0 -> raw deltas); codes 3 and 4 are cross-sector (fixed 15 / 10 margins).
    expected_margins = {2: (1.0, 1.5), 3: (15.0, 10.0), 4: (15.0, 10.0)}
    for row in range(selected.code_id.shape[0]):
        for slot, code in enumerate(selected.code_id[row].tolist()):
            assert selected.relation_margin[row, slot].item() == expected_margins[code][0]
            assert selected.distance_margin[row, slot].item() == expected_margins[code][1]
    assert len(host.health) == 1

# Production-shaped hierarchy (tests/fixtures/supervision.py HIERARCHY_CODES): code IDs follow
# lexicographic order. '311111' (4) excludes '321111' (11); '3111' (2) is its grandparent and
# '31111' (3) its parent; the '44' family (12-16) is cross-sector for it.
HIERARCHY_ANCHOR = 4
HIERARCHY_GRANDPARENT = 2
HIERARCHY_PARENT = 3
HIERARCHY_EXCLUSION = 11
CROSS_SECTOR_CODES = [12, 13, 14, 15, 16]

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

def _hierarchy_batch(index, pool, positive, selection_k):
    batch = _host_batch([pool], anchor=HIERARCHY_ANCHOR, positive=positive)
    batch['selection_k'] = selection_k
    batch['positive_structural_distance'] = torch.tensor(
        [float(index.structural_distance[HIERARCHY_ANCHOR, positive])]
    )
    batch['positive_structural_relation_id'] = index.structural_relation_id[HIERARCHY_ANCHOR,
                                                                            positive].reshape(1)
    return batch

def _code_embedding(code_id: int) -> torch.Tensor:
    '''The embedding ``_candidate_output`` gives a candidate with this code ID.'''
    return _candidate_output({'candidate_code_id': torch.tensor([[code_id]])})['embedding'][0]

def _hierarchy_candidate_output(batch: dict) -> dict:
    # Hierarchy code IDs reach 16, so gate probabilities scale by 1/20 to stay in [0, 1].
    output = _candidate_output(batch)
    code_ids = batch['candidate_code_id'].clamp_min(0).to(torch.float32).reshape(-1)
    output['gate_probs'] = torch.stack([code_ids / 20.0, 1.0 - code_ids / 20.0], dim=1)
    return output

def _select_with_flags(index, flags, pool, positive, selection_k, anchor_embedding, mix=None):
    host = _SelectionHost(index, flags)
    if mix is not None:
        host.current_schedule_scalars = {'router_mix_ratio': mix}
    batch = _hierarchy_batch(index, pool, positive, selection_k)
    candidate_output = _hierarchy_candidate_output(batch)
    selected = host._select_negative_batch(
        batch=batch,
        anchor_output={
            'embedding': anchor_embedding.unsqueeze(0),
            'gate_probs': torch.tensor([[0.5, 0.5]]),
        },
        candidate_output=candidate_output,
        candidate_uid=_local_uid(batch),
        batch_idx=0,
    )
    return host, selected

def _reasons(selected) -> list:
    return [SelectionReason(reason) for reason in selected.selection_reasons[0].tolist()]

def test_miners_choose_negatives_before_the_difficulty_proposal(hierarchy_index):
    pool = [HIERARCHY_EXCLUSION] + CROSS_SECTOR_CODES
    anchor_embedding = _code_embedding(13)
    mining = {'enable_hard_negative_mining': True, 'enable_router_guided_sampling': True}

    _, mined = _select_with_flags(
        hierarchy_index, mining, pool, HIERARCHY_GRANDPARENT, 4, anchor_embedding
    )
    _, unmined = _select_with_flags(
        hierarchy_index, {}, pool, HIERARCHY_GRANDPARENT, 4, anchor_embedding
    )

    # The difficulty proposal covers the whole pool, yet mining decides once it is enabled.
    assert _reasons(mined) == [
        SelectionReason.EXCLUSION_QUOTA,
        SelectionReason.GEOMETRIC,
        SelectionReason.GEOMETRIC,
        SelectionReason.ROUTER,
    ]
    assert _reasons(unmined) == [SelectionReason.EXCLUSION_QUOTA] + [SelectionReason.DIFFICULTY] * 3
    # Geometric slots go to the codes nearest the anchor embedding (code 13's): on the hyperboloid
    # d(x, y) = |asinh(x) - asinh(y)| here, so 1.4 (code 14) is nearer 1.3 than 1.2 (code 12).
    assert set(mined.code_id[0, 1:3].tolist()) == {13, 14}

@pytest.mark.parametrize(
    ('mix', 'expected'),
    [
        (0.0, [SelectionReason.GEOMETRIC] * 3),
        (1.0, [SelectionReason.ROUTER] * 3),
    ],
)
def test_router_mix_ratio_splits_mined_slots(hierarchy_index, mix, expected):
    pool = [HIERARCHY_EXCLUSION] + CROSS_SECTOR_CODES
    anchor_embedding = _code_embedding(13)
    mining = {'enable_hard_negative_mining': True, 'enable_router_guided_sampling': True}

    _, selected = _select_with_flags(
        hierarchy_index, mining, pool, HIERARCHY_GRANDPARENT, 4, anchor_embedding, mix=mix
    )

    assert _reasons(selected) == [SelectionReason.EXCLUSION_QUOTA] + expected

def test_repeated_codes_cannot_crowd_distinct_codes_out_of_mining(hierarchy_index):
    # A global pool repeats codes across rows and ranks. Three copies of code 13 sit at the anchor
    # embedding; the miner must still fill every mined slot with a distinct code instead of
    # leaving slots to the difficulty proposal.
    pool = [HIERARCHY_EXCLUSION, 13, 13, 13, 12, 14, 15]

    _, selected = _select_with_flags(
        hierarchy_index,
        {'enable_hard_negative_mining': True},
        pool,
        HIERARCHY_GRANDPARENT,
        4,
        _code_embedding(13),
    )

    assert _reasons(selected) == [SelectionReason.EXCLUSION_QUOTA] + [SelectionReason.GEOMETRIC] * 3
    assert selected.code_id[0].tolist() == [HIERARCHY_EXCLUSION, 13, 14, 12]
    # The duplicate code resolves to its smallest-UID occurrence (slot 1).
    assert selected.candidate_uid[0, 1].tolist() == [0, 0, 1]

def test_structurally_closer_relative_is_never_selected_even_when_nearest(hierarchy_index):
    # Positive: the grandparent. The parent is structurally closer than the positive, sits first
    # in the difficulty proposal, and is geometrically nearest to the anchor embedding.
    pool = [HIERARCHY_PARENT, HIERARCHY_EXCLUSION] + CROSS_SECTOR_CODES
    anchor_embedding = _code_embedding(HIERARCHY_PARENT)

    for flags in ({}, {'enable_hard_negative_mining': True}):
        host, selected = _select_with_flags(
            hierarchy_index, flags, pool, HIERARCHY_GRANDPARENT, 4, anchor_embedding
        )
        candidates, _, entity_valid = host.health[0]

        assert HIERARCHY_PARENT not in selected.code_id[0].tolist()
        assert HIERARCHY_EXCLUSION in selected.code_id[0].tolist()
        assert (entity_valid & ~candidates.valid_mask)[0].tolist() == [True] + [False] * 6

def test_select_negative_batch_rejects_router_mining_without_gate_probs(validated_bundle):
    host = _SelectionHost(
        SupervisionIndex.from_bundle(validated_bundle),
        {'enable_router_guided_sampling': True},
    )
    batch = _host_batch([[2, 3, 4]])
    candidate_output = _candidate_output(batch)

    with pytest.raises(ValueError, match='gate probabilities'):
        host._select_negative_batch(
            batch=batch,
            anchor_output={'embedding': candidate_output['embedding'][:1]},
            candidate_output=candidate_output,
            candidate_uid=_local_uid(batch),
            batch_idx=0,
        )

def test_router_miner_kl_prefers_matching_distribution(candidate_batch):
    '''KL-divergence metric should favor candidates with similar gate probs.'''

    miner = RouterGuidedNegativeMiner(metric='kl_divergence')

    anchor_gate_probs = torch.tensor([[0.7, 0.3]])
    negative_gate_probs = torch.tensor(
        [[
            [0.2, 0.8],  # Divergent distribution
            [0.7, 0.3],  # Matching distribution
            [0.5, 0.5],
        ]]
    )
    candidates = replace(candidate_batch, router_gate_probs=negative_gate_probs)

    proposal = miner.propose(anchor_gate_probs, candidates, k=1)

    # The matching distribution is proposed by source index, never as a gathered embedding
    assert proposal.source_indices.tolist() == [[1]]
    full_scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)
    assert full_scores[0, 1] > full_scores[0, 2] > full_scores[0, 0]

def test_router_miner_cosine_prefers_high_similarity(candidate_batch):
    '''Cosine metric should propose the highest-similarity gate distribution first.'''

    miner = RouterGuidedNegativeMiner(metric='cosine_similarity')

    anchor_gate_probs = torch.tensor([[0.2, 0.2, 0.6]])
    negative_gate_probs = torch.tensor(
        [
            [
                [0.2, 0.2, 0.6],  # Identical
                [0.6, 0.2, 0.2],  # Less similar
                [0.1, 0.8, 0.1],  # Least similar
            ]
        ]
    )
    candidates = replace(candidate_batch, router_gate_probs=negative_gate_probs)

    proposal = miner.propose(anchor_gate_probs, candidates, k=3)

    # Proposals are ranked by similarity
    assert proposal.source_indices.tolist() == [[0, 1, 2]]
    full_scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)
    assert full_scores[0, 0] >= full_scores[0, 1]
    assert full_scores[0, 1] >= full_scores[0, 2]

def test_confusion_scores_shape_consistency():
    '''Confusion scores should align with batch and candidate dimensions.'''

    miner = RouterGuidedNegativeMiner(metric='kl_divergence')

    batch_size = 4
    num_experts = 3
    num_negatives = 5
    anchor_gate_probs = torch.softmax(torch.randn(batch_size, num_experts), dim=1)
    negative_gate_probs = torch.softmax(torch.randn(batch_size, num_negatives, num_experts), dim=2)

    scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)

    assert scores.shape == (batch_size, num_negatives)
    # With KL-based metric, higher confusion corresponds to more similar distributions
    best_indices = scores.argmax(dim=1)
    assert torch.all(best_indices < num_negatives)
