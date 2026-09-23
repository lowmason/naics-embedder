from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.candidates import CandidateEntityBatch
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.selection import NegativeSelectionCoordinator
from naics_embedder.text_model.hard_negative_mining import (
    LorentzianHardNegativeMiner,
    RouterGuidedNegativeMiner,
)
from naics_embedder.text_model.mixins.curriculum import CurriculumMixin
from naics_embedder.text_model.mixins.distributed import (
    DistributedMixin,
    gather_candidate_entities,
)

def make_entity_batch(rank: int, code_id: int) -> CandidateEntityBatch:
    return CandidateEntityBatch(
        candidate_uid=torch.tensor([[[rank, 0, 0]]], dtype=torch.long),
        code_id=torch.tensor([[code_id]], dtype=torch.long),
        embedding=torch.tensor([[[float(rank), float(code_id)]]]),
        router_gate_probs=torch.tensor([[[0.25, 0.75]]]),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )

def _worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
    )
    try:
        local = make_entity_batch(rank=rank, code_id=rank + 1)
        gathered = gather_candidate_entities(local)
        queue.put(
            (
                rank,
                gathered.code_id.squeeze(0).cpu().tolist(),
                gathered.candidate_uid.squeeze(0).cpu().tolist(),
                gathered.valid_mask.squeeze(0).cpu().tolist(),
            )
        )
    finally:
        dist.destroy_process_group()

@pytest.mark.integration
def test_two_rank_gather_preserves_intrinsic_identity_only(tmp_path):
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(_worker, args=(2, init_file, queue), nprocs=2, join=True)
    results = sorted(queue.get() for _ in range(2))

    assert results[0][1] == results[1][1] == [1, 2]
    assert results[0][2][0][0] == 0
    assert results[0][2][1][0] == 1
    assert results[0][3] == [True, True]

def _uneven_worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
    )
    try:
        count = rank + 1
        local = CandidateEntityBatch(
            candidate_uid=torch.tensor(
                [[[rank, 0, slot] for slot in range(count)]], dtype=torch.long
            ),
            code_id=torch.tensor([[10 * rank + slot for slot in range(count)]]),
            embedding=torch.ones((1, count, 2), requires_grad=True) * (rank + 1),
            router_gate_probs=None,
            valid_mask=torch.ones((1, count), dtype=torch.bool),
        )
        gathered = gather_candidate_entities(local)
        gathered.embedding.sum().backward()
        queue.put(
            (
                rank,
                gathered.code_id.squeeze(0).tolist(),
                gathered.valid_mask.squeeze(0).tolist(),
                gathered.router_gate_probs is None,
                gathered.embedding.requires_grad,
            )
        )
    finally:
        dist.destroy_process_group()

@pytest.mark.integration
def test_uneven_entity_counts_are_padded_with_invalid_rows(tmp_path):
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(_uneven_worker, args=(2, init_file, queue), nprocs=2, join=True)
    results = sorted(queue.get() for _ in range(2))

    for _, code_ids, valid, no_router, requires_grad in results:
        assert valid == [True, False, True, True]
        assert [code for code, keep in zip(code_ids, valid) if keep] == [0, 10, 11]
        assert no_router
        assert requires_grad

# -------------------------------------------------------------------------------------------------
# End-to-end distributed selection: gather -> per-anchor join and eligibility -> one selection
# -------------------------------------------------------------------------------------------------

# Hierarchy fixture code IDs (tests/fixtures/supervision.py HIERARCHY_CODES, lexicographic):
# rank 0 anchors '311111' (4) with its grandparent '3111' (2) as positive and holds its exclusion
# '321111' (11) plus cross-sector '44'/'441' (12, 13); rank 1 anchors '441111' (16) with its
# grandparent '4411' (14) and holds its exclusion '311211' (7), rank 0's parent '31111' (3), and
# '311'/'31' (1, 0). Each anchor embedding sits nearest the other rank's candidates.
RANK_SETUPS = {
    0: {
        'anchor': 4,
        'positive': 2,
        'pool': [11, 12, 13],
        'anchor_value': 0.3
    },
    1: {
        'anchor': 16,
        'positive': 14,
        'pool': [7, 3, 1, 0],
        'anchor_value': 1.5
    },
}

def _lorentz_points(values: torch.Tensor) -> torch.Tensor:
    spatial = torch.stack([values, torch.zeros_like(values)], dim=1)
    return torch.cat([torch.sqrt(1.0 + spatial.square().sum(1, keepdim=True)), spatial], dim=1)

class _DistributedSelectionHost(DistributedMixin, CurriculumMixin):

    def __init__(self, index: SupervisionIndex):
        self.supervision_index = index
        self.current_curriculum_flags = {'enable_hard_negative_mining': True}
        self.current_schedule_scalars = {}
        self.current_epoch = 0
        self.hparams = SimpleNamespace(selection_seed=0)
        self.hard_negative_miner = LorentzianHardNegativeMiner()
        self.router_guided_miner = RouterGuidedNegativeMiner()
        self.selection_coordinator = NegativeSelectionCoordinator()
        self.ineligible = []

    def _log_selection_health(self, candidates, selected, batch_size, *, entity_valid_mask):
        self.ineligible.append(int((entity_valid_mask & ~candidates.valid_mask).sum()))

def _selection_worker(rank, world_size, init_file, manifest, queue):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
    )
    try:
        index = SupervisionIndex.from_bundle(load_validated_bundle(manifest))
        setup = RANK_SETUPS[rank]
        anchor, positive, pool = setup['anchor'], setup['positive'], setup['pool']
        slots = torch.arange(len(pool))
        batch = {
            'batch_size': 1,
            'k_candidates': len(pool),
            'selection_k': 3,
            'anchor_code_id': torch.tensor([anchor]),
            'positive_code_id': torch.tensor([positive]),
            'positive_structural_distance': index.structural_distance[anchor, positive].reshape(1),
            'positive_structural_relation_id': index.structural_relation_id[anchor,
                                                                            positive].reshape(1),
            'candidate_code_id': torch.tensor([pool]),
            'candidate_valid_mask': torch.ones((1, len(pool)), dtype=torch.bool),
            'candidate_source_slot': slots.unsqueeze(0),
            'candidate_sampling_role_id': torch.full((1, len(pool)), 2, dtype=torch.int8),
            'candidate_sampling_provenance_id': torch.full((1, len(pool)), 1, dtype=torch.int8),
            'difficulty_proposal_indices': slots.unsqueeze(0),
        }
        values = torch.tensor(pool, dtype=torch.float32) / 10.0
        candidate_output = {
            'embedding': _lorentz_points(values),
            'gate_probs': torch.stack([values / 2.0, 1.0 - values / 2.0], dim=1),
        }
        uid = torch.stack([torch.full_like(slots, rank),
                           torch.zeros_like(slots), slots], dim=-1).unsqueeze(0)
        host = _DistributedSelectionHost(index)
        selected = host._select_negative_batch(
            batch=batch,
            anchor_output={
                'embedding': _lorentz_points(torch.tensor([setup['anchor_value']])),
                'gate_probs': torch.tensor([[0.5, 0.5]]),
            },
            candidate_output=candidate_output,
            candidate_uid=uid,
            batch_idx=0,
        )
        aligned = torch.allclose(
            selected.embedding[0, :, 1], selected.code_id[0].to(torch.float32) / 10.0
        )
        queue.put(
            (
                rank,
                selected.code_id[0].tolist(),
                selected.candidate_uid[0, :, 0].tolist(),
                selected.is_explicit_exclusion[0].tolist(),
                aligned,
                host.ineligible[0],
            )
        )
    finally:
        dist.destroy_process_group()

@pytest.mark.integration
def test_two_rank_selection_mines_the_global_pool_under_local_eligibility(
    tmp_path, hierarchy_descriptions_parquet
):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(_selection_worker, args=(2, init_file, str(manifest), queue), nprocs=2, join=True)
    rank0, rank1 = sorted(queue.get() for _ in range(2))

    # Rank 0: its exclusion by quota, then the two geometrically nearest eligible codes, both
    # remote. Its parent (3) is nearest of all but structurally closer than the grandparent
    # positive, so it is ineligible for this anchor and never selected.
    assert rank0[1] == [11, 1, 0]
    assert rank0[2] == [0, 1, 1]
    assert rank0[3] == [True, False, False]
    assert rank0[5] >= 1
    # Rank 1: its exclusion by quota, then the nearest eligible codes, both from rank 0.
    assert rank1[1] == [7, 13, 12]
    assert rank1[2] == [1, 0, 0]
    assert rank1[3] == [True, False, False]
    # Every selected field stays on one occurrence identity.
    assert rank0[4] and rank1[4]
