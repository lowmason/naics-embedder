import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from naics_embedder.supervision.candidates import CandidateEntityBatch
from naics_embedder.text_model.mixins.distributed import gather_candidate_entities


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
