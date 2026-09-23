# -------------------------------------------------------------------------------------------------
# Distributed Utilities for Global Batch Sampling
# -------------------------------------------------------------------------------------------------
'''
Distributed training utilities for global batch sampling across multiple GPUs.

This module provides functions for gathering candidate entities across distributed workers,
enabling checked negative selection over the global candidate pool.
'''

from typing import Optional

import torch
import torch.distributed as dist

from naics_embedder.supervision.candidates import CandidateEntityBatch


def _all_gather_fixed(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    '''Non-differentiable all_gather of equal-shape tensors, concatenated rank-major.'''
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)

def gather_candidate_entities(local: CandidateEntityBatch) -> CandidateEntityBatch:
    '''
    Gather candidate-intrinsic entities from every rank into one rank-major candidate axis.

    Only identity (occurrence UID, code ID), embedding, router outputs, and validity travel.
    Structure, relation, margins, semantics, and exclusion direction are pair-dependent and must
    be recomputed for each local anchor after the gather. Embeddings use a differentiable
    all_gather so gradients return to their originating rank; ranks with fewer entities are padded
    with invalid rows (code ID and UID -1, zero embedding) before gathering.

    Returns:
        A ``[1, total_candidates]`` entity batch (the local entities alone when not distributed).
    '''

    uid = local.candidate_uid.reshape(-1, 3)
    code = local.code_id.reshape(-1)
    embedding = local.embedding.reshape(-1, local.embedding.shape[-1])
    router = None
    if local.router_gate_probs is not None:
        router = local.router_gate_probs.reshape(-1, local.router_gate_probs.shape[-1])
    valid = local.valid_mask.reshape(-1)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather as differentiable_all_gather

        world_size = dist.get_world_size()
        router_width = -1 if router is None else router.shape[-1]
        header = torch.tensor(
            [[code.shape[0], embedding.shape[-1], router_width]],
            dtype=torch.long,
            device=embedding.device,
        )
        headers = _all_gather_fixed(header, world_size)
        if (headers[:, 1] != embedding.shape[-1]).any():
            raise ValueError('candidate embedding widths differ across ranks')
        if (headers[:, 2] != router_width).any():
            raise ValueError('candidate router outputs differ in presence or width across ranks')
        padding = int(headers[:, 0].max()) - code.shape[0]
        if padding:
            uid = torch.cat([uid, uid.new_full((padding, 3), -1)])
            code = torch.cat([code, code.new_full((padding, ), -1)])
            embedding = torch.cat([embedding, embedding.new_zeros((padding, embedding.shape[-1]))])
            if router is not None:
                router = torch.cat([router, router.new_zeros((padding, router.shape[-1]))])
            valid = torch.cat([valid, valid.new_zeros((padding, ))])
        embedding = torch.cat(differentiable_all_gather(embedding), dim=0)
        uid = _all_gather_fixed(uid, world_size)
        code = _all_gather_fixed(code, world_size)
        valid = _all_gather_fixed(valid.to(torch.uint8), world_size).bool()
        if router is not None:
            router = _all_gather_fixed(router.detach(), world_size)

    return CandidateEntityBatch(
        candidate_uid=uid.unsqueeze(0),
        code_id=code.unsqueeze(0),
        embedding=embedding.unsqueeze(0),
        router_gate_probs=None if router is None else router.unsqueeze(0),
        valid_mask=valid.unsqueeze(0),
    )

def gather_embeddings_global(
    local_embeddings: torch.Tensor, world_size: Optional[int] = None
) -> torch.Tensor:
    '''
    Gather embeddings from all GPUs using all_gather with gradient support.

    Issue #19: Global Batch Sampling - Collect embeddings from all ranks
    to enable hard negative mining across the global batch.

    This function uses torch.distributed.all_gather which preserves gradients,
    ensuring that gradients flow back through the gather operation during backprop.

    Args:
        local_embeddings: Local embeddings tensor (N_local, D) with requires_grad=True
        world_size: Number of GPUs (auto-detected if None)

    Returns:
        Global embeddings tensor (N_global, D) where N_global = N_local * world_size
        Gradients will flow back through this operation during backprop.
    '''
    if not dist.is_initialized():
        # Single GPU case: return local embeddings as-is
        return local_embeddings

    if world_size is None:
        world_size = dist.get_world_size()

    if world_size == 1:
        return local_embeddings

    # Use torch.distributed.all_gather for gradient support
    # This preserves gradients: if local_embeddings requires grad, the gathered
    # tensors will also have gradients flowing back during backprop.
    gathered_list = [torch.zeros_like(local_embeddings) for _ in range(world_size)]

    # all_gather collects tensors from all ranks into gathered_list
    # Each rank receives all tensors, so gathered_list[i] contains the tensor from rank i
    # Gradients flow back: during backprop, gradients are scattered back to each rank
    dist.all_gather(gathered_list, local_embeddings)

    # Concatenate all gathered embeddings along the batch dimension
    # This concatenation also preserves gradients
    global_embeddings = torch.cat(gathered_list, dim=0)

    return global_embeddings

class DistributedMixin:
    '''
    Mixin providing distributed training utilities for global batch sampling.

    Global selection gathers candidate entities with :func:`gather_candidate_entities` and rejoins
    pair-dependent supervision for each local anchor.
    '''

    def _should_use_global_batch(self, enable_hnm: bool, enable_router: bool) -> bool:
        '''Whether mining should select from the global candidate pool across ranks.'''
        if not (enable_hnm or enable_router):
            return False
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return False
        return torch.distributed.get_world_size() > 1
