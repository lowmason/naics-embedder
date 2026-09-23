# -------------------------------------------------------------------------------------------------
# Distributed Utilities for Global Batch Sampling
# -------------------------------------------------------------------------------------------------
'''
Distributed training utilities for global batch sampling across multiple GPUs.

This module provides functions for gathering embeddings and metadata across
distributed workers, enabling hard negative mining over the global batch.
'''

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist

from naics_embedder.supervision.candidates import CandidateEntityBatch

if TYPE_CHECKING:
    from naics_embedder.text_model.mixins.logging import LoggingMixin

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

def gather_negative_codes_global(
    local_negative_codes: List[List[str]], world_size: Optional[int] = None
) -> List[List[str]]:
    '''
    Gather negative codes from all GPUs for false negative masking.

    Args:
        local_negative_codes: Local negative codes (batch_size, k_negatives)
        world_size: Number of GPUs (auto-detected if None)

    Returns:
        Global negative codes list
    '''
    if not dist.is_initialized():
        return local_negative_codes

    if world_size is None:
        world_size = dist.get_world_size()

    if world_size == 1:
        return local_negative_codes

    # Gather negative codes from all ranks
    # Note: all_gather_object is used for Python objects like lists
    gathered_codes: List[List[str]] = [None] * world_size  # type: ignore
    dist.all_gather_object(gathered_codes, local_negative_codes)

    # Flatten the list of lists from all ranks
    global_negative_codes = []
    for codes_per_rank in gathered_codes:
        if codes_per_rank is not None:
            global_negative_codes.extend(codes_per_rank)

    return global_negative_codes

@dataclass
class GlobalNegativeContext:
    '''Container for global negative embedding context in distributed training.'''

    negatives_reshaped: torch.Tensor
    negatives_flat: torch.Tensor
    global_batch_size: int
    global_k_negatives: int

class DistributedMixin:
    '''
    Mixin providing distributed training utilities for global batch sampling.

    This mixin adds methods for gathering embeddings across workers and
    managing global negative pools for hard negative mining.

    Note: This mixin is designed to be used with NAICSContrastiveModel which
    inherits from LightningModule. The type annotations use string forward
    references to avoid circular imports.
    '''

    # Type hints for attributes provided by LightningModule or other mixins
    if TYPE_CHECKING:
        _log_global_batch_stats: 'LoggingMixin._log_global_batch_stats'

    def _should_use_global_batch(self, enable_hnm: bool, enable_router: bool) -> bool:
        '''Determine if global batch sampling should be used.'''
        if not (enable_hnm or enable_router):
            return False
        if not torch.distributed.is_initialized():
            return False
        return torch.distributed.get_world_size() > 1

    def _gather_global_negative_pool(
        self,
        negative_emb: torch.Tensor,
        batch_size: int,
        k_negatives: int,
        batch_idx: int,
    ) -> Optional[GlobalNegativeContext]:
        '''
        Gather negative embeddings from all workers into a global pool.

        Args:
            negative_emb: Local negative embeddings
            batch_size: Local batch size
            k_negatives: Number of negatives per sample
            batch_idx: Current batch index

        Returns:
            GlobalNegativeContext with reshaped global negatives, or None if not distributed
        '''
        if not torch.distributed.is_initialized():
            return None

        global_negative_emb = gather_embeddings_global(negative_emb)
        world_size = torch.distributed.get_world_size()
        global_batch_size = batch_size * world_size
        if global_batch_size == 0:
            return None

        global_k_negatives = global_negative_emb.shape[0] // global_batch_size
        reshaped = global_negative_emb.view(global_batch_size, global_k_negatives, -1)
        flat = reshaped.view(-1, global_negative_emb.shape[-1])

        if batch_idx == 0:
            self._log_global_batch_stats(
                global_negative_emb, batch_size, global_batch_size, global_k_negatives
            )

        return GlobalNegativeContext(
            negatives_reshaped=reshaped,
            negatives_flat=flat,
            global_batch_size=global_batch_size,
            global_k_negatives=global_k_negatives,
        )
