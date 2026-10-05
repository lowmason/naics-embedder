# -------------------------------------------------------------------------------------------------
# Loss Computation Mixin
# -------------------------------------------------------------------------------------------------
'''
Loss mixin for NAICSContrastiveModel: the experts' load-balancing term, under ``moe`` only.

Req 11's three terms live in ``text_model/loss.py`` and ``compute_losses`` combines them. The one
extra term is the experts' load balancing, which only the MoE fusion has (spec 4.1, R11).
'''

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

class LossMixin:
    '''
    Mixin providing the MoE load-balancing term and its utilization logs.

    This mixin expects the following attributes on the class:
    - device: torch.device
    - trainer: PyTorch Lightning trainer (for distributed logging)
    - logger: PyTorch Lightning logger (for histograms)
    - global_step: int training step counter
    '''

    def _compute_load_balancing_loss(
        self,
        gate_probs_list: List[torch.Tensor],
        topk_indices_list: List[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        '''
        Compute load balancing loss for Mixture of Experts.

        This loss encourages uniform expert utilization to prevent expert collapse.

        Args:
            gate_probs_list: List of gate probability tensors from each forward pass
            topk_indices_list: List of top-k expert indices from each forward pass
            batch_size: Batch size for logging

        Returns:
            Load balancing loss tensor (scalar)
        '''
        if not gate_probs_list:
            return torch.tensor(0.0, device=self.device)

        gate_probs = torch.cat(gate_probs_list, dim=0)
        top_k_indices = torch.cat(topk_indices_list, dim=0)
        total_tokens = gate_probs.shape[0]
        num_experts = gate_probs.shape[1]

        prob_sum = gate_probs.sum(dim=0)
        expert_counts_micro = torch.zeros(num_experts, device=self.device)
        for i in range(num_experts):
            expert_counts_micro[i] = (top_k_indices == i).any(dim=1).sum()

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                global_prob_sum = prob_sum.clone()
                global_expert_counts = expert_counts_micro.clone()
                global_total_tokens = torch.tensor(
                    total_tokens, dtype=torch.float, device=self.device
                )
                torch.distributed.all_reduce(global_prob_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(
                    global_expert_counts, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(global_total_tokens, op=torch.distributed.ReduceOp.SUM)
                global_total_tokens_safe = torch.clamp(global_total_tokens, min=1.0)
                f = global_expert_counts / global_total_tokens_safe
                P = global_prob_sum / global_total_tokens_safe
                if self.trainer.is_global_zero:
                    logger.debug(f'Global load balancing: f={f.mean():.4f}, P={P.mean():.4f}')
            else:
                f = expert_counts_micro / total_tokens
                P = prob_sum / total_tokens
        else:
            f = expert_counts_micro / total_tokens
            P = prob_sum / total_tokens

        load_balancing_loss = num_experts * torch.sum(f * P)

        # Log expert utilization metrics (only on rank 0 in distributed)
        if (
            not torch.distributed.is_initialized() or not hasattr(self.trainer, 'is_global_zero')
            or self.trainer.is_global_zero
        ):
            self._log_expert_utilization(f, P, num_experts, batch_size)

        return load_balancing_loss

    def _log_expert_utilization(
        self,
        f: torch.Tensor,
        P: torch.Tensor,
        num_experts: int,
        batch_size: int,
    ) -> None:
        '''Log expert utilization and gating probability metrics.'''
        for i in range(num_experts):
            self.log(
                f'train/moe/expert_{i}_utilization',
                f[i].item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f'train/moe/expert_{i}_gating_prob',
                P[i].item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        experiment = getattr(self.logger, 'experiment', None) if self.logger is not None else None
        if experiment is not None and hasattr(experiment, 'add_histogram'):
            try:
                experiment.add_histogram(
                    'train/moe/expert_utilization_hist', f, global_step=self.global_step
                )
                experiment.add_histogram(
                    'train/moe/gating_prob_hist', P, global_step=self.global_step
                )
            except Exception as exc:
                logger.debug(f'Could not log histograms: {exc}')

        f_mean = f.mean().item()
        f_std = f.std().item()
        f_min = f.min().item()
        f_max = f.max().item()
        f_cv = (f_std / f_mean) if f_mean > 0 else 0.0

        self.log(
            'train/moe/utilization_mean',
            f_mean,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/utilization_std',
            f_std,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/utilization_cv',
            f_cv,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            'train/moe/utilization_min',
            f_min,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/utilization_max',
            f_max,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        P_mean = P.mean().item()
        P_std = P.std().item()
        P_min = P.min().item()
        P_max = P.max().item()
        P_cv = (P_std / P_mean) if P_mean > 0 else 0.0

        self.log(
            'train/moe/gating_prob_mean',
            P_mean,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/gating_prob_std',
            P_std,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/gating_prob_cv',
            P_cv,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/gating_prob_min',
            P_min,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/moe/gating_prob_max',
            P_max,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        ideal_utilization = 1.0 / num_experts
        utilization_imbalance = torch.abs(f - ideal_utilization).mean().item()
        self.log(
            'train/moe/utilization_imbalance',
            utilization_imbalance,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _collect_gate_outputs(self, outputs: List[Dict[str, torch.Tensor]]
                              ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        '''
        Extract gate probabilities and top-k indices from encoder outputs.

        Args:
            outputs: List of encoder output dictionaries

        Returns:
            Tuple of (gate_probs_list, topk_indices_list)
        '''
        gate_probs_list: List[torch.Tensor] = []
        topk_indices_list: List[torch.Tensor] = []
        for output in outputs:
            gate_probs = output.get('gate_probs')
            top_k_indices = output.get('top_k_indices')
            if gate_probs is not None and top_k_indices is not None:
                gate_probs_list.append(gate_probs)
                topk_indices_list.append(top_k_indices)
        return gate_probs_list, topk_indices_list
