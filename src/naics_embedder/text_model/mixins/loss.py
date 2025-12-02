# -------------------------------------------------------------------------------------------------
# Loss Computation Mixin
# -------------------------------------------------------------------------------------------------
'''
Loss computation mixin for NAICSContrastiveModel.

Provides methods for computing various loss components:
- Contrastive loss
- Hierarchy preservation loss
- LambdaRank loss for ranking optimization
- Radius regularization loss
- Load balancing loss for MoE
'''

import logging
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

class LossMixin:
    '''
    Mixin providing loss computation methods for the NAICS model.

    This mixin expects the following attributes on the class:
    - device: torch.device
    - hparams: hyperparameters with weight configurations
    - hierarchy_loss_fn: Optional hierarchy preservation loss function
    - lambdarank_loss_fn: Optional LambdaRank loss function
    - load_balancing_coef: float coefficient for load balancing loss
    - trainer: PyTorch Lightning trainer (for distributed logging)
    - logger: PyTorch Lightning logger (for histograms)
    - global_step: int training step counter
    '''

    def _compute_hierarchy_loss(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
    ) -> torch.Tensor:
        '''
        Compute hierarchy preservation loss based on tree distances.

        Args:
            anchor_emb: Anchor embeddings
            positive_emb: Positive embeddings
            batch: Training batch with anchor_code and optional positive_code
            batch_size: Batch size for logging

        Returns:
            Hierarchy loss tensor (scalar)
        '''
        if self.hierarchy_loss_fn is None or 'anchor_code' not in batch:
            return torch.tensor(0.0, device=self.device)

        try:
            all_codes = batch['anchor_code'].copy()
            if 'positive_code' in batch:
                all_codes.extend(batch['positive_code'])
            else:
                all_codes.extend(batch['anchor_code'])

            all_embeddings = torch.cat([anchor_emb, positive_emb])
            from naics_embedder.text_model.hyperbolic import LorentzDistance

            curvature = getattr(self.hparams, 'curvature', 1.0)
            lorentz_dist = LorentzDistance(curvature=curvature)

            hierarchy_loss = self.hierarchy_loss_fn(
                all_embeddings, all_codes, lambda x, y: lorentz_dist(x, y)
            )

            self.log('train/hierarchy_loss', hierarchy_loss, batch_size=batch_size)
            return hierarchy_loss
        except Exception as exc:
            logger.warning(f'Failed to compute hierarchy loss: {exc}')
            return torch.tensor(0.0, device=self.device)

    def _compute_lambdarank_loss(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
        k_negatives: int,
    ) -> torch.Tensor:
        '''
        Compute LambdaRank loss for global ranking optimization.

        Args:
            anchor_emb: Anchor embeddings
            positive_emb: Positive embeddings
            negative_emb: Negative embeddings
            batch: Training batch with codes
            batch_size: Batch size
            k_negatives: Number of negatives per sample

        Returns:
            LambdaRank loss tensor (scalar)
        '''
        if self.lambdarank_loss_fn is None or 'anchor_code' not in batch or 'positive_code' not in batch:
            return torch.tensor(0.0, device=self.device)

        try:
            negative_codes = batch.get('negative_codes', [])
            if not negative_codes or len(negative_codes) != batch_size:
                logger.debug('Skipping LambdaRank: negative_codes not available in batch')
                return torch.tensor(0.0, device=self.device)

            from naics_embedder.text_model.hyperbolic import LorentzDistance

            curvature = getattr(self.hparams, 'curvature', 1.0)
            lorentz_dist = LorentzDistance(curvature=curvature)

            lambdarank_loss = self.lambdarank_loss_fn(
                anchor_emb,
                positive_emb,
                negative_emb,
                batch['anchor_code'],
                batch['positive_code'],
                negative_codes,
                lambda x, y: lorentz_dist(x, y),
                batch_size,
                k_negatives,
            )
            self.log('train/lambdarank_loss', lambdarank_loss, batch_size=batch_size)
            return lambdarank_loss
        except Exception as exc:
            logger.warning(f'Failed to compute LambdaRank loss: {exc}')
            import traceback

            logger.debug(traceback.format_exc())
            return torch.tensor(0.0, device=self.device)

    def _compute_radius_regularization(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        '''
        Compute radius regularization loss to prevent embedding explosion.

        Args:
            anchor_emb: Anchor embeddings
            positive_emb: Positive embeddings
            negative_emb: Negative embeddings
            batch_size: Batch size for logging

        Returns:
            Radius regularization loss tensor (scalar)
        '''
        radius_reg_weight = getattr(self.hparams, 'radius_reg_weight', 0.0)
        if radius_reg_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        all_embeddings = torch.cat([anchor_emb, positive_emb, negative_emb])
        x0 = all_embeddings[:, 0]
        curvature = getattr(self.hparams, 'curvature', 1.0)
        radius_squared = torch.clamp(x0**2 - 1.0 / curvature, min=0.0)
        radius = torch.sqrt(radius_squared + 1e-8)

        radius_threshold = 10.0
        excess_radius = torch.clamp(radius - radius_threshold, min=0.0)
        radius_reg_loss = radius_reg_weight * torch.mean(excess_radius**2)

        self.log('train/radius_reg_loss', radius_reg_loss, batch_size=batch_size)
        self.log('train/mean_radius', radius.mean(), batch_size=batch_size)
        self.log('train/max_radius', radius.max(), batch_size=batch_size)
        return radius_reg_loss

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

    def _combine_loss_terms(
        self,
        contrastive_loss: torch.Tensor,
        load_balancing_loss: torch.Tensor,
        hierarchy_loss: torch.Tensor,
        lambdarank_loss: torch.Tensor,
        radius_reg_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Combine individual loss components into the final optimization target.

        Args:
            contrastive_loss: Main contrastive loss
            load_balancing_loss: MoE load balancing loss
            hierarchy_loss: Hierarchy preservation loss
            lambdarank_loss: Ranking optimization loss
            radius_reg_loss: Radius regularization loss

        Returns:
            Tuple containing the total loss and the scaled load balancing term.
        '''
        scaled_load_balancing_loss = self.load_balancing_coef * load_balancing_loss
        total_loss = (
            contrastive_loss + scaled_load_balancing_loss + hierarchy_loss + lambdarank_loss +
            radius_reg_loss
        )
        return total_loss, scaled_load_balancing_loss

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
