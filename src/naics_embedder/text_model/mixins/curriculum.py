# -------------------------------------------------------------------------------------------------
# Curriculum Learning Mixin
# -------------------------------------------------------------------------------------------------
'''
Curriculum learning mixin for NAICSContrastiveModel.

Provides methods for:
- Curriculum state management
- Hard negative mining (embedding-based and router-guided)
- False negative mask construction
- Pseudo-label clustering for false negative detection
'''

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from naics_embedder.text_model.false_negative_strategies import apply_false_negative_strategy
from naics_embedder.text_model.mixins.distributed import (
    GlobalNegativeContext,
    gather_embeddings_global,
)

logger = logging.getLogger(__name__)

class CurriculumMixin:
    '''
    Mixin providing curriculum learning functionality.

    This mixin expects the following attributes on the class:
    - curriculum_scheduler: Optional[CurriculumScheduler]
    - current_curriculum_flags: Dict[str, bool]
    - current_schedule_scalars: Dict[str, float]
    - previous_phase: Optional[int]
    - current_epoch: int
    - device: torch.device
    - code_to_pseudo_label: Dict[str, int]
    - hard_negative_miner: LorentzianHardNegativeMiner
    - router_guided_miner: RouterGuidedNegativeMiner
    - false_negative_config: FalseNegativeConfig
    - hparams: hyperparameters
    '''

    def _update_curriculum_state(self, batch_idx: int, batch_size: int) -> None:
        '''
        Update curriculum flags and scalars based on current epoch.

        Args:
            batch_idx: Current batch index
            batch_size: Batch size for logging
        '''
        if self.curriculum_scheduler is None:
            self.current_curriculum_flags = {}
            self.current_schedule_scalars = {}
            return

        self.current_curriculum_flags = self.curriculum_scheduler.get_curriculum_flags(
            self.current_epoch
        )
        self.current_schedule_scalars = self.curriculum_scheduler.get_schedule_scalars(
            self.current_epoch
        )

        self.curriculum_scheduler.log_phase_transition(self.current_epoch, self.previous_phase)
        self.previous_phase = self.curriculum_scheduler.get_phase(self.current_epoch)

        if batch_idx != 0 or not self.current_schedule_scalars:
            return

        anneal_progress = self.current_schedule_scalars.get('anneal_progress')
        if anneal_progress is not None:
            self.log(
                'train/curriculum/anneal_progress',
                anneal_progress,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        tree_alpha = self.current_schedule_scalars.get('tree_distance_alpha')
        if tree_alpha is not None:
            self.log(
                'train/curriculum/tree_distance_alpha',
                tree_alpha,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _build_false_negative_mask(self, batch: Dict[str, Any],
                                   batch_size: int) -> Optional[torch.Tensor]:
        '''
        Build a mask to identify false negatives using pseudo-labels.

        False negatives are negative samples that are semantically similar to the anchor.

        Args:
            batch: Training batch with anchor_code and negative_codes
            batch_size: Expected batch size

        Returns:
            Boolean mask tensor (batch_size, k_negatives) or None
        '''
        enable_clustering = self.current_curriculum_flags.get('enable_clustering', False)
        if not (
            enable_clustering and self.code_to_pseudo_label and 'negative_codes' in batch
            and 'anchor_code' in batch
        ):
            return None

        try:
            assert isinstance(batch['negative_codes'], list), 'negative_codes must be a list'
            assert len(batch['negative_codes']) == batch_size, (
                f"Expected {batch_size} groups, got {len(batch['negative_codes'])}"
            )
            assert all(isinstance(codes, list) for codes in batch['negative_codes']), (
                'Each entry must be a list of codes'
            )

            anchor_labels = torch.tensor(
                [self.code_to_pseudo_label.get(code, -1) for code in batch['anchor_code']],
                device=self.device,
            )

            neg_labels = torch.tensor(
                [
                    [self.code_to_pseudo_label.get(code, -2) for code in neg_codes_for_anchor]
                    for neg_codes_for_anchor in batch['negative_codes']
                ],
                device=self.device,
            )

            false_negative_mask = anchor_labels.unsqueeze(1) == neg_labels
            valid_anchor_mask = (anchor_labels > -1).unsqueeze(1)
            valid_neg_mask = neg_labels > -2

            return false_negative_mask & valid_anchor_mask & valid_neg_mask
        except Exception as exc:
            logger.warning(f'Failed to create false negative mask: {exc}')
            return None

    def _perform_hard_negative_mining(
        self,
        anchor_emb: torch.Tensor,
        negative_emb_reshaped: torch.Tensor,
        k_negatives: int,
        batch_size: int,
        batch_idx: int,
        global_context: Optional[GlobalNegativeContext],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Perform hard negative mining to select the most informative negatives.

        Args:
            anchor_emb: Anchor embeddings
            negative_emb_reshaped: Negative embeddings (batch_size, k_negatives, embed_dim)
            k_negatives: Number of negatives to select
            batch_size: Batch size
            batch_idx: Current batch index
            global_context: Optional global negative context for distributed training

        Returns:
            Tuple of (hard_negatives, hard_neg_distances)
        '''
        if global_context is not None:
            expanded = global_context.negatives_flat.unsqueeze(0).expand(batch_size, -1, -1)
            global_distances_flat = self.hard_negative_miner.lorentz_distance.batched_forward(
                anchor_emb,
                expanded,
            )
            _, topk_indices = torch.topk(
                global_distances_flat,
                k=k_negatives,
                dim=1,
                largest=False,
            )
            hard_negatives = global_context.negatives_flat[topk_indices]
            hard_neg_distances = global_distances_flat.gather(1, topk_indices)
            self._log_hard_negative_stats(
                hard_neg_distances, batch_idx, batch_size, used_global_batch=True
            )
            return hard_negatives, hard_neg_distances

        candidate_negatives = negative_emb_reshaped.clone()
        hard_negatives, hard_neg_distances = self.hard_negative_miner.mine_hard_negatives(
            anchor_emb=anchor_emb,
            candidate_negatives=candidate_negatives,
            k=k_negatives,
            return_distances=True,
        )
        self._log_hard_negative_stats(
            hard_neg_distances, batch_idx, batch_size, used_global_batch=False
        )
        return hard_negatives, hard_neg_distances

    def _apply_router_guided_sampling(
        self,
        anchor_output: Dict[str, torch.Tensor],
        negative_output: Dict[str, torch.Tensor],
        negative_emb_reshaped: torch.Tensor,
        hard_negatives: Optional[torch.Tensor],
        batch_size: int,
        k_negatives: int,
        batch_idx: int,
        global_context: Optional[GlobalNegativeContext],
    ) -> torch.Tensor:
        '''
        Apply router-guided sampling to select confusing negatives.

        Uses MoE router probabilities to find negatives that confuse the model.

        Args:
            anchor_output: Anchor encoder output with gate_probs
            negative_output: Negative encoder output with gate_probs
            negative_emb_reshaped: Negative embeddings
            hard_negatives: Optional hard negatives from embedding-based mining
            batch_size: Batch size
            k_negatives: Number of negatives
            batch_idx: Current batch index
            global_context: Optional global context for distributed training

        Returns:
            Selected negative embeddings
        '''
        anchor_gate_probs = anchor_output.get('gate_probs')
        negative_gate_probs = negative_output.get('gate_probs')
        if anchor_gate_probs is None or negative_gate_probs is None:
            if hard_negatives is not None:
                return hard_negatives
            logger.debug('Router-guided sampling skipped: gate probabilities not available')
            return negative_emb_reshaped

        if global_context is not None:
            global_negative_gate_probs = gather_embeddings_global(negative_gate_probs)
            global_negative_gate_probs_reshaped = global_negative_gate_probs.view(
                global_context.global_batch_size, global_context.global_k_negatives, -1
            )
            global_neg_gate_probs_flat = global_negative_gate_probs_reshaped.view(
                -1, anchor_gate_probs.shape[-1]
            )
            negative_gate_probs_expanded = global_neg_gate_probs_flat.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            confusion_scores = self.router_guided_miner.compute_confusion_scores(
                anchor_gate_probs, negative_gate_probs_expanded
            )
            _, router_topk_indices = torch.topk(
                confusion_scores,
                k=k_negatives,
                dim=1,
                largest=True,
            )
            router_hard_negatives = global_context.negatives_flat[router_topk_indices]
            router_confusion_scores = confusion_scores.gather(1, router_topk_indices)
        else:
            negative_gate_probs_local = negative_gate_probs.view(batch_size, k_negatives, -1)
            router_hard_negatives, router_confusion_scores = (
                self.router_guided_miner.mine_router_hard_negatives(
                    anchor_gate_probs=anchor_gate_probs,
                    negative_gate_probs=negative_gate_probs_local,
                    candidate_negatives=negative_emb_reshaped,
                    k=k_negatives,
                    return_scores=True,
                )
            )

        router_mix_ratio = self.current_schedule_scalars.get('router_mix_ratio', 0.5)
        router_mix_ratio = max(0.0, min(1.0, router_mix_ratio))

        if hard_negatives is not None:
            n_router = int(k_negatives * router_mix_ratio)
            n_embedding = k_negatives - n_router
            router_selected = router_hard_negatives[:, :n_router, :]
            embedding_selected = hard_negatives[:, :n_embedding, :]
            mixed_negatives = torch.cat([router_selected, embedding_selected], dim=1)
        else:
            mixed_negatives = router_hard_negatives

        if batch_idx == 0 and router_confusion_scores is not None:
            avg_confusion = router_confusion_scores.mean().item()
            min_confusion = router_confusion_scores.min().item()
            max_confusion = router_confusion_scores.max().item()
            self.log(
                'train/curriculum/router_confusion_mean',
                avg_confusion,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                'train/curriculum/router_confusion_min',
                min_confusion,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                'train/curriculum/router_confusion_max',
                max_confusion,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        return mixed_negatives

    def _prepare_negative_embeddings(
        self,
        anchor_output: Dict[str, torch.Tensor],
        negative_output: Dict[str, torch.Tensor],
        anchor_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        batch_size: int,
        k_negatives: int,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Prepare negative embeddings with optional hard negative mining.

        Args:
            anchor_output: Anchor encoder output
            negative_output: Negative encoder output
            anchor_emb: Anchor embeddings
            negative_emb: Negative embeddings (flat)
            batch_size: Batch size
            k_negatives: Number of negatives per sample
            batch_idx: Current batch index

        Returns:
            Tuple of (negative_emb_flat, negative_emb_reshaped)
        '''
        enable_hnm = self.current_curriculum_flags.get('enable_hard_negative_mining', False)
        enable_router = self.current_curriculum_flags.get('enable_router_guided_sampling', False)
        use_global_batch = self._should_use_global_batch(enable_hnm, enable_router)

        global_context = None
        if use_global_batch:
            global_context = self._gather_global_negative_pool(
                negative_emb, batch_size, k_negatives, batch_idx
            )

        negative_emb_reshaped = negative_emb.view(batch_size, k_negatives, -1)
        hard_negatives: Optional[torch.Tensor] = None

        if enable_hnm:
            hard_negatives, _ = self._perform_hard_negative_mining(
                anchor_emb,
                negative_emb_reshaped,
                k_negatives,
                batch_size,
                batch_idx,
                global_context,
            )
            negative_emb_reshaped = hard_negatives

        if enable_router:
            negative_emb_reshaped = self._apply_router_guided_sampling(
                anchor_output,
                negative_output,
                negative_emb_reshaped,
                hard_negatives,
                batch_size,
                k_negatives,
                batch_idx,
                global_context,
            )
        elif enable_hnm and hard_negatives is not None:
            negative_emb_reshaped = hard_negatives

        negative_emb = negative_emb_reshaped.view(batch_size * k_negatives, -1)
        return negative_emb, negative_emb_reshaped

    def _apply_false_negative_strategy_wrapper(
        self,
        anchor_emb: torch.Tensor,
        negative_emb_reshaped: torch.Tensor,
        false_negative_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        '''
        Apply false negative handling strategy.

        Args:
            anchor_emb: Anchor embeddings
            negative_emb_reshaped: Negative embeddings (batch_size, k_negatives, embed_dim)
            false_negative_mask: Optional mask of identified false negatives

        Returns:
            Tuple of (updated_mask, auxiliary_loss)
        '''
        if self.false_negative_config is None or negative_emb_reshaped is None:
            return false_negative_mask, None

        fn_mask, auxiliary_fn_loss = apply_false_negative_strategy(
            self.false_negative_config,
            anchor_emb,
            negative_emb_reshaped,
            false_negative_mask,
        )
        return fn_mask, auxiliary_fn_loss

    def _update_pseudo_labels(self) -> None:
        '''
        Run clustering on training data to generate pseudo-labels for false negative detection.

        Uses Hyperbolic K-Means compatible with Lorentz model to cluster embeddings
        directly in hyperbolic space using Lorentzian distances.
        '''
        if not hasattr(self.trainer, 'train_dataloader'):
            logger.warning('Trainer has no train_dataloader, cannot update pseudo-labels.')
            return

        logger.info('Generating embeddings for pseudo-label clustering (Hyperbolic K-Means)...')
        self.eval()
        all_embeddings = []
        all_codes = []

        # Sample a subset of batches for efficiency
        max_batches = 100
        batch_count = 0

        try:
            if self.trainer is None or self.trainer.train_dataloader is None:
                logger.warning('Trainer or train_dataloader is None, cannot update pseudo-labels.')
                return

            dataloader = self.trainer.train_dataloader

            for batch in dataloader:
                if batch_count >= max_batches:
                    break
                batch = self.transfer_batch_to_device(batch, self.device, 0)

                with torch.no_grad():
                    anchor_output = self(batch['anchor'])
                    # Use hyperbolic embeddings directly for hyperbolic K-Means
                    hyp_embs = anchor_output['embedding'].cpu()
                    all_embeddings.append(hyp_embs)
                    all_codes.extend(batch['anchor_code'])
                    batch_count += 1

            if not all_embeddings:
                logger.warning('No embeddings collected for pseudo-labeling')
                return

            all_embeddings = torch.cat(all_embeddings, dim=0)

            # Calculate cluster count
            fn_num_clusters = getattr(self.hparams, 'fn_num_clusters', 500)
            n_clusters = min(
                max(50,
                    len(all_embeddings) // 20),  # At least 50, at most 1 per 20 samples
                fn_num_clusters,
            )
            n_clusters = max(1, n_clusters)

            logger.info(
                f'Clustering {len(all_embeddings)} hyperbolic embeddings '
                f'into {n_clusters} clusters using Hyperbolic K-Means '
                f'(Lorentz model)...'
            )

            from naics_embedder.text_model.hyperbolic_clustering import HyperbolicKMeans

            curvature = getattr(self.hparams, 'curvature', 1.0)
            hyperbolic_kmeans = HyperbolicKMeans(
                n_clusters=n_clusters,
                curvature=curvature,
                max_iter=100,
                tol=1e-4,
                random_state=42,
                verbose=False,
            )
            labels = hyperbolic_kmeans.fit_predict(all_embeddings)

            self.code_to_pseudo_label = {code: int(label) for code, label in zip(all_codes, labels)}
            logger.info(
                f'Pseudo-label map updated with {len(self.code_to_pseudo_label)} entries. '
                f'Clustering inertia: {hyperbolic_kmeans.inertia_:.4f}, '
                f'iterations: {hyperbolic_kmeans.n_iter_}'
            )

        except Exception as e:
            logger.error(f'Failed to update pseudo-labels: {e}', exc_info=True)

        finally:
            self.train()
