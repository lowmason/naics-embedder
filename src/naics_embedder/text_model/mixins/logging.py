# -------------------------------------------------------------------------------------------------
# Logging Mixin
# -------------------------------------------------------------------------------------------------
'''
Logging mixin for NAICSContrastiveModel.

Provides methods for logging various training and validation metrics:
- Negative sample distribution
- Tree distance distribution
- Adaptive margin statistics
- Global batch statistics
- Hard negative mining statistics
- Router diversity metrics
- Loss breakdown
'''

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

class LoggingMixin:
    '''
    Mixin providing logging functionality for training metrics.

    This mixin expects the following attributes on the class:
    - curriculum_scheduler: Optional curriculum scheduler
    - current_epoch: int
    - current_curriculum_flags: Dict[str, bool]
    - ground_truth_distances: Optional[torch.Tensor]
    - code_to_idx: Optional[Dict[str, int]]
    - naics_hierarchy: Optional hierarchy for retrieval metrics
    '''

    def _to_python_scalar(self, value: Any) -> Any:
        '''Convert any numeric value to a Python scalar for logging.'''
        if isinstance(value, torch.Tensor):
            return value.item()
        elif isinstance(value, (bool, int)):
            return int(value)
        else:
            return float(value)

    def _log_multilevel_supervision_stats(
        self, batch: Dict[str, Any], batch_idx: int, batch_size: int
    ) -> None:
        '''Log statistics about multi-level positive supervision.'''
        if 'positive_levels' not in batch or batch_idx != 0:
            return

        level_counts: Dict[str, int] = {}
        for level in batch['positive_levels']:
            level_counts[level] = level_counts.get(level, 0) + 1

        for level, count in sorted(level_counts.items()):
            self.log(
                f'train/multilevel/positive_level_{level}_count',
                count,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _log_sampling_metadata(self, batch: Dict[str, Any], batch_size: int) -> None:
        '''Log SANS (Structure-Aware Negative Sampling) metadata.'''
        sampling_metadata = batch.get('sampling_metadata')
        if not sampling_metadata or sampling_metadata.get('strategy') != 'sans_static':
            return

        sampled_near = sampling_metadata.get('sampled_near', 0)
        sampled_far = sampling_metadata.get('sampled_far', 0)
        total_sampled = sampled_near + sampled_far
        if total_sampled > 0:
            near_pct = sampled_near / total_sampled
            self.log(
                'train/sans_static/sample_near_pct',
                near_pct,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        candidates_near = sampling_metadata.get('candidates_near', 0)
        candidates_far = sampling_metadata.get('candidates_far', 0)
        total_candidates = candidates_near + candidates_far
        if total_candidates > 0:
            candidate_near_pct = candidates_near / total_candidates
            self.log(
                'train/sans_static/candidate_near_pct',
                candidate_near_pct,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        effective_near_weight = sampling_metadata.get('avg_effective_near_weight')
        if effective_near_weight is not None:
            self.log(
                'train/sans_static/effective_near_weight',
                effective_near_weight,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _log_negative_sample_stats_if_needed(self, batch: Dict[str, Any], batch_idx: int) -> None:
        '''Log negative sample statistics at the start of each epoch.'''
        if batch_idx != 0 or 'negative_codes' not in batch:
            return
        self._log_negative_sample_distribution(batch)
        self._log_negative_tree_distance_distribution(batch)

    def _log_negative_sample_distribution(self, batch: Dict[str, Any]) -> None:
        '''
        Log distribution of negative sample types (child/sibling/cousin/distant).

        Issue #12: Track negative sample type distribution per curriculum phase.
        '''
        if self.curriculum_scheduler is None:
            return

        try:
            from naics_embedder.utils.utilities import get_relationship

            anchor_codes = batch['anchor_code']
            negative_codes = batch['negative_codes']

            # Classify negative samples by relationship type
            sample_types = {'child': 0, 'sibling': 0, 'cousin': 0, 'distant': 0, 'unknown': 0}
            total_samples = 0

            for anchor_code, neg_codes in zip(anchor_codes, negative_codes):
                for neg_code in neg_codes:
                    try:
                        relation = get_relationship(anchor_code, neg_code)

                        # Classify into categories
                        child_relations = [
                            'child',
                            'grandchild',
                            'great-grandchild',
                            'great-great-grandchild',
                        ]
                        if relation in child_relations:
                            sample_types['child'] += 1
                        elif relation == 'sibling':
                            sample_types['sibling'] += 1
                        elif relation in [
                            'cousin',
                            'nephew/niece',
                            'grand-nephew/niece',
                            'cousin_1_times_removed',
                            'second_cousin',
                        ]:
                            sample_types['cousin'] += 1
                        elif (
                            relation in ['unrelated'] or relation.startswith('third_cousin')
                            or relation.startswith('cousin_')
                        ):
                            sample_types['distant'] += 1
                        else:
                            sample_types['unknown'] += 1

                        total_samples += 1
                    except Exception:
                        sample_types['unknown'] += 1
                        total_samples += 1

            if total_samples > 0:
                phase = self.curriculum_scheduler.get_phase(self.current_epoch)

                # Log to TensorBoard
                for sample_type, count in sample_types.items():
                    self.log(
                        f'train/curriculum/negative_samples_{sample_type}',
                        count / total_samples,
                        batch_size=len(anchor_codes),
                        on_step=False,
                        on_epoch=True,
                    )

                # Log summary every 5 epochs to reduce noise
                if self.current_epoch % 5 == 0:
                    logger.info(
                        f'Negative sample distribution '
                        f'(Phase {phase}, Epoch {self.current_epoch}):\n'
                        f'  • Child: {sample_types["child"] / total_samples * 100:.1f}%\n'
                        f'  • Sibling: {sample_types["sibling"] / total_samples * 100:.1f}%\n'
                        f'  • Cousin: {sample_types["cousin"] / total_samples * 100:.1f}%\n'
                        f'  • Distant: {sample_types["distant"] / total_samples * 100:.1f}%\n'
                        f'  • Unknown: {sample_types["unknown"] / total_samples * 100:.1f}%'
                    )

        except Exception as e:
            logger.debug(f'Failed to log negative sample distribution: {e}')

    def _log_negative_tree_distance_distribution(self, batch: Dict[str, Any]) -> None:
        '''
        Log distribution of negative samples by tree distance bins.

        Issue #23: Track tree-distance categories to verify Phase 1 weighting.
        '''
        if self.ground_truth_distances is None or self.code_to_idx is None:
            return

        try:
            anchor_codes = batch['anchor_code']
            negative_codes = batch['negative_codes']

            bins = {'sibling_or_closer': 0, 'cousin': 0, 'distant': 0, 'unknown': 0}
            total = 0

            for anchor_code, neg_codes in zip(anchor_codes, negative_codes):
                anchor_idx = self.code_to_idx.get(anchor_code)
                if anchor_idx is None:
                    continue

                for neg_code in neg_codes:
                    neg_idx = self.code_to_idx.get(neg_code)
                    if neg_idx is None:
                        bins['unknown'] += 1
                        total += 1
                        continue

                    distance = self.ground_truth_distances[anchor_idx, neg_idx].item()

                    if distance <= 2.0:
                        bins['sibling_or_closer'] += 1
                    elif distance <= 4.0:
                        bins['cousin'] += 1
                    else:
                        bins['distant'] += 1

                    total += 1

            if total > 0:
                for name, count in bins.items():
                    self.log(
                        f'train/curriculum/tree_distance_{name}',
                        count / total,
                        batch_size=len(anchor_codes),
                        on_step=False,
                        on_epoch=True,
                    )
        except Exception as e:
            logger.debug(f'Failed to log tree distance distribution: {e}')

    def _log_global_batch_stats(
        self,
        global_negative_emb: torch.Tensor,
        batch_size: int,
        global_batch_size: int,
        global_k_negatives: int,
    ) -> None:
        '''Log memory usage and size statistics for global batch sampling.'''
        global_negatives_memory_mb = (
            global_negative_emb.numel() * global_negative_emb.element_size() / (1024**2)
        )
        similarity_matrix_memory_mb = batch_size * global_batch_size * global_k_negatives * 4 / (
            1024**2
        )
        self.log(
            'train/global_batch/global_negatives_memory_mb',
            global_negatives_memory_mb,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/global_batch/similarity_matrix_memory_mb',
            similarity_matrix_memory_mb,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/global_batch/global_batch_size',
            global_batch_size,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/global_batch/global_k_negatives',
            global_k_negatives,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

    def _log_hard_negative_stats(
        self,
        hard_neg_distances: Optional[torch.Tensor],
        batch_idx: int,
        batch_size: int,
        used_global_batch: bool,
    ) -> None:
        '''Log hard negative mining statistics.'''
        if hard_neg_distances is None or batch_idx != 0:
            return

        avg_hard_neg_dist = hard_neg_distances.mean().item()
        min_hard_neg_dist = hard_neg_distances.min().item()
        max_hard_neg_dist = hard_neg_distances.max().item()
        self.log(
            'train/curriculum/hard_neg_avg_distance',
            avg_hard_neg_dist,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/curriculum/hard_neg_min_distance',
            min_hard_neg_dist,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/curriculum/hard_neg_max_distance',
            max_hard_neg_dist,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        if used_global_batch:
            self.log(
                'train/global_batch/global_hard_negatives_used',
                True,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _log_adaptive_margin_stats(
        self, adaptive_margins: torch.Tensor, batch_idx: int, batch_size: int
    ) -> None:
        '''Log adaptive margin statistics.'''
        adaptive_margin_mean_value = adaptive_margins.mean().item()
        if batch_idx == 0:
            adaptive_margin_min_value = adaptive_margins.min().item()
            adaptive_margin_max_value = adaptive_margins.max().item()
            self.log(
                'train/curriculum/adaptive_margin_mean',
                adaptive_margin_mean_value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                'train/curriculum/adaptive_margin_min',
                adaptive_margin_min_value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                'train/curriculum/adaptive_margin_max',
                adaptive_margin_max_value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        if self.curriculum_scheduler is not None:
            self.curriculum_scheduler.update_metrics(
                {'adaptive_margin_mean': adaptive_margin_mean_value}
            )

    def _log_router_diversity(self, gate_probs_list: List[torch.Tensor], batch_size: int) -> None:
        '''Log router diversity metrics for MoE.'''
        if not gate_probs_list or not self.current_curriculum_flags.get(
            'enable_router_guided_sampling', False
        ):
            return

        gate_probs_combined = torch.cat(gate_probs_list, dim=0)
        log_probs = torch.log(gate_probs_combined + 1e-8)
        entropy_per_token = -(gate_probs_combined * log_probs).sum(dim=1)
        expert_diversity = entropy_per_token.mean()
        self.log(
            'train/curriculum/router_expert_diversity',
            expert_diversity.item(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

    def _log_loss_breakdown(
        self,
        contrastive_loss: torch.Tensor,
        scaled_load_balancing_loss: torch.Tensor,
        hierarchy_loss: torch.Tensor,
        lambdarank_loss: torch.Tensor,
        radius_reg_loss: torch.Tensor,
        total_loss: torch.Tensor,
        batch_size: int,
    ) -> None:
        '''Log breakdown of all loss components.'''
        self.log('train/contrastive_loss', contrastive_loss, prog_bar=True, batch_size=batch_size)
        self.log(
            'train/load_balancing_loss',
            scaled_load_balancing_loss,
            prog_bar=True,
            batch_size=batch_size,
        )
        if hierarchy_loss.item() > 0:
            self.log('train/hierarchy_loss', hierarchy_loss, prog_bar=False, batch_size=batch_size)
        if lambdarank_loss.item() > 0:
            self.log(
                'train/lambdarank_loss', lambdarank_loss, prog_bar=False, batch_size=batch_size
            )
        if radius_reg_loss.item() > 0:
            self.log(
                'train/radius_reg_loss', radius_reg_loss, prog_bar=False, batch_size=batch_size
            )
        self.log('train/total_loss', total_loss, prog_bar=True, batch_size=batch_size)

    def _log_radius_structure_metrics(
        self,
        embeddings: torch.Tensor,
        codes: Sequence[str],
        batch_size: int,
    ) -> Dict[str, float]:
        '''Log radius structure metrics for hyperbolic embeddings.'''
        if self.naics_hierarchy is None or not codes:
            return {}

        from naics_embedder.metrics.hierarchy_structure import compute_radius_structure_metrics

        metrics = compute_radius_structure_metrics(embeddings, codes, self.naics_hierarchy)
        for name, value in metrics.items():
            self.log(
                f'val/{name}',
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        return metrics

    def _log_hierarchy_retrieval_metrics(
        self,
        distance_matrix: torch.Tensor,
        codes: Sequence[str],
        batch_size: int,
    ) -> Dict[str, float]:
        '''Log hierarchy retrieval metrics (parent/child recall).'''
        if self.naics_hierarchy is None or distance_matrix.numel() == 0:
            return {}

        from naics_embedder.metrics.hierarchy_structure import compute_hierarchy_retrieval_metrics

        metrics = compute_hierarchy_retrieval_metrics(
            distance_matrix,
            codes,
            self.naics_hierarchy,
            parent_top_k=self.parent_eval_top_k,
            child_top_k=self.child_eval_top_k,
        )
        for name, value in metrics.items():
            self.log(
                f'val/{name}',
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        return metrics
