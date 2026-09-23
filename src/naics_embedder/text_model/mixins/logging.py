# -------------------------------------------------------------------------------------------------
# Logging Mixin
# -------------------------------------------------------------------------------------------------
'''
Logging mixin for NAICSContrastiveModel.

Provides methods for logging various training and validation metrics:
- Negative sample distribution (over the checked selection)
- Tree distance distribution (over the checked selection)
- Adaptive margin statistics
- Selection health counters
- Hard negative mining statistics
- Router diversity metrics
- Loss breakdown
'''

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from naics_embedder.supervision.candidates import NegativeCandidateBatch, SelectedNegativeBatch
from naics_embedder.supervision.schema import SelectionReason

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

    def _log_negative_sample_distribution(self, relation_names: List[str], batch_size: int) -> None:
        '''
        Log distribution of negative sample types (child/sibling/cousin/distant).

        Issue #12: Track negative sample type distribution per curriculum phase. Relation names come
        from the selected negatives' anchor-relative structural relation IDs in the bundle.
        '''
        if self.curriculum_scheduler is None or not relation_names:
            return

        child_relations = {'child', 'grandchild', 'great-grandchild', 'great-great-grandchild'}
        cousin_relations = {
            'cousin',
            'nephew/niece',
            'grand-nephew/niece',
            'cousin_1_times_removed',
            'second_cousin',
        }
        sample_types = {'child': 0, 'sibling': 0, 'cousin': 0, 'distant': 0, 'unknown': 0}
        for relation in relation_names:
            if relation in child_relations:
                sample_types['child'] += 1
            elif relation == 'sibling':
                sample_types['sibling'] += 1
            elif relation in cousin_relations:
                sample_types['cousin'] += 1
            elif (
                relation == 'cross_sector' or relation.startswith('third_cousin')
                or relation.startswith('cousin_')
            ):
                sample_types['distant'] += 1
            else:
                sample_types['unknown'] += 1

        total_samples = len(relation_names)
        phase = self.curriculum_scheduler.get_phase(self.current_epoch)
        for sample_type, count in sample_types.items():
            self.log(
                f'train/curriculum/negative_samples_{sample_type}',
                count / total_samples,
                batch_size=batch_size,
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

    def _log_negative_tree_distance_distribution(
        self, distances: torch.Tensor, batch_size: int
    ) -> None:
        '''
        Log distribution of negative samples by tree distance bins.

        Issue #23: Track tree-distance categories to verify Phase 1 weighting. Distances are the
        selected negatives' anchor-relative structural distances from the bundle.
        '''
        total = distances.numel()
        if total == 0:
            return

        bins = {
            'sibling_or_closer': int(distances.le(2.0).sum()),
            'cousin': int((distances.gt(2.0) & distances.le(4.0)).sum()),
            'distant': int(distances.gt(4.0).sum()),
            'unknown': 0,
        }
        for name, count in bins.items():
            self.log(
                f'train/curriculum/tree_distance_{name}',
                count / total,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def _log_selection_health(
        self,
        candidates: NegativeCandidateBatch,
        selected: SelectedNegativeBatch,
        batch_size: int,
        *,
        entity_valid_mask: torch.Tensor,
    ) -> None:
        '''
        Log low-cardinality integrity counters for one checked selection as epoch sums.

        ``entity_valid_mask`` marks real (non-padding) candidates; ``candidates.valid_mask``
        additionally applies anchor-relative structural eligibility. Counters carry no candidate
        identities in their names or values.
        '''
        selectable_codes = candidates.code_id.masked_fill(~candidates.valid_mask, -1)
        sorted_codes = selectable_codes.sort(dim=1).values
        duplicates = (
            sorted_codes[:, 1:].eq(sorted_codes[:, :-1]) & sorted_codes[:, 1:].ge(0)
        ).sum()
        reasons = selected.selection_reasons
        selection_metrics = {
            'train/integrity/anchors_with_exclusions':
            candidates.is_explicit_exclusion.any(dim=1).sum(),
            'train/integrity/quota_selections':
            reasons.eq(int(SelectionReason.EXCLUSION_QUOTA)).sum(),
            'train/integrity/geometric_selections':
            reasons.eq(int(SelectionReason.GEOMETRIC)).sum(),
            'train/integrity/router_selections':
            reasons.eq(int(SelectionReason.ROUTER)).sum(),
            'train/integrity/difficulty_selections':
            reasons.eq(int(SelectionReason.DIFFICULTY)).sum(),
            'train/integrity/deterministic_backfills':
            reasons.eq(int(SelectionReason.BACKFILL)).sum(),
            'train/integrity/invalid_candidates_ignored': (~entity_valid_mask).sum(),
            'train/integrity/structurally_ineligible_candidates':
            (entity_valid_mask & ~candidates.valid_mask).sum(),
            'train/integrity/duplicate_candidates_removed': duplicates,
        }
        for name, value in selection_metrics.items():
            self.log(
                name,
                value.to(torch.float32),
                on_step=False,
                on_epoch=True,
                reduce_fx='sum',
                batch_size=batch_size,
            )

    def _log_selected_negative_stats(
        self,
        batch: Dict[str, Any],
        anchor_emb: torch.Tensor,
        selected: SelectedNegativeBatch,
        batch_idx: int,
        batch_size: int,
    ) -> None:
        '''
        Epoch-start negative diagnostics computed on the checked selection.

        Relation types, tree distances, and hard-negative distances all come from the same selected
        identities and the bundle's anchor-relative structure, so they describe the negatives
        actually used; no legacy artifact is read.
        '''
        if batch_idx != 0:
            return
        valid = selected.valid_mask
        relation_names = [
            self.relation_id_to_name.get(int(relation_id), 'unknown')
            for relation_id in selected.structural_relation_id[valid].tolist()
        ]
        self._log_negative_sample_distribution(relation_names, batch_size)
        self._log_negative_tree_distance_distribution(
            selected.structural_distance[valid], batch_size
        )

        enable_hnm = self.current_curriculum_flags.get('enable_hard_negative_mining', False)
        if not enable_hnm:
            return
        enable_router = self.current_curriculum_flags.get('enable_router_guided_sampling', False)
        with torch.no_grad():
            distances = self.hard_negative_miner.lorentz_distance.batched_forward(
                anchor_emb,
                selected.embedding,
            )
        self._log_hard_negative_stats(
            distances[valid],
            batch_idx,
            batch_size,
            used_global_batch=self._should_use_global_batch(enable_hnm, enable_router),
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
        structural_preference_loss: torch.Tensor,
        radius_reg_loss: torch.Tensor,
        level_radius_loss: torch.Tensor,
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
        if structural_preference_loss.item() > 0:
            self.log(
                'train/structural_preference_loss',
                structural_preference_loss,
                prog_bar=False,
                batch_size=batch_size,
            )
        if radius_reg_loss.item() > 0:
            self.log(
                'train/radius_reg_loss', radius_reg_loss, prog_bar=False, batch_size=batch_size
            )
        if level_radius_loss.item() > 0:
            self.log(
                'train/level_radius_loss',
                level_radius_loss,
                prog_bar=False,
                batch_size=batch_size,
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
            scalar_value = self._to_python_scalar(value)
            self.log(
                f'val/{name}',
                scalar_value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
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
            scalar_value = self._to_python_scalar(value)
            self.log(
                f'val/{name}',
                scalar_value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return metrics
