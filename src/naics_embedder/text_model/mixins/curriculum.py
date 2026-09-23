# -------------------------------------------------------------------------------------------------
# Curriculum Learning Mixin
# -------------------------------------------------------------------------------------------------
'''
Curriculum learning mixin for NAICSContrastiveModel.

Provides methods for:
- Curriculum state management
- Canonical candidate pools and the one checked negative selection (difficulty, geometric, and
  router-guided proposals)
- Post-selection pseudo-related candidates
- Pseudo-label clustering for false negative detection
'''

import logging
from typing import Any, Dict, List, Optional

import torch

from naics_embedder.supervision.candidates import (
    CandidateEntityBatch,
    CandidateProposal,
    NegativeCandidateBatch,
    SelectedNegativeBatch,
)
from naics_embedder.supervision.schema import (
    SAMPLING_ROLE_TO_ID,
    SamplingProvenance,
    SamplingRole,
    SelectionReason,
)
from naics_embedder.text_model.mixins.distributed import gather_candidate_entities

logger = logging.getLogger(__name__)

def _proposal_from_local_uids(
    *,
    local_candidate_uid: torch.Tensor,
    active_candidates: NegativeCandidateBatch,
    local_source_indices: torch.Tensor,
) -> CandidateProposal:
    '''
    Translate collated difficulty proposals (local pool positions) into the active pool by UID.

    Rank-major distributed gathering changes source positions but preserves occurrence identity,
    so proposals are matched by candidate UID rather than position. Earlier proposals score higher;
    padding (index -1) becomes an ineligible ``-inf`` entry.

    Raises:
        ValueError: On a malformed or out-of-bounds proposal, or a UID that is missing from, or
            duplicated in, the active pool.
    '''
    if local_source_indices.ndim != 2:
        raise ValueError('difficulty proposal indices must have shape [batch, proposal]')
    if local_source_indices.ge(local_candidate_uid.shape[1]).any():
        raise ValueError('difficulty proposal contains an out-of-bounds local source index')
    safe = local_source_indices.clamp_min(0)
    local_uids = local_candidate_uid.gather(
        1,
        safe.unsqueeze(-1).expand(-1, -1, 3),
    )
    matches = active_candidates.candidate_uid.unsqueeze(2).eq(
        local_uids.unsqueeze(1)
    ).all(dim=-1)
    expected = local_source_indices.ge(0)
    match_count = matches.sum(dim=1)
    if (match_count[expected] != 1).any():
        raise ValueError('difficulty proposal UID is missing or duplicated in active pool')
    active_indices = matches.to(torch.int64).argmax(dim=1).masked_fill(~expected, -1)
    width = local_source_indices.shape[1]
    scores = torch.arange(
        width,
        0,
        -1,
        dtype=active_candidates.embedding.dtype,
        device=active_candidates.embedding.device,
    ).unsqueeze(0).expand_as(active_indices)
    scores = scores.masked_fill(~expected, -torch.inf)
    return CandidateProposal(
        source_indices=active_indices,
        scores=scores,
        reason=SelectionReason.DIFFICULTY,
    )

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
    - selection_coordinator: NegativeSelectionCoordinator
    - supervision_index: SupervisionIndex
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

    def _selection_seed(self) -> int:
        '''Global seed for deterministic exclusion rotation (the training seed).'''
        return int(getattr(self.hparams, 'selection_seed', 0))

    def _select_negative_batch(
        self,
        *,
        batch: Dict[str, Any],
        anchor_output: Dict[str, torch.Tensor],
        candidate_output: Dict[str, torch.Tensor],
        candidate_uid: torch.Tensor,
        batch_idx: int,
    ) -> SelectedNegativeBatch:
        '''
        Build the canonical candidate pool and perform the one checked negative selection.

        Flow: local candidate entities -> optional distributed entity gather -> anchor-relative
        supervision join -> ``NegativeCandidateBatch`` -> difficulty/geometric/router proposals ->
        coordinator selection (exclusion quota, dedup, backfill) -> the single gather.

        Args:
            batch: Repaired collated batch.
            anchor_output: Encoder output for the anchors.
            candidate_output: Encoder output for the flattened candidate pool.
            candidate_uid: ``[batch, candidate, 3]`` occurrence UIDs (rank, row, source slot).
            batch_idx: Current batch index.
        '''
        batch_size = int(batch['batch_size'])
        candidate_count = int(batch['k_candidates'])
        embedding = candidate_output['embedding'].reshape(batch_size, candidate_count, -1)
        raw_gate_probs = candidate_output.get('gate_probs')
        gate_probs = (
            None
            if raw_gate_probs is None
            else raw_gate_probs.reshape(batch_size, candidate_count, -1)
        )
        local_entities = CandidateEntityBatch(
            candidate_uid=candidate_uid,
            code_id=batch['candidate_code_id'],
            embedding=embedding,
            router_gate_probs=gate_probs,
            valid_mask=batch['candidate_valid_mask'],
        )

        enable_geometric = self.current_curriculum_flags.get(
            'enable_hard_negative_mining', False
        )
        enable_router = self.current_curriculum_flags.get(
            'enable_router_guided_sampling', False
        )
        if self._should_use_global_batch(enable_geometric, enable_router):
            gathered = gather_candidate_entities(local_entities)
            entities = CandidateEntityBatch(
                candidate_uid=gathered.candidate_uid.expand(batch_size, -1, -1),
                code_id=gathered.code_id.expand(batch_size, -1),
                embedding=gathered.embedding.expand(batch_size, -1, -1),
                router_gate_probs=(
                    None
                    if gathered.router_gate_probs is None
                    else gathered.router_gate_probs.expand(batch_size, -1, -1)
                ),
                valid_mask=gathered.valid_mask.expand(batch_size, -1),
            )
            sampling_role_id = torch.full_like(
                entities.code_id,
                SAMPLING_ROLE_TO_ID[SamplingRole.NEGATIVE],
                dtype=torch.int8,
            )
            sampling_provenance_id = torch.full_like(
                entities.code_id,
                int(SamplingProvenance.DISTRIBUTED_POOL),
                dtype=torch.int8,
            )
        else:
            entities = local_entities
            sampling_role_id = batch['candidate_sampling_role_id']
            sampling_provenance_id = batch['candidate_sampling_provenance_id']

        pair = self.supervision_index.join(
            batch['anchor_code_id'],
            entities.code_id,
            entities.valid_mask,
        )
        relation_margin = pair.structural_relation_id.to(torch.float32) - batch[
            'positive_structural_relation_id'
        ].to(torch.float32).unsqueeze(1)
        distance_margin = pair.structural_distance - batch[
            'positive_structural_distance'
        ].unsqueeze(1)
        candidates = NegativeCandidateBatch(
            candidate_uid=entities.candidate_uid,
            code_id=entities.code_id,
            embedding=entities.embedding,
            structural_distance=pair.structural_distance,
            structural_relation_id=pair.structural_relation_id,
            anchor_excludes_candidate=pair.anchor_excludes_candidate,
            candidate_excludes_anchor=pair.candidate_excludes_anchor,
            is_explicit_exclusion=pair.is_explicit_exclusion,
            semantic_target_id=pair.semantic_target_id,
            semantic_source_id=pair.semantic_source_id,
            sampling_role_id=sampling_role_id,
            sampling_provenance_id=sampling_provenance_id,
            relation_margin=relation_margin,
            distance_margin=distance_margin,
            router_gate_probs=entities.router_gate_probs,
            valid_mask=entities.valid_mask,
            runtime_fields={'difficulty': pair.structural_distance},
        )

        proposals: List[CandidateProposal] = [
            _proposal_from_local_uids(
                local_candidate_uid=candidate_uid,
                active_candidates=candidates,
                local_source_indices=batch['difficulty_proposal_indices'],
            )
        ]
        selection_k = int(batch['selection_k'])
        if enable_geometric:
            proposals.append(
                self.hard_negative_miner.propose(
                    anchor_output['embedding'],
                    candidates,
                    k=selection_k,
                )
            )
        if enable_router:
            anchor_gate_probs = anchor_output.get('gate_probs')
            if anchor_gate_probs is None or candidates.router_gate_probs is None:
                raise ValueError(
                    'router-guided selection requires anchor and candidate gate probabilities'
                )
            proposals.append(
                self.router_guided_miner.propose(
                    anchor_gate_probs=anchor_gate_probs,
                    candidates=candidates,
                    k=selection_k,
                )
            )

        selection = self.selection_coordinator.select(
            candidates,
            anchor_code_ids=batch['anchor_code_id'],
            positive_code_ids=batch['positive_code_id'],
            k=selection_k,
            epoch=int(self.current_epoch),
            global_seed=self._selection_seed(),
            proposals=tuple(proposals),
        )
        selected = candidates.select(selection)
        self._log_selection_health(candidates, selected, batch_size)
        return selected

    def _build_selected_pseudo_related_mask(
        self,
        anchor_code_ids: torch.Tensor,
        selected: SelectedNegativeBatch,
    ) -> Optional[torch.Tensor]:
        '''
        Pseudo-related flags for the checked selection, from clustering pseudo-labels.

        Built only after selection so every flag belongs to the selected candidate's identity.
        Explicit exclusions and invalid entries are never pseudo-related. Identity or shape
        errors propagate.

        Args:
            anchor_code_ids: ``[batch]`` anchor code IDs
            selected: The checked selected-negative batch

        Returns:
            ``[batch, selected]`` boolean mask, or None when clustering is inactive
        '''
        if not self.current_curriculum_flags.get('enable_clustering', False):
            return None
        if not self.code_to_pseudo_label:
            return None
        id_to_code = self.supervision_index.id_to_code
        device = selected.code_id.device
        anchor_labels = torch.tensor(
            [
                self.code_to_pseudo_label.get(id_to_code[int(code_id)], -1)
                for code_id in anchor_code_ids.tolist()
            ],
            device=device,
        )
        candidate_labels = torch.tensor(
            [
                [self.code_to_pseudo_label.get(id_to_code[int(code_id)], -2) for code_id in row]
                for row in selected.code_id.tolist()
            ],
            device=device,
        ).reshape(selected.code_id.shape)
        pseudo_related = (
            anchor_labels.unsqueeze(1).eq(candidate_labels)
            & anchor_labels.unsqueeze(1).ge(0)
            & candidate_labels.ge(0)
        )
        return pseudo_related & ~selected.is_explicit_exclusion & selected.valid_mask

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
