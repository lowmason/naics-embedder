# -------------------------------------------------------------------------------------------------
# NAICS Contrastive Learning Model
# -------------------------------------------------------------------------------------------------
'''
Main NAICS Contrastive Learning Model combining:
- MultiChannelEncoder with LoRA fine-tuning and MoE
- Hyperbolic embeddings using the Lorentz model
- Curriculum learning with structure-aware negative sampling
- Multi-level supervision and false negative detection

The model is decomposed into functional mixins for maintainability:
- DistributedMixin: Global batch sampling utilities
- LossMixin: Loss computation methods
- CurriculumMixin: Curriculum learning logic
- LoggingMixin: Logging utilities
- ValidationMixin: Validation step and evaluation
- OptimizerMixin: Optimizer configuration
'''

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import pytorch_lightning as pyl
import torch

from naics_embedder.metrics import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
)
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, load_validated_bundle
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
    CheckpointContract,
    containment_contract,
    contract_for_bundle,
    validate_checkpoint_contract,
)
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.mode import SupervisionModePolicy
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.supervision.selection import NegativeSelectionCoordinator
from naics_embedder.text_model.curriculum import CurriculumScheduler
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.hard_negative_mining import (
    LorentzianHardNegativeMiner,
    NormAdaptiveMargin,
    RouterGuidedNegativeMiner,
)
from naics_embedder.text_model.loss import (
    HierarchyPreservationLoss,
    HyperbolicInfoNCELoss,
    StructuralPreferenceLoss,
)
from naics_embedder.text_model.mixins import (
    CurriculumMixin,
    DistributedMixin,
    LoggingMixin,
    LossMixin,
    OptimizerMixin,
    ValidationMixin,
    gather_embeddings_global,
)

# Re-export distributed utilities for backward compatibility
__all__ = [
    'NAICSContrastiveModel',
    'gather_embeddings_global',
]
from naics_embedder.utils.config import FalseNegativeConfig
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy, load_naics_hierarchy

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Main NAICS Contrastive Learning Model
# -------------------------------------------------------------------------------------------------

class NAICSContrastiveModel(
    DistributedMixin,
    LossMixin,
    CurriculumMixin,
    LoggingMixin,
    ValidationMixin,
    OptimizerMixin,
    pyl.LightningModule,
):
    '''
    NAICS Contrastive Learning Model for learning hierarchical NAICS code embeddings.

    This model combines:
    - MultiChannelEncoder: LoRA-tuned transformer with Mixture of Experts
    - Hyperbolic embeddings: Lorentz model for hierarchical representation
    - Curriculum learning: Structure-aware dynamic curriculum (SADC)
    - Multiple loss functions: Contrastive, hierarchy preservation, structural preference

    Repaired Stage-3 supervision comes only from one validated supervision bundle: every training
    step encodes one canonical candidate pool, joins pair-dependent supervision by code identity,
    and gathers every loss field through one checked negative selection.

    The implementation is decomposed into functional mixins:
    - DistributedMixin: Multi-GPU global batch sampling
    - LossMixin: Loss computation (hierarchy, structural preference, regularization)
    - CurriculumMixin: Checked negative selection and pseudo-related candidates
    - LoggingMixin: Training and validation metric logging
    - ValidationMixin: Validation step and evaluation metrics
    - OptimizerMixin: Optimizer and scheduler configuration

    Args:
        base_model_name: HuggingFace model name for the base encoder
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        num_experts: Number of MoE experts
        top_k: Number of experts to select per token
        moe_hidden_dim: Hidden dimension of MoE layers
        temperature: Temperature for InfoNCE loss
        curvature: Hyperbolic space curvature
        hierarchy_weight: Weight for hierarchy preservation loss
        radius_reg_weight: Weight for radius regularization
        level_radius_weight: Weight for level-aware radius prior
        learning_rate: Base learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Number of warmup steps
        use_warmup_cosine: Use warmup + cosine decay scheduler
        load_balancing_coef: MoE load balancing coefficient
        fn_curriculum_start_epoch: Epoch to start false negative curriculum
        fn_cluster_every_n_epochs: Clustering frequency for pseudo-labels
        fn_num_clusters: Number of clusters for pseudo-labeling
        distance_matrix_path: Legacy ground truth distance matrix (not allowed in repaired mode,
            which reads structural distances from the supervision bundle)
        eval_every_n_epochs: Evaluation frequency
        eval_sample_size: Number of samples for evaluation
        tree_distance_alpha: Tree distance scaling factor
        base_margin: Base margin for adaptive margin
        curriculum_phase1_end: End of curriculum phase 1 (fraction)
        curriculum_phase2_end: End of curriculum phase 2 (fraction)
        curriculum_phase3_end: End of curriculum phase 3 (fraction)
        sibling_distance_threshold: Threshold for sibling relationships
        curriculum_phase_mode: Curriculum phase mode
        curriculum_anneal: Annealing configuration for curriculum
        false_negative_config: Configuration for false negative handling
        relations_parquet_path: Legacy NAICS relations parquet (not allowed in repaired mode, which
            reads the hierarchy from the supervision bundle)
        parent_eval_top_k: Top-k for parent retrieval evaluation
        child_eval_top_k: Top-k for child retrieval evaluation
        supervision_manifest_path: Manifest of the validated supervision bundle (required in
            repaired mode)
        supervision_contract_version: Expected supervision contract version
        supervision_mode: ``'repaired'`` (default) or the explicit, tagged
            ``'legacy_containment'`` mode (local unmined contrastive learning only)
        structural_preference_weight: Weight for the structural preference loss
        structural_preference_margin: Ordering margin for structural preference
        structural_preference_temperature: Softplus temperature for structural preference
        structural_preference_tie_tolerance: Structural distance tie tolerance
        selection_seed: Global seed for deterministic exclusion rotation
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
        supervision_bundle: Optional already-validated bundle for ``supervision_manifest_path``
            (not saved in hyperparameters)
    '''

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        temperature: float = 0.07,
        curvature: float = 1.0,
        hierarchy_weight: float = 0.1,
        radius_reg_weight: float = 0.01,
        level_radius_weight: float = 0.05,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        use_warmup_cosine: bool = False,
        load_balancing_coef: float = 0.01,
        fn_curriculum_start_epoch: int = 10,
        fn_cluster_every_n_epochs: int = 5,
        fn_num_clusters: int = 500,
        distance_matrix_path: Optional[str] = None,
        eval_every_n_epochs: int = 1,
        eval_sample_size: int = 500,
        tree_distance_alpha: float = 1.5,
        base_margin: float = 0.5,
        curriculum_phase1_end: float = 0.3,
        curriculum_phase2_end: float = 0.7,
        curriculum_phase3_end: float = 1.0,
        sibling_distance_threshold: float = 2.0,
        curriculum_phase_mode: str = 'three_phase',
        curriculum_anneal: Optional[Dict[str, float]] = None,
        false_negative_config: Optional[Union[FalseNegativeConfig, Dict[str, Any]]] = None,
        relations_parquet_path: Optional[str] = None,
        parent_eval_top_k: int = 1,
        child_eval_top_k: int = 5,
        supervision_manifest_path: Optional[str] = None,
        supervision_contract_version: str = CONTRACT_VERSION,
        supervision_mode: str = 'repaired',
        structural_preference_weight: float = 0.35,
        structural_preference_margin: float = 0.1,
        structural_preference_temperature: float = 1.0,
        structural_preference_tie_tolerance: float = 1e-6,
        selection_seed: int = 0,
        checkpoint_contract: Optional[CheckpointContract] = None,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
    ):
        super().__init__()

        self.supervision_policy = SupervisionModePolicy.from_name(supervision_mode)
        if self.supervision_policy.require_bundle:
            if supervision_manifest_path is None:
                raise ValueError(
                    'repaired supervision requires supervision_manifest_path; generate a bundle '
                    'with `naics-embedder data supervision` and set the printed manifest path'
                )
            if distance_matrix_path is not None or relations_parquet_path is not None:
                raise ValueError(
                    'repaired supervision reads structural distances and the NAICS hierarchy from '
                    'the supervision bundle; distance_matrix_path and relations_parquet_path are '
                    'legacy inputs'
                )

        # Bundle and contract objects stay out of hyperparameters: checkpoints record paths and
        # identifiers, and a restored model re-validates its bundle from the manifest path.
        self.save_hyperparameters(ignore=['checkpoint_contract', 'supervision_bundle'])
        self.supervision_mode = supervision_mode

        self.supervision_index: Optional[SupervisionIndex] = None
        self.selection_coordinator: Optional[NegativeSelectionCoordinator] = None
        self.naics_hierarchy: Optional[NaicsHierarchy] = None
        self.relation_id_to_name: Dict[int, str] = {}
        if self.supervision_policy.require_bundle:
            # Load the validated supervision bundle before any model construction: the single
            # authority for code identity, structural facts, exclusions, the evaluation
            # hierarchy, and ground-truth distances. A caller that already validated it may pass
            # it in.
            if supervision_bundle is None:
                bundle = load_validated_bundle(
                    supervision_manifest_path,
                    expected_contract=supervision_contract_version,
                )
            else:
                if supervision_bundle.manifest_path != Path(supervision_manifest_path).resolve():
                    raise ValueError(
                        f'pre-validated supervision bundle manifest '
                        f'{supervision_bundle.manifest_path} is not the configured manifest '
                        f'{supervision_manifest_path}'
                    )
                if supervision_bundle.manifest.contract_version != supervision_contract_version:
                    raise ValueError(
                        f'pre-validated supervision bundle has contract '
                        f'{supervision_bundle.manifest.contract_version}, expected '
                        f'{supervision_contract_version}'
                    )
                bundle = supervision_bundle
            runtime_contract = contract_for_bundle(bundle.manifest, supervision_mode)
            self.relation_id_to_name = {
                relation_id: name
                for name, relation_id in bundle.manifest.structural_relation_ids.items()
            }
            self.supervision_index = SupervisionIndex.from_bundle(bundle)
            self.selection_coordinator = NegativeSelectionCoordinator()
            self.naics_hierarchy = load_naics_hierarchy(str(bundle.artifact_path('relations')))
        else:
            runtime_contract = containment_contract()
            logger.warning(
                'LEGACY CONTAINMENT (%s): not contract-compliant Stage-3 training. Structural '
                'ranking and hierarchy losses, negative reordering, and pseudo-related handling '
                'are disabled; no supervision bundle is read.',
                self.supervision_policy.checkpoint_tag,
            )
            if relations_parquet_path:
                try:
                    self.naics_hierarchy = load_naics_hierarchy(relations_parquet_path)
                except FileNotFoundError:
                    logger.warning(
                        'NAICS relations parquet not found at %s; hierarchy diagnostics disabled',
                        relations_parquet_path,
                    )
        if checkpoint_contract is not None and checkpoint_contract != runtime_contract:
            raise ValueError(
                f'runtime checkpoint contract {checkpoint_contract.model_dump()} does not match '
                f'the model supervision contract {runtime_contract.model_dump()}'
            )
        self.checkpoint_contract = runtime_contract
        self.supervision_bundle_id = runtime_contract.bundle_id

        # Initialize encoder
        self.encoder = MultiChannelEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            curvature=curvature,
        )

        # Initialize loss function
        self.loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=self.encoder.embedding_dim,
            temperature=temperature,
            curvature=curvature,
        )

        # Initialize hard negative mining
        self.hard_negative_miner = LorentzianHardNegativeMiner(curvature=curvature)
        self.norm_adaptive_margin = NormAdaptiveMargin(base_margin=base_margin, curvature=curvature)
        self.router_guided_miner = RouterGuidedNegativeMiner(
            metric='kl_divergence',
            temperature=1.0,
        )

        # Store configuration
        self.load_balancing_coef = load_balancing_coef
        self.relations_parquet_path = relations_parquet_path
        self.parent_eval_top_k = parent_eval_top_k
        self.child_eval_top_k = child_eval_top_k

        # Initialize evaluation components
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()

        # Ground truth distances: the validated structural matrix in codebook order (legacy
        # containment may still evaluate against an old distance matrix)
        self.ground_truth_distances: Optional[torch.Tensor] = None
        self.code_to_idx: Optional[Dict[str, int]] = None
        if self.supervision_index is not None:
            self.ground_truth_distances = self.supervision_index.structural_distance
            self.code_to_idx = dict(self.supervision_index.code_to_id)
        elif distance_matrix_path:
            self._load_ground_truth_distances(distance_matrix_path)

        # Structural losses exist only where the policy permits them: legacy containment disables
        # them even when old weights are nonzero
        self.hierarchy_loss_fn = None
        self.structural_preference_loss_fn = None
        if self.supervision_policy.enable_structural_losses:
            if hierarchy_weight > 0:
                self.hierarchy_loss_fn = HierarchyPreservationLoss(
                    tree_distances=self.supervision_index.structural_distance,
                    code_to_idx=self.code_to_idx,
                    weight=hierarchy_weight,
                )
            # Structural preference over each anchor's positive plus selected negatives
            self.structural_preference_loss_fn = StructuralPreferenceLoss(
                curvature=curvature,
                margin=structural_preference_margin,
                temperature=structural_preference_temperature,
                tie_tolerance=structural_preference_tie_tolerance,
                weight=structural_preference_weight,
            )

        # Initialize validation state
        self.validation_embeddings: Dict[str, torch.Tensor] = {}
        self.validation_codes: List[str] = []

        # Initialize pseudo-label state
        self.code_to_pseudo_label: Dict[str, int] = {}

        # Initialize evaluation metrics history
        self.evaluation_metrics_history: List[Dict] = []

        # Initialize curriculum state
        self.curriculum_scheduler: Optional[CurriculumScheduler] = None
        self.current_curriculum_flags: Dict[str, bool] = {}
        self.current_schedule_scalars: Dict[str, float] = {}
        self.previous_phase: Optional[int] = None
        self.curriculum_anneal = curriculum_anneal
        self.curriculum_phase_mode = curriculum_phase_mode

        # Initialize false negative configuration
        if false_negative_config is None:
            self.false_negative_config = FalseNegativeConfig()
        elif isinstance(false_negative_config, FalseNegativeConfig):
            self.false_negative_config = false_negative_config
        else:
            self.false_negative_config = FalseNegativeConfig(**false_negative_config)

    def _load_ground_truth_distances(self, distance_matrix_path: str) -> None:
        '''
        Load ground truth NAICS tree distances for evaluation.

        Args:
            distance_matrix_path: Path to the distance matrix parquet file
        '''
        try:
            logger.info(f'Loading ground truth distances from: {distance_matrix_path}')

            df = pl.read_parquet(distance_matrix_path)
            n_codes = df.height

            ground_truth_distances = df.to_torch()
            logger.info(f'Distance matrix shape: [{n_codes}, {n_codes}]')

            code_to_idx = {}
            for col in df.columns:
                idx_col, code_col = col.split('-')
                idx = int(idx_col.replace('idx_', ''))
                code = code_col.replace('code_', '')
                code_to_idx[code] = idx

            self.ground_truth_distances = ground_truth_distances
            self.code_to_idx = code_to_idx

        except Exception as e:
            logger.error(f'Could not load ground truth distances: {e}')
            self.ground_truth_distances = None
            self.code_to_idx = None

    def forward(self, channel_inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        '''
        Forward pass through the encoder.

        Args:
            channel_inputs: Dictionary of channel inputs with tokenized text

        Returns:
            Dictionary containing:
            - embedding: Hyperbolic embeddings (batch_size, embed_dim + 1)
            - gate_probs: MoE gate probabilities (batch_size, num_experts)
            - top_k_indices: Selected expert indices (batch_size, top_k)
        '''
        return self.encoder(channel_inputs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''Record the supervision contract this checkpoint was trained under.'''
        checkpoint[CHECKPOINT_KEY] = self.checkpoint_contract.model_dump()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''
        Refuse to restore a checkpoint trained under any other supervision contract.

        Runs for Lightning exact resume and ``load_from_checkpoint``; weights-only migration
        never reaches this hook.
        '''
        validate_checkpoint_contract(checkpoint.get(CHECKPOINT_KEY), self.checkpoint_contract)

    def _forward_candidate_pool(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        '''
        Encode the collated candidate pool once and build occurrence UIDs.

        Only valid candidate rows are encoded; invalid padding rows receive zero outputs (the same
        convention as distributed padding) and are never selectable. UIDs are
        ``[rank, batch_row, source_slot]``, so invalid rows keep source slot ``-1``.

        Args:
            batch: Repaired collated batch

        Returns:
            Tuple of (flat candidate encoder output, ``[batch, candidate, 3]`` UIDs)
        '''
        valid = batch['candidate_valid_mask'].reshape(-1)
        valid_inputs = {
            channel: {name: value[valid] for name, value in inputs.items()}
            for channel, inputs in batch['candidate_inputs'].items()
        }
        valid_output = self(valid_inputs)
        candidate_output: Dict[str, torch.Tensor] = {}
        for name, value in valid_output.items():
            full = value.new_zeros((valid.shape[0], *value.shape[1:]))
            full[valid] = value
            candidate_output[name] = full

        source_slot = batch['candidate_source_slot']
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        )
        rank_component = torch.full_like(source_slot, rank)
        batch_component = torch.arange(
            int(batch['batch_size']), device=source_slot.device
        ).unsqueeze(1).expand_as(source_slot)
        candidate_uid = torch.stack([rank_component, batch_component, source_slot], dim=-1)
        return candidate_output, candidate_uid

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        '''
        Perform a single repaired training step over one checked negative selection.

        Order: encode anchors, positives, and the candidate pool once; select negatives by
        candidate identity; derive pseudo-related candidates from the selection only; then every
        loss consumes the same selected batch.

        Args:
            batch: Repaired collated batch (see ``collate_fn(..., supervision_mode='repaired')``)
            batch_idx: Batch index

        Returns:
            Total loss for optimization
        '''
        if self.supervision_policy.name == 'legacy_containment':
            return self._legacy_containment_training_step(batch, batch_idx)

        batch_size = int(batch['batch_size'])

        # Update curriculum state
        self._update_curriculum_state(batch_idx, batch_size)

        # Forward pass: anchors, positives, and the canonical candidate pool
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        candidate_output, candidate_uid = self._forward_candidate_pool(batch)
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']

        # Log training statistics
        self._log_multilevel_supervision_stats(batch, batch_idx, batch_size)
        self._log_sampling_metadata(batch, batch_size)

        # One checked selection; every negative field below is gathered by the same UIDs
        selected = self._select_negative_batch(
            batch=batch,
            anchor_output=anchor_output,
            candidate_output=candidate_output,
            candidate_uid=candidate_uid,
            batch_idx=batch_idx,
        )
        self._log_selected_negative_stats(anchor_emb, selected, batch_idx, batch_size)

        # Pseudo-related candidates exist only after selection; exclusions are never eligible
        pseudo_related = self._build_selected_pseudo_related_mask(batch['anchor_code_id'], selected)
        effective_mask, auxiliary_fn_loss = self._apply_false_negative_strategy_wrapper(
            anchor_emb,
            selected,
            pseudo_related,
        )
        contrastive_loss = self._compute_contrastive_loss(
            anchor_emb,
            positive_emb,
            selected,
            effective_mask,
        )
        if auxiliary_fn_loss is not None:
            contrastive_loss = contrastive_loss + auxiliary_fn_loss
        structural_preference_loss = self._compute_structural_preference_loss(
            anchor_emb,
            positive_emb,
            batch,
            selected,
        )

        # Adaptive margin diagnostics (feed metric-triggered curriculum annealing)
        adaptive_margins = self.norm_adaptive_margin(anchor_emb)
        self._log_adaptive_margin_stats(adaptive_margins, batch_idx, batch_size)

        # Auxiliary losses; selected-candidate regularizers see only valid selections
        hierarchy_loss = self._compute_hierarchy_loss(anchor_emb, positive_emb, batch, batch_size)
        radius_reg_loss = self._compute_radius_regularization(
            anchor_emb,
            positive_emb,
            selected.embedding[selected.valid_mask],
            batch_size,
        )
        level_radius_loss = self._compute_level_radius_alignment_loss(
            anchor_emb,
            positive_emb,
            batch,
            batch_size,
        )

        # MoE load balancing over anchors, positives, and valid candidate rows only
        valid_candidates = batch['candidate_valid_mask'].reshape(-1)
        valid_candidate_output = {
            name: candidate_output[name][valid_candidates]
            for name in ('gate_probs', 'top_k_indices') if name in candidate_output
        }
        gate_probs_list, topk_indices_list = self._collect_gate_outputs(
            [anchor_output, positive_output, valid_candidate_output]
        )
        self._log_router_diversity(gate_probs_list, batch_size)
        raw_load_balancing_loss = self._compute_load_balancing_loss(
            gate_probs_list,
            topk_indices_list,
            batch_size,
        )

        # Combine losses
        total_loss, scaled_load_balancing_loss = self._combine_loss_terms(
            contrastive_loss,
            raw_load_balancing_loss,
            hierarchy_loss,
            structural_preference_loss,
            radius_reg_loss,
            level_radius_loss,
        )

        # Log loss breakdown
        self._log_loss_breakdown(
            contrastive_loss,
            scaled_load_balancing_loss,
            hierarchy_loss,
            structural_preference_loss,
            radius_reg_loss,
            level_radius_loss,
            total_loss,
            batch_size,
        )

        return total_loss

    def _legacy_containment_training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        '''
        Contained legacy step: local unmined negatives in collated order plus supervision-free
        regularizers. No structural, hierarchy, reordering, or pseudo-related path runs.
        '''
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])
        batch_size = int(batch['batch_size'])
        k_negatives = int(batch['k_negatives'])
        negative_emb = negative_output['embedding'].reshape(batch_size, k_negatives, -1)
        valid = torch.ones(
            (batch_size, k_negatives),
            dtype=torch.bool,
            device=negative_emb.device,
        )
        explicit = torch.zeros_like(valid)
        contrastive = self.loss_fn(
            anchor_output['embedding'],
            positive_output['embedding'],
            negative_emb,
            valid_mask=valid,
            is_explicit_exclusion=explicit,
            pseudo_related_mask=None,
        )
        gate_probs, topk_indices = self._collect_gate_outputs(
            [anchor_output, positive_output, negative_output]
        )
        load_balancing = self._compute_load_balancing_loss(
            gate_probs,
            topk_indices,
            batch_size,
        )
        radius = self._compute_radius_regularization(
            anchor_output['embedding'],
            positive_output['embedding'],
            negative_output['embedding'],
            batch_size,
        )
        level_radius = self._compute_level_radius_alignment_loss(
            anchor_output['embedding'],
            positive_output['embedding'],
            batch,
            batch_size,
        )
        disabled = contrastive.new_zeros(())
        total, scaled_load_balancing = self._combine_loss_terms(
            contrastive,
            load_balancing,
            disabled,
            disabled,
            radius,
            level_radius,
        )
        self._log_loss_breakdown(
            contrastive,
            scaled_load_balancing,
            disabled,
            disabled,
            radius,
            level_radius,
            total,
            batch_size,
        )
        self.log(
            'train/integrity/legacy_containment',
            1.0,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return total
