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
from typing import Any, Dict, List, Optional, Union

import polars as pl
import pytorch_lightning as pyl
import torch

from naics_embedder.text_model.curriculum import CurriculumScheduler
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
)
from naics_embedder.text_model.hard_negative_mining import (
    LorentzianHardNegativeMiner,
    NormAdaptiveMargin,
    RouterGuidedNegativeMiner,
)
from naics_embedder.text_model.loss import HyperbolicInfoNCELoss
from naics_embedder.text_model.mixins import (
    CurriculumMixin,
    DistributedMixin,
    GlobalNegativeContext,
    LoggingMixin,
    LossMixin,
    OptimizerMixin,
    ValidationMixin,
    gather_embeddings_global,
    gather_negative_codes_global,
)

# Re-export distributed utilities for backward compatibility
__all__ = [
    'NAICSContrastiveModel',
    'gather_embeddings_global',
    'gather_negative_codes_global',
    'GlobalNegativeContext',
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
    - Multiple loss functions: Contrastive, hierarchy preservation, LambdaRank

    The implementation is decomposed into functional mixins:
    - DistributedMixin: Multi-GPU global batch sampling
    - LossMixin: Loss computation (hierarchy, ranking, regularization)
    - CurriculumMixin: Hard negative mining, router-guided sampling
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
        rank_order_weight: Weight for LambdaRank loss
        radius_reg_weight: Weight for radius regularization
        learning_rate: Base learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Number of warmup steps
        use_warmup_cosine: Use warmup + cosine decay scheduler
        load_balancing_coef: MoE load balancing coefficient
        fn_curriculum_start_epoch: Epoch to start false negative curriculum
        fn_cluster_every_n_epochs: Clustering frequency for pseudo-labels
        fn_num_clusters: Number of clusters for pseudo-labeling
        distance_matrix_path: Path to ground truth distance matrix
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
        relations_parquet_path: Path to NAICS relations parquet
        parent_eval_top_k: Top-k for parent retrieval evaluation
        child_eval_top_k: Top-k for child retrieval evaluation
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
        rank_order_weight: float = 0.15,
        radius_reg_weight: float = 0.01,
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
    ):
        super().__init__()

        self.save_hyperparameters()

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

        # Load NAICS hierarchy if available
        self.naics_hierarchy: Optional[NaicsHierarchy] = None
        if relations_parquet_path:
            try:
                self.naics_hierarchy = load_naics_hierarchy(relations_parquet_path)
            except FileNotFoundError:
                logger.warning(
                    'NAICS relations parquet not found at %s; hierarchy diagnostics disabled',
                    relations_parquet_path,
                )

        # Initialize evaluation components
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()

        # Load ground truth distances
        self.ground_truth_distances = None
        self.code_to_idx = None
        if distance_matrix_path:
            self._load_ground_truth_distances(distance_matrix_path)

        # Initialize hierarchy preservation loss
        self.hierarchy_loss_fn = None
        hierarchy_weight = getattr(self.hparams, 'hierarchy_weight', 0.1)
        if (
            self.ground_truth_distances is not None and self.code_to_idx is not None
            and hierarchy_weight > 0
        ):
            from naics_embedder.text_model.loss import HierarchyPreservationLoss

            self.hierarchy_loss_fn = HierarchyPreservationLoss(
                tree_distances=self.ground_truth_distances,
                code_to_idx=self.code_to_idx,
                weight=hierarchy_weight,
            )

        # Initialize LambdaRank loss
        self.lambdarank_loss_fn = None
        rank_order_weight = getattr(self.hparams, 'rank_order_weight', 0.15)
        if (
            self.ground_truth_distances is not None and self.code_to_idx is not None
            and rank_order_weight > 0
        ):
            from naics_embedder.text_model.loss import LambdaRankLoss

            self.lambdarank_loss_fn = LambdaRankLoss(
                tree_distances=self.ground_truth_distances,
                code_to_idx=self.code_to_idx,
                weight=rank_order_weight,
                sigma=1.0,
                ndcg_k=10,
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

    def _forward_batch(self, batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        '''
        Forward pass for anchor, positive, and negative samples.

        Args:
            batch: Training batch with anchor, positive, and negatives

        Returns:
            List of [anchor_output, positive_output, negative_output]
        '''
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])
        return [anchor_output, positive_output, negative_output]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        '''
        Perform a single training step.

        Args:
            batch: Training batch with anchor, positive, negatives, and metadata
            batch_idx: Batch index

        Returns:
            Total loss for optimization
        '''
        batch_size = batch['batch_size']
        k_negatives = batch['k_negatives']

        # Update curriculum state
        self._update_curriculum_state(batch_idx, batch_size)

        # Forward pass
        anchor_output, positive_output, negative_output = self._forward_batch(batch)
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']
        negative_emb = negative_output['embedding']

        # Log training statistics
        self._log_multilevel_supervision_stats(batch, batch_idx, batch_size)
        false_negative_mask = self._build_false_negative_mask(batch, batch_size)
        self._log_sampling_metadata(batch, batch_size)
        self._log_negative_sample_stats_if_needed(batch, batch_idx)

        # Prepare negatives with hard negative mining
        negative_emb, negative_emb_reshaped = self._prepare_negative_embeddings(
            anchor_output,
            negative_output,
            anchor_emb,
            negative_emb,
            batch_size,
            k_negatives,
            batch_idx,
        )

        # Apply false negative strategy
        false_negative_mask, auxiliary_fn_loss = self._apply_false_negative_strategy_wrapper(
            anchor_emb,
            negative_emb_reshaped,
            false_negative_mask,
        )

        # Compute adaptive margins
        adaptive_margins = self.norm_adaptive_margin(anchor_emb)
        self._log_adaptive_margin_stats(adaptive_margins, batch_idx, batch_size)

        # Compute contrastive loss
        contrastive_loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives,
            false_negative_mask=false_negative_mask,
        )

        if auxiliary_fn_loss is not None:
            contrastive_loss = contrastive_loss + auxiliary_fn_loss

        # Compute MoE load balancing loss
        gate_probs_list, topk_indices_list = self._collect_gate_outputs(
            [anchor_output, positive_output, negative_output]
        )
        self._log_router_diversity(gate_probs_list, batch_size)
        raw_load_balancing_loss = self._compute_load_balancing_loss(
            gate_probs_list,
            topk_indices_list,
            batch_size,
        )

        # Compute auxiliary losses
        hierarchy_loss = self._compute_hierarchy_loss(anchor_emb, positive_emb, batch, batch_size)
        lambdarank_loss = self._compute_lambdarank_loss(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch,
            batch_size,
            k_negatives,
        )
        radius_reg_loss = self._compute_radius_regularization(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
        )

        # Combine losses
        total_loss, scaled_load_balancing_loss = self._combine_loss_terms(
            contrastive_loss,
            raw_load_balancing_loss,
            hierarchy_loss,
            lambdarank_loss,
            radius_reg_loss,
        )

        # Log loss breakdown
        self._log_loss_breakdown(
            contrastive_loss,
            scaled_load_balancing_loss,
            hierarchy_loss,
            lambdarank_loss,
            radius_reg_loss,
            total_loss,
            batch_size,
        )

        return total_loss
