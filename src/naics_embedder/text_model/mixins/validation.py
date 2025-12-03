# -------------------------------------------------------------------------------------------------
# Validation Mixin
# -------------------------------------------------------------------------------------------------
'''
Validation mixin for NAICSContrastiveModel.

Provides methods for:
- Validation step
- Validation epoch end with comprehensive metric computation
- Evaluation metrics JSON logging
'''

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

class ValidationMixin:
    '''
    Mixin providing validation functionality.

    This mixin expects the following attributes on the class:
    - device: torch.device
    - hparams: hyperparameters
    - loss_fn: contrastive loss function
    - embedding_eval: EmbeddingEvaluator
    - embedding_stats: EmbeddingStatistics
    - hierarchy_metrics: HierarchyMetrics
    - ground_truth_distances: Optional[torch.Tensor]
    - code_to_idx: Optional[Dict[str, int]]
    - validation_embeddings: Dict[str, torch.Tensor]
    - validation_codes: List[str]
    - evaluation_metrics_history: List[Dict]
    - curriculum_scheduler: Optional[CurriculumScheduler]
    - current_epoch: int
    - trainer: PyTorch Lightning trainer
    - logger: PyTorch Lightning logger
    - naics_hierarchy: Optional NAICS hierarchy
    '''

    def _get_metrics_file_path(self) -> Optional[Path]:
        '''Get the path to save evaluation metrics JSON file.'''
        if self.logger is None:
            return None

        # Try to get log directory from logger
        if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
            return Path(self.logger.log_dir) / 'evaluation_metrics.json'
        elif hasattr(self.logger, 'save_dir'):
            # Fallback for TensorBoardLogger
            save_dir = getattr(self.logger, 'save_dir', None)
            if save_dir:
                version = getattr(self.logger, 'version', 0)
                name = getattr(self.logger, 'name', 'default')
                return Path(save_dir) / name / f'version_{version}' / 'evaluation_metrics.json'

        return None

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        '''
        Perform a single validation step.

        Args:
            batch: Validation batch with anchor, positive, and negatives
            batch_idx: Batch index

        Returns:
            Validation contrastive loss
        '''
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])

        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']
        negative_emb = negative_output['embedding']

        batch_size = batch['batch_size']
        k_negatives = batch['k_negatives']

        contrastive_loss = self.loss_fn(
            anchor_emb, positive_emb, negative_emb, batch_size, k_negatives
        )

        self.log(
            'val/contrastive_loss',
            contrastive_loss.detach(),
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        if 'anchor_code' in batch:
            for i, code in enumerate(batch['anchor_code']):
                if code not in self.validation_embeddings:
                    # Store hyperbolic embeddings (Lorentz model)
                    self.validation_embeddings[code] = anchor_emb[i].detach().cpu()
                    self.validation_codes.append(code)

        return contrastive_loss

    def on_validation_epoch_end(self) -> None:
        '''
        Compute evaluation metrics and trigger pseudo-label update based on the curriculum schedule.
        '''
        eval_every_n_epochs = getattr(self.hparams, 'eval_every_n_epochs', 1)
        if self.current_epoch % eval_every_n_epochs != 0:
            return

        if not self.validation_embeddings or self.ground_truth_distances is None:
            logger.warning('Skipping evaluation: missing embeddings or ground truth distances')
            return

        try:
            logger.info(f'\nRunning evaluation metrics (epoch {self.current_epoch})...')

            codes = sorted(self.validation_embeddings.keys())
            embeddings = torch.stack([self.validation_embeddings[code]
                                      for code in codes]).to(self.device)

            eval_sample_size = getattr(self.hparams, 'eval_sample_size', 500)
            if len(codes) > eval_sample_size:
                indices = torch.randperm(len(codes))[:eval_sample_size]
                embeddings = embeddings[indices]
                codes = [codes[i] for i in indices]

            if self.code_to_idx is None:
                logger.warning('code_to_idx is None, cannot evaluate')
                return

            code_indices = [self.code_to_idx[code] for code in codes if code in self.code_to_idx]
            if len(code_indices) < 2:
                logger.warning('Not enough codes in ground truth for evaluation')
                return

            gt_dists = self.ground_truth_distances[code_indices][:, code_indices].to(self.device)

            num_samples = len(embeddings)
            with torch.no_grad():
                epoch_metrics = self._compute_validation_metrics(
                    embeddings, codes, gt_dists, num_samples
                )

            # Handle clustering updates
            self._handle_clustering_update()

            # Save metrics to JSON
            self._save_evaluation_metrics(epoch_metrics)

        except Exception as e:
            logger.error(f'Error during evaluation: {e}', exc_info=True)

        finally:
            self.validation_embeddings.clear()
            self.validation_codes.clear()

    def _compute_validation_metrics(
        self,
        embeddings: torch.Tensor,
        codes: List[str],
        gt_dists: torch.Tensor,
        num_samples: int,
    ) -> Dict[str, Any]:
        '''
        Compute all validation metrics.

        Args:
            embeddings: Validation embeddings
            codes: NAICS codes for embeddings
            gt_dists: Ground truth distance matrix
            num_samples: Number of samples

        Returns:
            Dictionary of all computed metrics
        '''
        from naics_embedder.text_model.hyperbolic import (
            check_lorentz_manifold_validity,
            log_hyperbolic_diagnostics,
        )

        radius_metrics: Dict[str, float] = {}
        retrieval_metrics: Dict[str, float] = {}

        # Check manifold validity and log diagnostics
        curvature = getattr(self.hparams, 'curvature', 1.0)
        is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
            embeddings, curvature=curvature
        )

        # Log hyperbolic diagnostics
        diagnostics = log_hyperbolic_diagnostics(
            embeddings,
            curvature=curvature,
            level_labels=None,
            logger_instance=logger,
        )

        # Log manifold validity metrics
        self.log('val/manifold_valid', float(is_valid), batch_size=num_samples, sync_dist=True)
        self.log(
            'val/lorentz_norm_mean',
            diagnostics['lorentz_norm_mean'],
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/lorentz_norm_violation_max',
            self._to_python_scalar(diagnostics['violation_max']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/hyperbolic_radius_mean',
            self._to_python_scalar(diagnostics['radius_mean']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/hyperbolic_radius_std',
            self._to_python_scalar(diagnostics['radius_std']),
            batch_size=num_samples,
            sync_dist=True,
        )

        # Warn if manifold constraint is violated
        if not is_valid:
            logger.warning(
                f'⚠️  Hyperbolic embeddings violate manifold constraint! '
                f'Max violation: {diagnostics["violation_max"]:.6e}'
            )

        # Compute statistics on Euclidean projection for compatibility
        embeddings_euc = embeddings[:, 1:]  # Remove time coordinate
        stats = self.embedding_stats.compute_statistics(embeddings_euc)
        self.log(
            'val/mean_norm',
            self._to_python_scalar(stats['mean_norm']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/std_norm',
            self._to_python_scalar(stats['std_norm']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/mean_pairwise_distance',
            self._to_python_scalar(stats['mean_pairwise_distance']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/std_pairwise_distance',
            self._to_python_scalar(stats['std_pairwise_distance']),
            batch_size=num_samples,
            sync_dist=True,
        )

        # Check collapse on Euclidean projection
        collapse = self.embedding_stats.check_collapse(embeddings_euc)
        self.log(
            'val/variance_collapsed',
            self._to_python_scalar(collapse['variance_collapsed']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/norm_collapsed',
            self._to_python_scalar(collapse['norm_collapsed']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/distance_collapsed',
            self._to_python_scalar(collapse['distance_collapsed']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/mean_variance',
            self._to_python_scalar(collapse['mean_variance']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/norm_cv',
            self._to_python_scalar(collapse['norm_cv']),
            prog_bar=True,
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/distance_cv',
            self._to_python_scalar(collapse['distance_cv']),
            prog_bar=True,
            batch_size=num_samples,
            sync_dist=True,
        )

        # Use Lorentzian distances for hyperbolic embeddings
        emb_dists = self.embedding_eval.compute_pairwise_distances(
            embeddings, metric='lorentz', curvature=curvature
        )

        radius_metrics = self._log_radius_structure_metrics(embeddings, codes, num_samples)
        retrieval_metrics = self._log_hierarchy_retrieval_metrics(
            emb_dists,
            codes,
            num_samples,
        )

        cophenetic_result = self.hierarchy_metrics.cophenetic_correlation(emb_dists, gt_dists)
        self.log(
            'val/cophenetic_correlation',
            self._to_python_scalar(cophenetic_result['correlation']),
            prog_bar=True,
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/cophenetic_n_pairs',
            self._to_python_scalar(cophenetic_result['n_pairs']),
            batch_size=num_samples,
            sync_dist=True,
        )

        # Compute NDCG@k for ranking evaluation
        ndcg_result = self.hierarchy_metrics.ndcg_ranking(emb_dists, gt_dists, k_values=[5, 10, 20])
        for k in [5, 10, 20]:
            self.log(
                f'val/ndcg@{k}',
                self._to_python_scalar(ndcg_result[f'ndcg@{k}']),
                batch_size=num_samples,
                sync_dist=True,
            )
            self.log(
                f'val/ndcg@{k}_n_queries',
                self._to_python_scalar(ndcg_result[f'ndcg@{k}_n_queries']),
                batch_size=num_samples,
                sync_dist=True,
            )

        # Compute Spearman for backward compatibility
        spearman_result = self.hierarchy_metrics.spearman_correlation(emb_dists, gt_dists)
        self.log(
            'val/spearman_correlation',
            self._to_python_scalar(spearman_result['correlation']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/spearman_n_pairs',
            self._to_python_scalar(spearman_result['n_pairs']),
            batch_size=num_samples,
            sync_dist=True,
        )

        distortion = self.hierarchy_metrics.distortion(emb_dists, gt_dists)
        self.log(
            'val/mean_distortion',
            self._to_python_scalar(distortion['mean_distortion']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/std_distortion',
            self._to_python_scalar(distortion['std_distortion']),
            batch_size=num_samples,
            sync_dist=True,
        )
        self.log(
            'val/median_distortion',
            self._to_python_scalar(distortion['median_distortion']),
            prog_bar=True,
            batch_size=num_samples,
            sync_dist=True,
        )

        logger.info(
            f'Correlation metrics: \n'
            f'  • Hierarchy preservation: cophenetic={cophenetic_result["correlation"]:.4f} '
            f'({cophenetic_result["n_pairs"]} pairs)\n'
            f'  • Ranking quality: NDCG@5={ndcg_result["ndcg@5"]:.4f}, '
            f'NDCG@10={ndcg_result["ndcg@10"]:.4f}, NDCG@20={ndcg_result["ndcg@20"]:.4f}\n'
        )
        logger.info(
            f'Collapse detection metrics: \n'
            f'  • Norm CV: {collapse["norm_cv"]:.4f}\n'
            f'  • Distance CV: {collapse["distance_cv"]:.4f}\n'
            f'  • Collapse: {collapse["any_collapse"]}\n'
        )

        # Collect all evaluation metrics for JSON logging
        train_loss = None
        train_contrastive_loss = None
        val_loss = None

        if hasattr(self.trainer, 'callback_metrics'):
            train_loss = self.trainer.callback_metrics.get('train/total_loss', None)
            train_contrastive_loss = self.trainer.callback_metrics.get(
                'train/contrastive_loss', None
            )
            val_loss = self.trainer.callback_metrics.get('val/contrastive_loss', None)

        epoch_metrics = {
            'epoch':
            self.current_epoch,
            # Training metrics
            'train_loss': (self._to_python_scalar(train_loss) if train_loss is not None else None),
            'train_contrastive_loss': (
                self._to_python_scalar(train_contrastive_loss)
                if train_contrastive_loss is not None else None
            ),
            # Validation metrics
            'val_loss':
            self._to_python_scalar(val_loss) if val_loss is not None else None,
            # Hyperbolic metrics
            'hyperbolic_radius_mean':
            self._to_python_scalar(diagnostics['radius_mean']),
            'hyperbolic_radius_std':
            self._to_python_scalar(diagnostics['radius_std']),
            'lorentz_norm_mean':
            self._to_python_scalar(diagnostics['lorentz_norm_mean']),
            'lorentz_norm_std':
            self._to_python_scalar(diagnostics['lorentz_norm_std']),
            'lorentz_norm_violation_max':
            self._to_python_scalar(diagnostics['violation_max']),
            'manifold_valid':
            bool(is_valid),
            # Embedding statistics
            'mean_norm':
            self._to_python_scalar(stats['mean_norm']),
            'std_norm':
            self._to_python_scalar(stats['std_norm']),
            'mean_pairwise_distance':
            self._to_python_scalar(stats['mean_pairwise_distance']),
            'std_pairwise_distance':
            self._to_python_scalar(stats['std_pairwise_distance']),
            # Collapse detection
            'norm_cv':
            self._to_python_scalar(collapse['norm_cv']),
            'distance_cv':
            self._to_python_scalar(collapse['distance_cv']),
            'collapse_detected':
            bool(collapse['any_collapse']),
            # Hierarchy preservation
            'cophenetic_correlation':
            self._to_python_scalar(cophenetic_result['correlation']),
            'cophenetic_n_pairs':
            int(cophenetic_result['n_pairs']),
            'spearman_correlation':
            self._to_python_scalar(spearman_result['correlation']),
            'spearman_n_pairs':
            int(spearman_result['n_pairs']),
            # Ranking metrics
            'ndcg@5':
            self._to_python_scalar(ndcg_result['ndcg@5']),
            'ndcg@10':
            self._to_python_scalar(ndcg_result['ndcg@10']),
            'ndcg@20':
            self._to_python_scalar(ndcg_result['ndcg@20']),
            'ndcg@5_n_queries':
            int(ndcg_result['ndcg@5_n_queries']),
            'ndcg@10_n_queries':
            int(ndcg_result['ndcg@10_n_queries']),
            'ndcg@20_n_queries':
            int(ndcg_result['ndcg@20_n_queries']),
            # Distortion metrics
            'mean_distortion':
            self._to_python_scalar(distortion['mean_distortion']),
            'std_distortion':
            self._to_python_scalar(distortion['std_distortion']),
            'median_distortion':
            self._to_python_scalar(distortion['median_distortion']),
            # Sample size
            'num_samples':
            int(num_samples),
        }

        if radius_metrics:
            epoch_metrics.update(radius_metrics)
        if retrieval_metrics:
            epoch_metrics.update(retrieval_metrics)

        return epoch_metrics

    def _handle_clustering_update(self) -> None:
        '''Handle pseudo-label clustering updates based on curriculum schedule.'''
        if self.curriculum_scheduler is not None:
            fn_cluster_every_n_epochs = getattr(self.hparams, 'fn_cluster_every_n_epochs', 5)
            should_update = self.curriculum_scheduler.should_update_clustering(
                self.current_epoch, fn_cluster_every_n_epochs
            )
            if should_update:
                self._update_pseudo_labels()
        else:
            # Fallback to old behavior if scheduler not initialized
            fn_curriculum_start_epoch = getattr(self.hparams, 'fn_curriculum_start_epoch', 10)
            fn_cluster_every_n_epochs = getattr(self.hparams, 'fn_cluster_every_n_epochs', 5)
            if self.current_epoch >= fn_curriculum_start_epoch:
                if fn_cluster_every_n_epochs > 0:
                    epochs_since_start = self.current_epoch - fn_curriculum_start_epoch
                    if epochs_since_start % fn_cluster_every_n_epochs == 0:
                        self._update_pseudo_labels()

    def _save_evaluation_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        '''Save evaluation metrics to JSON file.'''
        self.evaluation_metrics_history.append(epoch_metrics)

        metrics_file = self._get_metrics_file_path()
        if metrics_file:
            try:
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_file, 'w') as f:
                    json.dump(self.evaluation_metrics_history, f, indent=2)
                logger.debug(f'Saved evaluation metrics to {metrics_file}')
            except Exception as e:
                logger.warning(f'Failed to save evaluation metrics to JSON: {e}')
