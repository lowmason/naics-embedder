# -------------------------------------------------------------------------------------------------
# Optimizer Configuration Mixin
# -------------------------------------------------------------------------------------------------
'''
Optimizer configuration mixin for NAICSContrastiveModel.

Provides methods for:
- Optimizer creation and configuration
- Learning rate scheduler setup (warmup + cosine decay, ReduceLROnPlateau)
- Curriculum scheduler initialization
'''

import logging
from typing import Any, Dict

import numpy as np
import torch

from naics_embedder.text_model.curriculum import CurriculumScheduler
from naics_embedder.utils.config import AnnealConfig

logger = logging.getLogger(__name__)

class OptimizerMixin:
    '''
    Mixin providing optimizer and scheduler configuration.

    This mixin expects the following attributes on the class:
    - hparams: hyperparameters with learning_rate, weight_decay, etc.
    - trainer: PyTorch Lightning trainer (optional, for total steps estimation)
    - parameters(): method returning model parameters
    '''

    def configure_optimizers(self) -> Dict[str, Any]:
        '''
        Configure optimizer and learning rate scheduler.

        Issue #4: Optimizer is reset when starting a new curriculum stage.
        This ensures fresh optimizer state for each curriculum stage.
        Issue #13: Add warmup + cosine decay for large training jobs.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration.
        '''
        learning_rate = getattr(self.hparams, 'learning_rate', 2e-4)
        weight_decay = getattr(self.hparams, 'weight_decay', 0.01)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        use_warmup_cosine = getattr(self.hparams, 'use_warmup_cosine', False)

        if use_warmup_cosine:
            return self._configure_warmup_cosine_scheduler(optimizer, learning_rate)
        else:
            return self._configure_reduce_lr_on_plateau(optimizer)

    def _configure_warmup_cosine_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
    ) -> Dict[str, Any]:
        '''
        Configure warmup + cosine annealing scheduler.

        Args:
            optimizer: The optimizer to schedule
            learning_rate: Base learning rate

        Returns:
            Configuration dictionary with optimizer and scheduler
        '''
        warmup_steps = getattr(self.hparams, 'warmup_steps', 500)
        base_lr = learning_rate
        min_lr = 1e-6

        # Capture self in closure for accessing trainer later
        model_self = self

        def lr_lambda(step: int) -> float:
            '''Compute learning rate multiplier for current step.'''
            if step < warmup_steps:
                # Linear warmup: from 0.01 * base_lr to base_lr
                return 0.01 + 0.99 * (step / warmup_steps)
            else:
                # Cosine annealing after warmup
                total_steps = model_self._estimate_total_steps()

                # Cosine annealing: from base_lr to min_lr
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(progress, 1.0)  # Clamp to [0, 1]
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                # Scale from [min_lr/base_lr, 1.0]
                return max(min_lr / base_lr, cosine_factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step-based for warmup and cosine decay
                'frequency': 1,
            },
        }

    def _configure_reduce_lr_on_plateau(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        '''
        Configure ReduceLROnPlateau scheduler for validation-based LR reduction.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Configuration dictionary with optimizer and scheduler
        '''
        scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # Reduce LR by 50%
                patience=3,  # Wait 3 epochs without improvement
                min_lr=1e-6,
            ),
            'monitor':
            'val/contrastive_loss',
            'interval':
            'epoch',
            'frequency':
            1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _estimate_total_steps(self) -> int:
        '''
        Estimate total training steps for scheduler calculation.

        Returns:
            Estimated total number of training steps
        '''
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                return self.trainer.estimated_stepping_batches
            elif hasattr(self.trainer, 'num_training_batches'):
                max_epochs = getattr(self.trainer, 'max_epochs', 15)
                return self.trainer.num_training_batches * max_epochs

        # Fallback: use a large number (will be updated dynamically)
        return 50000

    def on_train_start(self) -> None:
        '''
        Initialize curriculum scheduler and reset optimizer state when starting training.

        Issue #4: Ensures optimizer state is reset for curriculum learning.
        Issue #12: Initialize Structure-Aware Dynamic Curriculum (SADC) scheduler.
        '''
        # Initialize curriculum scheduler
        if hasattr(self, 'trainer') and self.trainer is not None:
            max_epochs = getattr(self.trainer, 'max_epochs', 15)
            anneal_cfg = getattr(self.hparams, 'curriculum_anneal', None)
            if anneal_cfg is not None and not isinstance(anneal_cfg, AnnealConfig):
                anneal_cfg = AnnealConfig(**anneal_cfg)
            self.curriculum_scheduler = CurriculumScheduler(
                max_epochs=max_epochs,
                phase1_end=getattr(self.hparams, 'curriculum_phase1_end', 0.3),
                phase2_end=getattr(self.hparams, 'curriculum_phase2_end', 0.7),
                phase3_end=getattr(self.hparams, 'curriculum_phase3_end', 1.0),
                tree_distance_alpha=getattr(self.hparams, 'tree_distance_alpha', 1.5),
                sibling_distance_threshold=getattr(self.hparams, 'sibling_distance_threshold', 2.0),
                phase_mode=getattr(self.hparams, 'curriculum_phase_mode', 'three_phase'),
                anneal_config=anneal_cfg,
            )
            logger.info('Curriculum scheduler initialized')

        # Reset optimizer state if this is a new curriculum stage
        # This is called by PyTorch Lightning when training starts
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Check if we're resuming from a checkpoint
            if hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path:
                # If resuming same stage, keep optimizer state
                # If starting new stage, optimizer will be recreated fresh
                pass
