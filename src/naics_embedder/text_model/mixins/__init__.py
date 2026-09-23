# -------------------------------------------------------------------------------------------------
# Mixins for NAICSContrastiveModel
# -------------------------------------------------------------------------------------------------
'''
Mixins module for the NAICSContrastiveModel.

This module provides functional mixins that decompose the model into smaller,
maintainable components:

- DistributedMixin: Global batch sampling utilities for multi-GPU training
- LossMixin: Loss computation methods (hierarchy, structural preference, radius regularization)
- CurriculumMixin: Curriculum learning logic (checked negative selection, pseudo-related candidates)
- LoggingMixin: Logging utilities for training and validation metrics
- ValidationMixin: Validation step and evaluation logic
- OptimizerMixin: Optimizer and scheduler configuration
'''

from naics_embedder.text_model.mixins.curriculum import CurriculumMixin
from naics_embedder.text_model.mixins.distributed import (
    DistributedMixin,
    gather_candidate_entities,
    gather_embeddings_global,
)
from naics_embedder.text_model.mixins.logging import LoggingMixin
from naics_embedder.text_model.mixins.loss import LossMixin
from naics_embedder.text_model.mixins.optimizer import OptimizerMixin
from naics_embedder.text_model.mixins.validation import ValidationMixin

__all__ = [
    'DistributedMixin',
    'gather_candidate_entities',
    'gather_embeddings_global',
    'LossMixin',
    'CurriculumMixin',
    'LoggingMixin',
    'ValidationMixin',
    'OptimizerMixin',
]
