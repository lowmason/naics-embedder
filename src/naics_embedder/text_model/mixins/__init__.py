# -------------------------------------------------------------------------------------------------
# Mixins for NAICSContrastiveModel
# -------------------------------------------------------------------------------------------------
'''
Mixins module for the NAICSContrastiveModel.

The model's three mixins:

- LossMixin: the experts' load-balancing term, under ``moe`` only (R11)
- LoggingMixin: the epoch's health logs (P20)
- OptimizerMixin: AdamW, the warmup, the plateau on the monitor's MRR and the logit scales' clamp
  (P16)
'''

from naics_embedder.text_model.mixins.logging import LoggingMixin
from naics_embedder.text_model.mixins.loss import LossMixin
from naics_embedder.text_model.mixins.optimizer import OUTCOME_MRR, OptimizerMixin

__all__ = [
    'OUTCOME_MRR',
    'LossMixin',
    'LoggingMixin',
    'OptimizerMixin',
]
