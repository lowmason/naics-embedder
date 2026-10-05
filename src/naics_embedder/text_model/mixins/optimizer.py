# -------------------------------------------------------------------------------------------------
# Optimizer Configuration Mixin
# -------------------------------------------------------------------------------------------------
'''
Optimizer mixin for NAICSContrastiveModel: AdamW, a linear warmup and a plateau on the monitor's
MRR (spec 4.4, P16).

- AdamW over two groups: the logit scales take no weight decay (spec 4.1).
- A linear warmup over the first ``warmup_epochs`` epochs, set by hand in ``optimizer_step``.
- ``ReduceLROnPlateau`` on ``val/outcome_mrr``, mode max. It is registered with Lightning, so its
  state is checkpointed, but Lightning never steps it: ``on_train_epoch_end`` steps it by hand
  with the monitor's MRR, before ModelCheckpoint saves, so an exact resume replays it (P16, P30).
- After each optimizer step, both logit scales' θ are clamped in place to their range.
'''

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from naics_embedder.text_model.loss import LogitScale

logger = logging.getLogger(__name__)

# The monitor's MRR: the key it is logged, stepped on and checkpointed under (spec 4.4, P18)
OUTCOME_MRR = 'val/outcome_mrr'

class OptimizerMixin:
    '''
    Mixin providing the optimizer, its schedule and the logit scales' clamp.

    This mixin expects the following attributes on the class:
    - hparams: ``learning_rate``, ``weight_decay``, ``warmup_epochs``, ``lr_plateau_factor`` and
      ``lr_plateau_patience``
    - logit_scale_task, logit_scale_code: the two ``LogitScale`` modules
    - trainer: inside a fit, its ``global_step`` and ``num_training_batches``
    '''

    def _logit_scales(self) -> Tuple[LogitScale, LogitScale]:
        return self.logit_scale_task, self.logit_scale_code

    def configure_optimizers(self) -> Dict[str, Any]:
        '''
        AdamW over two groups and the plateau, registered non-strict on ``val/outcome_mrr``.

        Returns:
            The optimizer and the plateau's scheduler config.
        '''

        scales = [scale.log_scale for scale in self._logit_scales()]
        scale_ids = {id(scale) for scale in scales}
        trainable = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in scale_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    'params': trainable,
                    'weight_decay': self.hparams.weight_decay
                },
                # The logit scales take no weight decay (spec 4.1)
                {
                    'params': scales,
                    'weight_decay': 0.0
                },
            ],
            lr=self.hparams.learning_rate,
        )
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.hparams.lr_plateau_factor,
            patience=self.hparams.lr_plateau_patience,
            threshold=0.0,
        )
        # Non-strict, so Lightning's own lookup of the monitor never raises (P30)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': plateau,
                'monitor': OUTCOME_MRR,
                'interval': 'epoch',
                'strict': False,
            },
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Any,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        '''
        One optimizer step, at the warmup's rate while it lasts, then the scales' clamp (P16).

        While ``global_step`` < W · S, for W warmup epochs of S steps, every group's rate is
        base · (global_step + 1) / (W · S), so the warmup's last step runs at the base rate. After
        it, the rate is the plateau's alone.

        Raises:
            ValueError: If the warmup has no step count to run over: the train loader is unsized.
        '''

        warmup_epochs = self.hparams.warmup_epochs
        if warmup_epochs > 0:
            steps = self.trainer.num_training_batches
            if not math.isfinite(steps):
                raise ValueError(
                    'the learning-rate warmup runs over an epoch of steps, but the train loader '
                    'has no length'
                )
            warmup_steps = warmup_epochs * int(steps)
            step = self.trainer.global_step
            if step < warmup_steps:
                rate = self.hparams.learning_rate * ((step + 1) / warmup_steps)
                for group in optimizer.param_groups:
                    group['lr'] = rate
        optimizer.step(closure=optimizer_closure)
        self._clamp_logit_scales()

    def _clamp_logit_scales(self) -> None:
        '''
        Clamp both logit scales' θ in place to [log low, log high], as CLIP does.

        The forward clamp passes no gradient beyond the range, and the scales take no weight
        decay, so a θ that momentum carried past a bound would otherwise stay there for the rest
        of the run. At the bound the forward clamp passes gradient again.
        '''

        with torch.no_grad():
            for scale in self._logit_scales():
                scale.log_scale.clamp_(math.log(scale.low), math.log(scale.high))

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        '''
        Lightning steps no plateau: ``on_train_epoch_end`` steps it with the monitor's MRR,
        before ModelCheckpoint saves the epoch (P16). Any other scheduler steps as Lightning's.
        '''

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return
        super().lr_scheduler_step(scheduler, metric)

    def _step_plateau(self, mrr: torch.Tensor) -> None:
        '''Step the plateau on the epoch's MRR: the value logged as ``val/outcome_mrr`` (P18).'''

        plateau = self.lr_schedulers()
        if not isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise RuntimeError(
                f'the model steps one ReduceLROnPlateau on {OUTCOME_MRR}, not {plateau!r}'
            )
        plateau.step(mrr)
