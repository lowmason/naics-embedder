# -------------------------------------------------------------------------------------------------
# Logging Mixin
# -------------------------------------------------------------------------------------------------
'''
Logging mixin for NAICSContrastiveModel: the epoch's health logs (P20).

Each epoch logs the mean of each term over its steps (``loss/task``, ``loss/code_code``,
``loss/radial``, ``loss/total``, and ``loss/load_balancing`` under ``moe``), the two logit scales
(``logit_scale/task``, ``logit_scale/code_code``), and r's mean and SD at each level from the
refreshed code cache (``radius/mean/level_<k>``, ``radius/sd/level_<k>``). Nothing selects on
them (spec 4.4).

A step has two leading sizes, its anchors and its queries, so Lightning's own epoch means would
weight each step by whichever it took for the batch size. The mixin keeps each step's values and
logs their plain mean once, at the epoch's end.
'''

import logging
from typing import TYPE_CHECKING, Dict, List

import torch

if TYPE_CHECKING:
    from naics_embedder.text_model.naics_model import StepLosses

logger = logging.getLogger(__name__)

# The terms whose epoch means are logged; load balancing exists under moe only (R11)
HEALTH_TERMS = ('task', 'code_code', 'radial', 'total', 'load_balancing')

class LoggingMixin:
    '''
    Mixin providing the health logs.

    This mixin expects the following attributes on the class:
    - logit_scale_task, logit_scale_code: the two ``LogitScale`` modules
    - code_cache: the ``CodeCache`` of the last refresh, or None
    - code_levels: each code's level, in codebook order (a buffer)
    - log: ``LightningModule.log``
    '''

    def _reset_health(self) -> None:
        '''Start new epoch means.'''

        self._health_steps: Dict[str, List[torch.Tensor]] = {}

    def _record_health(self, losses: 'StepLosses') -> None:
        '''Keep one step's term values, detached, for the epoch means.'''

        for name in HEALTH_TERMS:
            value = getattr(losses, name)
            if value is not None:
                self._health_steps.setdefault(name, []).append(value.detach())

    def _log_health(self) -> Dict[str, float]:
        '''
        Log the epoch's health values, once each, and start new epoch means.

        A term with no step this epoch is not logged. The SD is the population SD, so a level
        with one code has an SD of 0, not NaN.

        Returns:
            The values logged, by key.
        '''

        values: Dict[str, float] = {}
        for name in HEALTH_TERMS:
            steps = self._health_steps.get(name)
            if steps:
                # .cpu() before the cast: MPS has no float64
                values[f'loss/{name}'] = torch.stack(steps).cpu().to(torch.float64).mean().item()
        with torch.no_grad():
            values['logit_scale/task'] = self.logit_scale_task().item()
            values['logit_scale/code_code'] = self.logit_scale_code().item()
        if self.code_cache is not None:
            radius = self.code_cache.radius.detach().cpu().to(torch.float64)
            levels = self.code_levels.cpu()
            for level in torch.unique(levels).tolist():
                at_level = radius[levels == level]
                values[f'radius/mean/level_{level}'] = at_level.mean().item()
                values[f'radius/sd/level_{level}'] = at_level.std(correction=0).item()
        for name, value in values.items():
            self.log(name, value, on_step=False, on_epoch=True, batch_size=1)
        self._reset_health()
        return values
