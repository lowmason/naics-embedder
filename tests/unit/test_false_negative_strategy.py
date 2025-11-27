'''Unit tests for false negative strategy helper.'''

import torch

from naics_embedder.text_model.false_negative_strategies import apply_false_negative_strategy
from naics_embedder.utils.config import FalseNegativeConfig

def test_eliminate_strategy_leaves_mask_untouched():
    config = FalseNegativeConfig(strategy='eliminate')
    anchors = torch.randn(2, 4)
    negatives = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, True]])

    updated_mask, aux_loss = apply_false_negative_strategy(config, anchors, negatives, mask)

    assert updated_mask is not None
    assert torch.equal(updated_mask, mask)
    assert aux_loss is None

def test_attract_strategy_returns_aux_loss_and_disables_mask():
    config = FalseNegativeConfig(strategy='attract', attraction_weight=0.5, attraction_metric='l2')
    anchors = torch.randn(1, 4)
    negatives = torch.randn(1, 2, 4)
    mask = torch.tensor([[True, False]])

    updated_mask, aux_loss = apply_false_negative_strategy(config, anchors, negatives, mask)

    assert updated_mask is None
    assert aux_loss is not None
    assert aux_loss.item() >= 0
