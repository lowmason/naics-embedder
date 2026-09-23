'''Unit tests for false negative strategy helper.'''

import pytest
import torch

from naics_embedder.text_model.false_negative_strategies import apply_false_negative_strategy
from naics_embedder.text_model.loss import HyperbolicInfoNCELoss, effective_false_negative_mask
from naics_embedder.utils.config import FalseNegativeConfig

def _no_exclusions(mask: torch.Tensor) -> dict:
    return {
        'explicit_exclusion_mask': torch.zeros_like(mask),
        'valid_mask': torch.ones_like(mask),
    }

def test_eliminate_strategy_leaves_mask_untouched():
    config = FalseNegativeConfig(strategy='eliminate')
    anchors = torch.randn(2, 4)
    negatives = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, True]])

    updated_mask, aux_loss = apply_false_negative_strategy(
        config, anchors, negatives, mask, **_no_exclusions(mask)
    )

    assert updated_mask is not None
    assert torch.equal(updated_mask, mask)
    assert aux_loss is None

def test_attract_strategy_returns_aux_loss_and_disables_mask():
    config = FalseNegativeConfig(strategy='attract', attraction_weight=0.5, attraction_metric='l2')
    anchors = torch.randn(1, 4)
    negatives = torch.randn(1, 2, 4)
    mask = torch.tensor([[True, False]])

    updated_mask, aux_loss = apply_false_negative_strategy(
        config, anchors, negatives, mask, **_no_exclusions(mask)
    )

    assert updated_mask is None
    assert aux_loss is not None
    assert aux_loss.item() >= 0

# -------------------------------------------------------------------------------------------------
# Explicit exclusions are never false negatives
# -------------------------------------------------------------------------------------------------

def test_explicit_exclusion_overrides_pseudo_related_mask():
    pseudo_related = torch.tensor([[True, True, False]])
    explicit = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])

    effective = effective_false_negative_mask(pseudo_related, explicit, valid)

    assert effective.tolist() == [[False, True, False]]

def test_attraction_uses_only_valid_non_exclusion_pairs():
    anchor = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    pseudo_related = torch.tensor([[True, True, True]])
    explicit = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])

    updated_mask, attraction = apply_false_negative_strategy(
        FalseNegativeConfig(strategy='attract', attraction_metric='cosine'),
        anchor,
        negatives,
        pseudo_related,
        explicit_exclusion_mask=explicit,
        valid_mask=valid,
    )

    assert updated_mask is None
    assert attraction is not None and torch.isfinite(attraction)

def test_attraction_never_pulls_an_explicit_exclusion():
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    negatives = torch.tensor([[[0.0, 1.0], [0.5, 0.5]]], requires_grad=True)

    _, attraction = apply_false_negative_strategy(
        FalseNegativeConfig(strategy='attract', attraction_metric='l2'),
        anchor,
        negatives,
        torch.tensor([[True, True]]),
        explicit_exclusion_mask=torch.tensor([[True, False]]),
        valid_mask=torch.tensor([[True, True]]),
    )
    attraction.backward()

    assert torch.count_nonzero(negatives.grad[0, 0]) == 0
    assert torch.count_nonzero(negatives.grad[0, 1]) > 0

def test_hybrid_strategy_returns_the_exclusion_cleared_mask():
    mask = torch.tensor([[True, True]])

    updated_mask, attraction = apply_false_negative_strategy(
        FalseNegativeConfig(strategy='hybrid'),
        torch.randn(1, 3),
        torch.randn(1, 2, 3),
        mask,
        explicit_exclusion_mask=torch.tensor([[True, False]]),
        valid_mask=torch.tensor([[True, True]]),
    )

    assert updated_mask.tolist() == [[False, True]]
    assert attraction is not None

def test_mask_shapes_must_align():
    with pytest.raises(ValueError, match='align'):
        apply_false_negative_strategy(
            FalseNegativeConfig(strategy='eliminate'),
            torch.randn(1, 3),
            torch.randn(1, 2, 3),
            torch.tensor([[True, True]]),
            explicit_exclusion_mask=torch.tensor([[True, False, False]]),
            valid_mask=torch.tensor([[True, True]]),
        )

# -------------------------------------------------------------------------------------------------
# Contrastive denominator: exclusions stay repulsive, padding never contributes
# -------------------------------------------------------------------------------------------------

def _lorentz_point(value: float) -> torch.Tensor:
    spatial = torch.tensor([value])
    return torch.cat([torch.sqrt(1.0 + spatial.square()), spatial])

def test_explicit_pseudo_related_negative_remains_in_contrastive_denominator():
    loss_fn = HyperbolicInfoNCELoss(embedding_dim=1, temperature=0.5, curvature=1.0)
    anchor = _lorentz_point(0.0).unsqueeze(0)
    positive = _lorentz_point(0.1).unsqueeze(0)
    common = {
        'valid_mask': torch.tensor([[True]]),
        'is_explicit_exclusion': torch.tensor([[True]]),
        'pseudo_related_mask': torch.tensor([[True]]),
    }
    near_loss = loss_fn(
        anchor,
        positive,
        _lorentz_point(0.2).view(1, 1, -1),
        **common,
    )
    far_loss = loss_fn(
        anchor,
        positive,
        _lorentz_point(2.0).view(1, 1, -1),
        **common,
    )

    assert torch.isfinite(near_loss) and torch.isfinite(far_loss)
    assert not torch.allclose(near_loss, far_loss)

def test_invalid_padding_cannot_change_contrastive_loss_or_gradients():
    loss_fn = HyperbolicInfoNCELoss(embedding_dim=1, temperature=0.5, curvature=1.0)

    def run(invalid_value: float):
        anchor = _lorentz_point(0.0).unsqueeze(0).requires_grad_()
        positive = _lorentz_point(0.1).unsqueeze(0).requires_grad_()
        negatives = torch.stack([_lorentz_point(0.8),
                                 _lorentz_point(invalid_value)]).unsqueeze(0).requires_grad_()
        loss = loss_fn(
            anchor,
            positive,
            negatives,
            valid_mask=torch.tensor([[True, False]]),
            is_explicit_exclusion=torch.tensor([[False, False]]),
            pseudo_related_mask=None,
        )
        gradients = torch.autograd.grad(loss, (anchor, positive, negatives))
        return loss.detach(), tuple(gradient.detach() for gradient in gradients)

    first_loss, first_gradients = run(2.0)
    second_loss, second_gradients = run(20.0)

    assert torch.allclose(first_loss, second_loss)
    assert torch.allclose(first_gradients[0], second_gradients[0])
    assert torch.allclose(first_gradients[1], second_gradients[1])
    assert torch.allclose(first_gradients[2][:, 0], second_gradients[2][:, 0])
    assert torch.count_nonzero(first_gradients[2][:, 1]) == 0
    assert torch.count_nonzero(second_gradients[2][:, 1]) == 0
