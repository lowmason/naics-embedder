'''
Unit tests for loss functions.

Tests hyperbolic contrastive learning losses, hierarchy preservation losses,
and the structural-preference ranking loss used in training.

Req 11's three terms follow spec 4.1 and §6 "Task term" and "Code–code term". The task term sums
probability over each query's targets T within its candidates C. The code–code term is the cross
entropy to softmax(−D* / τ_t) over J_a, every code but the anchor and its unary partner. The radial
term pulls each code to ρ(λ − 1). The logit scales are learned, start at their init and stay in
their range.
'''

import math
from typing import Any, Dict

import pytest
import torch

from naics_embedder.supervision.candidates import SelectedNegativeBatch
from naics_embedder.text_model import loss as terms
from naics_embedder.text_model.hard_negative_mining import NormAdaptiveMargin
from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.text_model.loss import (
    HierarchyPreservationLoss,
    HyperbolicInfoNCELoss,
    StructuralPreferenceLoss,
    structural_preference_from_distances,
)

# -------------------------------------------------------------------------------------------------
# HyperbolicInfoNCELoss Tests
# -------------------------------------------------------------------------------------------------

def _contrastive(loss_fn, anchor, positive, negatives, batch_size, k_negatives, **kwargs):
    '''Call the loss with every negative valid and no explicit exclusion.'''
    negatives = negatives.view(batch_size, k_negatives, -1)
    mask_shape = (batch_size, k_negatives)
    return loss_fn(
        anchor,
        positive,
        negatives,
        valid_mask=torch.ones(mask_shape, dtype=torch.bool, device=anchor.device),
        is_explicit_exclusion=torch.zeros(mask_shape, dtype=torch.bool, device=anchor.device),
        **kwargs,
    )

@pytest.mark.unit
class TestHyperbolicInfoNCELoss:
    '''Test suite for Hyperbolic InfoNCE loss.'''

    @pytest.fixture
    def loss_fn(self):
        '''Create InfoNCE loss function.'''

        return HyperbolicInfoNCELoss(embedding_dim=384, temperature=0.07, curvature=1.0)

    @pytest.fixture
    def sample_triplet(self, test_device, random_seed):
        '''Generate sample anchor, positive, and negative embeddings.'''

        torch.manual_seed(random_seed)

        batch_size = 8
        k_negatives = 16
        dim = 384

        # Create tangent vectors
        anchor_tan = torch.randn(batch_size, dim + 1, device=test_device)
        positive_tan = torch.randn(batch_size, dim + 1, device=test_device)
        negative_tan = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)

        # Project to Lorentz hyperboloid
        anchor = LorentzOps.exp_map_zero(anchor_tan, c=1.0)
        positive = LorentzOps.exp_map_zero(positive_tan, c=1.0)
        negatives = LorentzOps.exp_map_zero(negative_tan, c=1.0)

        return anchor, positive, negatives, batch_size, k_negatives

    def test_loss_is_scalar(self, loss_fn, sample_triplet):
        '''Test that loss returns a scalar value.'''

        loss = _contrastive(loss_fn, *sample_triplet)

        assert loss.dim() == 0, 'Loss should be a scalar'
        assert loss.numel() == 1

    def test_loss_is_positive(self, loss_fn, sample_triplet):
        '''Test that loss is always non-negative.'''

        loss = _contrastive(loss_fn, *sample_triplet)

        assert loss >= 0, 'Loss should be non-negative'

    def test_loss_decreases_with_closer_positive(self, loss_fn, test_device):
        '''Test that loss decreases when positive is closer to anchor.'''

        batch_size = 4
        k_negatives = 8
        dim = 384

        # Anchor
        anchor_tan = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor_tan, c=1.0)

        # Close positive (small perturbation)
        close_positive_tan = anchor_tan + torch.randn_like(anchor_tan) * 0.1
        close_positive = LorentzOps.exp_map_zero(close_positive_tan, c=1.0)

        # Far positive (moderate perturbation to avoid numerical overflow)
        far_positive_tan = anchor_tan + torch.randn_like(anchor_tan) * 2.0
        far_positive = LorentzOps.exp_map_zero(far_positive_tan, c=1.0)

        # Negatives
        negative_tan = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negative_tan, c=1.0)

        # Compute losses
        loss_close = _contrastive(
            loss_fn, anchor, close_positive, negatives, batch_size, k_negatives
        )
        loss_far = _contrastive(loss_fn, anchor, far_positive, negatives, batch_size, k_negatives)

        assert loss_close < loss_far, 'Loss should be lower for closer positives'

    def test_false_negative_masking(self, loss_fn, sample_triplet, test_device):
        '''Test that pseudo-related masking of non-exclusion negatives changes the loss.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        # Loss without masking
        loss_no_mask = _contrastive(loss_fn, *sample_triplet)

        # Loss with masking (mask out some negatives)
        pseudo_related = torch.zeros(batch_size, k_negatives, dtype=torch.bool, device=test_device)
        pseudo_related[:, :4] = True  # Mask first 4 negatives for each anchor

        loss_with_mask = _contrastive(loss_fn, *sample_triplet, pseudo_related_mask=pseudo_related)

        # Loss with masking should be different (typically lower)
        assert loss_with_mask != loss_no_mask

    @pytest.mark.parametrize('temperature', [0.01, 0.05, 0.1, 0.5])
    def test_temperature_effect(self, sample_triplet, temperature):
        '''Test that temperature scaling affects loss magnitude.'''

        loss_fn = HyperbolicInfoNCELoss(embedding_dim=384, temperature=temperature, curvature=1.0)

        loss = _contrastive(loss_fn, *sample_triplet)

        # Loss should be computable for all valid temperatures
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, loss_fn, sample_triplet):
        '''Test that gradients flow through the loss.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        # Enable gradients
        anchor.requires_grad_(True)
        positive.requires_grad_(True)
        negatives.requires_grad_(True)

        loss = _contrastive(loss_fn, anchor, positive, negatives, batch_size, k_negatives)
        loss.backward()

        # Check that gradients exist and are non-zero
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None
        assert torch.any(anchor.grad != 0)

    def test_adaptive_margin_increases_contrastive_pressure(self, sample_triplet):
        '''Adaptive margins should make negatives harder (higher loss).'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        loss_fn = HyperbolicInfoNCELoss(embedding_dim=384, temperature=0.07, curvature=1.0)

        adaptive_margins = torch.full((batch_size, ), 0.5, device=anchor.device)

        loss_nomargin = _contrastive(loss_fn, *sample_triplet)
        loss_margin = _contrastive(loss_fn, *sample_triplet, adaptive_margins=adaptive_margins)

        assert loss_margin > loss_nomargin

    def test_negatives_must_be_batched(self, loss_fn, sample_triplet):
        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        with pytest.raises(ValueError, match='batch, selected'):
            loss_fn(
                anchor,
                positive,
                negatives,
                valid_mask=torch.ones((batch_size, k_negatives), dtype=torch.bool),
                is_explicit_exclusion=torch.zeros((batch_size, k_negatives), dtype=torch.bool),
            )

    def test_rows_without_eligible_negatives_keep_gradients_finite(self, loss_fn, sample_triplet):
        anchor, positive, negatives, batch_size, k_negatives = sample_triplet
        negatives = negatives.view(batch_size, k_negatives, -1).clone().requires_grad_()
        valid = torch.ones((batch_size, k_negatives), dtype=torch.bool)
        valid[0] = False

        loss = loss_fn(
            anchor,
            positive,
            negatives,
            valid_mask=valid,
            is_explicit_exclusion=torch.zeros_like(valid),
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(negatives.grad).all()
        assert torch.count_nonzero(negatives.grad[0]) == 0

class TestNormAdaptiveMargin:
    '''Tests for norm-adaptive margin computation.'''

    def test_margin_decays_with_radius(self, test_device):
        '''Margin should shrink as Lorentz norm grows.'''

        miner = NormAdaptiveMargin(base_margin=1.0, curvature=1.0).to(test_device)

        # Small norm (near origin)
        anchor_small = LorentzOps.exp_map_zero(torch.zeros(2, 385, device=test_device), c=1.0)

        # Larger norm via scaled tangent
        anchor_large = LorentzOps.exp_map_zero(torch.ones(2, 385, device=test_device) * 2.0, c=1.0)

        margin_small = miner(anchor_small)
        margin_large = miner(anchor_large)

        assert torch.all(margin_small > margin_large)

# -------------------------------------------------------------------------------------------------
# HierarchyPreservationLoss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyPreservationLoss:
    '''Test suite for Hierarchy Preservation loss.'''

    @pytest.fixture
    def sample_tree_distances(self, test_device):
        '''Create sample tree distance matrix.'''

        # Simple 4-node tree with known distances
        distances = torch.tensor(
            [
                [0.0, 0.5, 1.5, 2.5],
                [0.5, 0.0, 0.5, 1.5],
                [1.5, 0.5, 0.0, 0.5],
                [2.5, 1.5, 0.5, 0.0],
            ],
            device=test_device,
        )

        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
        }

        return distances, code_to_idx

    @pytest.fixture
    def hierarchy_loss_fn(self, sample_tree_distances):
        '''Create hierarchy preservation loss function.'''

        distances, code_to_idx = sample_tree_distances
        return HierarchyPreservationLoss(
            tree_distances=distances, code_to_idx=code_to_idx, weight=0.1, min_distance=0.1
        )

    def test_loss_is_scalar(self, hierarchy_loss_fn, test_device):
        '''Test that hierarchy loss returns a scalar.'''

        # Create sample embeddings
        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss.dim() == 0
        assert loss.numel() == 1

    def test_loss_is_non_negative(self, hierarchy_loss_fn, test_device):
        '''Test that hierarchy loss is non-negative.'''

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss >= 0

    def test_loss_zero_for_insufficient_codes(self, hierarchy_loss_fn, test_device):
        '''Test that loss is zero when there are fewer than 2 valid codes.'''

        embeddings = torch.randn(2, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        # Only one valid code
        codes = ['31', 'invalid_code']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss == 0.0

    def test_loss_respects_weight_parameter(self, sample_tree_distances, test_device):
        '''Test that loss scales with weight parameter.'''

        distances, code_to_idx = sample_tree_distances

        loss_fn_1 = HierarchyPreservationLoss(distances, code_to_idx, weight=0.1)
        loss_fn_2 = HierarchyPreservationLoss(distances, code_to_idx, weight=0.2)

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)
        codes = ['31', '311', '3111', '31111']

        loss_1 = loss_fn_1(embeddings, codes, LorentzOps.lorentz_distance)
        loss_2 = loss_fn_2(embeddings, codes, LorentzOps.lorentz_distance)

        # loss_2 should be approximately 2x loss_1
        assert torch.allclose(loss_2, 2 * loss_1, rtol=0.01)

# -------------------------------------------------------------------------------------------------
# Structural preference (replaces LambdaRank): direct gradient-contract tests
# -------------------------------------------------------------------------------------------------

def test_structural_preference_gradient_corrects_an_inversion():
    learned = torch.tensor([[4.0, 1.0]], requires_grad=True)
    structural = torch.tensor([[1.0, 3.0]])
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=structural,
        candidate_code_ids=torch.tensor([[11, 12]]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.zeros((1, 2), dtype=torch.bool),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
    )

    loss.backward()

    assert learned.grad[0, 0] > 0
    assert learned.grad[0, 1] < 0

def test_structural_preference_is_lower_for_correct_order():
    kwargs = {
        'structural_distances': torch.tensor([[1.0, 3.0]]),
        'candidate_code_ids': torch.tensor([[11, 12]]),
        'anchor_code_ids': torch.tensor([10]),
        'is_explicit_exclusion': torch.zeros((1, 2), dtype=torch.bool),
        'valid_mask': torch.ones((1, 2), dtype=torch.bool),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    correct = structural_preference_from_distances(
        learned_distances=torch.tensor([[1.0, 4.0]]), **kwargs
    )
    inverted = structural_preference_from_distances(
        learned_distances=torch.tensor([[4.0, 1.0]]), **kwargs
    )

    assert correct < inverted

@pytest.mark.parametrize(
    ('structural', 'explicit', 'valid', 'codes'),
    [
        ([2.0, 2.0], [False, False], [True, True], [11, 12]),
        ([1.0, 3.0], [True, False], [True, True], [11, 12]),
        ([1.0, 3.0], [False, False], [True, False], [11, 12]),
        ([1.0, 3.0], [False, False], [True, True], [10, 12]),
        ([1.0, 3.0], [False, False], [True, True], [11, 11]),
    ],
)
def test_structural_preference_returns_differentiable_zero_when_fully_masked(
    structural, explicit, valid, codes
):
    learned = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=torch.tensor([structural]),
        candidate_code_ids=torch.tensor([codes]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.tensor([explicit]),
        valid_mask=torch.tensor([valid]),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.equal(learned.grad, torch.zeros_like(learned))

def test_structural_preference_detaches_importance_weights():
    learned = torch.tensor([[3.0, 1.0]], requires_grad=True)
    weights = torch.tensor([[2.0]], requires_grad=True)
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=torch.tensor([[1.0, 3.0]]),
        candidate_code_ids=torch.tensor([[11, 12]]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.zeros((1, 2), dtype=torch.bool),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
        pair_weights=weights,
    )
    loss.backward()

    assert weights.grad is None

def test_structural_preference_is_invariant_to_joint_candidate_permutation():
    kwargs = {
        'learned_distances': torch.tensor([[3.0, 1.0, 2.0]]),
        'structural_distances': torch.tensor([[1.0, 3.0, 5.0]]),
        'candidate_code_ids': torch.tensor([[11, 12, 13]]),
        'anchor_code_ids': torch.tensor([10]),
        'is_explicit_exclusion': torch.tensor([[False, False, False]]),
        'valid_mask': torch.tensor([[True, True, True]]),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    original = structural_preference_from_distances(**kwargs)
    permutation = torch.tensor([2, 0, 1])
    permuted = structural_preference_from_distances(
        **{
            **kwargs,
            'learned_distances': kwargs['learned_distances'][:, permutation],
            'structural_distances': kwargs['structural_distances'][:, permutation],
            'candidate_code_ids': kwargs['candidate_code_ids'][:, permutation],
            'is_explicit_exclusion': kwargs['is_explicit_exclusion'][:, permutation],
            'valid_mask': kwargs['valid_mask'][:, permutation],
        }
    )

    assert torch.allclose(original, permuted)

def test_structural_preference_normalizes_each_anchor_before_batch_mean():
    kwargs = {
        'learned_distances': torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 9.0]]),
        'structural_distances': torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 7.0]]),
        'candidate_code_ids': torch.tensor([[11, 12, 13], [21, 22, 23]]),
        'anchor_code_ids': torch.tensor([10, 20]),
        'is_explicit_exclusion': torch.zeros((2, 3), dtype=torch.bool),
        'valid_mask': torch.tensor([[True, True, True], [True, True, False]]),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    combined = structural_preference_from_distances(**kwargs)
    individual = []
    for row in range(2):
        individual.append(
            structural_preference_from_distances(
                **{
                    key: value[row:row + 1] if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items()
                }
            )
        )

    assert torch.allclose(combined, torch.stack(individual).mean())

def test_structural_preference_rejects_invalid_hyperparameters():
    kwargs = {
        'learned_distances': torch.tensor([[1.0, 2.0]]),
        'structural_distances': torch.tensor([[1.0, 3.0]]),
        'candidate_code_ids': torch.tensor([[11, 12]]),
        'anchor_code_ids': torch.tensor([10]),
        'is_explicit_exclusion': torch.zeros((1, 2), dtype=torch.bool),
        'valid_mask': torch.ones((1, 2), dtype=torch.bool),
        'tie_tolerance': 1e-6,
    }

    with pytest.raises(ValueError, match='temperature'):
        structural_preference_from_distances(**kwargs, margin=0.2, temperature=0.0)
    with pytest.raises(ValueError, match='margin'):
        structural_preference_from_distances(**kwargs, margin=-0.1, temperature=1.0)

def _lorentz_row(values: list[float]) -> torch.Tensor:
    spatial = torch.tensor(values).unsqueeze(1)
    return torch.cat([torch.sqrt(1.0 + spatial.square()), spatial], dim=1)

def test_structural_preference_module_never_compares_an_exclusion():
    # The positive (structure 0.5) and two negatives; the structurally closest negative is an
    # explicit exclusion and must not participate in any comparison.
    selected_embedding = _lorentz_row([0.2, 3.0]).unsqueeze(0).requires_grad_()
    selected = SelectedNegativeBatch(
        candidate_uid=torch.tensor([[[0, 0, 0], [0, 0, 1]]]),
        code_id=torch.tensor([[12, 13]]),
        embedding=selected_embedding,
        structural_distance=torch.tensor([[1.0, 4.0]]),
        structural_relation_id=torch.tensor([[2, 7]], dtype=torch.int16),
        anchor_excludes_candidate=torch.tensor([[True, False]]),
        candidate_excludes_anchor=torch.tensor([[False, False]]),
        is_explicit_exclusion=torch.tensor([[True, False]]),
        semantic_target_id=torch.tensor([[2, 0]], dtype=torch.int8),
        semantic_source_id=torch.tensor([[2, 0]], dtype=torch.int8),
        sampling_role_id=torch.full((1, 2), 2, dtype=torch.int8),
        sampling_provenance_id=torch.full((1, 2), 1, dtype=torch.int8),
        relation_margin=torch.tensor([[1.0, 6.0]]),
        distance_margin=torch.tensor([[0.5, 3.5]]),
        router_gate_probs=None,
        valid_mask=torch.tensor([[True, True]]),
        selection_scores=torch.zeros((1, 2)),
        selection_reasons=torch.ones((1, 2), dtype=torch.int8),
        runtime_fields={},
    )
    loss_fn = StructuralPreferenceLoss(
        curvature=1.0, margin=0.1, temperature=1.0, tie_tolerance=1e-6, weight=0.35
    )

    loss = loss_fn(
        anchor_emb=_lorentz_row([0.0]),
        positive_emb=_lorentz_row([0.1]),
        anchor_code_id=torch.tensor([10]),
        positive_code_id=torch.tensor([11]),
        positive_structural_distance=torch.tensor([0.5]),
        selected=selected,
    )
    loss.backward()

    assert torch.isfinite(loss) and loss > 0
    assert torch.count_nonzero(selected_embedding.grad[0, 0]) == 0
    assert torch.count_nonzero(selected_embedding.grad[0, 1]) > 0

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLossIntegration:
    '''Integration tests combining multiple loss functions.'''

    def test_combined_losses(self, test_device):
        '''Test that multiple losses can be computed and combined.'''
        # Setup
        batch_size = 4
        k_negatives = 8
        dim = 384

        # Create embeddings
        anchor = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor, c=1.0)

        positive = torch.randn(batch_size, dim + 1, device=test_device)
        positive = LorentzOps.exp_map_zero(positive, c=1.0)

        negatives = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negatives, c=1.0)

        # InfoNCE loss
        infonce_loss_fn = HyperbolicInfoNCELoss(embedding_dim=dim, temperature=0.07, curvature=1.0)
        infonce_loss = _contrastive(
            infonce_loss_fn, anchor, positive, negatives, batch_size, k_negatives
        )

        # Hierarchy loss
        tree_distances = torch.rand(4, 4, device=test_device)
        tree_distances = (tree_distances + tree_distances.T) / 2  # Symmetric
        tree_distances.fill_diagonal_(0)

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)
        codes = ['code1', 'code2', 'code3', 'code4']
        code_to_idx = {'code1': 0, 'code2': 1, 'code3': 2, 'code4': 3}

        # Hierarchy loss
        hierarchy_loss_fn = HierarchyPreservationLoss(tree_distances, code_to_idx, weight=0.1)
        hierarchy_loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        # Structural preference over learned anchor-negative distances
        learned = LorentzOps.lorentz_distance(
            anchor.unsqueeze(1).expand(-1, k_negatives, -1).reshape(-1, dim + 1), negatives
        ).view(batch_size, k_negatives)
        structural_loss = structural_preference_from_distances(
            learned_distances=learned,
            structural_distances=torch.arange(k_negatives, dtype=torch.float32).expand(
                batch_size, -1
            ) + 1.0,
            candidate_code_ids=torch.arange(k_negatives).expand(batch_size, -1) + 100,
            anchor_code_ids=torch.arange(batch_size),
            is_explicit_exclusion=torch.zeros((batch_size, k_negatives), dtype=torch.bool),
            valid_mask=torch.ones((batch_size, k_negatives), dtype=torch.bool),
            margin=0.1,
            temperature=1.0,
            tie_tolerance=1e-6,
        )

        # Combined loss
        total_loss = infonce_loss + hierarchy_loss + structural_loss

        assert not torch.isnan(infonce_loss)
        assert not torch.isinf(infonce_loss)
        assert infonce_loss >= 0

        assert not torch.isnan(hierarchy_loss)
        assert not torch.isinf(hierarchy_loss)
        assert hierarchy_loss >= 0

        assert torch.isfinite(structural_loss)
        assert structural_loss >= 0

        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)
        assert total_loss >= 0

# -------------------------------------------------------------------------------------------------
# Req 11's terms: steps and references
# -------------------------------------------------------------------------------------------------

def _distances(rows: int, codes: int, seed: int) -> torch.Tensor:
    '''Seeded float64 distances in [0, 4).'''

    generator = torch.Generator().manual_seed(seed)
    return 4 * torch.rand(rows, codes, generator=generator, dtype=torch.float64)

def _scale(value: float) -> torch.Tensor:
    '''A float64 0-d scale, so a float64 reference meets no float32 rounding of the scale.'''

    return torch.tensor(value, dtype=torch.float64)

def _task_step() -> Dict[str, torch.Tensor]:
    '''
    Three queries over six codes. The second and third have two targets each, as a phrase that
    names two codes at its level does. Every query has a candidate that is not a target, so every
    candidate takes gradient.
    '''

    candidates = torch.tensor(
        [
            [True, True, True, False, False, True],
            [False, True, True, True, True, False],
            [True, False, True, True, False, True],
        ]
    )
    targets = torch.tensor(
        [
            [False, True, False, False, False, False],
            [False, False, True, False, True, False],
            [True, False, False, True, False, False],
        ]
    )
    return {
        'distances': _distances(3, 6, seed=11),
        'scale': _scale(1.5),
        'candidates': candidates,
        'targets': targets,
    }

def _reference_task_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    candidates: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    '''Spec 4.1(i) by hand: each query's softmax over its own C, summed over its T.'''

    per_query = []
    for row in range(distances.shape[0]):
        columns = candidates[row].nonzero().squeeze(1)
        probabilities = torch.softmax(-scale * distances[row, columns], dim=0)
        per_query.append(-torch.log(probabilities[targets[row, columns]].sum()))
    return torch.stack(per_query).mean()

def _code_code_step() -> Dict[str, Any]:
    '''
    Three anchors, codes 0, 2 and 4, among six. Codes 0 and 1 are a unary pair, as are 4 and 5, and
    code 2 has no partner, so J_a drops {0, 1}, {2} and {4, 5}. D*(a, a) and d(a, a) are 0, and each
    partner is the nearest code on both sides, so unmasked they would take most of the probability.
    '''

    rows = torch.arange(3)
    anchors = torch.tensor([0, 2, 4])
    partners = torch.tensor([1, -1, 5])
    paired = partners >= 0
    keep = torch.ones(3, 6, dtype=torch.bool)
    keep[rows, anchors] = False
    keep[rows[paired], partners[paired]] = False
    generator = torch.Generator().manual_seed(12)
    structural = 1 + 5 * torch.rand(3, 6, generator=generator, dtype=torch.float64)
    structural[rows, anchors] = 0.0
    structural[rows[paired], partners[paired]] = 0.5
    distances = _distances(3, 6, seed=13)
    distances[rows, anchors] = 0.0
    distances[rows[paired], partners[paired]] = 0.1
    return {
        'distances': distances,
        'scale': _scale(1.5),
        'structural': structural,
        'keep': keep,
        'target_temperature': 1.0,
    }

def _reference_code_code_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    structural: torch.Tensor,
    keep: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    '''Spec 4.1(ii) by hand: over each anchor's own J_a, CE(p_a, softmax of −s_c · d).'''

    per_anchor = []
    for row in range(distances.shape[0]):
        kept = keep[row]
        target = torch.softmax(-structural[row, kept] / target_temperature, dim=0)
        log_model = torch.log_softmax(-scale * distances[row, kept], dim=0)
        per_anchor.append(-(target * log_model).sum())
    return torch.stack(per_anchor).mean()

def _mean_target_entropy(structural: torch.Tensor, keep: torch.Tensor, temperature: float) -> float:
    '''The mean over anchors of the entropy of p_a = softmax over J_a of −D* / τ_t.'''

    entropies = []
    for row in range(structural.shape[0]):
        target = torch.softmax(-structural[row, keep[row]] / temperature, dim=0)
        entropies.append(-(target * target.log()).sum())
    return torch.stack(entropies).mean().item()

# -------------------------------------------------------------------------------------------------
# Req 11's task term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTaskLoss:
    '''Spec 4.1(i): the mean over queries of −log Σ_{t ∈ T} softmax_C(−s_q · d)_t.'''

    def test_two_targets_and_a_negative_at_one_distance_cost_minus_log_two_thirds(self):
        '''Probability is summed over T: the two targets hold 2/3 of it together.'''

        loss = terms.task_loss(
            distances=torch.ones(1, 3, dtype=torch.float64),
            scale=_scale(1.0),
            candidates=torch.tensor([[True, True, True]]),
            targets=torch.tensor([[True, True, False]]),
        )

        # The mean of each target's −log p would be log 3, as would a single target
        assert loss.item() == pytest.approx(-math.log(2 / 3), rel=1e-12)

    @pytest.mark.parametrize('scale', [0.5, 1.0, 3.0])
    def test_the_loss_is_minus_the_log_of_the_probability_summed_over_the_targets(self, scale):
        '''The value and its gradient in the distances and the scale are the by-hand reference's.'''

        step = _task_step()
        distances = step['distances'].clone().requires_grad_(True)
        scales = _scale(scale).requires_grad_(True)
        inputs = {**step, 'distances': distances, 'scale': scales}

        loss = terms.task_loss(**inputs)
        reference = _reference_task_loss(**inputs)

        torch.testing.assert_close(loss, reference, rtol=1e-12, atol=0.0)
        # A normalizer over T that took no gradient would keep the value and push every target away
        gradients = torch.autograd.grad(loss, (distances, scales))
        expected = torch.autograd.grad(reference, (distances, scales))
        for got, want in zip(gradients, expected):
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)

    def test_the_softmax_runs_over_exactly_the_candidates(self):
        '''
        C is exactly the candidates mask. A code outside it carries no probability and takes no
        gradient, however near it is, and every code inside it takes gradient.
        '''

        step = _task_step()
        outside = ~step['candidates']
        distances = step['distances'].clone().requires_grad_(True)
        loss = terms.task_loss(**{**step, 'distances': distances})
        loss.backward()

        for value in (0.0, 1e3):
            moved = step['distances'].clone()
            moved[outside] = value
            assert torch.equal(terms.task_loss(**{**step, 'distances': moved}), loss.detach())
        assert torch.count_nonzero(distances.grad[outside]) == 0
        assert torch.count_nonzero(distances.grad[step['candidates']]) == step['candidates'].sum()

    def test_the_task_loss_passes_gradient_to_the_distances_and_the_scale(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        step = _task_step()
        distances = step['distances'].float().requires_grad_(True)

        loss = terms.task_loss(distances, scale(), step['candidates'], step['targets'])
        loss.backward()

        assert torch.isfinite(distances.grad).all()
        assert torch.count_nonzero(distances.grad) > 0
        assert torch.isfinite(scale.log_scale.grad)
        assert scale.log_scale.grad.item() != 0.0

    def test_a_target_outside_the_candidates_is_refused(self):
        '''T lies in C: a target outside it would sum probability that no softmax gave it.'''

        step = _task_step()
        # Code 3 is not one of the first query's candidates
        step['targets'][0, 3] = True

        with pytest.raises(ValueError, match='query 0 has a target outside its candidates'):
            terms.task_loss(**step)

    def test_a_query_without_a_target_is_refused(self):
        step = _task_step()
        step['targets'][1] = False

        with pytest.raises(ValueError, match='query 1 has no target'):
            terms.task_loss(**step)

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            pytest.param(
                {
                    'distances': torch.zeros(0, 6, dtype=torch.float64),
                    'candidates': torch.zeros(0, 6, dtype=torch.bool),
                    'targets': torch.zeros(0, 6, dtype=torch.bool),
                },
                'task_loss needs a non-empty',
                id='an-empty-step',
            ),
            pytest.param(
                {'scale': torch.ones(1, dtype=torch.float64)},
                'task_loss needs a 0-d logit scale',
                id='a-scale-that-is-not-0-d',
            ),
            pytest.param(
                {'candidates': torch.ones(3, 5, dtype=torch.bool)},
                'candidates must be a bool mask shaped like the distances',
                id='misaligned-candidates',
            ),
            pytest.param(
                {'targets': torch.ones(3, 6)},
                'targets must be a bool mask shaped like the distances',
                id='targets-that-are-not-bool',
            ),
        ],
    )
    def test_a_malformed_step_is_refused(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            terms.task_loss(**{**_task_step(), **overrides})

# -------------------------------------------------------------------------------------------------
# Req 11's code–code term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCodeCodeLoss:
    '''
    Spec 4.1(ii): the mean over anchors of CE(p_a, softmax over J_a of −s_c · d(a, j)), with the
    target p_a = softmax over J_a of −D*_{aj} / τ_t.
    '''

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
    def test_the_loss_is_the_cross_entropy_to_softmax_of_minus_d_star_over_tau(self, temperature):
        step = {**_code_code_step(), 'target_temperature': temperature}

        loss = terms.code_code_loss(**step)

        torch.testing.assert_close(loss, _reference_code_code_loss(**step), rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
    def test_the_loss_is_stationary_where_the_model_reproduces_the_target(self, temperature):
        '''
        With d = D* / (τ_t · s) on J_a, the model's softmax is the target, so the gradient vanishes
        and the loss is the target's entropy. Any other target would leave a gradient here.
        '''

        step = _code_code_step()
        scale = 1.5
        distances = (step['structural'] / (temperature * scale)).requires_grad_(True)

        loss = terms.code_code_loss(
            distances, _scale(scale), step['structural'], step['keep'], temperature
        )
        loss.backward()

        entropy = _mean_target_entropy(step['structural'], step['keep'], temperature)
        assert distances.grad.abs().max().item() < 1e-12
        assert loss.item() == pytest.approx(entropy, rel=1e-12)

    def test_the_anchor_and_its_unary_partner_carry_no_probability_on_either_side(self):
        '''
        J_a drops the anchor and its unary partner (Req 9). Their D* never reaches the target and
        their d never reaches the model's softmax, though unmasked they would be the nearest codes
        on both sides. Every other code takes gradient.
        '''

        step = _code_code_step()
        masked, kept = ~step['keep'], step['keep']
        distances = step['distances'].clone().requires_grad_(True)
        loss = terms.code_code_loss(**{**step, 'distances': distances})
        loss.backward()

        for value in (0.0, 0.25, 50.0):
            target_side = {**step, 'structural': step['structural'].clone()}
            target_side['structural'][masked] = value
            model_side = {**step, 'distances': step['distances'].clone()}
            model_side['distances'][masked] = value
            assert torch.equal(terms.code_code_loss(**target_side), loss.detach())
            assert torch.equal(terms.code_code_loss(**model_side), loss.detach())
        assert torch.count_nonzero(distances.grad[masked]) == 0
        assert torch.count_nonzero(distances.grad[kept]) == kept.sum()

    def test_the_code_code_loss_passes_gradient_to_the_distances_and_the_scale(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        step = _code_code_step()
        distances = step['distances'].float().requires_grad_(True)

        loss = terms.code_code_loss(
            distances, scale(), step['structural'].float(), step['keep'], 1.0
        )
        loss.backward()

        assert torch.isfinite(distances.grad).all()
        assert torch.count_nonzero(distances.grad) > 0
        assert torch.isfinite(scale.log_scale.grad)
        assert scale.log_scale.grad.item() != 0.0

    def test_an_anchor_that_keeps_no_code_is_refused(self):
        step = _code_code_step()
        step['keep'][2] = False

        with pytest.raises(ValueError, match='anchor 2 keeps no code'):
            terms.code_code_loss(**step)

    @pytest.mark.parametrize('temperature', [0.0, -1.0, math.inf, math.nan])
    def test_a_target_temperature_that_is_not_positive_and_finite_is_refused(self, temperature):
        step = {**_code_code_step(), 'target_temperature': temperature}

        with pytest.raises(ValueError, match='target_temperature must be a positive finite number'):
            terms.code_code_loss(**step)

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            pytest.param(
                {
                    'distances': torch.zeros(0, 6, dtype=torch.float64),
                    'structural': torch.zeros(0, 6, dtype=torch.float64),
                    'keep': torch.zeros(0, 6, dtype=torch.bool),
                },
                'code_code_loss needs a non-empty',
                id='an-empty-step',
            ),
            pytest.param(
                {'scale': torch.ones(1, dtype=torch.float64)},
                'code_code_loss needs a 0-d logit scale',
                id='a-scale-that-is-not-0-d',
            ),
            pytest.param(
                {'structural': torch.zeros(3, 5, dtype=torch.float64)},
                'structural must be shaped like the distances',
                id='misaligned-structural-distances',
            ),
            pytest.param(
                {'keep': torch.ones(3, 6)},
                'keep must be a bool mask shaped like the distances',
                id='a-keep-mask-that-is-not-bool',
            ),
        ],
    )
    def test_a_malformed_step_is_refused(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            terms.code_code_loss(**{**_code_code_step(), **overrides})

# -------------------------------------------------------------------------------------------------
# Req 11's radial term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRadialLoss:
    '''
    Spec 4.1(iii): the mean over anchors of (r_a − ρ · (λ(a) − 1))², so the virtual root sits at o,
    the sectors at ρ and the six-digit codes at 5ρ.
    '''

    @pytest.mark.parametrize('rho', [0.5, 1.0, 2.0])
    def test_the_target_radius_is_rho_times_level_minus_one(self, rho):
        levels = torch.tensor([2, 3, 4, 5, 6])
        # The sectors at ρ, and so on out to the six-digit codes at 5ρ
        steps_out = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        radius = (rho * steps_out).requires_grad_()

        loss = terms.radial_loss(radius, levels, rho)
        loss.backward()

        assert loss.item() == 0.0
        assert torch.count_nonzero(radius.grad) == 0
        # At r = ρ · λ every code sits one step out, so the loss is ρ²
        one_out = terms.radial_loss(rho * levels.double(), levels, rho)
        assert one_out.item() == pytest.approx(rho**2, rel=1e-12)

    def test_the_loss_is_the_mean_squared_gap_to_the_target(self):
        generator = torch.Generator().manual_seed(23)
        levels = torch.randint(2, 7, (9, ), generator=generator)
        radius = 6 * torch.rand(9, generator=generator, dtype=torch.float64)

        loss = terms.radial_loss(radius, levels, 0.75)

        gaps = [r - 0.75 * (level - 1) for r, level in zip(radius.tolist(), levels.tolist())]
        assert loss.item() == pytest.approx(sum(gap**2 for gap in gaps) / 9, rel=1e-12)

    def test_the_radial_loss_passes_gradient_to_the_radii(self):
        radius = torch.tensor([0.5, 2.0, 7.5], requires_grad=True)

        terms.radial_loss(radius, torch.tensor([2, 4, 6]), 1.0).backward()

        # The gradient of the mean of (r − (λ − 1))² is 2 (r − (λ − 1)) / A
        torch.testing.assert_close(radius.grad, torch.tensor([-1 / 3, -2 / 3, 5 / 3]))

    @pytest.mark.parametrize('rho', [0.0, -1.0, math.inf, math.nan])
    def test_a_radial_step_that_is_not_positive_and_finite_is_refused(self, rho):
        with pytest.raises(ValueError, match='radial_step must be a positive finite number'):
            terms.radial_loss(torch.ones(3), torch.tensor([2, 4, 6]), rho)

    @pytest.mark.parametrize(
        ('radius', 'levels'),
        [
            pytest.param(torch.zeros(0), torch.zeros(0, dtype=torch.long), id='an-empty-step'),
            pytest.param(torch.ones(3, 1), torch.tensor([2, 4, 6]), id='radii-that-are-not-1-d'),
            pytest.param(torch.ones(3), torch.tensor([2, 4]), id='misaligned-levels'),
        ],
    )
    def test_a_malformed_step_is_refused(self, radius, levels):
        with pytest.raises(ValueError, match='radial_loss needs non-empty radii'):
            terms.radial_loss(radius, levels, 1.0)

# -------------------------------------------------------------------------------------------------
# Req 11's learned logit scales
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLogitScale:
    '''Spec 4.1: s = exp(θ), from its init and clamped to its range; R7 states 1 in [0.01, 100].'''

    def test_the_stated_defaults_start_at_1_with_one_parameter(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)

        assert [name for name, _ in scale.named_parameters()] == ['log_scale']
        assert list(scale.buffers()) == []
        assert scale.log_scale.item() == 0.0
        assert scale().ndim == 0
        assert scale().item() == 1.0

    @pytest.mark.parametrize('init', [0.5, 2.0, 8.0])
    def test_the_scale_starts_at_init_anywhere_in_its_range(self, init):
        scale = terms.LogitScale(init, 0.5, 8.0)

        assert scale.log_scale.item() == pytest.approx(math.log(init), rel=1e-6)
        assert scale().item() == pytest.approx(init, rel=1e-6)

    @pytest.mark.parametrize(
        ('theta', 'clamped'),
        [
            pytest.param(math.log(1e4), 100.0, id='above-the-range'),
            pytest.param(math.log(1e-4), 0.01, id='below-the-range'),
        ],
    )
    def test_the_scale_clamps_to_its_range(self, theta, clamped):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        with torch.no_grad():
            scale.log_scale.fill_(theta)

        assert scale().item() == pytest.approx(clamped, rel=1e-6)

    @pytest.mark.parametrize(
        ('init', 'low', 'high', 'message'),
        [
            pytest.param(1.0, 2.0, 1.0, 'must satisfy 0 < low < high', id='an-empty-range'),
            pytest.param(1.0, 1.0, 1.0, 'must satisfy 0 < low < high', id='a-one-point-range'),
            pytest.param(1.0, 0.0, 10.0, 'must satisfy 0 < low < high', id='a-zero-low'),
            pytest.param(1.0, -1.0, 10.0, 'must satisfy 0 < low < high', id='a-negative-low'),
            pytest.param(0.5, 1.0, 10.0, 'must start inside its range', id='an-init-below'),
            pytest.param(20.0, 1.0, 10.0, 'must start inside its range', id='an-init-above'),
            pytest.param(math.nan, 0.01, 100.0, 'takes finite numbers', id='a-nan-init'),
            pytest.param(1.0, 0.01, math.inf, 'takes finite numbers', id='an-infinite-high'),
        ],
    )
    def test_a_range_that_is_empty_or_not_positive_or_misses_its_init_is_refused(
        self, init, low, high, message
    ):
        with pytest.raises(ValueError, match=message):
            terms.LogitScale(init, low, high)

# -------------------------------------------------------------------------------------------------
# Req 11's terms under autocast
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_terms_and_the_scale_stay_float32_under_cpu_bf16_autocast():
    '''Spec §6 "Precision": float32 inputs give float32 losses under CPU bf16 autocast.'''

    scale = terms.LogitScale(1.0, 0.01, 100.0)
    task, code_code = _task_step(), _code_code_step()

    with torch.autocast('cpu', dtype=torch.bfloat16):
        values = {
            'scale': scale(),
            'task': terms.task_loss(
                task['distances'].float(), scale(), task['candidates'], task['targets']
            ),
            'code_code': terms.code_code_loss(
                code_code['distances'].float(),
                scale(),
                code_code['structural'].float(),
                code_code['keep'],
                1.0,
            ),
            'radial': terms.radial_loss(torch.rand(4), torch.tensor([2, 3, 4, 6]), 1.0),
        }

    dtypes = {name: value.dtype for name, value in values.items()}
    assert dtypes == dict.fromkeys(values, torch.float32)
