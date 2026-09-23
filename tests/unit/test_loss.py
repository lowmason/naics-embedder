'''
Unit tests for loss functions.

Tests hyperbolic contrastive learning losses, hierarchy preservation losses,
and the structural-preference ranking loss used in training.
'''

import pytest
import torch

from naics_embedder.supervision.candidates import SelectedNegativeBatch
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
        loss_close = _contrastive(loss_fn, anchor, close_positive, negatives, batch_size, k_negatives)
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

        loss_with_mask = _contrastive(
            loss_fn, *sample_triplet, pseudo_related_mask=pseudo_related
        )

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
                    key: value[row : row + 1]
                    if isinstance(value, torch.Tensor)
                    else value
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
