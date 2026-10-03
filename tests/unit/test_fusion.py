'''
Fusion over present channels (Req 14; Req 9): masked mean, attention pooling and the MoE ablation.
'''

import pytest
import torch

from naics_embedder.text_model.fusion import (
    FUSIONS,
    AttentionFusion,
    MaskedMeanFusion,
    MoEFusion,
    build_fusion,
    masked_mean,
)

pytestmark = pytest.mark.unit

# Row 0: channels 0 and 2 present; row 1: channel 1 only; row 2: no channel present
PRESENT = torch.tensor([[True, False, True], [False, True, False], [False, False, False]])

def _vectors() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]],
            [[50.0, 50.0], [5.0, 6.0], [70.0, 70.0]],
            [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )

def test_masked_mean_averages_the_present_channels_only():
    fused = masked_mean(_vectors(), PRESENT)

    assert fused.tolist() == [[2.0, 3.0], [5.0, 6.0], [0.0, 0.0]]

def test_masked_mean_has_no_parameters():
    assert list(MaskedMeanFusion().parameters()) == []

def test_attention_weights_present_channels_by_a_softmax_of_their_scores():
    fusion = AttentionFusion(hidden_size=2)
    with torch.no_grad():
        fusion.query.copy_(torch.tensor([1.0, 0.0]))

    fused = fusion(_vectors(), PRESENT).vector

    # Row 0's present channels score 1 and 3
    weights = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)
    expected = weights[0] * torch.tensor([1.0, 2.0]) + weights[1] * torch.tensor([3.0, 4.0])
    torch.testing.assert_close(fused[0], expected)
    torch.testing.assert_close(fused[1], torch.tensor([5.0, 6.0]))
    assert fused[2].tolist() == [0.0, 0.0]

def test_attention_starts_as_the_masked_mean():
    fused = AttentionFusion(hidden_size=2)(_vectors(), PRESENT).vector

    torch.testing.assert_close(fused, masked_mean(_vectors(), PRESENT))

@pytest.mark.parametrize('name', FUSIONS)
def test_an_absent_channel_never_contributes(name):
    torch.manual_seed(0)
    fusion = build_fusion(name, hidden_size=2, num_experts=2, top_k=1, moe_hidden_dim=4).eval()
    perturbed = _vectors().clone()
    perturbed[~PRESENT] = -1000.0

    assert torch.equal(fusion(_vectors(), PRESENT).vector, fusion(perturbed, PRESENT).vector)

@pytest.mark.parametrize('name', FUSIONS)
def test_a_row_with_no_present_channel_fuses_finitely_with_a_finite_gradient(name):
    torch.manual_seed(0)
    fusion = build_fusion(name, hidden_size=2, num_experts=2, top_k=1, moe_hidden_dim=4)
    vectors = _vectors().requires_grad_(True)

    fused = fusion(vectors, PRESENT).vector
    fused.sum().backward()

    assert torch.isfinite(fused).all()
    assert torch.isfinite(vectors.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in fusion.parameters()
    )

def test_moe_routes_the_masked_mean_and_is_the_only_option_with_gates():
    torch.manual_seed(0)
    fusion = MoEFusion(hidden_size=2, num_experts=2, top_k=1, hidden_dim=4).eval()

    output = fusion(_vectors(), PRESENT)

    expected, gate_probs, top_k_indices = fusion.moe(masked_mean(_vectors(), PRESENT))
    assert torch.equal(output.vector, expected)
    assert torch.equal(output.gate_probs, gate_probs)
    assert torch.equal(output.top_k_indices, top_k_indices)
    for other in (MaskedMeanFusion(), AttentionFusion(hidden_size=2)):
        result = other(_vectors(), PRESENT)
        assert result.gate_probs is None and result.top_k_indices is None

def test_an_unknown_fusion_is_refused():
    with pytest.raises(ValueError, match='unknown fusion'):
        build_fusion('concatenate', hidden_size=2)
