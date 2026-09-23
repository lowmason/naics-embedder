from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import naics_embedder.text_model.naics_model as model_module
from naics_embedder.supervision.candidates import NegativeSelection, SelectedNegativeBatch
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.text_model.dataloader.datamodule import collate_fn


def forced_selection(order: list[int]):
    def select(candidates, **_kwargs) -> NegativeSelection:
        indices = torch.tensor(
            [order],
            dtype=torch.long,
            device=candidates.code_id.device,
        ).expand(candidates.code_id.shape[0], -1)
        return NegativeSelection(
            source_indices=indices,
            source_candidate_uid=candidates.candidate_uid.gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 3),
            ),
            scores=torch.ones_like(indices, dtype=candidates.embedding.dtype),
            reasons=torch.full_like(
                indices,
                int(SelectionReason.GEOMETRIC),
                dtype=torch.int8,
            ),
        )

    return select


@dataclass
class SelectionSpyLoss:
    contrastive_uids: list[list[list[int]]] | None = None
    structural_uids: list[list[list[int]]] | None = None
    code_ids: list[int] | None = None
    structural_distances: list[float] | None = None
    exclusion_flags: list[bool] | None = None
    router_first_column: list[float] | None = None
    false_negative_flags: list[bool] | None = None

    def contrastive(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        selected: SelectedNegativeBatch,
        effective_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self.contrastive_uids = selected.candidate_uid.detach().cpu().tolist()
        self.false_negative_flags = (
            None if effective_mask is None else effective_mask[0].cpu().tolist()
        )
        return (
            anchor_emb.square().mean()
            + positive_emb.square().mean()
            + selected.embedding.square().mean()
        )

    def structural(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        batch: dict,
        selected: SelectedNegativeBatch,
    ) -> torch.Tensor:
        self.structural_uids = selected.candidate_uid.detach().cpu().tolist()
        self.code_ids = selected.code_id[0].cpu().tolist()
        self.structural_distances = selected.structural_distance[0].cpu().tolist()
        self.exclusion_flags = selected.is_explicit_exclusion[0].cpu().tolist()
        self.router_first_column = selected.router_gate_probs[0, :, 0].cpu().tolist()
        eligible = selected.valid_mask & ~selected.is_explicit_exclusion
        return (
            selected.embedding[eligible].square().mean()
            + anchor_emb.square().mean()
            + positive_emb.square().mean()
        ) * 0.01


class StubMultiChannelEncoder(nn.Module):
    embedding_dim = 2

    def __init__(self, **_kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, channel_inputs):
        raw = channel_inputs['title']['input_ids'][:, 0].to(torch.float32)
        value = raw * self.scale
        spatial = torch.stack([value, value / 2.0], dim=1)
        time = torch.sqrt(1.0 + spatial.square().sum(dim=1, keepdim=True))
        embedding = torch.cat([time, spatial], dim=1)
        first_gate = ((raw - 1.0) / 10.0).clamp(0.0, 1.0)
        gate_probs = torch.stack([first_gate, 1.0 - first_gate], dim=1)
        return {
            'embedding': embedding,
            'embedding_euc': spatial,
            'gate_probs': gate_probs,
            'top_k_indices': gate_probs.argmax(dim=1, keepdim=True),
        }


def _encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        channel: {
            'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
            'attention_mask': torch.ones(2, dtype=torch.long),
        }
        for channel in ('title', 'description', 'excluded', 'examples')
    }


def _repaired_item(candidate_ids: list[int]) -> dict:
    return {
        'anchor_code_id': 0,
        'anchor_code': '111111',
        'anchor_embedding': _encoded(0),
        'positive_code_id': 1,
        'positive_code': '111112',
        'positive_embedding': _encoded(1),
        'positive_structural_distance': 0.5,
        'positive_structural_relation_id': 1,
        'candidate_pool': [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_embedding': _encoded(code_id),
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in candidate_ids
        ],
        'difficulty_proposal_indices': list(range(len(candidate_ids))),
        'selection_k': 3,
    }


@pytest.fixture
def repaired_training_batch():
    batch = collate_fn(
        [
            _repaired_item([2, 3, 4]),
            _repaired_item([2, 3, 4, 4]),
        ],
        supervision_mode='repaired',
    )
    assert batch['candidate_valid_mask'][0].tolist() == [True, True, True, False]
    return batch


@pytest.fixture
def tiny_repaired_model(monkeypatch, generated_bundle):
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)
    model = model_module.NAICSContrastiveModel(
        base_model_name='test-stub',
        num_experts=2,
        top_k=1,
        moe_hidden_dim=4,
        hierarchy_weight=0.0,
        radius_reg_weight=0.0,
        level_radius_weight=0.0,
        load_balancing_coef=0.0,
        supervision_manifest_path=str(generated_bundle),
        supervision_mode='repaired',
        structural_preference_weight=0.35,
    )
    model.current_curriculum_flags = {
        'enable_hard_negative_mining': True,
        'enable_router_guided_sampling': False,
        'enable_clustering': True,
    }
    model.current_schedule_scalars = {}
    model.code_to_pseudo_label = {
        '111111': 7,
        '111113': 7,
        '222222': 7,
    }
    monkeypatch.setattr(model, '_update_curriculum_state', lambda *_args: None)
    monkeypatch.setattr(model, 'log', Mock())
    return model


def test_forced_reorder_preserves_uid_across_every_loss_field(
    tiny_repaired_model,
    repaired_training_batch,
    monkeypatch,
):
    spy = SelectionSpyLoss()
    monkeypatch.setattr(
        tiny_repaired_model,
        '_compute_contrastive_loss',
        spy.contrastive,
    )
    monkeypatch.setattr(
        tiny_repaired_model,
        '_compute_structural_preference_loss',
        spy.structural,
    )
    monkeypatch.setattr(
        tiny_repaired_model.selection_coordinator,
        'select',
        forced_selection([2, 0, 1]),
    )

    loss = tiny_repaired_model.training_step(repaired_training_batch, batch_idx=0)
    loss.backward()

    assert spy.contrastive_uids == spy.structural_uids
    assert spy.code_ids == [4, 2, 3]
    assert spy.structural_distances == [99.0, 2.0, 99.0]
    assert spy.exclusion_flags == [False, True, False]
    # The stub computes gate probabilities in float32, so compare approximately.
    assert spy.router_first_column == pytest.approx([0.3, 0.1, 0.2])
    assert spy.false_negative_flags == [False, False, True]
    assert all(
        uid[2] >= 0
        for row in spy.contrastive_uids
        for uid in row
    )
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in tiny_repaired_model.parameters()
    )


def test_old_parallel_arrays_misalign_but_checked_selection_does_not(candidate_batch):
    order = [2, 0, 1]
    reordered_embeddings = candidate_batch.embedding[:, order]
    old_parallel_pairs = list(
        zip(
            reordered_embeddings[0, :, 0].tolist(),
            candidate_batch.code_id[0].tolist(),
        )
    )
    expected_code_for_embedding = {1.0: 101, 2.0: 102, 3.0: 103}

    assert any(
        expected_code_for_embedding[embedding_value] != code_id
        for embedding_value, code_id in old_parallel_pairs
    )

    selected = candidate_batch.select(forced_selection(order)(candidate_batch))
    checked_pairs = list(
        zip(
            selected.embedding[0, :, 0].tolist(),
            selected.code_id[0].tolist(),
        )
    )
    assert checked_pairs == [(3.0, 103), (1.0, 101), (2.0, 102)]
