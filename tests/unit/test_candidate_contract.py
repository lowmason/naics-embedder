from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from naics_embedder.supervision.candidates import (
    CandidateEntityBatch,
    CandidateProposal,
    NegativeSelection,
)
from naics_embedder.supervision.schema import SelectionReason


def test_select_gathers_every_field_by_one_source_index(candidate_batch):
    indices = torch.tensor([[2, 0, 1]])
    selection = NegativeSelection(
        source_indices=indices,
        source_candidate_uid=candidate_batch.candidate_uid.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, 3)
        ),
        scores=torch.tensor([[0.9, 0.8, 0.7]]),
        reasons=torch.full((1, 3), SelectionReason.GEOMETRIC, dtype=torch.int8),
    )

    selected = candidate_batch.select(selection)

    assert selected.code_id.tolist() == [[103, 101, 102]]
    assert selected.structural_distance.tolist() == [[3.0, 1.0, 2.0]]
    assert selected.router_gate_probs[:, :, 0].tolist() == [[0.3, 0.1, 0.2]]
    assert selected.runtime_fields['difficulty'].tolist() == [[30.0, 10.0, 20.0]]
    assert torch.equal(selected.candidate_uid, selection.source_candidate_uid)


def test_select_rejects_uid_from_a_stale_pool(candidate_batch):
    indices = torch.tensor([[0]])
    wrong_uid = candidate_batch.candidate_uid[:, :1].clone()
    wrong_uid[0, 0, 2] += 1
    selection = NegativeSelection(
        source_indices=indices,
        source_candidate_uid=wrong_uid,
        scores=torch.ones((1, 1)),
        reasons=torch.full((1, 1), SelectionReason.BACKFILL, dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='UID mismatch.*row 0.*slot 0'):
        candidate_batch.select(selection)


def test_select_rejects_invalid_source_candidate(candidate_batch):
    invalid_batch = replace(
        candidate_batch,
        valid_mask=torch.tensor([[True, False, True]]),
    )
    selection = NegativeSelection(
        source_indices=torch.tensor([[1]]),
        source_candidate_uid=candidate_batch.candidate_uid[:, 1:2],
        scores=torch.ones((1, 1)),
        reasons=torch.full((1, 1), SelectionReason.BACKFILL, dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='invalid source candidate'):
        invalid_batch.select(selection)


# -------------------------------------------------------------------------------------------------
# Contract validation
# -------------------------------------------------------------------------------------------------

def _selection_for(batch, order):
    indices = torch.tensor([order])
    return NegativeSelection(
        source_indices=indices,
        source_candidate_uid=batch.candidate_uid.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, 3)
        ),
        scores=torch.zeros((1, len(order))),
        reasons=torch.full((1, len(order)), SelectionReason.BACKFILL, dtype=torch.int8),
    )


def test_every_selected_field_follows_the_same_order(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    order = [4, 0, 5, 2]

    selected = batch.select(_selection_for(batch, order))

    assert selected.code_id.tolist() == [[31, 20, 32, 22]]
    assert selected.is_explicit_exclusion.tolist() == [[False, True, False, True]]
    assert selected.anchor_excludes_candidate.tolist() == [[False, True, False, True]]
    assert selected.relation_margin.tolist() == [[5.0, 1.0, 6.0, 3.0]]
    assert selected.embedding[0, :, 0].tolist() == [5.0, 1.0, 6.0, 3.0]
    assert selected.valid_mask.all()


def test_misaligned_candidate_field_is_rejected(candidate_batch):
    with pytest.raises(ValueError, match='structural_distance does not align'):
        replace(candidate_batch, structural_distance=torch.zeros((1, 2)))


def test_exclusion_flag_must_equal_the_directional_or(candidate_batch):
    with pytest.raises(ValueError, match='directional OR'):
        replace(candidate_batch, is_explicit_exclusion=torch.tensor([[True, False, False]]))


def test_selection_indices_must_be_in_bounds(candidate_batch):
    selection = NegativeSelection(
        source_indices=torch.tensor([[3]]),
        source_candidate_uid=torch.zeros((1, 1, 3), dtype=torch.long),
        scores=torch.zeros((1, 1)),
        reasons=torch.zeros((1, 1), dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='out-of-bounds'):
        candidate_batch.select(selection)


def test_selection_batch_must_match_the_pool(candidate_batch):
    selection = NegativeSelection(
        source_indices=torch.zeros((2, 1), dtype=torch.long),
        source_candidate_uid=torch.zeros((2, 1, 3), dtype=torch.long),
        scores=torch.zeros((2, 1)),
        reasons=torch.zeros((2, 1), dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='batch dimension'):
        candidate_batch.select(selection)


def test_selection_components_must_align():
    with pytest.raises(ValueError, match='scores and reasons'):
        NegativeSelection(
            source_indices=torch.zeros((1, 2), dtype=torch.long),
            source_candidate_uid=torch.zeros((1, 2, 3), dtype=torch.long),
            scores=torch.zeros((1, 3)),
            reasons=torch.zeros((1, 2), dtype=torch.int8),
        )


def test_proposal_indices_and_scores_must_align():
    with pytest.raises(ValueError, match='share shape'):
        CandidateProposal(
            source_indices=torch.zeros((1, 2), dtype=torch.long),
            scores=torch.zeros((1, 3)),
            reason=SelectionReason.GEOMETRIC,
        )


def test_entity_batch_requires_aligned_intrinsic_fields():
    with pytest.raises(ValueError, match='embedding does not align'):
        CandidateEntityBatch(
            candidate_uid=torch.zeros((1, 2, 3), dtype=torch.long),
            code_id=torch.zeros((1, 2), dtype=torch.long),
            embedding=torch.zeros((1, 3, 4)),
            router_gate_probs=None,
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )


def test_valid_entities_need_nonnegative_identity():
    with pytest.raises(ValueError, match='nonnegative code IDs'):
        CandidateEntityBatch(
            candidate_uid=torch.zeros((1, 1, 3), dtype=torch.long),
            code_id=torch.tensor([[-1]]),
            embedding=torch.zeros((1, 1, 2)),
            router_gate_probs=None,
            valid_mask=torch.ones((1, 1), dtype=torch.bool),
        )


def test_candidate_batches_are_immutable(candidate_batch):
    with pytest.raises(FrozenInstanceError):
        candidate_batch.code_id = torch.zeros((1, 3), dtype=torch.long)
    with pytest.raises(TypeError):
        candidate_batch.runtime_fields['difficulty'] = torch.zeros((1, 3))
