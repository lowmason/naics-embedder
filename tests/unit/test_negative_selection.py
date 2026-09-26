import hashlib
import struct
from dataclasses import replace

import pytest
import torch

from naics_embedder.supervision.candidates import CandidateProposal
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.supervision.selection import (
    NegativeSelectionCoordinator,
    canonical_occurrence_mask,
    stable_hash,
)

def test_an_explicit_exclusion_is_never_selected(candidate_batch_with_exclusions):
    # Codes 20-22 are the anchor's exclusions (Req 8), so only 30-32 can fill the three slots
    selection = NegativeSelectionCoordinator().select(
        candidate_batch_with_exclusions,
        anchor_code_ids=torch.tensor([10]),
        positive_code_ids=torch.tensor([11]),
        k=3,
        proposals=(),
    )
    selected = candidate_batch_with_exclusions.select(selection)

    assert selected.code_id.tolist() == [[30, 31, 32]]
    assert not selected.is_explicit_exclusion.any()
    assert selection.reasons.tolist() == [[SelectionReason.BACKFILL] * 3]

def test_exclusions_add_no_selection_capacity(candidate_batch_with_exclusions):
    # Three non-exclusion codes cannot fill four slots, however many exclusions the pool holds
    with pytest.raises(ValueError, match='requested 4.*available 3 .unique non-exclusion codes'):
        _select(candidate_batch_with_exclusions, k=4)

def test_no_exclusion_uses_all_slots_for_ordinary_candidates(candidate_batch):
    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        proposals=(),
    )
    selected = candidate_batch.select(selection)

    assert selected.code_id.unique().numel() == 3
    assert not selected.is_explicit_exclusion.any()

def test_proposal_ties_break_by_code_then_uid(candidate_batch):
    proposal = CandidateProposal(
        source_indices=torch.tensor([[2, 1, 0]]),
        scores=torch.tensor([[1.0, 1.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )
    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        proposals=(proposal, ),
    )
    assert candidate_batch.select(selection).code_id.tolist() == [[101, 102, 103]]

def test_geometric_then_router_merge_is_deterministic_and_code_unique(candidate_batch):
    geometric = CandidateProposal(
        source_indices=torch.tensor([[2, 1]]),
        scores=torch.tensor([[0.9, 0.8]]),
        reason=SelectionReason.GEOMETRIC,
    )
    router = CandidateProposal(
        source_indices=torch.tensor([[1, 0]]),
        scores=torch.tensor([[0.95, 0.7]]),
        reason=SelectionReason.ROUTER,
    )

    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        proposals=(geometric, router),
    )

    assert candidate_batch.select(selection).code_id.tolist() == [[103, 102, 101]]
    assert selection.reasons.tolist() == [
        [
            SelectionReason.GEOMETRIC,
            SelectionReason.GEOMETRIC,
            SelectionReason.ROUTER,
        ]
    ]

def test_duplicate_codes_collapse_to_smallest_occurrence_uid(candidate_batch_with_duplicate_code):
    selection = NegativeSelectionCoordinator().select(
        candidate_batch_with_duplicate_code,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=2,
        proposals=(),
    )
    selected = candidate_batch_with_duplicate_code.select(selection)

    assert selected.code_id.unique().numel() == 2
    duplicate_slot = selected.code_id.eq(101)
    # One selected occurrence of code 101: a [1, 3] UID row, the smallest occurrence UID.
    assert selected.candidate_uid[duplicate_slot].tolist() == [[0, 0, 0]]

def test_insufficient_unique_candidates_is_fatal(candidate_batch):
    with pytest.raises(
        ValueError,
        match='anchor code ID 100.*requested 4.*available 3',
    ):
        NegativeSelectionCoordinator().select(
            candidate_batch,
            anchor_code_ids=torch.tensor([100]),
            positive_code_ids=torch.tensor([104]),
            k=4,
            proposals=(),
        )

# -------------------------------------------------------------------------------------------------
# Exclusions and eligibility
# -------------------------------------------------------------------------------------------------

def _select(batch, *, k, proposals=(), anchor=10, positive=11):
    return NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([anchor]),
        positive_code_ids=torch.tensor([positive]),
        k=k,
        proposals=proposals,
    )

def test_proposals_cannot_select_an_exclusion(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    greedy = CandidateProposal(
        source_indices=torch.tensor([[0, 1, 2, 3]]),
        scores=torch.tensor([[9.0, 8.0, 7.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )

    selected = batch.select(_select(batch, k=3, proposals=(greedy, )))

    # The three exclusions outscore code 30 but are skipped; backfill supplies 31 and 32
    reasons = selected.selection_reasons.tolist()[0]
    assert selected.code_id.tolist() == [[30, 31, 32]]
    assert reasons == [SelectionReason.GEOMETRIC] + [SelectionReason.BACKFILL] * 2

def test_anchor_positive_and_invalid_candidates_are_never_selected(candidate_batch):
    batch = replace(candidate_batch, valid_mask=torch.tensor([[True, True, False]]))

    selected = batch.select(_select(batch, k=1, anchor=101, positive=999))

    assert selected.code_id.tolist() == [[102]]
    with pytest.raises(ValueError, match='requested 2.*available 1'):
        _select(batch, k=2, anchor=101, positive=999)

def test_k_must_be_at_least_one(candidate_batch):
    with pytest.raises(ValueError, match='at least one'):
        _select(candidate_batch, k=0)

@pytest.mark.parametrize('bad_score', [float('nan'), float('inf')])
def test_malformed_proposal_scores_are_fatal(candidate_batch, bad_score):
    proposal = CandidateProposal(
        source_indices=torch.tensor([[2, 1, 0]]),
        scores=torch.tensor([[1.0, bad_score, 0.5]]),
        reason=SelectionReason.ROUTER,
    )

    with pytest.raises(ValueError, match='ROUTER proposal .* anchor code ID 100'):
        NegativeSelectionCoordinator().select(
            candidate_batch,
            anchor_code_ids=torch.tensor([100]),
            positive_code_ids=torch.tensor([104]),
            k=3,
            proposals=(proposal, ),
        )

def test_negative_infinity_marks_an_ineligible_proposal_entry(candidate_batch):
    proposal = CandidateProposal(
        source_indices=torch.tensor([[2, 1, 0]]),
        scores=torch.tensor([[1.0, float('-inf'), 0.5]]),
        reason=SelectionReason.GEOMETRIC,
    )

    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        proposals=(proposal, ),
    )

    reasons = [SelectionReason(reason) for reason in selection.reasons[0].tolist()]
    assert candidate_batch.select(selection).code_id.tolist() == [[103, 101, 102]]
    assert reasons == [SelectionReason.GEOMETRIC] * 2 + [SelectionReason.BACKFILL]

def test_canonical_occurrence_is_the_smallest_valid_uid_per_code():
    generator = torch.Generator().manual_seed(11)
    for _ in range(25):
        batch, width = 3, 40
        code_id = torch.randint(0, 6, (batch, width), generator=generator)
        uid = torch.stack(
            [
                torch.randint(0, 4, (batch, width), generator=generator),
                torch.randint(0, 5, (batch, width), generator=generator),
                torch.arange(width).expand(batch, width),
            ],
            dim=-1,
        )
        valid = torch.rand((batch, width), generator=generator).gt(0.3)

        mask = canonical_occurrence_mask(code_id, uid, valid)

        for row in range(batch):
            smallest = {}
            for slot in range(width):
                if not valid[row, slot]:
                    continue
                code, key = int(code_id[row, slot]), tuple(uid[row, slot].tolist())
                if code not in smallest or key < smallest[code][0]:
                    smallest[code] = (key, slot)
            assert set(torch.where(mask[row])[0].tolist()) == {
                slot
                for _, slot in smallest.values()
            }

def test_canonical_occurrence_handles_empty_pools_and_narrow_dtypes():
    empty = canonical_occurrence_mask(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros((2, 0, 3), dtype=torch.long),
        torch.zeros((2, 0), dtype=torch.bool),
    )
    # In int32, code 2**30 + 5 shifted by the 2-bit slot width wraps onto code 5's keys, which
    # would interleave the two code groups; the key must be built in int64.
    code_id = torch.tensor([[2**30 + 5, 5, 2**30 + 5]])
    uid = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2]]])
    valid = torch.ones((1, 3), dtype=torch.bool)

    narrow = canonical_occurrence_mask(code_id.to(torch.int32), uid.to(torch.int32), valid)

    assert empty.shape == (2, 0)
    assert narrow.tolist() == [[True, True, False]]

def test_malformed_score_in_an_unreached_proposal_is_still_fatal(candidate_batch):
    filling = CandidateProposal(
        source_indices=torch.tensor([[0, 1, 2]]),
        scores=torch.tensor([[3.0, 2.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )
    unreached = CandidateProposal(
        source_indices=torch.tensor([[2]]),
        scores=torch.tensor([[float('nan')]]),
        reason=SelectionReason.ROUTER,
    )

    with pytest.raises(ValueError, match='ROUTER proposal .* anchor code ID 100'):
        NegativeSelectionCoordinator().select(
            candidate_batch,
            anchor_code_ids=torch.tensor([100]),
            positive_code_ids=torch.tensor([104]),
            k=3,
            proposals=(filling, unreached),
        )

def test_stable_hash_matches_a_sha256_of_the_packed_seed_and_anchor():
    digest = hashlib.sha256(struct.pack('>qq', 7, 10)).digest()

    assert stable_hash(7, 10) == int.from_bytes(digest[:8], 'big', signed=False)
    assert stable_hash(7, 10) != stable_hash(7, 11)
