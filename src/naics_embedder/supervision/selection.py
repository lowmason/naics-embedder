'''
Deterministic final negative selection over non-exclusion candidates.

For each anchor the coordinator (1) merges strategy proposals in priority order; (2) deduplicates
by code ID, keeping the smallest occurrence UID; (3) breaks ties by code ID, then UID; and (4)
backfills deterministically. An explicit exclusion is never selected: an exclusion pair is not a
code-code negative (Req 8), so no slot is reserved for one. The coordinator returns one
``NegativeSelection`` of source indices; it never gathers candidate fields itself.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import hashlib
import math
import struct
from typing import Dict, List, Sequence, Set, Tuple

import torch

from naics_embedder.supervision.candidates import (
    CandidateProposal,
    NegativeCandidateBatch,
    NegativeSelection,
)
from naics_embedder.supervision.schema import SelectionReason

# -------------------------------------------------------------------------------------------------
# Stable hash
# -------------------------------------------------------------------------------------------------

def stable_hash(global_seed: int, anchor_code_id: int) -> int:
    '''Process- and platform-independent 64-bit hash of (seed, anchor); never Python ``hash()``.'''

    payload = struct.pack('>qq', global_seed, anchor_code_id)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big', signed=False)

# -------------------------------------------------------------------------------------------------
# Canonical occurrences
# -------------------------------------------------------------------------------------------------

def canonical_occurrence_mask(
    code_id: torch.Tensor,
    candidate_uid: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    '''
    Per row, mark the valid occurrence of each code with the smallest candidate UID.

    This is the occurrence the coordinator keeps when it deduplicates by code. Letting strategies
    score only these entries means a code repeated across rows and ranks (as in a distributed
    global pool) occupies one proposal slot instead of crowding out distinct codes.

    Args:
        code_id: ``[batch, candidate]`` code IDs.
        candidate_uid: ``[batch, candidate, 3]`` occurrence UIDs.
        valid_mask: ``[batch, candidate]`` validity.

    Returns:
        ``[batch, candidate]`` boolean mask on ``valid_mask``'s device.

    Raises:
        ValueError: If the identities cannot be ordered in one 64-bit key.
    '''

    if valid_mask.numel() == 0:
        return torch.zeros_like(valid_mask)
    device = valid_mask.device
    valid = valid_mask.cpu()
    # Build keys in int64 whatever the input dtype: narrower shifts would wrap and interleave codes.
    code = code_id.cpu().to(torch.int64).masked_fill(~valid, 0)
    uid = candidate_uid.cpu().to(torch.int64).masked_fill(~valid.unsqueeze(-1), 0)
    widths = [int(uid[..., component].max()).bit_length() for component in range(3)]
    code_width = int(code.max()).bit_length()
    if code_width + sum(widths) > 62:
        raise ValueError('candidate identities are too large to order in one 64-bit key')

    # Lexicographic (code, uid) order in one key: code is most significant, then UID components.
    key = code.clone()
    for component, width in enumerate(widths):
        key = (key << width) | uid[..., component]
    key = key.masked_fill(~valid, torch.iinfo(torch.int64).max)
    order = key.argsort(dim=1)
    sorted_code = code.gather(1, order)
    first = valid.gather(1, order).clone()
    first[:, 1:] &= sorted_code[:, 1:].ne(sorted_code[:, :-1])
    mask = torch.zeros_like(valid)
    mask.scatter_(1, order, first)
    return mask.to(device)

# -------------------------------------------------------------------------------------------------
# Coordinator
# -------------------------------------------------------------------------------------------------

Uid = Tuple[int, int, int]

class NegativeSelectionCoordinator:
    '''Turns a canonical candidate pool and strategy proposals into one checked selection.'''

    def select(
        self,
        candidates: NegativeCandidateBatch,
        *,
        anchor_code_ids: torch.Tensor,
        positive_code_ids: torch.Tensor,
        k: int,
        proposals: Sequence[CandidateProposal],
    ) -> NegativeSelection:
        '''
        Select ``k`` unique negative codes per anchor, never an explicit exclusion.

        Raises:
            ValueError: If ``k < 1``, shapes disagree, or an anchor lacks ``k`` selectable codes
                (unique non-exclusion codes).
        '''

        if k < 1:
            raise ValueError('negative selection K must be at least one')
        batch_size, pool_size = candidates.code_id.shape
        if anchor_code_ids.shape != (batch_size, ) or positive_code_ids.shape != (batch_size, ):
            raise ValueError('anchor and positive code IDs must have one value per batch row')
        for proposal in proposals:
            if proposal.source_indices.shape[0] != batch_size:
                raise ValueError('proposal batch dimension does not match candidate batch')
            # -inf marks an ineligible entry; any other non-finite score is malformed, whether or
            # not the merge would reach it.
            malformed = proposal.source_indices.ge(0) & (
                proposal.scores.isnan() | proposal.scores.isposinf()
            )
            if malformed.any():
                row, slot = torch.nonzero(malformed, as_tuple=False)[0].tolist()
                raise ValueError(
                    f'{SelectionReason(proposal.reason).name} proposal for anchor code ID '
                    f'{int(anchor_code_ids[row])} has a malformed score '
                    f'{float(proposal.scores[row, slot])} at source index '
                    f'{int(proposal.source_indices[row, slot])}'
                )

        code_rows = candidates.code_id.tolist()
        valid_rows = candidates.valid_mask.tolist()
        explicit_rows = candidates.is_explicit_exclusion.tolist()
        uid_rows = candidates.candidate_uid.tolist()
        anchors = anchor_code_ids.tolist()
        positives = positive_code_ids.tolist()
        proposal_rows = [
            (proposal.source_indices.tolist(), proposal.scores.tolist(), int(proposal.reason))
            for proposal in proposals
        ]

        selected_rows: List[List[int]] = []
        score_rows: List[List[float]] = []
        reason_rows: List[List[int]] = []
        for row in range(batch_size):
            codes = code_rows[row]
            uids: List[Uid] = [tuple(uid) for uid in uid_rows[row]]
            forbidden = {int(anchors[row]), int(positives[row])}

            available_by_code: Dict[int, int] = {}
            for index in range(pool_size):
                if not valid_rows[row][index] or explicit_rows[row][index]:
                    continue
                code_id = int(codes[index])
                if code_id in forbidden:
                    continue
                previous = available_by_code.get(code_id)
                if previous is None or uids[index] < uids[previous]:
                    available_by_code[code_id] = index

            chosen: List[int] = []
            chosen_scores: List[float] = []
            chosen_reasons: List[int] = []
            chosen_codes: Set[int] = set()

            for indices, scores, reason in proposal_rows:
                if len(chosen) == k:
                    break
                best_score_by_code: Dict[int, float] = {}
                for proposal_slot, index in enumerate(indices[row]):
                    if index < 0 or index >= pool_size or not valid_rows[row][index]:
                        continue
                    code_id = int(codes[index])
                    if code_id not in available_by_code or code_id in chosen_codes:
                        continue
                    score = float(scores[row][proposal_slot])
                    if score == -math.inf:
                        continue  # the ineligible marker used by every strategy
                    best_score_by_code[code_id] = max(
                        score,
                        best_score_by_code.get(code_id, -math.inf),
                    )
                entries = sorted(
                    (-score, code_id, uids[available_by_code[code_id]], code_id, score)
                    for code_id, score in best_score_by_code.items()
                )
                for _, _, _, code_id, score in entries:
                    chosen.append(available_by_code[code_id])
                    chosen_scores.append(score)
                    chosen_reasons.append(reason)
                    chosen_codes.add(code_id)
                    if len(chosen) == k:
                        break

            if len(chosen) < k:
                remaining = sorted(
                    (code_id, uids[index], index) for code_id, index in available_by_code.items()
                    if code_id not in chosen_codes
                )
                for code_id, _, index in remaining:
                    chosen.append(index)
                    chosen_scores.append(-math.inf)
                    chosen_reasons.append(int(SelectionReason.BACKFILL))
                    chosen_codes.add(code_id)
                    if len(chosen) == k:
                        break

            if len(chosen) != k:
                raise ValueError(
                    f'anchor code ID {int(anchors[row])} requested {k} unique negative codes; '
                    f'available {len(available_by_code)} (unique non-exclusion codes)'
                )
            selected_rows.append(chosen)
            score_rows.append(chosen_scores)
            reason_rows.append(chosen_reasons)

        device = candidates.code_id.device
        source_indices = torch.tensor(selected_rows, dtype=torch.long, device=device)
        uid_index = source_indices.unsqueeze(-1).expand(-1, -1, 3)
        source_uid = candidates.candidate_uid.gather(1, uid_index)
        return NegativeSelection(
            source_indices=source_indices,
            source_candidate_uid=source_uid,
            scores=torch.tensor(score_rows, dtype=candidates.embedding.dtype, device=device),
            reasons=torch.tensor(reason_rows, dtype=torch.int8, device=device),
        )
