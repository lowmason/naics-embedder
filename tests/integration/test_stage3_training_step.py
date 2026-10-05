'''
The checked negative selection keeps every field of a candidate together.

The tests that drove the old objective's training step through the selection coordinator went
with that step (Req 10, 11). This one checks the selection alone, which spec 4.5 deletes with the
rest of the old objective's machinery.
'''

import torch

from naics_embedder.supervision.candidates import NegativeSelection
from naics_embedder.supervision.schema import SelectionReason

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
    checked_pairs = list(zip(
        selected.embedding[0, :, 0].tolist(),
        selected.code_id[0].tolist(),
    ))
    assert checked_pairs == [(3.0, 103), (1.0, 101), (2.0, 102)]
