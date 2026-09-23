import json

import pyarrow.parquet as pq
import pytest
import torch

from naics_embedder.supervision.artifacts import (
    METADATA_BUNDLE,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.index import SupervisionIndex


def test_loader_rejects_mixed_bundle_metadata(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    member = manifest['artifacts']['pair_facts']['files'][0]
    pair_path = generated_bundle.parent / member['path']
    table = pq.read_table(pair_path)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_BUNDLE] = b'bundle-b'
    pq.write_table(table.replace_schema_metadata(metadata), pair_path)
    member['sha256'] = sha256_file(pair_path)
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*bundle-b'):
        load_validated_bundle(
            generated_bundle,
            expected_contract='stage3-supervision-v1',
        )


def test_join_maps_canonical_directions_into_anchor_view(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)
    anchor = torch.tensor([0, 3])
    candidate = torch.tensor([[2], [1]])
    valid = torch.ones((2, 1), dtype=torch.bool)

    joined = index.join(anchor, candidate, valid)

    assert joined.anchor_excludes_candidate.tolist() == [[True], [True]]
    assert joined.candidate_excludes_anchor.tolist() == [[False], [False]]
    assert torch.equal(
        joined.is_explicit_exclusion,
        joined.anchor_excludes_candidate | joined.candidate_excludes_anchor,
    )
    assert joined.structural_distance.tolist() == [[2.0], [99.0]]


def test_join_rejects_unknown_ids_with_anchor_context(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)

    with pytest.raises(ValueError, match='anchor row 0.*candidate code ID 999'):
        index.join(
            torch.tensor([0]),
            torch.tensor([[999]]),
            torch.tensor([[True]]),
        )


# -------------------------------------------------------------------------------------------------
# Additional index behavior
# -------------------------------------------------------------------------------------------------

def test_join_is_directional_from_the_candidate_side(validated_bundle):
    # '111112' (1) is excluded by '222222' (3): viewed from anchor 1 the candidate excludes it.
    index = SupervisionIndex.from_bundle(validated_bundle)

    joined = index.join(torch.tensor([1]), torch.tensor([[3]]), torch.tensor([[True]]))

    assert joined.anchor_excludes_candidate.tolist() == [[False]]
    assert joined.candidate_excludes_anchor.tolist() == [[True]]
    assert joined.semantic_target_id.tolist() == [[2]]
    assert joined.semantic_source_id.tolist() == [[2]]


def test_join_ignores_invalid_padding_ids(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)

    joined = index.join(
        torch.tensor([0]),
        torch.tensor([[2, -1]]),
        torch.tensor([[True, False]]),
    )

    assert joined.is_explicit_exclusion.tolist() == [[True, False]]
    assert joined.semantic_target_id.tolist() == [[2, 0]]


def test_exclusion_code_ids_are_symmetric_and_sorted(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)

    assert index.exclusion_code_ids(0) == (2,)
    assert index.exclusion_code_ids(1) == (3,)
    assert index.exclusion_code_ids(3) == (1,)
    assert index.exclusion_code_ids(4) == ()
    with pytest.raises(ValueError, match='unknown anchor code ID 5'):
        index.exclusion_code_ids(5)


def test_index_matrices_match_the_pair_facts(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)

    assert index.id_to_code == ('111111', '111112', '111113', '222222', '333333')
    assert index.code_to_id['222222'] == 3
    assert index.structural_distance[0, 1] == index.structural_distance[1, 0] == 0.5
    assert index.structural_relation_id[1, 2] == index.structural_relation_id[2, 1] == 3
    assert torch.diagonal(index.structural_distance).eq(0).all()
