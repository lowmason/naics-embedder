'''
The runtime structural-eligibility rule must equal the generator's training-pair rule.

Training pairs only keep negatives structurally farther from the anchor than the positive
(``create_triplets._structural_margins``). Candidates sourced at runtime — universe backfill and
the distributed global pool — must pass the same rule before they can occupy a negative slot.
'''

import polars as pl
import pytest
import torch

from naics_embedder.data.create_triplets import _structural_margins
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.margins import structural_margins, structurally_eligible
from naics_embedder.utils.config import SupervisionBuildConfig


def _margins(negative_distance, negative_relation, positive_distance, positive_relation):
    return structural_margins(
        negative_distance=torch.tensor([negative_distance]),
        negative_relation_id=torch.tensor([negative_relation], dtype=torch.int16),
        positive_distance=torch.tensor([positive_distance]),
        positive_relation_id=torch.tensor([positive_relation], dtype=torch.int16),
    )


@pytest.mark.parametrize(
    ('negative', 'positive', 'expected'),
    [
        # Cross-sector negatives always receive the fixed legacy margins.
        ((99.0, 99), (0.5, 1), (15.0, 10.0)),
        # Farther relation at an equal distance receives the fixed equal-distance margin.
        ((2.0, 3), (2.0, 2), (1.0, 0.3333)),
        # The -0.5 lineal adjustment receives its fixed margin when the relation is farther.
        ((1.5, 3), (2.0, 2), (1.0, 0.6667)),
        # Otherwise both margins are raw deltas.
        ((3.0, 7), (0.5, 1), (6.0, 2.5)),
    ],
)
def test_structural_margins_follow_the_generator_special_cases(negative, positive, expected):
    relation_margin, distance_margin = _margins(*negative, *positive)

    assert relation_margin.item() == pytest.approx(expected[0])
    assert distance_margin.item() == pytest.approx(expected[1])


@pytest.mark.parametrize(
    ('negative', 'positive'),
    [
        ((0.5, 1), (1.5, 3)),  # the anchor's parent while the positive is its grandparent
        ((2.0, 2), (2.0, 2)),  # another sibling of a sibling positive: no relation margin
        ((1.5, 3), (3.0, 7)),  # structurally closer by both measures
    ],
)
def test_structurally_closer_or_equal_negatives_are_ineligible(negative, positive):
    eligible = structurally_eligible(
        negative_distance=torch.tensor([negative[0]]),
        negative_relation_id=torch.tensor([negative[1]], dtype=torch.int16),
        positive_distance=torch.tensor([positive[0]]),
        positive_relation_id=torch.tensor([positive[1]], dtype=torch.int16),
    )

    assert eligible.tolist() == [False]


def test_runtime_rule_equals_the_generator_rule_over_every_hierarchy_triple(
    tmp_path, hierarchy_descriptions_parquet
):
    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    index = SupervisionIndex.from_bundle(load_validated_bundle(manifest))
    size = len(index.id_to_code)
    rows = [
        (anchor, positive, negative)
        for anchor in range(size)
        for positive in range(size)
        for negative in range(size)
        if len({anchor, positive, negative}) == 3
    ]
    anchor, positive, negative = (torch.tensor(column) for column in zip(*rows))
    distance = index.structural_distance
    relation = index.structural_relation_id

    relation_margin, distance_margin = structural_margins(
        negative_distance=distance[anchor, negative],
        negative_relation_id=relation[anchor, negative],
        positive_distance=distance[anchor, positive],
        positive_relation_id=relation[anchor, positive],
    )
    eligible = structurally_eligible(
        negative_distance=distance[anchor, negative],
        negative_relation_id=relation[anchor, negative],
        positive_distance=distance[anchor, positive],
        positive_relation_id=relation[anchor, positive],
    )
    kept = _structural_margins(
        pl.DataFrame(
            {
                'anchor': anchor.numpy(),
                'positive': positive.numpy(),
                'negative': negative.numpy(),
                'negative_structural_distance': distance[anchor, negative].numpy(),
                'negative_structural_relation_id': relation[anchor, negative].numpy(),
                'positive_structural_distance': distance[anchor, positive].numpy(),
                'positive_structural_relation_id': relation[anchor, positive].numpy(),
            }
        )
    )

    generator = {
        (row['anchor'], row['positive'], row['negative']):
        (row['relation_margin'], row['distance_margin'])
        for row in kept.iter_rows(named=True)
    }
    runtime = {
        (a, p, n): (relation_margin[i].item(), distance_margin[i].item())
        for i, (a, p, n) in enumerate(rows)
        if eligible[i]
    }
    assert 0 < len(generator) < len(rows)
    assert runtime.keys() == generator.keys()
    for key, (relation_value, distance_value) in generator.items():
        assert runtime[key][0] == pytest.approx(relation_value)
        assert runtime[key][1] == pytest.approx(distance_value)
