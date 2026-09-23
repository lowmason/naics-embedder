'''
Structural negative margins and eligibility at runtime.

The torch mirror of the generator rule ``create_triplets._structural_margins``: an ordinary
negative must be structurally farther from the anchor than the positive. Cross-sector
negatives receive fixed margins; equal distances and the -0.5 lineal adjustment receive fixed
distance margins when the relation margin is positive. Candidates sourced at runtime (universe
backfill, the distributed global pool) pass through the same rule, so the repaired pipeline never
repels an *ordinary* candidate that the generated supervision would not treat as a negative.

Explicit exclusions are exempt: their exclusion is authoritative regardless of structure (the
quota may select a structurally close exclusion, as the contract requires), so callers apply
eligibility to ordinary candidates only.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from typing import Tuple

import torch

from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE,
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
    LINEAL_ADJUSTED_DISTANCE_MARGIN,
    LINEAL_DISTANCE_DELTA,
)

# -------------------------------------------------------------------------------------------------
# Margins and eligibility
# -------------------------------------------------------------------------------------------------

def structural_margins(
    *,
    negative_distance: torch.Tensor,
    negative_relation_id: torch.Tensor,
    positive_distance: torch.Tensor,
    positive_relation_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Relation and distance margins of each negative relative to its anchor's positive.

    Inputs broadcast against each other (e.g. ``[batch, candidate]`` negatives against
    ``[batch, 1]`` positives). Structural values are exact in float32 (half-step distances, small
    relation IDs), so the special-case equality tests match the generator exactly on every device.

    Returns:
        ``(relation_margin, distance_margin)`` as float32 tensors.
    '''

    negative_distance = negative_distance.to(torch.float32)
    relation_delta = negative_relation_id.to(torch.float32) - positive_relation_id.to(torch.float32)
    distance_delta = negative_distance - positive_distance.to(torch.float32)
    negative_distance, relation_delta, distance_delta = torch.broadcast_tensors(
        negative_distance, relation_delta, distance_delta
    )
    cross_sector = negative_distance.eq(CROSS_SECTOR_DISTANCE)
    farther_relation = relation_delta.gt(0)

    relation_margin = relation_delta.masked_fill(cross_sector, CROSS_SECTOR_RELATION_MARGIN)
    # Fill in reverse precedence of the generator's when/then chain so earlier cases win.
    distance_margin = distance_delta.masked_fill(cross_sector, CROSS_SECTOR_DISTANCE_MARGIN)
    distance_margin = distance_margin.masked_fill(
        farther_relation & distance_delta.eq(LINEAL_DISTANCE_DELTA),
        LINEAL_ADJUSTED_DISTANCE_MARGIN,
    )
    distance_margin = distance_margin.masked_fill(
        farther_relation & distance_delta.eq(0.0),
        EQUAL_DISTANCE_MARGIN,
    )
    return relation_margin, distance_margin

def structurally_eligible(
    *,
    negative_distance: torch.Tensor,
    negative_relation_id: torch.Tensor,
    positive_distance: torch.Tensor,
    positive_relation_id: torch.Tensor,
) -> torch.Tensor:
    '''Whether each ordinary negative is structurally farther than the positive (both margins > 0).'''

    relation_margin, distance_margin = structural_margins(
        negative_distance=negative_distance,
        negative_relation_id=negative_relation_id,
        positive_distance=positive_distance,
        positive_relation_id=positive_relation_id,
    )
    return relation_margin.gt(0) & distance_margin.gt(0)
