'''
Shortlist scores for the ITA text-to-code panel (spec ``text-to-code-ita-panel``, 4.3).

Each pair's query ranks every candidate code under the encoder's own distance, as
``decoding.py`` ranks a query. A pair reports one or more codes, and its rank is the best of
their ranks. A reported code's rank counts every candidate at a distance no greater than its own,
so ties go against the truth. The pair's other reported codes rank behind its best one, so they
never count against it: ``rank`` is one plus the number of non-reported candidates at a distance
no greater than the nearest reported code's.

For each coarser level L in ``COARSE_LEVELS`` the same rule reads the shortlist by ancestor. A
candidate matches when its level-L ancestor is the level-L ancestor of a reported code, and
``rank_l{L}`` is one plus the number of non-matching candidates at a distance no greater than the
nearest matching one. Ancestors come from the codebook lineage, so the combined sectors 31-33,
44-45 and 48-49 are one sector each. Distances are computed in float64 on the CPU whatever the
device or dtype of the points.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from typing import Dict, Sequence, Union

import polars as pl
import torch

from naics_embedder.panels.decoding import (
    HIT_KS,
    DistanceFn,
    _as_float64,
    lca_level,
    resolve_distance,
)
from naics_embedder.utils.naics_hierarchy import code_lineage

COARSE_LEVELS = (2, 3, 4, 5)

# -------------------------------------------------------------------------------------------------
# Scorer
# -------------------------------------------------------------------------------------------------

def _rank_behind_matches(distances: torch.Tensor, matched: torch.Tensor) -> torch.Tensor:
    '''One plus the unmatched candidates at a distance no greater than the nearest matched one.'''

    nearest = distances.masked_fill(~matched, math.inf).min(dim=1).values
    return 1 + ((~matched) & (distances <= nearest[:, None])).sum(dim=1)

def _ancestor_matches(
    reported: Sequence[Sequence[str]],
    candidate_codes: Sequence[str],
    level: int,
) -> torch.Tensor:
    '''
    The (Q, C) mask of candidates whose level-L ancestor is a reported code's, per pair.

    Level L's ancestor of a six-digit code is ``code_lineage(code)[L - 2]``, the lineage running
    from the sector (level 2) to the code itself (level 6).
    '''

    ids: Dict[str, int] = {}
    candidate_ids = torch.tensor(
        [ids.setdefault(code_lineage(code)[level - 2], len(ids)) for code in candidate_codes]
    )
    matches = torch.zeros(len(reported), len(candidate_codes), dtype=torch.bool)
    for row, codes in enumerate(reported):
        wanted = torch.tensor([ids[code_lineage(code)[level - 2]] for code in codes])
        matches[row] = torch.isin(candidate_ids, wanted)
    return matches

def _rank_columns(
    ranks: torch.Tensor,
    suffix: str,
    data: Dict[str, list],
    schema: Dict[str, pl.DataType],
) -> None:
    '''Add a rank, its reciprocal and its hits at each k to the columns, named with a suffix.'''

    values = ranks.tolist()
    data[f'rank{suffix}'] = values
    schema[f'rank{suffix}'] = pl.Int64
    data[f'reciprocal_rank{suffix}'] = [1.0 / rank for rank in values]
    schema[f'reciprocal_rank{suffix}'] = pl.Float64
    for k in HIT_KS:
        data[f'hit_at_{k}{suffix}'] = [rank <= k for rank in values]
        schema[f'hit_at_{k}{suffix}'] = pl.Boolean

def score_shortlist(
    query_points: torch.Tensor,
    reported_codes: Sequence[Sequence[str]],
    candidate_points: torch.Tensor,
    candidate_codes: Sequence[str],
    pair_ids: Sequence[str],
    distance: Union[str, DistanceFn] = 'cosine',
) -> pl.DataFrame:
    '''
    Rank every candidate for each pair's query and score the pair's reported codes.

    Args:
        query_points: Query embeddings, (Q, D), one per pair.
        reported_codes: Each pair's reported codes, each a candidate. Order and repeats within a
            pair do not matter.
        candidate_points: Candidate embeddings, (C, D), in the same space.
        candidate_codes: Distinct six-digit candidate codes, one per row of ``candidate_points``.
        pair_ids: Identifiers carried into the result, in the order of the queries.
        distance: A name in ``decoding.DISTANCES`` or a callable taking float64 CPU (Q, D) and
            (C, D) tensors and returning (Q, C) distances.

    Returns:
        One row per pair, in the order given: ``pair_id``, ``codes`` (the reported codes, sorted
        and distinct), ``top1_code``, ``rank`` (Int64), ``reciprocal_rank``, ``hit_at_k`` for each
        k in ``HIT_KS``, ``lca_level`` (Int64) and, for each level L in ``COARSE_LEVELS``,
        ``rank_l{L}``, ``reciprocal_rank_l{L}`` and ``hit_at_{k}_l{L}``. ``top1_code`` is the
        nearest reported code when ``rank`` is 1 and otherwise the nearest candidate that is not
        reported. A tie at that distance goes to the code first in ``candidate_codes``.
        ``lca_level`` is the highest LCA level between ``top1_code`` and any reported code.

    Raises:
        ValueError: On mismatched shapes, repeated candidates, a pair with no reported code, a
            reported code that is not a candidate, or a distance that is not finite.
    '''

    name, distance_fn = resolve_distance(distance)
    queries = _as_float64(query_points, 'query_points')
    candidates = _as_float64(candidate_points, 'candidate_points')
    pair_ids = [str(pair_id) for pair_id in pair_ids]
    candidate_codes = [str(code) for code in candidate_codes]
    if queries.shape[0] != len(reported_codes) or len(pair_ids) != len(reported_codes):
        raise ValueError(
            f'{queries.shape[0]} query points, {len(reported_codes)} reported code lists and '
            f'{len(pair_ids)} pair ids must agree'
        )
    if candidates.shape[0] != len(candidate_codes):
        raise ValueError(
            f'{candidates.shape[0]} candidate points for {len(candidate_codes)} candidate codes'
        )
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            f'queries have {queries.shape[1]} coordinates, candidates {candidates.shape[1]}'
        )
    position = {code: index for index, code in enumerate(candidate_codes)}
    if len(position) != len(candidate_codes):
        raise ValueError('candidate codes must be distinct')
    reported = [sorted({str(code) for code in codes}) for codes in reported_codes]
    empty = [pair_id for pair_id, codes in zip(pair_ids, reported) if not codes]
    if empty:
        raise ValueError(f'{len(empty):,} pairs report no code, e.g. {empty[:5]}')
    unknown = sorted({code for codes in reported for code in codes if code not in position})
    if unknown:
        raise ValueError(f'{len(unknown):,} reported codes are not candidates, e.g. {unknown[:5]}')

    distances = distance_fn(queries, candidates)
    if distances.shape != (len(pair_ids), len(candidate_codes)):
        raise ValueError(f'distance returned shape {tuple(distances.shape)}')
    if not torch.isfinite(distances).all():
        raise ValueError(f'distance {name!r} returned non-finite values')

    is_reported = torch.zeros(len(pair_ids), len(candidate_codes), dtype=torch.bool)
    for row, codes in enumerate(reported):
        is_reported[row, [position[code] for code in codes]] = True
    ranks = _rank_behind_matches(distances, is_reported)
    # argmin returns the first minimum, so a tie goes to the code first in the candidate order
    nearest_reported = distances.masked_fill(~is_reported, math.inf).argmin(dim=1)
    nearest_other = distances.masked_fill(is_reported, math.inf).argmin(dim=1)
    top1_codes = [
        candidate_codes[index]
        for index in torch.where(ranks == 1, nearest_reported, nearest_other).tolist()
    ]

    data: Dict[str, list] = {
        'pair_id': pair_ids,
        'codes': reported,
        'top1_code': top1_codes,
    }
    schema: Dict[str, pl.DataType] = {
        'pair_id': pl.Utf8,
        'codes': pl.List(pl.Utf8),
        'top1_code': pl.Utf8,
    }
    _rank_columns(ranks, '', data, schema)
    data['lca_level'] = [
        max(lca_level(top1, code) for code in codes) for top1, codes in zip(top1_codes, reported)
    ]
    schema['lca_level'] = pl.Int64
    for level in COARSE_LEVELS:
        matches = _ancestor_matches(reported, candidate_codes, level)
        _rank_columns(_rank_behind_matches(distances, matches), f'_l{level}', data, schema)
    return pl.DataFrame(data, schema=schema)
