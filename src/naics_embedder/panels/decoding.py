'''
Text-to-code decoding scores for the outcome panel (Req 3).

Each query decodes to the nearest candidate code under the arm's own distance. Ties are broken
against the truth: the true code's rank counts every candidate at a distance no greater than its
own, so an encoder that places every point alike ranks the truth last, never first. Distances
are computed in float64 on the CPU whatever the device or dtype of the points.

Metrics per query: exact top-1 accuracy, reciprocal rank, Hit@k for k in ``HIT_KS``, and the
level of the lowest common ancestor of the top-1 code and the truth (6 for the same six-digit
code, 2 for a shared sector, 1 for the virtual root above the sectors).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F

from naics_embedder.utils.naics_hierarchy import naics_parent_code

HIT_KS = (1, 5, 10)
METRIC_NAMES = ('top1', 'mrr', *(f'hit_at_{k}' for k in HIT_KS), 'lca_level')
DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# -------------------------------------------------------------------------------------------------
# Distances (float64 CPU inputs, one row per point)
# -------------------------------------------------------------------------------------------------

def euclidean_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''Euclidean distances, (Q, C), without the matrix-product shortcut that loses precision.'''

    return torch.cdist(queries, candidates, compute_mode='donot_use_mm_for_euclid_dist')

def cosine_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''One minus cosine similarity, (Q, C).'''

    return 1.0 - F.normalize(queries, dim=1) @ F.normalize(candidates, dim=1).T

def lorentz_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''
    Geodesic distances on the curvature -1 hyperboloid, (Q, C), for (time, space) rows.

    The time coordinate is re-derived from the spatial ones, as ``metrics.core``'s
    ``lorentz_distance_matrix`` does, so a point rounded off the hyperboloid is read at its
    spatial position.
    '''

    q_space, c_space = queries[:, 1:], candidates[:, 1:]
    q_time = torch.sqrt(1.0 + (q_space * q_space).sum(dim=1))
    c_time = torch.sqrt(1.0 + (c_space * c_space).sum(dim=1))
    return torch.acosh(torch.clamp(torch.outer(q_time, c_time) - q_space @ c_space.T, min=1.0))

DISTANCES: Dict[str, DistanceFn] = {
    'euclidean': euclidean_distances,
    'cosine': cosine_distances,
    'lorentz': lorentz_distances,
}

def resolve_distance(distance: Union[str, DistanceFn]) -> Tuple[str, DistanceFn]:
    '''A registered distance by name, or a callable with its ``__name__``.'''

    if callable(distance):
        return getattr(distance, '__name__', 'custom'), distance
    if distance not in DISTANCES:
        raise ValueError(f'unknown distance {distance!r}; expected one of {sorted(DISTANCES)}')
    return distance, DISTANCES[distance]

# -------------------------------------------------------------------------------------------------
# Hierarchical partial credit
# -------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def code_lineage(code: str) -> Tuple[str, ...]:
    '''The code's ancestors from its sector down to the code itself.'''

    chain = [code]
    parent = naics_parent_code(code)
    while parent is not None:
        chain.append(parent)
        parent = naics_parent_code(parent)
    return tuple(reversed(chain))

def lca_level(code_a: str, code_b: str) -> int:
    '''
    Level of the lowest common ancestor of two codes, counting the virtual root as level 1.

    A shared sector is level 2 and the same six-digit code level 6; combined sectors (31-33,
    44-45, 48-49) count as one sector.
    '''

    shared = 0
    for ancestor_a, ancestor_b in zip(code_lineage(code_a), code_lineage(code_b)):
        if ancestor_a != ancestor_b:
            break
        shared += 1
    return shared + 1

# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodingResult:
    '''
    Decoding scores at three grains.

    Attributes:
        per_query: One row per query: ``query_id``, ``code``, ``top1_code``, ``rank``,
            ``reciprocal_rank``, ``hit_at_k`` per k and ``lca_level``. Stage 4 resamples it by
            code.
        per_code: One row per true code: ``code``, ``n_queries`` and each metric's mean.
        summary: Query-weighted means of every metric, with ``n_queries``, ``n_codes``,
            ``n_candidates`` and the distance's name.
    '''

    per_query: pl.DataFrame
    per_code: pl.DataFrame
    summary: Dict[str, Union[int, float, str]]

def _as_float64(points: torch.Tensor, name: str) -> torch.Tensor:
    if points.dim() != 2 or points.shape[0] == 0:
        raise ValueError(f'{name} must be a non-empty 2-D tensor, got shape {tuple(points.shape)}')
    # Move before casting: MPS has no float64
    return points.detach().cpu().to(torch.float64).contiguous()

def _metric_exprs():
    return [
        (pl.col('rank') == 1).alias('top1'),
        pl.col('reciprocal_rank').alias('mrr'),
        *(pl.col(f'hit_at_{k}') for k in HIT_KS),
        pl.col('lca_level'),
    ]

def score_decoding(
    query_points: torch.Tensor,
    query_codes: Sequence[str],
    candidate_points: torch.Tensor,
    candidate_codes: Sequence[str],
    distance: Union[str, DistanceFn] = 'cosine',
    query_ids: Optional[Sequence[int]] = None,
) -> DecodingResult:
    '''
    Decode every query to its nearest candidate and score it against the query's true code.

    Args:
        query_points: Query embeddings, (Q, D).
        query_codes: The true code of each query.
        candidate_points: Candidate embeddings, (C, D), in the same space.
        candidate_codes: Distinct candidate codes, one per row of ``candidate_points``.
        distance: A name in ``DISTANCES`` or a callable taking float64 CPU (Q, D) and (C, D)
            tensors and returning (Q, C) distances.
        query_ids: Identifiers carried into ``per_query`` (index entry ids); 0..Q-1 if omitted.

    Raises:
        ValueError: On mismatched shapes, repeated candidates, a query code that is not a
            candidate, or a distance that is not finite.
    '''

    name, distance_fn = resolve_distance(distance)
    queries = _as_float64(query_points, 'query_points')
    candidates = _as_float64(candidate_points, 'candidate_points')
    query_codes = [str(code) for code in query_codes]
    candidate_codes = [str(code) for code in candidate_codes]
    query_ids = list(range(len(query_codes))) if query_ids is None else list(query_ids)
    if queries.shape[0] != len(query_codes) or len(query_ids) != len(query_codes):
        raise ValueError(
            f'{queries.shape[0]} query points, {len(query_codes)} query codes and '
            f'{len(query_ids)} query ids must agree'
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
    unknown = sorted({code for code in query_codes if code not in position})
    if unknown:
        raise ValueError(f'{len(unknown):,} query codes are not candidates, e.g. {unknown[:5]}')

    distances = distance_fn(queries, candidates)
    if distances.shape != (len(query_codes), len(candidate_codes)):
        raise ValueError(f'distance returned shape {tuple(distances.shape)}')
    if not torch.isfinite(distances).all():
        raise ValueError(f'distance {name!r} returned non-finite values')

    rows = torch.arange(len(query_codes))
    true_index = torch.tensor([position[code] for code in query_codes])
    ranks = (distances <= distances[rows, true_index][:, None]).sum(dim=1)
    runner_up = distances.clone()
    runner_up[rows, true_index] = math.inf
    top1_index = torch.where(ranks == 1, true_index, runner_up.argmin(dim=1))
    top1_codes = [candidate_codes[index] for index in top1_index.tolist()]

    per_query = pl.DataFrame(
        {
            'query_id': query_ids,
            'code': query_codes,
            'top1_code': top1_codes,
            'rank': ranks.tolist(),
        },
        schema={
            'query_id': pl.Int64,
            'code': pl.Utf8,
            'top1_code': pl.Utf8,
            'rank': pl.Int64
        },
    ).with_columns(
        reciprocal_rank=1.0 / pl.col('rank'),
        **{f'hit_at_{k}': pl.col('rank') <= k
           for k in HIT_KS},
        lca_level=pl.Series(
            [lca_level(truth, top1) for truth, top1 in zip(query_codes, top1_codes)],
            dtype=pl.Int64,
        ),
    )
    means = per_query.select(expr.mean() for expr in _metric_exprs()).row(0, named=True)
    # yapf: disable
    per_code = (
        per_query
        .group_by('code')
        .agg(pl.len().alias('n_queries'), *(expr.mean() for expr in _metric_exprs()))
        .sort('code')
    )
    # yapf: enable
    summary: Dict[str, Union[int, float, str]] = {
        'distance': name,
        'n_queries': per_query.height,
        'n_codes': per_code.height,
        'n_candidates': len(candidate_codes),
        **{
            metric: float(means[metric])
            for metric in METRIC_NAMES
        },
    }
    return DecodingResult(per_query=per_query, per_code=per_code, summary=summary)
