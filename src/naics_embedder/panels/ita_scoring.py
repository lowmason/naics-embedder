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

``summarize_ita`` reads a part's scores in four reports (4.3). A pair is copied when its title or
index is copied (U38). The headline is the mean over the pairs that are not copied, each pair once.
The copied report is the same over the copied pairs. The phrasing report is the mean over
description groups of each group's mean over its non-copied pairs. The counts cover every pair of
the part. Mojibake, company-fallback and conflict flags only add to the counts.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

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

# -------------------------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------------------------

def _metric_columns() -> Dict[str, str]:
    '''Each summary metric and the per-pair column it averages, in the order reports list them.'''

    columns = {
        'mrr': 'reciprocal_rank',
        'top1': 'hit_at_1',
        **{
            f'hit_at_{k}': f'hit_at_{k}'
            for k in HIT_KS
        },
        'lca_level': 'lca_level',
    }
    for level in COARSE_LEVELS:
        columns[f'mrr_l{level}'] = f'reciprocal_rank_l{level}'
        for k in HIT_KS:
            columns[f'hit_at_{k}_l{level}'] = f'hit_at_{k}_l{level}'
    return columns

# A boolean metric averages as a fraction. ``top1`` and ``hit_at_1`` are one quantity
_METRIC_COLUMNS = _metric_columns()
SUMMARY_METRICS = tuple(_METRIC_COLUMNS)

# Each count and the pair-frame flag it adds up
_COUNT_FLAGS = {
    'n_copied': 'copied',
    'n_conflict': 'conflict',
    'n_mojibake': 'mojibake',
    'n_company_fallback': 'company_fallback',
}

@dataclass(frozen=True)
class ItaResult:
    '''
    A part's ITA scores at three grains and in four reports.

    Every mean is correctly rounded and independent of row order, so the same scores always give
    the same values, bit for bit.

    Attributes:
        per_pair: One row per pair, in the scorer's order: the scorer's columns, then
            ``description_key`` and ``copied``. The decision's bootstrap resamples it by
            description group.
        per_code: One row per reported code, sorted by ``code``: ``code``, ``n_pairs`` and each
            metric's mean over the non-copied pairs that report it. A pair that reports several
            codes counts under each.
        headline: ``n_pairs``, ``distance`` and each ``SUMMARY_METRICS`` mean over the pairs that
            are not copied.
        copied: The same over the copied pairs. With none, ``n_pairs`` is 0 and every metric None.
        phrasing: ``n_groups``, the description groups with a non-copied pair, and each metric's
            mean over those groups of the group's mean over its non-copied pairs.
        counts: ``n_pairs``, ``n_groups``, ``n_copied``, ``n_conflict``, ``n_mojibake`` and
            ``n_company_fallback``, over every pair of the part, copied ones included.
        data_fingerprint: The part's data fingerprint, which names the table build the scores were
            read on. ``ItaPanel`` sets it from its read's logged detail. ``summarize_ita`` leaves
            it None.
    '''

    per_pair: pl.DataFrame
    per_code: pl.DataFrame
    headline: Dict[str, Union[int, float, str, None]]
    copied: Dict[str, Union[int, float, str, None]]
    phrasing: Dict[str, Union[int, float, str, None]]
    counts: Dict[str, int]
    data_fingerprint: Optional[str] = None

# The distinct per-pair columns the metrics average: ``top1`` and ``hit_at_1`` share one
_METRIC_SOURCES = list(dict.fromkeys(_METRIC_COLUMNS.values()))

def _mean(values: Sequence[float]) -> Optional[float]:
    '''
    The correctly rounded mean of the values, whatever their order, or None when there are none.

    A float sum depends on its order and, in Polars, on how the column is split into chunks, which a
    parallel ``group_by`` varies from one run to the next. ``math.fsum`` has one result for the same
    values, so the same scores always give the same report, bit for bit.
    '''

    return math.fsum(values) / len(values) if values else None

def _metric_means(frame: pl.DataFrame) -> Dict[str, Optional[float]]:
    return {
        name: _mean(frame.get_column(column).to_list())
        for name, column in _METRIC_COLUMNS.items()
    }

def _group_values(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    '''One row per ``key``: ``n_pairs`` and, for each source column, the group's values as a list.'''

    return frame.group_by(key).agg(
        pl.len().cast(pl.Int64).alias('n_pairs'), *(pl.col(column) for column in _METRIC_SOURCES)
    )

def _group_means(groups: pl.DataFrame) -> Dict[str, List[Optional[float]]]:
    '''Each metric's mean within each group of ``_group_values``, in the groups' order.'''

    return {
        name: [_mean(values) for values in groups.get_column(column).to_list()]
        for name, column in _METRIC_COLUMNS.items()
    }

def _check_summary_inputs(scores: pl.DataFrame, pairs: pl.DataFrame) -> None:
    for label, frame in (('scores', scores), ('pairs', pairs)):
        repeated = frame.filter(pl.col('pair_id').is_duplicated()).get_column('pair_id')
        if not repeated.is_empty():
            raise ValueError(f'{label} repeat pair ids: {repeated.unique().sort().to_list()[:5]}')
    scored = set(scores.get_column('pair_id'))
    listed = set(pairs.get_column('pair_id'))
    if scored != listed:
        raise ValueError(
            f'scores and pairs name different pair ids: {len(scored - listed):,} only scored, '
            f'e.g. {sorted(scored - listed)[:5]}, and {len(listed - scored):,} only in pairs, '
            f'e.g. {sorted(listed - scored)[:5]}'
        )
    for column in ('description_key', *_COUNT_FLAGS.values()):
        if pairs.get_column(column).null_count():
            raise ValueError(f'pairs have a null {column!r}')
    for column in _METRIC_SOURCES:
        if scores.get_column(column).null_count():
            raise ValueError(f'scores have a null {column!r}')

def summarize_ita(scores: pl.DataFrame, pairs: pl.DataFrame, distance: str) -> ItaResult:
    '''
    Read a part's shortlist scores as the headline, copied, phrasing and count reports.

    Args:
        scores: ``score_shortlist``'s frame for the part, one row per panel pair.
        pairs: The part's panel pairs, in ``PANEL_PAIR_SCHEMA``, with the same pair ids.
        distance: The name of the distance the scores were read under, carried into the headline
            and the copied report.

    Returns:
        The part's ``ItaResult``.

    Raises:
        ValueError: When a pair id repeats in either frame, the two frames' pair ids differ, a
            column the reports read is null in either frame, or the part has no pair that is not
            copied, so no headline pair.
    '''

    _check_summary_inputs(scores, pairs)
    per_pair = (
        scores.with_row_index('_row').join(
            pairs.select('pair_id', 'description_key', 'copied'), on='pair_id', how='left'
        ).sort('_row').drop('_row')
    )
    headline_pairs = per_pair.filter(~pl.col('copied'))
    if headline_pairs.is_empty():
        raise ValueError(
            f'the part has no headline pair: none of its {per_pair.height:,} pairs is without a '
            f'copied title or index'
        )

    def report(frame: pl.DataFrame) -> Dict[str, Union[int, float, str, None]]:
        return {'n_pairs': frame.height, 'distance': distance, **_metric_means(frame)}

    groups = _group_values(headline_pairs, 'description_key')
    pair_codes = (
        headline_pairs.select('codes', *_METRIC_SOURCES).explode('codes').rename({'codes': 'code'})
    )
    by_code = _group_values(pair_codes, 'code').sort('code')
    per_code = pl.DataFrame(
        {
            'code': by_code.get_column('code'),
            'n_pairs': by_code.get_column('n_pairs'),
            **_group_means(by_code),
        },
        schema={
            'code': pl.Utf8,
            'n_pairs': pl.Int64,
            **{
                name: pl.Float64
                for name in SUMMARY_METRICS
            },
        },
    )
    counts = {
        'n_pairs': pairs.height,
        'n_groups': pairs.get_column('description_key').n_unique(),
        **{
            name: int(pairs.get_column(flag).sum())
            for name, flag in _COUNT_FLAGS.items()
        },
    }
    return ItaResult(
        per_pair=per_pair,
        per_code=per_code,
        headline=report(headline_pairs),
        copied=report(per_pair.filter(pl.col('copied'))),
        phrasing={
            'n_groups': groups.height,
            **{
                name: _mean(means)
                for name, means in _group_means(groups).items()
            },
        },
        counts=counts,
    )
