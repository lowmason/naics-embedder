'''
Each panel's per-unit scores and its decision statistic (D10).

Scores are long: one row per panel, statistic and item, with the item's resampling unit.

- **Outcome panel.** An item is a validation query and its unit is its true code. Every Req 3
  metric is a statistic; the decision statistic is ``mrr``, the per-query reciprocal rank that D6
  already selects checkpoints on.
- **Regressor regimes.** An item is a level-6 code in a feature year and its unit is its
  four-digit group. Every Req 2 comparator is a statistic, valued at the row's squared error
  averaged over the read's repeats, so a row counts once however many repeats scored it. The
  decision statistic is the mean squared error of ``covariates+embedding``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Dict, Tuple

import polars as pl

from naics_embedder.panels.decoding import HIT_KS
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import DECISION_LEVEL, PANEL_NAMES, VALIDATION, Regime

SEEN_PANEL = PANEL_NAMES[Regime.SEEN]
HELDOUT_PANEL = PANEL_NAMES[Regime.HELDOUT]
# D8: the outcome panel and the two regressor regimes
PANELS = (OUTCOME_PANEL, SEEN_PANEL, HELDOUT_PANEL)
REGRESSOR_PANELS = (SEEN_PANEL, HELDOUT_PANEL)
EMBEDDING_COMPARATOR = 'covariates+embedding'
DECISION_STATISTIC = {
    OUTCOME_PANEL: 'mrr',
    SEEN_PANEL: EMBEDDING_COMPARATOR,
    HELDOUT_PANEL: EMBEDDING_COMPARATOR,
}
# Δ = orientation × (A − B) is positive when it favours A: MRR is higher-better, MSE lower-better
ORIENTATION = {OUTCOME_PANEL: 1.0, SEEN_PANEL: -1.0, HELDOUT_PANEL: -1.0}
# Req 1's gain is over the sparse encoding each regime can use (D10)
SPARSE_COMPARATOR = {
    SEEN_PANEL: 'covariates+one_hot',
    HELDOUT_PANEL: 'covariates+ancestors',
}
STATISTIC_DEFINITIONS = {
    OUTCOME_PANEL: 'mean reciprocal rank over validation queries; unit: a code with its queries',
    SEEN_PANEL: (
        'mean squared error of covariates+embedding on level-6 log employment, each row averaged '
        'over its repeats; unit: a four-digit group'
    ),
    HELDOUT_PANEL: (
        'mean squared error of covariates+embedding on level-6 log employment, each row averaged '
        'over its repeats; unit: a four-digit group'
    ),
}
SCORE_SCHEMA = {
    'panel': pl.Utf8,
    'statistic': pl.Utf8,
    'unit': pl.Utf8,
    'item': pl.Utf8,
    'feature_year': pl.Int32,
    'value': pl.Float64,
}
SCORE_COLUMNS: Tuple[str, ...] = tuple(SCORE_SCHEMA)

# -------------------------------------------------------------------------------------------------
# Per-unit scores
# -------------------------------------------------------------------------------------------------

def outcome_scores(per_query: pl.DataFrame) -> pl.DataFrame:
    '''
    The outcome panel's scores from ``DecodingResult.per_query``: every Req 3 metric per query.

    Raises:
        ValueError: If a query id repeats.
    '''

    if per_query.get_column('query_id').is_duplicated().any():
        raise ValueError('the decoding scores repeat a query id')
    metrics = {
        'top1': pl.col('rank') == 1,
        'mrr': pl.col('reciprocal_rank'),
        **{
            f'hit_at_{k}': pl.col(f'hit_at_{k}')
            for k in HIT_KS
        },
        'lca_level': pl.col('lca_level'),
    }
    wide = per_query.select(
        unit=pl.col('code'),
        item=pl.col('query_id').cast(pl.Utf8),
        **{
            name: expr.cast(pl.Float64)
            for name, expr in metrics.items()
        },
    )
    long = wide.unpivot(
        index=['unit', 'item'], on=list(metrics), variable_name='statistic', value_name='value'
    )
    return long.with_columns(
        panel=pl.lit(OUTCOME_PANEL),
        feature_year=pl.lit(None, dtype=pl.Int32),
    ).select(SCORE_COLUMNS).cast(SCORE_SCHEMA)

def regressor_scores(predictions: pl.DataFrame, repeats: int) -> pl.DataFrame:
    '''
    Both regimes' scores from level-6 validation predictions: each comparator's squared error per
    row (a code in a feature year), averaged over the row's repeats.

    Raises:
        ValueError: If a row is not a level-6 validation row of a regressor panel, or a row does not
            have exactly ``repeats`` predictions under every comparator of its panel.
    '''

    outside = predictions.filter(
        (pl.col('split') != VALIDATION) | (pl.col('level') != DECISION_LEVEL)
        | ~pl.col('panel').is_in(list(REGRESSOR_PANELS))
    )
    if outside.height:
        raise ValueError(
            f'{outside.height:,} predictions are not level-{DECISION_LEVEL} validation rows of '
            f'{list(REGRESSOR_PANELS)}'
        )
    keys = ['panel', 'comparator', 'group', 'code', 'feature_year']
    # yapf: disable
    rows = (
        predictions
        .with_columns(error=(pl.col('prediction') - pl.col('outcome'))**2)
        .group_by(keys)
        .agg(n=pl.len(), value=pl.col('error').mean())
    )
    # yapf: enable
    uneven = rows.filter(pl.col('n') != repeats)
    if uneven.height:
        first = uneven.row(0, named=True)
        raise ValueError(
            f'{uneven.height:,} rows do not have {repeats} predictions each, e.g. '
            f'{first["panel"]} {first["comparator"]} {first["code"]}/{first["feature_year"]}: '
            f'{first["n"]}'
        )
    # A comparator entirely absent for a row forms no group above, so it would silently vanish
    # from the output rather than being caught as uneven; every row of a panel must appear under
    # every comparator that panel has
    expected = rows.select('panel', 'code', 'feature_year').unique().join(
        rows.select('panel', 'comparator').unique(), on='panel', how='inner'
    )
    missing = expected.join(
        rows.select('panel', 'comparator', 'code', 'feature_year'),
        on=['panel', 'comparator', 'code', 'feature_year'],
        how='anti',
    )
    if missing.height:
        first = missing.row(0, named=True)
        raise ValueError(
            f'{missing.height:,} (panel, comparator, row) combinations are missing entirely, e.g. '
            f'{first["panel"]} {first["comparator"]} {first["code"]}/{first["feature_year"]}'
        )
    return rows.select(
        'panel',
        statistic=pl.col('comparator'),
        unit=pl.col('group'),
        item=pl.concat_str(pl.col('code'), pl.col('feature_year').cast(pl.Utf8), separator='/'),
        feature_year=pl.col('feature_year'),
        value=pl.col('value'),
    ).cast(SCORE_SCHEMA).sort('panel', 'statistic', 'unit', 'item')

def seed_scores(per_query: pl.DataFrame, predictions: pl.DataFrame, repeats: int) -> pl.DataFrame:
    '''
    One seed's scores on all three panels.

    Raises:
        ValueError: If a panel has no scores.
    '''

    scores = pl.concat([outcome_scores(per_query), regressor_scores(predictions, repeats)])
    missing = sorted(set(PANELS) - set(scores.get_column('panel').unique().to_list()))
    if missing:
        raise ValueError(f'no scores for {missing}')
    return scores

def statistic_values(scores: pl.DataFrame, panel: str, statistic: str) -> pl.DataFrame:
    '''
    One statistic's items (``unit``, ``item``, ``feature_year``, ``value``) on one panel.

    Raises:
        ValueError: If the scores hold no such items.
    '''

    rows = scores.filter((pl.col('panel') == panel) & (pl.col('statistic') == statistic))
    if not rows.height:
        raise ValueError(f'no {statistic!r} scores on {panel}')
    return rows.select('unit', 'item', 'feature_year', 'value')

def panel_statistic(scores: pl.DataFrame, panel: str, statistic: str) -> float:
    '''A statistic's mean over its items: one seed's value on every unit.'''

    return float(statistic_values(scores, panel, statistic).get_column('value').mean())

def statistic_means(scores: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    '''Every statistic's mean over its items, by panel.'''

    means: Dict[str, Dict[str, float]] = {}
    for panel, statistic, value in scores.group_by('panel', 'statistic').agg(
        pl.col('value').mean()
    ).sort('panel', 'statistic').iter_rows():
        means.setdefault(panel, {})[statistic] = float(value)
    return means
