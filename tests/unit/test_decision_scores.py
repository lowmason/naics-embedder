'''
Each panel's per-unit scores (D10): per-query metrics by code, and per-row squared errors by
four-digit group with each row's repeats averaged first.
'''

import polars as pl
import pytest
import torch

from naics_embedder.decision.scores import (
    PANELS,
    SCORE_COLUMNS,
    outcome_scores,
    panel_statistic,
    regressor_scores,
    seed_scores,
)
from naics_embedder.panels.decoding import METRIC_NAMES, score_decoding
from naics_embedder.panels.regressor import PREDICTION_COLUMNS

pytestmark = pytest.mark.unit

def _per_query():
    # Query 7 decodes to its code (rank 1); query 8 ties its code with two others (rank 3)
    codes = ['111110', '111120', '211111']
    return score_decoding(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ['111110', '211111'],
        torch.eye(3),
        codes,
        distance='euclidean',
        query_ids=[7, 8],
    ).per_query

def _prediction(panel, comparator, repeat, code, year, prediction, outcome=1.0):
    return {
        'panel': panel,
        'split': 'validation',
        'level': 6,
        'comparator': comparator,
        'repeat': repeat,
        'fold': 0,
        'code': code,
        'group': code[:4],
        'feature_year': year,
        'outcome_year': year + 1,
        'alpha': 1.0,
        'outcome': outcome,
        'prediction': prediction,
    }

def _predictions(rows):
    schema = {
        'panel': pl.Utf8,
        'split': pl.Utf8,
        'level': pl.Int32,
        'comparator': pl.Utf8,
        'repeat': pl.Int32,
        'fold': pl.Int32,
        'code': pl.Utf8,
        'group': pl.Utf8,
        'feature_year': pl.Int32,
        'outcome_year': pl.Int32,
        'alpha': pl.Float64,
        'outcome': pl.Float64,
        'prediction': pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient='row').select(PREDICTION_COLUMNS)

def test_outcome_items_are_queries_resampled_by_their_code():
    scores = outcome_scores(_per_query())

    assert scores.columns == list(SCORE_COLUMNS)
    assert sorted(scores.get_column('statistic').unique().to_list()) == sorted(METRIC_NAMES)
    mrr = scores.filter(pl.col('statistic') == 'mrr').sort('item')
    assert mrr.select('unit', 'item', 'value').rows() == [
        ('111110', '7', 1.0), ('211111', '8', 1 / 3)
    ]
    assert panel_statistic(scores, 'outcome', 'mrr') == pytest.approx((1 + 1 / 3) / 2)

def test_a_rows_repeats_are_averaged_before_it_counts_once():
    predictions = _predictions(
        [
            # 111111 in 2022: errors 1 and 9 over its two repeats, so 5; in 2023: 0 and 4, so 2
            _prediction('regressor_heldout', 'covariates+embedding', 0, '111111', 2022, 2.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '111111', 2022, 4.0),
            _prediction('regressor_heldout', 'covariates+embedding', 0, '111111', 2023, 1.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '111111', 2023, 3.0),
            _prediction('regressor_heldout', 'covariates+embedding', 0, '222211', 2022, 1.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '222211', 2022, 1.0),
        ]
    )

    scores = regressor_scores(predictions, repeats=2)

    assert scores.select('unit', 'item', 'feature_year', 'value').rows() == [
        ('1111', '111111/2022', 2022, 5.0),
        ('1111', '111111/2023', 2023, 2.0),
        ('2222', '222211/2022', 2022, 0.0),
    ]
    assert panel_statistic(scores, 'regressor_heldout',
                           'covariates+embedding') == pytest.approx(7 / 3)

def test_a_row_without_every_repeat_is_refused():
    predictions = _predictions(
        [_prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 1.0)]
    )

    with pytest.raises(ValueError, match='do not have 2 predictions'):
        regressor_scores(predictions, repeats=2)

@pytest.mark.parametrize(
    'column, value', [('split', 'test'), ('level', 5), ('panel', 'regressor_other')]
)
def test_only_level_six_validation_rows_of_the_two_regimes_are_scored(column, value):
    row = _prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 1.0)

    with pytest.raises(ValueError, match='validation rows'):
        regressor_scores(_predictions([{**row, column: value}]), repeats=1)

def test_a_seed_is_scored_on_all_three_panels():
    rows = [
        _prediction(panel, 'covariates+embedding', 0, '111111', 2023, 1.0)
        for panel in ('regressor_seen', 'regressor_heldout')
    ]

    scores = seed_scores(_per_query(), _predictions(rows), repeats=1)

    assert sorted(scores.get_column('panel').unique().to_list()) == sorted(PANELS)
    with pytest.raises(ValueError, match='regressor_heldout'):
        seed_scores(_per_query(), _predictions(rows[:1]), repeats=1)
