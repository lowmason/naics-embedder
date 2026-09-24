'''
The outcome panel and its selection log (Req 3; Req 4; roadmap Stage 2 Exit).

A one-hot stub encoder stands in for a trained arm: each code sits on its own axis, and each query
sits on the axis of the code the stub assigns it, so every rank below is worked out by hand.
'''

import json

import polars as pl
import pytest
import torch

from naics_embedder.panels.decoding import METRIC_NAMES
from naics_embedder.panels.index_roles import role_table_fingerprint
from naics_embedder.panels.outcome import (
    OutcomePanel,
    SealedSplitError,
    SplitAlreadyOpenedError,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

pytestmark = pytest.mark.unit

CANDIDATES = ['111110', '111120', '112130', '211111', '311111']
ROWS = [
    (0, '111110', 'Soybean farming', 'examples'),
    (1, '111110', 'Soybean seed production', 'training'),
    (2, '111110', 'Edamame farming', 'validation'),
    (3, '111110', 'Soybeans, organic', 'test'),
    (4, '111120', 'Oilseed farming', 'examples'),
    (5, '111120', 'Canola farming', 'validation'),
    (6, '111120', 'Sunflower farming', 'test'),
    (7, '211111', 'Crude petroleum extraction', 'examples'),
    (8, '211111', 'Oil sands mining', 'test'),
    (9, '311111', 'Dog food manufacturing', 'examples'),
    (10, '311111', 'Cat food manufacturing', 'validation'),
]
# The stub decodes 'Canola farming' and 'Oil sands mining' to the wrong code
ASSIGNED = {
    'Edamame farming': '111110',
    'Canola farming': '111110',
    'Cat food manufacturing': '311111',
    'Soybeans, organic': '111110',
    'Sunflower farming': '111120',
    'Oil sands mining': '111120',
}

class OneHotStubEncoder:
    '''Each code on its own axis; each query on the axis of the code the stub assigns it.'''

    def __init__(self, candidates, assigned):
        self.axis = {code: index for index, code in enumerate(candidates)}
        self.assigned = assigned

    def _points(self, codes):
        indices = torch.tensor([self.axis[code] for code in codes])
        return torch.nn.functional.one_hot(indices, len(self.axis)).to(torch.float32)

    def encode_codes(self, codes):
        return self._points(codes)

    def encode_queries(self, texts):
        return self._points([self.assigned[text] for text in texts])

@pytest.fixture
def role_rows():
    return pl.DataFrame(
        ROWS,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'logs' / 'selection_log.jsonl')

@pytest.fixture
def panel(role_rows, log):
    return OutcomePanel(role_rows, CANDIDATES, log)

@pytest.fixture
def encoder():
    return OneHotStubEncoder(CANDIDATES, ASSIGNED)

def _events(log):
    return [(record['event'], record['split']) for record in log.records()]

# -------------------------------------------------------------------------------------------------
# Roles and candidates
# -------------------------------------------------------------------------------------------------

def test_every_entry_holds_one_role_and_entryless_codes_are_never_queries(panel):
    panel.open_test('final configuration')
    splits = {
        'training': panel.training_queries(),
        'validation': panel.validation_queries('check the splits'),
        'test': panel.test_queries('check the splits'),
    }
    query_ids = [entry_id for frame in splits.values() for entry_id in frame['entry_id']]

    assert panel.candidates == tuple(CANDIDATES)
    assert panel.entryless_candidates == ('112130', )
    assert sorted(query_ids) == [1, 2, 3, 5, 6, 8, 10]  # entries 0, 4, 7, 9 are examples text
    assert len(query_ids) == len(set(query_ids))
    for frame in splits.values():
        assert '112130' not in frame['code'].to_list()

def test_the_fingerprint_identifies_the_assignment(panel, role_rows):
    assert panel.fingerprint == role_table_fingerprint(role_rows)

@pytest.mark.parametrize(
    'candidates',
    [CANDIDATES[:1] + CANDIDATES, CANDIDATES + ['11111'], CANDIDATES[1:]],
)
def test_candidates_must_be_distinct_six_digit_codes_covering_the_queries(
    role_rows, log, candidates
):
    with pytest.raises(ValueError):
        OutcomePanel(role_rows, candidates, log)

# -------------------------------------------------------------------------------------------------
# Selection log and sealing
# -------------------------------------------------------------------------------------------------

def test_validation_reads_are_logged_and_training_reads_are_not(panel, log):
    assert panel.training_queries().height == 1
    assert log.records() == []

    validation = panel.validation_queries('tune the learning rate')

    assert validation['entry_id'].to_list() == [2, 5, 10]
    [record] = log.records()
    assert record['event'] == 'read'
    assert record['panel'] == 'outcome'
    assert record['split'] == 'validation'
    assert record['purpose'] == 'tune the learning rate'
    assert record['fingerprint'] == panel.fingerprint
    assert record['n_queries'] == 3

def test_the_test_split_is_sealed_until_a_logged_opening(panel, log, encoder):
    with pytest.raises(SealedSplitError):
        panel.test_queries('peek')
    with pytest.raises(SealedSplitError):
        panel.score(encoder, 'test', 'peek')
    assert log.records() == []

    panel.open_test('final configuration')

    assert panel.test_queries('final configuration')['entry_id'].to_list() == [3, 6, 8]
    assert _events(log) == [('open', 'test'), ('read', 'test')]

def test_every_panel_object_must_open_the_test_split_itself(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    later = OutcomePanel(role_rows, CANDIDATES, log)

    with pytest.raises(SealedSplitError):
        later.test_queries('reuse the earlier opening')

def test_a_second_opening_needs_a_reason_and_is_logged_as_a_reopen(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    later = OutcomePanel(role_rows, CANDIDATES, log)

    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        later.open_test('final configuration')
    later.open_test('final configuration', reopen_reason='the first run crashed before scoring')

    assert _events(log) == [('open', 'test'), ('reopen', 'test')]
    assert log.records()[-1]['detail'] == {'reason': 'the first run crashed before scoring'}

def test_openings_are_counted_per_role_assignment(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    reassigned = role_rows.with_columns(
        role=pl.when(pl.col('entry_id') == 1).then(pl.lit('test')).otherwise('role')
    )

    OutcomePanel(reassigned, CANDIDATES, log).open_test('final configuration')

    assert _events(log) == [('open', 'test'), ('open', 'test')]

def test_only_validation_and_test_splits_are_scored(panel, encoder):
    with pytest.raises(ValueError, match='only validation and test'):
        panel.score(encoder, 'training', 'fit check')

def test_log_records_need_a_purpose(log):
    with pytest.raises(ValueError, match='purpose'):
        log.append(
            SelectionEvent.READ,
            panel='outcome',
            split='validation',
            purpose='  ',
            fingerprint='f',
            n_queries=0,
        )

def test_the_log_is_append_only_json_lines(panel, log):
    panel.validation_queries('first')
    panel.validation_queries('second')

    lines = log.path.read_text().splitlines()
    assert [json.loads(line)['purpose'] for line in lines] == ['first', 'second']

# -------------------------------------------------------------------------------------------------
# Exit: a stub encoder scored on the sealed splits
# -------------------------------------------------------------------------------------------------

def test_a_stub_encoder_scores_every_metric_on_both_splits(panel, log, encoder):
    validation = panel.score(encoder, 'validation', 'stub check')
    panel.open_test('stub check on the fixture panel')
    test = panel.score(encoder, 'test', 'stub check on the fixture panel')

    # Validation: entries 2 and 10 decode correctly (rank 1); entry 5 lands on 111110, so its
    # code ties the three other off-axis candidates and ranks last of 5
    assert validation.per_query.select('query_id', 'rank', 'lca_level').rows() == [
        (2, 1, 6),
        (5, 5, 4),
        (10, 1, 6),
    ]
    assert set(METRIC_NAMES) <= set(validation.summary)
    assert validation.summary['top1'] == pytest.approx(2 / 3)
    assert validation.summary['mrr'] == pytest.approx((1 + 1 / 5 + 1) / 3)
    assert validation.summary['hit_at_5'] == 1.0
    assert validation.summary['lca_level'] == pytest.approx(16 / 3)
    # Test: entry 8 (211111) lands on 111120, which shares only the virtual root with it
    assert test.per_query.select('query_id', 'rank', 'lca_level').rows() == [
        (3, 1, 6),
        (6, 1, 6),
        (8, 5, 1),
    ]
    assert test.summary['n_candidates'] == 5
    assert test.summary['lca_level'] == pytest.approx(13 / 3)
    assert _events(log) == [('read', 'validation'), ('open', 'test'), ('read', 'test')]
    assert log.records()[0]['detail'] == {'encoder': 'OneHotStubEncoder', 'distance': 'cosine'}

# -------------------------------------------------------------------------------------------------
# Loading from preprocessing outputs
# -------------------------------------------------------------------------------------------------

def _descriptions(examples):
    return pl.DataFrame(
        {
            'code': CANDIDATES + ['11111'],
            'examples': examples + [None],
        },
        schema={
            'code': pl.Utf8,
            'examples': pl.Utf8
        },
    )

def test_from_files_reads_the_preprocessing_outputs(tmp_path, role_rows):
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    role_rows.write_parquet(roles_path)
    _descriptions(
        [
            'Soybean farming',
            'Oilseed farming',
            None,
            'Crude petroleum extraction',
            'Dog food manufacturing',
        ]
    ).write_parquet(descriptions_path)

    panel = OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')

    assert panel.candidates == tuple(CANDIDATES)
    assert panel.fingerprint == role_table_fingerprint(role_rows)

def test_from_files_refuses_descriptions_that_hold_every_entry(tmp_path, role_rows):
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    role_rows.write_parquet(roles_path)
    everything = role_rows.sort('entry_id').group_by('code', maintain_order=True).agg(
        pl.col('text').str.join('; ')
    )
    by_code = dict(everything.iter_rows())
    _descriptions([by_code.get(code) for code in CANDIDATES]).write_parquet(descriptions_path)

    with pytest.raises(ValueError, match='examples channel other than'):
        OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')
