'''
The ITA panel: logged reads, the data fingerprint and the sealed opening (plan 17, Task 5; spec
text-to-code-ita-panel, 4.4).

``ItaPanel`` mirrors ``OutcomePanel`` under its own panel name, ``ita_text``. Every read is logged
before a query is encoded, the sealed part needs a logged opening by the same panel object, and a
second opening is keyed on the split fingerprint, so a rebuilt pair table cannot pass it as a
first. Each test builds its tables from ``tests/fixtures/ita.py`` and writes under ``tmp_path``.
'''

import hashlib
import json
from typing import Sequence

import polars as pl
import pytest
import torch

from naics_embedder.decision.sweep import _log_records
from naics_embedder.panels.ita import READ_ROLES, ItaPanel
from naics_embedder.panels.ita_split import ITA_PANEL, ItaPart, split_fingerprint
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
from tests.fixtures.ita import (
    ITA_CANDIDATES,
    ITA_SETTINGS,
    AssignedStubEncoder,
    ita_tables,
    write_ita_tables,
)

pytestmark = pytest.mark.unit

ARM_DETAIL = {'arm': 'lexical-stub', 'role': 'candidate', 'run': 'ita/lexical-stub/1'}

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

def _node(code: str) -> str:
    return f'naics:2022:national_industry:{code}'

def _reported(code: str) -> dict:
    '''A headline-basis row on a code, to build a pair-code row with ``ita_tables``.'''

    return {
        'naics_code': code,
        'naics_year': 2022,
        'naics_2022_node_id': _node(code),
        'flag_headline_basis': True,
    }

def _panel(log: SelectionLog, *tables: pl.DataFrame, candidates=ITA_CANDIDATES) -> ItaPanel:
    pairs, codes = tables or ita_tables()
    return ItaPanel(pairs, codes, candidates, ITA_SETTINGS, log)

class RaisingEncoder:
    '''An encoder that fails as soon as it is asked to encode anything.'''

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        raise RuntimeError('encode_queries failed')

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        raise RuntimeError('encode_codes failed')

class ScaledStubEncoder(AssignedStubEncoder):
    '''
    The stub with 111110's point ten times as long. Cosine reads directions only, so it ranks as
    the stub does; the Euclidean distance from a query on 111110's axis to that point is 9, beyond
    every other code's sqrt(2).
    '''

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        points = super().encode_codes(codes)
        points[[code == '111110' for code in codes]] *= 10.0
        return points

def _rank(result, pair_id: str) -> int:
    return result.per_pair.filter(pl.col('pair_id') == pair_id).get_column('rank').item()

# -------------------------------------------------------------------------------------------------
# Reads
# -------------------------------------------------------------------------------------------------

def test_a_validation_read_is_logged_before_any_query_is_encoded(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)

    with pytest.raises(RuntimeError, match='encode_'):
        panel.score(RaisingEncoder(), 'validation', 'check', detail=ARM_DETAIL)

    (record, ) = log.records()
    assert record['event'] == SelectionEvent.READ.value
    assert record['panel'] == 'ita_text'
    assert record['split'] == 'validation'
    assert record['purpose'] == 'check'
    assert record['fingerprint'] == split_fingerprint(ITA_SETTINGS)
    assert record['n_queries'] == 6
    assert record['detail'] == {
        'encoder': 'RaisingEncoder',
        'distance': 'cosine',
        'data_fingerprint': panel.data_fingerprint('validation'),
        **ARM_DETAIL,
    }

def test_the_stub_is_scored_by_cosine_and_an_arm_by_its_own_distance(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)
    arm = AssignedStubEncoder()
    arm.distance = 'euclidean'

    stub_result, stub_record = panel.score_logged(AssignedStubEncoder(), 'validation', 'stub')
    arm_result, arm_record = panel.score_logged(arm, 'validation', 'arm')

    assert stub_record['detail']['distance'] == 'cosine'
    assert stub_result.headline['distance'] == 'cosine'
    assert arm_record['detail']['distance'] == 'euclidean'
    assert arm_result.headline['distance'] == 'euclidean'

def test_the_scores_are_read_under_the_distance_the_read_logs(tmp_path):
    panel = _panel(SelectionLog(tmp_path / 'log.jsonl'))
    by_cosine = ScaledStubEncoder()
    by_euclidean = ScaledStubEncoder()
    by_euclidean.distance = 'euclidean'

    # p04 reports 111110 and 311111 and its query sits on 111110's axis: cosine finds 111110 at
    # distance 0, while under the Euclidean distance 311111 is the nearest reported code, at
    # sqrt(2), tied with the six codes that are not reported
    assert _rank(panel.score(by_cosine, 'validation', 'cosine'), 'p04') == 1
    assert _rank(panel.score(by_euclidean, 'validation', 'euclidean'), 'p04') == 7

def test_the_stub_scores_the_validation_part_by_hand(tmp_path):
    result = _panel(SelectionLog(tmp_path / 'log.jsonl')
                    ).score(AssignedStubEncoder(), 'validation', 'check')

    # p03's query decodes to 111110, and its reported 211120 is behind 111110 and the six other
    # codes at the stub's cosine distance 1: rank 8. Every other headline pair ranks 1
    assert result.per_pair.get_column('pair_id').to_list() == [
        'p01', 'p02', 'p03', 'p04', 'p05', 'p07'
    ]
    assert result.per_pair.get_column('rank').to_list() == [1, 1, 8, 1, 1, 1]
    assert result.headline['n_pairs'] == 5
    assert result.headline['mrr'] == pytest.approx((4 + 1 / 8) / 5)

def test_a_role_outside_reference_and_candidate_is_refused_before_logging(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)

    for role in ('baseline', None, 'Reference'):
        with pytest.raises(ValueError, match='role'):
            panel.score(AssignedStubEncoder(), 'validation', 'check', detail={'role': role})

    assert not log.path.exists()

@pytest.mark.parametrize(
    'detail',
    [{
        'role': 'reference'
    }, {
        'role': 'candidate'
    }, {
        'arm': 'lexical-stub'
    }, None],
    ids=['reference', 'candidate', 'no-role', 'no-detail'],
)
def test_reference_candidate_and_a_missing_role_are_read(tmp_path, detail):
    log = SelectionLog(tmp_path / 'log.jsonl')

    _panel(log).score(AssignedStubEncoder(), 'validation', 'check', detail=detail)

    (record, ) = log.records()
    assert {key: record['detail'][key] for key in (detail or {})} == (detail or {})

def test_the_read_roles_are_reference_and_candidate():
    assert READ_ROLES == ('reference', 'candidate')

def test_a_detail_that_would_replace_a_logged_key_is_refused_before_logging(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')

    with pytest.raises(ValueError, match='cannot replace'):
        _panel(log).score(
            AssignedStubEncoder(), 'validation', 'check', detail={'data_fingerprint': 'x'}
        )

    assert not log.path.exists()

def test_a_part_that_is_not_validation_or_sealed_is_refused_before_logging(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')

    with pytest.raises(ValueError, match='training'):
        _panel(log).score(AssignedStubEncoder(), 'training', 'check')

    assert not log.path.exists()

# -------------------------------------------------------------------------------------------------
# The sealed part
# -------------------------------------------------------------------------------------------------

def test_the_sealed_part_is_refused_before_it_is_opened(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)
    stub = AssignedStubEncoder()

    with pytest.raises(SealedSplitError):
        panel.score(stub, 'sealed', 'peek')
    assert not log.path.exists()

def test_an_opened_sealed_part_is_read_and_logged(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)

    panel.open_sealed('decision')
    result = panel.score(AssignedStubEncoder(), ItaPart.SEALED, 'decision', detail=ARM_DETAIL)

    opening, read = log.records()
    assert (opening['event'], opening['panel'], opening['split']) == ('open', 'ita_text', 'sealed')
    assert opening['fingerprint'] == split_fingerprint(ITA_SETTINGS)
    assert opening['n_queries'] == 4
    assert opening['detail'] == {}
    assert (read['event'], read['split'], read['n_queries']) == ('read', 'sealed', 4)
    assert read['detail']['data_fingerprint'] == panel.data_fingerprint('sealed')
    assert result.per_pair.get_column('pair_id').to_list() == ['p08', 'p09', 'p10', 'p11']

def test_an_opening_by_another_panel_object_does_not_unseal_this_one(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    _panel(log).open_sealed('decision')
    other = _panel(log)

    with pytest.raises(SealedSplitError):
        other.score(AssignedStubEncoder(), 'sealed', 'peek')
    assert len(log.records()) == 1

def test_a_second_opening_needs_a_reason_also_after_the_tables_are_rebuilt(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel_a = _panel(log)
    panel_a.open_sealed('decision')
    rebuilt = ita_tables(
        extra_pairs=[
            {
                'pair_id': 'p12',
                'description_key': 'pet food maker',
                'representative_string': 'Pet food maker',
                'flag_headline': True,
            }
        ],
        extra_codes=[{
            'pair_id': 'p12',
            **_reported('311111')
        }],
    )
    panel_b = _panel(log, *rebuilt)

    assert panel_b.fingerprint == panel_a.fingerprint
    assert panel_b.data_fingerprint('validation') != panel_a.data_fingerprint('validation')
    with pytest.raises(SplitAlreadyOpenedError):
        panel_b.open_sealed('again')
    assert len(log.records()) == 1
    panel_b.open_sealed('again', reopen_reason='audit')

    reopen = log.records()[-1]
    assert reopen['event'] == SelectionEvent.REOPEN.value
    assert reopen['panel'] == 'ita_text'
    assert reopen['split'] == 'sealed'
    assert reopen['fingerprint'] == panel_a.fingerprint
    assert reopen['detail'] == {'reason': 'audit'}

def test_a_blank_reopen_reason_is_no_reason(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    _panel(log).open_sealed('decision')

    with pytest.raises(SplitAlreadyOpenedError):
        _panel(log).open_sealed('again', reopen_reason='   ')
    assert len(log.records()) == 1

def test_an_opening_of_another_split_or_panel_does_not_count(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    fingerprint = split_fingerprint(ITA_SETTINGS)
    log.append(
        SelectionEvent.OPEN,
        panel='outcome',
        split='test',
        purpose='other panel',
        fingerprint=fingerprint,
        n_queries=1,
    )
    log.append(
        SelectionEvent.OPEN,
        panel=ITA_PANEL,
        split='sealed',
        purpose='other salt',
        fingerprint='0' * 64,
        n_queries=1,
    )

    _panel(log).open_sealed('decision')

    assert log.records()[-1]['event'] == SelectionEvent.OPEN.value

# -------------------------------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'code, message',
    [
        (_node('999999'), 'not a candidate'),
        ('naics:2022:industry:517112', 'national_industry'),
    ],
    ids=['code-outside-candidates', 'node-of-another-level'],
)
def test_refusals_happen_before_anything_is_logged(tmp_path, code, message):
    log = SelectionLog(tmp_path / 'log.jsonl')
    pairs, codes = ita_tables(
        replace_codes={
            'p03': [
                {
                    'naics_code': code.rsplit(':', 1)[-1],
                    'naics_2022_node_id': code,
                    'flag_headline_basis': True,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match=message):
        ItaPanel(pairs, codes, ITA_CANDIDATES, ITA_SETTINGS, log)
    assert not log.path.exists()

@pytest.mark.parametrize(
    'candidates',
    [(*ITA_CANDIDATES, '111110'), (*ITA_CANDIDATES, '11111')],
    ids=['repeated', 'five-digit'],
)
def test_candidates_must_be_distinct_six_digit_codes(tmp_path, candidates):
    log = SelectionLog(tmp_path / 'log.jsonl')

    with pytest.raises(ValueError, match='distinct six-digit'):
        _panel(log, candidates=candidates)
    assert not log.path.exists()

def test_the_candidates_are_kept_sorted(tmp_path):
    panel = _panel(SelectionLog(tmp_path / 'log.jsonl'), candidates=ITA_CANDIDATES[::-1])

    assert panel.candidates == ITA_CANDIDATES
    assert panel.settings == ITA_SETTINGS
    assert panel.fingerprint == split_fingerprint(ITA_SETTINGS)

# -------------------------------------------------------------------------------------------------
# The data fingerprint
# -------------------------------------------------------------------------------------------------

_PART_PAIRS = {
    'validation': [
        ['p01', 'We make dog food', ['311111'], False],
        ['p02', 'we make dog food', ['311111'], False],
        ['p03', 'Oil and gas wells', ['211120'], False],
        ['p04', 'Cat food plant', ['111110', '311111'], False],
        ['p05', 'Software publisher', ['513210'], True],
        ['p07', 'General contractor', ['236220'], False],
    ],
    'sealed': [
        ['p08', 'Soybean farm', ['111110'], False],
        ['p09', 'Canola growers', ['111120'], False],
        ['p10', 'Sawmill', ['321113'], False],
        ['p11', 'Dental office', ['621210'], True],
    ],
}

@pytest.mark.parametrize('part', ['validation', 'sealed'])
def test_the_data_fingerprint_hashes_the_pairs_and_the_candidates(tmp_path, part):
    log = SelectionLog(tmp_path / 'log.jsonl')
    payload = {'pairs': _PART_PAIRS[part], 'candidates': list(ITA_CANDIDATES)}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))

    fingerprint = _panel(log).data_fingerprint(ItaPart(part))

    assert fingerprint == hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    assert not log.path.exists()

def test_the_data_fingerprint_names_the_candidates(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    pairs, codes = ita_tables()
    wider = _panel(log, pairs, codes, candidates=(*ITA_CANDIDATES, '999999'))

    assert wider.data_fingerprint('validation') != _panel(log).data_fingerprint('validation')

def test_row_order_and_extra_columns_leave_the_data_fingerprint_alone(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    pairs, codes = ita_tables()
    shuffled_pairs = pairs.sample(fraction=1.0, shuffle=True,
                                  seed=17).with_columns(pl.lit('x').alias('extra'))
    shuffled_codes = codes.sample(fraction=1.0, shuffle=True,
                                  seed=17).with_columns(pl.lit(1).alias('extra'))
    assert shuffled_pairs.get_column('pair_id').to_list() != pairs.get_column('pair_id').to_list()
    assert shuffled_codes.get_column('pair_id').to_list() != codes.get_column('pair_id').to_list()

    panel = _panel(log, pairs, codes)
    shuffled = _panel(log, shuffled_pairs, shuffled_codes)

    assert shuffled.pairs.equals(panel.pairs)
    for part in ('validation', 'sealed'):
        assert shuffled.data_fingerprint(part) == panel.data_fingerprint(part)
    stub = AssignedStubEncoder()
    assert (
        shuffled.score(stub, 'validation',
                       'check').headline == panel.score(stub, 'validation', 'check').headline
    )

# -------------------------------------------------------------------------------------------------
# The shared selection log
# -------------------------------------------------------------------------------------------------

def test_ita_reads_stay_out_of_another_runs_log_records(tmp_path):
    log = SelectionLog(tmp_path / 'log.jsonl')
    panel = _panel(log)
    outcome = log.append(
        SelectionEvent.READ,
        panel='outcome',
        split='validation',
        purpose='stage 7 read',
        fingerprint='f' * 64,
        n_queries=3,
        detail={'run': 'stage7/x'},
    )
    _, ita = panel.score_logged(AssignedStubEncoder(), 'validation', 'check', detail=ARM_DETAIL)
    panel.open_sealed('decision')

    assert _log_records([log], 'stage7/x') == [outcome]
    assert _log_records([log], ARM_DETAIL['run']) == [ita]
    assert len(log.records()) == 3

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def test_from_files_reads_the_tables_and_the_bundle_codebook(generated_bundle, tmp_path):
    bundle = load_validated_bundle(generated_bundle)
    headline = ('p01', 'p02', 'p03', 'p05', 'p07', 'p08', 'p09', 'p10', 'p11')
    pairs, codes = ita_tables(
        replace_codes={
            'p04': [_reported('111111'), _reported('111112')],
            **{
                pair_id: [_reported('111111')]
                for pair_id in headline
            },
        }
    )
    pairs_path, codes_path = write_ita_tables(tmp_path, pairs, codes)
    settings_path = tmp_path / 'ita_panel.yaml'
    settings_path.write_text(json.dumps(ITA_SETTINGS.model_dump()), encoding='utf-8')

    panel = ItaPanel.from_files(
        pairs_path, codes_path, bundle, settings_path, tmp_path / 'log.jsonl'
    )

    assert panel.candidates == ('111111', '111112', '111113', '222222', '333333')
    assert panel.settings == ITA_SETTINGS
    assert panel.log.path == tmp_path / 'log.jsonl'
    assert panel.pairs.height == 10
    assert panel.pairs.filter(pl.col('pair_id') == 'p04').get_column('codes').to_list() == [
        ['111111', '111112']
    ]
    assert not panel.log.path.exists()
