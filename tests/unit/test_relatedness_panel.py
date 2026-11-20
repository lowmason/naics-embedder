'''
The relatedness panel: its parts, counts, logged reads, pins and the sealed opening (plan 21,
Task 6; spec relatedness-artifact, 4.2 to 4.5).

``RelatednessPanel`` mirrors ``ItaPanel`` under its own panel name, ``relatedness_co_ownership``.
Every read is logged before the arm is asked for a similarity, the sealed part needs a logged
opening by the same panel object, and a second opening is keyed on the split fingerprint, so a
rebuilt table cannot pass it as a first. The oracles are written here from ``BASE_PAIRS`` and
``BASE_MULTI``: a gain is J / m(candidate). Each test builds its tables from
``tests/fixtures/relatedness.py`` and writes under ``tmp_path``. No test reads a real table.
'''

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import pytest

from naics_embedder.decision.sweep import _log_records
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.relatedness import READ_ROLES, RelatednessPanel, co_ownership_inputs
from naics_embedder.panels.relatedness_arms import TreeArm, VectorArm
from naics_embedder.panels.relatedness_scoring import (
    COUNT_KEYS,
    RelatednessResult,
    tie_aware_auc,
    tie_aware_ndcg,
)
from naics_embedder.panels.relatedness_split import (
    RelatednessPanelSettings,
    RelatednessPart,
    split_fingerprint,
    split_part,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.relatedness import (
    BASE_MULTI,
    BASE_PAIRS,
    RELATEDNESS_CODES,
    RELATEDNESS_SETTINGS,
    co_ownership_tables,
    tree_distance,
    write_co_ownership_tables,
)

pytestmark = pytest.mark.unit

Tables = Tuple[pl.DataFrame, pl.DataFrame]

TABLE_SHA256 = {'ita_co_ownership': 'a' * 64, 'co_ownership_marginals': 'b' * 64}

ARM_DETAIL = {'arm': 'tree', 'seed': 3, 'role': 'candidate', 'run': 'relatedness/tree/3'}

# The 14 codes with a marginal row, and the two parts' anchors (the plan's golden parts, P1)
CANDIDATES = tuple(sorted(BASE_MULTI))
VALIDATION_ANCHORS = ('111110', '111120', '111191', '311211', '311221', '312111', '445110')
SEALED_ANCHORS = ('111150', '311111', '311119', '311212', '312112', '621210')

# The base pairs whose two industries are both in one part
VALIDATION_ONLY_PAIRS = tuple(
    pair for pair in BASE_PAIRS if {pair[0], pair[1]} <= set(VALIDATION_ANCHORS)
)
SEALED_ONLY_PAIRS = tuple(pair for pair in BASE_PAIRS if {pair[0], pair[1]} <= set(SEALED_ANCHORS))

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

class UnwritableLog(SelectionLog):
    '''
    A log for a panel that is never read: a record it is asked to write fails the test. Its
    directory does not exist, so no file can be made there either.
    '''

    def __init__(self):
        super().__init__(Path(__file__).with_name('no-such-directory') / 'never-written.jsonl')

    def append(self, *args, **kwargs):
        raise AssertionError('this panel was not meant to log anything')

def panel(
    log: Optional[SelectionLog] = None,
    tables: Optional[Tables] = None,
    *,
    table_sha256: Mapping[str, str] = TABLE_SHA256,
    settings: RelatednessPanelSettings = RELATEDNESS_SETTINGS,
) -> RelatednessPanel:
    '''The panel over the base tables, or over ``tables``.'''

    targets, marginals = tables or co_ownership_tables()
    return RelatednessPanel(
        targets,
        marginals,
        RELATEDNESS_CODES,
        settings,
        log or UnwritableLog(),
        table_sha256=table_sha256,
    )

def logged(tmp_path: Path) -> SelectionLog:
    return SelectionLog(tmp_path / 'log.jsonl')

def rebuilt_tables() -> Tables:
    '''The base tables rebuilt with one more industry, 236220, paired with 312111.'''

    return co_ownership_tables((*BASE_PAIRS, ('236220', '312111', 1)), BASE_MULTI | {'236220': 1})

def tree_arm() -> TreeArm:
    return TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES))

def hand_gain(anchor: str, candidate: str, pairs=BASE_PAIRS, multi=BASE_MULTI) -> float:
    '''J(anchor, candidate) / m(candidate), from the fixture's pairs, and 0.0 with no pair.'''

    for code_a, code_b, j in pairs:
        if {code_a, code_b} == {anchor, candidate}:
            return j / multi[candidate]
    return 0.0

def hand_partners(anchor: str) -> List[str]:
    return [code for code in CANDIDATES if hand_gain(anchor, code) > 0]

def hand_fingerprint(anchors: Sequence[str], table_sha256: Mapping[str, str] = TABLE_SHA256) -> str:
    '''P6's payload, built from the fixture's pairs, as canonical JSON.'''

    payload = {
        'candidates': list(CANDIDATES),
        'anchors': [
            [anchor, [[partner, hand_gain(anchor, partner)] for partner in hand_partners(anchor)]]
            for anchor in anchors
        ],
        'tables': dict(table_sha256),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def without_time(record: Mapping) -> Dict:
    return {key: value for key, value in record.items() if key != 'time'}

class RaisingArm:
    '''An arm that fails as soon as it is asked for a similarity.'''

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        raise RuntimeError('similarity failed')

class RecordingArm:
    '''The tree arm, keeping what each call asked for.'''

    def __init__(self):
        self.calls: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        self.calls.append((tuple(anchors), tuple(candidates)))
        return tree_arm().similarity(anchors, candidates)

class NotFiniteArm:
    '''The tree arm with some cells replaced: ``cells`` maps (anchor, candidate) to a value.'''

    def __init__(self, cells: Mapping[Tuple[str, str], float]):
        self.cells = dict(cells)

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        matrix = tree_arm().similarity(anchors, candidates)
        for (anchor, candidate), value in self.cells.items():
            matrix[list(anchors).index(anchor), list(candidates).index(candidate)] = value
        return matrix

class NarrowArm:
    '''The tree arm, one candidate column short.'''

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        return tree_arm().similarity(anchors, candidates)[:, :-1]

def settings_file(tmp_path: Path, settings: RelatednessPanelSettings) -> Path:
    path = tmp_path / 'relatedness_panel.yaml'
    path.write_text(json.dumps(settings.model_dump()), encoding='utf-8')
    return path

# -------------------------------------------------------------------------------------------------
# The plan's tests
# -------------------------------------------------------------------------------------------------

def test_each_industry_falls_in_one_part_and_a_rebuild_keeps_it_there():
    base = panel()
    assert base.anchors('validation') == (
        '111110', '111120', '111191', '311211', '311221', '312111', '445110'
    )
    assert base.anchors('sealed') == ('111150', '311111', '311119', '311212', '312112', '621210')
    # rebuilt with BASE_MULTI | {'236220': 1} and the pair ('236220', '312111', 1): every base
    # code keeps its part, 236220 is in validation, the fingerprint is equal, and the validation
    # data fingerprint differs
    rebuilt = panel(tables=rebuilt_tables())
    for code in RELATEDNESS_CODES:
        assert rebuilt.part_of(code) is base.part_of(code)
    assert rebuilt.part_of('236220') is RelatednessPart.VALIDATION
    assert rebuilt.anchors('validation') == tuple(sorted((*base.anchors('validation'), '236220')))
    assert rebuilt.anchors('sealed') == base.anchors('sealed')
    assert rebuilt.fingerprint == base.fingerprint
    assert rebuilt.data_fingerprint('validation') != base.data_fingerprint('validation')

def test_the_counts_of_each_part(tmp_path):
    # validation: n_industries 8, n_anchors 7, n_no_partner 1, n_candidates 14, n_relevant 16,
    # n_crossing 6; sealed (after open_sealed): 6, 6, 0, 14, 8, 6
    base = panel(logged(tmp_path))

    validation = base.score(tree_arm(), 'validation', 'check')
    base.open_sealed('decision')
    sealed = base.score(tree_arm(), 'sealed', 'decision')

    assert validation.counts == dict(zip(COUNT_KEYS, (8, 7, 1, 14, 16, 6)))
    assert sealed.counts == dict(zip(COUNT_KEYS, (6, 6, 0, 14, 8, 6)))

def test_the_tree_arms_scores_are_the_scorers(tmp_path):
    # TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES)) on validation: 311211's ndcg_10
    # equals tie_aware_ndcg of its tree row and gains at k 10, and headline['ndcg_10'] is the fsum
    # mean of the seven anchors
    arm = TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES))
    result = panel(logged(tmp_path)).score(arm, 'validation', 'check')

    others = [code for code in CANDIDATES if code != '311211']
    tree_row = -tree_distance(('311211', *others))[0, 1:]
    gains = np.array([hand_gain('311211', code) for code in others])
    row = result.per_anchor.filter(pl.col('anchor') == '311211').row(0, named=True)
    assert row['ndcg_10'] == tie_aware_ndcg(tree_row, gains, 10)
    ndcg_10 = result.per_anchor.get_column('ndcg_10').to_list()
    assert len(ndcg_10) == 7
    assert result.headline['ndcg_10'] == math.fsum(ndcg_10) / 7

def test_a_read_is_logged_before_the_arm_is_asked(tmp_path):
    # an arm whose similarity raises RuntimeError: the log holds one read with panel
    # 'relatedness_co_ownership', split 'validation', fingerprint split_fingerprint(...),
    # n_queries 7, and detail arm_type, data_fingerprint and the caller's arm, role and run
    log = logged(tmp_path)
    base = panel(log)

    with pytest.raises(RuntimeError, match='similarity failed'):
        base.score(RaisingArm(), 'validation', 'check', detail=ARM_DETAIL)

    (record, ) = log.records()
    assert without_time(record) == {
        'event': 'read',
        'panel': 'relatedness_co_ownership',
        'split': 'validation',
        'purpose': 'check',
        'fingerprint': split_fingerprint(RELATEDNESS_SETTINGS),
        'n_queries': 7,
        'detail': {
            'arm_type': 'RaisingArm',
            'data_fingerprint': base.data_fingerprint('validation'),
            **ARM_DETAIL,
        },
    }

def test_a_non_finite_similarity_is_refused_naming_the_arm_and_anchor(tmp_path):
    # a stub arm with NaN on 311211's row: ValueError naming the stub's type and '311211'; the
    # read record exists
    log = logged(tmp_path)
    arm = NotFiniteArm({('311211', '111120'): float('nan')})

    with pytest.raises(ValueError, match='NotFiniteArm') as refusal:
        panel(log).score(arm, 'validation', 'check', detail=ARM_DETAIL)

    assert '311211' in str(refusal.value)
    (record, ) = log.records()
    assert (record['event'], record['split']) == ('read', 'validation')

def test_a_role_outside_reference_and_candidate_is_refused_before_logging(tmp_path):
    log = logged(tmp_path)
    base = panel(log)

    for role in ('baseline', None, 'Reference', ''):
        with pytest.raises(ValueError, match='role'):
            base.score(tree_arm(), 'validation', 'check', detail={'role': role})

    assert not log.path.exists()

def test_the_sealed_part_is_refused_before_it_is_opened(tmp_path):
    # SealedSplitError, and the log file does not exist
    log = logged(tmp_path)
    base = panel(log)

    for part in ('sealed', RelatednessPart.SEALED):
        with pytest.raises(SealedSplitError):
            base.score(tree_arm(), part, 'peek')

    assert not log.path.exists()

def test_a_second_opening_needs_a_reason_also_after_the_tables_are_rebuilt(tmp_path):
    # panel A opens; panel B over the rebuilt tables (as above) has the same fingerprint;
    # B.open_sealed('again') → SplitAlreadyOpenedError; with reopen_reason='audit' → event 'reopen',
    # detail reason 'audit'
    log = logged(tmp_path)
    panel_a = panel(log)
    panel_a.open_sealed('decision')
    panel_b = panel(log, rebuilt_tables())

    assert panel_b.fingerprint == panel_a.fingerprint
    assert panel_b.data_fingerprint('validation') != panel_a.data_fingerprint('validation')
    with pytest.raises(SplitAlreadyOpenedError, match="'decision'.*reopen_reason"):
        panel_b.open_sealed('again')
    assert len(log.records()) == 1
    panel_b.open_sealed('again', reopen_reason='audit')

    reopen = log.records()[-1]
    assert without_time(reopen) == {
        'event': 'reopen',
        'panel': 'relatedness_co_ownership',
        'split': 'sealed',
        'purpose': 'again',
        'fingerprint': panel_a.fingerprint,
        'n_queries': 6,
        'detail': {
            'reason': 'audit'
        },
    }
    assert len(log.records()) == 2

def test_a_part_with_no_anchor_is_refused_before_logging(tmp_path):
    # tables whose pairs are all inside the validation part: open_sealed, then a sealed read raises
    # ValueError 'no anchor'; the log holds the opening only
    log = logged(tmp_path)
    base = panel(log, co_ownership_tables(VALIDATION_ONLY_PAIRS))
    assert base.anchors('sealed') == ()

    base.open_sealed('decision')
    with pytest.raises(ValueError, match='no anchor'):
        base.score(tree_arm(), 'sealed', 'decision')

    (opening, ) = log.records()
    assert (opening['event'], opening['split'], opening['n_queries']) == ('open', 'sealed', 0)

def test_constructor_refusals_log_nothing(tmp_path):
    # the j-above-m tables of Task 2: RelatednessPanel(...) raises and the log file does not exist
    log = logged(tmp_path)
    pairs = [(a, b, 4 if (a, b) == ('311211', '311212') else j) for a, b, j in BASE_PAIRS]

    with pytest.raises(ValueError, match='exceeds m of 311212'):
        panel(log, co_ownership_tables(pairs))

    assert not log.path.exists()

def test_row_order_and_extra_columns_leave_the_data_fingerprint_alone(tmp_path):
    # both tables shuffled with an 'extra' column, same table_sha256: equal data fingerprints for
    # both parts and an equal validation headline
    log = logged(tmp_path)
    targets, marginals = co_ownership_tables()
    shuffled_targets = targets.sample(fraction=1.0, shuffle=True,
                                      seed=17).with_columns(pl.lit('x').alias('extra'))
    shuffled_marginals = marginals.sample(fraction=1.0, shuffle=True,
                                          seed=17).with_columns(pl.lit(1).alias('extra'))
    assert shuffled_targets.get_column('unit_a').to_list() != targets.get_column('unit_a').to_list()
    assert (
        shuffled_marginals.get_column('node_id').to_list()
        != marginals.get_column('node_id').to_list()
    )

    base = panel(log, (targets, marginals))
    shuffled = panel(log, (shuffled_targets, shuffled_marginals))

    for part in ('validation', 'sealed'):
        assert shuffled.data_fingerprint(part) == base.data_fingerprint(part)
    assert (
        shuffled.score(tree_arm(), 'validation',
                       'check').headline == base.score(tree_arm(), 'validation', 'check').headline
    )

def test_the_data_fingerprint_names_the_table_hashes():
    # another table_sha256 value: another data fingerprint, the same split fingerprint
    base = panel()
    other = panel(table_sha256={'ita_co_ownership': 'c' * 64, 'co_ownership_marginals': 'b' * 64})

    for part in ('validation', 'sealed'):
        assert other.data_fingerprint(part) != base.data_fingerprint(part)
    assert other.fingerprint == base.fingerprint

def test_from_files_checks_the_pins_before_reading(tmp_path, validated_bundle):
    # tables over the bundle's six-digit codes (111111, 111112, 111113, 222222, 333333):
    # multi {'111111': 2, '111112': 2, '222222': 1, '333333': 0}, pairs
    # [('111111', '111112', 2), ('111111', '222222', 1)], written with write_co_ownership_tables.
    # The right pins (sha256_file of each file) set panel.table_sha256 to them. A wrong pin raises
    # ValueError naming the table, the pin and the hash found, and logs nothing
    multi = {'111111': 2, '111112': 2, '222222': 1, '333333': 0}
    pairs = [('111111', '111112', 2), ('111111', '222222', 1)]
    targets_path, marginals_path = write_co_ownership_tables(
        tmp_path, *co_ownership_tables(pairs, multi)
    )
    settings_path = settings_file(tmp_path, RELATEDNESS_SETTINGS)
    log_path = tmp_path / 'log.jsonl'
    pins = {
        'ita_co_ownership': sha256_file(targets_path),
        'co_ownership_marginals': sha256_file(marginals_path),
    }

    read = RelatednessPanel.from_files(
        targets_path,
        marginals_path,
        validated_bundle,
        settings_path,
        log_path,
        expected_sha256=pins,
    )
    assert read.table_sha256 == pins
    assert read.candidates == ('111111', '111112', '222222', '333333')

    wrong = {**pins, 'ita_co_ownership': 'd' * 64}
    with pytest.raises(ValueError) as refusal:
        RelatednessPanel.from_files(
            targets_path,
            marginals_path,
            validated_bundle,
            settings_path,
            log_path,
            expected_sha256=wrong,
        )
    for named in ('ita_co_ownership', 'd' * 64, pins['ita_co_ownership']):
        assert named in str(refusal.value)
    assert not log_path.exists()

def test_relatedness_reads_stay_out_of_another_runs_log_records(tmp_path):
    # a relatedness validation read and a sealed opening in a log with an outcome read under run
    # 'stage7/x': decision.sweep._log_records(..., 'stage7/x') returns that run's records only
    log = logged(tmp_path)
    base = panel(log)
    outcome = log.append(
        SelectionEvent.READ,
        panel='outcome',
        split='validation',
        purpose='stage 7 read',
        fingerprint='f' * 64,
        n_queries=3,
        detail={'run': 'stage7/x'},
    )
    _, relatedness = base.score_logged(tree_arm(), 'validation', 'check', detail=ARM_DETAIL)
    base.open_sealed('decision')

    assert _log_records([log], 'stage7/x') == [outcome]
    assert _log_records([log], ARM_DETAIL['run']) == [relatedness]
    assert len(log.records()) == 3

# -------------------------------------------------------------------------------------------------
# The panel's names and attributes
# -------------------------------------------------------------------------------------------------

def test_the_panel_name_and_the_read_roles():
    assert RelatednessPanel.name == 'relatedness_co_ownership'
    assert READ_ROLES == ('reference', 'candidate')

def test_the_panel_keeps_its_inputs():
    targets, marginals = co_ownership_tables()
    base = panel(tables=(targets, marginals))

    assert base.co_ownership == co_ownership_inputs(targets, marginals, RELATEDNESS_CODES)
    assert base.candidates == CANDIDATES
    assert base.settings == RELATEDNESS_SETTINGS
    assert base.fingerprint == split_fingerprint(RELATEDNESS_SETTINGS)
    assert base.table_sha256 == TABLE_SHA256

def test_each_code_is_in_the_part_its_salted_hash_gives():
    base = panel()

    for code in RELATEDNESS_CODES:
        assert isinstance(base.part_of(code), RelatednessPart)
        assert base.part_of(code) is split_part(code, RELATEDNESS_SETTINGS.salt)
    in_validation = {
        code
        for code in CANDIDATES if base.part_of(code) is RelatednessPart.VALIDATION
    }
    assert in_validation == {*VALIDATION_ANCHORS, '445131'}

def test_the_anchors_accept_a_part_or_its_name():
    base = panel()

    assert base.anchors(RelatednessPart.VALIDATION) == VALIDATION_ANCHORS
    assert base.anchors(RelatednessPart.SEALED) == SEALED_ANCHORS
    with pytest.raises(ValueError, match='training'):
        base.anchors('training')

# -------------------------------------------------------------------------------------------------
# Reads
# -------------------------------------------------------------------------------------------------

def test_every_anchor_is_scored_on_its_tree_row_and_its_gains(tmp_path):
    result = panel(logged(tmp_path)).score(tree_arm(), 'validation', 'check')

    expected_10, expected_20 = [], []
    for anchor in VALIDATION_ANCHORS:
        others = [code for code in CANDIDATES if code != anchor]
        tree_row = -tree_distance((anchor, *others))[0, 1:]
        gains = np.array([hand_gain(anchor, code) for code in others])
        expected_10.append(tie_aware_ndcg(tree_row, gains, 10))
        expected_20.append(tie_aware_ndcg(tree_row, gains, 20))
        row = result.per_anchor.filter(pl.col('anchor') == anchor).row(0, named=True)
        assert row['part'] == 'validation'
        assert row['n_relevant'] == len(hand_partners(anchor))
        assert row['ndcg_10'] == expected_10[-1]
        assert row['ndcg_20'] == expected_20[-1]
        assert row['auc'] == tie_aware_auc(tree_row, gains > 0)
    assert result.per_anchor.get_column('anchor').to_list() == list(VALIDATION_ANCHORS)
    assert result.headline == {'n_anchors': 7, 'ndcg_10': math.fsum(expected_10) / 7}
    assert result.reports['ndcg_20'] == math.fsum(expected_20) / 7
    assert (result.panel, result.part) == ('relatedness_co_ownership', 'validation')

def test_the_arm_is_asked_for_the_parts_anchors_against_every_candidate(tmp_path):
    base = panel(logged(tmp_path))
    arm = RecordingArm()

    base.score(arm, 'validation', 'check')
    base.open_sealed('decision')
    base.score(arm, RelatednessPart.SEALED, 'decision')

    assert arm.calls == [(VALIDATION_ANCHORS, CANDIDATES), (SEALED_ANCHORS, CANDIDATES)]

def test_an_opened_sealed_part_is_read_and_logged(tmp_path):
    log = logged(tmp_path)
    base = panel(log)

    base.open_sealed('decision')
    result, record = base.score_logged(
        tree_arm(), RelatednessPart.SEALED, 'decision', detail=ARM_DETAIL
    )

    opening, read = log.records()
    assert without_time(opening) == {
        'event': 'open',
        'panel': 'relatedness_co_ownership',
        'split': 'sealed',
        'purpose': 'decision',
        'fingerprint': split_fingerprint(RELATEDNESS_SETTINGS),
        'n_queries': 6,
        'detail': {},
    }
    assert read == record
    assert (read['event'], read['split'], read['n_queries']) == ('read', 'sealed', 6)
    assert read['detail']['data_fingerprint'] == base.data_fingerprint('sealed')
    assert result.per_anchor.get_column('anchor').to_list() == list(SEALED_ANCHORS)
    assert set(result.per_anchor.get_column('part').to_list()) == {'sealed'}
    assert (result.panel, result.part) == ('relatedness_co_ownership', 'sealed')

def test_a_result_carries_the_data_fingerprint_its_read_logged(tmp_path):
    base = panel(logged(tmp_path))

    validation, record = base.score_logged(tree_arm(), 'validation', 'check')
    assert isinstance(validation, RelatednessResult)
    assert validation.data_fingerprint == base.data_fingerprint('validation')
    assert validation.data_fingerprint == record['detail']['data_fingerprint']
    base.open_sealed('decision')
    sealed = base.score(tree_arm(), 'sealed', 'decision')
    assert sealed.data_fingerprint == base.data_fingerprint('sealed')
    assert sealed.data_fingerprint != validation.data_fingerprint

def test_score_logs_its_read_and_returns_what_score_logged_returns(tmp_path):
    log = logged(tmp_path)
    base = panel(log)

    result = base.score(tree_arm(), 'validation', 'first', detail=ARM_DETAIL)
    again, record = base.score_logged(tree_arm(), 'validation', 'second', detail=ARM_DETAIL)

    first, second = log.records()
    assert (first['event'], first['purpose'], first['detail']) == (
        'read', 'first', record['detail']
    )
    assert second == record
    assert result.headline == again.headline
    assert result.counts == again.counts
    assert result.per_anchor.equals(again.per_anchor)

@pytest.mark.parametrize(
    'detail',
    [{
        'role': 'reference'
    }, {
        'role': 'candidate'
    }, {
        'arm': 'tree'
    }, {}, None],
    ids=['reference', 'candidate', 'no-role', 'empty', 'no-detail'],
)
def test_reference_candidate_and_a_missing_role_are_read(tmp_path, detail):
    log = logged(tmp_path)
    base = panel(log)

    base.score(tree_arm(), 'validation', 'check', detail=detail)

    (record, ) = log.records()
    assert record['detail'] == {
        'arm_type': 'TreeArm',
        'data_fingerprint': base.data_fingerprint('validation'),
        **(detail or {}),
    }

@pytest.mark.parametrize('key', ['data_fingerprint', 'arm_type'])
def test_a_detail_that_would_replace_a_logged_key_is_refused_before_logging(tmp_path, key):
    log = logged(tmp_path)

    with pytest.raises(ValueError, match='cannot replace'):
        panel(log).score(tree_arm(), 'validation', 'check', detail={key: 'x'})

    assert not log.path.exists()

def test_a_part_that_is_not_validation_or_sealed_is_refused_before_logging(tmp_path):
    log = logged(tmp_path)

    with pytest.raises(ValueError, match='training'):
        panel(log).score(tree_arm(), 'training', 'check')

    assert not log.path.exists()

def test_an_arms_own_error_leaves_the_read_logged_and_passes_unchanged(tmp_path):
    log = logged(tmp_path)
    vectors = np.random.default_rng(7).normal(size=(len(RELATEDNESS_CODES) - 1, 4))
    arm = VectorArm([code for code in RELATEDNESS_CODES if code != '445131'], vectors)

    with pytest.raises(ValueError, match=r'^requested codes not in the arm \(1\): 445131$'):
        panel(log).score(arm, 'validation', 'check', detail=ARM_DETAIL)

    (record, ) = log.records()
    assert (record['event'], record['detail']['arm_type']) == ('read', 'VectorArm')

@pytest.mark.parametrize(
    'cells',
    [
        {
            ('311211', '111120'): float('nan')
        },
        {
            ('311211', '311211'): float('inf')
        },
        {
            ('311211', '445131'): -float('inf')
        },
    ],
    ids=['nan', 'inf-in-its-own-column', 'minus-inf'],
)
def test_every_non_finite_value_is_refused_by_the_panel_naming_the_arm(tmp_path, cells):
    log = logged(tmp_path)

    with pytest.raises(ValueError) as refusal:
        panel(log).score(NotFiniteArm(cells), 'validation', 'check')

    assert str(
        refusal.value
    ) == 'NotFiniteArm gave a similarity that is not finite for anchor 311211'
    (record, ) = log.records()
    assert record['event'] == 'read'

def test_the_first_anchor_with_a_non_finite_similarity_is_named(tmp_path):
    cells = {('445110', '111110'): float('nan'), ('311211', '111110'): float('nan')}

    with pytest.raises(ValueError) as refusal:
        panel(logged(tmp_path)).score(NotFiniteArm(cells), 'validation', 'check')

    assert 'anchor 311211' in str(refusal.value)
    assert '445110' not in str(refusal.value)

def test_a_similarity_of_another_shape_is_refused_naming_the_arm(tmp_path):
    log = logged(tmp_path)

    with pytest.raises(ValueError) as refusal:
        panel(log).score(NarrowArm(), 'validation', 'check')

    message = str(refusal.value)
    assert 'NarrowArm' in message
    assert '(7, 13)' in message
    assert '(7, 14)' in message
    (record, ) = log.records()
    assert record['event'] == 'read'

def test_a_part_with_one_anchor_is_read(tmp_path):
    log = logged(tmp_path)
    base = panel(log, co_ownership_tables([('311211', '311212', 3)]))

    result = base.score(tree_arm(), 'validation', 'check')

    assert result.per_anchor.get_column('anchor').to_list() == ['311211']
    assert result.counts == dict(zip(COUNT_KEYS, (8, 1, 7, 14, 1, 1)))
    (record, ) = log.records()
    assert record['n_queries'] == 1

# -------------------------------------------------------------------------------------------------
# Counts on other tables
# -------------------------------------------------------------------------------------------------

def test_the_counts_after_a_rebuild(tmp_path):
    # 236220-312111 is a pair inside validation: 7 pairs do not cross the parts, 6 do
    rebuilt = panel(logged(tmp_path), rebuilt_tables())

    validation = rebuilt.score(tree_arm(), 'validation', 'check')
    rebuilt.open_sealed('decision')
    sealed = rebuilt.score(tree_arm(), 'sealed', 'decision')

    assert validation.counts == dict(zip(COUNT_KEYS, (9, 8, 1, 15, 18, 6)))
    assert sealed.counts == dict(zip(COUNT_KEYS, (6, 6, 0, 15, 8, 6)))

def test_the_counts_when_no_pair_crosses_the_parts(tmp_path):
    base = panel(logged(tmp_path), co_ownership_tables(VALIDATION_ONLY_PAIRS))

    result = base.score(tree_arm(), 'validation', 'check')

    assert result.counts == dict(zip(COUNT_KEYS, (8, 6, 2, 14, 10, 0)))

# -------------------------------------------------------------------------------------------------
# The order of the refusals
# -------------------------------------------------------------------------------------------------

def test_a_validation_part_with_no_anchor_is_refused_and_logs_nothing(tmp_path):
    log = logged(tmp_path)
    base = panel(log, co_ownership_tables(SEALED_ONLY_PAIRS))

    with pytest.raises(ValueError, match='no anchor'):
        base.score(tree_arm(), 'validation', 'check')

    assert not log.path.exists()

def test_a_bad_role_is_refused_before_the_seal_is_checked(tmp_path):
    log = logged(tmp_path)

    with pytest.raises(ValueError, match='role'):
        panel(log).score(tree_arm(), 'sealed', 'peek', detail={'role': 'baseline'})

    assert not log.path.exists()

def test_an_unopened_sealed_part_with_no_anchor_is_refused_as_sealed(tmp_path):
    log = logged(tmp_path)

    with pytest.raises(SealedSplitError):
        panel(log, co_ownership_tables(VALIDATION_ONLY_PAIRS)).score(tree_arm(), 'sealed', 'peek')

    assert not log.path.exists()

# -------------------------------------------------------------------------------------------------
# The sealed part
# -------------------------------------------------------------------------------------------------

def test_an_opening_by_another_panel_object_does_not_unseal_this_one(tmp_path):
    log = logged(tmp_path)
    panel(log).open_sealed('decision')
    other = panel(log)

    with pytest.raises(SealedSplitError):
        other.score(tree_arm(), 'sealed', 'peek')
    assert len(log.records()) == 1

@pytest.mark.parametrize('reason', ['', '   '], ids=['empty', 'blank'])
def test_a_blank_reopen_reason_is_no_reason(tmp_path, reason):
    log = logged(tmp_path)
    panel(log).open_sealed('decision')

    with pytest.raises(SplitAlreadyOpenedError):
        panel(log).open_sealed('again', reopen_reason=reason)
    assert len(log.records()) == 1

def test_a_reopen_reason_is_kept_without_its_surrounding_spaces(tmp_path):
    log = logged(tmp_path)
    panel(log).open_sealed('decision')

    panel(log).open_sealed('again', reopen_reason='  audit  ')

    assert log.records()[-1]['detail'] == {'reason': 'audit'}

def test_a_first_opening_with_a_reason_is_an_open_that_keeps_it(tmp_path):
    log = logged(tmp_path)

    panel(log).open_sealed('decision', reopen_reason='audit')

    (opening, ) = log.records()
    assert (opening['event'], opening['detail']) == ('open', {'reason': 'audit'})

def test_an_opening_of_another_split_or_panel_does_not_count(tmp_path):
    log = logged(tmp_path)
    fingerprint = split_fingerprint(RELATEDNESS_SETTINGS)
    log.append(
        SelectionEvent.OPEN,
        panel='ita_text',
        split='sealed',
        purpose='other panel',
        fingerprint=fingerprint,
        n_queries=1,
    )
    log.append(
        SelectionEvent.OPEN,
        panel='relatedness_co_ownership',
        split='sealed',
        purpose='other salt',
        fingerprint='0' * 64,
        n_queries=1,
    )

    panel(log).open_sealed('decision')

    assert log.records()[-1]['event'] == 'open'

def test_a_second_opening_by_the_same_panel_needs_a_reason(tmp_path):
    log = logged(tmp_path)
    base = panel(log)
    base.open_sealed('decision')

    with pytest.raises(SplitAlreadyOpenedError):
        base.open_sealed('again')
    assert len(log.records()) == 1

# -------------------------------------------------------------------------------------------------
# The data fingerprint
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'part, anchors',
    [('validation', VALIDATION_ANCHORS), ('sealed', SEALED_ANCHORS)],
)
def test_the_data_fingerprint_hashes_the_anchors_gains_candidates_and_tables(
    tmp_path, part, anchors
):
    log = logged(tmp_path)
    base = panel(log)

    assert base.data_fingerprint(part) == hand_fingerprint(anchors)
    assert base.data_fingerprint(RelatednessPart(part)) == hand_fingerprint(anchors)
    assert not log.path.exists()

@pytest.mark.parametrize('table', ['ita_co_ownership', 'co_ownership_marginals'])
def test_the_data_fingerprint_names_each_tables_hash(table):
    table_sha256 = {**TABLE_SHA256, table: 'e' * 64}
    other = panel(table_sha256=table_sha256)

    for part, anchors in (('validation', VALIDATION_ANCHORS), ('sealed', SEALED_ANCHORS)):
        assert other.data_fingerprint(part) == hand_fingerprint(anchors, table_sha256)
        assert other.data_fingerprint(part) != panel().data_fingerprint(part)

def test_the_data_fingerprint_names_the_candidates_without_a_partner():
    # 445131 has a marginal row and no partner: dropping its row changes only the candidates
    targets, marginals = co_ownership_tables()
    fewer = marginals.filter(pl.col('node_id') != 'naics:2022:national_industry:445131')

    narrower = panel(tables=(targets, fewer))

    assert narrower.anchors('validation') == VALIDATION_ANCHORS
    assert narrower.data_fingerprint('validation') != panel().data_fingerprint('validation')
    assert narrower.data_fingerprint('sealed') != panel().data_fingerprint('sealed')

# -------------------------------------------------------------------------------------------------
# The table hashes
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'table_sha256, named',
    [
        ({
            'ita_co_ownership': 'a' * 64
        }, 'exactly'),
        ({
            **TABLE_SHA256, 'other': 'c' * 64
        }, 'exactly'),
        ({
            **TABLE_SHA256, 'co_ownership_marginals': 'B' * 64
        }, 'co_ownership_marginals'),
        ({
            **TABLE_SHA256, 'ita_co_ownership': 'a' * 63
        }, 'ita_co_ownership'),
        ({
            **TABLE_SHA256, 'ita_co_ownership': 'a' * 65
        }, 'ita_co_ownership'),
        ({
            **TABLE_SHA256, 'co_ownership_marginals': 'g' * 64
        }, 'co_ownership_marginals'),
        ({
            **TABLE_SHA256, 'co_ownership_marginals': None
        }, 'co_ownership_marginals'),
    ],
    ids=['missing', 'extra', 'uppercase', 'short', 'long', 'not-hex', 'not-text'],
)
def test_table_hashes_must_be_the_two_tables_sha256(tmp_path, table_sha256, named):
    log = logged(tmp_path)

    with pytest.raises(ValueError, match='table_sha256') as refusal:
        panel(log, table_sha256=table_sha256)

    assert named in str(refusal.value)
    assert not log.path.exists()

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def _bundle_tables(tmp_path: Path) -> Tuple[Path, Path]:
    multi = {'111111': 2, '111112': 2, '222222': 1, '333333': 0}
    pairs = [('111111', '111112', 2), ('111111', '222222', 1)]
    return write_co_ownership_tables(tmp_path, *co_ownership_tables(pairs, multi))

def test_from_files_reads_the_tables_settings_and_log(tmp_path, validated_bundle):
    targets_path, marginals_path = _bundle_tables(tmp_path)
    targets, marginals = pl.read_parquet(targets_path), pl.read_parquet(marginals_path)
    settings = RELATEDNESS_SETTINGS.model_copy(update={'relative_margin': 0.1})
    pins = {
        'co_ownership_marginals': sha256_file(marginals_path),
        'ita_co_ownership': sha256_file(targets_path),
    }

    read = RelatednessPanel.from_files(
        str(targets_path),
        str(marginals_path),
        validated_bundle,
        settings_file(tmp_path, settings),
        str(tmp_path / 'log.jsonl'),
        expected_sha256=pins,
    )

    assert read.co_ownership == co_ownership_inputs(
        targets, marginals, ('111111', '111112', '111113', '222222', '333333')
    )
    assert read.settings == settings
    assert read.log.path == tmp_path / 'log.jsonl'
    assert read.fingerprint == split_fingerprint(settings)
    assert read.table_sha256 == pins
    assert not read.log.path.exists()

def test_from_files_refuses_a_wrong_marginals_pin(tmp_path, validated_bundle):
    targets_path, marginals_path = _bundle_tables(tmp_path)
    found = sha256_file(marginals_path)
    pins = {'ita_co_ownership': sha256_file(targets_path), 'co_ownership_marginals': 'f' * 64}

    with pytest.raises(ValueError) as refusal:
        RelatednessPanel.from_files(
            targets_path,
            marginals_path,
            validated_bundle,
            settings_file(tmp_path, RELATEDNESS_SETTINGS),
            tmp_path / 'log.jsonl',
            expected_sha256=pins,
        )

    for named in ('co_ownership_marginals', 'f' * 64, found):
        assert named in str(refusal.value)
    assert 'ita_co_ownership' not in str(refusal.value)
    assert not (tmp_path / 'log.jsonl').exists()

def test_from_files_checks_every_pin_before_it_reads_either_table(tmp_path, validated_bundle):
    # The target table is not a parquet file but has its right pin, and the marginal table's pin is
    # wrong: the pin is refused, so no table was read
    targets_path, marginals_path = _bundle_tables(tmp_path)
    targets_path.write_bytes(b'not a parquet file')
    pins = {'ita_co_ownership': sha256_file(targets_path), 'co_ownership_marginals': 'f' * 64}

    with pytest.raises(ValueError, match='co_ownership_marginals'):
        RelatednessPanel.from_files(
            targets_path,
            marginals_path,
            validated_bundle,
            settings_file(tmp_path, RELATEDNESS_SETTINGS),
            tmp_path / 'log.jsonl',
            expected_sha256=pins,
        )

@pytest.mark.parametrize(
    'expected_sha256',
    [{
        'ita_co_ownership': 'a' * 64
    }, {
        **TABLE_SHA256, 'ita_co_ownership': 'A' * 64
    }],
    ids=['missing', 'uppercase'],
)
def test_from_files_refuses_pins_that_are_not_the_two_tables_sha256(
    tmp_path, validated_bundle, expected_sha256
):
    targets_path, marginals_path = _bundle_tables(tmp_path)

    with pytest.raises(ValueError, match='expected_sha256'):
        RelatednessPanel.from_files(
            targets_path,
            marginals_path,
            validated_bundle,
            settings_file(tmp_path, RELATEDNESS_SETTINGS),
            tmp_path / 'log.jsonl',
            expected_sha256=expected_sha256,
        )

    assert not (tmp_path / 'log.jsonl').exists()

class _CodebookBundle:
    '''A bundle whose only member is a codebook, as ``from_files`` reads it.'''

    def __init__(self, codebook: Path):
        self.codebook = codebook

    def artifact_path(self, name: str) -> Path:
        assert name == 'codebook'
        return self.codebook

def test_from_files_takes_only_the_codebooks_six_digit_codes(tmp_path):
    codebook = tmp_path / 'codebook.parquet'
    coarser = ['11', '111', '1111', '11111', '31-33']
    pl.DataFrame({'code': [*coarser, *RELATEDNESS_CODES[::-1]]}).write_parquet(codebook)
    targets_path, marginals_path = write_co_ownership_tables(tmp_path, *co_ownership_tables())
    pins = {
        'ita_co_ownership': sha256_file(targets_path),
        'co_ownership_marginals': sha256_file(marginals_path),
    }

    read = RelatednessPanel.from_files(
        targets_path,
        marginals_path,
        _CodebookBundle(codebook),
        settings_file(tmp_path, RELATEDNESS_SETTINGS),
        tmp_path / 'log.jsonl',
        expected_sha256=pins,
    )

    assert read.candidates == CANDIDATES
    assert read.anchors('validation') == VALIDATION_ANCHORS
