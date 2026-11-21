'''
Relatedness's reference choice and margin record (plan 21, Task 7; spec relatedness-artifact, 4.6).

The reference is the arm with the highest validation headline NDCG@10 among the tree and the
text-only table under plain and centered cosine, and a tie goes to the earlier arm of
``REFERENCE_ARMS``. δ = r × that headline, fixed from the margin record's own fresh reference reads
before any candidate read. The margin-first check keys on the record's ``panel`` (P4), so a second
family's record is judged on its own panel's reads, and it counts each reference arm's read once
(P9). Every table is built from ``tests/fixtures/relatedness.py``, and every log is under
``tmp_path``. No test reads a real table.
'''

import copy
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from naics_embedder.decision.decide import read_time
from naics_embedder.decision.relatedness import (
    REFERENCE_ARMS,
    TEXT_ONLY,
    TEXT_ONLY_CENTERED,
    TREE,
    RelatednessMarginRecord,
    check_relatedness_margin,
    check_relatedness_margin_first,
    choose_reference,
    fix_relatedness_margin,
)
from naics_embedder.panels.relatedness import RelatednessPanel
from naics_embedder.panels.relatedness_arms import RelatednessArm, TreeArm, VectorArm
from naics_embedder.panels.relatedness_split import RelatednessPart, split_fingerprint
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from tests.fixtures.relatedness import (
    BASE_MULTI,
    BASE_PAIRS,
    RELATEDNESS_CODES,
    RELATEDNESS_SETTINGS,
    SECOND_FAMILY,
    co_ownership_tables,
    tree_distance,
)

pytestmark = pytest.mark.unit

PANEL = 'relatedness_co_ownership'
PURPOSE = 'fix the relatedness margin'
COMMIT = 'e6e12b2'
NAME = 'relatedness-margins'
TABLE_SHA256 = {'ita_co_ownership': 'a' * 64, 'co_ownership_marginals': 'b' * 64}

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

def logged(tmp_path: Path) -> SelectionLog:
    '''A log that does not exist until the first record is appended.'''

    return SelectionLog(tmp_path / 'selection_log.jsonl')

def panel(log: SelectionLog) -> RelatednessPanel:
    '''The panel over the base tables, writing to ``log``.'''

    targets, marginals = co_ownership_tables()
    return RelatednessPanel(
        targets, marginals, RELATEDNESS_CODES, RELATEDNESS_SETTINGS, log, table_sha256=TABLE_SHA256
    )

def arms() -> Dict[str, RelatednessArm]:
    '''The three reference arms: the tree, and the text-only table under plain and centered cosine.'''

    vectors = np.random.default_rng(7).normal(size=(16, 8))
    return {
        TREE: TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES)),
        TEXT_ONLY: VectorArm(RELATEDNESS_CODES, vectors),
        TEXT_ONLY_CENTERED: VectorArm(RELATEDNESS_CODES, vectors, center=True),
    }

def hand_gain(anchor: str, candidate: str) -> float:
    '''J(anchor, candidate) / m(candidate) from the fixture's pairs, 0.0 with no pair.'''

    for code_a, code_b, j in BASE_PAIRS:
        if {code_a, code_b} == {anchor, candidate}:
            return j / BASE_MULTI[candidate]
    return 0.0

class OracleArm:
    '''An arm that ranks each anchor's candidates by their true gain, so its NDCG@10 is the best.'''

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        return np.array(
            [[hand_gain(anchor, candidate) for candidate in candidates] for anchor in anchors]
        )

class FlatArm:
    '''An arm that ties every candidate, so every flat arm has one headline.'''

    def similarity(self, anchors: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        return np.zeros((len(anchors), len(candidates)))

def fix(
    panel: RelatednessPanel,
    reference_arms: Optional[Mapping[str, RelatednessArm]] = None,
    r: float = 0.05,
) -> RelatednessMarginRecord:
    return fix_relatedness_margin(
        panel,
        arms() if reference_arms is None else reference_arms,
        r,
        NAME,
        embedding_data_commit=COMMIT,
        purpose=PURPOSE,
    )

def read(
    panel: RelatednessPanel,
    role: Optional[str] = None,
    part: str = 'validation',
    purpose: str = 'a stage 11 read',
    arm: str = 'c',
) -> Dict[str, Any]:
    '''One read of ``part`` through the panel, returning the record the log appended.'''

    detail = {'arm': arm, 'seed': 1, 'run': f'stage11/{arm}'}
    if role is not None:
        detail['role'] = role
    if part == 'sealed':
        panel.open_sealed('the decision')
    _, record = panel.score_logged(FlatArm(), part, purpose, detail)
    return record

def append(
    log: SelectionLog,
    *,
    panel: str = PANEL,
    split: str = 'validation',
    event: SelectionEvent = SelectionEvent.READ,
    purpose: str = 'an outside read',
    detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    '''A record that an outside caller appends straight to the log.'''

    return log.append(
        event,
        panel=panel,
        split=split,
        purpose=purpose,
        fingerprint=split_fingerprint(RELATEDNESS_SETTINGS),
        n_queries=7,
        detail=detail,
    )

def _hand_record(
    log: SelectionLog,
    panel_name: str,
    headlines: Mapping[str, float],
    data_fingerprint: str,
    *,
    fixed_at: Optional[datetime] = None,
) -> RelatednessMarginRecord:
    '''
    A second family's margin record (P4): one validation reference read of each arm under
    ``panel_name``, all under one run, with the split fingerprint of ``RELATEDNESS_SETTINGS`` and
    ``data_fingerprint``. It passes ``check_relatedness_margin``, and ``fixed_at`` is now by default.
    '''

    run_id = f'relatedness-margin/{uuid.uuid4().hex}'
    reads = [
        append(
            log,
            panel=panel_name,
            purpose=PURPOSE,
            detail={
                'arm': arm,
                'role': 'reference',
                'run': run_id,
                'data_fingerprint': data_fingerprint,
            },
        ) for arm in REFERENCE_ARMS
    ]
    reference = choose_reference(headlines)
    margin = RelatednessMarginRecord(
        panel=panel_name,
        name=f'{panel_name}-margins',
        r=0.05,
        run_id=run_id,
        reference_headlines=dict(headlines),
        reference=reference,
        margin=0.05 * headlines[reference],
        split_fingerprint=split_fingerprint(RELATEDNESS_SETTINGS),
        data_fingerprint=data_fingerprint,
        embedding_data_commit=COMMIT,
        table_sha256=TABLE_SHA256,
        log_records=reads,
        fixed_at=datetime.now(timezone.utc) if fixed_at is None else fixed_at,
    )
    check_relatedness_margin(margin)
    return margin

HEADLINES = {TREE: 0.30, TEXT_ONLY: 0.42, TEXT_ONLY_CENTERED: 0.40}

# -------------------------------------------------------------------------------------------------
# The reference
# -------------------------------------------------------------------------------------------------

def test_the_reference_is_the_highest_headline_and_a_tie_goes_to_the_simpler_arm():
    assert choose_reference(
        {
            'tree': 0.30,
            'text-only': 0.42,
            'text-only-centered': 0.40
        }
    ) == 'text-only'
    assert choose_reference({'tree': 0.42, 'text-only': 0.42, 'text-only-centered': 0.42}) == 'tree'
    assert choose_reference(
        {
            'tree': 0.30,
            'text-only': 0.42,
            'text-only-centered': 0.42
        }
    ) == 'text-only'
    assert choose_reference(
        {
            'tree': 0.30,
            'text-only': 0.40,
            'text-only-centered': 0.42
        }
    ) == 'text-only-centered'

def test_the_tie_order_is_the_reference_arms_not_the_mappings_order():
    assert REFERENCE_ARMS == ('tree', 'text-only', 'text-only-centered')
    backwards = {TEXT_ONLY_CENTERED: 0.42, TEXT_ONLY: 0.42, TREE: 0.42}
    assert choose_reference(backwards) == TREE
    assert choose_reference({TEXT_ONLY_CENTERED: 0.42, TEXT_ONLY: 0.42, TREE: 0.30}) == TEXT_ONLY
    assert choose_reference({
        TEXT_ONLY_CENTERED: 0.40,
        TEXT_ONLY: 0.30,
        TREE: 0.30
    }) == (TEXT_ONLY_CENTERED)

@pytest.mark.parametrize(
    'headlines, chosen',
    [
        ({
            TREE: 0.0,
            TEXT_ONLY: 0.0,
            TEXT_ONLY_CENTERED: 0.0
        }, TREE),
        ({
            TREE: -0.30,
            TEXT_ONLY: -0.20,
            TEXT_ONLY_CENTERED: -0.20
        }, TEXT_ONLY),
        ({
            TREE: -0.30,
            TEXT_ONLY: -0.40,
            TEXT_ONLY_CENTERED: -0.35
        }, TREE),
        ({
            TREE: 0.0,
            TEXT_ONLY: -0.1,
            TEXT_ONLY_CENTERED: 1e-300
        }, TEXT_ONLY_CENTERED),
    ],
    ids=['all-zero', 'negative-tie', 'all-negative', 'a-tiny-positive-beats-zero'],
)
def test_the_reference_is_the_highest_even_when_a_headline_is_zero_or_negative(headlines, chosen):
    assert choose_reference(headlines) == chosen

CHOOSE_REFUSED = {
    'no-tree': {
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: 0.4
    },
    'no-text-only': {
        TREE: 0.4,
        TEXT_ONLY_CENTERED: 0.4
    },
    'no-centered': {
        TREE: 0.4,
        TEXT_ONLY: 0.4
    },
    'empty': {},
    'another-arm': {
        TREE: 0.4,
        TEXT_ONLY: 0.4,
        'text-only-d16': 0.4
    },
    'a-fourth-arm': {
        TREE: 0.4,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: 0.4,
        'euclidean-d16': 0.5
    },
    'a-key-that-is-not-text': {
        TREE: 0.4,
        TEXT_ONLY: 0.4,
        3: 0.4
    },
    'tree-nan': {
        TREE: math.nan,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: 0.4
    },
    'text-only-nan': {
        TREE: 0.4,
        TEXT_ONLY: math.nan,
        TEXT_ONLY_CENTERED: 0.4
    },
    'centered-nan': {
        TREE: 0.4,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: math.nan
    },
    'tree-inf': {
        TREE: math.inf,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: 0.4
    },
    'text-only-negative-inf': {
        TREE: 0.4,
        TEXT_ONLY: -math.inf,
        TEXT_ONLY_CENTERED: 0.4
    },
    'centered-inf': {
        TREE: 0.4,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: math.inf
    },
}

@pytest.mark.parametrize('headlines', CHOOSE_REFUSED.values(), ids=CHOOSE_REFUSED.keys())
def test_choose_reference_refuses_other_arms_or_a_nan(headlines):
    with pytest.raises(ValueError):
        choose_reference(headlines)

def test_choose_reference_names_what_is_wrong():
    with pytest.raises(ValueError, match="the reference arms are .*'tree'.*not \\['tree'\\]"):
        choose_reference({TREE: 0.4})
    with pytest.raises(ValueError, match="arm 'text-only' has a headline that is not finite"):
        choose_reference({TREE: 0.4, TEXT_ONLY: math.nan, TEXT_ONLY_CENTERED: 0.4})

# -------------------------------------------------------------------------------------------------
# Fixing the margin
# -------------------------------------------------------------------------------------------------

def test_the_margin_is_r_times_the_chosen_references_headline(tmp_path):
    p = panel(logged(tmp_path))
    record = fix_relatedness_margin(
        p,
        arms(),
        0.05,
        'relatedness-margins',
        embedding_data_commit='e6e12b2',
        purpose='fix the margin'
    )
    assert record.reference == choose_reference(record.reference_headlines)
    assert record.margin == record.r * record.reference_headlines[record.reference]
    assert record.panel == 'relatedness_co_ownership' and record.table_sha256 == p.table_sha256
    assert sorted(r['detail']['arm'] for r in record.log_records) == sorted(REFERENCE_ARMS)
    check_relatedness_margin_first(record, p.log)

def test_the_record_holds_the_headlines_the_pins_and_the_reads_it_made(tmp_path):
    p = panel(logged(tmp_path))
    record = fix(p)
    # The headlines are NDCG@10 over the validation part, scored again here under no run
    scored = {arm: p.score(a, RelatednessPart.VALIDATION, 'check') for arm, a in arms().items()}
    expected = {arm: result.headline['ndcg_10'] for arm, result in scored.items()}
    assert len(set(expected.values())) == 3
    assert record.reference_headlines == expected
    assert list(record.reference_headlines) == list(REFERENCE_ARMS)
    assert record.reference == TREE

    assert record.kind == 'relatedness_margin'
    assert record.panel == PANEL and record.name == NAME and record.r == 0.05
    assert record.margin == 0.05 * expected[TREE]
    assert record.embedding_data_commit == COMMIT
    assert record.table_sha256 == TABLE_SHA256
    assert record.split_fingerprint == p.fingerprint
    assert record.data_fingerprint == p.data_fingerprint(RelatednessPart.VALIDATION)
    assert record.run_id.startswith('relatedness-margin/')
    assert len(record.run_id) == len('relatedness-margin/') + 32
    assert record.fixed_at.utcoffset() is not None

    reads = [r for r in p.log.records() if r['detail'].get('run') == record.run_id]
    assert len(reads) == 3
    assert record.log_records == reads
    assert [r['detail']['arm'] for r in reads] == list(REFERENCE_ARMS)
    for read_ in reads:
        assert (read_['event'], read_['panel'], read_['split'], read_['purpose']) == (
            'read', PANEL, 'validation', PURPOSE
        )
        assert read_['fingerprint'] == p.fingerprint
        assert read_['detail']['role'] == 'reference'
        assert read_['detail']['data_fingerprint'] == record.data_fingerprint
        assert 'seed' not in read_['detail']
    assert [r['detail']['arm_type'] for r in reads] == ['TreeArm', 'VectorArm', 'VectorArm']

def test_fix_reads_nothing_but_the_three_reference_arms_on_validation(tmp_path):
    p = panel(logged(tmp_path))
    fix(p)
    assert [(r['panel'], r['split'], r['event'])
            for r in p.log.records()] == [(PANEL, 'validation', 'read')] * 3

def test_the_arms_are_read_in_the_tie_order_whatever_the_callers_order(tmp_path):
    given = arms()
    backwards = {arm: given[arm] for arm in reversed(REFERENCE_ARMS)}
    assert list(backwards) == list(reversed(REFERENCE_ARMS))
    record = fix(panel(logged(tmp_path)), backwards)
    assert [r['detail']['arm'] for r in record.log_records] == list(REFERENCE_ARMS)
    assert list(record.reference_headlines) == list(REFERENCE_ARMS)

def test_each_margin_reads_under_a_fresh_run_id(tmp_path):
    p = panel(logged(tmp_path))
    first, second = fix(p), fix(p)
    assert first.run_id != second.run_id
    assert len(first.log_records) == len(second.log_records) == 3
    assert len(p.log.records()) == 6

def test_the_margin_scales_with_r(tmp_path):
    record = fix(panel(logged(tmp_path)), r=0.2)
    assert record.r == 0.2
    assert record.margin == 0.2 * record.reference_headlines[record.reference]

# The arms' kinds in tie order, the reference, and how many arms share the best headline
REFERENCE_CASES = {
    'tree': (OracleArm, FlatArm, FlatArm, TREE, 1),
    'text-only': (FlatArm, OracleArm, FlatArm, TEXT_ONLY, 1),
    'text-only-centered': (FlatArm, FlatArm, OracleArm, TEXT_ONLY_CENTERED, 1),
    'tree-and-text-only-tie': (OracleArm, OracleArm, FlatArm, TREE, 2),
    'tree-and-centered-tie': (OracleArm, FlatArm, OracleArm, TREE, 2),
    'text-only-and-centered-tie': (FlatArm, OracleArm, OracleArm, TEXT_ONLY, 2),
    'every-arm-ties-flat': (FlatArm, FlatArm, FlatArm, TREE, 3),
    'every-arm-ties-best': (OracleArm, OracleArm, OracleArm, TREE, 3),
}

@pytest.mark.parametrize('case', REFERENCE_CASES.values(), ids=REFERENCE_CASES.keys())
def test_each_arm_is_the_reference_when_it_reads_best_and_ties_go_to_the_earlier_arm(
    tmp_path, case
):
    tree, text_only, centered, expected, tied = case
    reference_arms = {TREE: tree(), TEXT_ONLY: text_only(), TEXT_ONLY_CENTERED: centered()}
    p = panel(logged(tmp_path))
    record = fix(p, reference_arms)
    headlines = record.reference_headlines
    best = max(headlines.values())
    assert sum(headline == best for headline in headlines.values()) == tied
    assert record.reference == expected
    assert headlines[expected] == best
    assert record.margin == record.r * headlines[expected]
    check_relatedness_margin_first(record, p.log)

def test_the_margin_follows_the_reference_headline_not_the_trees(tmp_path):
    # The text-only arm reads best, so δ is r × its headline and not r × the tree's
    record = fix(
        panel(logged(tmp_path)), {
            TREE: FlatArm(),
            TEXT_ONLY: OracleArm(),
            TEXT_ONLY_CENTERED: FlatArm()
        }
    )
    assert record.reference_headlines[TEXT_ONLY] > record.reference_headlines[TREE]
    assert record.margin == 0.05 * record.reference_headlines[TEXT_ONLY]
    assert record.margin != 0.05 * record.reference_headlines[TREE]

REFUSED_BEFORE_READING = {
    'r-zero': (0.0, None, COMMIT),
    'r-negative': (-0.05, None, COMMIT),
    'r-nan': (math.nan, None, COMMIT),
    'r-inf': (math.inf, None, COMMIT),
    'r-negative-inf': (-math.inf, None, COMMIT),
    'no-arms': (0.05, {}, COMMIT),
    'no-tree': (0.05, {
        TEXT_ONLY: FlatArm(),
        TEXT_ONLY_CENTERED: FlatArm()
    }, COMMIT),
    'no-text-only': (0.05, {
        TREE: FlatArm(),
        TEXT_ONLY_CENTERED: FlatArm()
    }, COMMIT),
    'no-centered': (0.05, {
        TREE: FlatArm(),
        TEXT_ONLY: FlatArm()
    }, COMMIT),
    'another-arm': (
        0.05,
        {
            TREE: FlatArm(),
            TEXT_ONLY: FlatArm(),
            'text-only-d16': FlatArm()
        },
        COMMIT,
    ),
    'a-fourth-arm': (
        0.05,
        {
            TREE: FlatArm(),
            TEXT_ONLY: FlatArm(),
            TEXT_ONLY_CENTERED: FlatArm(),
            'euclidean-d16': FlatArm()
        },
        COMMIT,
    ),
    'a-key-that-is-not-text': (
        0.05,
        {
            TREE: FlatArm(),
            TEXT_ONLY: FlatArm(),
            3: FlatArm()
        },
        COMMIT,
    ),
    'commit-empty': (0.05, None, ''),
    'commit-short': (0.05, None, 'e6e12b'),
    'commit-long': (0.05, None, 'e' * 41),
    'commit-uppercase': (0.05, None, 'E6E12B2'),
    'commit-not-hex': (0.05, None, 'e6e12bg'),
    'commit-with-a-newline': (0.05, None, 'e6e12b2\n'),
}

@pytest.mark.parametrize(
    'r, reference_arms, commit', REFUSED_BEFORE_READING.values(), ids=REFUSED_BEFORE_READING.keys()
)
def test_fix_refuses_a_bad_r_or_other_arms_before_reading(tmp_path, r, reference_arms, commit):
    p = panel(logged(tmp_path))
    with pytest.raises(ValueError):
        fix_relatedness_margin(
            p,
            arms() if reference_arms is None else reference_arms,
            r,
            NAME,
            embedding_data_commit=commit,
            purpose=PURPOSE,
        )
    assert not p.log.path.exists()

def test_fix_names_what_it_refuses(tmp_path):
    p = panel(logged(tmp_path))
    with pytest.raises(ValueError, match='positive and finite'):
        fix(p, r=0.0)
    with pytest.raises(ValueError, match='the reference arms are'):
        fix(p, {TREE: FlatArm()})
    with pytest.raises(ValueError, match='embedding_data_commit'):
        fix_relatedness_margin(
            p, arms(), 0.05, NAME, embedding_data_commit='not-a-commit', purpose=PURPOSE
        )

def test_fix_refuses_a_blank_purpose_before_anything_is_logged(tmp_path):
    p = panel(logged(tmp_path))
    with pytest.raises(ValueError, match='purpose'):
        fix_relatedness_margin(p, arms(), 0.05, NAME, embedding_data_commit=COMMIT, purpose='  ')
    assert not p.log.path.exists()

def test_a_short_and_a_full_commit_pass(tmp_path):
    for commit in ('e6e12b2', 'a' * 40):
        record = fix_relatedness_margin(
            panel(logged(tmp_path)),
            arms(),
            0.05,
            NAME,
            embedding_data_commit=commit,
            purpose=PURPOSE
        )
        assert record.embedding_data_commit == commit

# -------------------------------------------------------------------------------------------------
# The record's own check
# -------------------------------------------------------------------------------------------------

def _broken_read(index: int, change):

    def update(margin: RelatednessMarginRecord) -> dict:
        records = [copy.deepcopy(record) for record in margin.log_records]
        change(records[index])
        return {'log_records': records}

    return update

def _set_detail(key, value):

    def change(record):
        record['detail'][key] = value

    return change

def _drop_detail(key):

    def change(record):
        del record['detail'][key]

    return change

def _set(key, value):

    def change(record):
        record[key] = value

    return change

# What is wrong with a record, in each case, and the words its refusal uses. A read's cases run at
# each of the three reads, so no clause checks only one of them
READ_CASES = {
    'not-a-read': (_set('event', 'open'), 'not a read of the validation part'),
    'other-panel': (_set('panel', 'ita_text'), 'not a read of the validation part'),
    'second-family-panel': (_set('panel', SECOND_FAMILY), 'not a read of the validation part'),
    'sealed-read': (_set('split', 'sealed'), 'not a read of the validation part'),
    'other-run': (_set_detail('run', 'stage11/c'), 'under run'),
    'no-run': (_drop_detail('run'), 'under run'),
    'candidate-role': (_set_detail('role', 'candidate'), 'has role'),
    'no-role': (_drop_detail('role'), 'has role'),
    'other-arm': (_set_detail('arm', 'c'), 'reads arm'),
    'no-arm': (_drop_detail('arm'), 'reads arm'),
    'euclidean-arm': (_set_detail('arm', 'euclidean-d16'), 'reads arm'),
    'read-split-fingerprint': (_set('fingerprint', 'f' * 64), 'split fingerprint'),
    'read-data-fingerprint': (_set_detail('data_fingerprint', 'f' * 64), 'data fingerprint'),
    'no-data-fingerprint': (_drop_detail('data_fingerprint'), 'data fingerprint'),
}

RECORD_CASES = {
    'extra-arm': (
        lambda m: {
            'reference_headlines': {
                **m.reference_headlines, 'euclidean-d16': 0.9
            }
        },
        'not the reference arms',
    ),
    'missing-tree': (
        lambda m: {
            'reference_headlines': {
                k: v
                for k, v in m.reference_headlines.items() if k != TREE
            }
        },
        'not the reference arms',
    ),
    'missing-centered': (
        lambda m: {
            'reference_headlines': {
                k: v
                for k, v in m.reference_headlines.items() if k != TEXT_ONLY_CENTERED
            }
        },
        'not the reference arms',
    ),
    'no-headlines': (lambda m: {
        'reference_headlines': {}
    }, 'not the reference arms'),
    'a-headline-key-that-is-not-text': (
        lambda m: {
            'reference_headlines': {
                TREE: m.reference_headlines[TREE],
                TEXT_ONLY: m.reference_headlines[TEXT_ONLY],
                3: 0.9
            }
        },
        'not the reference arms',
    ),
    'other-reference': (lambda m: {
        'reference': TEXT_ONLY
    }, 'names the reference'),
    'centered-reference': (lambda m: {
        'reference': TEXT_ONLY_CENTERED
    }, 'names the reference'),
    'reference-not-an-arm': (lambda m: {
        'reference': 'euclidean-d16'
    }, 'names the reference'),
    'headlines-flip-the-choice': (
        lambda m: {
            'reference_headlines': {
                **m.reference_headlines, TEXT_ONLY: 1.0
            }
        },
        'names the reference',
    ),
    'a-tie-flips-the-choice': (
        lambda m: {
            'reference': TEXT_ONLY,
            'reference_headlines': {
                **m.reference_headlines, TEXT_ONLY: m.reference_headlines[TREE]
            }
        },
        'names the reference',
    ),
    'margin': (lambda m: {
        'margin': math.nextafter(m.margin, 1.0)
    }, 'not r ×'),
    'margin-zero': (lambda m: {
        'margin': 0.0
    }, 'not r ×'),
    'r': (lambda m: {
        'r': 0.1
    }, 'not r ×'),
    'headline-moves-delta': (
        lambda m: {
            'reference_headlines': {
                **m.reference_headlines, TREE: 0.99
            }
        },
        'not r ×',
    ),
    'margin-from-another-headline': (
        lambda m: {
            'margin': m.r * m.reference_headlines[TEXT_ONLY]
        },
        'not r ×',
    ),
    'no-reads': (lambda m: {
        'log_records': []
    }, 'holds no read'),
    'record-panel': (lambda m: {
        'panel': SECOND_FAMILY
    }, 'not a read of the validation part'),
    'record-split-fingerprint': (lambda m: {
        'split_fingerprint': 'f' * 64
    }, 'split fingerprint'),
    'record-data-fingerprint': (lambda m: {
        'data_fingerprint': 'f' * 64
    }, 'data fingerprint'),
    'record-run': (lambda m: {
        'run_id': 'relatedness-margin/other'
    }, 'under run'),
    'an-arm-read-twice': (
        lambda m: {
            'log_records': [*m.log_records[:-1],
                            copy.deepcopy(m.log_records[0])]
        },
        'not once',
    ),
    'the-last-arm-read-twice': (
        lambda m: {
            'log_records': [*m.log_records[:-1],
                            copy.deepcopy(m.log_records[1])]
        },
        'not once',
    ),
    'an-arm-unread': (lambda m: {
        'log_records': m.log_records[:-1]
    }, 'not once'),
    'the-first-arm-unread': (lambda m: {
        'log_records': m.log_records[1:]
    }, 'not once'),
    'one-more-read': (
        lambda m: {
            'log_records': [*m.log_records, copy.deepcopy(m.log_records[1])]
        },
        'not once',
    ),
    'the-same-read-three-times': (
        lambda m: {
            'log_records': [copy.deepcopy(m.log_records[0]) for _ in range(3)]
        },
        'not once',
    ),
}

def _inconsistent_margins() -> Dict[str, Any]:
    cases = dict(RECORD_CASES)
    for key, (change, words) in READ_CASES.items():
        for index in range(3):
            cases[f'{key}-read-{index}'] = (_broken_read(index, change), words)
    return cases

INCONSISTENT_MARGINS = _inconsistent_margins()

@pytest.mark.parametrize(
    'update, disagrees', INCONSISTENT_MARGINS.values(), ids=INCONSISTENT_MARGINS.keys()
)
def test_a_record_that_disagrees_with_itself_is_refused(tmp_path, update, disagrees):
    p = panel(logged(tmp_path))
    margin = fix(p)
    check_relatedness_margin(margin)
    broken = margin.model_copy(update=update(margin))
    with pytest.raises(ValueError, match=f"the margin '{NAME}'.*{re.escape(disagrees)}"):
        check_relatedness_margin(broken)
    with pytest.raises(ValueError, match=f"the margin '{NAME}'.*{re.escape(disagrees)}"):
        check_relatedness_margin_first(broken, p.log)

def test_a_headline_that_is_not_finite_is_refused(tmp_path):
    margin = fix(panel(logged(tmp_path)))
    nan = margin.model_copy(
        update={'reference_headlines': {
            **margin.reference_headlines, TEXT_ONLY: math.nan
        }}
    )
    with pytest.raises(ValueError, match='not finite'):
        check_relatedness_margin(nan)

def test_each_check_names_the_read_and_what_it_found(tmp_path):
    margin = fix(panel(logged(tmp_path)))
    broken = margin.model_copy(update=_broken_read(1, _set_detail('role', 'candidate'))(margin))
    with pytest.raises(ValueError, match="its read 1 has role 'candidate', not 'reference'"):
        check_relatedness_margin(broken)
    other = margin.model_copy(update=_broken_read(2, _set_detail('arm', 'c'))(margin))
    with pytest.raises(ValueError, match="its read 2 reads arm 'c'"):
        check_relatedness_margin(other)

def test_the_second_arm_read_twice_after_the_first_is_refused_for_the_arm_it_names(tmp_path):
    margin = fix(panel(logged(tmp_path)))
    twice = margin.model_copy(
        update={
            'log_records': [margin.log_records[0], margin.log_records[1], margin.log_records[1]]
        }
    )
    with pytest.raises(ValueError, match="arm 'text-only' is read 2 times"):
        check_relatedness_margin(twice)
    unread = margin.model_copy(update={'log_records': margin.log_records[:-1]})
    with pytest.raises(ValueError, match="arm 'text-only-centered' is read 0 times"):
        check_relatedness_margin(unread)

def test_a_consistent_record_passes_each_check_after_a_json_round_trip(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    check_relatedness_margin(margin)
    check_relatedness_margin_first(margin, p.log)
    again = RelatednessMarginRecord.model_validate_json(margin.model_dump_json())
    check_relatedness_margin(again)

def test_the_record_round_trips_through_json(tmp_path):
    p = panel(logged(tmp_path))
    record = fix(p)
    again = RelatednessMarginRecord.model_validate_json(record.model_dump_json())
    assert again == record
    check_relatedness_margin_first(again, p.log)

RECORD_FIELDS: Dict[str, Any] = {
    'panel': PANEL,
    'name': NAME,
    'r': 0.05,
    'run_id': 'relatedness-margin/x',
    'reference_headlines': {
        TREE: 0.5,
        TEXT_ONLY: 0.4,
        TEXT_ONLY_CENTERED: 0.3
    },
    'reference': TREE,
    'margin': 0.025,
    'split_fingerprint': 'a' * 64,
    'data_fingerprint': 'b' * 64,
    'embedding_data_commit': COMMIT,
    'table_sha256': TABLE_SHA256,
    'log_records': [],
    'fixed_at': datetime(2026, 11, 21, tzinfo=timezone.utc),
}

def test_the_record_refuses_extra_keys_a_bad_r_a_nan_and_a_bad_commit():
    record = RelatednessMarginRecord(**RECORD_FIELDS)
    assert record.kind == 'relatedness_margin'
    with pytest.raises(ValidationError):
        record.margin = 0.1
    bad_cases: List[Dict[str, Any]] = [
        {
            'extra': 1
        },
        {
            'r': 0.0
        },
        {
            'r': -0.05
        },
        {
            'r': math.inf
        },
        {
            'margin': math.nan
        },
        {
            'margin': math.inf
        },
        {
            'reference_headlines': {
                TREE: math.nan,
                TEXT_ONLY: 0.4,
                TEXT_ONLY_CENTERED: 0.3
            }
        },
        {
            'kind': 'ita_margin'
        },
        {
            'embedding_data_commit': ''
        },
        {
            'embedding_data_commit': 'e6e12b'
        },
        {
            'embedding_data_commit': 'e' * 41
        },
        {
            'embedding_data_commit': 'E6E12B2'
        },
        {
            'embedding_data_commit': 'e6e12bg'
        },
    ]
    for bad in bad_cases:
        with pytest.raises(ValidationError):
            RelatednessMarginRecord(**{**RECORD_FIELDS, **bad})
    for missing in ('panel', 'embedding_data_commit', 'table_sha256', 'fixed_at'):
        with pytest.raises(ValidationError):
            RelatednessMarginRecord(**{k: v for k, v in RECORD_FIELDS.items() if k != missing})

# -------------------------------------------------------------------------------------------------
# Margin first
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('arm', ['c', None], ids=['other-arm', 'no-arm'])
def test_a_margin_after_a_reference_read_of_another_arm_is_refused(tmp_path, arm):
    p = panel(logged(tmp_path))
    detail = {'role': 'reference', 'run': 'outside/c'}
    if arm is not None:
        detail['arm'] = arm
    early = append(p.log, purpose='a mislabeled candidate read', detail=detail)
    margin = fix(p)
    with pytest.raises(ValueError, match='a mislabeled candidate read') as refused:
        check_relatedness_margin_first(margin, p.log)
    assert early['time'] in str(refused.value)

@pytest.mark.parametrize('arm', REFERENCE_ARMS)
def test_a_margin_after_an_outside_reference_read_of_a_reference_arm_passes(tmp_path, arm):
    p = panel(logged(tmp_path))
    append(
        p.log,
        purpose='a stage 11 gap read',
        detail={
            'arm': arm,
            'role': 'reference',
            'run': 'outside/ref'
        },
    )
    check_relatedness_margin_first(fix(p), p.log)

@pytest.mark.parametrize('arm', REFERENCE_ARMS)
@pytest.mark.parametrize('role', ['candidate', None, 'monitor'])
def test_a_margin_after_a_read_of_a_reference_arm_that_is_not_a_reference_read_is_refused(
    tmp_path, arm, role
):
    p = panel(logged(tmp_path))
    detail = {'arm': arm, 'run': 'outside/ref'}
    if role is not None:
        detail['role'] = role
    early = append(p.log, purpose='a candidate read of a reference arm', detail=detail)
    margin = fix(p)
    with pytest.raises(ValueError, match='a candidate read of a reference arm') as refused:
        check_relatedness_margin_first(margin, p.log)
    assert early['time'] in str(refused.value)

def test_a_margin_after_a_candidate_read_is_refused(tmp_path):
    p = panel(logged(tmp_path))
    early = read(p, role='candidate', purpose='an early candidate read')
    margin = fix(p)
    with pytest.raises(ValueError, match='an early candidate read') as refused:
        check_relatedness_margin_first(margin, p.log)
    assert early['time'] in str(refused.value)

def test_a_margin_after_a_read_with_no_role_is_refused(tmp_path):
    p = panel(logged(tmp_path))
    early = read(p, role=None, purpose='a monitor read')
    assert 'role' not in early['detail']
    margin = fix(p)
    with pytest.raises(ValueError, match='a monitor read') as refused:
        check_relatedness_margin_first(margin, p.log)
    assert early['time'] in str(refused.value)

def test_a_margin_after_a_read_with_another_role_is_refused(tmp_path):
    p = panel(logged(tmp_path))
    # The panel refuses such a role, but an outside caller can append the read to the log
    append(
        p.log,
        purpose='an outside monitor read',
        detail={
            'arm': 'c',
            'role': 'monitor',
            'run': 'outside/c'
        },
    )
    with pytest.raises(ValueError, match='an outside monitor read'):
        check_relatedness_margin_first(fix(p), p.log)

def test_a_margin_after_a_validation_read_with_no_detail_is_refused(tmp_path):
    p = panel(logged(tmp_path))
    append(p.log, purpose='a bare read')
    with pytest.raises(ValueError, match='a bare read'):
        check_relatedness_margin_first(fix(p), p.log)

def test_ita_reads_do_not_block_a_relatedness_margin(tmp_path):
    p = panel(logged(tmp_path))
    # An ita_text validation read with role candidate, appended first
    ita = append(
        p.log,
        panel='ita_text',
        purpose='a stage 5 ita read',
        detail={
            'arm': 'c',
            'role': 'candidate',
            'run': 'stage5/c'
        },
    )
    margin = fix(p)
    assert ita['panel'] == 'ita_text'
    check_relatedness_margin_first(margin, p.log)

def test_earlier_reads_of_other_panels_the_sealed_part_and_openings_do_not_count(tmp_path):
    p = panel(logged(tmp_path))
    read(p, role='candidate', part='sealed', purpose='a sealed candidate read')
    append(p.log, panel='outcome', purpose='a stage 7 read', detail={'run': 'stage7/x'})
    append(
        p.log,
        panel='ita_text',
        purpose='an ita read with no role',
        detail={
            'arm': 'c',
            'run': 'stage5/c'
        }
    )
    append(
        p.log,
        panel=SECOND_FAMILY,
        purpose='a second family candidate read',
        detail={
            'arm': 'c',
            'role': 'candidate',
            'run': 'stage11/c'
        },
    )
    append(
        p.log,
        event=SelectionEvent.OPEN,
        purpose='an opening of the validation part',
        detail={'reason': 'a record that is not a read'},
    )
    append(
        p.log,
        event=SelectionEvent.REOPEN,
        purpose='a reopening of the validation part',
        detail={'reason': 'a record that is not a read'},
    )
    check_relatedness_margin_first(fix(p), p.log)

def test_earlier_reference_reads_and_later_candidate_reads_pass(tmp_path):
    p = panel(logged(tmp_path))
    read(p, role='reference', purpose='a stage 11 gap read', arm=TEXT_ONLY)
    margin = fix(p)
    read(p, role='candidate')
    read(p, role=None)
    read(p, role='candidate', part='sealed')
    check_relatedness_margin_first(margin, p.log)

def test_a_log_with_no_record_leaves_nothing_to_refuse(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    empty = SelectionLog(tmp_path / 'no-such-log.jsonl')
    check_relatedness_margin_first(margin, empty)
    assert not empty.path.exists()

def test_a_candidate_read_before_fixed_at_is_refused_when_fixed_at_is_moved_later(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    late = read(p, role='candidate', purpose='a later candidate read')
    later = margin.model_copy(update={'fixed_at': read_time(late, 'read') + timedelta(seconds=1)})
    with pytest.raises(ValueError, match='a later candidate read') as refused:
        check_relatedness_margin_first(later, p.log)
    assert late['time'] in str(refused.value)

def test_a_candidate_read_exactly_at_fixed_at_is_not_before_it(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    late = read(p, role='candidate', purpose='a later candidate read')
    at = margin.model_copy(update={'fixed_at': read_time(late, 'read')})
    check_relatedness_margin_first(at, p.log)
    after = margin.model_copy(
        update={'fixed_at': read_time(late, 'read') + timedelta(microseconds=1)}
    )
    with pytest.raises(ValueError, match='a later candidate read'):
        check_relatedness_margin_first(after, p.log)

def test_a_read_in_another_timezone_is_compared_as_an_instant(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    # An hour before fixed_at, written at UTC+14, so that its text sorts after fixed_at's
    early = (margin.fixed_at - timedelta(hours=1)).astimezone(timezone(timedelta(hours=14)))
    assert early.isoformat() > margin.fixed_at.isoformat()
    log = SelectionLog(tmp_path / 'outside.jsonl')
    record = append(log, purpose='a candidate read at UTC+14', detail={'arm': 'c', 'run': 'x/c'})
    log.path.write_text(
        json.dumps({
            **record, 'time': early.isoformat()
        }, sort_keys=True) + '\n', encoding='utf-8'
    )
    with pytest.raises(ValueError, match='a candidate read at UTC\\+14'):
        check_relatedness_margin_first(margin, log)

@pytest.mark.parametrize(
    'time', [None, '2026-11-21T10:00:00', 'yesterday'], ids=['none', 'naive', 'bad']
)
def test_a_scanned_read_with_no_usable_time_is_refused(tmp_path, time):
    p = panel(logged(tmp_path))
    margin = fix(p)
    log = SelectionLog(tmp_path / 'outside.jsonl')
    record = append(
        log, purpose='a candidate read with a bad time', detail={
            'arm': 'c',
            'run': 'x/c'
        }
    )
    log.path.write_text(
        json.dumps({
            **record, 'time': time
        }, sort_keys=True) + '\n', encoding='utf-8'
    )
    with pytest.raises(ValueError, match='a candidate read with a bad time'):
        check_relatedness_margin_first(margin, log)

def test_a_naive_fixed_at_or_a_read_after_fixed_at_is_refused(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    naive = margin.model_copy(update={'fixed_at': margin.fixed_at.replace(tzinfo=None)})
    with pytest.raises(ValueError, match='timezone'):
        check_relatedness_margin_first(naive, p.log)
    first = read_time(margin.log_records[0], 'read')
    early = margin.model_copy(update={'fixed_at': first - timedelta(seconds=1)})
    with pytest.raises(ValueError, match='before its read'):
        check_relatedness_margin_first(early, p.log)
    check_relatedness_margin_first(margin, p.log)

def test_each_of_the_margins_own_reads_must_come_by_fixed_at(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    times = [read_time(record, 'read') for record in margin.log_records]
    last = max(times)
    # fixed_at at the last read passes, and a microsecond before it refuses, whichever read is last
    check_relatedness_margin_first(margin.model_copy(update={'fixed_at': last}), p.log)
    with pytest.raises(ValueError, match='before its read'):
        check_relatedness_margin_first(
            margin.model_copy(update={'fixed_at': last - timedelta(microseconds=1)}), p.log
        )
    for index, record in enumerate(margin.log_records):
        later = copy.deepcopy(list(margin.log_records))
        later[index]['time'] = (last + timedelta(seconds=5)).isoformat()
        with pytest.raises(ValueError, match='before its read'):
            check_relatedness_margin_first(margin.model_copy(update={'log_records': later}), p.log)

def test_a_naive_fixed_at_is_refused_before_the_log_is_read(tmp_path):
    margin = fix(panel(logged(tmp_path)))
    naive = margin.model_copy(update={'fixed_at': margin.fixed_at.replace(tzinfo=None)})
    unread = SelectionLog(tmp_path / 'no-such-log.jsonl')
    with pytest.raises(ValueError, match='timezone'):
        check_relatedness_margin_first(naive, unread)

# -------------------------------------------------------------------------------------------------
# A second family
# -------------------------------------------------------------------------------------------------

def test_a_second_familys_margin_is_judged_on_its_own_reads(tmp_path):
    p = panel(logged(tmp_path))
    # Candidate reads on the first family, and then a second family's record
    read(p, role='candidate', purpose='a first family candidate read')
    read(p, role=None, purpose='a first family monitor read')
    second = _hand_record(p.log, SECOND_FAMILY, HEADLINES, 'c' * 64)
    assert second.panel == SECOND_FAMILY and second.reference == TEXT_ONLY
    check_relatedness_margin_first(second, p.log)

    # The first family has not fixed its margin, so its record is refused for those reads
    first = fix(p)
    with pytest.raises(ValueError, match='a first family candidate read'):
        check_relatedness_margin_first(first, p.log)

    # A candidate read on the second family, before a second hand record's fixed_at
    early = append(
        p.log,
        panel=SECOND_FAMILY,
        purpose='a second family candidate read',
        detail={
            'arm': 'c',
            'role': 'candidate',
            'run': 'stage11/c'
        },
    )
    refused = _hand_record(p.log, SECOND_FAMILY, HEADLINES, 'c' * 64)
    with pytest.raises(ValueError, match='a second family candidate read') as error:
        check_relatedness_margin_first(refused, p.log)
    assert early['time'] in str(error.value)

def test_a_second_familys_candidate_read_does_not_block_the_first_familys_margin(tmp_path):
    p = panel(logged(tmp_path))
    append(
        p.log,
        panel=SECOND_FAMILY,
        purpose='a second family candidate read',
        detail={
            'arm': 'c',
            'role': 'candidate',
            'run': 'stage11/c'
        },
    )
    check_relatedness_margin_first(fix(p), p.log)

def test_a_second_familys_record_is_refused_for_a_read_of_the_first_familys_panel_name(tmp_path):
    p = panel(logged(tmp_path))
    second = _hand_record(p.log, SECOND_FAMILY, HEADLINES, 'c' * 64)
    as_first = second.model_copy(update={'panel': PANEL})
    with pytest.raises(ValueError, match='not a read of the validation part'):
        check_relatedness_margin_first(as_first, p.log)

def test_a_hand_records_fixed_at_can_be_given(tmp_path):
    p = panel(logged(tmp_path))
    later = datetime.now(timezone.utc) + timedelta(days=1)
    check_relatedness_margin_first(
        _hand_record(p.log, SECOND_FAMILY, HEADLINES, 'c' * 64, fixed_at=later), p.log
    )
    earlier = datetime.now(timezone.utc) - timedelta(days=1)
    before = _hand_record(p.log, SECOND_FAMILY, HEADLINES, 'c' * 64, fixed_at=earlier)
    with pytest.raises(ValueError, match='before its read'):
        check_relatedness_margin_first(before, p.log)
