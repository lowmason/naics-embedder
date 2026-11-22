'''
Relatedness's reference choice and margin record (plan 21, Task 7; spec relatedness-artifact, 4.6).

The reference is the arm with the highest validation headline NDCG@10 among the tree and the
text-only table under plain and centered cosine, and a tie goes to the earlier arm of
``REFERENCE_ARMS``. δ = r × that headline, fixed from the margin record's own fresh reference reads
before any candidate read. The margin-first check keys on the record's ``panel`` (P4), so a second
family's record is judged on its own panel's reads, and it counts each reference arm's read once
(P9). Every table is built from ``tests/fixtures/relatedness.py``, and every log is under
``tmp_path``. No test reads a real table.

Task 8 adds the beat-by-δ test (spec 4.6): a candidate beats the reference on a judging family when
the lower bound of the interval on Δ exceeds +δ, and it goes forward when it beats on every family,
at the level 1 − 0.05/n for n families. Each family has its own unit stream and its own margin.

Task 9 adds the provenance check (spec 4.6): a table whose provenance names a judging input, such
as co-ownership, is never decided on. Each of its eight checks runs in the brief's order, and each
refusal names its check first. The check reads no file, so every provenance here is a dictionary.
'''

import copy
import dataclasses
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from naics_embedder.decision import relatedness
from naics_embedder.decision.decide import read_time
from naics_embedder.decision.relatedness import (
    CODEBOOK_INPUTS,
    PROVENANCE_ENTRIES,
    REFERENCE_ARMS,
    RELATEDNESS_EXPORT,
    TEXT_ONLY,
    TEXT_ONLY_CENTERED,
    TRAINING_FAMILY_INPUTS,
    TREE,
    FamilyDelta,
    FamilyVerdict,
    RelatednessMarginRecord,
    RelatednessVerdict,
    arm_replicates,
    beat_by_margin,
    check_relatedness_margin,
    check_relatedness_margin_first,
    check_relatedness_provenance,
    choose_reference,
    fix_relatedness_margin,
    headline_values,
    relatedness_test,
)
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import superiority_level
from naics_embedder.panels.relatedness import RelatednessPanel
from naics_embedder.panels.relatedness_arms import RelatednessArm, TreeArm, VectorArm
from naics_embedder.panels.relatedness_scoring import RelatednessResult
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

def test_the_refusal_for_a_candidate_read_names_fixed_at_and_the_reads_purpose_and_time(tmp_path):
    p = panel(logged(tmp_path))
    early = read(p, role='candidate', purpose='an early candidate read')
    margin = fix(p)
    with pytest.raises(ValueError) as refused:
        check_relatedness_margin_first(margin, p.log)
    message = str(refused.value)
    assert f'was fixed at {margin.fixed_at.isoformat()}, after the candidate read' in message
    assert f"'an early candidate read' at {early['time']}" in message
    assert early['time'] != margin.fixed_at.isoformat()

def test_the_refusal_for_an_own_read_after_fixed_at_names_fixed_at_and_the_reads_time(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    first = read_time(margin.log_records[0], 'read')
    early = margin.model_copy(update={'fixed_at': first - timedelta(seconds=1)})
    with pytest.raises(ValueError) as refused:
        check_relatedness_margin_first(early, p.log)
    assert f'was fixed at {early.fixed_at.isoformat()}, before its read at {first.isoformat()}' in (
        str(refused.value)
    )

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

# -------------------------------------------------------------------------------------------------
# The beat-by-δ test: helpers
# -------------------------------------------------------------------------------------------------

REPLICATES = 2000
BOOTSTRAP_SEED = 20260924
FLOOR = 5
SECOND_FINGERPRINT = 'd' * 64
OTHER_FINGERPRINT = 'f' * 64

class Inputs(NamedTuple):
    '''The three mappings ``relatedness_test`` takes, keyed by family.'''

    candidate_seeds: Dict[str, List[RelatednessResult]]
    references: Dict[str, RelatednessResult]
    margins: Dict[str, RelatednessMarginRecord]

    def only(self, *keys: str) -> 'Inputs':
        '''The same inputs on the given families, in that order.'''

        return Inputs(
            {key: self.candidate_seeds[key]
             for key in keys},
            {key: self.references[key]
             for key in keys},
            {key: self.margins[key]
             for key in keys},
        )

def offset(result: RelatednessResult, amount: float) -> RelatednessResult:
    '''``result`` with every anchor's NDCG@10 raised by ``amount``.'''

    return dataclasses.replace(
        result, per_anchor=result.per_anchor.with_columns(pl.col('ndcg_10') + amount)
    )

def candidate_seeds_of(reference: RelatednessResult, count: int = FLOOR) -> List[RelatednessResult]:
    '''Candidate seeds i = 0 to ``count`` - 1: ``reference`` with NDCG@10 + 0.1 + 0.001 × i.'''

    return [offset(reference, 0.1 + 0.001 * i) for i in range(count)]

def two_families(tmp_path: Path) -> Inputs:
    '''
    Two judging families, five candidate seeds on each, all read on the validation part.

    The first family is the panel's: its margin is fixed from the three reference reads, its
    reference is the arm that margin chooses (the tree on the fixture), and its fingerprint is the
    panel's. The second family is ``SECOND_FAMILY`` with the text-only arm's validation read as its
    reference, a hand-made margin that chooses text-only (δ = 0.05 × 0.42) and a fingerprint of its
    own, so the two families differ in every input.
    '''

    p = panel(logged(tmp_path))
    margin = fix(p)
    reads = {
        name: p.score(
            arm, 'validation', 'a reference read', {
                'arm': name,
                'role': 'reference',
                'run': 'stage11/reference'
            }
        )
        for name, arm in arms().items()
    }
    first = reads[margin.reference]
    second = dataclasses.replace(
        reads[TEXT_ONLY], panel=SECOND_FAMILY, data_fingerprint=SECOND_FINGERPRINT
    )
    return Inputs(
        {
            PANEL: candidate_seeds_of(first),
            SECOND_FAMILY: candidate_seeds_of(second)
        },
        {
            PANEL: first,
            SECOND_FAMILY: second
        },
        {
            PANEL: margin,
            SECOND_FAMILY: _hand_record(p.log, SECOND_FAMILY, HEADLINES, SECOND_FINGERPRINT)
        },
    )

def one_family(tmp_path: Path) -> Inputs:
    '''The first family of ``two_families``.'''

    return two_families(tmp_path).only(PANEL)

def run(inputs: Inputs, candidate: str = 'c', **overrides: Any) -> RelatednessVerdict:
    '''``relatedness_test`` on the inputs, with the floor, replicates and seed the tests use.'''

    settings = {
        'min_seeds': FLOOR,
        'replicates': REPLICATES,
        'bootstrap_seed': BOOTSTRAP_SEED,
        **overrides
    }
    return relatedness_test(
        candidate, inputs.candidate_seeds, inputs.references, inputs.margins, **settings
    )

def expected_delta(inputs: Inputs, key: str, candidate: str = 'c') -> FamilyDelta:
    '''
    The family's Δ built from ``arm_replicates``, which its own tests pin to the draw primitives:
    the candidate's seeds and the one reference result, both on the stream of the family.
    '''

    reference = inputs.references[key]
    margin = inputs.margins[key]
    items = PanelItems.from_values(headline_values(reference))
    settings = {'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    candidate_point, candidate_draws = arm_replicates(
        items, key, candidate, inputs.candidate_seeds[key], **settings
    )
    reference_point, reference_draws = arm_replicates(
        items, key, margin.reference, [reference], **settings
    )
    return FamilyDelta(
        key,
        margin.reference,
        candidate_point - reference_point,
        candidate_draws - reference_draws,
        margin.margin,
    )

def with_seed(inputs: Inputs, key: str, position: int, **changes: Any) -> Inputs:
    '''``inputs`` with ``changes`` made to one candidate seed of the family.'''

    seeds = list(inputs.candidate_seeds[key])
    seeds[position] = dataclasses.replace(seeds[position], **changes)
    return inputs._replace(candidate_seeds={**inputs.candidate_seeds, key: seeds})

def with_reference(inputs: Inputs, key: str, **changes: Any) -> Inputs:
    '''``inputs`` with ``changes`` made to the family's reference result.'''

    reference = dataclasses.replace(inputs.references[key], **changes)
    return inputs._replace(references={**inputs.references, key: reference})

def without_anchor(result: RelatednessResult, anchor: str) -> RelatednessResult:
    '''``result`` with the anchor's row dropped.'''

    return dataclasses.replace(
        result, per_anchor=result.per_anchor.filter(pl.col('anchor') != anchor)
    )

def renamed_anchor(result: RelatednessResult, old: str, new: str) -> RelatednessResult:
    '''``result`` with the anchor ``old`` named ``new``: the same count of anchors, another set.'''

    return dataclasses.replace(
        result, per_anchor=result.per_anchor.with_columns(pl.col('anchor').str.replace(old, new))
    )

def on_anchors(
    inputs: Inputs, key: str, change: Callable[[RelatednessResult], RelatednessResult]
) -> Inputs:
    '''``inputs`` with ``change`` made to the family's reference and to each of its candidate seeds.'''

    seeds = [change(seed) for seed in inputs.candidate_seeds[key]]
    return inputs._replace(
        candidate_seeds={
            **inputs.candidate_seeds, key: seeds
        },
        references={
            **inputs.references, key: change(inputs.references[key])
        },
    )

# -------------------------------------------------------------------------------------------------
# Headline values and replicates
# -------------------------------------------------------------------------------------------------

def test_the_headline_values_are_each_anchors_ndcg_10_with_the_anchor_as_unit_and_item(tmp_path):
    result = one_family(tmp_path).references[PANEL]
    values = headline_values(result)
    assert values.columns == ['unit', 'item', 'value']
    anchors = result.per_anchor.get_column('anchor').to_list()
    assert values.get_column('unit').to_list() == anchors
    assert values.get_column('item').to_list() == anchors
    assert values.get_column('value').to_list() == result.per_anchor.get_column('ndcg_10').to_list()
    # NDCG@20 is a report, not the headline
    assert values.get_column('value').to_list() != result.per_anchor.get_column('ndcg_20').to_list()
    items = PanelItems.from_values(values)
    assert len(items.items) == len(items.units) == result.headline['n_anchors'] == 7
    assert items.sizes.tolist() == [1.0] * 7

def test_the_point_statistic_is_the_headline(tmp_path):
    p = panel(logged(tmp_path))
    p.open_sealed('the decision')
    for part in ('validation', 'sealed'):
        for name, arm in arms().items():
            result = p.score(arm, part, 'a read', {'arm': name})
            values = headline_values(result)
            items = PanelItems.from_values(values)
            point = point_statistic(items.sums(items.values(values)), items.sizes)
            # The panel's headline is a correctly rounded fsum mean, and the point statistic a
            # float sum over the same values: they agree to the last bit or two
            assert point == pytest.approx(result.headline['ndcg_10'], rel=0, abs=1e-15)

@pytest.mark.parametrize('panel_name', [PANEL, SECOND_FAMILY])
@pytest.mark.parametrize('arm', ['c', 'd'])
def test_arm_replicates_draw_units_on_the_panel_and_seeds_on_the_panel_and_arm(
    tmp_path, panel_name, arm
):
    reference = one_family(tmp_path).references[PANEL]
    seeds = candidate_seeds_of(reference, 3)
    items = PanelItems.from_values(headline_values(reference))
    point, replicates = arm_replicates(
        items, panel_name, arm, seeds, replicates=REPLICATES, bootstrap_seed=BOOTSTRAP_SEED
    )
    sums = items.sums(np.stack([items.values(headline_values(seed)) for seed in seeds]))
    units = unit_draws(panel_name, 7, REPLICATES, BOOTSTRAP_SEED)
    seed_counts = seed_draws(panel_name, arm, 3, REPLICATES, BOOTSTRAP_SEED)
    assert point == point_statistic(sums, items.sizes)
    np.testing.assert_array_equal(
        replicates, replicate_statistics(sums, items.sizes, units, seed_counts)
    )
    assert replicates.shape == (REPLICATES, )

def test_a_one_seed_arm_draws_no_seeds(tmp_path):
    # Its point is the seed's headline and its replicates the headline on each unit resample
    reference = one_family(tmp_path).references[PANEL]
    items = PanelItems.from_values(headline_values(reference))
    point, replicates = arm_replicates(
        items, PANEL, 'tree', [reference], replicates=REPLICATES, bootstrap_seed=BOOTSTRAP_SEED
    )
    assert point == pytest.approx(reference.headline['ndcg_10'], abs=1e-15)
    values = items.values(headline_values(reference))
    units = unit_draws(PANEL, 7, REPLICATES, BOOTSTRAP_SEED)
    np.testing.assert_allclose(replicates, units @ values / 7, rtol=0, atol=1e-14)

def test_arm_replicates_refuses_an_arm_with_no_seed(tmp_path):
    items = PanelItems.from_values(headline_values(one_family(tmp_path).references[PANEL]))
    with pytest.raises(ValueError, match="arm 'c' has no seed"):
        arm_replicates(items, PANEL, 'c', [], replicates=REPLICATES, bootstrap_seed=BOOTSTRAP_SEED)

def test_arm_replicates_refuses_a_seed_on_other_anchors(tmp_path):
    reference = one_family(tmp_path).references[PANEL]
    items = PanelItems.from_values(headline_values(reference))
    renamed = dataclasses.replace(
        reference,
        per_anchor=reference.per_anchor.with_columns(
            pl.col('anchor').str.replace('111110', '999999')
        )
    )
    settings = {'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    for seeds in (
        [without_anchor(reference, '111110')],
        [reference, without_anchor(reference, '445110')],
        [renamed],
    ):
        with pytest.raises(ValueError, match='items'):
            arm_replicates(items, PANEL, 'c', seeds, **settings)

# -------------------------------------------------------------------------------------------------
# The verdict over n families
# -------------------------------------------------------------------------------------------------

def test_the_verdict_at_n_1_and_n_2():
    a = FamilyDelta('relatedness_co_ownership', 'tree', 0.02, np.linspace(0.0, 0.04, 1001), 0.0008)
    one = beat_by_margin('c', [a])
    assert one.level == 0.95 and one.families[0].interval[0] == pytest.approx(0.001) and one.beats
    b = FamilyDelta(SECOND_FAMILY, 'tree', 0.04, np.linspace(0.02, 0.06, 1001), 0.01)
    two = beat_by_margin('c', [a, b])
    assert two.level == pytest.approx(0.975)
    assert two.families[0].interval[0] == pytest.approx(0.0005) and not two.families[0].beats
    assert two.families[1].beats and not two.beats

def test_a_superior_candidate_that_does_not_beat_by_delta_does_not_go_forward():
    a = FamilyDelta('relatedness_co_ownership', 'tree', 0.02, np.linspace(0.0, 0.04, 1001), 0.0015)
    verdict = beat_by_margin('c', [a])
    assert verdict.families[0].interval[0] > 0 and not verdict.beats

def test_beat_by_margin_refuses_no_family_or_a_repeated_one():
    spread = np.linspace(0.0, 0.04, 1001)
    a = FamilyDelta(PANEL, 'tree', 0.02, spread, 0.001)
    b = FamilyDelta(SECOND_FAMILY, 'tree', 0.02, spread, 0.001)
    with pytest.raises(ValueError, match='got none'):
        beat_by_margin('c', [])
    with pytest.raises(ValueError, match='repeats'):
        beat_by_margin('c', [a, a])
    # A family repeated beside another one, and not next to its first
    with pytest.raises(ValueError, match='repeats') as raised:
        beat_by_margin('c', [a, b, dataclasses.replace(a, reference='text-only')])
    assert repr(PANEL) in str(raised.value) and SECOND_FAMILY not in str(raised.value)
    assert beat_by_margin('c', [a, b]).level == pytest.approx(0.975)

def test_the_level_is_one_minus_five_percent_over_the_number_of_families():
    spread = np.linspace(0.0, 0.04, 1001)
    names = [PANEL, SECOND_FAMILY, 'relatedness_third_family']
    deltas = [FamilyDelta(name, 'tree', 0.02, spread, 0.0) for name in names]
    for n in (1, 2, 3):
        verdict = beat_by_margin('c', deltas[:n])
        assert verdict.level == superiority_level(n) == 1.0 - 0.05 / n
        assert len(verdict.families) == n
        for family in verdict.families:
            assert family.interval == percentile_interval(spread, verdict.level)

def test_a_family_beats_only_when_the_lower_bound_exceeds_its_margin():

    def beats(margin, replicates):
        deltas = [FamilyDelta(PANEL, 'tree', 0.03, replicates, margin)]
        return beat_by_margin('c', deltas).families[0].beats

    flat = np.full(100, 0.03)
    assert beats(0.03, flat) is False
    assert beats(0.0299999, flat) is True
    assert beats(0.0300001, flat) is False
    # The upper bound is not the test: this interval is (0.001, 0.039)
    spread = np.linspace(0.0, 0.04, 1001)
    assert beats(0.01, spread) is False
    assert beats(0.0009, spread) is True

@pytest.mark.parametrize('failing', [0, 1, 2])
def test_the_candidate_goes_forward_only_when_it_beats_on_every_family(failing):
    spread = np.linspace(0.0, 0.04, 1001)
    names = [PANEL, SECOND_FAMILY, 'relatedness_third_family']
    deltas = [
        FamilyDelta(name, 'tree', 0.02, spread, 0.04 if index == failing else 0.0002)
        for index, name in enumerate(names)
    ]
    verdict = beat_by_margin('c', deltas)
    assert [family.beats for family in verdict.families] == [index != failing for index in range(3)]
    assert verdict.beats is False
    everyone = [dataclasses.replace(delta, margin=0.0002) for delta in deltas]
    assert beat_by_margin('c', everyone).beats is True

def test_the_verdict_keeps_each_familys_fields_in_the_order_given():
    spread = np.linspace(0.0, 0.04, 1001)
    first = FamilyDelta(SECOND_FAMILY, 'text-only', 0.011, spread, 0.0004)
    second = FamilyDelta(PANEL, 'tree', 0.022, spread + 0.001, 0.0007)
    verdict = beat_by_margin('a candidate', [first, second])
    assert verdict.candidate == 'a candidate'
    assert [(f.panel, f.reference, f.delta, f.margin) for f in verdict.families] == [
        (SECOND_FAMILY, 'text-only', 0.011, 0.0004),
        (PANEL, 'tree', 0.022, 0.0007),
    ]
    assert verdict.families[1].interval == percentile_interval(spread + 0.001, verdict.level)

def test_the_verdict_holds_plain_python_values():
    spread = np.linspace(0.0, 0.04, 1001)
    delta = FamilyDelta(PANEL, 'tree', np.float64(0.02), spread, np.float64(0.0008))
    verdict = beat_by_margin('c', [delta])
    family = verdict.families[0]
    assert type(verdict.level) is float and type(verdict.beats) is bool
    assert type(verdict.families) is tuple
    assert type(family.beats) is bool and type(family.delta) is float
    assert type(family.margin) is float and type(family.interval) is tuple
    assert [type(end) for end in family.interval] == [float, float]
    assert isinstance(family, FamilyVerdict) and isinstance(verdict, RelatednessVerdict)

def test_the_verdicts_and_the_deltas_are_frozen():
    delta = FamilyDelta(PANEL, 'tree', 0.02, np.linspace(0.0, 0.04, 1001), 0.0008)
    verdict = beat_by_margin('c', [delta])
    for frozen, name in ((delta, 'delta'), (verdict, 'beats'), (verdict.families[0], 'margin')):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(frozen, name, 1)

# -------------------------------------------------------------------------------------------------
# The test over the judging families
# -------------------------------------------------------------------------------------------------

def test_relatedness_test_is_reproducible_and_paired(tmp_path):
    inputs = one_family(tmp_path)
    first = run(inputs)
    assert first == run(inputs)
    family = first.families[0]
    assert family.delta == pytest.approx(0.102) and first.level == 0.95
    assert family.panel == PANEL and family.reference == inputs.margins[PANEL].reference
    assert family.margin == inputs.margins[PANEL].margin and first.candidate == 'c'
    # The paired difference of the two arms' resamples, each on the family's own streams
    assert first == beat_by_margin('c', [expected_delta(inputs, PANEL)])
    assert first.beats is True and family.interval[0] > family.margin

def test_a_candidate_that_copies_the_reference_differs_by_zero_on_every_resample(tmp_path):
    # Paired arms share each unit draw, so a one-seed copy of the reference has Δ = 0 throughout
    inputs = one_family(tmp_path)
    reference = inputs.references[PANEL]
    copy_of = inputs._replace(candidate_seeds={PANEL: [reference]})
    verdict = run(copy_of, candidate='copy', min_seeds=1)
    assert verdict.families[0].delta == 0.0
    assert verdict.families[0].interval == (0.0, 0.0)
    assert verdict.beats is False

def test_the_candidates_name_sets_its_seed_draws_and_not_its_delta(tmp_path):
    inputs = one_family(tmp_path)
    under_c = run(inputs, candidate='c')
    under_d = run(inputs, candidate='d')
    assert under_c.families[0].delta == under_d.families[0].delta
    assert under_d == beat_by_margin('d', [expected_delta(inputs, PANEL, candidate='d')])
    assert under_c.families[0].interval != under_d.families[0].interval

def test_relatedness_test_at_two_families(tmp_path):
    inputs = two_families(tmp_path)
    verdict = run(inputs)
    assert verdict.level == pytest.approx(0.975)
    assert len(verdict.families) == 2
    assert [family.panel for family in verdict.families] == [PANEL, SECOND_FAMILY]
    assert verdict == beat_by_margin(
        'c', [expected_delta(inputs, key) for key in (PANEL, SECOND_FAMILY)]
    )
    assert verdict.beats is True and all(family.beats for family in verdict.families)

def test_each_family_is_judged_on_its_own_reference_margin_and_data(tmp_path):
    inputs = two_families(tmp_path)
    verdict = run(inputs)
    first, second = verdict.families
    assert (first.reference, second.reference) == (TREE, TEXT_ONLY)
    assert first.margin == inputs.margins[PANEL].margin
    assert second.margin == inputs.margins[SECOND_FAMILY].margin == 0.05 * 0.42
    assert first.margin != second.margin
    # Each Δ is that family's candidate minus its reference, on its own results
    for family, key in ((first, PANEL), (second, SECOND_FAMILY)):
        expected = expected_delta(inputs, key)
        assert family.delta == expected.delta and family.delta == pytest.approx(0.102)
    # The two references read differently, so the families do not share their replicates
    assert first.interval != second.interval

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
@pytest.mark.parametrize('change', ['drops an anchor', 'renames an anchor'])
def test_each_family_takes_its_own_anchors(tmp_path, key, change):
    # Anchors are the units, so a family's items are its own reference's: one family has 6 anchors
    # and the other 7, or both have 7 and the sets differ
    alter = {
        'drops an anchor': lambda result: without_anchor(result, '445110'),
        'renames an anchor': lambda result: renamed_anchor(result, '445110', '999999'),
    }[change]
    inputs = on_anchors(two_families(tmp_path), key, alter)
    first, second = (
        set(inputs.references[name].per_anchor.get_column('anchor'))
        for name in (PANEL, SECOND_FAMILY)
    )
    assert first != second
    assert sorted((len(first), len(second))) == ([6, 7] if change == 'drops an anchor' else [7, 7])
    verdict = run(inputs)
    expected = beat_by_margin(
        'c', [expected_delta(inputs, name) for name in (PANEL, SECOND_FAMILY)]
    )
    assert verdict == expected
    assert [family.panel for family in verdict.families] == [PANEL, SECOND_FAMILY]
    assert verdict.beats is True

def test_a_second_family_leaves_the_first_familys_draws_alone(tmp_path):
    inputs = two_families(tmp_path)
    one, two = run(inputs.only(PANEL)), run(inputs)
    assert one.level == 0.95 and two.level == pytest.approx(0.975)
    assert one.families[0].delta == two.families[0].delta
    # The same draws, read at the higher level: the interval widens on both ends
    spread = expected_delta(inputs, PANEL).replicates
    assert one.families[0].interval == percentile_interval(spread, 0.95)
    assert two.families[0].interval == percentile_interval(spread, two.level)
    assert two.families[0].interval[0] < one.families[0].interval[0]
    assert two.families[0].interval[1] > one.families[0].interval[1]

@pytest.mark.parametrize('order', [(PANEL, SECOND_FAMILY), (SECOND_FAMILY, PANEL)])
def test_the_verdict_does_not_depend_on_the_mappings_order(tmp_path, order):
    inputs = two_families(tmp_path)
    reordered = inputs.only(*order)
    assert list(reordered.margins) == list(order)
    verdict = run(reordered)
    assert verdict == run(inputs)
    assert [family.panel for family in verdict.families] == sorted(order)

def test_relatedness_test_beats_only_on_every_family_by_its_own_margin(tmp_path):
    inputs = two_families(tmp_path)
    margins = {key: inputs.margins[key].margin for key in inputs.margins}

    # Every seed of a family is its reference plus one amount, so Δ is that amount on every resample
    def lifted(first_amount: float, second_amount: float) -> Inputs:
        seeds = {
            key: [offset(inputs.references[key], amount)] * FLOOR
            for key, amount in ((PANEL, first_amount), (SECOND_FAMILY, second_amount))
        }
        return inputs._replace(candidate_seeds=seeds)

    big = {key: 2 * margin for key, margin in margins.items()}
    small = {key: margin / 2 for key, margin in margins.items()}
    both = run(lifted(big[PANEL], big[SECOND_FAMILY]))
    assert [family.beats for family in both.families] == [True, True] and both.beats is True
    only_first = run(lifted(big[PANEL], small[SECOND_FAMILY]))
    assert [family.beats for family in only_first.families] == [True, False]
    assert only_first.beats is False
    only_second = run(lifted(small[PANEL], big[SECOND_FAMILY]))
    assert [family.beats for family in only_second.families] == [False, True]
    assert only_second.beats is False
    # Superior to the reference on the second family, and still not by δ
    assert only_first.families[1].interval[0] > 0 and not only_first.families[1].beats

def test_a_candidate_worse_than_the_reference_has_a_negative_delta_and_does_not_beat_it(tmp_path):
    inputs = one_family(tmp_path)
    worse = inputs._replace(
        candidate_seeds={PANEL: [offset(inputs.references[PANEL], -0.1)] * FLOOR}
    )
    verdict = run(worse)
    family = verdict.families[0]
    assert family.delta == pytest.approx(-0.1)
    assert family.interval == pytest.approx((-0.1, -0.1), abs=1e-12)
    assert family.beats is False and verdict.beats is False

def test_relatedness_test_runs_on_the_sealed_part_with_the_margin_fixed_on_validation(tmp_path):
    p = panel(logged(tmp_path))
    margin = fix(p)
    p.open_sealed('the decision')
    reference = p.score(
        arms()[margin.reference], 'sealed', 'the opening', {
            'arm': margin.reference,
            'role': 'reference',
            'run': 'stage12/reference'
        }
    )
    assert reference.part == 'sealed' and reference.headline['n_anchors'] == 6
    assert reference.data_fingerprint != margin.data_fingerprint
    verdict = relatedness_test(
        'c',
        {PANEL: candidate_seeds_of(reference)},
        {PANEL: reference},
        {PANEL: margin},
        min_seeds=FLOOR,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    assert verdict.beats is True and verdict.families[0].margin == margin.margin
    assert verdict.families[0].delta == pytest.approx(0.102)

# -------------------------------------------------------------------------------------------------
# The test's refusals
# -------------------------------------------------------------------------------------------------

def test_fewer_than_min_seeds_is_refused(tmp_path):
    inputs = one_family(tmp_path)
    seeds = inputs.candidate_seeds[PANEL]
    with pytest.raises(ValueError, match='4 .*5'):
        relatedness_test(
            'c', {PANEL: seeds[:4]},
            inputs.references,
            inputs.margins,
            min_seeds=5,
            replicates=REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED
        )

def test_the_floor_names_the_family_the_count_and_the_floor(tmp_path):
    inputs = two_families(tmp_path)
    short = inputs._replace(
        candidate_seeds={
            **inputs.candidate_seeds, SECOND_FAMILY: inputs.candidate_seeds[SECOND_FAMILY][:3]
        }
    )
    with pytest.raises(ValueError, match='candidate seeds, fewer than') as raised:
        run(short)
    assert f'the family {SECOND_FAMILY!r} has 3 candidate seeds, fewer than the 5' in str(
        raised.value
    )

def test_exactly_min_seeds_passes_and_one_fewer_is_refused(tmp_path):
    inputs = one_family(tmp_path)
    assert len(run(inputs, min_seeds=5).families) == 1
    assert len(run(inputs, min_seeds=1).families) == 1
    with pytest.raises(ValueError, match='has 5 candidate seeds, fewer than the 6'):
        run(inputs, min_seeds=6)
    four = inputs._replace(candidate_seeds={PANEL: inputs.candidate_seeds[PANEL][:4]})
    assert run(four, min_seeds=4).families[0].delta == pytest.approx(0.1015)
    with pytest.raises(ValueError, match='has 4 candidate seeds, fewer than the 5'):
        run(four, min_seeds=5)

@pytest.mark.parametrize('floor', [0, -1])
def test_a_floor_below_one_is_refused(tmp_path, floor):
    with pytest.raises(ValueError, match='min_seeds must be at least 1'):
        run(one_family(tmp_path), min_seeds=floor)

@pytest.mark.parametrize(
    'empty, named',
    [
        (('candidate_seeds', ), 'candidate_seeds'),
        (('references', ), 'references'),
        (('margins', ), 'margins'),
        (('candidate_seeds', 'references', 'margins'), 'candidate_seeds, references, margins'),
    ],
)
def test_empty_mappings_are_refused(tmp_path, empty, named):
    inputs = one_family(tmp_path)
    emptied = inputs._replace(**{name: {} for name in empty})
    with pytest.raises(ValueError, match='no family is in') as raised:
        run(emptied)
    assert named in str(raised.value)

def _keyed_differently(inputs: Inputs, field: str, change: str) -> Inputs:
    mapping = getattr(inputs, field)
    if change == 'extra':
        extra = {**mapping, 'relatedness_extra_family': next(iter(mapping.values()))}
        return inputs._replace(**{field: extra})
    return inputs._replace(
        **{field: {
            key: v
            for key, v in mapping.items() if key != SECOND_FAMILY
        }}
    )

@pytest.mark.parametrize('change', ['extra', 'missing'])
@pytest.mark.parametrize('field', ['candidate_seeds', 'references', 'margins'])
def test_mappings_keyed_differently_are_refused(tmp_path, field, change):
    keyed = _keyed_differently(two_families(tmp_path), field, change)
    with pytest.raises(ValueError, match='must name the same families'):
        run(keyed)

def test_a_margin_for_another_panel_is_refused(tmp_path):
    inputs = two_families(tmp_path)
    swapped = inputs._replace(
        margins={
            PANEL: inputs.margins[SECOND_FAMILY],
            SECOND_FAMILY: inputs.margins[PANEL]
        }
    )
    with pytest.raises(ValueError, match='is for panel') as raised:
        run(swapped)
    assert repr(SECOND_FAMILY) in str(raised.value) and repr(PANEL) in str(raised.value)
    assert inputs.margins[SECOND_FAMILY].name in str(raised.value)

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
def test_a_margin_that_disagrees_with_itself_is_refused(tmp_path, key):
    inputs = two_families(tmp_path)
    broken = inputs.margins[key].model_copy(update={'margin': 1.0})
    with pytest.raises(ValueError, match='holds δ') as raised:
        run(inputs._replace(margins={**inputs.margins, key: broken}))
    assert inputs.margins[key].name in str(raised.value)

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
@pytest.mark.parametrize('position', [0, 2, 4])
def test_a_candidate_seed_of_another_panel_is_refused(tmp_path, key, position):
    broken = with_seed(two_families(tmp_path), key, position, panel='relatedness_other')
    with pytest.raises(ValueError, match='was read on panel') as raised:
        run(broken)
    assert f'candidate seed {position}' in str(raised.value)
    assert repr(key) in str(raised.value) and repr('relatedness_other') in str(raised.value)

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
def test_a_reference_result_of_another_panel_is_refused(tmp_path, key):
    broken = with_reference(two_families(tmp_path), key, panel='relatedness_other')
    with pytest.raises(ValueError, match='was read on panel') as raised:
        run(broken)
    assert 'the reference' in str(raised.value) and repr(key) in str(raised.value)

def test_a_result_read_on_the_other_familys_panel_is_refused(tmp_path):
    inputs = two_families(tmp_path)
    with pytest.raises(ValueError, match='was read on panel'):
        run(with_seed(inputs, SECOND_FAMILY, 1, panel=PANEL))

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
@pytest.mark.parametrize('position', [0, 1, 4])
def test_a_candidate_seed_of_another_part_is_refused(tmp_path, key, position):
    broken = with_seed(two_families(tmp_path), key, position, part='sealed')
    with pytest.raises(ValueError, match='more than one part') as raised:
        run(broken)
    assert 'sealed' in str(raised.value) and 'validation' in str(raised.value)

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
def test_a_reference_of_another_part_is_refused(tmp_path, key):
    with pytest.raises(ValueError, match='more than one part'):
        run(with_reference(two_families(tmp_path), key, part='sealed'))

def on_part(inputs: Inputs, part: str, *keys: str) -> Inputs:
    '''``inputs`` with every result of the given families read on ``part``.'''

    seeds, references = dict(inputs.candidate_seeds), dict(inputs.references)
    for key in keys:
        seeds[key] = [dataclasses.replace(seed, part=part) for seed in seeds[key]]
        references[key] = dataclasses.replace(references[key], part=part)
    return inputs._replace(candidate_seeds=seeds, references=references)

def test_a_second_family_on_another_part_than_the_first_is_refused(tmp_path):
    # Each family is one part in itself: the parts differ across the families
    inputs = two_families(tmp_path)
    with pytest.raises(ValueError, match='more than one part'):
        run(on_part(inputs, 'sealed', SECOND_FAMILY))
    with pytest.raises(ValueError, match='more than one part'):
        run(on_part(inputs, 'sealed', PANEL))
    # Every result on the sealed part is one part, so the test runs
    assert len(run(on_part(inputs, 'sealed', PANEL, SECOND_FAMILY)).families) == 2

def test_seeds_read_on_another_build_or_part_are_refused(tmp_path):
    inputs = one_family(tmp_path)
    with pytest.raises(ValueError, match='data fingerprint'):
        run(with_seed(inputs, PANEL, 2, data_fingerprint=OTHER_FINGERPRINT))
    with pytest.raises(ValueError, match='more than one part'):
        run(with_seed(inputs, PANEL, 2, part='sealed'))

@pytest.mark.parametrize('key', [PANEL, SECOND_FAMILY])
@pytest.mark.parametrize('position', [0, 2, 4])
def test_a_seed_read_on_another_build_is_refused_naming_the_family_and_the_seed(
    tmp_path, key, position
):
    inputs = two_families(tmp_path)
    broken = with_seed(inputs, key, position, data_fingerprint=OTHER_FINGERPRINT)
    with pytest.raises(ValueError, match='data fingerprint') as raised:
        run(broken)
    message = str(raised.value)
    assert f'the family {key!r} candidate seed {position}' in message
    assert OTHER_FINGERPRINT in message and inputs.references[key].data_fingerprint in message

def test_a_seed_is_compared_with_its_own_familys_reference(tmp_path):
    # The two families are on different builds: a seed on the first family's build is another build
    inputs = two_families(tmp_path)
    first_build = inputs.references[PANEL].data_fingerprint
    assert first_build != SECOND_FINGERPRINT
    with pytest.raises(ValueError, match='data fingerprint'):
        run(with_seed(inputs, SECOND_FAMILY, 3, data_fingerprint=first_build))

def test_a_reference_on_another_build_than_its_seeds_is_refused(tmp_path):
    inputs = one_family(tmp_path)
    broken = with_reference(inputs, PANEL, data_fingerprint=OTHER_FINGERPRINT)
    with pytest.raises(ValueError, match='data fingerprint') as raised:
        run(broken)
    assert 'candidate seed 0' in str(raised.value)

def test_seeds_scored_on_other_anchors_are_refused(tmp_path):
    inputs = one_family(tmp_path)
    for position in (0, 3):
        per_anchor = inputs.candidate_seeds[PANEL][position].per_anchor
        for other in (
            per_anchor.filter(pl.col('anchor') != '445110'),
            per_anchor.with_columns(pl.col('anchor').str.replace('445110', '999999')),
        ):
            with pytest.raises(ValueError, match='items'):
                run(with_seed(inputs, PANEL, position, per_anchor=other))

# One breaking of a valid pair of families for each refusal, in the order the brief lists them, and
# the words only that refusal's message uses. The first family's seeds 1, 2 and 3 take the part, the
# panel and the fingerprint, so the floor's truncation to four seeds leaves each of them
def _break_keys(inputs: Inputs) -> Inputs:
    return inputs._replace(
        references={
            key: v
            for key, v in inputs.references.items() if key != SECOND_FAMILY
        }
    )

def _break_margin(inputs: Inputs) -> Inputs:
    return inputs._replace(margins={**inputs.margins, PANEL: inputs.margins[SECOND_FAMILY]})

def _break_margin_check(inputs: Inputs) -> Inputs:
    broken = inputs.margins[SECOND_FAMILY].model_copy(update={'margin': 1.0})
    return inputs._replace(margins={**inputs.margins, SECOND_FAMILY: broken})

def _break_floor(inputs: Inputs) -> Inputs:
    return inputs._replace(
        candidate_seeds={
            **inputs.candidate_seeds, PANEL: inputs.candidate_seeds[PANEL][:4]
        }
    )

def _break_panel(inputs: Inputs) -> Inputs:
    return with_seed(inputs, PANEL, 2, panel='relatedness_other')

def _break_part(inputs: Inputs) -> Inputs:
    return with_seed(inputs, PANEL, 1, part='sealed')

def _break_fingerprint(inputs: Inputs) -> Inputs:
    return with_seed(inputs, PANEL, 3, data_fingerprint=OTHER_FINGERPRINT)

BREAKS = {
    'keys': (_break_keys, 'must name the same families'),
    'margin': (_break_margin, 'is for panel'),
    'margin check': (_break_margin_check, 'holds δ'),
    'floor': (_break_floor, 'candidate seeds, fewer than'),
    'panel': (_break_panel, 'was read on panel'),
    'part': (_break_part, 'more than one part'),
    'fingerprint': (_break_fingerprint, 'data fingerprint'),
}
BREAK_PAIRS = [(a, b) for index, a in enumerate(BREAKS) for b in list(BREAKS)[index + 1:]]

@pytest.mark.parametrize('earlier, later', BREAK_PAIRS)
def test_the_refusals_come_in_the_briefs_order(tmp_path, earlier, later):
    inputs = two_families(tmp_path)
    for kind in (later, earlier):
        inputs = BREAKS[kind][0](inputs)
    with pytest.raises(ValueError, match=BREAKS[earlier][1]):
        run(inputs)

@pytest.mark.parametrize('kind', list(BREAKS))
def test_each_refusal_says_its_own_words_only(tmp_path, kind):
    broken = BREAKS[kind][0](two_families(tmp_path))
    with pytest.raises(ValueError) as raised:
        run(broken)
    for other, (_, words) in BREAKS.items():
        assert (re.search(words, str(raised.value)) is not None) == (other == kind)

@pytest.mark.parametrize('kind', list(BREAKS))
def test_nothing_is_drawn_before_a_refusal(tmp_path, monkeypatch, kind):

    def draws(*args, **kwargs):
        raise AssertionError('a draw ran before the refusal')

    monkeypatch.setattr(relatedness, 'unit_draws', draws)
    monkeypatch.setattr(relatedness, 'seed_draws', draws)
    with pytest.raises(ValueError):
        run(BREAKS[kind][0](two_families(tmp_path)))

# -------------------------------------------------------------------------------------------------
# The provenance check: helpers
# -------------------------------------------------------------------------------------------------

RULE = '(relatedness §4.6: codebook, text-only table and training families only)'
SHAS = {'codebook_sha256': 'c' * 64, 'text_only_sha256': 'd' * 64}
OTHER_SHA = 'e' * 64
AREA_MIX_RUN = 'stage6_ita_area_mix_20261014_1000_csv'
REF = {'path': 'x', 'sha256': 'y'}

# A provenance as the export writes it, with the area-mix input present: the codebook, the text-only
# table and the area mix, each a path and a sha256, and the area mix's run
VALID: Dict[str, Any] = {
    'writer': RELATEDNESS_EXPORT,
    'codes': 2125,
    'dimension': 16,
    'inputs': {
        'codebook': {
            'path': 'data/codebook.parquet',
            'sha256': SHAS['codebook_sha256']
        },
        'text_only': {
            'path': 'checkpoints/plan9_exit/text_only.parquet',
            'sha256': SHAS['text_only_sha256']
        },
        'ita_area_mix': {
            'path': 'data/ita_area_mix.parquet',
            'sha256': 'a' * 64
        },
    },
    'runs': {
        'ita_area_mix': AREA_MIX_RUN
    },
    'configuration': {
        'epochs': 20,
        'learning_rate': 0.001
    },
    'seed': 3,
    'checkpoint': {
        'path': 'checkpoints/relatedness/seed3.pt',
        'sha256': 'b' * 64
    },
    'table_sha256': 'f' * 64,
    'generated_at': '2026-11-21T00:00:00+00:00',
    'library_versions': {
        'torch': '2.9.1'
    },
}

# The entries and the input names of the brief, written out so that the cases below do not read the
# constants under test
BRIEF_ENTRIES = (
    'checkpoint', 'codes', 'configuration', 'dimension', 'generated_at', 'inputs',
    'library_versions', 'runs', 'seed', 'table_sha256', 'writer'
)
BRIEF_INPUTS = ('codebook', 'text_only', 'ita_area_mix')

# The eight checks, in the brief's order
CHECKS = ('writer', 'entries', 'inputs', 'codebook', 'text_only', 'runs', 'checkpoint', 'seed')

def valid() -> Dict[str, Any]:
    '''A copy of ``VALID`` that a test can change.'''

    return copy.deepcopy(VALID)

def with_input(name: str, ref: Optional[Any] = None) -> Dict[str, Any]:
    '''``VALID`` with one more input, ``name``, which is a path and a sha256 unless ``ref`` is given.'''

    content = valid()
    content['inputs'][name] = {
        'path': f'data/{name}.parquet',
        'sha256': OTHER_SHA
    } if ref is None else ref
    return content

def without_area_mix() -> Dict[str, Any]:
    '''``VALID`` with no area-mix input and so no run.'''

    content = valid()
    del content['inputs']['ita_area_mix']
    content['runs'] = {}
    return content

def edited(path: Sequence[Any], value: Any,
           content: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '''``VALID``, or ``content``, with the value at ``path`` replaced by ``value``.'''

    content = valid() if content is None else content
    holder = content
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return content

def dropped(path: Sequence[Any], content: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '''``VALID``, or ``content``, with the entry at ``path`` removed.'''

    content = valid() if content is None else content
    holder = content
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    return content

def refusal(content: Any, **shas: str) -> str:
    '''The message of the refusal that ``check_relatedness_provenance`` raises for ``content``.'''

    with pytest.raises(ValueError) as raised:
        check_relatedness_provenance(content, **{**SHAS, **shas})
    return str(raised.value)

class Case(NamedTuple):
    '''A provenance that one check refuses, and the words that its message holds.'''

    id: str
    content: Any
    check: str
    words: str

# -------------------------------------------------------------------------------------------------
# The provenance check
# -------------------------------------------------------------------------------------------------

def test_the_constants_are_the_briefs():
    assert RELATEDNESS_EXPORT == 'relatedness_export'
    assert PROVENANCE_ENTRIES == frozenset(
        {
            'writer', 'codes', 'dimension', 'inputs', 'runs', 'configuration', 'seed', 'checkpoint',
            'table_sha256', 'generated_at', 'library_versions'
        }
    )
    assert CODEBOOK_INPUTS == ('codebook', 'text_only')
    assert TRAINING_FAMILY_INPUTS == ('ita_area_mix', )

def test_the_valid_provenance_is_the_exports_entries():
    assert set(VALID) == PROVENANCE_ENTRIES

def test_a_provenance_that_names_co_ownership_is_refused():
    with pytest.raises(ValueError, match='inputs.*ita_co_ownership'):
        check_relatedness_provenance(with_input('ita_co_ownership'), **SHAS)
    with pytest.raises(ValueError, match='entries.*co_ownership'):
        check_relatedness_provenance({**VALID, 'co_ownership': {}}, **SHAS)

def test_a_valid_provenance_passes_with_and_without_the_area_mix():
    assert check_relatedness_provenance(VALID, **SHAS) is None
    assert check_relatedness_provenance(without_area_mix(), **SHAS) is None

PROVENANCE_CASES = [
    Case(
        'writer-export', {
            **VALID, 'writer': 'export'
        }, 'writer', "the writer is 'export', not 'relatedness_export'"
    ),
    Case(
        'missing-seed', dropped(['seed']), 'entries', r"no entry \['seed'\], which the writer writes"
    ),
    Case(
        'inputs-without-text-only', dropped(['inputs', 'text_only']), 'inputs',
        r"no input \['text_only'\]"
    ),
    Case(
        'input-with-a-run-key', edited(['inputs', 'codebook', 'run'], AREA_MIX_RUN), 'inputs',
        "the input 'codebook' is not exactly a string 'path' and a string 'sha256'"
    ),
    Case(
        'another-codebook-sha256', edited(['inputs', 'codebook', 'sha256'], OTHER_SHA), 'codebook',
        'the codebook input has sha256'
    ),
    Case(
        'another-text-only-sha256', edited(['inputs', 'text_only', 'sha256'], OTHER_SHA),
        'text_only', 'the text_only input has sha256'
    ),
    Case(
        'no-runs-with-the-area-mix', edited(['runs'], {}), 'runs',
        r"keyed by \[\], but the training-family inputs present are \['ita_area_mix'\]"
    ),
    Case(
        'runs-naming-an-absent-area-mix', edited(['inputs'],
                                                 without_area_mix()['inputs']), 'runs',
        r"keyed by \['ita_area_mix'\], but the training-family inputs present are \[\]"
    ),
    Case(
        'checkpoint-without-sha256', dropped(['checkpoint', 'sha256']), 'checkpoint',
        "the checkpoint is not exactly a string 'path' and a string 'sha256'"
    ),
    Case('seed-true', edited(['seed'], True), 'seed', 'the seed is True, not an int'),
]

def assert_refused_by(case: Case) -> None:
    '''The check refuses ``case.content`` by ``case.check`` alone, with ``case.words`` and the tag.'''

    message = refusal(case.content)
    assert message.startswith(f'{case.check}: ')
    assert re.search(case.words, message), message
    assert message.endswith(f' {RULE}')
    assert message.count(RULE) == 1

@pytest.mark.parametrize('case', PROVENANCE_CASES, ids=[case.id for case in PROVENANCE_CASES])
def test_each_rule_refuses(case):
    assert_refused_by(case)

def _quoted(value: Any) -> str:
    '''``value`` as a regular expression that matches its ``repr``.'''

    return re.escape(repr(value))

def _file_shapes(good_sha256: str) -> Dict[str, Any]:
    '''What a file reference must not be: each a wrong form of a path and a sha256.'''

    return {
        'an-extra-key': {
            'path': 'x',
            'sha256': good_sha256,
            'run': AREA_MIX_RUN
        },
        'no-path': {
            'sha256': good_sha256
        },
        'no-sha256': {
            'path': 'x'
        },
        'a-renamed-key': {
            'file': 'x',
            'sha256': good_sha256
        },
        'empty': {},
        'a-path-that-is-an-int': {
            'path': 3,
            'sha256': good_sha256
        },
        'a-path-that-is-none': {
            'path': None,
            'sha256': good_sha256
        },
        'a-path-that-is-bytes': {
            'path': b'x',
            'sha256': good_sha256
        },
        'a-sha256-that-is-an-int': {
            'path': 'x',
            'sha256': 3
        },
        'a-sha256-that-is-none': {
            'path': 'x',
            'sha256': None
        },
        'a-sha256-that-is-bytes': {
            'path': 'x',
            'sha256': b'x'
        },
        'the-two-key-names-in-a-list': ['path', 'sha256'],
        'none': None,
        'a-string': 'data/x.parquet',
    }

def _further_cases() -> List[Case]:
    '''
    The shapes of each check's rule beyond the plan's ten: boundaries, empties, wrong types and
    values that look like the right one.
    '''

    cases = []

    def add(case_id: str, content: Any, check: str, words: str) -> None:
        cases.append(Case(f'{check}-{case_id}', content, check, words))

    # writer
    for position, value in enumerate([[], None, 'relatedness_export', 3, ['writer']]):
        add(f'content-{position}-{type(value).__name__}', value, 'writer', 'is not a mapping')
    add('no-writer-entry', dropped(['writer']), 'writer', "has no 'writer' entry")
    add('no-writer-and-nothing-else', {}, 'writer', "has no 'writer' entry")
    for position, writer in enumerate(
        [
            'Relatedness_Export', ' relatedness_export', 'relatedness_export\n',
            'relatedness_export_v2', 'x_relatedness_export', 'relatedness', '', None, 3,
            ['relatedness_export'], 'tree_builder',
            Path('relatedness_export')
        ]
    ):
        add(f'writer-{position}', edited(['writer'], writer), 'writer', 'the writer is')
    # entries
    for entry in [name for name in BRIEF_ENTRIES if name != 'writer']:
        add(f'missing-{entry}', dropped([entry]), 'entries', rf"no entry \['{entry}'\]")
    add(
        'missing-everything-but-the-writer', {'writer': RELATEDNESS_EXPORT}, 'entries',
        re.escape(f"no entry {[name for name in BRIEF_ENTRIES if name != 'writer']}")
    )
    add(
        'missing-two-sorted', dropped(['seed'], dropped(['checkpoint'])), 'entries',
        r"no entry \['checkpoint', 'seed'\]"
    )
    for name in ['co_ownership', 'ita_co_ownership', 'selection_log', 'qcew', 'builder', 'Seed']:
        add(
            f'unknown-{name}', {
                **VALID, name: {}
            }, 'entries', rf"the entries \['{name}'\], which the writer does not write"
        )
    add(
        'unknown-two-sorted', {
            **VALID, 'qcew': 1,
            'co_ownership': 2
        }, 'entries', r"the entries \['co_ownership', 'qcew'\], which"
    )
    add(
        'unknown-and-missing', {
            **dropped(['seed']), 'co_ownership': {}
        }, 'entries', r"the entries \['co_ownership'\], which the writer does not write and has "
        r"no entry \['seed'\], which the writer writes"
    )
    add(
        'unknown-keys-of-two-types', {
            **VALID, 1: 'x',
            'co_ownership': {}
        }, 'entries', r"\['1', 'co_ownership'\]"
    )
    # inputs
    for position, value in enumerate(
        [None, [], ['codebook', 'text_only'], 'codebook', 3, ('codebook', 'text_only')]
    ):
        add(f'not-a-mapping-{position}', edited(['inputs'], value), 'inputs', 'is not a mapping')
    for name in [
        'ita_co_ownership', 'co_ownership_marginals', 'co_ownership', 'qcew', 'selection_log',
        'ITA_AREA_MIX', 'ita_area_mix_2023', 'Codebook', ''
    ]:
        add(
            f'outside-name-{name or "empty"}', with_input(name), 'inputs',
            rf"the inputs \['{name}'\], which are outside"
        )
    add(
        'outside-name-beside-no-area-mix',
        edited(['runs'], {}, dropped(['inputs', 'ita_area_mix'], with_input('qcew'))), 'inputs',
        r"the inputs \['qcew'\]"
    )
    add(
        'outside-names-sorted', edited(['inputs', 'qcew'], REF, with_input('co_ownership')),
        'inputs', r"the inputs \['co_ownership', 'qcew'\], which"
    )
    add(
        'outside-name-before-a-malformed-file', with_input('ita_co_ownership', 'a string'),
        'inputs', r"the inputs \['ita_co_ownership'\], which"
    )
    add(
        'outside-name-before-a-malformed-codebook',
        edited(['inputs', 'codebook'], None, with_input('ita_co_ownership')), 'inputs',
        r"the inputs \['ita_co_ownership'\], which"
    )
    add(
        'missing-input-before-a-malformed-file',
        edited(['inputs', 'codebook'], None, dropped(['inputs', 'text_only'])), 'inputs',
        r"no input \['text_only'\]"
    )
    add('a-name-that-is-an-int', edited(['inputs', 7], REF), 'inputs', r"the inputs \['7'\], which")
    add('no-codebook', dropped(['inputs', 'codebook']), 'inputs', r"no input \['codebook'\]")
    add(
        'no-codebook-and-no-text-only',
        edited(['inputs'], {'ita_area_mix': VALID['inputs']['ita_area_mix']}), 'inputs',
        r"no input \['codebook', 'text_only'\]"
    )
    add(
        'no-inputs-at-all', edited(['inputs'], {}), 'inputs', r"no input \['codebook', 'text_only'\]"
    )
    add(
        'outside-and-missing', dropped(['inputs', 'text_only'], with_input('qcew')), 'inputs',
        r"the inputs \['qcew'\], which are outside \['codebook', 'text_only', "
        r"'ita_area_mix'\] and names no input \['text_only'\], which the table must read"
    )
    for name in BRIEF_INPUTS:
        for shape, value in _file_shapes('a' * 64).items():
            add(
                f'{name}-is-{shape}', edited(['inputs', name], value), 'inputs',
                f"the input '{name}' is not exactly a string 'path' and a string 'sha256'"
            )
    # codebook and text_only
    codebook = SHAS['codebook_sha256']
    text_only = SHAS['text_only_sha256']
    for sha_id, sha in {
        'another': OTHER_SHA,
        'a-prefix': codebook[:-1],
        'a-longer-one': codebook + '0',
        'upper-case': codebook.upper(),
        'padded': f'{codebook} ',
        'empty': '',
        'the-other-inputs': text_only
    }.items():
        add(
            f'its-sha256-is-{sha_id}', edited(['inputs', 'codebook', 'sha256'], sha), 'codebook',
            rf'the codebook input has sha256 {_quoted(sha)}, not the {_quoted(codebook)}'
        )
    for sha_id, sha in {
        'another': OTHER_SHA,
        'a-prefix': text_only[:-1],
        'a-longer-one': text_only + '0',
        'upper-case': text_only.upper(),
        'padded': f'{text_only} ',
        'empty': '',
        'the-other-inputs': codebook
    }.items():
        add(
            f'its-sha256-is-{sha_id}', edited(['inputs', 'text_only', 'sha256'], sha), 'text_only',
            rf'the text_only input has sha256 {_quoted(sha)}, not the {_quoted(text_only)}'
        )
    # runs
    for position, value in enumerate([None, [], [AREA_MIX_RUN], AREA_MIX_RUN, 3]):
        add(f'not-a-mapping-{position}', edited(['runs'], value), 'runs', 'is not a mapping')
    add(
        'other-name-with-a-blank-run', edited(['runs'], {'ita_co_ownership': ' '}), 'runs',
        r"keyed by \['ita_co_ownership'\], but"
    )
    for run_id, runs in {
        'another-name': {
            'ita_co_ownership': AREA_MIX_RUN
        },
        'the-area-mix-and-co-ownership': {
            'ita_area_mix': AREA_MIX_RUN,
            'ita_co_ownership': 'x'
        },
        'the-area-mix-and-a-codebook-input': {
            'ita_area_mix': AREA_MIX_RUN,
            'codebook': 'x'
        },
    }.items():
        add(
            run_id, edited(['runs'], runs), 'runs',
            r"but the training-family inputs present are \['ita_area_mix'\]"
        )
    for run_id, runs in {
        'a-codebook-input': {
            'codebook': AREA_MIX_RUN
        },
        'the-area-mix-and-another': {
            'ita_area_mix': AREA_MIX_RUN,
            'x': 'y'
        },
    }.items():
        add(
            f'no-area-mix-with-{run_id}', edited(['runs'], runs, without_area_mix()), 'runs',
            r'but the training-family inputs present are \[\]'
        )
    for position, blank in enumerate(['', ' ', '\t\n', ' ', '  ']):
        add(
            f'blank-run-{position}', edited(['runs', 'ita_area_mix'], blank), 'runs',
            "the run of 'ita_area_mix' is not a non-blank string"
        )
    for position, value in enumerate(
        [None, 3, 0, False, [AREA_MIX_RUN], {
            'path': AREA_MIX_RUN
        }, b'run']
    ):
        add(
            f'non-string-run-{position}', edited(['runs', 'ita_area_mix'], value), 'runs',
            "the run of 'ita_area_mix' is not a non-blank string"
        )
    # checkpoint
    for shape, value in _file_shapes('b' * 64).items():
        add(
            f'is-{shape}', edited(['checkpoint'], value), 'checkpoint',
            "the checkpoint is not exactly a string 'path' and a string 'sha256'"
        )
    # seed
    for value in [True, False, 3.0, 0.0, -1.5, '3', None, [3], 3 + 0j, b'3', {3}]:
        add(
            f'seed-{type(value).__name__}-{value!r}', edited(['seed'], value), 'seed',
            rf'the seed is {_quoted(value)}, not an int'
        )
    return cases

FURTHER_CASES = _further_cases()

@pytest.mark.parametrize('case', FURTHER_CASES, ids=[case.id for case in FURTHER_CASES])
def test_each_further_shape_of_each_rule_refuses(case):
    assert_refused_by(case)

def _frozen(value: Any) -> Any:
    '''``value`` with every dictionary replaced by a read-only mapping that is not a ``dict``.'''

    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    return value

@pytest.mark.parametrize('seed', [0, 1, -1, -2**31, 2**63, 20261121])
def test_any_int_seed_passes(seed):
    assert check_relatedness_provenance(edited(['seed'], seed), **SHAS) is None
    assert check_relatedness_provenance(edited(['seed'], seed, without_area_mix()), **SHAS) is None

@pytest.mark.parametrize('run', ['x', ' x ', 'a b', 'stage6_ita_area_mix_20261014_1000_csv', '0'])
def test_any_run_with_a_character_other_than_whitespace_passes(run):
    assert check_relatedness_provenance(edited(['runs', 'ita_area_mix'], run), **SHAS) is None

def test_a_mapping_that_is_not_a_dict_passes_at_every_level():
    for content in (valid(), without_area_mix()):
        assert check_relatedness_provenance(_frozen(content), **SHAS) is None

def test_the_arguments_are_what_the_inputs_are_compared_with():
    content = valid()
    content['inputs']['codebook']['sha256'] = OTHER_SHA
    content['inputs']['text_only']['sha256'] = 'a' * 64
    assert check_relatedness_provenance(
        content, codebook_sha256=OTHER_SHA, text_only_sha256='a' * 64
    ) is None
    swapped = {
        'codebook_sha256': SHAS['text_only_sha256'],
        'text_only_sha256': SHAS['codebook_sha256']
    }
    assert refusal(VALID, **swapped).startswith('codebook: ')
    assert refusal(VALID, text_only_sha256=OTHER_SHA).startswith('text_only: ')
    assert refusal(VALID, codebook_sha256=OTHER_SHA).startswith('codebook: ')

def test_a_refusal_names_the_sha256_it_found_and_the_one_it_expected():
    found = valid()
    found['inputs']['codebook']['sha256'] = OTHER_SHA
    message = refusal(found)
    assert OTHER_SHA in message and SHAS['codebook_sha256'] in message
    assert SHAS['text_only_sha256'] not in message
    found = valid()
    found['inputs']['text_only']['sha256'] = OTHER_SHA
    message = refusal(found)
    assert OTHER_SHA in message and SHAS['text_only_sha256'] in message
    assert SHAS['codebook_sha256'] not in message

@pytest.mark.parametrize('run', ['', '  ', '\t\n', 3, None, b'run', ['x']])
def test_a_refused_run_is_shown_in_the_message(run):
    message = refusal(edited(['runs', 'ita_area_mix'], run))
    assert f': {run!r} {RULE}' in message

def test_the_shas_are_keyword_only():
    with pytest.raises(TypeError):
        check_relatedness_provenance(VALID, SHAS['codebook_sha256'], SHAS['text_only_sha256'])
    with pytest.raises(TypeError):
        check_relatedness_provenance(VALID)

# One breaking of a valid provenance for each check, which changes only what its own check reads
def _break_provenance_writer(content: Dict[str, Any]) -> None:
    content['writer'] = 'export'

def _break_provenance_entries(content: Dict[str, Any]) -> None:
    content['co_ownership'] = {}

def _break_provenance_inputs(content: Dict[str, Any]) -> None:
    content['inputs']['ita_co_ownership'] = REF

def _break_provenance_codebook(content: Dict[str, Any]) -> None:
    content['inputs']['codebook']['sha256'] = OTHER_SHA

def _break_provenance_text_only(content: Dict[str, Any]) -> None:
    content['inputs']['text_only']['sha256'] = OTHER_SHA

def _break_provenance_runs(content: Dict[str, Any]) -> None:
    content['runs'] = {}

def _break_provenance_checkpoint(content: Dict[str, Any]) -> None:
    del content['checkpoint']['sha256']

def _break_provenance_seed(content: Dict[str, Any]) -> None:
    content['seed'] = True

PROVENANCE_BREAKS = {
    'writer': _break_provenance_writer,
    'entries': _break_provenance_entries,
    'inputs': _break_provenance_inputs,
    'codebook': _break_provenance_codebook,
    'text_only': _break_provenance_text_only,
    'runs': _break_provenance_runs,
    'checkpoint': _break_provenance_checkpoint,
    'seed': _break_provenance_seed,
}
PROVENANCE_BREAK_PAIRS = [(a, b) for index, a in enumerate(CHECKS) for b in CHECKS[index + 1:]]

def broken_provenance(*kinds: str) -> Dict[str, Any]:
    '''``VALID`` broken in each of the given ways.'''

    content = valid()
    for kind in kinds:
        PROVENANCE_BREAKS[kind](content)
    return content

@pytest.mark.parametrize('kind', CHECKS)
def test_each_break_is_refused_by_its_own_check_alone(kind):
    message = refusal(broken_provenance(kind))
    assert message.startswith(f'{kind}: ') and message.endswith(f' {RULE}')
    assert broken_provenance(kind) != VALID

@pytest.mark.parametrize('earlier, later', PROVENANCE_BREAK_PAIRS)
def test_the_checks_run_in_the_briefs_order(earlier, later):
    for kinds in ((earlier, later), (later, earlier)):
        assert refusal(broken_provenance(*kinds)).startswith(f'{earlier}: ')

def test_every_check_refuses_with_the_earlier_ones_passed_and_the_later_ones_broken():
    for position, kind in enumerate(CHECKS):
        assert refusal(broken_provenance(*CHECKS[position:])).startswith(f'{kind}: ')

def test_the_rule_describes_the_provenance_and_changes_nothing():
    padded = edited(['runs', 'ita_area_mix'], ' x ')
    for content in (
        valid(), without_area_mix(), padded, broken_provenance('writer'),
        broken_provenance('seed', 'runs'), broken_provenance('seed')
    ):
        before = copy.deepcopy(content)
        try:
            check_relatedness_provenance(content, **SHAS)
        except ValueError:
            pass
        assert content == before
