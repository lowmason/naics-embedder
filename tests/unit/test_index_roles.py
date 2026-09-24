'''
Index-entry roles (Req 3; roadmap D4): quotas, eligibility, assignment and the frozen table.

Expected counts are worked out by hand from the allocation rule, never from production code.
'''

from fractions import Fraction

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.index_roles import (
    RoleFractions,
    allocate_role_counts,
    assign_index_roles,
    attach_role_text,
    held_out_eligibility,
    read_role_table,
    role_table_fingerprint,
    verify_examples_channel,
    verify_role_leakage,
    write_role_table,
)
from naics_embedder.supervision.artifacts import sha256_file, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole

pytestmark = pytest.mark.unit

E, TR, V, TE = IndexRole.EXAMPLES, IndexRole.TRAINING, IndexRole.VALIDATION, IndexRole.TEST
STAGE2 = RoleFractions.from_mapping(
    {
        'examples': 0.30,
        'training': 0.35,
        'validation': 0.20,
        'test': 0.15
    }
)
QUARTERS = RoleFractions.from_mapping(
    {
        'examples': 0.25,
        'training': 0.25,
        'validation': 0.25,
        'test': 0.25
    }
)
NO_TIES = (0.1, 0.2, 0.3, 0.4)

# -------------------------------------------------------------------------------------------------
# Fractions and quotas
# -------------------------------------------------------------------------------------------------

def test_fractions_are_exact_decimals():
    assert STAGE2.of(E) == Fraction(3, 10)
    assert STAGE2.of(TR) == Fraction(7, 20)
    assert STAGE2.of(V) == Fraction(1, 5)
    assert STAGE2.of(TE) == Fraction(3, 20)

@pytest.mark.parametrize(
    'mapping',
    [
        {
            'examples': 0.30,
            'training': 0.35,
            'validation': 0.20,
            'test': 0.14
        },
        {
            'examples': 0.60,
            'training': 0.45,
            'validation': 0.10,
            'test': -0.15
        },
    ],
)
def test_fractions_must_be_non_negative_and_sum_to_one(mapping):
    with pytest.raises(ValueError, match='sum to 1'):
        RoleFractions.from_mapping(mapping)

@pytest.mark.parametrize(
    ('n_entries', 'n_eligible', 'expected'),
    [
        # 0.30/0.35/0.20/0.15 of 1: training wins the remainder, then the floor moves it
        (1, 1, {
            E: 1,
            TR: 0,
            V: 0,
            TE: 0
        }),
        # of 2: remainders 0.7 (training) and 0.6 (examples) win
        (2, 2, {
            E: 1,
            TR: 1,
            V: 0,
            TE: 0
        }),
        # of 3: training floors to 1; examples (0.9) and validation (0.6) win the remainders
        (3, 3, {
            E: 1,
            TR: 1,
            V: 1,
            TE: 0
        }),
        # of 4: examples and training floor to 1; validation (0.8) and test (0.6) win
        (4, 4, {
            E: 1,
            TR: 1,
            V: 1,
            TE: 1
        }),
        (20, 20, {
            E: 6,
            TR: 7,
            V: 4,
            TE: 3
        }),
        # 7 held out but 5 eligible: validation (larger) gives one, then test (tie) gives one
        (20, 5, {
            E: 6,
            TR: 9,
            V: 3,
            TE: 2
        }),
        # nothing eligible: every held-out quota moves to training
        (4, 0, {
            E: 1,
            TR: 3,
            V: 0,
            TE: 0
        }),
    ],
)
def test_quotas_follow_largest_remainder_floor_and_eligibility(n_entries, n_eligible, expected):
    assert allocate_role_counts(n_entries, n_eligible, STAGE2, NO_TIES) == expected

@pytest.mark.parametrize(
    ('tie_break', 'winner'),
    [((0.5, 0.4, 0.3, 0.1), TE), ((0.5, 0.4, 0.1, 0.3), V)],
)
def test_remainder_ties_go_to_the_smallest_draw(tie_break, winner):
    counts = allocate_role_counts(1, 1, QUARTERS, tie_break, examples_floor=0)

    assert counts == {role: int(role == winner) for role in (E, TR, V, TE)}

def test_floor_takes_from_test_before_validation_when_training_is_empty():
    # Quarters of 2 with validation and test drawing lowest: V1 TE1, then the floor needs one
    counts = allocate_role_counts(2, 2, QUARTERS, (0.9, 0.8, 0.1, 0.2))

    assert counts == {E: 1, TR: 0, V: 1, TE: 0}

def test_quotas_reject_inconsistent_inputs():
    with pytest.raises(ValueError, match='eligible count'):
        allocate_role_counts(2, 3, STAGE2, NO_TIES)
    with pytest.raises(ValueError, match='one draw per role'):
        allocate_role_counts(2, 2, STAGE2, (0.1, 0.2))

# -------------------------------------------------------------------------------------------------
# Eligibility
# -------------------------------------------------------------------------------------------------

def _entries(rows):
    return pl.DataFrame(
        rows,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8
        },
        orient='row',
    )

def test_eligibility_withholds_exact_and_near_duplicate_matches():
    entries = _entries(
        [
            (0, '111110', 'Soybean farming'),  # equals a static title
            (1, '111110', 'Soybeans, organic'),
            (2, '111120', 'Oilseed farming'),  # reordered twin of entry 3
            (3, '111120', 'Farming, oilseed'),
            (4, '111120', 'Growing corn for grain'),  # contains a static phrase: fine
            (5, '111130', 'Rye farming'),  # occurs inside entry 6
            (6, '111130', 'Rye farming, organic'),
        ]
    )

    report = held_out_eligibility(entries, ['soybean farming', 'growing corn'])

    assert report.eligible.tolist() == [False, True, False, False, True, False, True]
    assert report.counts == {
        'entries': 7,
        'exact_static': 1,
        'near_duplicate_static': 1,
        'exact_entry': 1,
        'near_duplicate_entry': 2,
        'withheld_exact': 2,  # entries 0 and 5
        'withheld_near_duplicate': 2,  # entries 2 and 3
        'withheld': 4,
        'eligible': 3,
    }

# -------------------------------------------------------------------------------------------------
# Assignment
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def assignment_entries():
    rows = [(i, '111110', f'soybean entry {i}') for i in range(20)]
    rows.append((20, '111120', 'oilseed entry'))
    rows += [(i, '111130', f'rye entry {i}') for i in range(21, 25)]
    return _entries(rows)

def _eligible(entries):
    return entries.get_column('code').ne('111130').to_numpy()

def _counts(roles, code):
    grouped = roles.filter(pl.col('code') == code).group_by('role').len()
    return {IndexRole(role): count for role, count in grouped.iter_rows()}

def test_every_entry_gets_exactly_one_role_in_per_code_quotas(assignment_entries):
    roles = assign_index_roles(assignment_entries, _eligible(assignment_entries), STAGE2, seed=7)

    assert roles.columns == ['entry_id', 'code', 'role']
    assert roles.get_column('entry_id').to_list() == list(range(25))
    assert _counts(roles, '111110') == {E: 6, TR: 7, V: 4, TE: 3}
    assert _counts(roles, '111120') == {E: 1}
    assert _counts(roles, '111130') == {E: 1, TR: 3}

def test_ineligible_entries_are_never_held_out(assignment_entries):
    roles = assign_index_roles(assignment_entries, _eligible(assignment_entries), STAGE2, seed=7)

    held_out = roles.filter(pl.col('role').is_in([V.value, TE.value]))
    assert '111130' not in held_out.get_column('code').to_list()

def test_assignment_is_deterministic_per_seed(assignment_entries):
    eligible = _eligible(assignment_entries)
    first = assign_index_roles(assignment_entries, eligible, STAGE2, seed=7)
    again = assign_index_roles(assignment_entries, eligible, STAGE2, seed=7)
    other = assign_index_roles(assignment_entries, eligible, STAGE2, seed=8)

    assert first.equals(again)
    assert not first.equals(other)

def test_assignment_needs_one_flag_per_entry(assignment_entries):
    with pytest.raises(ValueError, match='flags for 25 entries'):
        assign_index_roles(assignment_entries, np.ones(3, dtype=bool), STAGE2, seed=7)

# -------------------------------------------------------------------------------------------------
# The frozen table
# -------------------------------------------------------------------------------------------------

def test_role_table_round_trips_with_string_codes(tmp_path):
    roles = pl.DataFrame(
        {
            'entry_id': [3, 1],
            'code': ['111110', '111120'],
            'role': ['test', 'examples']
        }
    )
    path = tmp_path / 'conf' / 'index_roles.csv'

    digest = write_role_table(roles, path)

    assert digest == sha256_file(path) == role_table_fingerprint(roles)
    assert path.read_text().splitlines() == [
        'entry_id,code,role',
        '1,111120,examples',
        '3,111110,test',
    ]
    assert read_role_table(path).equals(roles.sort('entry_id'))

def test_attach_role_text_joins_on_entry_id():
    entries = _entries([(0, '111110', 'Soybean farming'), (1, '111120', 'Oilseed farming')])
    roles = pl.DataFrame(
        {
            'entry_id': [1, 0],
            'code': ['111120', '111110'],
            'role': ['test', 'examples']
        }
    )

    joined = attach_role_text(roles, entries)

    assert joined.rows() == [
        (0, '111110', 'Soybean farming', 'examples'),
        (1, '111120', 'Oilseed farming', 'test'),
    ]

@pytest.mark.parametrize(
    ('roles', 'message'),
    [
        ({
            'entry_id': [0],
            'code': ['111110'],
            'role': ['examples']
        }, 'different entries'),
        (
            {
                'entry_id': [0, 1],
                'code': ['111110', '111110'],
                'role': ['examples', 'test']
            }, 'different code'
        ),
    ],
)
def test_attach_role_text_rejects_a_table_for_other_entries(roles, message):
    entries = _entries([(0, '111110', 'Soybean farming'), (1, '111120', 'Oilseed farming')])

    with pytest.raises(ValueError, match=message):
        attach_role_text(pl.DataFrame(roles), entries)

# -------------------------------------------------------------------------------------------------
# Table invariants (shared with the bundle loader)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def role_rows():
    return pl.DataFrame(
        {
            'entry_id': [0, 1, 2, 3],
            'code': ['111110', '111110', '111110', '111120'],
            'text': ['Soybeans, organic', 'Soybean seed', 'Edamame', 'Oilseed farming'],
            'role': ['examples', 'examples', 'validation', 'examples'],
        }
    )

def test_a_valid_table_passes(role_rows):
    validate_index_role_table(role_rows, ['111110', '111120', '112130'])

@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (lambda f: f.with_columns(entry_id=pl.lit(0)), 'more than one role'),
        (lambda f: f.with_columns(role=pl.lit('holdout')), 'unknown roles'),
        (lambda f: f.with_columns(code=pl.lit('11111')), 'outside the six-digit codebook'),
        (lambda f: f.with_columns(role=pl.lit(None, pl.Utf8)), 'null values'),
        (lambda f: f.with_columns(text=pl.lit('  ')), 'empty entry text'),
        (lambda f: f.with_columns(role=pl.lit('training')), 'fewer than 1 examples-role'),
        (lambda f: f.drop('text'), 'lack required columns'),
    ],
)
def test_table_violations_fail_closed(role_rows, change, message):
    with pytest.raises(ValueError, match=message):
        validate_index_role_table(change(role_rows), ['111110', '111120', '112130'])

# -------------------------------------------------------------------------------------------------
# Consistency with descriptions
# -------------------------------------------------------------------------------------------------

def _descriptions(examples_by_code, title='Soybean Farming'):
    codes = sorted(examples_by_code)
    return pl.DataFrame(
        {
            'code': codes,
            'title': [title] * len(codes),
            'description': ['This industry comprises farms.'] * len(codes),
            'examples': [examples_by_code[code] for code in codes],
            'excluded': [None] * len(codes),
        },
        schema={
            'code': pl.Utf8,
            'title': pl.Utf8,
            'description': pl.Utf8,
            'examples': pl.Utf8,
            'excluded': pl.Utf8,
        },
    )

def test_examples_channel_must_hold_exactly_the_examples_role_entries(role_rows):
    good = _descriptions({'111110': 'Soybeans, organic; Soybean seed', '111120': 'Oilseed farming'})
    verify_examples_channel(good, role_rows)

    stale = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed; Edamame',
            '111120': 'Oilseed farming'
        }
    )
    with pytest.raises(ValueError, match='examples channel other than'):
        verify_examples_channel(stale, role_rows)

def test_role_leakage_reports_zero_for_clean_splits(role_rows):
    descriptions = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed',
            '111120': 'Oilseed farming'
        }
    )

    report = verify_role_leakage(descriptions, role_rows)

    assert report == {
        'validation': {
            'exact': 0,
            'near_duplicate': 0
        },
        'test': {
            'exact': 0,
            'near_duplicate': 0
        },
    }

def test_role_leakage_fails_on_a_held_out_query_in_training_text(role_rows):
    leaky = role_rows.with_columns(
        text=pl.when(pl.col('entry_id') == 2).then(pl.lit('Soybean farming')).otherwise('text')
    )
    descriptions = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed',
            '111120': 'Oilseed farming'
        }
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        verify_role_leakage(descriptions, leaky)
