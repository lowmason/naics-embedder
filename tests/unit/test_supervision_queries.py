'''
Req 11's task queries (spec 4.1(i); section 6, "Task term", the data half).

Every expected query is written by hand from the reference bundle's rows
(``tests/fixtures/supervision.py``): a training entry is a level-6 query with its code as the one
target, and a phrase row sends its phrase to its named codes that are not lineal to its own code,
with its own code as a forced negative.
'''

import logging
import re

import polars as pl
import pytest

from naics_embedder.supervision.activity import activity_phrase
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.queries import TaskQuery, build_task_queries
from tests.fixtures.supervision import (
    REFERENCE_INDEX_ROLE_ROWS,
    REFERENCE_REDIRECTION_ROWS,
    build_reference_bundle,
)

# The reference bundle's eleven queries, sorted by level, then text
REFERENCE_QUERIES = [
    TaskQuery('Retailing new cars', 2, ('44', ), ('311211', )),
    TaskQuery('Wood products manufacturing', 3, ('321', ), ('441111', )),
    TaskQuery('Wood flour grinding', 4, ('3112', ), ('321111', )),
    TaskQuery('Dealing in new cars', 5, ('44111', ), ('311111', )),
    TaskQuery('Cat food manufacturing', 6, ('311111', ), ()),
    TaskQuery('Corn meal milling', 6, ('311211', ), ()),
    TaskQuery('Dealing in new cars', 6, ('441111', ), ()),
    TaskQuery('Food manufacturing', 6, ('311111', '311211'), ('441111', )),
    TaskQuery('New car dealers', 6, ('441111', ), ()),
    TaskQuery('Sawmilling', 6, ('321111', ), ('311111', )),
    TaskQuery('Wood flour grinding', 6, ('311211', ), ('321111', )),
]

def _queries_named(bundle, text):
    return [query for query in build_task_queries(bundle) if query.text == text]

# -------------------------------------------------------------------------------------------------
# The queries, their targets and their forced negatives
# -------------------------------------------------------------------------------------------------

def test_the_reference_bundle_gives_exactly_these_task_queries(reference_bundle):
    assert build_task_queries(reference_bundle) == REFERENCE_QUERIES

def test_lineal_and_withheld_rows_never_become_queries(reference_bundle):
    rows = pl.read_parquet(reference_bundle.artifact_path('redirections'))
    lineal = rows.row(2, named=True)
    withheld = rows.row(3, named=True)
    texts = {query.text for query in build_task_queries(reference_bundle)}

    # Row 2's one named code, '3111', is its code's ancestor, so its phrase stays text only
    assert lineal['activity'] == 'Mixed food making'
    assert lineal['named_codes'] == lineal['lineal_codes'] == ['3111']
    assert 'Mixed food making' not in texts
    # Row 3's text gives a phrase, which the table drops because the row is withheld
    assert withheld['withheld'] and withheld['activity'] is None
    assert activity_phrase(withheld['text']) == 'Rice flour milling'
    assert 'Rice flour milling' not in texts

def test_a_withheld_row_gives_no_query_even_with_a_phrase(reference_bundle):
    # Past the loader, which refuses a phrase on a withheld row: the builder skips the row itself
    path = reference_bundle.artifact_path('redirections')
    phrased = pl.when(pl.col('reference_id') == 3).then(pl.lit('Rice flour milling'))
    pl.read_parquet(path).with_columns(activity=phrased.otherwise('activity')).write_parquet(path)

    assert build_task_queries(reference_bundle) == REFERENCE_QUERIES

def test_an_index_entry_and_a_phrase_with_one_text_at_level_6_are_one_query(reference_bundle):
    # The training entry gives the target '321111'; row 0, from '311111', adds its code as the
    # forced negative
    assert _queries_named(reference_bundle, 'Sawmilling') == [
        TaskQuery('Sawmilling', 6, ('321111', ), ('311111', )),
    ]

def test_an_index_entry_and_a_phrase_with_one_text_at_other_levels_stay_apart(reference_bundle):
    # Row 7 sends the training entry's text to '441111''s parent, '44111', at level 5
    assert _queries_named(reference_bundle, 'Dealing in new cars') == [
        TaskQuery('Dealing in new cars', 5, ('44111', ), ('311111', )),
        TaskQuery('Dealing in new cars', 6, ('441111', ), ()),
    ]

def test_a_phrase_with_destinations_at_two_levels_is_a_query_at_each(reference_bundle):
    assert _queries_named(reference_bundle, 'Wood flour grinding') == [
        TaskQuery('Wood flour grinding', 4, ('3112', ), ('321111', )),
        TaskQuery('Wood flour grinding', 6, ('311211', ), ('321111', )),
    ]

def test_a_phrase_from_two_rows_takes_each_levels_referencing_codes_as_negatives(tmp_path):
    # A second row, from '441111', sends wood flour grinding to '311' and '311211'. Its code joins N
    # only where its own destinations are, levels 3 and 6, beside row 4's '321111'
    second = (
        9,
        'cross_reference',
        '441111',
        'Wood flour grinding--are classified in Subsector 311 and Industry 311211.',
        'Wood flour grinding',
        ['311', '311211'],
        [],
        False,
    )
    rows = (*REFERENCE_REDIRECTION_ROWS, second)
    bundle = load_validated_bundle(build_reference_bundle(tmp_path, redirection_rows=rows))

    assert _queries_named(bundle, 'Wood flour grinding') == [
        TaskQuery('Wood flour grinding', 3, ('311', ), ('441111', )),
        TaskQuery('Wood flour grinding', 4, ('3112', ), ('321111', )),
        TaskQuery('Wood flour grinding', 6, ('311211', ), ('321111', '441111')),
    ]

def test_a_phrase_with_two_destinations_at_one_level_has_two_targets(reference_bundle):
    assert _queries_named(reference_bundle, 'Food manufacturing') == [
        TaskQuery('Food manufacturing', 6, ('311111', '311211'), ('441111', )),
    ]

def test_a_query_whose_targets_and_negatives_overlap_is_refused(tmp_path):
    # A training entry files sawmilling under '311111', which row 0 sends sawmilling away from
    rows = (*REFERENCE_INDEX_ROLE_ROWS, (15, '311111', 'Sawmilling', 'training'))
    bundle = load_validated_bundle(build_reference_bundle(tmp_path, index_role_rows=rows))
    refusal = (
        "task query 'Sawmilling' at level 6 names ['311111'] as a target and a forced "
        'negative'
    )

    with pytest.raises(ValueError, match=re.escape(refusal)):
        build_task_queries(bundle)

def test_building_the_queries_logs_one_line_of_counts(reference_bundle, caplog):
    with caplog.at_level(logging.INFO, logger='naics_embedder.supervision.queries'):
        build_task_queries(reference_bundle)

    logged = [
        record.getMessage() for record in caplog.records
        if record.name == 'naics_embedder.supervision.queries'
    ]
    assert logged == [
        'Task queries: 11 (5 from index entries, 1 merged with a phrase, 6 phrase-only); '
        'by level: 2: 1, 3: 1, 4: 1, 5: 1, 6: 7'
    ]

# -------------------------------------------------------------------------------------------------
# The fixture: what the monitor needs from the reference bundle
# -------------------------------------------------------------------------------------------------

def test_the_reference_bundle_has_a_validation_entry_for_each_six_digit_code(reference_bundle):
    roles = pl.read_parquet(reference_bundle.artifact_path('index_roles'))
    validation = roles.filter(pl.col('role') == 'validation')

    # So an outcome read ranks each code's entry against the other three: a non-degenerate MRR
    assert validation.get_column('code').sort().to_list() == [
        '311111',
        '311211',
        '321111',
        '441111',
    ]
