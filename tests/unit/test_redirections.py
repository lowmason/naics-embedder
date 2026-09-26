'''
The redirection table and the exclusion channel (Req 8).

Expected values are worked out by hand from the rules in ``data/redirections.py``.
'''

import polars as pl
import pytest

from naics_embedder.data.redirections import (
    REDIRECTIONS_SCHEMA,
    activity_phrase,
    build_redirections,
    exclusion_channel,
    lineal_codes,
    named_codes,
)

pytestmark = pytest.mark.unit

CODES = {'11', '111', '1111', '11111', '111110', '11112', '111120'}
NO_PARAGRAPHS = pl.DataFrame(schema={'code': pl.Utf8, 'text': pl.Utf8})

SOYBEANS = (
    'Growing soybeans for green manure--are classified in Industry 111120, Oilseed (except '
    'Soybean) Farming.'
)
COMBINATIONS = (
    'Growing oilseed and grain combinations--are classified in Industry Group 1111, Oilseed and '
    'Grain Farming.'
)
MANAGEMENT = 'Farm management services are classified in the Agriculture sector.'
PARAGRAPH = 'Excluded from this industry group are soybean farms, classified in Industry 111110.'

# -------------------------------------------------------------------------------------------------
# One row's parts
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'text, activity',
    [
        pytest.param(
            'Growing soybeans--are classified in Industry 111110, Soybean Farming.',
            'Growing soybeans',
            id='dashes',
        ),
        pytest.param(
            'Establishments primarily engaged in growing hay are classified in Industry 111940.',
            'Establishments primarily engaged in growing hay',
            id='sentence',
        ),
        pytest.param(
            'Growing hay--are lclassified in Industry 111940, Hay Farming.',
            'Growing hay',
            id='misspelled',
        ),
        pytest.param(
            'Tax return preparation is included in Industry 541213.',
            'Tax return preparation',
            id='is-included',
        ),
        pytest.param('See Industry 111110 for soybean farming.', None, id='no-redirection'),
        pytest.param('--are classified in Industry 111110.', None, id='no-activity'),
    ],
)
def test_activity_phrase_is_the_text_before_the_redirection(text, activity):
    assert activity_phrase(text) == activity

def test_named_codes_are_other_codebook_codes_in_order_of_first_appearance():
    text = (
        'Growing hay--are classified in Industry 111120, Industry Group 1111, Industry 111120 '
        'again, Industry 999999 or Industry 111110.'
    )

    assert named_codes('111110', text, CODES) == ['111120', '1111']
    # A combined sector is named by its first code
    assert named_codes('111110', 'Retailing--are classified in Sector 44-45.', {'44'}) == ['44']

def test_lineal_codes_are_named_ancestors_and_descendants():
    assert lineal_codes('111120', ['1111', '111110', '11']) == ['1111', '11']
    # 711's "Excluded" paragraph names its child 7113
    assert lineal_codes('711', ['7113', '722']) == ['7113']

# -------------------------------------------------------------------------------------------------
# The table
# -------------------------------------------------------------------------------------------------

def test_the_table_lists_every_row_once_in_order():
    references = pl.DataFrame(
        {
            'code': ['111110', '111120', '111120'],
            'text': [SOYBEANS, COMBINATIONS, MANAGEMENT]
        }
    )
    paragraphs = pl.DataFrame({'code': ['1111'], 'text': [PARAGRAPH]})

    table = build_redirections(references, paragraphs, CODES)

    soybeans = 'Growing soybeans for green manure'
    combinations = 'Growing oilseed and grain combinations'
    assert table.schema == pl.Schema(REDIRECTIONS_SCHEMA)
    assert table.rows() == [
        (0, 'cross_reference', '111110', SOYBEANS, soybeans, ['111120'], [], False),
        (1, 'cross_reference', '111120', COMBINATIONS, combinations, ['1111'], ['1111'], False),
        # Names no code: its text stays, but it has no activity phrase
        (2, 'cross_reference', '111120', MANAGEMENT, None, [], [], False),
        (3, 'description', '1111', PARAGRAPH, None, ['111110'], ['111110'], False),
    ]

def test_a_row_a_held_out_query_leaks_into_is_withheld_and_loses_its_activity():
    references = pl.DataFrame(
        {
            'code': ['111110', '111120', '111120'],
            'text': [
                # The first query reorders this activity phrase, but no sentence of the text
                'Establishments primarily engaged in growing soybeans for green manure are '
                'classified in Industry 111120.',
                'Growing hay--are classified in Industry 111110.',
                # The second query occurs in this text as whole words
                MANAGEMENT,
            ],
        }
    )
    queries = [
        'Growing soybeans for green manure, establishments primarily engaged in',
        'Farm management services',
    ]

    table = build_redirections(references, NO_PARAGRAPHS, CODES, queries)

    assert table.get_column('withheld').to_list() == [True, False, True]
    assert table.get_column('activity').to_list() == [None, 'Growing hay', None]
    # A withheld row still names its destination, so the pair stays an exclusion
    assert table.get_column('named_codes').to_list() == [['111120'], ['111110'], []]

# -------------------------------------------------------------------------------------------------
# The exclusion channel
# -------------------------------------------------------------------------------------------------

def test_the_channel_joins_each_kept_text_once_and_keeps_withheld_destinations():
    soybeans = 'Growing soybeans for green manure--are classified in Industry 111120.'
    hay = 'Growing hay--are classified in Industry 111940 or Industry 111120.'
    peanuts = 'Excluded are peanut farms, classified in Industry 111992.'
    table = pl.DataFrame(
        [
            (0, 'cross_reference', '111110', soybeans, 'x', ['111120'], [], False),
            (1, 'cross_reference', '111110', hay, 'x', ['111940', '111120'], [], False),
            (2, 'cross_reference', '111120', 'Growing soybeans.', None, ['111110'], [], True),
            (3, 'description', '111110', peanuts, None, ['111992'], [], False),
            (4, 'cross_reference', '111940', MANAGEMENT, None, [], [], False),
        ],
        schema=REDIRECTIONS_SCHEMA,
        orient='row',
    )

    channel = exclusion_channel(table)

    assert channel.rows() == [
        ('111110', f'{soybeans} {hay} {peanuts}', ['111120', '111940', '111992']),
        # Every row withheld: no text, but the destination stays an exclusion
        ('111120', None, ['111110']),
        # Names no code: text only
        ('111940', MANAGEMENT, None),
    ]
