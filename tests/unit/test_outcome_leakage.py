'''
Leakage matching between held-out queries and training text (Req 3, "Leakage").

Expected values are worked out by hand from the matching rules, never from production code.
'''

from fractions import Fraction

import polars as pl
import pytest

from naics_embedder.panels.leakage import (
    find_leakage,
    find_leakage_within,
    normalize_text,
    text_segments,
    training_text_segments,
)

pytestmark = pytest.mark.unit

def test_normalize_text_keeps_ascii_letters_and_digits():
    assert normalize_text('Soybean farming, field & seed production') == (
        'soybean farming field seed production'
    )
    assert normalize_text('Nurse practitioners’ offices (e.g., centers)') == (
        'nurse practitioners offices e g centers'
    )
    assert normalize_text('  T-shirts  ') == 't shirts'

def test_text_segments_split_sentences_and_keep_activity_phrases():
    text = (
        'Growing soybeans--are classified in Industry 111110, Soybean Farming; '
        'Growing corn--are classified in Industry 111150. See also.'
    )

    assert text_segments(text) == [
        'growing soybeans are classified in industry 111110 soybean farming',
        'growing soybeans',
        'growing corn are classified in industry 111150',
        'growing corn',
        'see also',
    ]
    assert text_segments(None) == []
    assert text_segments('') == []

def test_training_text_segments_cover_every_channel_and_extra_texts():
    descriptions = pl.DataFrame(
        {
            'title': ['Soybean Farming'],
            'description': ['This industry grows soybeans. It also sells seed.'],
            'examples': ['Soybean farming, field; Soybean seed production'],
            'excluded': [None],
        },
        schema={name: pl.Utf8
                for name in ('title', 'description', 'examples', 'excluded')},
    )

    assert training_text_segments(descriptions, extra_texts=['Dry pea farming']) == [
        'dry pea farming',
        'it also sells seed',
        'soybean farming',
        'soybean farming field',
        'soybean seed production',
        'this industry grows soybeans',
    ]

def test_training_text_segments_require_the_text_columns():
    with pytest.raises(ValueError, match='text columns'):
        training_text_segments(pl.DataFrame({'title': ['Soybean Farming']}))

def test_exact_matches_need_whole_words():
    matches = find_leakage(
        ['art supplies', 'Card shops', 'party'],
        ['Greeting card shops', 'party supplies stores'],
    )

    # 'art supplies' sits inside 'party supplies' but not on a word boundary
    assert matches.exact.tolist() == [False, True, True]
    assert matches.near_duplicate.tolist() == [False, False, False]

def test_reordered_words_are_near_duplicates_but_not_exact():
    matches = find_leakage(['Card shops, greeting'], ['greeting card shops'])

    assert matches.exact.tolist() == [False]
    assert matches.near_duplicate.tolist() == [True]

def test_near_duplicate_threshold_is_inclusive():
    # 'abcdefghi' has 9 distinct trigrams; ' z ' adds 1 (Jaccard 9/10), 'yz' adds 2 (9/11)
    matches = find_leakage(['abcdefghi z', 'abcdefghi yz'], ['abcdefghi'])

    assert matches.exact.tolist() == [False, False]
    assert matches.near_duplicate.tolist() == [True, False]
    assert matches.leaked.tolist() == [True, False]

def test_within_a_list_self_matches_are_skipped_but_duplicates_flag_each_other():
    matches = find_leakage_within(
        ['Soybean farming', 'soybean  farming!', 'Dry pea farming', 'pea farming']
    )

    # 'pea farming' occurs inside 'dry pea farming'; the two soybean texts normalize equal
    assert matches.exact.tolist() == [True, True, False, True]
    # 'dry pea farming' and 'pea farming' share 10 of 13 trigrams: below 9/10
    assert matches.near_duplicate.tolist() == [True, True, False, False]

def test_queries_without_letters_or_digits_are_rejected():
    with pytest.raises(ValueError, match='no letters or digits'):
        find_leakage(['--'], ['soybean farming'])

@pytest.mark.parametrize('threshold', [Fraction(0), Fraction(3, 2)])
def test_threshold_must_lie_in_the_unit_interval(threshold):
    with pytest.raises(ValueError, match='min_jaccard'):
        find_leakage(['soybean farming'], ['soybean farming'], min_jaccard=threshold)

def test_empty_inputs_flag_nothing():
    assert find_leakage([], ['soybean farming']).leaked.tolist() == []
    assert find_leakage(['soybean farming'], []).leaked.tolist() == [False]
