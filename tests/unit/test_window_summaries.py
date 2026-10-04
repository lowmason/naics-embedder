'''
Window-fitting summaries (roadmap Stage 6b): the pin and its identity (spec 4.6), the units
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.leakage import SENTENCE_BREAK
from naics_embedder.panels.window_summaries import (
    SummariesPin,
    summaries_identity,
    summary_budget,
    text_units,
    token_counter,
)
from tests.fixtures.window_summaries import words

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

@pytest.fixture(scope='module')
def minilm_tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

# -------------------------------------------------------------------------------------------------
# The pin and its identity
# -------------------------------------------------------------------------------------------------

def test_the_identity_is_the_pins_sha256_and_none_without_a_pin(monkeypatch):
    pin = SummariesPin(path='summaries.csv', sha256='a' * 64, window=16)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, 'tiny-backbone', pin)

    assert summaries_identity('tiny-backbone') == 'a' * 64
    assert summaries_identity('unpinned/backbone') is None

def test_the_seam_pins_minilm_alone_to_a_pin_no_test_can_read():
    '''tests/conftest.py's autouse seam (spec section 6).'''

    assert list(window_summaries.WINDOW_SUMMARIES) == [MINILM]
    pin = window_summaries.WINDOW_SUMMARIES[MINILM]
    assert summaries_identity(MINILM) == pin.sha256
    assert pin.sha256 is not None
    assert pin.window == 128
    assert not Path(pin.path).exists()

@pytest.mark.real_window_summaries
def test_the_marker_leaves_the_committed_pins_alone():
    # Every committed pin names its artifact; the seam's dummy names a file that does not exist
    for pin in window_summaries.WINDOW_SUMMARIES.values():
        assert Path(pin.path).is_file()

def test_the_module_imports_no_torch():
    '''The resolver runs in the token cache and the text-only builder; it loads no model.'''

    imported = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import naics_embedder.panels.window_summaries; '
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert imported.stdout.strip() == 'False'

# -------------------------------------------------------------------------------------------------
# Units
# -------------------------------------------------------------------------------------------------

def test_the_budget_is_the_window_less_the_marker_and_special_tokens(minilm_tokenizer):
    count_marked = token_counter(minilm_tokenizer)

    for channel in ('title', 'description', 'examples', 'excluded'):
        assert summary_budget(count_marked, channel, 128) == 124

def test_the_counter_counts_special_tokens_only_when_asked(minilm_tokenizer):
    # 'soybean' is three word pieces: soy, ##be, ##an
    assert token_counter(minilm_tokenizer)(['soybean farming', 'corn']) == [6, 3]
    assert token_counter(minilm_tokenizer, special_tokens=False)(['soybean farming']) == [4]
    # The tokenizer itself raises on an empty batch
    assert token_counter(minilm_tokenizer)([]) == []

@pytest.mark.parametrize(
    ('channel', 'text', 'expected'),
    [
        pytest.param(
            'description',
            'Farms in the U.S. grow corn. Growers, i.e. farmers, sell it. Fruit, e.g. apples, is '
            'excluded.',
            [
                'Farms in the U.S. grow corn.',
                'Growers, i.e. farmers, sell it.',
                'Fruit, e.g. apples, is excluded.',
            ],
            id='abbreviations',
        ),
        pytest.param(
            'description',
            'Mills grade wheat No. 2 and corn, etc. for feed. Bakers buy flour vs. meal.',
            ['Mills grade wheat No. 2 and corn, etc. for feed.', 'Bakers buy flour vs. meal.'],
            id='no-etc-vs',
        ),
        pytest.param(
            'description',
            'Farms do: 1. growing crops. 2. raising animals. Ranches are included.',
            ['Farms do: 1. growing crops.', '2. raising animals.', 'Ranches are included.'],
            id='numbered-list',
        ),
        pytest.param(
            'description',
            'Farms that grow onions are classified in Industry 111113. Others are not.',
            ['Farms that grow onions are classified in Industry 111113.', 'Others are not.'],
            id='a-code-closes-a-unit',
        ),
        pytest.param(
            'excluded',
            'Growing crops (1); raising animals (2); fishing--are classified in Industry 114111.',
            [
                'Growing crops (1); raising animals (2); fishing--are classified in Industry '
                '114111.'
            ],
            id='parenthesized-numerals',
        ),
        pytest.param(
            'excluded',
            'Growing soybeans--are classified in Industry 111110, Soybean Farming; Growing '
            'wheat--are classified in Industry 111140. Growing rice (except wild rice; see '
            '111199)--are classified in Industry 111160, Rice Farming.',
            [
                'Growing soybeans--are classified in Industry 111110, Soybean Farming;',
                'Growing wheat--are classified in Industry 111140.',
                'Growing rice (except wild rice; see 111199)--are classified in Industry 111160, '
                'Rice Farming.',
            ],
            id='cross-references',
        ),
    ],
)
def test_units_are_sentences_that_close_only_at_a_real_break(channel, text, expected):
    units = text_units(channel, text, words, budget=100)

    assert units == expected
    # Every unit boundary is a boundary of the leakage segmenter (spec 4.2)
    pieces = [piece for unit in units for piece in SENTENCE_BREAK.split(unit)]
    assert pieces == SENTENCE_BREAK.split(text)

def test_a_sentence_over_the_budget_is_re_split_at_its_clauses():
    text = 'Farms grow corn; farms grow wheat; farms grow rice. Ranches raise cattle.'

    assert text_units('description', text, words, budget=6) == [
        'Farms grow corn;',
        'farms grow wheat;',
        'farms grow rice.',
        'Ranches raise cattle.',
    ]

def test_a_clause_over_the_budget_is_re_split_at_its_pieces_without_the_guards():
    # The parentheses keep the sentence one clause; its pieces split inside them
    text = 'Farms grow (corn; wheat; rice) here. Ranches raise cattle.'

    assert text_units('description', text, words, budget=4) == [
        'Farms grow (corn;',
        'wheat;',
        'rice) here.',
        'Ranches raise cattle.',
    ]

def test_a_piece_over_the_budget_is_refused():
    with pytest.raises(ValueError, match='a description piece is over the 3-token budget'):
        text_units('description', 'Farms grow corn and wheat. Ranches raise cattle.', words, 3)

def test_examples_units_are_the_entries():
    assert text_units('examples', 'Corn farming; ; Wheat farming', words, 100) == [
        'Corn farming',
        'Wheat farming',
    ]

def test_an_examples_entry_over_the_budget_is_refused():
    with pytest.raises(ValueError, match='an examples entry is over the 1-token budget'):
        text_units('examples', 'Corn; Wheat farming', words, 1)

def test_a_title_is_one_unit_and_one_over_the_window_is_refused():
    assert text_units('title', 'Soybean Farming', words, 2) == ['Soybean Farming']
    with pytest.raises(ValueError, match='a title over the window cannot be summarized'):
        text_units('title', 'Soybean Farming', words, 1)

def test_a_channel_without_units_is_refused():
    with pytest.raises(ValueError, match="no units are defined for channel 'query'"):
        text_units('query', 'soybeans', words, 100)
