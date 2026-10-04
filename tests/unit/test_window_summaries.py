'''
Window-fitting summaries (roadmap Stage 6b): the pin and its identity (spec 4.6), the units
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import logging
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.leakage import SENTENCE_BREAK
from naics_embedder.panels.window_summaries import (
    SUMMARIES_SCHEMA,
    SummariesPin,
    over_window,
    read_window_summaries,
    resolve_channel_texts,
    summaries_identity,
    summary_budget,
    text_sha256,
    text_units,
    token_counter,
    write_window_summaries,
)
from naics_embedder.text_model.fields import marked_text
from tests.fixtures.window_summaries import WordTokenizer, pin_artifact, words

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

# -------------------------------------------------------------------------------------------------
# The artifact and the resolver
# -------------------------------------------------------------------------------------------------

WINDOW = 10
STUB = 'stub-backbone'
# 'description: ' and three three-word sentences: 1 + 9 words and [CLS] [SEP], 12 > 10 tokens
LONG = 'Farms grow corn. Farms grow wheat. Farms sell grain.'
SUMMARY = 'Farms grow corn. Farms sell grain.'
FITS = 'Farms grow oilseeds.'

def descriptions_frame(description=LONG, examples='Soybeans; Beans'):
    return pl.DataFrame(
        {
            'code': ['111110', '111120'],
            'title': ['Soybean Farming', 'Oilseed Farming'],
            'description': [description, FITS],
            'examples': [examples, None],
            'excluded': [None, '  '],
        }
    )

def summary_row(code='111110', channel='description', summary=SUMMARY, source=LONG, **overrides):
    row = {
        'code': code,
        'channel': channel,
        'source_sha256': text_sha256(source),
        'window': WINDOW,
        'summary': summary,
        'source_tokens': 12,
        'summary_tokens': 9,
        'units_kept': 2,
        'units_total': 3,
    }
    row.update(overrides)
    return row

def pin_rows(tmp_path, rows, window=WINDOW):
    return pin_artifact(tmp_path / 'window_summaries.csv', rows, window=window)

def resolve(descriptions, pin):
    return resolve_channel_texts(descriptions, WordTokenizer(), STUB, WINDOW, pin=pin)

def test_the_artifact_round_trips_sorted_by_channel_then_code(tmp_path):
    rows = [
        summary_row(code='222220', channel='excluded'),
        summary_row(code='111120'),
        summary_row(code='111110', channel='excluded'),
    ]
    path = tmp_path / 'window_summaries.csv'

    sha256 = write_window_summaries(pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), path)
    table = read_window_summaries(path)

    assert sha256 == text_sha256(path.read_text(encoding='utf-8'))
    assert table.schema == pl.Schema(SUMMARIES_SCHEMA)
    assert table.select('channel', 'code').rows() == [
        ('description', '111120'),
        ('excluded', '111110'),
        ('excluded', '222220'),
    ]
    # Rewriting what was read reproduces the bytes
    assert write_window_summaries(table, tmp_path / 'again.csv') == sha256

@pytest.mark.parametrize(
    ('rows', 'refusal'),
    [
        pytest.param(
            [summary_row(channel='title')],
            "a channel outside \\('description', 'examples', 'excluded'\\): title",
            id='channel',
        ),
        pytest.param(
            [summary_row(), summary_row(summary='Farms grow corn.')],
            "summarizes code 111110's description more than once",
            id='repeated',
        ),
    ],
)
def test_the_reader_refuses_a_malformed_artifact(tmp_path, rows, refusal):
    path = tmp_path / 'window_summaries.csv'
    pl.DataFrame(rows, schema=SUMMARIES_SCHEMA).write_csv(path)

    with pytest.raises(ValueError, match=refusal):
        read_window_summaries(path)

def test_over_window_lists_each_present_text_beyond_the_window_titles_included():
    count_marked = token_counter(WordTokenizer())
    # A marker adds a word, and [CLS] and [SEP] two tokens: seven words fill the window exactly
    at_window = 'Farms grow corn wheat rice oats rye'
    past_window = 'Farms grow corn wheat rice oats rye barley'
    long_title = 'Soybean Farming and Oilseed Growing Services for Others'
    assert count_marked([marked_text('description', at_window)]) == [WINDOW]
    assert count_marked([marked_text('description', past_window)]) == [WINDOW + 1]
    assert count_marked([marked_text('title', long_title)]) == [WINDOW + 1]
    descriptions = pl.DataFrame(
        {
            'code': ['111110', '222220'],
            'title': [long_title, 'Wheat Farming'],
            'description': [at_window, past_window],
            'examples': ['Soybeans; Beans', '  '],
            'excluded': [None, 'Dairy farming'],
        }
    )

    # Sorted by channel, then code: the loop meets the title first, and the sort puts it last
    assert over_window(descriptions, count_marked, WINDOW) == [
        ('222220', 'description'),
        ('111110', 'title'),
    ]

def test_an_over_window_text_is_replaced_by_its_summary(tmp_path, caplog):
    descriptions = descriptions_frame()

    with caplog.at_level(logging.INFO, logger='naics_embedder.panels.window_summaries'):
        resolved = resolve(descriptions, pin_rows(tmp_path, [summary_row()]))

    assert resolved.get_column('description').to_list() == [SUMMARY, FITS]
    assert resolved.drop('description').equals(descriptions.drop('description'))
    assert "{'description': 1, 'examples': 0, 'excluded': 0}" in caplog.text

def test_an_examples_text_is_replaced_by_its_entries(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    row = summary_row(channel='examples', summary='Soybeans; Corn', source=examples)

    descriptions = descriptions_frame(description='Farms grow corn.', examples=examples)

    resolved = resolve(descriptions, pin_rows(tmp_path, [row]))

    assert resolved.get_column('examples').to_list() == ['Soybeans; Corn', None]

def test_texts_that_fit_pass_through_without_a_pin_or_an_artifact():
    descriptions = descriptions_frame(description='Farms grow corn.')
    unreadable = SummariesPin(path='/nonexistent/window_summaries.csv', sha256='f' * 64, window=99)

    assert resolve(descriptions, None).equals(descriptions)
    assert resolve(descriptions, unreadable).equals(descriptions)
    # The seam's MiniLM pin names no file either
    assert resolve_channel_texts(descriptions, WordTokenizer(), MINILM, WINDOW).equals(descriptions)

def test_the_default_pin_is_the_backbones_entry_at_call_time(tmp_path, monkeypatch):
    pin = pin_rows(tmp_path, [summary_row()])
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, STUB, pin)

    resolved = resolve_channel_texts(descriptions_frame(), WordTokenizer(), STUB, WINDOW)

    assert resolved.get_column('description').to_list()[0] == SUMMARY

def test_an_over_window_text_without_a_pin_is_refused():
    with pytest.raises(ValueError, match="code 111110's description is over the 10-token window"):
        resolve(descriptions_frame(), None)

def test_none_names_no_pin_even_for_a_backbone_that_has_one(tmp_path, monkeypatch):
    # The same texts resolve under the backbone's pin, which is the default
    monkeypatch.setitem(
        window_summaries.WINDOW_SUMMARIES, STUB, pin_rows(tmp_path, [summary_row()])
    )
    default = resolve_channel_texts(descriptions_frame(), WordTokenizer(), STUB, WINDOW)
    assert default.get_column('description').to_list()[0] == SUMMARY

    with pytest.raises(ValueError, match='no window summaries are pinned for stub-backbone'):
        resolve(descriptions_frame(), None)

def test_a_pin_for_another_window_is_refused(tmp_path):
    pin = pin_rows(tmp_path, [summary_row()], window=12)

    with pytest.raises(ValueError, match='fit a 12-token window, but texts are tokenized at 10'):
        resolve(descriptions_frame(), pin)

def test_a_missing_artifact_is_refused_by_its_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin = SummariesPin(path='conf/window_summaries.csv', sha256='f' * 64, window=WINDOW)

    with pytest.raises(ValueError, match='are missing') as refusal:
        resolve(descriptions_frame(), pin)

    assert str((tmp_path / 'conf/window_summaries.csv').resolve()) in str(refusal.value)

def test_an_artifact_with_another_sha256_is_refused(tmp_path):
    pin = pin_rows(tmp_path, [summary_row()])
    Path(pin.path).write_text(Path(pin.path).read_text() + '\n')

    with pytest.raises(ValueError, match='but the pin names'):
        resolve(descriptions_frame(), pin)

@pytest.mark.parametrize(
    ('rows', 'refusal'),
    [
        pytest.param([], "code 111110's description is over the window", id='missing'),
        pytest.param(
            [summary_row(), summary_row(code='111120', source=FITS)],
            "code 111120's description fits the window",
            id='fits',
        ),
        pytest.param(
            [summary_row(), summary_row(code='999999')],
            "code 999999's description is not in the descriptions",
            id='unknown-code',
        ),
        pytest.param(
            [summary_row(), summary_row(channel='excluded')],
            "code 111110's excluded is not in the descriptions",
            id='absent-text',
        ),
    ],
)
def test_the_rows_must_be_exactly_the_over_window_texts(tmp_path, rows, refusal):
    with pytest.raises(ValueError, match=refusal):
        resolve(descriptions_frame(), pin_rows(tmp_path, rows))

@pytest.mark.parametrize(
    ('row', 'refusal'),
    [
        pytest.param(
            summary_row(source='Farms grow corn.'),
            "code 111110's description: the summary was built from another source text",
            id='source',
        ),
        pytest.param(
            summary_row(window=12),
            "code 111110's description: the summary fits a 12-token window, not 10",
            id='window',
        ),
    ],
)
def test_each_row_must_match_its_source_and_window(tmp_path, row, refusal):
    with pytest.raises(ValueError, match=refusal):
        resolve(descriptions_frame(), pin_rows(tmp_path, [row]))

def test_every_row_is_checked_against_its_source_before_any_is_checked_as_an_extract(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    rows = [
        # Reordered, so not an extract (step 6), and sorted first
        summary_row(summary='Farms sell grain. Farms grow corn.'),
        # Built from another text (step 5)
        summary_row(channel='examples', summary='Soybeans; Corn', source='Soybeans; Corn'),
    ]

    with pytest.raises(ValueError, match="code 111110's examples: the summary was built from"):
        resolve(descriptions_frame(examples=examples), pin_rows(tmp_path, rows))

@pytest.mark.parametrize(
    'summary',
    [
        pytest.param('Farms sell grain. Farms grow corn.', id='reordered'),
        pytest.param('Farms grow corn. Farms grow corn.', id='repeated'),
        pytest.param('Farms grow maize. Farms sell grain.', id='edited'),
    ],
)
def test_a_summary_that_is_not_an_extract_is_refused(tmp_path, summary):
    pin = pin_rows(tmp_path, [summary_row(summary=summary)])

    with pytest.raises(ValueError, match="code 111110's description: the summary is not an"):
        resolve(descriptions_frame(), pin)

def test_reordered_examples_entries_are_refused(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    row = summary_row(channel='examples', summary='Corn; Soybeans', source=examples)

    descriptions = descriptions_frame(description='Farms grow corn.', examples=examples)

    with pytest.raises(ValueError, match="code 111110's examples: the summary is not an extract"):
        resolve(descriptions, pin_rows(tmp_path, [row]))

def test_a_summary_over_the_window_is_refused(tmp_path):
    # The whole text is an extract of itself, and still over the window
    with pytest.raises(ValueError, match="code 111110's description does not fit the 10-token"):
        resolve(descriptions_frame(), pin_rows(tmp_path, [summary_row(summary=LONG)]))
