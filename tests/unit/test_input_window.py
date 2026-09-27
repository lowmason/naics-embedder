'''
The backbone's trained input window (Req 9; Verification "Backbone input window").
'''

import pytest
from transformers import BertTokenizerFast

from naics_embedder.utils.input_window import (
    check_window,
    overflow_shares,
    token_counter,
    trained_window,
)

pytestmark = pytest.mark.unit

BACKBONE = 'sentence-transformers/all-MiniLM-L6-v2'

def test_the_backbones_trained_window_is_128_tokens():
    # Its model card at revision 1110a243: "The sequence length was limited to 128 tokens."
    assert trained_window(BACKBONE) == 128

def test_a_backbone_without_a_recorded_window_is_refused():
    with pytest.raises(ValueError, match='no trained input window'):
        trained_window('some/other-backbone')

def test_a_null_max_length_is_the_window_and_a_longer_one_is_refused():
    assert check_window(BACKBONE, None) == 128
    assert check_window(BACKBONE, 64) == 64
    assert check_window(BACKBONE, 128) == 128
    with pytest.raises(ValueError, match='exceeds the trained input window'):
        check_window(BACKBONE, 129)

def test_overflow_shares_count_present_texts_beyond_the_window():

    def words_and_two_special_tokens(texts):
        return [len(text.split()) + 2 for text in texts]

    shares = overflow_shares(
        {
            'title': ['a b', 'a b c d e'],
            'excluded': [None, '  ', 'a b c'],
            'examples': [None, None],
        },
        words_and_two_special_tokens,
        window=5,
    )

    assert list(shares) == ['title', 'excluded', 'examples']
    assert shares['title'] == {'present': 2, 'over': 1, 'share': 0.5}
    # 'a b c' is five tokens: at the window, not beyond it
    assert shares['excluded'] == {'present': 1, 'over': 0, 'share': 0.0}
    assert shares['examples'] == {'present': 0, 'over': 0, 'share': 0.0}

def test_token_counts_include_the_special_tokens(tmp_path):
    tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'soybean', 'farming']
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(tokens))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))

    assert token_counter(tokenizer)(['soybean farming', 'soybean']) == [4, 3]
