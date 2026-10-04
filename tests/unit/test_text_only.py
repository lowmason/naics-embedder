'''
The text-only comparator (Req 2; roadmap D9): the arm's backbone, frozen, reading the arm's text.

A one-layer BERT with a seventeen-word vocabulary stands in for the backbone, so these run
offline and fast; the real table uses the backbone from the local Hugging Face cache.
'''

import hashlib
import json

import numpy as np
import polars as pl
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from naics_embedder.panels import text_only, window_summaries
from naics_embedder.panels.text_only import (
    CHANNELS,
    build_text_only_table,
    encode_code_texts,
    pca_reduce,
    provenance_path,
    text_only_fingerprint,
)
from naics_embedder.panels.window_summaries import summaries_identity, text_sha256
from naics_embedder.utils.input_window import TRAINED_WINDOWS
from tests.fixtures.window_summaries import pin_artifact

pytestmark = pytest.mark.unit

WORDS = [
    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'soybean', 'farming', 'grows', 'soybeans',
    'oilseed', 'canola', 'cattle', 'raises', 'not', 'here', ',', '.'
]

@pytest.fixture(scope='module')
def tokenizer(tmp_path_factory):
    vocab = tmp_path_factory.mktemp('backbone') / 'vocab.txt'
    vocab.write_text('\n'.join(WORDS) + '\n')
    return BertTokenizerFast(vocab_file=str(vocab))

@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    return BertModel(config)

def _descriptions(rows):
    return pl.DataFrame(
        rows,
        schema={
            'code': pl.Utf8,
            **{
                channel: pl.Utf8
                for channel in CHANNELS
            }
        },
        orient='row',
    )

def _encode(frame, model, tokenizer, batch_size=32):
    return encode_code_texts(frame, model, tokenizer, max_length=16, batch_size=batch_size)

def test_a_codes_vector_is_the_mean_of_its_present_channels(model, tokenizer):
    title = _encode(
        _descriptions([('111110', 'soybean farming', None, None, None)]), model, tokenizer
    )
    description = _encode(
        _descriptions([('111110', None, 'grows soybeans', None, None)]), model, tokenizer
    )

    # A blank channel is absent: masked out, never encoded as a placeholder
    both = _encode(
        _descriptions([('111110', 'soybean farming', 'grows soybeans', '   ', None)]), model,
        tokenizer
    )

    np.testing.assert_allclose(both, (title + description) / 2, atol=1e-6)

def test_the_batch_size_does_not_change_the_vectors(model, tokenizer):
    frame = _descriptions(
        [
            ('111110', 'soybean farming', 'grows soybeans here .', None, 'soybean'),
            ('111120', 'oilseed farming', 'grows canola , not soybeans .', 'soybean farming', None),
            ('112111', 'cattle', 'raises cattle', None, None),
        ]
    )

    one = _encode(frame, model, tokenizer, batch_size=1)
    many = _encode(frame, model, tokenizer, batch_size=32)

    assert one.shape == (3, 8)
    assert one.dtype == np.float64
    np.testing.assert_allclose(one, many, atol=1e-5)

def test_a_code_with_no_text_is_refused(model, tokenizer):
    frame = _descriptions(
        [('111110', 'soybean farming', None, None, None), ('111120', None, ' ', None, None)]
    )

    with pytest.raises(ValueError, match='at least one present text channel'):
        _encode(frame, model, tokenizer)

def test_the_backbone_stays_frozen(model, tokenizer):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()

    _encode(_descriptions([('111110', 'soybean farming', None, None, None)]), model, tokenizer)

    assert not model.training
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())

def test_the_table_and_its_provenance_are_written(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 16)
    descriptions = tmp_path / 'naics_descriptions.parquet'
    _descriptions(
        [
            ('111120', 'oilseed farming', 'grows canola', None, None),
            ('111110', 'soybean farming', 'grows soybeans', None, 'soybean'),
        ]
    ).write_parquet(descriptions)
    output = tmp_path / 'text_only.parquet'

    path = build_text_only_table(
        descriptions,
        output,
        backbone='tiny-bert',
        max_length=16,
        batch_size=1,
        model=model,
        tokenizer=tokenizer,
        revision='abc123',
    )

    table = pl.read_parquet(path)
    assert table.columns == ['code', *[f't{index}' for index in range(8)]]
    assert table.get_column('code').to_list() == ['111110', '111120']
    provenance = json.loads(provenance_path(path).read_text())
    assert provenance_path(path).name == 'text_only_provenance.json'
    assert provenance['backbone'] == 'tiny-bert'
    assert provenance['revision'] == 'abc123'
    assert provenance['channels'] == list(CHANNELS)
    assert provenance['max_length'] == 16
    assert (provenance['codes'], provenance['hidden_size']) == (2, 8)
    assert provenance['table_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance['descriptions']['sha256'] == hashlib.sha256(descriptions.read_bytes()
                                                                  ).hexdigest()
    assert set(provenance['library_versions']) == {'torch', 'transformers', 'polars'}
    # The name a regressor read logs the table by, so a logged read matches this file
    assert provenance['matrix_fingerprint'] == text_only_fingerprint(table)

def test_the_table_reads_an_over_window_text_as_its_summary(
    tmp_path, monkeypatch, model, tokenizer
):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 16)
    # 19 tokens marked, over the 16-token window; the summary is 10
    text = 'grows soybeans. grows canola. raises cattle. not here. soybean farming.'
    summary = 'grows soybeans. raises cattle.'
    row = {
        'code': '111110',
        'channel': 'description',
        'source_sha256': text_sha256(text),
        'window': 16,
        'summary': summary,
        'source_tokens': 19,
        'summary_tokens': 10,
        'units_kept': 2,
        'units_total': 5,
    }
    pin = pin_artifact(tmp_path / 'window_summaries.csv', [row], window=16)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, 'tiny-bert', pin)
    descriptions = tmp_path / 'naics_descriptions.parquet'
    _descriptions([('111110', 'soybean farming', text, None, None)]).write_parquet(descriptions)

    path = build_text_only_table(
        descriptions,
        tmp_path / 'text_only.parquet',
        backbone='tiny-bert',
        max_length=16,
        model=model,
        tokenizer=tokenizer,
    )

    expected = _encode(
        _descriptions([('111110', 'soybean farming', summary, None, None)]), model, tokenizer
    )
    np.testing.assert_array_equal(pl.read_parquet(path).drop('code').to_numpy(), expected)
    assert json.loads(provenance_path(path).read_text())['summaries'] == pin.sha256

def test_the_provenance_records_the_backbones_summaries(text_only_comparator_table):
    provenance = json.loads(provenance_path(text_only_comparator_table).read_text())

    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(provenance['backbone'])
    assert provenance['summaries'] is not None

def test_a_max_length_beyond_the_trained_window_is_refused(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 8)
    output = tmp_path / 'text_only.parquet'

    # The window is checked before the descriptions are read
    with pytest.raises(ValueError, match='trained input window'):
        build_text_only_table(
            tmp_path / 'naics_descriptions.parquet',
            output,
            backbone='tiny-bert',
            max_length=16,
            model=model,
            tokenizer=tokenizer,
        )
    assert not output.exists()

def test_the_backbone_is_read_from_the_local_cache_only(monkeypatch, model, tokenizer):
    import transformers

    calls = []

    def fake(loaded):

        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return loaded

        return from_pretrained

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', fake(tokenizer))
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', fake(model))
    model.config._commit_hash = 'cafe'

    loaded_model, loaded_tokenizer, revision = text_only.load_backbone('some/backbone')

    assert calls == [('some/backbone', {'local_files_only': True})] * 2
    assert (loaded_model, loaded_tokenizer, revision) == (model, tokenizer, 'cafe')
    assert not loaded_model.training

def test_pca_reduces_to_the_arms_dimension():
    vectors = np.random.default_rng(0).normal(size=(10, 6))

    assert pca_reduce(vectors, 3).shape == (10, 3)
    for dimension in (0, 7, 11):
        with pytest.raises(ValueError, match='cannot reduce'):
            pca_reduce(vectors, dimension)
