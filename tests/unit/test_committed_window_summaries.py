'''
The committed window-fitting summaries (roadmap Stage 6b): the artifact MiniLM's pin names, and,
where the real descriptions and the cached tokenizer are present, the resolver accepting it.

Every test here reads the committed pin, so the module opts out of the dummy-pin seam
(``tests/conftest.py``).
'''

import hashlib
import json
import logging
from pathlib import Path

import polars as pl
import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import (
    SUMMARY_CHANNELS,
    WINDOW_SUMMARIES_PATH,
    read_window_summaries,
    resolve_channel_texts,
)

pytestmark = [pytest.mark.unit, pytest.mark.real_window_summaries]

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
DESCRIPTIONS = Path('data/naics_descriptions.parquet')

@pytest.fixture
def pin():
    return window_summaries.WINDOW_SUMMARIES[MINILM]

@pytest.fixture
def provenance(pin):
    return json.loads(provenance_path(Path(pin.path)).read_text())

def test_the_pin_names_the_committed_artifact(pin, provenance):
    assert pin.path == WINDOW_SUMMARIES_PATH
    assert hashlib.sha256(Path(pin.path).read_bytes()).hexdigest() == pin.sha256
    assert pin.window == 128
    assert provenance['artifact_sha256'] == pin.sha256
    assert (provenance['backbone'], provenance['window']) == (MINILM, pin.window)

def test_each_summarized_text_has_one_row_that_fits_the_window(pin):
    rows = read_window_summaries(Path(pin.path))

    assert rows.select('code', 'channel').is_duplicated().sum() == 0
    assert set(rows.get_column('channel').unique().to_list()) <= set(SUMMARY_CHANNELS)
    assert (rows.get_column('window') == pin.window).all()
    assert (rows.get_column('summary_tokens') <= pin.window).all()
    assert (rows.get_column('source_tokens') > pin.window).all()
    assert (rows.get_column('units_kept') <= rows.get_column('units_total')).all()

def test_the_provenance_counts_every_over_window_text_as_summarized(provenance):
    channels = provenance['channels']

    assert channels['title']['over_window'] == 0
    for channel in SUMMARY_CHANNELS:
        assert channels[channel]['summarized'] == channels[channel]['over_window'] > 0

def test_the_resolver_accepts_the_artifact_on_the_real_descriptions(pin, provenance, caplog):
    '''
    Local only: the resolver re-tokenizes every row and re-checks the source hashes, S5, the
    segment subset and the fit (spec 4.7).
    '''

    if not DESCRIPTIONS.is_file():
        pytest.skip(f'{DESCRIPTIONS} is not here')
    if hashlib.sha256(DESCRIPTIONS.read_bytes()).hexdigest() != provenance['descriptions']['sha256']:
        pytest.skip(f'{DESCRIPTIONS} is not the descriptions the summaries were built from')
    try:
        tokenizer = AutoTokenizer.from_pretrained(MINILM, local_files_only=True)
    except OSError:
        pytest.skip(f"{MINILM}'s tokenizer is not in the local Hugging Face cache")

    with caplog.at_level(logging.INFO, logger='naics_embedder.panels.window_summaries'):
        resolve_channel_texts(pl.read_parquet(DESCRIPTIONS), tokenizer, MINILM, pin.window)

    assert "{'description': 162, 'examples': 106, 'excluded': 485}" in caplog.text
