'''
Building the window-fitting summaries (roadmap Stage 6b): the centrality selection (spec 4.3), the
artifact's rows (4.5) and the build (4.6), with a stub tokenizer and a stub embedder.
'''

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from naics_embedder.data import window_summaries as build
from naics_embedder.data.window_summaries import (
    Selection,
    generate_window_summaries,
    select_units,
    summary_rows,
)
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import (
    NO_BREAK_PATTERN,
    SummariesPin,
    read_window_summaries,
    resolve_channel_texts,
    text_sha256,
)
from naics_embedder.utils.input_window import TRAINED_WINDOWS
from tests.fixtures.window_summaries import WordTokenizer

pytestmark = pytest.mark.unit

WINDOW = 10
STUB = 'stub-backbone'
# 'description: ' and three three-word sentences: 12 tokens with [CLS] and [SEP], over 10
LONG = 'Farms grow corn. Farms grow wheat. Farms sell grain.'
# 'examples: ' and eight entries: 11 tokens
EIGHT = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
TEXT_SCHEMA = {name: pl.Utf8 for name in ('code', 'title', 'description', 'examples', 'excluded')}

# Unit vectors of the stub embedder. The description's target is (2, 1) up to scale: wheat is the
# most central unit (sell grain ties it and loses to the earlier unit), then corn raises the
# cosine most. The examples' target is (1, 1): soybeans, then beans, which reaches cosine 1.
VECTORS = {
    'Farms grow corn.': [0.0, 1.0],
    'Farms grow wheat.': [1.0, 0.0],
    'Farms sell grain.': [1.0, 0.0],
    'Soybeans': [1.0, 0.0],
    'Beans': [0.0, 1.0],
    'Corn': [1.0, 0.0],
    'Wheat': [0.0, 1.0],
    'Rice': [1.0, 0.0],
    'Oats': [0.0, 1.0],
    'Rye': [1.0, 0.0],
    'Barley': [0.0, 1.0],
}

def stub_embed(units):
    return np.array([VECTORS[unit] for unit in units])

def descriptions_frame(**columns):
    texts = {
        'code': ['111110', '111120'],
        'title': ['Soybean Farming', 'Oilseed Farming'],
        'description': [LONG, 'Farms grow oilseeds.'],
        'examples': [EIGHT, None],
        'excluded': [None, None],
    }
    texts.update(columns)
    return pl.DataFrame(texts, schema=TEXT_SCHEMA)

# -------------------------------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------------------------------

def fits_at_most(count):
    '''A stub fit: a candidate fits when it has at most ``count`` units.'''

    return lambda candidates: [len(candidate) <= count for candidate in candidates]

def test_the_greedy_adds_the_unit_that_most_raises_the_cosine_while_one_fits():
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    # Target (2, 1): unit 1 first, then unit 0; with three units allowed, unit 2 raises it to 1
    assert select_units(vectors, np.ones(3), fits_at_most(2)) == Selection([0, 1], plateau=False)
    assert select_units(vectors, np.ones(3), fits_at_most(3)) == Selection([0, 1, 2], False)

def test_an_exact_tie_goes_to_the_earlier_unit():
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    assert select_units(vectors, np.ones(3), fits_at_most(1)) == Selection([1], plateau=False)

def test_the_selection_stops_when_no_unit_raises_the_cosine():
    # Target (3, 0): unit 0 alone has cosine 1, and either other unit lowers it
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    weights = np.array([3.0, 1.0, 1.0])

    assert select_units(vectors, weights, fits_at_most(3)) == Selection([0], plateau=True)

def test_a_text_with_no_unit_that_fits_alone_is_refused():
    with pytest.raises(ValueError, match='no unit fits the window on its own'):
        select_units(np.eye(2), np.ones(2), fits_at_most(0))

# -------------------------------------------------------------------------------------------------
# Rows
# -------------------------------------------------------------------------------------------------

def test_a_summary_keeps_whole_units_in_source_order_and_fits_with_its_marker():
    rows, plateau = summary_rows(descriptions_frame(), WordTokenizer(), stub_embed, window=WINDOW)

    assert rows.to_dicts() == [
        {
            'code': '111110',
            'channel': 'description',
            'source_sha256': text_sha256(LONG),
            'window': WINDOW,
            # Picked wheat first, then corn; emitted in source order
            'summary': 'Farms grow corn. Farms grow wheat.',
            'source_tokens': 12,
            'summary_tokens': 9,
            'units_kept': 2,
            'units_total': 3,
        },
        {
            'code': '111110',
            'channel': 'examples',
            'source_sha256': text_sha256(EIGHT),
            'window': WINDOW,
            'summary': 'Soybeans; Beans',
            'source_tokens': 11,
            'summary_tokens': 5,
            'units_kept': 2,
            'units_total': 8,
        },
    ]
    assert plateau == {'description': 0, 'examples': 1, 'excluded': 0}

def test_identical_texts_get_one_summary_and_each_unit_is_embedded_once():
    embedded = []

    def counting_embed(units):
        embedded.append(list(units))
        return stub_embed(units)

    frame = descriptions_frame(description=[LONG, LONG], examples=[None, None])
    rows, _ = summary_rows(frame, WordTokenizer(), counting_embed, window=WINDOW)

    assert rows.select('code', 'summary').rows() == [
        ('111110', 'Farms grow corn. Farms grow wheat.'),
        ('111120', 'Farms grow corn. Farms grow wheat.'),
    ]
    assert embedded == [['Farms grow corn.', 'Farms grow wheat.', 'Farms sell grain.']]

def test_a_title_over_the_window_cannot_be_summarized():
    frame = descriptions_frame(title=['Farms that grow soybeans and other oilseed crops', 'Oil'])

    with pytest.raises(ValueError, match='a title over the window cannot be summarized'):
        summary_rows(frame, WordTokenizer(), stub_embed, window=WINDOW)

# -------------------------------------------------------------------------------------------------
# The build
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def stub_backbone(monkeypatch):
    '''A backbone with a 10-token window whose units the stub embedder reads.'''

    monkeypatch.setitem(TRAINED_WINDOWS, STUB, WINDOW)
    monkeypatch.setattr(build, 'backbone_embedder', lambda model, tokenizer: stub_embed)

@pytest.fixture
def descriptions_path(tmp_path):
    path = tmp_path / 'naics_descriptions.parquet'
    descriptions_frame().write_parquet(path)
    return path

def generate(tmp_path, descriptions_path, **options):
    return generate_window_summaries(
        descriptions_path,
        tmp_path / 'conf' / 'window_summaries.csv',
        backbone=STUB,
        model=object(),
        tokenizer=WordTokenizer(),
        revision='stub-revision',
        **options,
    )

def test_the_build_writes_a_checked_artifact_and_its_provenance(
    tmp_path, stub_backbone, descriptions_path
):
    pin = generate(tmp_path, descriptions_path)

    artifact = tmp_path / 'conf' / 'window_summaries.csv'
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert pin == SummariesPin(path=str(artifact), sha256=sha256, window=WINDOW)
    assert read_window_summaries(artifact).select('channel', 'summary').rows() == [
        ('description', 'Farms grow corn. Farms grow wheat.'),
        ('examples', 'Soybeans; Beans'),
    ]
    resolve_channel_texts(descriptions_frame(), WordTokenizer(), STUB, WINDOW, pin=pin)
    assert sorted(path.name for path in artifact.parent.iterdir()) == [
        'window_summaries.csv',
        'window_summaries_provenance.json',
    ]

    provenance = json.loads(provenance_path(artifact).read_text())
    assert provenance['artifact_sha256'] == sha256
    assert provenance['descriptions'] == {
        'path': str(descriptions_path),
        'sha256': hashlib.sha256(descriptions_path.read_bytes()).hexdigest(),
    }
    assert [provenance[key] for key in ('backbone', 'revision', 'tokenizer', 'window')] == [
        STUB,
        'stub-revision',
        STUB,
        WINDOW,
    ]
    assert provenance['budget'] == {channel: 7 for channel in TEXT_SCHEMA if channel != 'code'}
    assert provenance['units'] == {
        'rule': 'sentence-clause-piece-v1',
        'no_break_pattern': NO_BREAK_PATTERN
    }
    assert provenance['selection']['rule'] == 'centrality-v1'
    assert provenance['channels']['description'] == {
        'present': 2,
        'over_window': 1,
        'summarized': 1,
        'mean_kept_share': 0.75,
        'min_summary_tokens': 9,
        'p10_summary_tokens': 9,
        'plateau_stops': 0,
    }
    assert provenance['channels']['examples']['plateau_stops'] == 1
    assert provenance['channels']['title']['summarized'] == 0

def test_a_backbone_with_no_recorded_window_is_refused(tmp_path, descriptions_path):
    # No stub_backbone: the stub has no entry in TRAINED_WINDOWS
    with pytest.raises(ValueError, match="no trained input window is recorded for 'stub-backbone'"):
        generate(tmp_path, descriptions_path)
    assert not (tmp_path / 'conf').exists()

def test_an_existing_artifact_is_kept_without_force(tmp_path, stub_backbone, descriptions_path):
    first = generate(tmp_path, descriptions_path)

    with pytest.raises(FileExistsError, match='--force'):
        generate(tmp_path, descriptions_path)

    assert generate(tmp_path, descriptions_path, force=True) == first

def test_a_failing_check_leaves_no_artifact(
    tmp_path, stub_backbone, descriptions_path, monkeypatch
):

    def reordered(*args, **kwargs):
        rows, plateau = summary_rows(*args, **kwargs)
        swapped = pl.when(pl.col('channel') == 'description').then(
            pl.lit('Farms grow wheat. Farms grow corn.')
        ).otherwise(pl.col('summary'))
        return rows.with_columns(swapped.alias('summary')), plateau

    monkeypatch.setattr(build, 'summary_rows', reordered)

    with pytest.raises(ValueError, match="code 111110's description: the summary is not an"):
        generate(tmp_path, descriptions_path)

    assert list((tmp_path / 'conf').iterdir()) == []
