'''
A miniature set of the four Census NAICS source files, as ``load_naics_sources`` returns them.

Nine codes under one sector. 111110 and 111120 have index entries; 111191 has none, so it is a
decoding candidate only; 11119 has no entries either and falls back to its description's
illustrative examples. One entry of 111120 repeats its code's title, so it can never be held out,
and one sheet row is a "see" cross-reference row, which is not an entry.
'''

import polars as pl
import pytest

from naics_embedder.data.download_data import NaicsSources

TITLES = [
    ('11', 'Agriculture, Forestry, Fishing and Hunting'),
    ('111', 'Crop Production'),
    ('1111', 'Oilseed and Grain Farming'),
    ('11111', 'Soybean Farming'),
    ('111110', 'Soybean Farming'),
    ('11112', 'Oilseed (except Soybean) Farming'),
    ('111120', 'Oilseed (except Soybean) Farming'),
    ('11119', 'Other Grain Farming'),
    ('111191', 'Oilseed and Grain Combination Farming'),
]
DESCRIPTIONS = {
    '11': 'The Sector as a Whole\nThe Agriculture sector comprises farms and ranches.',
    '111': 'This subsector comprises establishments growing crops.',
    '1111': 'This industry group comprises establishments growing oilseeds and grains.',
    '11111': 'This industry comprises establishments growing soybeans.',
    '111110': 'This industry comprises establishments primarily engaged in growing soybeans.',
    '11112': 'This industry comprises establishments growing fibrous oilseed plants.',
    '111120': 'This industry comprises establishments growing oilseed plants except soybeans.',
    '11119': (
        'This industry comprises establishments growing grains not elsewhere classified.\n'
        'Illustrative Examples:\nBarley farming\nRye farming'
    ),
    '111191': 'This industry comprises establishments growing a combination of oilseeds and grains.',
}
# Sheet order is entry order: entry_id is the row position
INDEX_ROWS = [
    ('111110', 'Soybean farming, field and seed production'),
    ('111110', 'Edamame farming'),
    ('111110', 'Soybeans, organic'),
    ('111110', '  Soybean seed production '),
    ('111120', 'Canola farming'),
    ('111120', 'Flaxseed farming'),
    ('111120', 'Sunflower farming'),
    ('111120', 'Safflower farming'),
    ('******', 'Grain farming--see Industry Group 1111'),
    ('111120', 'Rapeseed farming'),
    ('111120', 'Oilseed (except soybean) farming'),
]
EXCLUSIONS = [
    (
        '111110',
        'Growing soybeans for green manure--are classified in Industry 111120, Oilseed (except '
        'Soybean) Farming.',
    ),
]

@pytest.fixture
def naics_sources() -> NaicsSources:
    return NaicsSources(
        titles=pl.DataFrame(
            {
                'index': [number for number, _ in enumerate(TITLES, start=1)],
                'code': [code for code, _ in TITLES],
                'title': [title for _, title in TITLES],
            },
            schema={
                'index': pl.UInt32,
                'code': pl.Utf8,
                'title': pl.Utf8
            },
        ),
        descriptions=pl.DataFrame(
            {
                'code': list(DESCRIPTIONS),
                'description': list(DESCRIPTIONS.values()),
            }
        ),
        index=pl.DataFrame(
            {
                'code': [code for code, _ in INDEX_ROWS],
                'examples': [text for _, text in INDEX_ROWS],
            }
        ),
        exclusions=pl.DataFrame(
            {
                'code': [code for code, _ in EXCLUSIONS],
                'excluded': [text for _, text in EXCLUSIONS],
            }
        ),
    )
