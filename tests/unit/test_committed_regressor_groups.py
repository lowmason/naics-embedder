'''
The committed held-out groups: the regressor panel's real outer sets (Req 2; Req 4; Stage 3).

These read only ``conf/data/regressor_heldout_groups.csv`` and its provenance, so they run in CI
without the QCEW files. The pinned hash fails any accidental redraw: a redraw moves both outer
sets.
'''

import json
from collections import Counter
from pathlib import Path

import pytest

from naics_embedder.panels.regressor_splits import (
    SECTOR_LEVEL,
    ancestor_at,
    group_table_fingerprint,
    read_group_table,
)
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import RegressorPanelConfig, load_config

pytestmark = pytest.mark.unit

TABLE = Path('conf/data/regressor_heldout_groups.csv')
PROVENANCE = Path('conf/data/regressor_heldout_groups_provenance.json')
TABLE_SHA256 = 'deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4'

@pytest.fixture(scope='module')
def groups():
    return read_group_table(TABLE)

@pytest.fixture(scope='module')
def provenance():
    return json.loads(PROVENANCE.read_text())

def test_the_committed_table_is_the_recorded_draw(groups, provenance):
    assert sha256_file(TABLE) == TABLE_SHA256
    assert group_table_fingerprint(groups) == TABLE_SHA256
    assert provenance['heldout_groups'] == {
        'path': 'conf/data/regressor_heldout_groups.csv',
        'sha256': TABLE_SHA256,
        'groups': 60,
    }

def test_a_fifth_of_each_sectors_groups_is_held_out(groups, provenance):
    by_sector = Counter(ancestor_at(group, SECTOR_LEVEL) for group in groups)

    assert dict(sorted(by_sector.items())) == provenance['groups_by_sector']
    assert provenance['groups_by_sector'] == {
        '11': 4,
        '21': 1,
        '22': 1,
        '23': 2,
        '31': 17,
        '42': 4,
        '44': 5,
        '48': 6,
        '51': 2,
        '52': 2,
        '53': 2,
        '54': 2,
        '56': 2,
        '61': 1,
        '62': 3,
        '71': 2,
        '72': 1,
        '81': 3,
    }
    assert (provenance['seed'], provenance['fraction']) == (20260924, '1/5')
    assert provenance['six_digit_population'] == 980

def test_the_draw_used_the_configured_inputs(provenance):
    cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')

    assert provenance['qcew_sha256'] == cfg.qcew_sha256
    assert provenance['codebook_codes_sha256'] == cfg.codebook_codes_sha256
    assert provenance['seed'] == cfg.seed

def test_the_partition_counts_are_recorded(provenance):
    assert provenance['rows_by_level_and_split'] == {
        '2': {
            'remainder': 2,
            'seen_outer': 1,
            'heldout_outer': 54
        },
        '3': {
            'remainder': 92,
            'seen_outer': 46,
            'heldout_outer': 126
        },
        '4': {
            'remainder': 480,
            'seen_outer': 240,
            'heldout_outer': 180
        },
        '5': {
            'remainder': 1046,
            'seen_outer': 523,
            'heldout_outer': 405
        },
        '6': {
            'remainder': 1554,
            'seen_outer': 777,
            'heldout_outer': 609
        },
    }
