'''
The committed index-entry role table: the outcome panel's real sealed splits (Req 3; roadmap D4).

These read only ``conf/data/index_roles.csv`` and its provenance, so they run in CI without the
Census files. The pinned hash fails any accidental redraw: redrawing unseals the splits.
'''

import json
from pathlib import Path

import polars as pl
import pytest

from naics_embedder.panels.index_roles import read_role_table, role_table_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

pytestmark = pytest.mark.unit

TABLE = Path('conf/data/index_roles.csv')
PROVENANCE = Path('conf/data/index_roles_provenance.json')
TABLE_SHA256 = '05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a'

@pytest.fixture(scope='module')
def roles() -> pl.DataFrame:
    return read_role_table(TABLE)

@pytest.fixture(scope='module')
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text())

def test_every_index_entry_holds_exactly_one_role(roles):
    assert roles.height == 20_373
    assert roles.get_column('entry_id').n_unique() == 20_373
    assert roles.get_column('code').n_unique() == 1_010
    assert set(roles.get_column('role').unique()) == {'examples', 'training', 'validation', 'test'}

def test_the_entryless_codes_are_never_queries(roles):
    assert roles.filter(pl.col('code').is_in(['112130', '541120'])).height == 0

def test_every_code_keeps_an_examples_channel_entry(roles):
    with_examples = roles.filter(pl.col('role') == 'examples').get_column('code').n_unique()

    assert with_examples == roles.get_column('code').n_unique()

def test_the_table_is_the_one_its_provenance_describes(roles, provenance):
    assert sha256_file(TABLE) == TABLE_SHA256
    assert role_table_fingerprint(roles) == TABLE_SHA256
    assert provenance['role_table']['sha256'] == TABLE_SHA256
    assert dict(roles.group_by('role').len().iter_rows()) == provenance['roles']

def test_held_out_queries_were_checked_against_training_text(provenance):
    assert provenance['roles'] == {
        'examples': 6_118,
        'training': 7_200,
        'validation': 4_042,
        'test': 3_013,
    }
    assert provenance['eligibility']['withheld_exact'] == 1_539
    assert provenance['eligibility']['withheld_near_duplicate'] == 2_516
    assert provenance['held_out_leakage'] == {
        'validation': {
            'exact': 0,
            'near_duplicate': 0
        },
        'test': {
            'exact': 0,
            'near_duplicate': 0
        },
    }
