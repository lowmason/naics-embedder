'''
Drawing the frozen index-entry role table on the miniature Census sources (Req 3; roadmap D4).

Per-code quotas are worked out by hand: 111110 has 4 entries, all eligible, so E1 TR1 V1 TE1;
111120 has 6 entries, one of which repeats its title and can never be held out, so E2 TR2 V1 TE1.
'''

import json

import polars as pl
import pytest

from naics_embedder.data import index_role_table
from naics_embedder.panels.index_roles import read_role_table, role_table_fingerprint
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig

pytestmark = pytest.mark.unit

@pytest.fixture
def configs(tmp_path, monkeypatch, naics_sources):
    monkeypatch.setattr(index_role_table, 'load_naics_sources', lambda cfg: naics_sources)
    download_cfg = DownloadConfig(index_roles_csv=str(tmp_path / 'conf' / 'index_roles.csv'))
    panel_cfg = OutcomePanelConfig(provenance_json=str(tmp_path / 'conf' / 'provenance.json'))
    return download_cfg, panel_cfg

def test_every_entry_gets_one_role_in_per_code_quotas(configs):
    download_cfg, panel_cfg = configs

    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    roles = read_role_table(table_path)

    assert roles.get_column('entry_id').to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    counts = {
        code: dict(frame.group_by('role').len().sort('role').iter_rows())
        for (code, ), frame in roles.group_by('code')
    }
    assert counts == {
        '111110': {
            'examples': 1,
            'test': 1,
            'training': 1,
            'validation': 1
        },
        '111120': {
            'examples': 2,
            'test': 1,
            'training': 2,
            'validation': 1
        },
    }
    # Entry 10 repeats its code's title, so it is never held out
    assert roles.filter(pl.col('entry_id') == 10)['role'][0] in ('examples', 'training')

def test_provenance_records_the_draw_and_its_checks(configs):
    download_cfg, panel_cfg = configs

    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    provenance = json.loads(open(panel_cfg.provenance_json).read())

    assert provenance['role_table'] == {'path': str(table_path), 'sha256': sha256_file(table_path)}
    assert provenance['role_table']['sha256'] == role_table_fingerprint(read_role_table(table_path))
    assert provenance['index_file']['sha256'] == download_cfg.index_sha256
    assert provenance['seed'] == 20260924
    assert provenance['fractions'] == {
        'examples': '3/10',
        'training': '7/20',
        'validation': '1/5',
        'test': '3/20',
    }
    assert provenance['near_duplicate_min_jaccard'] == '9/10'
    assert provenance['eligibility'] == {
        'entries': 10,
        'exact_static': 1,
        'near_duplicate_static': 1,
        'exact_entry': 0,
        'near_duplicate_entry': 0,
        'withheld_exact': 1,
        'withheld_near_duplicate': 0,
        'withheld': 1,
        'eligible': 9,
    }
    assert provenance['roles'] == {'examples': 3, 'training': 3, 'validation': 2, 'test': 2}
    assert provenance['codes_with_role'] == {
        'examples': 2,
        'training': 2,
        'validation': 2,
        'test': 2,
    }
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

def test_the_table_is_drawn_once(configs):
    download_cfg, panel_cfg = configs
    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    first = sha256_file(table_path)

    with pytest.raises(FileExistsError, match='--force'):
        index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    index_role_table.generate_index_role_table(download_cfg, panel_cfg, force=True)

    # The same seed redraws the same table
    assert sha256_file(table_path) == first
