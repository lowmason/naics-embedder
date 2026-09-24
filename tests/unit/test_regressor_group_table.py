'''
``data regressor-groups``: the held-out draw, written once with its provenance (roadmap Stage 3).
'''

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.panels.regressor_splits import (
    assign_splits,
    codes_fingerprint,
    read_group_table,
    split_counts,
)
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig
from tests.fixtures.regressor_panel import BRANCH_RECORD, CODEBOOK, write_qcew_slices

pytestmark = pytest.mark.unit

@pytest.fixture
def draw_inputs(tmp_path, regressor_cells):
    pins = write_qcew_slices(tmp_path / 'qcew', regressor_cells)
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    cfg = RegressorPanelConfig(
        qcew_dir=str(tmp_path / 'qcew'),
        qcew_sha256=pins,
        codebook_codes_sha256=codes_fingerprint(CODEBOOK),
        heldout_groups_csv=str(tmp_path / 'conf' / 'regressor_heldout_groups.csv'),
        provenance_json=str(tmp_path / 'conf' / 'regressor_heldout_groups_provenance.json'),
        branch_record=RegressorBranchRecord(**BRANCH_RECORD),
    )
    return cfg, codebook

def test_the_draw_writes_the_table_and_its_provenance(draw_inputs, regressor_rows):
    cfg, codebook = draw_inputs

    path = generate_regressor_group_table(cfg, codebook)

    groups = read_group_table(path)
    # A fifth of 6, 1, 6 and 4 groups floors to 2; a fifth of 17, rounded, is 3
    assert len(groups) == 3
    provenance = json.loads(Path(cfg.provenance_json).read_text())
    assert provenance['heldout_groups'] == {
        'path': cfg.heldout_groups_csv,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'groups': 3,
    }
    assert provenance['groups_by_sector'] == {'11': 1, '31': 1, '52': 1}
    assert (provenance['seed'], provenance['fraction']) == (20260924, '1/5')
    assert provenance['six_digit_population'] == 32
    assert provenance['rows_by_level_and_split'] == {
        str(level): split_counts(assign_splits(rows, groups))
        for level, rows in regressor_rows.items()
    }
    assert provenance['qcew_sha256'] == cfg.qcew_sha256
    assert provenance['codebook_codes_sha256'] == cfg.codebook_codes_sha256
    assert {'generator_revision', 'library_versions', 'generated_at'} <= set(provenance)

def test_an_existing_table_is_redrawn_only_with_force(draw_inputs):
    cfg, codebook = draw_inputs
    path = generate_regressor_group_table(cfg, codebook)
    first = path.read_bytes()

    with pytest.raises(FileExistsError, match='drawn once'):
        generate_regressor_group_table(cfg, codebook)
    generate_regressor_group_table(cfg, codebook, force=True)

    assert path.read_bytes() == first

def test_the_draw_needs_the_branch_records_population(draw_inputs):
    cfg, codebook = draw_inputs
    wrong = RegressorBranchRecord(**{**BRANCH_RECORD, 'excluded_codes': []})

    with pytest.raises(ValueError, match='branch record mismatch'):
        generate_regressor_group_table(cfg.model_copy(update={'branch_record': wrong}), codebook)
    with pytest.raises(ValueError, match='no branch_record'):
        generate_regressor_group_table(cfg.model_copy(update={'branch_record': None}), codebook)
    assert not Path(cfg.heldout_groups_csv).exists()
