'''
Draw the regressor panel's held-out four-digit groups (roadmap Stage 3; Req 2; Req 4).

``naics-embedder data regressor-groups`` runs this once. The table it writes
(``conf/data/regressor_heldout_groups.csv``) is committed, and the panel reads it from then on,
so the held-out regime's outer set never moves. The seen regime's outer set leaves out the
held-out groups too, so a redraw moves both outer sets, and its new fingerprint would not count
as a reopening: an existing table is replaced only with ``force``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

from naics_embedder.data.supervision_bundle import generator_revision
from naics_embedder.panels.qcew_rows import (
    level_cells,
    load_national_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    LEVELS,
    require_branch_record,
    verify_branch_record,
)
from naics_embedder.panels.regressor_splits import (
    SECTOR_LEVEL,
    ancestor_at,
    assign_splits,
    draw_heldout_groups,
    read_codebook_codes,
    split_counts,
    write_group_table,
)
from naics_embedder.utils.config import RegressorPanelConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_regressor_group_table(
    cfg: RegressorPanelConfig,
    codebook_path: Path,
    *,
    force: bool = False,
) -> Path:
    '''
    Draw the held-out four-digit groups and write the table and its provenance.

    Returns:
        The table's path (``cfg.heldout_groups_csv``).

    Raises:
        FileExistsError: If the table exists and ``force`` is False.
        ValueError: If a pinned hash differs or the data are not the branch record's population.
    '''

    table_path = Path(cfg.heldout_groups_csv)
    if table_path.exists() and not force:
        raise FileExistsError(
            f'{table_path} exists: the held-out groups are drawn once and committed. Redrawing '
            'moves both regressor outer sets; pass --force only to do that deliberately.'
        )
    record = require_branch_record(cfg)
    codes = read_codebook_codes(Path(codebook_path), cfg.codebook_codes_sha256)
    cells = load_national_cells(Path(cfg.qcew_dir), cfg.qcew_sha256)
    six_digit = population(level_cells(cells, codes, DECISION_LEVEL))
    verify_branch_record(record.model_dump(), codes, six_digit)

    fraction = Fraction(str(cfg.heldout_fraction))
    groups = draw_heldout_groups(six_digit, fraction, cfg.seed)
    fingerprint = write_group_table(groups, table_path)

    partition: Dict[str, Dict[str, int]] = {}
    for level in LEVELS:
        cells_at_level = level_cells(cells, codes, level)
        rows = assign_splits(panel_rows(cells_at_level, population(cells_at_level)), groups)
        partition[str(level)] = split_counts(rows)
    by_sector: Dict[str, int] = {}
    for group in groups:
        sector = ancestor_at(group, SECTOR_LEVEL)
        by_sector[sector] = by_sector.get(sector, 0) + 1

    provenance: Dict[str, Any] = {
        'heldout_groups': {
            'path': str(table_path),
            'sha256': fingerprint,
            'groups': len(groups)
        },
        'seed': cfg.seed,
        'fraction': str(fraction),
        'groups_by_sector': dict(sorted(by_sector.items())),
        'six_digit_population': len(six_digit),
        'rows_by_level_and_split': partition,
        'qcew_sha256': dict(sorted(cfg.qcew_sha256.items())),
        'codebook_codes_sha256': cfg.codebook_codes_sha256,
        'generator_revision': generator_revision(),
        'library_versions': {
            name: version(name)
            for name in ('numpy', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = Path(cfg.provenance_json)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')

    logger.info(f'Held-out groups: {len(groups)} ({fingerprint}) written to {table_path}')
    logger.info(f'Rows by level and split: {partition}')
    logger.info(f'Provenance written to: {provenance_path}\n')
    return table_path
