'''
Draw the frozen index-entry role table (roadmap Stage 2; Req 3; D4).

``naics-embedder data roles`` runs this once. The table it writes (``conf/data/index_roles.csv``)
is committed and ``data preprocess`` applies it from then on, so the sealed validation and test
splits never move. Redrawing reassigns every entry and unseals both splits, so an existing table
is replaced only with ``force``.
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

import polars as pl

from naics_embedder.data.download_data import (
    build_descriptions,
    load_naics_sources,
    naics_index_entries,
)
from naics_embedder.data.supervision_bundle import generator_revision
from naics_embedder.panels.index_roles import (
    ROLE_ORDER,
    RoleFractions,
    assign_index_roles,
    attach_role_text,
    held_out_eligibility,
    verify_examples_channel,
    verify_role_leakage,
    write_role_table,
)
from naics_embedder.panels.leakage import training_text_segments
from naics_embedder.supervision.artifacts import validate_index_role_table
from naics_embedder.supervision.schema import INDEX_ROLES_SCHEMA_VERSION, IndexRole
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_index_role_table(
    download_cfg: DownloadConfig,
    panel_cfg: OutcomePanelConfig,
    *,
    force: bool = False,
) -> Path:
    '''
    Draw every index entry's role, check the result, and write the table and its provenance.

    Held-out (validation and test) queries are drawn only from entries that match no title,
    description, exclusion text, fallback examples text or other index entry, exactly or as a
    near-duplicate; the provenance counts the entries withheld for each reason.

    Returns:
        The role table's path (``download_cfg.index_roles_csv``).

    Raises:
        FileExistsError: If the table exists and ``force`` is False.
        ValueError: If the drawn roles fail a Req 3 check.
    '''

    table_path = Path(download_cfg.index_roles_csv)
    if table_path.exists() and not force:
        raise FileExistsError(
            f'{table_path} exists: the role table is drawn once and committed. Redrawing it '
            'reassigns every entry and unseals the validation and test splits; pass --force only '
            'to do that deliberately.'
        )
    fractions = RoleFractions.from_mapping(panel_cfg.fractions)
    min_jaccard = Fraction(str(panel_cfg.near_duplicate_min_jaccard))

    sources = load_naics_sources(download_cfg)
    entries = naics_index_entries(sources)

    # Training text whatever the roles: titles, descriptions, exclusion text, and the illustrative
    # examples of codes without index entries
    static = build_descriptions(sources, entries.clear())
    eligibility = held_out_eligibility(entries, training_text_segments(static), min_jaccard)
    roles = assign_index_roles(
        entries, eligibility.eligible, fractions, panel_cfg.seed, panel_cfg.examples_floor
    )

    role_rows = attach_role_text(roles, entries)
    descriptions = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value)
    )
    validate_index_role_table(
        role_rows,
        descriptions.filter(pl.col('level') == 6).get_column('code').to_list(),
        min_examples_per_code=panel_cfg.examples_floor,
    )
    verify_examples_channel(descriptions, role_rows)
    leakage = verify_role_leakage(descriptions, role_rows, min_jaccard)

    fingerprint = write_role_table(roles, table_path)
    provenance = _provenance(
        download_cfg, panel_cfg, fractions, min_jaccard, eligibility.counts, roles, leakage
    )
    provenance['role_table'] = {'path': str(table_path), 'sha256': fingerprint}
    provenance_path = Path(panel_cfg.provenance_json)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')

    logger.info(f'Index-entry roles: {provenance["roles"]}')
    logger.info(f'Held-out eligibility: {eligibility.counts}')
    logger.info(f'Role table ({fingerprint}) written to: {table_path}')
    logger.info(f'Provenance written to: {provenance_path}\n')
    return table_path

def _provenance(
    download_cfg: DownloadConfig,
    panel_cfg: OutcomePanelConfig,
    fractions: RoleFractions,
    min_jaccard: Fraction,
    eligibility: Dict[str, int],
    roles: pl.DataFrame,
    leakage: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    by_role = {role.value: roles.filter(pl.col('role') == role.value) for role in ROLE_ORDER}
    return {
        'schema_version': INDEX_ROLES_SCHEMA_VERSION,
        'index_file': {
            'url': download_cfg.url_index,
            'sheet': download_cfg.sheet_index,
            'sha256': download_cfg.index_sha256,
        },
        'seed': panel_cfg.seed,
        'fractions': {
            role.value: str(fractions.of(role))
            for role in ROLE_ORDER
        },
        'examples_floor': panel_cfg.examples_floor,
        'near_duplicate_min_jaccard': str(min_jaccard),
        'eligibility': eligibility,
        'roles': {
            role: frame.height
            for role, frame in by_role.items()
        },
        'codes_with_role': {
            role: frame.get_column('code').n_unique()
            for role, frame in by_role.items()
        },
        'held_out_leakage': leakage,
        'generator_revision': generator_revision(),
        'library_versions': {
            name: version(name)
            for name in ('numpy', 'polars', 'scikit-learn')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
