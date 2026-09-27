# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging

import polars as pl

from naics_embedder.utils.config import DistancesConfig
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Structural distances
# -------------------------------------------------------------------------------------------------

def compute_structural_distances(input_parquet: str, cfg: DistancesConfig) -> pl.DataFrame:
    '''
    D* (Req 7) for every unordered pair of distinct codes.

    Rows follow the canonical pair orientation (shallower code first, code order on ties), so
    each unordered pair appears exactly once. Every value comes from
    :func:`~naics_embedder.utils.naics_hierarchy.tree_distance_matrix`, the tree path length
    through a virtual root above the sectors: no half-step for lineal pairs and no cross-sector
    constant. No exclusion processing happens here and nothing is written.

    Args:
        input_parquet: Descriptions parquet with ``index``, ``level``, and ``code`` columns.
        cfg: Distance configuration (logged for provenance).

    Returns:
        DataFrame with ``idx_i``, ``idx_j``, ``code_i``, ``code_j``, ``structural_distance``.
    '''

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    naics = pl.read_parquet(input_parquet).select('index', 'level', 'code').with_row_index('row')
    distances = tree_distance_matrix(naics.get_column('code').to_list())
    naics_i = naics.select(
        row_i=pl.col('row'), idx_i=pl.col('index'), lvl_i=pl.col('level'), code_i=pl.col('code')
    )
    naics_j = naics.select(
        row_j=pl.col('row'), idx_j=pl.col('index'), lvl_j=pl.col('level'), code_j=pl.col('code')
    )

    # Canonical orientation: shallower code first, numeric code order on ties.
    pairs = naics_i.join(naics_j, how='cross').filter(
        (pl.col('lvl_i') < pl.col('lvl_j'))
        | (
            (pl.col('lvl_i') == pl.col('lvl_j'))
            & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
        )
    )
    values = distances[pairs.get_column('row_i').to_numpy(), pairs.get_column('row_j').to_numpy()]
    logger.info(f'D* for {pairs.height:,} pairs of {naics.height:,} codes')

    return pairs.select(
        pl.col('idx_i'),
        pl.col('idx_j'),
        pl.col('code_i'),
        pl.col('code_j'),
        structural_distance=pl.Series(values, dtype=pl.Float32),
    ).sort('idx_i', 'idx_j')
