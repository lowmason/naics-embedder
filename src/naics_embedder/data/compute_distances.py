# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx
import polars as pl

from naics_embedder.supervision.schema import CROSS_SECTOR_DISTANCE
from naics_embedder.utils.config import DistancesConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Distance utilities
# -------------------------------------------------------------------------------------------------

def _sectors(input_parquet: str) -> List[str]:
    return (
        pl.read_parquet(input_parquet).filter(pl.col('level').eq(2)).select('code').sort(
            pl.col('code').cast(pl.UInt32)
        ).unique(maintain_order=True).get_column('code').to_list()
    )

def _sector_codes(sector: str, input_parquet: str) -> List[str]:
    if sector == '31':
        sector_list = ['31', '32', '33']

    elif sector == '44':
        sector_list = ['44', '45']

    elif sector == '48':
        sector_list = ['48', '49']

    else:
        sector_list = [sector]

    return (
        pl.read_parquet(input_parquet).filter(
            pl.col('level').eq(6),
            pl.col('code').str.slice(0, 2).is_in(sector_list)
        ).select('code').sort(pl.col('code').cast(pl.UInt32)).unique(maintain_order=True
                                                                     ).get_column('code').to_list()
    )

def _join_sectors(code: str) -> str:
    if code in ['31', '32', '33']:
        return '31'

    elif code in ['44', '45']:
        return '44'

    elif code in ['48', '49']:
        return '48'

    else:
        return code

def _sector_tree(sector: str, input_parquet: str) -> nx.DiGraph:
    if sector == '31':
        sector_list = ['31', '32', '33']

    elif sector == '44':
        sector_list = ['44', '45']

    elif sector == '48':
        sector_list = ['48', '49']

    else:
        sector_list = [sector]

    code_6 = _sector_codes(sector, input_parquet)
    code_5 = sorted(set(n[:5] for n in code_6))
    code_4 = sorted(set(n[:4] for n in code_6))
    code_3 = sorted(set(n[:3] for n in code_6))
    code_2 = sorted(set(n[:2] for n in code_6))

    if not code_6:
        sector_df = pl.read_parquet(input_parquet).filter(
            pl.col('code').str.slice(0, 2).is_in(sector_list)
        )

        codes = sector_df.select('code').unique(maintain_order=True).get_column('code').to_list()

        graph = nx.DiGraph()
        graph.add_nodes_from(codes)

        for code in codes:
            if len(code) <= 2:
                continue

            parent = code[:-1]
            parent = _join_sectors(parent) if len(parent) == 2 else parent

            if parent in codes:
                graph.add_edge(parent, code)

        root = _join_sectors(sector)
        if not graph.has_node(root):
            graph.add_node(root)

        return graph

    edge_list = []
    for c2 in code_2:
        s = _join_sectors(c2)
        for c3 in [c3 for c3 in code_3 if c3[:2] == c2]:
            edge_list.append((s, c3))
            for c4 in [c4 for c4 in code_4 if c4[:3] == c3]:
                edge_list.append((c3, c4))
                for c5 in [c5 for c5 in code_5 if c5[:4] == c4]:
                    edge_list.append((c4, c5))
                    for c6 in [c6 for c6 in code_6 if c6[:5] == c5]:
                        edge_list.append((c5, c6))

    edges = set(edge_list)

    return nx.DiGraph(edges)

def _compute_tree_metadata(tree: nx.DiGraph, root: str) -> Tuple[Dict, Dict, Dict]:
    depths, ancestors, parents = {}, {}, {}
    queue = [(root, 0, [root])]
    while queue:
        node, depth, ancestor_path = queue.pop(0)
        depths[node] = depth
        ancestors[node] = ancestor_path.copy()
        parents[node] = ancestor_path[-2] if len(ancestor_path) > 1 else None

        for child in tree.successors(node):
            queue.append((child, depth + 1, ancestor_path + [child]))

    return depths, ancestors, parents

def _find_common_ancestor(i: str, j: str, ancestors: Dict[str, List[str]]) -> Optional[str]:
    ancestors_i, ancestors_j = set(ancestors[i]), ancestors[j]
    for ancestor in reversed(ancestors_j):
        if ancestor in ancestors_i:
            return ancestor

    return None

# -------------------------------------------------------------------------------------------------
# 2. Compute relationships
# -------------------------------------------------------------------------------------------------

def _get_distance(i: str, j: str, depths: Dict[str, int], ancestors: Dict[str, List[str]]) -> float:
    if i == j:
        return 0.0

    depth_i, depth_j = depths[i], depths[j]
    common_ancestor = _find_common_ancestor(i, j, ancestors)
    if common_ancestor is None:
        return float(depth_i + depth_j)

    depth_ancestor = depths[common_ancestor]

    distance = (depth_i - depth_ancestor) + (depth_j - depth_ancestor)

    is_lineal = (i in ancestors[j]) or (j in ancestors[i])
    if is_lineal:
        distance -= 0.5

    return float(max(distance, 0.0))

# -------------------------------------------------------------------------------------------------
# Structural distances
# -------------------------------------------------------------------------------------------------

def compute_structural_distances(input_parquet: str, cfg: DistancesConfig) -> pl.DataFrame:
    '''
    Compute structural distances for every unordered pair of distinct codes.

    Rows follow the canonical pair orientation (shallower code first, code order on ties), so
    each unordered pair appears exactly once. Codes in different sector trees receive the explicit
    cross-sector distance. No exclusion processing happens here and nothing is written.

    Args:
        input_parquet: Descriptions parquet with ``index``, ``level``, and ``code`` columns.
        cfg: Distance configuration (logged for provenance).

    Returns:
        DataFrame with ``idx_i``, ``idx_j``, ``code_i``, ``code_j``, ``structural_distance``.
    '''

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    df_list = []
    for sector in _sectors(input_parquet):
        G = _sector_tree(sector, input_parquet)

        pairs = [(i, j) if int(i) < int(j) else (j, i) for i, j in combinations(G.nodes, 2)]

        depths, ancestors, parents = _compute_tree_metadata(G, sector)

        distances = []
        for i, j in sorted(pairs, key=lambda x: (x[0], x[1])):
            distance = _get_distance(i, j, depths, ancestors)
            distances.append({'code_i': i, 'code_j': j, 'distance': distance})

        df = pl.DataFrame(
            data=distances,
            schema={
                'code_i': pl.Utf8,
                'code_j': pl.Utf8,
                'distance': pl.Float32
            },
        )

        logger.info(f'Sector {sector}: [{len(depths): ,} nodes, {df.height: ,} pairs]')

        df_list.append(df)

    pair_relations = pl.concat(df_list).select(
        pl.col('code_i'), pl.col('code_j'), distance=pl.col('distance')
    )

    naics_i = pl.scan_parquet(input_parquet).select(
        idx_i=pl.col('index'), lvl_i=pl.col('level'), code_i=pl.col('code')
    )

    naics_j = pl.scan_parquet(input_parquet).select(
        idx_j=pl.col('index'), lvl_j=pl.col('level'), code_j=pl.col('code')
    )

    # Canonical orientation: shallower code first, numeric code order on ties. Tree pairs above
    # use the same orientation, so every in-tree pair joins its structural value exactly once.
    return (
        naics_i.join(naics_j, how='cross').filter(
            (pl.col('lvl_i') < pl.col('lvl_j'))
            | (
                (pl.col('lvl_i') == pl.col('lvl_j'))
                & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
            )
        ).collect().join(pair_relations, how='left', on=['code_i', 'code_j']).select(
            pl.col('idx_i'),
            pl.col('idx_j'),
            pl.col('code_i'),
            pl.col('code_j'),
            structural_distance=pl.col('distance').fill_null(CROSS_SECTOR_DISTANCE),
        ).sort('idx_i', 'idx_j')
    )
