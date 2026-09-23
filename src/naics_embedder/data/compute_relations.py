# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Tuple

import networkx as nx
import polars as pl

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Relations utilities
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
    code_6 = _sector_codes(sector, input_parquet)
    code_5 = sorted(set(n[:5] for n in code_6))
    code_4 = sorted(set(n[:4] for n in code_6))
    code_3 = sorted(set(n[:3] for n in code_6))
    code_2 = sorted(set(n[:2] for n in code_6))
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

def _get_relations(i: str, j: str, depths: Dict[str, int], ancestors: Dict[str, List[str]]) -> str:
    depth_i, depth_j = depths[i], depths[j]
    ancestors_j = ancestors[j]

    if i in ancestors_j:
        generation_gap = depth_j - depth_i

        if generation_gap == 1:
            return 'child'

        elif generation_gap == 2:
            return 'grandchild'

        elif generation_gap == 3:
            return 'great-grandchild'

        else:
            return 'great-great-grandchild'

    common_ancestor = _find_common_ancestor(i, j, ancestors)

    distance_i = depth_i - depths[common_ancestor]  # type: ignore
    distance_j = depth_j - depths[common_ancestor]  # type: ignore

    if distance_i == 1 and distance_j == 1:
        return 'sibling'

    elif distance_i == 1 and distance_j == 2:
        return 'nephew/niece'

    elif distance_i == 2 and distance_j == 2:
        return 'cousin'

    elif distance_i == 1 and distance_j > 2:
        num_grands = distance_j - 2
        return f'{"grand-" * num_grands}nephew/niece'

    elif distance_i == 2 and distance_j > 2:
        times_removed = distance_j - 2
        return f'cousin_{times_removed}_times_removed'

    else:
        degree = min(distance_i, distance_j) - 1
        removed = abs(distance_j - distance_i)
        ordinals = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}
        degree_name = ordinals.get(degree, f'{degree}th')

        if removed == 0:
            return f'{degree_name}_cousin'

        else:
            return f'{degree_name}_cousin_{removed}_times_removed'

# -------------------------------------------------------------------------------------------------
# Structural relations
# -------------------------------------------------------------------------------------------------

CROSS_SECTOR_RELATION_NAME = 'cross_sector'
CROSS_SECTOR_RELATION_ID = 99

def compute_structural_relations(
    input_parquet: str,
    relation_ids: Mapping[str, int],
) -> pl.DataFrame:
    '''
    Compute canonical structural relations for every unordered pair of distinct codes.

    Rows follow the canonical pair orientation (shallower code first, code order on ties); the
    relation name describes ``code_j`` relative to ``code_i`` and is not a bidirectional label.
    Codes in different sector trees receive the explicit ``cross_sector`` relation. No exclusion
    processing happens here and nothing is written.

    Args:
        input_parquet: Descriptions parquet with ``index``, ``level``, and ``code`` columns.
        relation_ids: Mapping of structural relation names to IDs. Every relation name produced
            by the hierarchy must be mapped; ``cross_sector`` defaults to 99.

    Returns:
        DataFrame with ``idx_i``, ``idx_j``, ``code_i``, ``code_j``,
        ``structural_relation_id``, ``structural_relation_name``.

    Raises:
        ValueError: If the hierarchy produces a relation name absent from ``relation_ids``.
    '''

    cross_sector_id = relation_ids.get(CROSS_SECTOR_RELATION_NAME, CROSS_SECTOR_RELATION_ID)

    df_list = []
    for sector in _sectors(input_parquet):
        G = _sector_tree(sector, input_parquet)

        pairs = [(i, j) if int(i) < int(j) else (j, i) for i, j in combinations(G.nodes, 2)]

        depths, ancestors, _ = _compute_tree_metadata(G, sector)

        relationships = []
        for i, j in sorted(pairs, key=lambda x: (x[0], x[1])):
            relationship = _get_relations(i, j, depths, ancestors)
            relationships.append({'code_i': i, 'code_j': j, 'relationship': relationship})

        df = pl.DataFrame(
            data=relationships,
            schema={
                'code_i': pl.Utf8,
                'code_j': pl.Utf8,
                'relationship': pl.Utf8
            },
        )

        logger.info(f'Sector {sector}: [{len(depths): ,} nodes, {df.height: ,} pairs]')

        df_list.append(df)

    pair_relations = pl.concat(df_list).select(
        pl.col('code_i'),
        pl.col('code_j'),
        relation_id=pl.col('relationship').replace_strict(dict(relation_ids),
                                                          default=None).cast(pl.Int8),
        relation=pl.col('relationship'),
    )

    unmapped = sorted(
        pair_relations.filter(pl.col('relation_id').is_null()).get_column('relation').unique()
    )
    if unmapped:
        raise ValueError(
            f'hierarchy relations missing from the relation_id mapping: {unmapped}; '
            'an unmapped relation must never fall back to the cross_sector ID'
        )

    naics_i = pl.scan_parquet(input_parquet).select(
        idx_i=pl.col('index'), lvl_i=pl.col('level'), code_i=pl.col('code')
    )

    naics_j = pl.scan_parquet(input_parquet).select(
        idx_j=pl.col('index'), lvl_j=pl.col('level'), code_j=pl.col('code')
    )

    # Canonical orientation: shallower code first, numeric code order on ties. Tree pairs above
    # use the same orientation, so every in-tree pair joins its structural relation exactly once.
    return (
        naics_i.join(naics_j, how='cross').filter(
            (pl.col('lvl_i') < pl.col('lvl_j'))
            | (
                (pl.col('lvl_i') == pl.col('lvl_j'))
                & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
            )
        ).collect().join(pair_relations, how='left', on=['code_i', 'code_j']).select(
            idx_i=pl.col('idx_i'),
            idx_j=pl.col('idx_j'),
            code_i=pl.col('code_i'),
            code_j=pl.col('code_j'),
            structural_relation_id=pl.col('relation_id').fill_null(cross_sector_id),
            structural_relation_name=pl.col('relation').fill_null(CROSS_SECTOR_RELATION_NAME),
        ).sort('idx_i', 'idx_j')
    )
