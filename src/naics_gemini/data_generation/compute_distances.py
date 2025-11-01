from dataclasses import asdict, dataclass
from itertools import combinations
import json
from typing import Dict, List, Optional, Tuple
import logging

import networkx as nx
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import track

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration for computing pairwise distances."""
    input_parquet: str = './data/naics_descriptions.parquet'
    output_parquet: str = './data/naics_distances.parquet'


# -------------------------------------------------------------------------------------------------
# 1. Relationship utilities
# -------------------------------------------------------------------------------------------------

def sectors(input_parquet: str) -> List[str]:
    """Get the list of 2-digit sector codes."""
    return (
        pl.read_parquet(input_parquet)
        .filter(pl.col('level') == 2)
        .select('code')
        .sort(pl.col('code').cast(pl.UInt32))
        .unique(maintain_order=True)
        .get_column('code')
        .to_list()
    )

def sector_codes(sector: str, input_parquet: str) -> List[str]:
    """Get all codes belonging to a specific sector."""
    if sector == '31':
        sector_list = ['31', '32', '33']
    elif sector == '44':
        sector_list = ['44', '45']
    elif sector == '48':
        sector_list = ['48', '49']
    else:
        sector_list = [sector]

    return (
        pl.read_parquet(input_parquet)
        .filter(
            pl.col('code').str.slice(0, 2).is_in(sector_list)
        )
        .get_column('code')
        .to_list()
    )

def is_descendent(parent: str, child: str) -> bool:
    """Check if one code is a hierarchical descendent of another."""
    return (
        (len(child) > len(parent)) 
        & (child[:len(parent)] == parent)
    )

def get_relations(sector_nodes: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Builds graph edges (relationships) for all codes in a sector."""
    nodes = sorted(sector_nodes, key=len)
    relations = []
    lineal_children = []

    for parent, child in combinations(nodes, 2):
        if is_descendent(parent, child):
            relations.append((parent, child))
            
            # Check for direct parent-child (lineal)
            is_lineal = True
            for other in nodes:
                if (
                    (other != parent) 
                    & (other != child) 
                    & is_descendent(parent, other) 
                    & is_descendent(other, child)
                ):
                    is_lineal = False
                    break
            
            if is_lineal:
                lineal_children.append(child)

    return relations, lineal_children


# -------------------------------------------------------------------------------------------------
# 2. Graph distance computation
# -------------------------------------------------------------------------------------------------

def compute_distances(
    relations: List[Tuple[str, str]], 
    lineal_children: List[str]
) -> pl.DataFrame:
    """Computes graph distances for all pairs in a sector."""
    G = nx.Graph()
    G.add_edges_from(relations)
    
    pairs = list(G.nodes())
    distances = []

    for code_i, code_j in combinations(pairs, 2):
        try:
            distance = nx.shortest_path_length(G, source=code_i, target=code_j)
            
            lineal = 1 if (
                (is_descendent(code_i, code_j) and code_j in lineal_children) or
                (is_descendent(code_j, code_i) and code_i in lineal_children)
            ) else 0

            distances.append(
                (code_i, code_j, distance, lineal)
            )
        except nx.NetworkXNoPath:
            # This shouldn't happen in a connected component (sector)
            logger.warning(f"No path found between {code_i} and {code_j}")

    return pl.DataFrame(
        distances,
        schema=["code_i", "code_j", "distance", "lineal"]
    )

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def calculate_pairwise_distances() -> None:
    """
    Main entry point for computing pairwise distances for all codes.
    """
    # Configuration
    cfg = Config()
    logger.info('Configuration:')
    logger.info(json.dumps(asdict(cfg), indent=2))

    # Get sector list
    sector_list = sectors(cfg.input_parquet)
    logger.info(f"Found {len(sector_list)} sectors.")

    # Compute distances within each sector
    all_sector_distances = []
    
    desc = "Processing sectors"
    for sector in track(sector_list, description=desc):
        nodes = sector_codes(sector, cfg.input_parquet)
        relations, lineal_children = get_relations(nodes)
        
        if relations:
            sector_distances = compute_distances(relations, lineal_children)
            all_sector_distances.append(sector_distances)

    pair_relations = pl.concat(all_sector_distances)
    logger.info(f"Computed {pair_relations.height:,} within-sector distances.")

    # Create full cross-join for all pairs
    logger.info("Creating full pairwise matrix...")
    naics_i = (
        pl.scan_parquet(cfg.input_parquet)
        .select(
            idx_i=pl.col('index'),
            lvl_i=pl.col('level'), 
            code_i=pl.col('code')
        )
    )
    naics_j = (
        pl.scan_parquet(cfg.input_parquet)
        .select(
            idx_j=pl.col('index'), 
            lvl_j=pl.col('level'), 
            code_j=pl.col('code')
        )
    )

    # Cross join and filter to unique pairs
    naics_distances = (
        naics_i.join(naics_j, how='cross')
        .filter(pl.col('idx_i') < pl.col('idx_j')) # Keep only (i, j) where i < j
        .with_columns(
            sector_i=pl.col('code_i').str.slice(0, 2),
            sector_j=pl.col('code_j').str.slice(0, 2)
        )
        .collect() # Collect after cross-join
        .join(
            pair_relations, 
            how='left', 
            on=['code_i', 'code_j']
        )
        .select(
            pl.col('idx_i'),
            pl.col('idx_j'),
            pl.col('code_i'),
            pl.col('code_j'),
            pl.col('lvl_i'),
            pl.col('lvl_j'),
            pl.col('sector_i'),
            pl.col('sector_j'),
            pl.col('distance'),
            pl.col('lineal')
        )
        .with_columns(
            unrelated=(pl.col('sector_i') != pl.col('sector_j')),
            distance=pl.col('distance').fill_null(9) # 9 for unrelated
        )
    )
    
    logger.info(f"Total pairs: {naics_distances.height:,}")

    # Stats table
    console = Console()
    table = Table(
        title='Pairwise Distance Stats', 
        show_header=True, 
        header_style='bold magenta'
    )
    table.add_column('Unrelated', style='dim')
    table.add_column('Distance', style='cyan')
    table.add_column('Lineal', style='green')
    table.add_column('N', justify='right', style='bold')
    table.add_column('Percent', justify='right')

    stats = (
        naics_distances
        .group_by('unrelated', 'distance', 'lineal')
        .agg(pl.count().alias('n'))
        .sort('unrelated', 'distance', 'lineal')
    )
    total = stats['n'].sum()
    
    for row in stats.iter_rows(named=True):
        pct = f"{row['n'] / total:.2%}"
        table.add_row(
            str(row['unrelated']),
            str(row['distance']),
            str(row['lineal']),
            f"{row['n']:,}",
            pct
        )
    
    console.print(table)

    # Write output
    logger.info(f"Writing distances to {cfg.output_parquet}...")
    naics_distances.write_parquet(cfg.output_parquet)

    logger.info(f"\nSuccessfully generated {cfg.output_parquet}")


if __name__ == '__main__':
    configure_logging()
    calculate_pairwise_distances()
