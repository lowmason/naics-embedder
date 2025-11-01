import json
from dataclasses import asdict, dataclass
from typing import Tuple
import logging

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
    """Configuration for generating training triplets."""
    # Input
    distances_parquet: str = './data/naics_distances.parquet'
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    
    # Output
    output_parquet: str = './data/naics_training_pairs.parquet'


# -------------------------------------------------------------------------------------------------
# Input
# -------------------------------------------------------------------------------------------------

def input_parquet_files(
    descriptions_parquet: str, 
    distances_parquet: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load description and distance parquet files."""
    
    logger.info(f"Loading descriptions from {descriptions_parquet}")
    descriptions = pl.read_parquet(descriptions_parquet)

    logger.info(f"Loading distances from {distances_parquet}")
    distances = (
        pl.read_parquet(distances_parquet)
        .select(
            code_i=pl.col('code_i'),
            code_j=pl.col('code_j'),
            distance=pl.col('distance').add(pl.col('lineal')) # Add lineal bonus
        )
    )
    logger.info('Input data loaded.')
    return descriptions, distances

# -------------------------------------------------------------------------------------------------
# Get exclusions
# -------------------------------------------------------------------------------------------------

def get_exclusions(
    descriptions: pl.DataFrame, 
    distances: pl.DataFrame
) -> pl.DataFrame:
    """Identify codes that are explicitly cross-referenced as exclusions."""
    
    logger.info("Processing exclusions...")
    # Get codes from descriptions
    codes = (
        descriptions
        .select(
            code=pl.col('code'), 
            excluded=pl.col('excluded').str.split(';')
        )
        .explode('excluded')
        .with_columns(
            excluded=pl.col('excluded')
                        .str.extract(r'(\d{2,6})') # Extract code
                        .str.strip()
        )
        .filter(pl.col('excluded').is_not_null())
        .filter(pl.col('code') != pl.col('excluded'))
        .select(
            code_i=pl.col('code'),
            code_j=pl.col('excluded')
        )
    )
    
    # Get codes from distances (symmetric)
    codes_sym = codes.rename({'code_i': 'code_j', 'code_j': 'code_i'})
    
    # Combine and join with distances
    exclusions = (
        pl.concat([codes, codes_sym])
        .unique()
        .join(distances, on=['code_i', 'code_j'])
        .select('code_i', 'code_j')
        .with_columns(excluded=pl.lit(True))
    )
    
    logger.info(f"Found {exclusions.height:,} explicit exclusion pairs.")
    return exclusions

# -------------------------------------------------------------------------------------------------
# Get distances
# -------------------------------------------------------------------------------------------------

def get_distances(distances: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Separate distances into positive (close) and negative (far)."""
    
    logger.info("Separating positive and negative distance pairs...")
    # Positive distances
    positive_distances = (
        distances
        .filter(pl.col('distance') <= 8) # distance < 9 (i.e., not 9)
        .select(
            code_i=pl.col('code_i'), 
            code_j=pl.col('code_j'), 
            positive_distance=pl.col('distance')
        )
    )
    
    # Negative distances
    negative_distances = (
        distances
        .filter(pl.col('distance') >= 2) # distance > 1
        .select(
            code_i=pl.col('code_i'), 
            code_j=pl.col('code_j'), 
            negative_distance=pl.col('distance')
        )
    )
    
    return positive_distances, negative_distances

# -------------------------------------------------------------------------------------------------
# Get pairs
# -------------------------------------------------------------------------------------------------

def get_pairs(
    distances: pl.DataFrame, 
    exclusions: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Get all valid positive and negative pairs, excluding explicit exclusions."""
    
    logger.info("Identifying valid positive and negative pairs...")
    # Positive pairs
    positives = (
        distances
        .filter(pl.col('distance') <= 8) # distance < 9
        .join(exclusions, on=['code_i', 'code_j'], how='anti') # Exclude
        .select(
            code_i=pl.col('code_i'),
            code_j=pl.col('code_j')
        )
    )

    # Negative pairs
    negatives = (
        distances
        .filter(pl.col('distance') >= 2) # distance > 1
        .select(
            code_i=pl.col('code_i'), 
            code_j=pl.col('code_j')
        )
    )
    
    logger.info(f"Found {positives.height:,} valid positive pairs.")
    logger.info(f"Found {negatives.height:,} valid negative pairs.")
    return positives, negatives

# -------------------------------------------------------------------------------------------------
# Get triplets
# -------------------------------------------------------------------------------------------------

def get_triplets(
    positives: pl.DataFrame, 
    negatives: pl.DataFrame, 
    positive_distances: pl.DataFrame,
    negative_distances: pl.DataFrame
) -> pl.DataFrame:
    """Create (anchor, positive, negative) triplets by joining pairs."""
    
    logger.info("Generating triplets...")
    
    # Join positives and negatives on the anchor (code_i)
    triplets_df = (
        positives
        .join(
            negatives, 
            on='code_i'
        )
        .rename({
            'code_i': 'anchor', 
            'code_j': 'positive', 
            'code_j_right': 'negative'
        })
        .filter(pl.col('positive') != pl.col('negative')) # Ensure pos != neg
        .join(
            positive_distances.rename({
                'code_i': 'anchor', 
                'code_j': 'positive'
            }), 
            on=['anchor', 'positive'], 
            how='left'
        )
        .join(
            negative_distances.rename({
                'code_i': 'anchor', 
                'code_j': 'negative'
            }), 
            on=['anchor', 'negative'], 
            how='left'
        )
        .select(
            'anchor', 
            'positive', 
            'negative', 
            'positive_distance', 
            'negative_distance'
        )
    )
    
    logger.info(f"Generated {triplets_df.height:,} raw triplets.")
    return triplets_df

# -------------------------------------------------------------------------------------------------
# Stats
# -------------------------------------------------------------------------------------------------

def triplet_stats(triplets_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate and display statistics about the generated triplets."""
    
    logger.info("Calculating triplet statistics...")
    # Add stats columns
    triplets_stats = (
        triplets_df
        .with_columns(
            distance_diff=(
                pl.col('negative_distance') - pl.col('positive_distance')
            )
        )
        .with_columns(
            unrelated=pl.col('negative_distance') == 10, # 9 (dist) + 1 (lineal) = 10
            lineal=pl.col('positive_distance') % 1 != 0,
            positive_distance=pl.col('positive_distance').floor(),
            negative_distance=pl.col('negative_distance').floor(),
            distance_diff=pl.col('distance_diff').floor()
        )
        .with_columns(
            difficulty_bucket=pl.col('distance_diff').cast(pl.Int64)
        )
    )

    # Stats table
    console = Console()
    table = Table(
        title='Triplet Difficulty Stats', 
        show_header=True, 
        header_style='bold magenta'
    )
    table.add_column('Difficulty Bucket\n(Neg - Pos Dist)', style='cyan')
    table.add_column('Lineal', style='green')
    table.add_column('Unrelated', style='dim')
    table.add_column('Example\nPos-Neg Dist', style='dim')
    table.add_column('N', justify='right', style='bold')
    table.add_column('Percent', justify='right')

    stats = (
        triplets_stats
        .group_by('difficulty_bucket', 'lineal', 'unrelated')
        .agg(
            pl.col('distance_diff').min().alias('min_diff'),
            pl.col('distance_diff').max().alias('max_diff'),
            pl.count().alias('n')
        )
        .sort('difficulty_bucket', descending=True)
    )
    
    total = stats['n'].sum()

    for row in stats.iter_rows(named=True):
        pct = f"{row['n'] / total:.2%}"
        diff_range = (
            f"{row['min_diff']}" 
            if row['min_diff'] == row['max_diff'] 
            else f"{row['min_diff']}-{row['max_diff']}"
        )
        
        table.add_row(
            str(row['difficulty_bucket']),
            str(row['lineal']),
            str(row['unrelated']),
            diff_range,
            f"{row['n']:,}",
            pct
        )
    
    console.print(table)
    return triplets_stats


# -------------------------------------------------------------------------------------------------
# Parquet
# -------------------------------------------------------------------------------------------------

def parquet_stats(parquet_df: pl.DataFrame, output_parquet: str) -> None:
    """Log statistics about the final Parquet file."""
    
    rows = list(zip(parquet_df.columns, parquet_df.dtypes))
    
    logger.info('Parquet schema: Schema([')
    for name, dtype in rows:
        logger.info(f"    ('{name}', {dtype}),")
    logger.info('])\n')

    logger.info(f'{parquet_df.height: ,} NAICS contrastive triplets written to:')
    logger.info(f'  {output_parquet}\n')


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def generate_training_triplets() -> None:
    """
    Main entry point for generating (anchor, positive, negative) triplets.
    """
    # Configuration
    cfg = Config()
    logger.info('Configuration:')
    logger.info(json.dumps(asdict(cfg), indent=2))

    # Load data
    descriptions, distances = input_parquet_files(
        cfg.descriptions_parquet,
        cfg.distances_parquet
    )

    # Exclusions
    exclusions = get_exclusions(descriptions, distances)

    # All positive and negative distances
    positive_distances, negative_distances = get_distances(distances)

    # All positive and negative pairs
    positives, negatives = get_pairs(distances, exclusions)

    # Combine positives and negatives into triplets
    triplets_df = get_triplets(
        positives,
        negatives,
        positive_distances,
        negative_distances
    )

    # Calculate stats
    triplets_with_stats = triplet_stats(triplets_df)

    # Write to parquet
    logger.info(f"Writing triplets to {cfg.output_parquet}...")
    (
        triplets_with_stats
        .write_parquet(cfg.output_parquet)
    )

    parquet_stats(triplets_with_stats, cfg.output_parquet)

if __name__ == '__main__':
    configure_logging()
    generate_training_triplets()
