import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.text import Text

from naics_gemini.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------


@dataclass
class Config:
    # Input
    distances_parquet: str = './data/naics_distances.parquet'
    descriptions_parquet: str = './data/naics_descriptions.parquet'

    # Output
    output_parquet: str = './data/naics_training_pairs'


# -------------------------------------------------------------------------------------------------
# Input
# -------------------------------------------------------------------------------------------------


def _input_parquet_files(
    descriptions_parquet: str, distances_parquet: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    descriptions = pl.read_parquet(descriptions_parquet)

    distances = pl.read_parquet(distances_parquet).select(
        idx_i=pl.col('idx_i'),
        idx_j=pl.col('idx_j'),
        code_i=pl.col('code_i'),
        code_j=pl.col('code_j'),
        distance=pl.col('distance'),
    )

    logger.info('Number of input observations:')
    logger.info(f'  descriptions: {descriptions.height: ,}')
    logger.info(f'  distances: {distances.height: ,}\n')

    return descriptions, distances


# -------------------------------------------------------------------------------------------------
# Exclusions
# -------------------------------------------------------------------------------------------------


def _get_exclusions(descriptions_df: pl.DataFrame, distances_df: pl.DataFrame) -> pl.DataFrame:
    codes = set(descriptions_df.get_column('code').unique().sort().to_list())

    exclusions = (
        descriptions_df.filter(pl.col('excluded').is_not_null())
        .select(
            positive_code=pl.col('code'),
            negative_code=pl.col('excluded').str.extract_all(r'\b\d{2,6}\b'),
        )
        .explode('negative_code')
        .sort('negative_code')
        .filter(pl.col('negative_code').is_not_null(), pl.col('negative_code').is_in(codes))
        .join(
            descriptions_df.select(negative_code=pl.col('code')),
            on='negative_code',
            how='inner',
        )
        .join(
            distances_df.select(positive_code=pl.col('code_i'), negative_code=pl.col('code_j')),
            on=['positive_code', 'negative_code'],
            how='inner',
        )
        .select(
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.lit(True),
        )
        .unique()
        .sort('positive_code', 'negative_code')
    )

    logger.info(f'Number of exclusions: {exclusions.height: ,}\n')

    return exclusions


# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------


def _get_distances(distances_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    positive_distances = (
        distances_df.filter(pl.col('distance').ne(pl.col('distance').max()))
        .select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_distance=pl.col('distance'),
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negative_distances = (
        distances_df.select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_distance=pl.col('distance'),
        )
        .unique()
        .sort('anchor_code', 'negative_code')
    )

    logger.info('Number of distances:')
    logger.info(f'  positives: {positive_distances.height: ,}')
    logger.info(f'  negatives: {negative_distances.height: ,}\n')

    return positive_distances, negative_distances


# -------------------------------------------------------------------------------------------------
# Pairs
# -------------------------------------------------------------------------------------------------


def _get_pairs(
    distances_df: pl.DataFrame, exclusions_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    positives = (
        distances_df.filter(pl.col('distance').ne(pl.col('distance').max()))
        .select(
            anchor_idx=pl.col('idx_i'),
            positive_idx=pl.col('idx_j'),
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negatives = (
        distances_df.select(
            positive_idx=pl.col('idx_i'),
            negative_idx=pl.col('idx_j'),
            positive_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            distance=pl.col('distance'),
        )
        .join(exclusions_df, on=['positive_code', 'negative_code'], how='left')
        .select(
            negative_idx=pl.col('negative_idx'),
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.col('excluded').fill_null(False),
            unrelated=pl.col('distance').eq(pl.col('distance').max()),
        )
        .unique()
        .sort('positive_code', 'negative_code')
    )

    logger.info('Number of pairs:')
    logger.info(f'  positives: {positives.height: ,}')
    logger.info(f'  negatives: {negatives.height: ,}\n')

    return positives, negatives


# -------------------------------------------------------------------------------------------------
# Triplets
# -------------------------------------------------------------------------------------------------


def _get_triplets(
    positives_df: pl.DataFrame,
    negatives_df: pl.DataFrame,
    positive_distances_df: pl.DataFrame,
    negative_distances_df: pl.DataFrame,
) -> pl.DataFrame:
    triplets = positives_df.join(negatives_df, how='inner', on='positive_code')

    triplets = triplets.join(
        positive_distances_df, how='inner', on=['anchor_code', 'positive_code']
    )

    triplets = (
        triplets.
        join(
            negative_distances_df, 
            how='inner', 
            on=['anchor_code', 'negative_code']
        )
        .with_columns(
            distance_diff=pl.col('negative_distance').sub(pl.col('positive_distance'))
        )
        .with_columns(
            unrelated=pl.when(pl.col('excluded'))
                        .then(pl.lit(False))
                        .otherwise(pl.col('unrelated')),
            distance_diff=pl.when(pl.col('excluded')).then(pl.lit(0.125))
                            .when(pl.col('unrelated')).then(pl.lit(7.0))
                            .when(pl.col('distance_diff').eq(-0.5)).then(pl.lit(0.25))
                            .otherwise(pl.col('distance_diff')),
        )
        .filter(
            pl.col('distance_diff').gt(0.0)
        )
        .sort('anchor_idx', 'positive_idx')
        .with_columns(batch=pl.col('anchor_idx').floordiv(21.25))
        .select(
            'batch',
            'anchor_idx', 'positive_idx', 'negative_idx',
            'anchor_code', 'positive_code', 'negative_code',
            'excluded', 'unrelated',
            'positive_distance', 'negative_distance', 
            'distance_diff',
        )
    )

    logger.info(f'Number of triplets: {triplets.height: ,}\n')

    return triplets


# -------------------------------------------------------------------------------------------------
# Triplet stats
# -------------------------------------------------------------------------------------------------

def _triplet_stats(triplets_df: pl.DataFrame):

    stats_df = (
        triplets_df
        .group_by('excluded', 'unrelated', 'distance_diff')
        .agg(
            count=pl.len()
        )
        .with_columns(
            pct=pl.col('count')
                  .truediv(pl.col('count').sum())
        )
        .sort('distance_diff', 'excluded', 'unrelated', descending=[False, False, True])
    )

    dists = stats_df.get_column('distance_diff').unique().sort().to_list()

    logger.info(
        f'Observed differences in positive and negative distances: {", ".join(map(str, dists))}\n'
    )

    console = Console()

    def _render_triplet_table(rows):
        
        title = Text('Triplet Statistics:', style='bold')

        table = Table(title=title, title_justify='left', show_lines=True, show_footer=True)

        total_count = sum(row.get('count', 0) for row in rows)
        total_pct = 100 * sum(row.get('pct', 0) for row in rows)

        table.add_column('Exclusion', justify='center')
        table.add_column('Unrelated', justify='center')
        table.add_column('Distance Difference', justify='center')
        table.add_column('Frequency', justify='right', footer=f'[bold]{total_count: ,}[/bold]')
        table.add_column('Percent', justify='right', footer=f'[bold]{total_pct: .4f}%[/bold]')

        for row in rows:
            excluded = 'True' if row.get('excluded', False) else 'False'
            unrelated = 'True' if row.get('unrelated', False) else 'False'

            distance_diff = row.get('distance_diff')
            dd_cell = Text(f'{distance_diff: .3f}', style='bold')

            n = row.get('count', 0)
            pct = row.get('pct', 0)

            n_cell = Text(f'{n: ,}')
            pct_cell = Text(f'{100 * pct: .4f}%', style='bold')

            table.add_row(excluded, unrelated, dd_cell, n_cell, pct_cell)

        console.print(table)

    _render_triplet_table(stats_df.to_dicts())


# -------------------------------------------------------------------------------------------------
# Generate triplets
# -------------------------------------------------------------------------------------------------


def generate_training_triplets() -> pl.DataFrame:
    # Configuration
    cfg = Config()

    logger.info('Configuration:')
    logger.info(json.dumps(asdict(cfg), indent=2))
    logger.info('')

    # Load data
    descriptions, distances = _input_parquet_files(cfg.descriptions_parquet, cfg.distances_parquet)

    # Exclusions
    exclusions = _get_exclusions(descriptions, distances)

    # All positive and negative distances
    positive_distances, negative_distances = _get_distances(distances)

    # All positive and negative pairs
    positives, negatives = _get_pairs(distances, exclusions)

    # Combine positives and negatives into triplets
    triplets_df = _get_triplets(positives, negatives, positive_distances, negative_distances)

    _triplet_stats(triplets_df)

    _parquet_stats(
        parquet_df=triplets_df,
        message='NAICS triplets written to',
        output_parquet=cfg.output_parquet,
        logger=logger,
    )

    output_path = Path(cfg.output_parquet)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)     

    (
        triplets_df
        .write_parquet(
            cfg.output_parquet,
            use_pyarrow=True,
            pyarrow_options={
                'partition_cols': ['batch']
            }
        )
    )

    return triplets_df


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    generate_training_triplets()