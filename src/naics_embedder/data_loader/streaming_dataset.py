# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Any, Dict, Iterator, List

import polars as pl

from naics_embedder.utils.config import StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def _taxonomy(codes_parquet: str) -> pl.DataFrame:

    return (
        pl
        .read_parquet(
            codes_parquet
        )
        .filter(
            pl.col('level').eq(6)
        )
        .select('code')
        .sort(pl.col('code').cast(pl.UInt32))
        .unique(maintain_order=True)
        .select(
            code_2=pl.col('code').str.slice(0, 2),
            code_3=pl.col('code').str.slice(0, 3),
            code_4=pl.col('code').str.slice(0, 4),
            code_5=pl.col('code').str.slice(0, 5),
            code_6=pl.col('code').str.slice(0, 6)
        )
        .with_columns(
            code_2=pl.when(pl.col('code_2').is_in(['31', '32', '33'])).then(pl.lit('31'))
                     .when(pl.col('code_2').is_in(['44', '45'])).then(pl.lit('44'))
                     .when(pl.col('code_2').is_in(['48', '49'])).then(pl.lit('48'))
                     .otherwise(pl.col('code_2'))
        )
        .with_columns(
            code=pl.concat_str(
                pl.col('code_2'),
                pl.col('code_6').str.slice(2, 4),
                separator=''
            )
        )
    )


def _anchors(triplets_parquet: str) -> pl.DataFrame:

    return (
        pl
        .read_parquet(
            triplets_parquet
        )
        .select(
            level=pl.col('anchor_level'),
            anchor=pl.col('anchor_code')
        )
        .unique()
        .sort(
            pl.col('level'),
            pl.col('anchor').cast(pl.UInt32)
        )
    )

    
def _linear_skip(anchor: str, taxonomy: pl.DataFrame) -> List[str]:

    lvl = len(anchor)
    anchor_code = f'code_{lvl}'
    codes = [f'code_{i}' for i in range(lvl + 1, 7)]

    for code in codes:
        candidate = (
            taxonomy
            .filter(pl.col(anchor_code).eq(anchor))
            .get_column(code)
            .unique()
            .to_list()
        )

        if lvl == 5:
            return candidate
        elif len(candidate) > 1:
            return sorted(set(candidate))
    
    return (
        taxonomy
        .filter(pl.col(anchor_code).eq(anchor))
        .get_column('code_6')
        .unique()
        .to_list()
    )


# -------------------------------------------------------------------------------------------------
# Descendants
# -------------------------------------------------------------------------------------------------

def _descendants(
    anchors: pl.DataFrame, 
    taxonomy: pl.DataFrame
) -> pl.DataFrame:

    parent_anchors = (
        anchors
        .filter(pl.col('level').lt(6))
        .get_column('anchor')
        .unique()
        .sort()
        .to_list()
    )

    parent_stratum = []
    for anchor in parent_anchors:
        parent_stratum.append({
            'anchor': anchor,
            'stratum': _linear_skip(anchor, taxonomy)
        })

    return (
        pl.DataFrame(
            data=parent_stratum,
            schema={
                'anchor': pl.Utf8,
                'stratum': pl.List(pl.Utf8)
            }
        )
        .filter(
            pl.col('stratum').is_not_null()
        )
        .explode('stratum')
        .select(
            level=pl.col('anchor')
                    .str.len_chars(),
            anchor=pl.col('anchor'),
            positive=pl.col('stratum')
        )
    )
# -------------------------------------------------------------------------------------------------
# Ancestors
# -------------------------------------------------------------------------------------------------

def _ancestors(
    anchors: pl.DataFrame, 
    taxonomy: pl.DataFrame,
) -> pl.DataFrame:
    return (
        anchors
        .filter(
            pl.col('level').eq(6)
        )
        .join(
            taxonomy,
            left_on='anchor',
            right_on='code_6',
            how='inner'
        )
        .select(
            level=pl.col('level'),
            anchor=pl.col('anchor'),
            code_5=pl.col('code_5'),
            code_4=pl.col('code_4'),
            code_3=pl.col('code_3'),
            code_2=pl.col('code_2')
        )
        .unpivot(
            ['code_5', 'code_4', 'code_3', 'code_2'],
            index=['level', 'anchor'],
            variable_name='ancestor_level',
            value_name='ancestor'
        )
        .with_columns(
            ancestor_level=pl.col('ancestor_level')
                             .str.slice(5, 1)
                             .cast(pl.Int8)
                             .add(-6)
                             .mul(-1)
        )
        .sort('level', 'anchor', 'ancestor_level')
        .group_by('level', 'anchor', maintain_order=True)
        .agg(
            positive=pl.col('ancestor')
        )
    )

def sample_positives(
    descriptions_path: str = './data/naics_descriptions.parquet',
    triplets_path: str = './data/naics_training_pairs/*/*.parquet'
) -> pl.DataFrame:

    taxonomy = _taxonomy(descriptions_path)
    anchors = _anchors(triplets_path)
    code_to_idx = get_indices_codes('code_to_idx')
    descendants = _descendants(anchors, taxonomy)
    ancestors = _ancestors(anchors, taxonomy)

    return (
        pl
        .concat([
            descendants,
            ancestors
        ])
        .explode('positive')
        .select(
            anchor_idx=pl.col('anchor')
                        .replace(code_to_idx)
                        .cast(pl.UInt32),
            positive_idx=pl.col('positive')
                        .replace(code_to_idx)
                        .cast(pl.UInt32),
            anchor_code=pl.col('anchor'),
            positive_code=pl.col('positive'),
            anchor_level=pl.col('level'),
            positive_level=pl.col('positive')
                            .str.len_chars()
        )
        .sort('anchor_idx', 'positive_idx')
    )



# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def create_streaming_generator(cfg: StreamingConfig) -> Iterator[Dict[str, Any]]:

    '''Create a generator that yields triplets for training.'''
    


# -------------------------------------------------------------------------------------------------
# Streaming dataset generator
# -------------------------------------------------------------------------------------------------

def create_streaming_dataset(
    token_cache: Dict[int, Dict[str, Any]],
    cfg: StreamingConfig
) -> Iterator[Dict[str, Any]]:

    '''Create streaming dataset that yields triplets with tokenized embeddings.'''
    
    # Get triplets iterator
    triplets_iterator = create_streaming_generator(cfg)
    
    # Yield triplets with tokenized embeddings
    for triplets in triplets_iterator:
        
        anchor_idx = triplets['anchor_idx']
        anchor_code = triplets['anchor_code']
        try:
            anchor_embedding = {k: v for k, v in token_cache[anchor_idx].items() if k != 'code'}
        except KeyError as e:
            logger.error(f'{worker_id} KeyError accessing token_cache[{anchor_idx}]: {e}')
            raise
        
        positive_idx = triplets['positive_idx']
        positive_code = triplets['positive_code']
        positive_embedding = {k: v for k, v in token_cache[positive_idx].items() if k != 'code'}
        
        negatives = []
        for negative in triplets['negatives']:
            negative_idx = negative['negative_idx']
            negative_code = negative['negative_code']
            negative_embedding = {k: v for k, v in token_cache[negative_idx].items() if k != 'code'}
            
            negatives.append({
                'negative_idx': negative_idx,
                'negative_code': negative_code,
                'negative_embedding': negative_embedding,
                'relation_margin': negative['relation_margin'],
                'distance_margin': negative['distance_margin']
            })
        
        yield {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': positive_code,
            'positive_embedding': positive_embedding,
            'negatives': negatives
        }
