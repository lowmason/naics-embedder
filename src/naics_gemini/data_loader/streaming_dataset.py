# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import operator
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from functools import reduce
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import polars as pl
import pyarrow as pa
import torch
from pyarrow import dataset as ds

from naics_gemini.utils.utilities import get_indices_codes

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class CurriculumConfig:

    codes_parquet: str = './data/naics_descriptions.parquet'
    distance_parquet: str = './data/naics_distances.parquet'
    relation_parquet: str = './data/naics_relations.parquet'
    triplets_parquet: str = './data/naics_training_pairs'

    anchor_level: Optional[List[int]] = None
    positive_level: Optional[List[int]] = None
    negative_level: Optional[List[int]] = None

    anchor_distance: Optional[List[float]] = None
    positive_distance: Optional[List[float]] = None
    negative_distance: Optional[List[float]] = None
    
    n_positives: int = 2125
    n_negatives: int = 2125

    seed: int = 42

    def items(self):
        for f in fields(self):
            if f.name != 'input_parquet':
                v = getattr(self, f.name)
                if v is not None:
                    yield f.name, v


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------
    
def _get_file_list(
    codes_parquet: str,
    triplets_parquet: str,
    anchor_level: Optional[List[int]] = None
) -> List[str]:
        
    codes = get_indices_codes(codes_parquet, return_type='codes')
    code_to_idx = get_indices_codes(codes_parquet, return_type='code_to_idx')

    level_dict = defaultdict(list)
    for code in codes:
        level = len(code)
        level_dict[level].append(code)

    if anchor_level is not None:
        dataset_files = []   
        for level in anchor_level:
            for code in level_dict[level]:
                idx = code_to_idx[code]
                for pq_path in Path(f'{triplets_parquet}/anchor={idx}/').glob('*.parquet'):
                    dataset_files.append(pq_path.as_posix())
    
    else:
        dataset_files = []
        for pq_path in Path(f'{triplets_parquet}/').glob('**/*.parquet'):
            dataset_files.append(pq_path.as_posix())
    
    return dataset_files


def _create_dataset(
    codes_parquet: str,
    triplets_parquet: str,
    anchor_level: Optional[List[int]] = None
) -> ds.Dataset:
    
    dataset_files = _get_file_list(
        codes_parquet=codes_parquet,
        triplets_parquet=triplets_parquet,
        anchor_level=anchor_level
    )

    print(f'Number of batches (parquet files): {len(dataset_files):,}')
        
    return (
        ds
        .dataset(
            dataset_files, 
            format='parquet',
            partitioning=ds.partitioning(
                flavor='hive',
                schema=pa.schema([
                    pa.field('anchor', pa.uint32())
                ])
            )        
        )
    )


def _get_file_filters(
    curriculum: CurriculumConfig
) -> Optional[ds.Expression]:

    exprs = []
    for k, v in curriculum.items():

        if isinstance(v, list):
            exprs.append(
                ds.field(k).isin(v)
            )

    if not exprs:
        return None
    
    return reduce(operator.and_, exprs)


def _tuple_dicts(
    distance_parquet: str,
    relation_parquet: str
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:

    distance_df = (
        pl
        .read_parquet(
            distance_parquet
        )
        .select('idx_i', 'idx_j', 'distance')
        .unique()
    )

    relation_df = (
        pl
        .read_parquet(
            relation_parquet
        )
        .select('idx_i', 'idx_j', 'relation_id')
        .unique()
    )

    tuple_iter = (
        distance_df
        .join(
            relation_df,
            on=['idx_i', 'idx_j'],
            how='inner'
        )
        .with_columns(
            distance=pl.when(pl.col('relation_id').eq(2)).then(1.0)
                       .when(pl.col('distance').is_in([1.0, 1.5]))
                       .then(pl.col('distance').add(0.5))
                       .otherwise(pl.col('distance'))
        )
        .sort('idx_i', 'idx_j')
        .iter_rows(named=True)
    )

    relation_dict, distance_dict = {}, {}
    for row in tuple_iter:
        key = (row['idx_i'], row['idx_j'])
        relation_dict[key] = row['relation_id']
        distance_dict[key] = row['distance']

    return relation_dict, distance_dict


def _fill_incomplete(
    incomplete_df: pl.DataFrame,
    triplets_parquet: str,
    curriculum: CurriculumConfig
):

    incomplete_anchors = (
        incomplete_df
        .get_column('anchor_idx')
        .sort()
        .to_list()
    )

    incomplete_files = []
    for idx in incomplete_anchors:
        for pq_path in Path(f'{triplets_parquet}/anchor={idx}/').glob('*.parquet'):
            incomplete_files.append(pq_path.as_posix())
        

    curriculum_keys = [k for k, v in curriculum.items()]
    if 'anchor_distance' in curriculum_keys:

        all_distances = [
            0.125, 0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 
            3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5
        ]
        max_anchor_distance = max(curriculum.anchor_distance)

        incomplete_distance = [d for d in all_distances if d > max_anchor_distance]
        len_incomplete_distance = min(len(curriculum.anchor_distance), len(incomplete_distance))

        incomplete_distance = sorted(list(incomplete_distance))[:len_incomplete_distance] + [7.0]

    else:
        incomplete_distance = [7.0]

    incomplete_curriculum = replace(
        curriculum,
        anchor_distance=incomplete_distance
    )

    filter_expr = _get_file_filters(incomplete_curriculum)

    incomplete_dataset = (
        ds
        .dataset(
            incomplete_files, 
            format='parquet',
            partitioning=ds.partitioning(
                flavor='hive',
                schema=pa.schema([
                    pa.field('anchor', pa.uint32())
                ])
            )        
        )
        .filter(filter_expr)
    )

    incomplete_added = (
        pl
        .from_arrow(
            incomplete_dataset
            .to_table()
        )
        .sort('anchor_idx', 'positive_idx', 'negative_idx', 'anchor_distance')
        .group_by('anchor_idx', 'positive_idx', maintain_order=True)
        .agg(
            negative_added=pl.col('negative_idx')  
        )
    )

    completed = (
        incomplete_df
        .explode('positive_idx', 'negative_idx')
        .with_columns(
            to_add=pl.col('negative_idx')
                    .list.len()
                    .add(-curriculum.n_negatives)
                    .mul(-1)
        )
        .join(
            incomplete_added,
            how='left',
            on=['anchor_idx', 'positive_idx']
        )
        .with_columns(
            pl.col('negative_added')
            .fill_null([])
            .list.sample(
                pl.col('to_add'),
                with_replacement=False,
                shuffle=True, 
                seed=curriculum.seed
            )
        )
        .select(
            anchor_idx=pl.col('anchor_idx'),
            positive_idx=pl.col('positive_idx'),
            negative_idx=pl.col('negative_idx')
                            .list.set_union(pl.col('negative_added'))
                            .list.unique()
        )
        .group_by('anchor_idx')
        .agg(
            pl.col('positive_idx'),
            pl.col('negative_idx')
        )
    )

    print(f'    Incomplete: {len(incomplete_files):,}, Completed: {completed.height:,}')

    return completed


# -------------------------------------------------------------------------------------------------
# Generator function
# -------------------------------------------------------------------------------------------------

def iter_file_batches(
    dataset: Optional[ds.Dataset],
    filter_expr: Optional[ds.Expression],
    codes_parquet: Optional[str],
    triplets_parquet: Optional[str],
    curriculum: CurriculumConfig,
    anchor_level: Optional[List[int]] = None
) -> Iterator[Tuple[ds.FileFragment, pa.Table]]:
    
    if dataset is None:
        dataset = _create_dataset(
            codes_parquet=codes_parquet, 
            triplets_parquet=triplets_parquet, 
            anchor_level=anchor_level
        )

    
    if filter_expr is None:
        filter_expr = _get_file_filters(curriculum)
    
    for file_fragment in dataset.get_fragments(): 

        table = file_fragment.to_table(
            filter=filter_expr,
            columns=['anchor_idx', 'positive_idx', 'negative_idx', 'anchor_distance']
        )
        
        yield file_fragment, table


# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def triplet_batches(
    iter_files: Iterator[Tuple[ds.FileFragment, pa.Table]],
    curriculum: CurriculumConfig,
    idx_to_code: Dict[int, str],
    relation_dict: Dict[Tuple[int, int], int],
    distance_dict: Dict[Tuple[int, int], float],
    rng: Optional[torch.Generator] = None,
) -> Iterator[List[Dict[str, any]]]:
    
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(curriculum.seed)

    n_neg = curriculum.n_negatives
    n_pos = curriculum.n_positives

    for file_num, (file, file_batch) in enumerate(iter_files, start=1):

        df_batch = (
            pl
            .from_arrow(file_batch)
        )

        if df_batch.height > n_pos:
            df_batch = df_batch.sample(
                n=curriculum.n_positives,
                with_replacement=False,
                shuffle=True,
                seed=curriculum.seed
            )

        df = (
            df_batch
            .group_by('anchor_idx', 'positive_idx', maintain_order=True)  
            .agg(
                pl.col('negative_idx')
            )
            .group_by('anchor_idx', maintain_order=True)
            .agg(
                pl.col('positive_idx'),
                pl.col('negative_idx')
            )
            .with_columns(
                fallback=pl.col('negative_idx')
                           .list.len()
                           .lt(n_neg)
            )
        )

        print(
            f'  Batch {file_num} '
            f'[{Path(file.path).parent.stem}]: '
            f'triplets = {df_batch.height:,}, '
            f'grouped triplets = {df.height:,}'
        )

        complete = df.filter(pl.col('fallback')).drop('fallback')
        incomplete = df.filter(~pl.col('fallback')).drop('fallback')

        print(f'    Complete {complete.height:,}, Incomplete = {incomplete.height:,}')

        if incomplete.height == 0:
            completed = complete

        elif complete.height == 0:
            completed = _fill_incomplete(incomplete, curriculum.triplets_parquet, curriculum)

        else:
            _completed = _fill_incomplete(incomplete, curriculum.triplets_parquet, curriculum)

            completed = (
                pl
                .concat([
                    complete, 
                    _completed
                ])
            )

        triplet_iter = (
            completed
            .explode('positive_idx', 'negative_idx')
            .explode('negative_idx')
            .iter_rows(named=True)
        )

        triplets = []
        for row in triplet_iter:
            anchor_idx = row['anchor_idx']
            anchor_code = idx_to_code[anchor_idx]

            positive_idx = row['positive_idx']
            positive_code = idx_to_code[positive_idx]

            positive_relation = relation_dict.get((anchor_idx, positive_idx), None)
            positive_distance = distance_dict.get((anchor_idx, positive_idx), None)

            negative_idx = row['negative_idx']
            negative_code = idx_to_code[negative_idx]

            negative_relation = relation_dict.get((anchor_idx, negative_idx), None)
            negative_distance = distance_dict.get((anchor_idx, negative_idx), None)

            triplets.append({
                'anchor_idx': anchor_idx,
                'anchor_code': anchor_code,
                'positive_idx': positive_idx,
                'positive_code': positive_code,
                'positive_relation': positive_relation,
                'positive_distance': positive_distance,
                'negative_idx': negative_idx,
                'negative_code': negative_code,
                'negative_relation': negative_relation,
                'negative_distance': negative_distance            
            })

        yield triplets


# -------------------------------------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------------------------------------

curriculum = CurriculumConfig(
    anchor_level=[2, 3],
    positive_level=[3],
    n_negatives=10
)

file_iterator = iter_file_batches(
    dataset=None,
    filter_expr=None,
    codes_parquet=curriculum.codes_parquet,
    triplets_parquet=curriculum.triplets_parquet,
    curriculum=curriculum,
    anchor_level=curriculum.anchor_level
 )

relation_dict, distance_dict = _tuple_dicts(
    curriculum.distance_parquet, 
    curriculum.relation_parquet
)

idx_to_code = get_indices_codes(curriculum.codes_parquet, 'idx_to_code')

triplets_iterator = triplet_batches(
    iter_files=file_iterator,
    curriculum=curriculum,
    idx_to_code=idx_to_code,
    relation_dict=relation_dict,
    distance_dict=distance_dict,
    rng=None
)