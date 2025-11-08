# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from naics_gemini.data_loader.streaming_dataset import (
    CurriculumConfig,
    create_streaming_dataset,
)
from naics_gemini.data_loader.tokenization_cache import tokenization_cache

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    '''Collate function for batching training examples.'''
    
    channels = ['title', 'description', 'excluded', 'examples']
    
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect anchor and positive codes for evaluation
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    
    for channel in channels:
        anchor_ids = []
        anchor_masks = []
        positive_ids = []
        positive_masks = []
        
        for item in batch:
            anchor_ids.append(item['anchor'][channel]['input_ids'])
            anchor_masks.append(item['anchor'][channel]['attention_mask'])
            positive_ids.append(item['positive'][channel]['input_ids'])
            positive_masks.append(item['positive'][channel]['attention_mask'])
        
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)
        
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_tokens in item['negatives']:
                all_neg_ids.append(neg_tokens[channel]['input_ids'])
                all_neg_masks.append(neg_tokens[channel]['attention_mask'])
        
        negatives_batch[channel]['input_ids'] = torch.stack(all_neg_ids)
        negatives_batch[channel]['attention_mask'] = torch.stack(all_neg_masks)
    
    # Extract codes from batch items
    for item in batch:
        anchor_codes.append(item.get('anchor_code', ''))
        positive_codes.append(item.get('positive_code', ''))
        if 'negative_codes' in item:
            negative_codes.extend(item['negative_codes'])
    
    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': len(batch[0]['negatives']),
        # Add codes for evaluation tracking
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes
    }


# -------------------------------------------------------------------------------------------------
# Wrapper to make generator function work with DataLoader
# -------------------------------------------------------------------------------------------------

class GeneratorDataset(IterableDataset):
    '''Wrapper to make a generator function work with PyTorch DataLoader.'''
    
    def __init__(self, generator_fn, *args, **kwargs):
        self.generator_fn = generator_fn
        self.args = args
        self.kwargs = kwargs
    
    def __iter__(self):
        return self.generator_fn(*self.args, **self.kwargs)


# -------------------------------------------------------------------------------------------------
# Setup and dataloader creation functions
# -------------------------------------------------------------------------------------------------

def setup_data(
    descriptions_path: str = './data/naics_descriptions.parquet',
    tokenizer_name: str = 'sentence-transformers/all-mpnet-base-v2',
    cache_dir: str = './data/token_cache'
) -> Dict[int, torch.Tensor]:
    '''Setup data by loading or creating tokenization cache.'''
    
    token_cache = tokenization_cache(
        fields_path=descriptions_path,
        tokenizer_name=tokenizer_name,
        max_length=512,
        cache_dir=cache_dir
    )
    
    return token_cache


def create_train_dataset(
    descriptions_path: str,
    triplets_path: str,
    token_cache: Dict[int, torch.Tensor],
    curriculum_config: Optional[Dict] = None,
    seed: int = 42
) -> IterableDataset:
    '''Create training dataset.'''
    
    curriculum_config = curriculum_config or {}
    curriculum = CurriculumConfig(**curriculum_config)
    
    dataset = GeneratorDataset(
        create_streaming_dataset,
        descriptions_path=descriptions_path,
        triplets_path=triplets_path,
        token_cache=token_cache,
        curriculum=curriculum,
        seed=seed
    )
    
    return dataset


def create_val_dataset(
    descriptions_path: str,
    triplets_path: str,
    token_cache: Dict[str, torch.Tensor],
    curriculum_config: Optional[Dict] = None,
    seed: int = 42
) -> IterableDataset:
    '''Create validation dataset.'''
    
    curriculum_config = curriculum_config or {}
    curriculum = CurriculumConfig(**curriculum_config)
    
    dataset = GeneratorDataset(
        create_streaming_dataset,
        descriptions_path=descriptions_path,
        triplets_path=triplets_path,
        token_cache=token_cache,
        curriculum=curriculum,
        seed=seed + 1  # Different seed for validation
    )
    
    return dataset


def create_train_dataloader(
    train_dataset: IterableDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    '''Create training dataloader.'''
    
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )


def create_val_dataloader(
    val_dataset: IterableDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    '''Create validation dataloader.'''
    
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )


def create_dataloaders(
    descriptions_path: str = './data/naics_descriptions.parquet',
    triplets_path: str = './data/naics_training_pairs.parquet',
    tokenizer_name: str = 'sentence-transformers/all-mpnet-base-v2',
    curriculum_config: Optional[Dict] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    '''
    Create train and validation dataloaders.
    
    This is a convenience function that sets up everything needed.
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    '''
    
    # Setup tokenization cache
    token_cache = setup_data(
        descriptions_path=descriptions_path,
        tokenizer_name=tokenizer_name
    )
    
    # Create datasets
    train_dataset = create_train_dataset(
        descriptions_path=descriptions_path,
        triplets_path=triplets_path,
        token_cache=token_cache,
        curriculum_config=curriculum_config,
        seed=seed
    )
    
    val_dataset = create_val_dataset(
        descriptions_path=descriptions_path,
        triplets_path=triplets_path,
        token_cache=token_cache,
        curriculum_config=curriculum_config,
        seed=seed
    )
    
    # Create dataloaders
    train_dataloader = create_train_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_dataloader = create_val_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader