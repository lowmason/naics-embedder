# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from naics_gemini.data_loader.streaming_dataset import create_streaming_dataset
from naics_gemini.utils.config import StreamingConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    
    channels = ['title', 'description', 'excluded', 'examples']
    
    # Initialize batch dictionaries
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect codes and indices for evaluation tracking
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    
    # Process each channel
    for channel in channels:
        anchor_ids = []
        anchor_masks = []
        positive_ids = []
        positive_masks = []
        
        # Collect anchor and positive for this channel
        for item in batch:
            anchor_ids.append(item['anchor_embedding'][channel]['input_ids'])
            anchor_masks.append(item['anchor_embedding'][channel]['attention_mask'])
            positive_ids.append(item['positive_embedding'][channel]['input_ids'])
            positive_masks.append(item['positive_embedding'][channel]['attention_mask'])
        
        # Stack anchor
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)
        
        # Stack positive
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)
        
        # Collect all negatives for this channel
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_dict in item['negatives']:
                all_neg_ids.append(neg_dict['negative_embedding'][channel]['input_ids'])
                all_neg_masks.append(neg_dict['negative_embedding'][channel]['attention_mask'])
        
        # Stack negatives
        negatives_batch[channel]['input_ids'] = torch.stack(all_neg_ids)
        negatives_batch[channel]['attention_mask'] = torch.stack(all_neg_masks)
    
    # Extract codes from batch items
    for item in batch:
        anchor_codes.append(item['anchor_code'])
        positive_codes.append(item['positive_code'])
        for neg_dict in item['negatives']:
            negative_codes.append(neg_dict['negative_code'])
    
    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': len(batch[0]['negatives']),
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes
    }


# -------------------------------------------------------------------------------------------------
# Wrapper to make generator function work with DataLoader
# -------------------------------------------------------------------------------------------------

class GeneratorDataset(IterableDataset):
    
    def __init__(self, generator_fn, *args, **kwargs):
        self.generator_fn = generator_fn
        self.args = args
        self.kwargs = kwargs
    
    def __iter__(self):
        return self.generator_fn(*self.args, **self.kwargs)


# -------------------------------------------------------------------------------------------------
# Main DataModule for PyTorch Lightning (optional but recommended)
# -------------------------------------------------------------------------------------------------

class NAICSDataModule(LightningDataModule):
    
    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        triplets_path: str = './data/naics_training_pairs',
        tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        streaming_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        val_split: float = 0.1,
        **kwargs: Any
    ):

        super().__init__()

        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        if streaming_config is not None:    
            val_streaming_config = streaming_config.copy()
            val_streaming_config['seed'] = seed + 1

            curriculum = StreamingConfig(**streaming_config)
            val_curriculum = StreamingConfig(**val_streaming_config)
        
        else:
            curriculum = StreamingConfig()
            val_curriculum = StreamingConfig(seed=seed + 1)

        logger.info('  • Creating training dataset')
        self.train_dataset = GeneratorDataset(
            create_streaming_dataset,
            curriculum
        )           
        
        logger.info('  • Creating validation dataset\n')
        self.val_dataset = GeneratorDataset(
            create_streaming_dataset,
            val_curriculum
        )
    
    
    def train_dataloader(self) -> DataLoader:

        '''Create training dataloader.'''
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        
        '''Create validation dataloader.'''
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )