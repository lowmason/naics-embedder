# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from dataclasses import replace
from typing import Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from naics_gemini.data_loader.streaming_dataset import (
    CurriculumConfig,
    create_streaming_dataset,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:

    '''
    Collate function for batching training examples.
    
    Args:
        batch: List of training examples from streaming dataset
        
    Returns:
        Batched tensors ready for model input
    '''
    
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

    '''Wrapper to make a generator function work with PyTorch DataLoader.'''
    
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

    '''
    Data module for NAICS contrastive learning.
    
    This class encapsulates all data loading logic including:
    - Tokenization caching
    - Curriculum-based filtering
    - Streaming dataset creation
    - DataLoader configuration
    '''
    
    def __init__(
        self,
        curriculum_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        # CLI compatibility parameters (ignored for now, handled by streaming dataset)
        descriptions_path: Optional[str] = None,
        triplets_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        val_split: float = 0.1,
        **kwargs  # Catch any other unexpected parameters
    ):
        '''
        Initialize NAICS DataModule.
        
        Args:
            curriculum_config: Dictionary with curriculum (will be passed to CurriculumConfig)
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            seed: Random seed for training dataset
            descriptions_path: Path to descriptions (for CLI compatibility)
            triplets_path: Path to triplets (for CLI compatibility) 
            tokenizer_name: Tokenizer name (for CLI compatibility)
            val_split: Validation split ratio (for CLI compatibility)
            **kwargs: Additional arguments (ignored)
        '''

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_split = val_split
        
        # Store paths for potential future use
        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        
        # Convert curriculum config dict to CurriculumConfig
        curriculum_config = curriculum_config or {}
        
        # Map CLI parameter names to CurriculumConfig parameter names
        param_mapping = {
            'positive_levels': 'positive_level',
            'k_negatives': 'n_negatives',
            'max_positives': 'n_positives'
        }
        
        # Apply parameter mapping and filter out unsupported parameters
        mapped_config = {}
        supported_params = set(CurriculumConfig.__dataclass_fields__.keys())
        
        for key, value in curriculum_config.items():
            # Map parameter name if needed
            mapped_key = param_mapping.get(key, key)
            # Only include if it's a supported parameter
            if mapped_key in supported_params:
                mapped_config[mapped_key] = value
            else:
                logger.warning(f'Ignoring unsupported curriculum parameter: {key}')
        
        # Set default seed if not provided
        if 'seed' not in mapped_config:
            mapped_config['seed'] = seed
            
        self.curriculum = CurriculumConfig(**mapped_config)
        
        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
    
    
    def setup(self, stage: Optional[str] = None):

        '''
        Setup datasets.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        '''
        
        logger.info('Setting up NAICS DataModule...')
        
        if stage == 'fit' or stage is None:

            # Create training dataset
            logger.info('Creating training dataset...')
            self.train_dataset = GeneratorDataset(
                create_streaming_dataset,
                self.curriculum
            )
            
            # Create validation curriculum with different seed
            val_curriculum = replace(
                self.curriculum,
                seed=self.seed + 1
            )
            
            logger.info('Creating validation dataset...')
            self.val_dataset = GeneratorDataset(
                create_streaming_dataset,
                val_curriculum
            )
            
            logger.info('DataModule setup complete!')
    
    
    def train_dataloader(self) -> DataLoader:

        '''Create training dataloader.'''
        
        if self.train_dataset is None:
            raise RuntimeError('Training dataset not initialized. Call setup() first.')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        
        '''Create validation dataloader.'''
        
        if self.val_dataset is None:
            raise RuntimeError('Validation dataset not initialized. Call setup() first.')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )