# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import polars as pl
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:
    tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    max_length: Optional[Union[int, str]] = None
    
    def __post_init__(self):
        if self.max_length is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.max_length = tokenizer.model_max_length


# -------------------------------------------------------------------------------------------------
# Tokenization functions
# -------------------------------------------------------------------------------------------------

def tokenize_text(
    dict: Dict[str, str],
    field: str,
    counter: Dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    
    '''Tokenize a single text field.'''

    text = dict.get(field, '')
    
    if not text or text == '':
        text = '[EMPTY]'
    else:
        counter[field] += 1
        
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ) 
    
    return (
        {
            'input_ids': encoded['input_ids'].squeeze(0), #type: ignore
            'attention_mask': encoded['attention_mask'].squeeze(0) #type: ignore
        },
        counter
    )


def build_tokenization_cache(
    fields_path: str,
    tokenizer_name: str,
    max_length: int
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Build tokenization cache from descriptions file.'''
    
    logger.info('Building tokenization cache...')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # DataFrame iterator
    df_iter = (
        pl
        .read_parquet(
            fields_path
        )
        .sort('index')
        .iter_rows(named=True)
    )
    
    # Tokenization cache
    cache, cnt = {}, { 'title': 0, 'description': 0, 'excluded': 0, 'examples': 0 }
    for row in df_iter:

        idx, code = row['index'], row['code']
        
        title, cnt = tokenize_text(row, 'title', cnt, tokenizer, 24)
        description, cnt = tokenize_text(row, 'description', cnt, tokenizer, max_length)
        excluded, cnt = tokenize_text(row, 'excluded', cnt, tokenizer, max_length)
        examples, cnt = tokenize_text(row, 'examples', cnt, tokenizer, max_length)
        
        cache[idx] = {
            'code': code,
            'title': title,
            'description': description,
            'excluded': excluded,
            'examples': examples
        }
    
    logger.info('Cache built with:')
    logger.info(f'  {cnt["title"]: ,} titles')
    logger.info(f'  {cnt["description"]: ,} descriptions')
    logger.info(f'  {cnt["excluded"]: ,} exclusions')
    logger.info(f'  {cnt["examples"]: ,} examples')

    return cache


def save_tokenization_cache(
    cache: Dict[int, Dict[str, torch.Tensor]],
    cache_dir: str = './data/token_cache'
) -> Path:
    
    '''Save tokenization cache to disk.'''
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = Path(f'{cache_path}/token_cache.pt')
    
    torch.save(cache, cache_file)
    logger.info(f'Saved tokenization cache to: {cache_file.resolve()}')
    
    return cache_file


def load_tokenization_cache(
    cache_dir: str = './data/token_cache'
) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    
    '''Load tokenization cache from disk if it exists.'''
    
    cache_file = Path(f'{cache_dir}/token_cache.pt')
    
    if cache_file.exists():
        logger.info(f'Loading tokenization cache from: {cache_file.resolve()}')
        return torch.load(cache_file, weights_only=True)
    
    return None


def tokenization_cache(
    fields_path: str,
    tokenizer_name: str,
    max_length: int,
    cache_dir: str = './data/token_cache'
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Get tokenization cache, loading from disk or building if necessary.'''
    
    # Try to load from cache
    cache = load_tokenization_cache(cache_dir)
    if cache is not None:
        return cache
    
    # Build cache if it doesn't exist
    cache = build_tokenization_cache(fields_path, tokenizer_name, max_length)
    save_tokenization_cache(cache, cache_dir)
    
    return cache


def get_tokens(
    idx_code: Union[int, str],
    cache: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Get tokens for a specific NAICS index or code from cache.'''

    if isinstance(idx_code, int):
        key = idx_code

    if isinstance(idx_code, str):
        for k, v in cache.items():
            if v['code'] == idx_code:
                key = k
                break

    return {k: v for k, v in cache[key].items() if k != 'code'}