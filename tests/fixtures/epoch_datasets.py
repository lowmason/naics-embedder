'''
Epoch-aware datasets for testing train-epoch propagation through DataLoader workers.

Defined at module level so that DataLoader workers started with spawn or forkserver can unpickle
them by qualified name.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict

import torch
from torch.utils.data import Dataset

CHANNELS = ['title', 'description', 'excluded', 'examples']

# -------------------------------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------------------------------

def _embedding() -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        channel: {
            'input_ids': torch.zeros(4, dtype=torch.long),
            'attention_mask': torch.ones(4, dtype=torch.long),
        }
        for channel in CHANNELS
    }


class EpochRecordingDataset(Dataset):
    '''
    Stand-in for Phase1MapDataset: exposes set_epoch() and samples items that depend on the
    epoch. Each item's anchor_code records the epoch it was sampled with, and collate_fn carries
    it into the batch as batch['anchor_code'].
    '''

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'anchor_code': str(self.epoch),
            'anchor_embedding': _embedding(),
            'positive_code': f'{idx:06d}',
            'positive_embedding': _embedding(),
            'negatives': [{'negative_code': '999999', 'negative_embedding': _embedding()}],
        }
