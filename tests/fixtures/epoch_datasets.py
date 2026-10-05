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
            'present': True,
        }
        for channel in CHANNELS
    }

class EpochRecordingDataset(Dataset):
    '''
    Stand-in for RepairedPhase1Dataset: exposes set_epoch() and samples repaired items (one
    candidate pool each) that depend on the epoch. Each item's anchor_code records the epoch it
    was sampled with, and collate_fn carries it into the batch as batch['anchor_code'].
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
            'anchor_code_id': 0,
            'anchor_code': str(self.epoch),
            'anchor_embedding': _embedding(),
            'positive_code_id': 1,
            'positive_code': f'{idx:06d}',
            'positive_embedding': _embedding(),
            'positive_structural_distance': 1.0,
            'positive_structural_relation_id': 1,
            'candidate_pool': [
                {
                    'negative_code_id': 2,
                    'negative_code': '999999',
                    'negative_embedding': _embedding(),
                    'sampling_role_id': 2,
                    'sampling_provenance_id': 2,
                }
            ],
            'difficulty_proposal_indices': [0],
            'selection_k': 1,
        }
