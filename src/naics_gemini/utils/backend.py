#!/usr/bin/env python3

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import platform
import sys

import torch

# -------------------------------------------------------------------------------------------------
# Backend GPU availability tests
# -------------------------------------------------------------------------------------------------

def get_device():

    print(f'Python version: {sys.version.split()[0]} on {platform.system()} {platform.processor()}')
    print(f'Torch version: {torch.__version__}')

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    if cuda_ok:
        print(f'GPU version: CUDA : {torch.version.cuda}') # type: ignore
    elif mps_ok:
        print('GPU: MPS (Apple Silicon Metal) available')

    else:
        print('No GPU backend detected.')

    device = 'cuda' if cuda_ok else 'mps' if mps_ok else 'cpu'

    return device

if __name__ == '__main__':
    device = get_device()
