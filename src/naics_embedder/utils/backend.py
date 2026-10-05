# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import platform
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Backend GPU availability tests
# -------------------------------------------------------------------------------------------------

def get_device(log_info: bool = False, *,
               cuda_precision: str = 'bf16-mixed') -> Tuple[str, str, int]:
    '''
    Detect the accelerator and the precision a trainer runs at on it.

    Args:
        log_info: If True, log the Python, torch and GPU backend versions.
        cuda_precision: The precision on CUDA. ``train`` passes ``training.trainer.precision``
            (spec 4.2); the default is the shipped ``bf16-mixed``.

    Returns:
        ``(device, precision, num_gpus)``. The device is ``cuda``, ``mps`` or ``cpu``. The
        precision is ``cuda_precision`` on CUDA and ``32-true`` everywhere else. ``num_gpus`` is
        1 on MPS and 0 on CPU.
    '''
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    if cuda_ok:
        num_gpus = torch.cuda.device_count()
        gpu = f'  • GPU:\n    - CUDA ({torch.version.cuda})'  # type: ignore
    elif mps_ok:
        num_gpus = 1
        gpu = '  • GPU:\n    - MPS (Apple Silicon Metal)'
    else:
        num_gpus = 0
        gpu = '  • GPU:\n    - No GPU backend detected, using CPU'

    device = 'cuda' if cuda_ok else 'mps' if mps_ok else 'cpu'
    precision = cuda_precision if cuda_ok else '32-true'

    if log_info:
        logger.info(
            '  • Python:\n'
            f'    - version: {sys.version.split()[0]}\n'
            f'    - platform: {platform.system()} {platform.processor()}'
        )
        logger.info(f'  • Torch:\n    - version: {torch.__version__}')
        logger.info(f'{gpu}\n')

    return device, precision, num_gpus

if __name__ == '__main__':
    device, precision, num_gpus = get_device()
