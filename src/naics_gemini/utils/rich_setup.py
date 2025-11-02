import logging

from rich.logging import RichHandler


def configure_logging(level='INFO'):
    '''
    Configures logging to use RichHandler for beautiful output.
    '''
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[
                    'typer', 
                    'click', 
                    'hydra',
                    'pytorch_lightning',
                    'torch'
                ],
                show_path=False,
            )
        ],
    )
    
    # Set lower levels for noisy libraries
    logging.getLogger('hydra').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)

    # Re-enable our own logger
    logging.getLogger('naics_gemini').setLevel(level)
