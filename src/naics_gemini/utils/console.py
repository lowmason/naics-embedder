# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------------------------------------

class CustomFormatter(logging.Formatter):
    '''
    Prints time only when it changes (by second):

    [12:34:56]
    first message
    second message
    [12:34:57]
    next message
    '''

    def __init__(self, timefmt='[%H:%M:%S]'):
        super().__init__()
        self.timefmt = timefmt
        self._last_time_str = None

    def format(self, record: logging.LogRecord) -> str:
        time_str = datetime.fromtimestamp(record.created).strftime(self.timefmt)
        message = record.getMessage()

        if time_str != self._last_time_str:
            self._last_time_str = time_str
            return f'{time_str}\n{message}'
        
        else:
            return message



def configure_logging(level='INFO'):

    '''
    Configures logging to use RichHandler for beautiful output with module context.
    '''

    console = Console(markup=False)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_suppress=[
            'typer',
            'click',
            'hydra',
            'pytorch_lightning',
            'torch'
        ],
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True
    )
    
    handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=level,
        handlers=[handler]
    )

    # Quiet down noisy libs
    for noisy in ['hydra', 'pytorch_lightning', 'transformers', 'httpx']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Explicitly re-enable your own package
    logging.getLogger('naics_gemini').setLevel(level)
