"""Top-level Typer application that wires together project subcommands.

The module configures the global Typer instance and attaches command groups for
data preparation, tooling, and training. Warnings from PyTorch Lightning are
suppressed to keep CLI output focused on actionable information for users.
"""

import logging
import os
import warnings

import typer
from rich.console import Console
from rich.panel import Panel

from naics_embedder.cli.commands import data, tools, training

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

console = Console()
logger = logging.getLogger(__name__)


warnings.filterwarnings(
    'ignore',
    message='.*Precision.*is not supported by the model summary.*',
    category=UserWarning,
    module='pytorch_lightning.utilities.model_summary.model_summary'
)

warnings.filterwarnings(
    'ignore',
    message='.*Found .* module.*in eval mode.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Checkpoint directory.*exists and is not empty.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Trying to infer the.*batch_size.*',
    category=UserWarning,
    module='pytorch_lightning'
)


app = typer.Typer(
    help=Panel.fit(
        '[bold cyan]NAICS Embedder[/bold cyan]\n\nText-enhanced Hyperbolic NAICS Embedding System',
        border_style='cyan',
        padding=(1, 2),
    )  # type: ignore
)

app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

app.command('train')(training.train)
app.command('train-seq')(training.train_sequential)