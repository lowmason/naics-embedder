# -------------------------------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------------------------------
'''
CLI command modules for NAICS Embedder.

This package organizes CLI commands into logical groups:
- data: Data generation and preprocessing commands
- tools: Utility tools for configuration, GPU optimization, and metrics
- training: Model training commands

Importing this package configures the process for the CLI: it sets ``PYTORCH_ALLOC_CONF``
before torch is loaded and applies the centralized warning filters from
``naics_embedder.utils.warnings``.
'''

import os

# Set before anything imports torch so its CUDA allocator sees it. That includes
# naics_embedder.utils, whose __init__ loads torch through utils.backend.
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import typer

from naics_embedder.utils.warnings import configure_warnings

from .commands import data, tools, training

# Applied after the imports above: filterwarnings() prepends, and torch/Lightning register
# filters of their own on import, so running last keeps these suppressions ahead of them.
configure_warnings()

# Create main Typer app. Typer requires help to be a string (it runs inspect.cleandoc on it), so
# the banner is styled with Rich markup rather than passed as a Panel renderable.
app = typer.Typer(
    help=(
        '[bold cyan]NAICS Embedder[/bold cyan]\n\n'
        'Text-enhanced Hyperbolic NAICS Embedding System'
    ),
    rich_markup_mode='rich',
)

# Add sub-apps
app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

# Register training commands directly on main app
app.command('train')(training.train)

__all__ = ['app', 'data', 'tools', 'training']
