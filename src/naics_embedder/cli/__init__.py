# -------------------------------------------------------------------------------------------------
# Commands Package
# -------------------------------------------------------------------------------------------------
'''
CLI command modules for NAICS Embedder.

This package organizes CLI commands into logical groups:
- data: Data generation and preprocessing commands
- tools: Utility tools for configuration, GPU optimization, and metrics
- training: Model training commands
'''

import typer

from .commands import data, tools, training

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
