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

# Create main Typer app
app = typer.Typer(
    help='NAICS Embedder\n\nText-enhanced Hyperbolic NAICS Embedding System'
)

# Add sub-apps
app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

# Register training commands directly on main app
app.command('train')(training.train)

__all__ = ['app', 'data', 'tools', 'training']
