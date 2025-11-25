# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import typer
from rich.console import Console

from naics_embedder.data.compute_distances import calculate_pairwise_distances
from naics_embedder.data.compute_relations import calculate_pairwise_relations
from naics_embedder.data.create_triplets import generate_training_triplets
from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Data generation sub-commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(help='Manage and generate project datasets.')


# -------------------------------------------------------------------------------------------------
# Download and preprocess data
# -------------------------------------------------------------------------------------------------

@app.command('preprocess')
def preprocess():
    """Download and preprocess all raw NAICS data files.

    Runs the extraction and cleaning pipeline that produces
    ``data/naics_descriptions.parquet`` for downstream steps.
    """

    configure_logging('data_preprocess.log')

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')


# -------------------------------------------------------------------------------------------------
# Compute pairwise graph relationships
# -------------------------------------------------------------------------------------------------

@app.command('relations')
def relations():
    """Compute pairwise graph relationships between all NAICS codes.

    Requires the preprocessed descriptions parquet and produces
    ``data/naics_relations.parquet`` with relationship annotations.
    """

    configure_logging('data_relations.log')

    console.rule('[bold green]Stage 2: Computing Relations[/bold green]')

    calculate_pairwise_relations()

    console.print('\n[bold]Relation computation complete.[/bold]\n')


# -------------------------------------------------------------------------------------------------
# Compute pairwise graph distances
# -------------------------------------------------------------------------------------------------

@app.command('distances')
def distances():
    """Compute pairwise graph distances between all NAICS codes.

    Reads ``data/naics_descriptions.parquet`` and saves graph distances to
    ``data/naics_distances.parquet`` for training and evaluation.
    """

    configure_logging('data_distances.log')

    console.rule('[bold green]Stage 2: Computing Distances[/bold green]')

    calculate_pairwise_distances()

    console.print('\n[bold]Distance computation complete.[/bold]\n')


# -------------------------------------------------------------------------------------------------
# Generate training triplets
# -------------------------------------------------------------------------------------------------

@app.command('triplets')
def triplets():
    """Generate (anchor, positive, negative) training triplets.

    Builds the triplet parquet required for contrastive learning using the
    description and distance parquet files.
    """

    configure_logging('data_triplets.log')

    console.rule('[bold green]Stage 3: Generating Triplets[/bold green]')

    generate_training_triplets()

    console.print('\n[bold]Triplet generation complete.[/bold]\n')


# -------------------------------------------------------------------------------------------------
# Run full data generation pipeline
# -------------------------------------------------------------------------------------------------

@app.command('all')
def all_data():
    """Run the full data generation pipeline."""

    configure_logging('data_all.log')

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')

    preprocess()
    relations()
    distances()
    triplets()

    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')
