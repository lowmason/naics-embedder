# -------------------------------------------------------------------------------------------------
# Data Generation Commands
# -------------------------------------------------------------------------------------------------
'''
CLI commands for NAICS data generation and preprocessing.

This module provides the ``data`` command group that orchestrates the data
preparation pipeline. Commands should be run in order or via ``data all``.

Pipeline Stages:
    1. preprocess: Download and clean raw NAICS data files
    2. supervision: Build one immutable, validated Stage-3 supervision bundle

Commands:
    roles: Draw the frozen index-entry role table, once; it is committed and preprocess applies
        it.
    regressor-groups: Draw the regressor panel's held-out four-digit groups, once; they are
        committed and the panel reads them.
    preprocess: Download raw NAICS files and produce descriptions parquet.
    supervision: Build codebook, pair facts, compatibility distance/relation artifacts,
        training pairs, and curriculum thresholds as one versioned bundle.
    relations, distances, triplets: Deprecated stage commands; each prints a migration notice
        and builds the complete supervision bundle.
    all: Run the complete data generation pipeline.
'''

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    SupervisionBuildConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Data generation sub-commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    help='Data generation and preprocessing commands for NAICS taxonomy.', no_args_is_help=True
)

SUPERVISION_CONFIG = 'data/supervision.yaml'
DOWNLOAD_CONFIG = 'data/download.yaml'
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'

SourceDirOption = Annotated[
    Optional[str],
    typer.Option(
        '--source-dir',
        help='Read the Census files from this directory, by file name, instead of downloading',
    ),
]

def _download_config(source_dir: Optional[str]) -> DownloadConfig:
    cfg = load_config(DownloadConfig, DOWNLOAD_CONFIG)
    if source_dir is not None:
        cfg = cfg.model_copy(update={'source_dir': source_dir})
    return cfg

# -------------------------------------------------------------------------------------------------
# Download and preprocess data
# -------------------------------------------------------------------------------------------------

@app.command('preprocess')
def preprocess(
    source_dir: SourceDirOption = None,
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Overwrite a descriptions file that a configured supervision bundle pins',
        ),
    ] = False,
):
    '''
    Download and preprocess all raw NAICS data files.

    Downloads the official 2022 NAICS taxonomy files from the U.S. Census
    Bureau and processes them into a unified descriptions parquet file.

    The output file contains columns for code, title, description, examples,
    and exclusions for each NAICS code at all hierarchy levels (2-6 digit).
    Each code's examples channel holds its examples-role index entries only,
    per the committed role table (``data roles``).

    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.
        ``data/naics_index_roles.parquet`` - Every index entry with its role.

    Example:
        Download and preprocess NAICS data::

            $ uv run naics-embedder data preprocess
    '''

    configure_logging('data_preprocess.log')

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    try:
        download_preprocess_data(_download_config(source_dir), force=force)
    except FileExistsError as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    console.print('\n[bold]Preprocessing complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Draw the index-entry role table
# -------------------------------------------------------------------------------------------------

@app.command('roles')
def roles(
    source_dir: SourceDirOption = None,
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Redraw an existing table: reassigns every entry and unseals the splits',
        ),
    ] = False,
):
    '''
    Draw the frozen index-entry role table, once.

    Gives every Census index entry exactly one role, per code and stratified: examples-channel
    text, or a training, validation or test query for the outcome panel (roadmap D4). No
    validation or test query matches any training text. The table is committed, and preprocess
    applies it from then on.

    Output:
        ``conf/data/index_roles.csv`` and ``conf/data/index_roles_provenance.json``.

    Example:
        Draw the table from local copies of the Census files::

            $ uv run naics-embedder data roles --source-dir ~/Downloads/Data
    '''

    configure_logging('data_roles.log')

    console.rule('[bold green]Drawing Index-Entry Roles[/bold green]')

    try:
        table_path = generate_index_role_table(
            _download_config(source_dir),
            load_config(OutcomePanelConfig, OUTCOME_PANEL_CONFIG),
            force=force,
        )
    except FileExistsError as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    typer.echo(f'Index-entry role table: {table_path}')

# -------------------------------------------------------------------------------------------------
# Draw the regressor panel's held-out groups
# -------------------------------------------------------------------------------------------------

@app.command('regressor-groups')
def regressor_groups(
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Redraw an existing table: moves both regressor outer sets',
        ),
    ] = False,
):
    '''
    Draw the regressor panel's held-out four-digit groups, once.

    Holds out a fifth of each sector's four-digit groups (largest remainder, seeded) from the
    six-digit population Stage 1's finding names. The table is committed, and the regressor
    panel reads it from then on (roadmap Stage 3).

    Output:
        ``conf/data/regressor_heldout_groups.csv`` and
        ``conf/data/regressor_heldout_groups_provenance.json``.

    Example:
        Draw the groups over bundle 18403d29's codebook::

            $ uv run naics-embedder data regressor-groups --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('data_regressor_groups.log')

    console.rule('[bold green]Drawing Regressor Held-Out Groups[/bold green]')

    try:
        table_path = generate_regressor_group_table(
            load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG),
            Path(codebook),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    typer.echo(f'Regressor held-out groups: {table_path}')

# -------------------------------------------------------------------------------------------------
# Build the Stage-3 supervision bundle
# -------------------------------------------------------------------------------------------------

def _build_supervision_bundle() -> Path:
    cfg = load_config(SupervisionBuildConfig, SUPERVISION_CONFIG)
    manifest_path = generate_supervision_bundle(cfg)
    typer.echo(f'Supervision manifest: {manifest_path}')
    typer.echo(
        'Set supervision.manifest_path to this immutable manifest before repaired Stage-3 '
        'training.'
    )
    return manifest_path

@app.command('supervision')
def supervision():
    '''
    Build one immutable, validated Stage-3 supervision bundle.

    Computes structural distances and relations in canonical pair orientation, attaches both
    directional exclusion flags, and derives the codebook, pair facts, compatibility
    distance/relation artifacts and matrices, training pairs, and curriculum difficulty
    thresholds from those facts. Every artifact carries the bundle ID and schema version; the
    manifest is written only after all artifacts validate, and an existing bundle is never
    overwritten.

    Requires:
        ``data/naics_descriptions.parquet`` - From the preprocess stage.

    Output:
        ``data/supervision/stage3-supervision-v1/<bundle-id>/manifest.json`` and its artifacts.
        The manifest path is printed; configure it as ``supervision.manifest_path``.

    Example:
        Build a supervision bundle::

            $ uv run naics-embedder data supervision
    '''

    configure_logging('data_supervision.log')

    console.rule('[bold green]Stage 2: Building Supervision Bundle[/bold green]')

    _build_supervision_bundle()

    console.print('\n[bold]Supervision bundle complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Deprecated partial-authority stages
# -------------------------------------------------------------------------------------------------

def _migrate_stage(stage: str) -> None:
    typer.echo(
        f'`data {stage}` no longer publishes a standalone artifact; relations, distances, and '
        'triplets are generated together by `naics-embedder data supervision`. Building the '
        'complete supervision bundle now.'
    )
    supervision()

@app.command('relations')
def relations():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('relations')

@app.command('distances')
def distances():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('distances')

@app.command('triplets')
def triplets():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('triplets')

# -------------------------------------------------------------------------------------------------
# Run full data generation pipeline
# -------------------------------------------------------------------------------------------------

@app.command('all')
def all_data():
    '''
    Run the complete data generation pipeline.

    Executes both data preparation stages in order: preprocess, then one complete supervision
    bundle build. This is the recommended way to prepare data for training from scratch.

    Output:
        ``data/naics_descriptions.parquet`` and a new supervision bundle whose manifest path is
        printed.

    Example:
        Run the full pipeline::

            $ uv run naics-embedder data all

    Note:
        This command may take 10-30 minutes depending on your system.
        Progress is logged to ``logs/data_all.log``.
    '''

    configure_logging('data_all.log')

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')

    preprocess()
    supervision()

    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')
