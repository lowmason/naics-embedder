# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------
'''
CLI utility commands for configuration, GPU optimization, and metrics analysis.

This module provides the ``tools`` command group with utilities for inspecting
configuration, visualizing training metrics, and investigating model behavior.

Commands:
    config: Display current training configuration.
    visualize: Generate visualizations from training log files.
    investigate: Analyze hierarchy preservation metrics.
'''

from pathlib import Path
from typing import Optional, Tuple

import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    resolve_graph_supervision_paths,
)
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.embeddings_verification import Stage4VerificationConfig, verify_stage4
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    help='Utility tools for configuration, metrics analysis, and debugging.', no_args_is_help=True
)

# -------------------------------------------------------------------------------------------------
# View configuration
# -------------------------------------------------------------------------------------------------

@app.command('config')
def config(
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
):
    '''
    Display the current training and curriculum configuration.

    Loads the specified configuration file and displays a formatted summary
    of all settings including data paths, model architecture, training
    hyperparameters, and loss function weights.

    Args:
        config_file: Path to the YAML configuration file to display.
            Defaults to ``conf/config.yaml``.

    Example:
        Display default configuration::

            $ uv run naics-embedder tools config

        Display custom configuration::

            $ uv run naics-embedder tools config --config conf/custom.yaml
    '''

    configure_logging('tools_config.log')

    show_current_config(config_file)

# -------------------------------------------------------------------------------------------------
# Visualize metrics
# -------------------------------------------------------------------------------------------------

@app.command('visualize')
def visualize(
    stage: Annotated[
        str,
        typer.Option(
            '--stage',
            '-s',
            help="Stage name to filter (e.g., '02_text')",
        ),
    ] = '02_text',
    log_file: Annotated[
        Optional[str],
        typer.Option(
            '--log-file',
            help='Path to log file (default: logs/train_sequential.log)',
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            '--output-dir',
            help='Output directory for plots (default: outputs/visualizations/)',
        ),
    ] = None,
):
    '''
    Visualize training metrics from log files.

    Parses training log files and generates visualizations showing the
    progression of key metrics including contrastive loss, hierarchy
    correlation, embedding statistics, and learning rate schedules.

    Output visualizations are saved as PNG files in the specified output
    directory.

    Args:
        stage: Stage identifier used to filter metrics. Use this to focus
            on a specific training stage like ``02_text``.
        log_file: Path to the training log file to parse. When omitted,
            defaults to ``logs/train_sequential.log``.
        output_dir: Directory for saving visualization files. When omitted,
            defaults to ``outputs/visualizations/``.

    Example:
        Visualize metrics from default log::

            $ uv run naics-embedder tools visualize --stage 02_text

        Visualize custom log file::

            $ uv run naics-embedder tools visualize --log-file logs/train.log
    '''

    configure_logging('tools_visualize.log')

    try:
        log_path = Path(log_file) if log_file else None
        output_path = Path(output_dir) if output_dir else None

        result = visualize_metrics(stage=stage, log_file=log_path, output_dir=output_path)

        if result.get('output_file'):
            console.print(
                '\n[bold green]✓[/bold green] Visualization saved to: '
                f'[cyan]{result["output_file"]}[/cyan]\n'
            )

    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)

# -------------------------------------------------------------------------------------------------
# Investigate hierarchy preservation metrics
# -------------------------------------------------------------------------------------------------

@app.command('investigate')
def investigate(
    distance_matrix: Annotated[
        Optional[str],
        typer.Option(
            '--distance-matrix',
            help='Path to ground truth distance matrix',
        ),
    ] = None,
    config_file: Annotated[
        Optional[str],
        typer.Option(
            '--config',
            help='Path to config file (default: conf/config.yaml)',
        ),
    ] = None,
):
    '''
    Analyze why hierarchy preservation correlations might be low.
    
    Investigates potential causes for poor hierarchy preservation metrics
    by analyzing the ground truth distance matrix, evaluation configuration,
    and providing diagnostic recommendations.
    
    Use this command when training produces unexpectedly low hierarchy
    correlation metrics to identify configuration or data issues.
    
    Args:
        distance_matrix: Path to the ground truth distance matrix parquet.
            When omitted, uses the path from the configuration file.
        config_file: Path to the configuration file. When omitted, uses
            the default ``conf/config.yaml``.
    
    Example:
        Investigate hierarchy metrics::
        
            $ uv run naics-embedder tools investigate
        
        Use custom distance matrix::
        
            $ uv run naics-embedder tools investigate \\
                --distance-matrix data/custom_distances.parquet
    '''

    configure_logging('tools_investigate.log')

    try:
        dist_path = Path(distance_matrix) if distance_matrix else None
        config_path = Path(config_file) if config_file else None

        result = investigate_hierarchy(distance_matrix_path=dist_path, config_path=config_path)
        for key, value in result.items():
            console.print(f'[bold green]{key}:[/bold green] {value}')

        console.print('\n[bold green]Investigation complete![/bold green]\n')

    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)

# -------------------------------------------------------------------------------------------------
# Verify Stage 4 against Stage 3
# -------------------------------------------------------------------------------------------------

def _stage4_structural_inputs(
    supervision_manifest: Optional[str],
    distance_matrix: Optional[str],
    relations_parquet: Optional[str],
) -> Tuple[Path, Path]:
    '''
    The distance matrix and relations parquet that verify-stage4 reads.

    With a supervision manifest both come from that one validated bundle (an explicitly supplied
    path must be the bundle's own artifact). Without one, unset paths fall back to the legacy
    ``./data`` files.
    '''
    if supervision_manifest:
        paths = resolve_graph_supervision_paths(
            supervision_manifest,
            distance_matrix_path=distance_matrix,
            relations_path=relations_parquet,
        )
        return paths.distance_matrix, paths.relations
    return (
        Path(distance_matrix or './data/naics_distance_matrix.parquet'),
        Path(relations_parquet or './data/naics_relations.parquet'),
    )

@app.command('verify-stage4')
def verify_stage4_command(
    stage3_parquet: Annotated[
        str,
        typer.Option(
            '--pre',
            help='Path to Stage 3 (pre-HGCN) embeddings parquet',
        ),
    ] = './output/hyperbolic_projection/encodings.parquet',
    stage4_parquet: Annotated[
        str,
        typer.Option(
            '--post',
            help='Path to Stage 4 (HGCN) embeddings parquet',
        ),
    ] = './output/hgcn/encodings.parquet',
    distance_matrix: Annotated[
        Optional[str],
        typer.Option(
            '--distance-matrix',
            help=(
                'Path to ground truth distance matrix parquet (default: the bundle artifact with '
                '--supervision-manifest, else ./data/naics_distance_matrix.parquet)'
            ),
        ),
    ] = None,
    relations_parquet: Annotated[
        Optional[str],
        typer.Option(
            '--relations',
            help=(
                'Path to relations parquet, used for the parent retrieval metric (default: the '
                'bundle artifact with --supervision-manifest, else ./data/naics_relations.parquet)'
            ),
        ),
    ] = None,
    supervision_manifest: Annotated[
        Optional[str],
        typer.Option(
            '--supervision-manifest',
            help='Supervision bundle manifest; the distance matrix and relations come from it',
        ),
    ] = None,
    max_cophenetic_drop: Annotated[
        float,
        typer.Option('--max-cophenetic-drop', help='Allowed drop in cophenetic correlation'),
    ] = 0.02,
    max_ndcg_drop: Annotated[
        float,
        typer.Option('--max-ndcg-drop', help='Allowed drop in NDCG@10'),
    ] = 0.01,
    min_local_improvement: Annotated[
        float,
        typer.Option('--min-local-improvement', help='Required parent retrieval improvement'),
    ] = 0.05,
    ndcg_k: Annotated[
        int,
        typer.Option('--ndcg-k', help='NDCG@K to evaluate'),
    ] = 10,
    parent_top_k: Annotated[
        int,
        typer.Option('--parent-top-k', help='Top-K used for parent retrieval accuracy'),
    ] = 1,
):
    '''
    Compare Stage 3 and Stage 4 embeddings at curvature 1.0.

    Enforce cophenetic, NDCG, and parent-retrieval thresholds. Report structural
    Spearman v1 separately; undefined values and deltas display as N/A.
    '''

    configure_logging('tools_verify_stage4.log')

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=max_cophenetic_drop,
        max_ndcg_degradation=max_ndcg_drop,
        min_local_improvement=min_local_improvement,
        ndcg_k=ndcg_k,
        parent_top_k=parent_top_k,
    )

    try:
        distance_matrix_path, relations_path = _stage4_structural_inputs(
            supervision_manifest, distance_matrix, relations_parquet
        )
        result = verify_stage4(
            Path(stage3_parquet),
            Path(stage4_parquet),
            distance_matrix_path,
            relations_path,
            cfg,
        )
    except Exception as exc:
        console.print(f'[bold red]Verification failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Stage 4 Verification[/bold cyan]\n')
    console.print('[bold]Pre-HGCN metrics:[/bold]')
    for key, value in result['pre'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Post-HGCN metrics:[/bold]')
    for key, value in result['post'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Deltas:[/bold]')
    for key, value in result['delta'].items():
        formatted = 'N/A' if value is None else f'{value:+.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Threshold checks:[/bold]')
    for key, passed in result['checks'].items():
        status = '[green]PASS[/green]' if passed else '[red]FAIL[/red]'
        console.print(f'  • {key}: {status}')

    if result['passed']:
        console.print('\n[bold green]✓ Stage 4 verification passed![/bold green]\n')
    else:
        console.print('\n[bold red]✗ Stage 4 verification failed thresholds[/bold red]\n')
        raise typer.Exit(code=1)
