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
    outcome-baseline: Score the lexical stub encoder on the outcome panel's validation split.
    text-only-table: Embed every code's text with the arm's backbone, frozen (roadmap D9).
    regressor-panel: Score an arm on the regressor panel's validation or sealed test split.
    margins: Fix each panel's non-inferiority margin from a reference arm (Req 5).
    decide: Decide among arms under Req 5's rule over D8's three panels.
'''

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import ArmRecord, MarginRecord, read_record, write_record
from naics_embedder.decision.rule import TieUnresolvedError
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    resolve_graph_supervision_paths,
)
from naics_embedder.panels.lexical_encoder import (
    LexicalTrigramEncoder,
    code_texts_from_descriptions,
)
from naics_embedder.panels.outcome import OutcomePanel, SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    TEST,
    VALIDATION,
    ArmTables,
    Regime,
    load_regressor_panel,
    summarize,
)
from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.text_only import provenance_path as text_only_provenance_path
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.embeddings_verification import Stage4VerificationConfig, verify_stage4
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.config import (
    DecisionConfig,
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    help='Utility tools for configuration, metrics analysis, and debugging.', no_args_is_help=True
)

REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'
DECISION_CONFIG = 'data/decision.yaml'

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

# -------------------------------------------------------------------------------------------------
# Outcome panel: lexical baseline
# -------------------------------------------------------------------------------------------------

@app.command('outcome-baseline')
def outcome_baseline(
    purpose: Annotated[
        str,
        typer.Option('--purpose', help='Why this read happens; recorded in the selection log'),
    ] = 'lexical baseline on the validation split',
    index_roles: Annotated[
        Optional[str],
        typer.Option(
            '--index-roles',
            help='Index roles parquet from data preprocess (default: the download config)',
        ),
    ] = None,
    descriptions: Annotated[
        Optional[str],
        typer.Option(
            '--descriptions',
            help='Descriptions parquet from data preprocess (default: the download config)',
        ),
    ] = None,
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the outcome-panel config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the summary as JSON to this path'),
    ] = None,
):
    '''
    Score the training-free lexical encoder on the outcome panel's validation split.

    Decodes every validation query to the nearest six-digit code by hashed character trigrams
    under cosine distance, and reports top-1 accuracy, MRR, Hit@1/5/10 and the level of the
    lowest common ancestor. The read is logged in the selection log. The test split stays sealed:
    this command never opens it.

    Example:
        Score the baseline on the preprocessing outputs::

            $ uv run naics-embedder tools outcome-baseline
    '''

    configure_logging('tools_outcome_baseline.log')

    download_cfg = load_config(DownloadConfig, 'data/download.yaml')
    panel_cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')
    descriptions_path = Path(descriptions or download_cfg.output_parquet)

    try:
        panel = OutcomePanel.from_files(
            index_roles or download_cfg.index_roles_parquet,
            descriptions_path,
            log or panel_cfg.selection_log,
        )
        encoder = LexicalTrigramEncoder(
            code_texts_from_descriptions(pl.read_parquet(descriptions_path))
        )
        result = panel.score(encoder, IndexRole.VALIDATION, purpose)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f'[bold red]Outcome baseline failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Outcome panel: lexical baseline, validation split[/bold cyan]\n')
    for key, value in result.summary.items():
        formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
        console.print(f'  • {key}: {formatted}')
    console.print(f'\nRead logged to {panel.log.path}\n')

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'fingerprint': panel.fingerprint, 'summary': result.summary}
        path.write_text(json.dumps(payload, indent=2) + '\n')

# -------------------------------------------------------------------------------------------------
# Regressor panel
# -------------------------------------------------------------------------------------------------

@app.command('text-only-table')
def text_only_table(
    descriptions: Annotated[
        str,
        typer.Option('--descriptions', help="The arm's descriptions parquet: the text it reads"),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the table (parquet)'),
    ],
    backbone: Annotated[
        Optional[str],
        typer.Option('--backbone', help="The arm's backbone (default: the regressor config)"),
    ] = None,
):
    '''
    Embed every code's text with the arm's backbone, frozen (roadmap D9).

    Each of the four channels is mean-pooled over its tokens, and a code's vector is the mean of
    its present channels. The backbone is read from the local Hugging Face cache. The regressor
    panel reduces the table to the arm's dimension by PCA.

    Output:
        The table, and ``<stem>_provenance.json`` beside it.

    Example:
        Embed the text bundle 18403d29 was built from::

            $ uv run naics-embedder tools text-only-table \\
                --descriptions data/naics_descriptions.parquet --output /tmp/text_only.parquet
    '''

    configure_logging('tools_text_only_table.log')

    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    try:
        path = build_text_only_table(
            Path(descriptions),
            Path(output),
            backbone=backbone or cfg.text_only.backbone,
            max_length=cfg.text_only.max_length,
            batch_size=cfg.text_only.batch_size,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f'[bold red]Text-only table failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'Text-only table: {path}')
    console.print(f'Provenance: {text_only_provenance_path(path)}')

def _require_writable(path: Path) -> None:
    '''
    Create the file's directory and require that the file can be written there.

    Raises:
        OSError: If the path is a directory, or neither it nor its new directory is writable.
    '''

    if path.is_dir():
        raise IsADirectoryError(f'{path} is a directory, not a file')
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if path.exists() else path.parent
    if not os.access(target, os.W_OK):
        raise PermissionError(f'{target} is not writable')

@app.command('regressor-panel')
def regressor_panel(
    coordinates: Annotated[
        str,
        typer.Option(
            '--coordinates',
            help="The arm's code table in the export form (tangent coordinates if hyperbolic)",
        ),
    ],
    text_only: Annotated[
        str,
        typer.Option('--text-only', help='The text-only table (tools text-only-table)'),
    ],
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    regime: Annotated[
        Optional[List[Regime]],
        typer.Option('--regime', help='Regime to score (repeatable; default: both)'),
    ] = None,
    level: Annotated[
        Optional[List[int]],
        typer.Option('--level', help='NAICS level 2-6 (repeatable; default: 6)'),
    ] = None,
    split: Annotated[
        str,
        typer.Option('--split', help='validation, or test (sealed: needs --open-purpose)'),
    ] = VALIDATION,
    purpose: Annotated[
        Optional[str],
        typer.Option(
            '--purpose',
            help='Why this read happens; recorded in the selection log '
            '(default: regressor panel <split> read)',
        ),
    ] = None,
    open_purpose: Annotated[
        Optional[str],
        typer.Option('--open-purpose', help='Why the outer sets are opened (test split only)'),
    ] = None,
    reopen_reason: Annotated[
        Optional[str],
        typer.Option('--reopen-reason', help='Required to open an outer set a second time'),
    ] = None,
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the regressor config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Write the per-row predictions to this parquet'),
    ] = None,
):
    '''
    Score an arm on the regressor panel: out-of-sample predictions per row (roadmap Stage 3).

    Every Req 2 comparator is fitted by ridge on standardized features, the penalty tuned inside
    the remainder. The validation split reads only the remainder; the test split opens each
    regime's sealed outer set first, and both the opening and the read are logged.

    Example:
        Score an arm's table on the validation split of both regimes at six digits::

            $ uv run naics-embedder tools regressor-panel --coordinates arm.parquet \\
                --text-only text_only.parquet --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('tools_regressor_panel.log')

    if split not in (VALIDATION, TEST):
        console.print(f'[bold red]--split must be {VALIDATION} or {TEST}, not {split!r}[/bold red]')
        raise typer.Exit(code=1)
    if split == TEST and not (open_purpose or '').strip():
        console.print(
            '[bold red]The outer sets are sealed: --split test needs --open-purpose, which is '
            'logged.[/bold red]'
        )
        raise typer.Exit(code=1)

    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    # A repeated regime is scored once: opening it twice would be refused after the first read
    regimes = list(dict.fromkeys(regime or list(Regime)))
    levels = sorted(set(level or [DECISION_LEVEL]))
    read_purpose = (purpose or '').strip() or f'regressor panel {split} read'
    output_path = Path(output) if output else None
    try:
        # The output path, the arm and every opening are checked before any opening: a test
        # read that failed after its opening would use the opening up
        if output_path is not None:
            _require_writable(output_path)
        panel = load_regressor_panel(cfg, codebook, log_path=log, levels=levels)
        arm = ArmTables.from_tables(pl.read_parquet(coordinates), pl.read_parquet(text_only))
        panel.require_arm(arm)
        defined, undefined = {}, []
        for chosen in regimes:
            defined[chosen] = []
            for number in levels:
                reason = panel.cell_status(chosen, number)
                if reason is None:
                    defined[chosen].append(number)
                else:
                    undefined.append((chosen.value, number, reason))
        to_open = [chosen for chosen, numbers in defined.items() if split == TEST and numbers]
        for chosen in to_open:
            panel.require_openable(chosen, reopen_reason)
        results = []
        for chosen, numbers in defined.items():
            if chosen in to_open:
                panel.open_outer(chosen, open_purpose or '', reopen_reason=reopen_reason)
            read = panel.validation if split == VALIDATION else panel.test
            results.extend(read(chosen, number, arm, read_purpose) for number in numbers)
    except (OSError, ValueError, SealedSplitError, SplitAlreadyOpenedError) as exc:
        console.print(f'[bold red]Regressor panel failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    for name, number, reason in undefined:
        console.print(f'  • {name}, level {number}: undefined ({reason})')
    if not results:
        console.print('[bold yellow]No regime was defined at the requested levels.[/bold yellow]')
        raise typer.Exit(code=1)

    predictions = pl.concat(results)
    console.print(f'\n[bold cyan]Regressor panel: {split} split[/bold cyan]\n')
    for row in summarize(predictions).iter_rows(named=True):
        console.print(
            f'  • {row["panel"]}, level {row["level"]}, {row["comparator"]}: rows {row["rows"]:,}, '
            f'RMSE {row["rmse"]:.4f}, R² {row["r2"]:.4f}, median alpha {row["median_alpha"]:g}'
        )
    console.print(f'\nReads logged to {panel.log.path} (fingerprint {panel.fingerprint})\n')

    if output_path is not None:
        try:
            predictions.write_parquet(output_path)
        except OSError as exc:
            console.print(f'[bold red]Predictions not written:[/bold red] {exc}')
            raise typer.Exit(code=1)
        console.print(f'Predictions written to {output_path}')

# -------------------------------------------------------------------------------------------------
# Decisions (Req 5)
# -------------------------------------------------------------------------------------------------

@app.command('margins')
def margins_command(
    reference: Annotated[
        str,
        typer.Option('--reference', help="The reference configuration's arm record (JSON)"),
    ],
    multiple: Annotated[
        float,
        typer.Option('--multiple', help="Each δ as a multiple of the reference's across-seed SD"),
    ],
    name: Annotated[
        str,
        typer.Option('--name', help='Names the margins in the decision records that use them'),
    ],
    store: Annotated[
        str,
        typer.Option('--store', help='The artifact store the arm record references'),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the margin record (JSON)'),
    ],
):
    '''
    Fix each panel's non-inferiority margin δ from a reference arm (Req 5).

    δ is the multiple times the reference arm's across-seed standard deviation of the panel's
    decision statistic (D10). Fix the margins before any other arm of a decision reads a panel: a
    decision refuses every run that read before its margins were fixed.

    Example:
        Fix the margins at half a standard deviation::

            $ uv run naics-embedder tools margins --reference reference.json --multiple 0.5 \\
                --name reference-margins --store ~/naics-artifacts --output margins.json
    '''

    configure_logging('tools_margins.log')

    cfg = load_config(DecisionConfig, DECISION_CONFIG)
    try:
        record = fix_margins(
            read_record(reference, ArmRecord),
            multiple,
            name,
            ArtifactStore(store),
            min_seeds=cfg.min_seeds,
        )
        path = write_record(record, output)
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]Margins failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(
        f'\n[bold cyan]Margins {name!r}, fixed {record.fixed_at.isoformat()}[/bold cyan]\n'
    )
    for entry in record.margins:
        console.print(
            f'  • {entry.panel}: δ {entry.margin:.6g} = {multiple:g} × SD {entry.sd:.6g} of '
            f'{entry.statistic} over {len(entry.per_seed)} seeds'
        )
    console.print(f'\nMargin record: {path}\n')

@app.command('decide')
def decide_command(
    arm: Annotated[
        List[str],
        typer.Option('--arm', help='An arm record (JSON); repeat for each arm'),
    ],
    margins: Annotated[
        str,
        typer.Option('--margins', help='The margin record (tools margins)'),
    ],
    name: Annotated[
        str,
        typer.Option('--name', help='Names the decision'),
    ],
    question: Annotated[
        str,
        typer.Option('--question', help='What the decision settles, in a sentence'),
    ],
    store: Annotated[
        str,
        typer.Option('--store', help='The artifact store the arm records reference'),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the decision record (JSON)'),
    ],
):
    '''
    Decide among two or more arms under Req 5's rule over D8's three panels.

    Each A-against-B comparison reads Δ on paired resamples of each panel's units, seeds nested.
    A is adopted over B when it is non-inferior on all three panels (the 95 % interval's lower
    bound above −δ) and superior on at least one (the 98⅓ % interval above zero). The survivors
    are the arms no other arm is adopted over, and the tie order picks among them. The record
    carries the arms with their selection-log records and artifact references, the margins, every
    comparison, the non-dominated set, the tie order and the chosen arm.

    Example:
        Decide between a candidate and the reference::

            $ uv run naics-embedder tools decide --arm candidate.json --arm reference.json \\
                --margins margins.json --name dimension-8 --question "Is dimension 8 enough?" \\
                --store ~/naics-artifacts --output decision.json
    '''

    configure_logging('tools_decide.log')

    cfg = load_config(DecisionConfig, DECISION_CONFIG)
    try:
        record = decide(
            name,
            question,
            [read_record(path, ArmRecord) for path in arm],
            read_record(margins, MarginRecord),
            ArtifactStore(store),
            replicates=cfg.replicates,
            bootstrap_seed=cfg.bootstrap_seed,
            min_seeds=cfg.min_seeds,
        )
        path = write_record(record, output)
    except (OSError, ValueError, TieUnresolvedError) as exc:
        console.print(f'[bold red]Decision failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'\n[bold cyan]Decision {name!r}[/bold cyan]\n')
    for comparison in record.comparisons:
        verdict = 'adopted' if comparison.adopted else 'not adopted'
        console.print(f'  • {comparison.a} over {comparison.b}: {verdict}')
        for panel in comparison.panels:
            low, high = panel.noninferiority_interval
            upper_low, upper_high = panel.superiority_interval
            console.print(
                f'      {panel.panel}: Δ {panel.delta:+.4g}; 95 % [{low:+.4g}, {high:+.4g}] '
                f'against −δ {-panel.margin:.4g}; 98⅓ % [{upper_low:+.4g}, {upper_high:+.4g}]'
            )
    cycle = ' (dominance cycled)' if record.cycle else ''
    console.print(f'\nNon-dominated: {", ".join(record.non_dominated)}{cycle}')
    console.print(f'Tie order: {", ".join(record.tie_order)}')
    console.print(f'[bold]Chosen: {record.chosen}[/bold]')
    console.print(f'\nDecision record: {path}\n')
