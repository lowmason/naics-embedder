# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------
'''
CLI utility commands for configuration, GPU optimization, and metrics analysis.

This module provides the ``tools`` command group with utilities for inspecting
configuration, visualizing training metrics, and investigating model behavior.

Commands:
    config: Display current training configuration.
    visualize: Plot the durable monitor and epoch health summary.
    outcome-baseline: Score the lexical stub encoder on the outcome panel's validation split.
    text-only-table: Embed every code's text with the arm's backbone, frozen (roadmap D9).
    regressor-panel: Score an arm on the regressor panel's validation or sealed test split.
    margins: Fix each panel's non-inferiority margin from a reference arm (Req 5).
    decide: Decide among arms under Req 5's rule over D8's three panels.
    diagnostics: Report Req 6's structural diagnostics over every codebook code.
    export-table: Export an arm's code table in Req 2's form, with its provenance (Stage 6).
    outcome-panel: Score an arm on the outcome panel's validation split (Stage 6).
    sweep: Read trained seeds on all three validation panels and write an arm record.
    radius-report: Check radius variation, geometry and loss gradients for a selected checkpoint.
'''

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import polars as pl
import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.decision.decide import _check_monitor_records, decide, fix_margins
from naics_embedder.decision.records import ArmRecord, ArmSpec, MarginRecord, read_record, write_record
from naics_embedder.decision.rule import TieUnresolvedError
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import run_seed_sweep
from naics_embedder.metrics.diagnostics import GEOMETRIES, diagnostics_report
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
from naics_embedder.panels.text_only import build_text_only_table, load_backbone, provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.checkpoint_runner import CheckpointRunner
from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.text_model.radius_report import ANCHOR_GRADIENT_PREFIX, radius_report, term_gradients
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.metrics_tools import visualize_metrics
from naics_embedder.utils.config import (
    Config,
    DecisionConfig,
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.training import effective_precision, parse_config_overrides, run_settings
from naics_embedder.utils.utilities import pick_device
from naics_embedder.utils.validation import ValidationError, require_valid_supervision_bundle

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
    Display the training configuration a run would use.

    Validates the configuration file over the defaults and displays the run's name, seed and
    inputs, then the settings every run records. A file that sets a key the configuration
    no longer has is refused (spec 4.5). A missing file, or one the configuration refuses,
    prints the error and exits 1.

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

    if not show_current_config(config_file):
        raise typer.Exit(code=1)

# -------------------------------------------------------------------------------------------------
# Visualize metrics
# -------------------------------------------------------------------------------------------------

@app.command('visualize')
def visualize(
    summary: Annotated[str, typer.Option('--summary', help="The run's epoch_summary.jsonl")],
    output_dir: Annotated[Optional[str],
                          typer.Option(
                              '--output-dir',
                              help='Plot directory (default: visualizations beside the summary)'
                          )] = None,
):
    '''
    Plot the run's monitor MRR, loss means, logit scales and radius mean and SD per level.

    Read ``epoch_summary.jsonl`` beside the run's checkpoints. The figure is ``epoch_metrics.png``
    in ``--output-dir``, or ``visualizations`` beside the summary by default.
    '''

    configure_logging('tools_visualize.log')
    try:
        result = visualize_metrics(
            summary=Path(summary), output_dir=Path(output_dir) if output_dir else None
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        console.print(f'[bold red]Error:[/bold red] {exc}')
        raise typer.Exit(code=1)
    console.print(f'Epoch metrics: {result["output_file"]} ({result["num_epochs"]} epochs)')

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
    console.print(f'Provenance: {provenance_path(path)}')

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

def _require_new_record(path: Path) -> None:
    '''
    Refuse a record file that exists, and require that it can be written.

    Raises:
        FileExistsError: If the path exists: a record is written once.
        OSError: As ``_require_writable``.
    '''

    if path.exists():
        raise FileExistsError(f'{path} exists; a record is written once')
    _require_writable(path)

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
        # Before the reference is read and resampled
        _require_new_record(Path(output))
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
        # Before any arm is read and resampled
        _require_new_record(Path(output))
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

# -------------------------------------------------------------------------------------------------
# Diagnostics (Req 6)
# -------------------------------------------------------------------------------------------------

@app.command('diagnostics')
def diagnostics_command(
    table: Annotated[
        str,
        typer.Option(
            '--table',
            help="The arm's code table in the export form (tangent coordinates if hyperbolic)",
        ),
    ],
    geometry: Annotated[
        str,
        typer.Option('--geometry', help='euclidean, spherical or hyperbolic'),
    ],
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    curvature: Annotated[
        float,
        typer.Option('--curvature', help="A hyperbolic arm's curvature magnitude"),
    ] = 1.0,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the report as JSON to this path'),
    ] = None,
):
    '''
    Report Req 6's structural diagnostics over every codebook code.

    Sector separation (an AUC), within-sector rank correlation (over queries and over sectors),
    MAP over ancestors, NDCG@5/10/20 with integer lowest-common-ancestor grades, the Pearson
    correlation of distance with D*, and parent retrieval@1/5 without the 522 unary pairs. The
    report describes an arm: nothing selects on it, and no statistic in it has a threshold.

    Example:
        Report on a hyperbolic arm's export::

            $ uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \\
                --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('tools_diagnostics.log')

    if geometry not in GEOMETRIES:
        console.print(f'[bold red]--geometry must be one of {list(GEOMETRIES)}[/bold red]')
        raise typer.Exit(code=1)
    try:
        codes = pl.read_parquet(codebook).get_column('code').to_list()
        report = diagnostics_report(
            pl.read_parquet(table), geometry, codebook_codes=codes, curvature=curvature
        )
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]Diagnostics failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    separation = report.sector_separation
    within = report.within_sector_rank_correlation
    ancestors = report.map_over_ancestors
    parents = report.parent_retrieval

    def formatted(value: Optional[float]) -> str:
        return 'undefined' if value is None else f'{value:.4f}'

    console.print(
        f'\n[bold cyan]Structural diagnostics (Req 6): {report.codes:,} codes, '
        f'{report.geometry}[/bold cyan]\n'
    )
    console.print(
        f'  • sector separation AUC: {separation.auc:.4f} ({separation.same_sector_pairs:,} '
        f'same-sector, {separation.cross_sector_pairs:,} cross-sector pairs)'
    )
    console.print(
        f'  • within-sector rank correlation: {formatted(within.mean_over_queries)} over '
        f'{within.queries - within.undefined_queries:,} queries, '
        f'{formatted(within.mean_over_sectors)} over {len(within.by_sector)} sectors '
        f'({within.undefined_queries:,} undefined)'
    )
    levels = ', '.join(f'level {level} {value:.4f}' for level, value in ancestors.by_level.items())
    console.print(
        f'  • MAP over ancestors: {ancestors.value:.4f} over {ancestors.queries:,} queries '
        f'({levels})'
    )
    ndcg = ', '.join(f'{k} {value.value:.4f}' for k, value in report.ndcg.items())
    console.print(f'  • NDCG: {ndcg}')
    console.print(
        f'  • distance Pearson with D*: {formatted(report.distance_pearson.value)} over '
        f'{report.distance_pearson.pairs:,} pairs'
    )
    at = ', '.join(f'@{k} {value:.4f}' for k, value in parents.at.items())
    console.print(
        f'  • parent retrieval: {at} over {parents.queries:,} queries '
        f'({parents.unary_pairs_excluded} unary pairs excluded)'
    )
    console.print('\nDescriptive only: nothing selects on these, and none has a threshold.\n')

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2) + '\n')
        console.print(f'Report written to {path}')

# -------------------------------------------------------------------------------------------------
# Shared-encoder arms: export and the outcome read (roadmap Stage 6)
# -------------------------------------------------------------------------------------------------

def _run_config(config_file: str, overrides: Optional[List[str]]) -> Config:
    '''
    The run's config as ``train`` resolves it: the YAML file, then ``key=value`` overrides.

    Raises:
        ValueError: If an override has no ``=``. ``train`` skips one with a warning, but a logged
            read must run on exactly the config it names (P18).
    '''

    cfg = Config.from_yaml(config_file)
    override_dict, invalid = parse_config_overrides(overrides)
    if invalid:
        raise ValueError(f'overrides take the form key=value, not {invalid}')
    return cfg.override(override_dict) if override_dict else cfg

def _run_bundle(cfg: Config) -> ValidatedSupervisionBundle:
    '''
    The configured supervision bundle, through ``train``'s gate.

    Raises:
        ValidationError: As ``require_valid_supervision_bundle``.
    '''

    return require_valid_supervision_bundle(cfg)

def _sweep_spec(cfg: Config, *, name: str, accelerator: str) -> ArmSpec:
    '''The arm's settings and text identities, from its config and cached backbone (P21, P32).'''

    _, _, revision = load_backbone(cfg.model.base_model_name)
    return ArmSpec(
        name=name,
        components=1,
        dimension=cfg.model.dimension,
        geometry='hyperbolic',
        backbone=cfg.model.base_model_name,
        backbone_revision=revision,
        descriptions_sha256=sha256_file(cfg.data_loader.streaming.descriptions_parquet),
        summaries_sha256=summaries_identity(cfg.data_loader.tokenization.tokenizer_name),
        max_length=cfg.data_loader.streaming.max_length,
        settings=run_settings(
            cfg, accelerator=accelerator, precision=effective_precision(cfg, accelerator)
        )
    )

@app.command('sweep')
def sweep_command(
    runs: Annotated[str,
                    typer.Option('--runs', help='Run directory pattern containing {seed}')],
    seed: Annotated[List[int],
                    typer.Option('--seed', help='Trained seed to read (repeatable)')],
    text_only: Annotated[str,
                         typer.Option('--text-only', help='Frozen-backbone comparator table')],
    store: Annotated[str, typer.Option('--store', help='Content-addressed artifact store')],
    output: Annotated[str,
                      typer.Option('--output', help='New arm record JSON; never overwritten')],
    purpose: Annotated[str,
                       typer.Option('--purpose', help='Why these validation reads happen')],
    name: Annotated[str,
                    typer.Option('--name', help='Arm name in its decision record')] = 'reference',
    accelerator: Annotated[str,
                           typer.Option(
                               '--accelerator', help='Training accelerator: cuda, mps or cpu'
                           )] = 'cuda',
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the outcome-panel config)')] = None,
    config_file: Annotated[
        str, typer.Option('--config', help='Training config YAML')] = 'conf/config.yaml',
    overrides: Annotated[Optional[List[str]],
                         typer.Argument(help='Training config overrides, as key=value')] = None,
):
    '''
    Read each trained seed once on all three validation panels and save its arm record.

    Every seed's checkpoints and monitor reads are checked before the first decision read. The
    record carries the monitor reads that selected each checkpoint. ``--accelerator`` describes
    training; the export and decision reads use this machine's device.
    '''

    configure_logging('tools_sweep.log')
    try:
        _require_new_record(Path(output))
        if not purpose.strip():
            raise ValueError('every selection-log record needs a purpose')
        if accelerator not in ('cuda', 'mps', 'cpu'):
            raise ValueError('--accelerator must be cuda, mps or cpu')
        if '{seed}' not in runs:
            raise ValueError('--runs must contain {seed}')
        if len(set(seed)) != len(seed):
            raise ValueError(f'a seed repeats: {seed}')
        directories = {number: Path(runs.format(seed=number)) for number in seed}
        cfg = _run_config(config_file, overrides)
        bundle = _run_bundle(cfg)
        spec = _sweep_spec(cfg, name=name, accelerator=accelerator)
        panel_cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')
        log_path = log or panel_cfg.selection_log
        panel = OutcomePanel.from_bundle(bundle, log_path)
        runner = CheckpointRunner(
            cfg, bundle, run_directory=directories.__getitem__, device=pick_device('auto')
        )
        for number in seed:
            selected = runner.check(spec, number)
            # Reuse check_arm's monitor gate before artifacts or decision reads exist. Only these
            # fields are read: no fake artifact references or incomplete records are constructed.
            arm = SimpleNamespace(spec=spec, panels=SimpleNamespace(outcome=panel.fingerprint))
            run = SimpleNamespace(
                seed=number,
                training_run=selected.training_run,
                checkpoint_epoch=selected.epoch,
                monitor_records=selected.monitor_records
            )
            _check_monitor_records(arm, run)
        regressor_cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
        regressor = load_regressor_panel(
            regressor_cfg,
            bundle.artifact_path('codebook'),
            log_path=log_path,
            levels=[DECISION_LEVEL]
        )
        record = run_seed_sweep(
            spec,
            seed,
            runner,
            outcome_panel=panel,
            regressor_panel=regressor,
            text_only_table=text_only,
            store=ArtifactStore(store),
            purpose=purpose
        )
        write_record(record, output)
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        console.print(f'[bold red]Sweep failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'Arm record: {output} ({len(record.runs)} seeds)')

@app.command('radius-report')
def radius_report_command(
    checkpoint: Annotated[str,
                          typer.Option('--checkpoint', help='Selected checkpoint to check')],
    table: Annotated[str, typer.Option('--table', help='Table exported from that checkpoint')],
    output: Annotated[Optional[str],
                      typer.Option('--output', help='Radius and gradient report JSON')] = None,
    config_file: Annotated[str,
                           typer.Option(
                               '--config', help='Config naming the bundle and token cache'
                           )] = 'conf/config.yaml',
    overrides: Annotated[Optional[List[str]],
                         typer.Argument(help='Config overrides, as key=value')] = None,
):
    '''
    Check radius variation, geometry and loss gradients for one selected checkpoint.

    Gradients use epoch 0, step 0 of the saved seed and saved query chunk size. The checkpoint
    and its table must share an export provenance. The report is written even when a measured
    criterion fails; a failure exits 1. No evaluation split is scored.
    '''

    configure_logging('tools_radius_report.log')
    try:
        if output:
            _require_writable(Path(output))
        cfg = _run_config(config_file, overrides)
        bundle = _run_bundle(cfg)
        tokens = code_token_config(cfg)
        # This checks the checkpoint/table and preprocessing identities without a panel read.
        encoder = ArmEncoder.from_files(checkpoint, table, bundle, tokens, device='cpu')
        model = encoder.model
        saved = model.hparams.run_settings or {}
        data = NAICSDataModule(
            tokens,
            seed=model.hparams.seed,
            queries_per_step=saved.get('queries_per_step', cfg.data_loader.queries_per_step),
            supervision_manifest_path=str(bundle.manifest_path),
            supervision_bundle=bundle
        )
        data.prepare_data()
        data.setup('fit')
        data.set_train_epoch(0)
        batch = data.train_dataset[0]
        model.refresh_code_cache(data.code_rows)
        gradients = term_gradients(model, batch)
        anchors = [
            gradients.pop(f'{ANCHOR_GRADIENT_PREFIX}{row}')
            for row in range(len(batch['codes']['ids']))
        ]
        radius = radius_report(pl.read_parquet(table), anchor_radius_gradient=np.array(anchors))
        inert = [
            name for name, value in gradients.items() if not (np.isfinite(value) and value > 0)
        ]
        report = {
            'checkpoint': str(Path(checkpoint).resolve()),
            'table': str(Path(table).resolve()),
            'seed': int(model.hparams.seed),
            'epoch': 0,
            'step': 0,
            'anchor_ids': batch['codes']['ids'].tolist(),
            'radius': asdict(radius),
            'term_gradients': {
                name: value if np.isfinite(value) else None
                for name, value in gradients.items()
            },
            'inert_terms': inert,
            'passed': radius.passed and not inert
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        console.print_json(rendered)
    except (OSError, ValueError, RuntimeError, KeyError, ValidationError) as exc:
        console.print(f'[bold red]Radius report failed:[/bold red] {exc}')
        raise typer.Exit(code=1)
    if not report['passed']:
        raise typer.Exit(code=1)

@app.command('export-table')
def export_table(
    checkpoint: Annotated[
        str,
        typer.Option('--checkpoint', help="The arm's checkpoint: a training run's last.ckpt, say"),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='The table parquet; its provenance is written beside it'),
    ],
    config_file: Annotated[
        str,
        typer.Option('--config', help='Config YAML naming the bundle and the token cache'),
    ] = 'conf/config.yaml',
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Config overrides, as train takes them (e.g., 'model.dimension=8')"),
    ] = None,
):
    '''
    Export an arm's code table in Req 2's form, with its provenance (roadmap Stage 6).

    Every code goes through the checkpoint's model in eval mode. The table holds ``code``,
    ``index``, ``level`` and ``e0 … e{d-1}``: each code's tangent vector at the origin, in the
    bundle's codebook order. The checkpoint's supervision contract must match the configured
    bundle. Its encoder record is its own, so a d = 8 checkpoint exports under a d = 16 config.

    Example:
        Export a run's last checkpoint::

            $ uv run naics-embedder tools export-table \\
                --checkpoint checkpoints/sadc_default/last.ckpt \\
                --output data/plan8/arm_table.parquet supervision.manifest_path=PATH
    '''

    configure_logging('tools_export_table.log')

    output_path = Path(output)
    try:
        # A bad output path fails before the whole codebook is encoded
        _require_writable(output_path)
        cfg = _run_config(config_file, overrides)
        table = export_code_table(
            checkpoint,
            _run_bundle(cfg),
            code_token_config(cfg),
            output_path,
            device=pick_device('auto'),
        )
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f'[bold red]Export failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'Code table: {table}')
    console.print(f'Provenance: {provenance_path(table)}')

@app.command('outcome-panel')
def outcome_panel(
    checkpoint: Annotated[
        str,
        typer.Option('--checkpoint', help='The arm checkpoint the table was exported from'),
    ],
    table: Annotated[
        str,
        typer.Option('--table', help='The table tools export-table wrote from the checkpoint'),
    ],
    purpose: Annotated[
        str,
        typer.Option('--purpose', help='Why this read happens; recorded in the selection log'),
    ],
    config_file: Annotated[
        str,
        typer.Option('--config', help='Config YAML naming the bundle and the token cache'),
    ] = 'conf/config.yaml',
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the outcome-panel config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the summary as JSON to this path'),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Config overrides, as train takes them (e.g., 'model.dimension=8')"),
    ] = None,
):
    '''
    Score an arm on the outcome panel's validation split under its own distance (Stage 6).

    Queries go through the checkpoint's model; codes are decoded from the table exported from it.
    The read is logged with the table's ``matrix_fingerprint`` and the checkpoint's SHA-256. The
    test split stays sealed: this command never opens it.

    Example:
        Read the validation split for an exported table::

            $ uv run naics-embedder tools outcome-panel \\
                --checkpoint checkpoints/sadc_default/last.ckpt \\
                --table data/plan8/arm_table.parquet --purpose 'Stage 6 Exit reading' \\
                supervision.manifest_path=PATH
    '''

    configure_logging('tools_outcome_panel.log')

    panel_cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')
    try:
        # A bad output path fails before the read is logged
        if output:
            _require_writable(Path(output))
        cfg = _run_config(config_file, overrides)
        bundle = _run_bundle(cfg)
        encoder = ArmEncoder.from_files(
            checkpoint, table, bundle, code_token_config(cfg), device=pick_device('auto')
        )
        panel = OutcomePanel.from_bundle(bundle, log or panel_cfg.selection_log)
        result = read_outcome_validation(encoder, panel, purpose)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f'[bold red]Outcome panel failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Outcome panel: arm, validation split[/bold cyan]\n')
    for key, value in result.summary.items():
        formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
        console.print(f'  • {key}: {formatted}')
    console.print(f'\nRead logged to {panel.log.path} (table {encoder.table_fingerprint})\n')

    if output:
        path = Path(output)
        payload = {
            'fingerprint': panel.fingerprint,
            'table': encoder.table_fingerprint,
            'checkpoint': encoder.checkpoint_sha256,
            'summary': result.summary,
        }
        path.write_text(json.dumps(payload, indent=2) + '\n')
