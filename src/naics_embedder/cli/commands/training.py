# -------------------------------------------------------------------------------------------------
# Training Commands
# -------------------------------------------------------------------------------------------------
'''
CLI commands for training NAICS embedding models.

``train`` trains the text stage: Req 11's three terms over the query and code streams, on a code
cache refreshed from the live model, with each epoch read by D6's monitor on the outcome panel's
validation split. That MRR, ``val/outcome_mrr``, keeps the checkpoint, stops the run early and
steps the learning-rate plateau (spec 4.4).
'''

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import pytorch_lightning as pyl
import typer
from rich.console import Console
from rich.panel import Panel
from transformers import AutoTokenizer
from typing_extensions import Annotated

from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    EncoderArchitecture,
    contract_for_bundle,
    shared_encoder_architecture,
    validate_exact_resume,
)
from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import code_token_config, encode_token_rows
from naics_embedder.text_model.mixins import OUTCOME_MRR
from naics_embedder.text_model.monitor import MONITOR_RECORDS, OutcomeMonitor
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
    OutcomePanelConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.training import (
    TrainingResult,
    create_trainer,
    detect_hardware,
    effective_precision,
    parse_config_overrides,
    read_checkpoint,
    refuse_a_fresh_start_into_a_used_directory,
    refuse_a_resume_from_another_directory,
    refuse_a_resume_of_a_stopped_run,
    refuse_a_resume_under_other_settings,
    refuse_other_constructor_settings,
    resolve_checkpoint,
    run_settings,
    save_training_summary,
)
from naics_embedder.utils.utilities import STAGE3_EMBEDDING_PREFIX, pick_device
from naics_embedder.utils.validation import (
    require_valid_supervision_bundle,
    validate_training_config,
)

console = Console()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Model and DataModule Construction
# -------------------------------------------------------------------------------------------------

def build_model_from_config(
    cfg: Config,
    runtime_contract: CheckpointContract,
    bundle: ValidatedSupervisionBundle,
    *,
    run_settings: Dict[str, Any],
    monitor: Optional[OutcomeMonitor],
) -> NAICSContrastiveModel:
    '''
    Construct a fresh model under the validated bundle (P15).

    The model takes supervision only from the bundle, and every setting from the config: Req 11's
    three terms and their two logit scales, the radius bound, and the optimizer's AdamW, warmup
    and plateau (spec 4.5). ``run_settings`` (``utils/training.run_settings``) is saved in its
    hyperparameters, which the exact-resume guard compares (P19); the monitor reads each epoch
    and is not saved.
    '''

    return NAICSContrastiveModel(
        base_model_name=cfg.model.base_model_name,
        lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
        moe_hidden_dim=cfg.model.moe.hidden_dim,
        radius_bound=cfg.model.radius_bound,
        code_code_weight=cfg.loss.code_code_weight,
        radial_weight=cfg.loss.radial_weight,
        target_temperature=cfg.loss.target_temperature,
        radial_step=cfg.loss.radial_step,
        logit_scale_init=cfg.loss.logit_scale_init,
        logit_scale_range=tuple(cfg.loss.logit_scale_range),
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        lr_plateau_factor=cfg.training.lr_plateau_factor,
        lr_plateau_patience=cfg.training.lr_plateau_patience,
        load_balancing_coef=cfg.model.moe.load_balancing_coef,
        seed=cfg.seed,
        run_settings=run_settings,
        supervision_manifest_path=cfg.supervision.manifest_path,
        supervision_contract_version=cfg.supervision.contract_version,
        supervision_bundle=bundle,
        # The key the token cache resolves under (spec 4.8)
        summaries=summaries_identity(cfg.data_loader.tokenization.tokenizer_name),
        checkpoint_contract=runtime_contract,
        monitor=monitor,
    )

def build_datamodule_from_config(
    cfg: Config,
    bundle: ValidatedSupervisionBundle,
) -> NAICSDataModule:
    '''
    Construct the two-stream datamodule of the validated bundle (spec 4.3).

    It reads the token cache the export and the reads load (``code_token_config``), so an arm is
    exported from the token rows it trained on, and it draws each epoch's permutations from the
    run's seed, ``data_loader.queries_per_step`` queries a step.
    '''

    return NAICSDataModule(
        token_config=code_token_config(cfg),
        seed=cfg.seed,
        queries_per_step=cfg.data_loader.queries_per_step,
        supervision_manifest_path=cfg.supervision.manifest_path,
        supervision_contract_version=cfg.supervision.contract_version,
        supervision_bundle=bundle,
    )

def build_monitor_from_config(
    cfg: Config,
    bundle: ValidatedSupervisionBundle,
    checkpoint_dir: Path,
    *,
    selection_log: Union[str, Path],
) -> OutcomeMonitor:
    '''
    The run's D6 monitor (spec 4.4): the bundle's outcome panel, its reads logged to
    ``selection_log`` and kept in ``monitor_reads.jsonl`` in the run's checkpoint directory.

    It tokenizes queries with the token cache's tokenizer and window (``code_token_config``), as
    an arm's reads do. Building it writes nothing: the first read writes the log, and the first
    epoch's end the records file.

    Args:
        cfg: The run's configuration.
        bundle: The validated bundle, whose ``index_roles`` member holds the panel.
        checkpoint_dir: The run's checkpoint directory.
        selection_log: ``conf/data/outcome_panel.yaml``'s ``selection_log``.
    '''

    token_config = code_token_config(cfg)
    return OutcomeMonitor(
        OutcomePanel.from_bundle(bundle, selection_log),
        AutoTokenizer.from_pretrained(token_config.tokenizer_name),
        token_config.max_length,
        Path(checkpoint_dir) / MONITOR_RECORDS,
        purpose=(
            f'D6 monitor: the validation MRR that selects an epoch of {cfg.experiment_name} '
            f'(seed {cfg.seed})'
        ),
    )

def encoder_architecture_for(cfg: Config) -> EncoderArchitecture:
    '''
    The configured run's encoder record.

    It comes from the same helper the model builds its own record with, so the two cannot drift
    (spec 4.4).
    '''

    return shared_encoder_architecture(
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        backbone=cfg.model.base_model_name,
    )

def runtime_contract_for(cfg: Config, bundle: ValidatedSupervisionBundle) -> CheckpointContract:
    '''
    The checkpoint contract of the configured run, its encoder record and summaries included.

    Training's exact resume and the HGCN feeder compare this whole contract with a checkpoint's;
    export and reads take the encoder record from the checkpoint instead (spec 4.4).
    '''

    return contract_for_bundle(
        bundle.manifest,
        encoder=encoder_architecture_for(cfg),
        summaries=summaries_identity(cfg.data_loader.tokenization.tokenizer_name),
    )

# -------------------------------------------------------------------------------------------------
# Embedding Generation
# -------------------------------------------------------------------------------------------------

def generate_embeddings_from_checkpoint(
    checkpoint_path: str, config: Config, output_path: Optional[str] = None, batch_size: int = 32
) -> str:
    '''Generate hyperbolic embeddings parquet file from a trained checkpoint.

    Loads a trained model checkpoint, runs inference on all NAICS codes, and
    writes the resulting embeddings to a parquet file compatible with HGCN
    training. The checkpoint must carry the supervision contract of the configured run, the
    contract of its validated bundle; a checkpoint without one, or with another, is refused
    before its model loads. One trained under another objective, as every checkpoint saved
    before Stage 7 was, is refused first, and nothing migrates it (spec 4.5, D2).

    Args:
        checkpoint_path: Filesystem path to the PyTorch Lightning checkpoint
            that contains the trained contrastive model weights.
        config: Project configuration containing data paths used for token
            caching and parquet loading.
        output_path: Optional path for the embeddings parquet. When omitted,
            ``output/hyperbolic_projection/encodings.parquet`` is used.
        batch_size: Batch size to use during inference to balance throughput
            and memory usage.

    Returns:
        str: Filesystem path to the generated embeddings parquet file.
    '''

    logger.info('=' * 80)
    logger.info('GENERATING EMBEDDINGS FROM CHECKPOINT')
    logger.info('=' * 80)
    logger.info(f'Checkpoint: {checkpoint_path}')

    # Determine output path
    if output_path is None:
        # Use default location: ./output/hyperbolic_projection/encodings.parquet
        output_dir = Path('./output/hyperbolic_projection')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'encodings.parquet')

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'Output: {output_path}')

    # The checkpoint must carry the contract of the configured run's validated bundle
    bundle = require_valid_supervision_bundle(config)
    validate_exact_resume(checkpoint_path, runtime_contract_for(config, bundle))
    token_fingerprints = {
        'description_fingerprint': bundle.manifest.description_fingerprint,
        'codebook_fingerprint': bundle.manifest.codebook_fingerprint,
    }

    # Load device
    device = pick_device('auto')  # Auto-detect device
    logger.info(f'Device: {device}')

    # Load model from checkpoint
    logger.info('Loading model from checkpoint...')
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        supervision_manifest_path=str(bundle.manifest_path),
        supervision_bundle=bundle,
    )
    model.to(device).eval()
    logger.info('Model loaded successfully')

    # Load descriptions parquet
    descriptions_path = config.data_loader.streaming.descriptions_parquet
    logger.info(f'Loading NAICS descriptions from: {descriptions_path}')

    df = pl.read_parquet(descriptions_path).sort('index')
    logger.info(f'Loaded {df.height:,} NAICS codes')

    # The cache training reads; a missing or stale one is rebuilt (spec §5)
    logger.info('Loading tokenization cache...')
    token_cache = tokenization_cache(code_token_config(config), **token_fingerprints)
    logger.info('Tokenization cache loaded')

    # Every code through the shared encoder, in eval mode and without gradient
    logger.info(f'Generating embeddings (batch_size={batch_size})...')
    rows = [token_cache[index] for index in df.get_column('index').to_list()]
    embeddings = encode_token_rows(model, rows, batch_size=batch_size)['embedding']
    embedding_dim = embeddings.shape[1]
    logger.info(f'Generated embeddings: shape={tuple(embeddings.shape)}')

    # d + 1 Lorentz coordinates as hyp_e* columns, which HGCN finds by prefix
    emb_schema = {f'{STAGE3_EMBEDDING_PREFIX}{i}': pl.Float64 for i in range(embedding_dim)}
    emb_df = pl.DataFrame(embeddings.numpy(), schema=emb_schema, orient='row')

    # Combine with metadata, in the Int64 the feeder has always written
    base_df = df.select(
        pl.col('index').cast(pl.Int64),
        pl.col('level').cast(pl.Int64),
        pl.col('code'),
    )

    result_df = base_df.hstack(emb_df)

    # Save to parquet
    logger.info(f'Saving embeddings to: {output_path}')
    result_df.write_parquet(output_path)

    logger.info('=' * 80)
    logger.info('EMBEDDING GENERATION COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Embeddings saved: {output_path}')
    logger.info(f'Total codes: {df.height:,}')
    logger.info(f'Embedding dimension: {embedding_dim}')

    return output_path

# -------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------

def _stdin_is_terminal() -> bool:
    '''
    Whether stdin is a terminal, so that a question can be asked and answered.

    A remote launch reads stdin from ``/dev/null`` (spec 4.6), where ``typer.confirm`` would abort
    a finished run. A closed stdin (``sys.stdin`` is None) or a closed stream is no terminal
    either.
    '''
    stdin = sys.stdin
    if stdin is None:
        return False
    try:
        return stdin.isatty()
    except ValueError:  # A closed stream
        return False

def train(
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
    ckpt_path: Annotated[
        Optional[str],
        typer.Option(
            '--ckpt-path',
            help='Path to checkpoint file to resume from, or "last" to auto-detect last checkpoint',
        ),
    ] = None,
    checkpoint_load_mode: Annotated[
        CheckpointLoadMode,
        typer.Option(
            '--checkpoint-load-mode',
            help=(
                'exact, the one mode: resume the optimizer, epoch and monitor state (requires '
                'a matching supervision contract, checkpoint directory and run settings, and a '
                'run early stopping has not ended); nothing migrates weights (D2)'
            ),
        ),
    ] = CheckpointLoadMode.EXACT,
    skip_validation: Annotated[
        bool,
        typer.Option(
            '--skip-validation',
            help=(
                'Skip advisory pre-flight checks of data files and cache; the supervision '
                'bundle gate always runs'
            ),
        ),
    ] = False,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help=(
                "Config overrides (e.g., 'training.learning_rate=1e-4 "
                "data_loader.queries_per_step=64')"
            )
        ),
    ] = None,
):
    '''
    Train the NAICS text encoder: Req 11's three terms, selected on the validation MRR (D6).

    Loads the configuration, gates the supervision bundle, and builds the two-stream datamodule,
    the outcome monitor (the bundle's panel, logging to ``conf/data/outcome_panel.yaml``'s
    ``selection_log``, its records in the checkpoint directory) and the model, which records the
    run's settings. ``create_trainer``'s trainer keeps the earliest epoch with the highest
    ``val/outcome_mrr`` and ``last.ckpt``, and stops early on the same MRR.

    Before anything is built, P19's guards refuse a fresh start into a checkpoint directory that
    is not empty, and an exact resume from a checkpoint saved in another directory, under other
    run settings or another seed, or of a run that early stopping ended. Each refusal exits 1. A
    run that spent its epoch budget without an early stop is not refused: its resume trains
    nothing.

    Args:
        config_file: Path to the base YAML configuration file that describes
            data, model, and training settings. Defaults to ``conf/config.yaml``.
        ckpt_path: Optional checkpoint path to resume training. Use ``last`` to
            automatically pick up the latest checkpoint for the configured
            experiment. A checkpoint saved in another directory is refused, and so is one of a
            run that early stopping ended.
        checkpoint_load_mode: ``exact``, the one mode, resumes full training state and requires
            the checkpoint's supervision contract, checkpoint directory and run settings to match
            this run's, and early stopping not to have ended the run. The weights-only migration
            is deleted (roadmap D2); the option keeps its name and default.
        skip_validation: Skip advisory pre-flight checks for data files and tokenization
            cache. The mandatory supervision bundle gate is never skipped.
        overrides: Optional list of key-value override strings. Use dot notation
            to specify nested config values like ``training.learning_rate=1e-4``.

    Example:
        Train with default configuration::

            $ uv run naics-embedder train

        Resume from the last checkpoint, under the settings the run started with::

            $ uv run naics-embedder train --ckpt-path last

        Train under another learning rate, as a run of its own::

            $ uv run naics-embedder train experiment_name=lr-1e-5 training.learning_rate=1e-5
    '''

    configure_logging('train.log')

    console.rule('[bold green]Training NAICS Embedder[/bold green]')

    try:
        # Load configuration
        logger.info('Loading configuration...')
        cfg = Config.from_yaml(config_file)

        # Apply command-line overrides using centralized parsing
        if overrides:
            logger.info('Applying command-line overrides:')
            override_dict, invalid_overrides = parse_config_overrides(overrides)

            for invalid in invalid_overrides:
                console.print(f'[yellow]Warning:[/yellow] Skipping invalid override: {invalid}')

            if override_dict:
                logger.info('')
                cfg = cfg.override(override_dict)

        # Detect hardware once the config is final: on CUDA the trainer runs at the configured
        # training.trainer.precision, elsewhere at 32-true (spec 4.2)
        logger.info('Determining infrastructure...')
        hardware = detect_hardware(log_info=True, cuda_precision=cfg.training.trainer.precision)

        # Log GPU memory if available
        if hardware.gpu_memory:
            logger.info(
                f'GPU Memory: {hardware.gpu_memory["reserved_gb"]:.1f} GB used / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB total '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization, '
                f'{hardware.gpu_memory["free_gb"]:.1f} GB free)'
            )

        # Mandatory supervision gate: validate the bundle before any DataModule, checkpoint, or
        # model work. There is no fallback to legacy files, and no mode without a bundle (D2).
        bundle = require_valid_supervision_bundle(cfg)
        runtime_contract = runtime_contract_for(cfg, bundle)
        logger.info(
            f'Supervision bundle {runtime_contract.bundle_id} '
            f'({runtime_contract.contract_version}) validated'
        )

        # Run advisory pre-flight validation
        if not skip_validation:
            validation_result = validate_training_config(cfg)
            if not validation_result.valid:
                console.print('\n[bold red]Pre-flight validation failed:[/bold red]')
                for error in validation_result.errors:
                    console.print(f'  [red]✗[/red] {error}')
                console.print('\n[dim]Use --skip-validation to bypass these checks[/dim]')
                raise typer.Exit(code=1)

            for warning in validation_result.warnings:
                console.print(f'[yellow]⚠[/yellow] {warning}')

        # Display configuration summary
        summary_list_1 = [
            f'[bold]Experiment:[/bold] {cfg.experiment_name}',
            f'[bold]Seed:[/bold] {cfg.seed}\n',
            '[cyan]Data:[/cyan]',
            f'  • Queries per step: {cfg.data_loader.queries_per_step}\n',
        ]

        summary_list_3 = [
            '[cyan]Model:[/cyan]',
            f'  • Base: {cfg.model.base_model_name.split("/")[-1]}',
            f'  • LoRA rank: {cfg.model.lora.r}',
            f'  • Fusion: {cfg.model.fusion}',
            f'  • Dimension: {cfg.model.dimension}\n',
            '[cyan]Training:[/cyan]',
            f'  • Learning rate: {cfg.training.learning_rate}',
            f'  • Max epochs: {cfg.training.trainer.max_epochs}',
            f'  • Accelerator: {hardware.accelerator}',
            f'  • Precision: {hardware.precision}',
        ]

        # Add GPU memory info
        summary_list_4 = []
        if hardware.gpu_memory:
            summary_list_4.append('\n[cyan]GPU Memory:[/cyan]')
            summary_list_4.append(
                f'  • Used: {hardware.gpu_memory["reserved_gb"]:.1f} GB / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization)'
            )
            summary_list_4.append(f'  • Free: {hardware.gpu_memory["free_gb"]:.1f} GB')

        summary = '\n'.join(summary_list_1 + summary_list_3 + summary_list_4)

        console.print(
            Panel(
                summary,
                title='[yellow]Configuration Summary[/yellow]',
                border_style='yellow',
                expand=False,
            )
        )
        console.print('')

        # Seed for reproducibility
        logger.info(f'Setting random seed: {cfg.seed}\n')
        pyl.seed_everything(cfg.seed, verbose=False)

        # Handle checkpoint resumption using centralized utility
        checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
        checkpoint_info = resolve_checkpoint(
            ckpt_path, Path(cfg.dirs.checkpoint_dir), cfg.experiment_name
        )
        checkpoint_path = checkpoint_info.path

        if checkpoint_info.exists:
            console.print(f'[green]✓[/green] Using checkpoint: [cyan]{checkpoint_path}[/cyan]\n')
        elif ckpt_path:
            console.print(f'[yellow]Warning:[/yellow] Checkpoint not found at {ckpt_path}')
            console.print('Starting training from scratch.\n')

        # The settings the model records and an exact resume must match (P21, P31)
        settings = run_settings(
            cfg,
            accelerator=hardware.accelerator,
            precision=effective_precision(cfg, hardware.accelerator),
        )

        # P19's guards, before anything is built. Exact resume, the one checkpoint load mode,
        # restores the optimizer, epoch and monitor state, so it requires the checkpoint's
        # supervision contract, its checkpoint directory and its run settings to match, and a run
        # that early stopping has not ended: Lightning does not restore the stop, so that run
        # would train on (a run that spent its epoch budget without one trains nothing). A fresh
        # start requires an unused checkpoint directory
        exact_resume = bool(checkpoint_path)
        if exact_resume:
            validate_exact_resume(checkpoint_path, runtime_contract)
            saved = read_checkpoint(checkpoint_path)
            refuse_a_resume_from_another_directory(saved, checkpoint_dir)
            refuse_a_resume_under_other_settings(saved, settings, seed=cfg.seed)
            refuse_other_constructor_settings(saved, cfg)
            refuse_a_resume_of_a_stopped_run(saved, cfg.training.early_stopping_patience)
            logger.info(
                'Supervision contract, checkpoint directory and run settings match, and early '
                'stopping has not ended the run - resuming training from checkpoint'
            )
            console.print(
                '[cyan]Resuming training from checkpoint (will continue from saved epoch)[/cyan]\n'
            )
        else:
            refuse_a_fresh_start_into_a_used_directory(checkpoint_dir)

        # Initialize DataModule
        logger.info('Initializing DataModule...')
        datamodule = build_datamodule_from_config(cfg, bundle)

        # D6's monitor reads the bundle's outcome panel and logs to its selection log
        selection_log = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml').selection_log
        logger.info(f'Initializing the outcome monitor (selection log: {selection_log})...')
        monitor = build_monitor_from_config(
            cfg, bundle, checkpoint_dir, selection_log=selection_log
        )

        # Initialize a fresh model; Lightning restores exact-resume state in trainer.fit
        logger.info('Initializing Model...\n')
        model = build_model_from_config(
            cfg, runtime_contract, bundle, run_settings=settings, monitor=monitor
        )

        # Initialize Trainer: checkpointing and early stopping on the monitor's MRR (P17)
        logger.info('Initializing PyTorch Lightning Trainer...\n')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer, checkpoint_callback, early_stopping = create_trainer(cfg, hardware, checkpoint_dir)

        # Start training
        logger.info('Starting model training...\n')
        console.print('[bold cyan]Selection (D6):[/bold cyan]')
        console.print(
            f"  • {OUTCOME_MRR}: the outcome panel's validation MRR, read after each epoch"
        )
        console.print(
            '  • Keeps the earliest epoch with the highest MRR, and last.ckpt; stops after '
            f'{cfg.training.early_stopping_patience} epochs without a gain\n'
        )

        console.print(
            f'[bold yellow]Training for {cfg.training.trainer.max_epochs} epochs...[/bold yellow]\n'
        )

        # Exact resume passes the checkpoint to trainer.fit(); a fresh run passes None
        trainer.fit(model, datamodule, ckpt_path=checkpoint_path)

        # Training complete
        logger.info('Training complete!')
        logger.info(f'Best model checkpoint: {checkpoint_callback.best_model_path}')

        # The kept epoch's MRR, and whether early stopping ended the run
        early_stop_triggered = early_stopping.stopped_epoch > 0
        best_score = checkpoint_callback.best_model_score
        best_mrr = float(best_score) if best_score is not None else None

        if early_stop_triggered:
            logger.info(f'Early stopping triggered at epoch {early_stopping.stopped_epoch}')

        console.print(
            f'\n[bold green]✓ Training completed successfully![/bold green]\n'
            f'Best checkpoint: [cyan]{checkpoint_callback.best_model_path}[/cyan]\n'
        )

        # Print the MRR that kept the checkpoint as the final metric
        if best_mrr is not None:
            if early_stop_triggered:
                label = 'Final evaluation metric (early stopping)'
            else:
                label = 'Final evaluation metric'
            console.print(f'[bold]{label}:[/bold] [cyan]{OUTCOME_MRR} = {best_mrr:.6f}[/cyan]\n')
            logger.info(f'{label}: {OUTCOME_MRR} = {best_mrr:.6f}')

        # Save final config
        config_output_path = checkpoint_dir / 'config.yaml'
        cfg.to_yaml(str(config_output_path))
        console.print(f'Config saved: [cyan]{config_output_path}[/cyan]\n')

        # Save training summary artifacts for downstream evaluation and documentation
        training_result = TrainingResult(
            best_checkpoint_path=checkpoint_callback.best_model_path,
            last_checkpoint_path=str(checkpoint_dir / 'last.ckpt'),
            config_path=str(config_output_path),
            best_score=best_mrr,
            stopped_epoch=early_stopping.stopped_epoch if early_stop_triggered else -1,
            early_stopped=early_stop_triggered,
            metrics={'best_val_outcome_mrr': best_mrr},
        )

        summary_paths = save_training_summary(
            result=training_result, config=cfg, hardware=hardware, output_dir=checkpoint_dir
        )
        console.print(
            'Training summary saved: [cyan]'
            f'{summary_paths.get("yaml", summary_paths.get("json"))}[/cyan]\n'
        )

        # Ask about embeddings for HGCN training (the feeder stays until Stage 11), but only on a
        # terminal: a remote launch reads stdin from /dev/null, where typer.confirm would abort the
        # finished run (spec 4.5, "Stays"). Without one, the answer is the question's default, no.
        generate_embeddings = False
        if _stdin_is_terminal():
            console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
            generate_embeddings = typer.confirm(
                'Generate embeddings parquet file from this checkpoint?', default=False
            )
        else:
            logger.info('stdin is not a terminal: no HGCN embeddings question, so none generated')

        if generate_embeddings:
            logger.info('Generating embeddings from checkpoint...')
            embeddings_path = generate_embeddings_from_checkpoint(
                checkpoint_path=checkpoint_callback.best_model_path,
                config=cfg,
                output_path=None,  # Will use default location
            )
            console.print(
                f'\n[bold green]✓ Embeddings generated successfully![/bold green]\n'
                f'Embeddings saved to: [cyan]{embeddings_path}[/cyan]\n'
                f'This file can be used for HGCN training.\n'
            )

    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        console.print(f'\n[bold red]✗ Training failed:[/bold red] {e}\n')
        raise typer.Exit(code=1)
