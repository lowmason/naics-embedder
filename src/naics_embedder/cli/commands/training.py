# -------------------------------------------------------------------------------------------------
# Training Commands
# -------------------------------------------------------------------------------------------------
'''
CLI commands for training NAICS embedding models.

The ``train`` command is the supported entry point and runs the dynamic
Structure-Aware Dynamic Curriculum (SADC) workflow. The legacy sequential
command is retained only for backwards compatibility and is hidden from the
public help output.
'''

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
import pytorch_lightning as pyl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    MigrationReport,
    containment_contract,
    contract_for_bundle,
    load_weights_only,
    validate_exact_resume,
)
from naics_embedder.text_model.dataloader.datamodule import (
    NAICSDataModule,
    legacy_token_fingerprints,
)
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
    TokenizationConfig,
)
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.training import (
    TrainingResult,
    detect_hardware,
    parse_config_overrides,
    resolve_checkpoint,
    save_training_summary,
)
from naics_embedder.utils.utilities import pick_device
from naics_embedder.utils.validation import (
    require_valid_supervision_bundle,
    validate_training_config,
)
from naics_embedder.utils.warnings import configure_warnings

# Apply centralized warning configuration
configure_warnings()

console = Console()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Model Construction and Checkpoint Migration
# -------------------------------------------------------------------------------------------------

def build_model_from_config(
    cfg: Config,
    runtime_contract: CheckpointContract,
    bundle: Optional[ValidatedSupervisionBundle],
) -> NAICSContrastiveModel:
    '''
    Construct a fresh model for the configured supervision mode.

    Repaired models take supervision only from the validated bundle: legacy structural inputs
    (distance matrix, relations parquet, LambdaRank weight) are never passed. Explicit legacy
    containment reads the old distance matrix and relations file for evaluation only.
    '''

    if cfg.supervision.mode == 'repaired':
        supervision_inputs = {
            'supervision_manifest_path': cfg.supervision.manifest_path,
            'supervision_bundle': bundle,
        }
    else:
        supervision_inputs = {
            'supervision_manifest_path': None,
            'supervision_bundle': None,
            'distance_matrix_path': cfg.data_loader.streaming.distance_matrix_parquet,
            'relations_parquet_path': cfg.data_loader.streaming.relations_parquet,
        }
    structural_preference = cfg.loss.structural_preference
    return NAICSContrastiveModel(
        base_model_name=cfg.model.base_model_name,
        lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
        moe_hidden_dim=cfg.model.moe.hidden_dim,
        temperature=cfg.loss.temperature,
        curvature=cfg.loss.curvature,
        hierarchy_weight=cfg.loss.hierarchy_weight,
        radius_reg_weight=cfg.loss.radius_reg_weight,
        level_radius_weight=cfg.loss.level_radius_weight,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        load_balancing_coef=cfg.model.moe.load_balancing_coef,
        eval_every_n_epochs=cfg.model.eval_every_n_epochs,
        eval_sample_size=cfg.model.eval_sample_size,
        base_margin=cfg.loss.base_margin,  # pyright: ignore[reportAttributeAccessIssue]
        tree_distance_alpha=cfg.curriculum.tree_distance_alpha,
        curriculum_phase1_end=cfg.curriculum.phase1_end,
        curriculum_phase2_end=cfg.curriculum.phase2_end,
        curriculum_phase3_end=cfg.curriculum.phase3_end,
        sibling_distance_threshold=cfg.curriculum.sibling_distance_threshold,
        curriculum_phase_mode=cfg.curriculum.phase_mode,
        curriculum_anneal=cfg.curriculum.anneal.model_dump(),
        fn_curriculum_start_epoch=cfg.curriculum.fn_curriculum_start_epoch,
        fn_cluster_every_n_epochs=cfg.curriculum.fn_cluster_every_n_epochs,
        fn_num_clusters=cfg.curriculum.fn_num_clusters,
        false_negative_config=cfg.false_negatives.model_dump(),
        parent_eval_top_k=cfg.model.parent_eval_top_k,
        child_eval_top_k=cfg.model.child_eval_top_k,
        supervision_contract_version=cfg.supervision.contract_version,
        supervision_mode=cfg.supervision.mode,
        structural_preference_weight=structural_preference.weight,
        structural_preference_margin=structural_preference.margin,
        structural_preference_temperature=structural_preference.temperature,
        structural_preference_tie_tolerance=structural_preference.tie_tolerance,
        selection_seed=cfg.seed,
        checkpoint_contract=runtime_contract,
        **supervision_inputs,
    )

def announce_legacy_containment() -> None:
    '''Prominently tag a legacy-containment run in logs and on the console.'''

    message = (
        'LEGACY CONTAINMENT: not contract-compliant Stage-3 training. Only local unmined '
        'contrastive learning and supervision-independent regularizers run; checkpoints are '
        'tagged legacy-containment and can never exact-resume into repaired training.'
    )
    logger.warning(message)
    console.print(f'[bold red]{message}[/bold red]\n')

def runtime_contract_for(
    cfg: Config, bundle: Optional[ValidatedSupervisionBundle]
) -> CheckpointContract:
    '''
    The checkpoint contract of the configured run.

    The supervision gate returns no bundle only for explicit legacy containment.
    '''

    if bundle is None:
        return containment_contract()
    return contract_for_bundle(bundle.manifest, cfg.supervision.mode)

def log_migration_report(report: MigrationReport) -> None:
    '''Report what a weights-only migration loaded, skipped, and left freshly initialized.'''

    logger.info(
        f'Weights-only migration: loaded {len(report.loaded)} encoder tensors, skipped '
        f'{len(report.skipped)} excluded tensors, {len(report.missing)} encoder tensors freshly '
        f'initialized, {len(report.unexpected)} unexpected'
    )
    for name in report.skipped:
        logger.info(f'  • skipped (excluded group): {name}')
    for name in report.missing:
        logger.warning(f'  • freshly initialized (absent from checkpoint): {name}')
    console.print(
        f'[cyan]Weights-only migration:[/cyan] loaded {len(report.loaded)}, skipped '
        f'{len(report.skipped)}, freshly initialized {len(report.missing)}; optimizer, epoch, '
        'curriculum, and sampler state start fresh\n'
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
    training. The checkpoint must carry the supervision contract of the configured run: the
    repaired contract of its validated bundle, or (only under explicit legacy containment) the
    legacy-containment tag. Untagged legacy or mismatched checkpoints are refused.

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

    # The checkpoint must carry the contract of the configured run: the validated repaired bundle,
    # or the explicit legacy-containment tag
    bundle = require_valid_supervision_bundle(config)
    validate_exact_resume(checkpoint_path, runtime_contract_for(config, bundle))
    if bundle is None:
        announce_legacy_containment()
        load_overrides = {}
        token_fingerprints = legacy_token_fingerprints(
            config.data_loader.streaming.descriptions_parquet
        )
    else:
        load_overrides = {
            'supervision_manifest_path': str(bundle.manifest_path),
            'supervision_bundle': bundle,
        }
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
        map_location=device,
        **load_overrides,
    )
    model.eval()
    model.to(device)
    logger.info('Model loaded successfully')

    # Load descriptions parquet
    descriptions_path = config.data_loader.streaming.descriptions_parquet
    logger.info(f'Loading NAICS descriptions from: {descriptions_path}')

    df = pl.read_parquet(descriptions_path).sort('index')
    logger.info(f'Loaded {df.height:,} NAICS codes')

    # Load tokenization cache
    tokenization_cfg = TokenizationConfig(
        descriptions_parquet=descriptions_path,
        tokenizer_name=config.data_loader.tokenization.tokenizer_name,
        max_length=config.data_loader.tokenization.max_length,
    )

    logger.info('Loading tokenization cache...')
    token_cache = tokenization_cache(tokenization_cfg, **token_fingerprints, use_locking=False)
    logger.info('Tokenization cache loaded')

    # Generate embeddings in batches
    logger.info(f'Generating embeddings (batch_size={batch_size})...')
    all_embeddings = []
    all_indices = []
    all_levels = []
    all_codes = []

    num_batches = (df.height + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, df.height)
            batch_df = df.slice(start_idx, end_idx - start_idx)

            # Prepare batch inputs
            channel_inputs = {
                'title': {
                    'input_ids': [],
                    'attention_mask': []
                },
                'description': {
                    'input_ids': [],
                    'attention_mask': []
                },
                'excluded': {
                    'input_ids': [],
                    'attention_mask': []
                },
                'examples': {
                    'input_ids': [],
                    'attention_mask': []
                },
            }

            batch_indices = []
            batch_levels = []
            batch_codes = []

            for row in batch_df.iter_rows(named=True):
                idx = row['index']
                batch_indices.append(idx)
                batch_levels.append(row['level'])
                batch_codes.append(row['code'])

                # Get tokenized inputs from cache
                tokens = token_cache[idx]

                for channel in ['title', 'description', 'excluded', 'examples']:
                    channel_inputs[channel]['input_ids'].append(
                        tokens[channel]['input_ids']  # pyright: ignore[reportArgumentType]
                    )
                    channel_inputs[channel]['attention_mask'].append(
                        tokens[channel]['attention_mask']  # pyright: ignore[reportArgumentType]
                    )

            # Stack tensors
            for channel in channel_inputs:
                channel_inputs[channel]['input_ids'] = torch.stack(  # pyright: ignore[reportArgumentType]
                    channel_inputs[channel]['input_ids']
                ).to(device)
                channel_inputs[channel]['attention_mask'] = torch.stack(  # pyright: ignore[reportArgumentType]
                    channel_inputs[channel]['attention_mask']
                ).to(device)

            # Run inference
            output = model(channel_inputs)
            embeddings = output['embedding']  # Hyperbolic embeddings (batch_size, embedding_dim+1)

            # Store embeddings
            all_embeddings.append(embeddings.cpu())
            all_indices.extend(batch_indices)
            all_levels.extend(batch_levels)
            all_codes.extend(batch_codes)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(
                    f'  Processed {end_idx:,} / {df.height:,} codes ({(end_idx / df.height) * 100:.1f}%)'
                )

    # Concatenate all embeddings
    logger.info('Concatenating embeddings...')
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # (N, embedding_dim+1)
    embedding_dim = all_embeddings_tensor.shape[1]

    logger.info(f'Generated embeddings: shape={all_embeddings_tensor.shape}')

    # Convert to numpy
    embeddings_np = all_embeddings_tensor.numpy()

    # Create DataFrame with hyp_e* columns
    emb_schema = {f'hyp_e{i}': pl.Float64 for i in range(embedding_dim)}
    emb_df = pl.DataFrame(embeddings_np, schema=emb_schema)

    # Combine with metadata
    base_df = pl.DataFrame({'index': all_indices, 'level': all_levels, 'code': all_codes})

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
                'exact: resume optimizer/epoch/curriculum state (requires a matching supervision '
                'contract); weights_only: load allowlisted encoder weights into a fresh run'
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
            help="Config overrides (e.g., 'training.learning_rate=1e-4 data.batch_size=64')"
        ),
    ] = None,
):
    '''
    Train the NAICS text encoder with contrastive learning.

    Orchestrates the complete training workflow including configuration loading,
    hardware detection, checkpoint management, and training execution with
    PyTorch Lightning. Supports resumption from checkpoints and runtime
    configuration overrides.

    Args:
        config_file: Path to the base YAML configuration file that describes
            data, model, and training settings. Defaults to ``conf/config.yaml``.
        ckpt_path: Optional checkpoint path to resume training. Use ``last`` to
            automatically pick up the latest checkpoint for the configured
            experiment. Specify a full path for cross-experiment resumption.
        checkpoint_load_mode: ``exact`` resumes full training state and requires the
            checkpoint's supervision contract to match the runtime bundle; ``weights_only``
            loads allowlisted encoder weights into a fresh run starting at epoch zero.
        skip_validation: Skip advisory pre-flight checks for data files and tokenization
            cache. The mandatory supervision bundle gate is never skipped.
        overrides: Optional list of key-value override strings. Use dot notation
            to specify nested config values like ``training.learning_rate=1e-4``.

    Example:
        Train with default configuration::

            $ uv run naics-embedder train

        Resume from last checkpoint with custom learning rate::

            $ uv run naics-embedder train --ckpt-path last training.learning_rate=1e-5
    '''

    configure_logging('train.log')

    console.rule('[bold green]Training NAICS Embedder[/bold green]')

    try:
        # Detect hardware using centralized utility
        logger.info('Determining infrastructure...')
        hardware = detect_hardware(log_info=True)

        # Log GPU memory if available
        if hardware.gpu_memory:
            logger.info(
                f'GPU Memory: {hardware.gpu_memory["reserved_gb"]:.1f} GB used / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB total '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization, '
                f'{hardware.gpu_memory["free_gb"]:.1f} GB free)'
            )

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

        # Mandatory supervision gate: validate the bundle before any DataModule, checkpoint, or
        # model work. There is no fallback from repaired training to legacy files; only an
        # explicit legacy_containment mode runs without a bundle.
        bundle = require_valid_supervision_bundle(cfg)
        runtime_contract = runtime_contract_for(cfg, bundle)
        if bundle is None:
            announce_legacy_containment()
        else:
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
            f'  • Batch size: {cfg.data_loader.batch_size}',
            f'  • Num workers: {cfg.data_loader.num_workers}\n',
        ]

        summary_list_3 = [
            '[cyan]Model:[/cyan]',
            f'  • Base: {cfg.model.base_model_name.split("/")[-1]}',
            f'  • LoRA rank: {cfg.model.lora.r}',
            '  • MoE: ',
            f'    - {cfg.model.moe.num_experts} experts\n',
            '[cyan]Training:[/cyan]',
            f'  • Learning rate: {cfg.training.learning_rate}',
            f'  • Max epochs: {cfg.training.trainer.max_epochs}',
            f'  • Accelerator: {hardware.accelerator}',
            f'  • Precision: {hardware.precision}',
        ]

        # Add GPU memory info and batch size suggestions
        summary_list_4 = []
        if hardware.gpu_memory:
            summary_list_4.append('\n[cyan]GPU Memory:[/cyan]')
            summary_list_4.append(
                f'  • Used: {hardware.gpu_memory["reserved_gb"]:.1f} GB / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization)'
            )
            summary_list_4.append(f'  • Free: {hardware.gpu_memory["free_gb"]:.1f} GB')

            # Conservative batch size suggestion
            current_batch_size = cfg.data_loader.batch_size
            if hardware.gpu_memory['free_gb'] > 8.0 and current_batch_size < 12:
                # Suggest 2x-3x current batch size conservatively
                suggested_batch = min(12, current_batch_size * 2)
                if suggested_batch > current_batch_size:
                    summary_list_4.append('\n[yellow]Batch Size Suggestion:[/yellow]')
                    summary_list_4.append(f'  • Current: {current_batch_size}')
                    summary_list_4.append(
                        f'  • Suggested: {suggested_batch} (conservative estimate)'
                    )
                    summary_list_4.append(
                        '  • [dim]Note: gpu_tools.py estimates are optimistic; '
                        'reduce suggested values by ~50%[/dim]'
                    )
                    summary_list_4.append(
                        f'  • [dim]Override with: data.batch_size={suggested_batch}[/dim]'
                    )

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

        # Initialize DataModule
        logger.info('Initializing DataModule...')

        datamodule = NAICSDataModule(
            descriptions_path=cfg.data_loader.streaming.descriptions_parquet,
            triplets_path=cfg.data_loader.streaming.triplets_parquet,
            tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
            streaming_config=cfg.data_loader.streaming.model_dump(),
            sampling_config=cfg.sampling.model_dump(),
            batch_size=cfg.data_loader.batch_size,
            num_workers=cfg.data_loader.num_workers,
            val_split=cfg.data_loader.val_split,
            seed=cfg.seed,
            supervision_mode=cfg.supervision.mode,
            supervision_manifest_path=cfg.supervision.manifest_path,
            supervision_contract_version=cfg.supervision.contract_version,
            supervision_bundle=bundle,
        )

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

        # Exact resume restores optimizer, epoch, curriculum, and sampler state, so it requires the
        # checkpoint's supervision contract to match before anything is constructed.
        exact_resume = bool(checkpoint_path) and checkpoint_load_mode is CheckpointLoadMode.EXACT
        if exact_resume:
            validate_exact_resume(checkpoint_path, runtime_contract)

        # Initialize a fresh model; Lightning restores exact-resume state in trainer.fit
        logger.info('Initializing Model with evaluation metrics...\n')
        model = build_model_from_config(cfg, runtime_contract, bundle)

        if checkpoint_path and not exact_resume:
            report = load_weights_only(model, checkpoint_path)
            log_migration_report(report)

        # Setup callbacks
        logger.info('Setting up callbacks and checkpointing...\n')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if exact_resume:
            logger.info('Supervision contract matches - resuming training from checkpoint')
            console.print(
                '[cyan]Resuming training from checkpoint (will continue from saved epoch)[/cyan]\n'
            )

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='naics-{epoch:02d}-{val/contrastive_loss:.4f}',
            monitor='val/contrastive_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )

        early_stopping = EarlyStopping(
            monitor='val/contrastive_loss',
            patience=3,  # Reduced from 5 to prevent training beyond optimal point
            mode='min',
            min_delta=0.0001,  # Minimum improvement required
            verbose=True,
        )

        # TensorBoard logger - ensure directory exists first
        tb_log_dir = Path(cfg.dirs.output_dir) / cfg.experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)

        tb_logger = TensorBoardLogger(save_dir=cfg.dirs.output_dir, name=cfg.experiment_name)

        # Don't add epoch progress callback - PyTorch Lightning's progress bar
        # already shows epoch info

        # Initialize Trainer
        logger.info('Initializing PyTorch Lightning Trainer...\n')

        # Use only 1 device as specified in config, even if multiple GPUs are available
        devices_to_use = cfg.training.trainer.devices if hasattr(
            cfg.training.trainer, 'devices'
        ) else 1

        # If using multiple devices, need to handle unused parameters in DDP
        strategy = 'auto'
        if devices_to_use > 1 and hardware.accelerator in ['cuda', 'gpu']:
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=True)

        trainer = pyl.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            accelerator=hardware.accelerator,
            devices=devices_to_use,
            strategy=strategy,
            precision=hardware.precision,  # type: ignore
            gradient_clip_val=cfg.training.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
            log_every_n_steps=cfg.training.trainer.log_every_n_steps,
            val_check_interval=cfg.training.trainer.val_check_interval,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            default_root_dir=cfg.dirs.output_dir,
        )

        # Start training
        logger.info('Starting model training with evaluation metrics...\n')
        console.print('[bold cyan]Evaluation metrics enabled:[/bold cyan]')
        console.print('  • Cophenetic correlation (hierarchy preservation)')
        console.print('  • NDCG@k (ranking quality: position-aware metric)')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        console.print('  • Distortion metrics (mean, std)\n')

        console.print(
            f'[bold yellow]Training for {cfg.training.trainer.max_epochs} epochs...[/bold yellow]\n'
        )

        # Only exact resume passes the checkpoint to trainer.fit(); a weights-only migration has
        # already loaded its encoder weights and must start fresh at epoch zero
        trainer_ckpt_path = checkpoint_path if exact_resume else None
        trainer.fit(model, datamodule, ckpt_path=trainer_ckpt_path)

        # Training complete
        logger.info('Training complete!')
        logger.info(f'Best model checkpoint: {checkpoint_callback.best_model_path}')

        # Check if early stopping was triggered and get the best loss
        early_stop_triggered = early_stopping.stopped_epoch > 0
        best_loss = early_stopping.best_score if early_stopping.best_score is not None else None

        if early_stop_triggered and best_loss is not None:
            logger.info(
                f'Early stopping triggered at epoch {early_stopping.stopped_epoch} '
                f'with best loss: {best_loss:.6f}'
            )

        console.print(
            f'\n[bold green]✓ Training completed successfully![/bold green]\n'
            f'Best checkpoint: [cyan]{checkpoint_callback.best_model_path}[/cyan]\n'
        )

        # Print the loss that decided early stopping as the final metric
        if best_loss is not None:
            if early_stop_triggered:
                label = 'Final evaluation metric (early stopping)'
            else:
                label = 'Final evaluation metric'
            console.print(
                f'[bold]{label}:[/bold] [cyan]val/contrastive_loss = {best_loss:.6f}[/cyan]\n'
            )
            logger.info(f'{label}: val/contrastive_loss = {best_loss:.6f}')

        # Save final config
        config_output_path = checkpoint_dir / 'config.yaml'
        cfg.to_yaml(str(config_output_path))
        console.print(f'Config saved: [cyan]{config_output_path}[/cyan]\n')

        # Save training summary artifacts for downstream evaluation and documentation
        training_result = TrainingResult(
            best_checkpoint_path=checkpoint_callback.best_model_path,
            last_checkpoint_path=str(checkpoint_dir / 'last.ckpt'),
            config_path=str(config_output_path),
            best_loss=float(best_loss) if best_loss is not None else None,
            stopped_epoch=early_stopping.stopped_epoch if early_stop_triggered else -1,
            early_stopped=early_stop_triggered,
            metrics={'best_val_loss': float(best_loss) if best_loss is not None else None},
        )

        summary_paths = save_training_summary(
            result=training_result, config=cfg, hardware=hardware, output_dir=checkpoint_dir
        )
        console.print(
            'Training summary saved: [cyan]'
            f'f{summary_paths.get("yaml", summary_paths.get("json"))}[/cyan]\n'
        )

        # Prompt to generate embeddings for HGCN training
        console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
        generate_embeddings = typer.confirm(
            'Generate embeddings parquet file from this checkpoint?', default=False
        )

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
