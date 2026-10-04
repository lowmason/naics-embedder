'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query. The HGCN feeder, the table export and the arm encoder all encode through it, so a
code embeds the same way wherever it is read.

``export_code_table`` writes Req 2's form of an arm: ``code``, ``index``, ``level`` and
``e0 … e{d-1}``, each code's capped tangent vector at the origin (R6), in the bundle's codebook
order. Its provenance ties the table to its checkpoint.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import polars as pl
import torch

from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
    CheckpointContract,
    validate_supervision_contract,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
COORDINATES = 'the capped tangent vector at the origin (spec R6); no time coordinate'

# -------------------------------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------------------------------

def code_token_config(cfg: Config) -> TokenizationConfig:
    '''
    The tokenization cache training reads, as ``NAICSDataModule`` builds it.

    The descriptions and the window are the streaming ones, the tokenizer is the tokenization
    one, and the path is the default. Export and reads therefore load the cache file that
    training built.
    '''

    return TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.streaming.max_length,
    )

def encode_token_rows(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    fields: Sequence[str] = CHANNELS,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    '''
    Encode token rows through the model in batches, in eval mode and without gradient.

    A row maps each field to its tokens (``input_ids``, ``attention_mask`` and ``present``), as
    the tokenization cache stores a code or ``tokenize_field`` returns a query. The batches go to
    the model's device, and the model is left in eval mode.

    Args:
        model: A model whose forward returns ``tangent`` and ``embedding``: the shared encoder,
            or the Lightning module that holds it.
        rows: The token rows, in output order.
        fields: The fields read from each row.
        batch_size: Rows per forward pass.

    Returns:
        ``tangent`` (N, d) and ``embedding`` (N, d + 1), float64 on the CPU, in row order.

    Raises:
        ValueError: If there are no rows, or ``batch_size`` is not positive.
    '''

    if not rows:
        raise ValueError('there are no token rows to encode')
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, not {batch_size}')
    device = next(model.parameters()).device
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {'tangent': [], 'embedding': []}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = stack_text_inputs(rows[start:start + batch_size], fields)
            inputs = {
                field: {
                    name: tensor.to(device)
                    for name, tensor in tensors.items()
                }
                for field, tensors in batch.items()
            }
            output = model(inputs)
            for name, collected in parts.items():
                # .cpu() before the cast: casting an MPS tensor to float64 raises
                collected.append(output[name].cpu().to(torch.float64))
    return {name: torch.cat(collected) for name, collected in parts.items()}

# -------------------------------------------------------------------------------------------------
# Loading an arm
# -------------------------------------------------------------------------------------------------

def require_unit_curvature(hyper_parameters: Mapping[str, Any]) -> None:
    '''
    Refuse a checkpoint trained at a curvature other than 1 (spec R8).

    The table's tangent coordinates and the scorer's ``lorentz`` distance both assume c = 1. An
    absent curvature is the model's default, 1.

    Raises:
        ValueError: If the saved curvature is not 1.
    '''

    curvature = float(hyper_parameters.get('curvature', 1.0))
    if curvature != 1.0:
        raise ValueError(
            f'the checkpoint was trained at curvature {curvature:g}; export and reads take c = 1 '
            'only (spec R8)'
        )

def load_arm_model(
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    *,
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[NAICSContrastiveModel, CheckpointContract]:
    '''
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). Its supervision fields must match ``bundle``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        device: Where the model runs.

    Returns:
        The model, in eval mode on ``device``, and the checkpoint's saved contract.

    Raises:
        ValueError: If the curvature is not 1 (R8), the supervision contract is not the bundle's,
            or the checkpoint is of another encoder architecture (D2).
    '''

    # Lightning checkpoints carry pickled hyperparameters; they are trusted artifacts of this
    # project's own training runs
    raw = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    require_unit_curvature(raw.get('hyper_parameters', {}))
    contract = validate_supervision_contract(raw.get(CHECKPOINT_KEY), bundle.manifest)
    # on_load_checkpoint refuses another encoder architecture before the state dict loads (D2)
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        supervision_manifest_path=str(bundle.manifest_path),
        supervision_bundle=bundle,
    )
    return model.to(device).eval(), contract

# -------------------------------------------------------------------------------------------------
# The code table
# -------------------------------------------------------------------------------------------------

def export_code_table(
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    token_config: TokenizationConfig,
    output_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = 'cpu',
    batch_size: int = 32,
) -> Path:
    '''
    Export an arm's code table in Req 2's form, with its provenance beside it (spec 4.3).

    Every code goes through the checkpoint's model in eval mode, without gradient. The table
    holds ``code``, then ``index`` and ``level`` from the descriptions (Int64), then ``e0 …
    e{d-1}`` (float64): each code's capped tangent vector at the origin, in the bundle's codebook
    order. The provenance is ``<stem>_provenance.json``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        token_config: The token cache training read (``code_token_config``). Its
            ``descriptions_parquet`` is the arm's descriptions.
        output_path: The table's parquet path.
        device: Where the model runs.
        batch_size: Codes per forward pass.

    Returns:
        The table's path.

    Raises:
        ValueError: As ``load_arm_model``; if the descriptions' ``(index, code)`` rows are not
            the codebook's ``(code_id, code)`` rows; or if ``coordinate_matrix`` refuses the
            table, which is then not written.
    '''

    model, contract = load_arm_model(checkpoint_path, bundle, device=device)
    descriptions_path = Path(token_config.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path).sort('index')
    codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
    described = descriptions.select(pl.col('index').cast(pl.Int64), pl.col('code').cast(pl.Utf8))
    coded = codebook.select(pl.col('code_id').cast(pl.Int64), pl.col('code').cast(pl.Utf8))
    if described.rows() != coded.rows():
        raise ValueError(
            f"the descriptions' (index, code) rows are not bundle "
            f"{bundle.manifest.bundle_id}'s codebook (code_id, code) rows"
        )

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    rows = [cache[index] for index in descriptions.get_column('index').to_list()]
    tangent = encode_token_rows(model, rows, batch_size=batch_size)['tangent']
    schema = {f'{TABLE_PREFIX}{index}': pl.Float64 for index in range(tangent.shape[1])}
    table = descriptions.select(
        pl.col('code').cast(pl.Utf8),
        pl.col('index').cast(pl.Int64),
        pl.col('level').cast(pl.Int64),
    ).hstack(pl.DataFrame(tangent.numpy(), schema=schema, orient='row'))
    # Fingerprinted before the write: coordinate_matrix refuses a table no panel could read
    fingerprint = table_fingerprint(table)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    checkpoint_path = Path(checkpoint_path)
    provenance: Dict[str, Any] = {
        'checkpoint': {
            'path': str(checkpoint_path),
            'sha256': sha256_file(checkpoint_path)
        },
        'contract': contract.model_dump(mode='json'),
        'backbone': contract.encoder.backbone,
        'revision': model.encoder.backbone_revision,
        'max_length': token_config.max_length,
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': summaries_identity(token_config.tokenizer_name),
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
        'table_sha256': sha256_file(output_path),
        'matrix_fingerprint': fingerprint,
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'peft', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(f'Code table ({table.height:,} codes, dimension {tangent.shape[1]}): {output_path}')
    return output_path
