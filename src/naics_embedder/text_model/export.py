'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query (``encode_query_texts``). The HGCN feeder, the table export, the arm encoder and the
training monitor all encode through it, so a code or a query embeds the same way wherever it is
read. The last three default to batches of ``ENCODE_BATCH_SIZE``.

``export_code_table`` writes Req 2's form of an arm: ``code``, ``index``, ``level`` and
``e0 … e{d-1}``, each code's bounded tangent vector at the origin (R6, Req 13), in the bundle's
codebook order. Its provenance ties the table to its checkpoint.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
from naics_embedder.text_model.fields import CHANNELS, QUERY, tokenize_field
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
COORDINATES = (
    'the bounded tangent vector at the origin, r * u with r = R * tanh(|v| / R) (Req 13); '
    'no time coordinate'
)
# Rows per forward pass wherever an arm is encoded: the export, the reads, the training cache and
# the monitor. A chunk is trimmed to its longest text, so the batches set the backbone's shapes,
# and one size makes a live read and a read of the export agree bit for bit on the CPU (spec 4.4)
ENCODE_BATCH_SIZE = 32

# -------------------------------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------------------------------

def code_token_config(cfg: Config) -> TokenizationConfig:
    '''
    The tokenization cache training reads: ``train`` hands it to ``NAICSDataModule``.

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
    batch_size: int = ENCODE_BATCH_SIZE,
) -> Dict[str, torch.Tensor]:
    '''
    Encode token rows through the model in batches, in eval mode and without gradient.

    A row maps each field to its tokens (``input_ids``, ``attention_mask`` and ``present``), as
    the tokenization cache stores a code or ``tokenize_field`` returns a query. The batches go to
    the model's device, and the model is left in eval mode.

    Args:
        model: A model whose forward returns ``tangent``, ``embedding``, ``radius`` and
            ``direction``: the shared encoder, or the Lightning module that holds it.
        rows: The token rows, in output order.
        fields: The fields read from each row.
        batch_size: Rows per forward pass.

    Returns:
        ``tangent`` (N, d), ``embedding`` (N, d + 1), ``radius`` (N,) and ``direction`` (N, d),
        float64 on the CPU, in row order.

    Raises:
        ValueError: If there are no rows, or ``batch_size`` is not positive.
    '''

    if not rows:
        raise ValueError('there are no token rows to encode')
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, not {batch_size}')
    device = next(model.parameters()).device
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {
        'tangent': [],
        'embedding': [],
        'radius': [],
        'direction': [],
    }
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

def encode_query_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> torch.Tensor:
    '''
    Encode query texts through the model, each marked ``query:`` and tokenized at ``max_length``.

    The arm encoder and the training monitor both encode their queries here, so a read of the live
    model and a read of its exported arm put each query at the same point.

    Args:
        model: As ``encode_token_rows``.
        tokenizer: The token cache's tokenizer.
        texts: The query texts, in output order.
        max_length: The token cache's window.
        batch_size: Queries per forward pass.

    Returns:
        The queries' bounded tangent vectors at the origin (Q, d), float64 on the CPU.

    Raises:
        ValueError: As ``encode_token_rows``, if there are no texts.
    '''

    rows = [{QUERY: tokenize_field(tokenizer, QUERY, text, max_length)} for text in texts]
    return encode_token_rows(model, rows, fields=(QUERY, ), batch_size=batch_size)['tangent']

# -------------------------------------------------------------------------------------------------
# Loading an arm
# -------------------------------------------------------------------------------------------------

def load_arm_model(
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    *,
    summaries: Optional[str],
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[NAICSContrastiveModel, CheckpointContract]:
    '''
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). It must have been trained under Req 11's
    objective, its supervision fields must match ``bundle``, and its summaries ``summaries``.
    Curvature is fixed at 1 with no parameter (spec 4.2), so there is none to check.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        summaries: The sha256 of the summaries the read's token cache applies
            (``summaries_identity`` of its tokenizer); keyword-only with no default.
        device: Where the model runs.

    Returns:
        The model, in eval mode on ``device``, and the checkpoint's saved contract.

    Raises:
        ValueError: If the checkpoint was trained under another objective, such as every one
            saved before Stage 7 (spec 4.5; checked first), its supervision contract is not the
            bundle's, it was trained under other summaries, or it is of another encoder
            architecture. Nothing migrates such a checkpoint (D2).
    '''

    # Lightning checkpoints carry pickled hyperparameters; they are trusted artifacts of this
    # project's own training runs
    raw = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    contract = validate_supervision_contract(
        raw.get(CHECKPOINT_KEY), bundle.manifest, summaries=summaries
    )
    # Callback scores can be float64, which MPS cannot deserialize; move only the model below.
    # on_load_checkpoint refuses another encoder architecture before the state dict loads (D2)
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
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
    batch_size: int = ENCODE_BATCH_SIZE,
) -> Path:
    '''
    Export an arm's code table in Req 2's form, with its provenance beside it (spec 4.3).

    Every code goes through the checkpoint's model in eval mode, without gradient. The table
    holds ``code``, then ``index`` and ``level`` from the descriptions (Int64), then ``e0 …
    e{d-1}`` (float64): each code's bounded tangent vector at the origin, in the bundle's codebook
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

    model, contract = load_arm_model(
        checkpoint_path,
        bundle,
        summaries=summaries_identity(token_config.tokenizer_name),
        device=device,
    )
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
        # The tokenizer the codes were read with, which a read's queries must share
        'tokenizer': token_config.tokenizer_name,
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
