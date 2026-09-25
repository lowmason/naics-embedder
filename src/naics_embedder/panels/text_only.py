'''
The text-only comparator of the regressor panel (Req 2; roadmap D9).

D9: "the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
dimension". The table holds one vector per code at the backbone's hidden size; the panel reduces
it to the arm's dimension when it scores the arm.

- **Text.** The four channels the arm reads (``title``, ``description``, ``excluded``,
  ``examples``) from the arm's own descriptions file.
- **Pooling.** Each channel is mean-pooled over its tokens under the attention mask, as the arm's
  encoder pools (``text_model/encoder.py``). A code's vector is the mean over its present
  channels: an absent (null or blank) channel is masked out, never encoded as a placeholder.
- **Frozen.** The backbone runs in evaluation mode under ``torch.no_grad``, on the CPU in
  float32, with no adapter.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA

from naics_embedder.supervision.artifacts import sha256_file

logger = logging.getLogger(__name__)

CHANNELS = ('title', 'description', 'excluded', 'examples')
POOLING = 'attention-masked mean over tokens per channel, then the mean over present channels'
TEXT_ONLY_PREFIX = 't'

# -------------------------------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------------------------------

def _present(text: Optional[str]) -> bool:
    return text is not None and bool(text.strip())

def encode_code_texts(
    descriptions: pl.DataFrame,
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    '''
    Embed every code's text with a frozen backbone.

    Args:
        descriptions: One row per code with the four channel columns.
        model: A Hugging Face encoder returning ``last_hidden_state``.
        tokenizer: Its tokenizer.
        max_length: Tokens kept per channel text.
        batch_size: Texts per forward pass.

    Returns:
        One row per code, in ``descriptions`` order: the mean of its present channels'
        mean-pooled vectors (float64).

    Raises:
        ValueError: If a code has no present channel.
    '''

    model.eval()
    total: Optional[np.ndarray] = None
    counts = np.zeros(descriptions.height, dtype=np.int64)
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        present = [index for index, text in enumerate(texts) if _present(text)]
        for start in range(0, len(present), batch_size):
            batch = present[start:start + batch_size]
            tokens = tokenizer(
                [texts[index] for index in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = pooled.to(torch.float64).numpy()
            if total is None:
                total = np.zeros((descriptions.height, vectors.shape[1]), dtype=np.float64)
            total[batch] += vectors
            counts[batch] += 1
    if total is None or (counts == 0).any():
        raise ValueError('every code needs at least one present text channel')
    return total / counts[:, None]

def load_backbone(name: str, *, local_files_only: bool = True) -> Tuple[Any, Any, Optional[str]]:
    '''
    The backbone's model and tokenizer, from the local Hugging Face cache by default.

    Returns:
        ``(model, tokenizer, revision)``; ``revision`` is the resolved snapshot's commit hash.
    '''

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(name, local_files_only=local_files_only)
    return model.to('cpu').eval(), tokenizer, getattr(model.config, '_commit_hash', None)

# -------------------------------------------------------------------------------------------------
# The table
# -------------------------------------------------------------------------------------------------

def text_only_frame(codes: List[str], vectors: np.ndarray) -> pl.DataFrame:
    '''``code`` plus ``t0`` … ``t{h-1}`` (float64).'''

    schema = {f'{TEXT_ONLY_PREFIX}{index}': pl.Float64 for index in range(vectors.shape[1])}
    values = pl.DataFrame(vectors, schema=schema, orient='row')
    return pl.DataFrame({'code': codes}, schema={'code': pl.Utf8}).hstack(values)

def provenance_path(table_path: Path) -> Path:
    '''The provenance JSON written beside a text-only table.'''

    table_path = Path(table_path)
    return table_path.with_name(f'{table_path.stem}_provenance.json')

def build_text_only_table(
    descriptions_path: Path,
    output_path: Path,
    *,
    backbone: str,
    max_length: int,
    batch_size: int = 32,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Any = None,
    revision: Optional[str] = None,
) -> Path:
    '''
    Embed every code's text with the frozen backbone and write the table and its provenance.

    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
        The table's path.
    '''

    descriptions_path = Path(descriptions_path)
    descriptions = pl.read_parquet(descriptions_path).sort('code')
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    vectors = encode_code_texts(
        descriptions, model, tokenizer, max_length=max_length, batch_size=batch_size
    )
    table = text_only_frame(descriptions.get_column('code').to_list(), vectors)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)

    provenance: Dict[str, Any] = {
        'backbone': backbone,
        'revision': revision,
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'channels': list(CHANNELS),
        'pooling': POOLING,
        'max_length': max_length,
        'codes': table.height,
        'hidden_size': vectors.shape[1],
        'table_sha256': sha256_file(output_path),
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(
        f'Text-only table ({table.height:,} codes, width {vectors.shape[1]}): {output_path}'
    )
    return output_path

# -------------------------------------------------------------------------------------------------
# Reduction
# -------------------------------------------------------------------------------------------------

def pca_reduce(vectors: np.ndarray, dimension: int) -> np.ndarray:
    '''
    Reduce the vectors to ``dimension`` principal components (D9: the arm's dimension).

    Raises:
        ValueError: If ``dimension`` exceeds the number of vectors or their width.
    '''

    vectors = np.asarray(vectors, dtype=np.float64)
    if not 1 <= dimension <= min(vectors.shape):
        raise ValueError(
            f'cannot reduce {vectors.shape[0]:,} vectors of width {vectors.shape[1]} '
            f'to {dimension} components'
        )
    return PCA(n_components=dimension, svd_solver='full').fit_transform(vectors)
