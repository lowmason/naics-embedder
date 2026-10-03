'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query. The HGCN feeder, the table export and the arm encoder all encode through it, so a
code embeds the same way wherever it is read.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Mapping, Sequence

import torch

from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.utils.config import Config, TokenizationConfig

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
