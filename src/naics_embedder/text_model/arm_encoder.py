'''
An arm as the outcome panel reads it: queries through its checkpoint, codes from its table
(spec 4.3).

``ArmEncoder`` implements ``QueryCodeEncoder`` (``panels/outcome.py``):

- A query is marked ``query:`` and goes through the checkpoint's model.
- A code's vector is decoded from the table ``tools export-table`` wrote from the checkpoint.

Both pass through one float64 exp map at the origin. The code vectors a read decodes against are
therefore a fixed function of the table, and the ``matrix_fingerprint`` the read logs names them.
Checkpoint, table, encoder and distance are the pieces of Stage 4's ``SeedArtifacts``
(``decision/sweep.py``).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, Sequence, Union

import polars as pl
import torch
from transformers import AutoTokenizer

from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.text_model.export import encode_token_rows, load_arm_model
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
# The exp map at the origin
# -------------------------------------------------------------------------------------------------

def exp_map_origin(tangent: torch.Tensor) -> torch.Tensor:
    '''
    The exponential map at the origin of the curvature -1 hyperboloid (c = 1), in float64.

    It is ``HyperbolicHead``'s map, computed in float64 on the CPU whatever the tangent's device
    and dtype.

    Args:
        tangent: Tangent vectors at the origin (N, d).

    Returns:
        (time, space) rows (N, d + 1), float64 on the CPU.
    '''

    # .cpu() before the cast: casting an MPS tensor to float64 raises
    tangent = tangent.cpu().to(torch.float64)
    norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp(min=1e-8)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) / norm * tangent], dim=1)

# -------------------------------------------------------------------------------------------------
# The arm encoder
# -------------------------------------------------------------------------------------------------

class ArmEncoder:
    '''
    ``QueryCodeEncoder`` for one arm: its checkpoint's model and the table exported from it.

    Args:
        model: The arm's model, in eval mode (``load_arm_model``).
        tokenizer: The token cache's tokenizer.
        max_length: The token cache's window.
        table: The arm's exported table.
        checkpoint_sha256: The checkpoint file's SHA-256.
        batch_size: Queries per forward pass.

    Attributes:
        distance: The head's distance, ``'lorentz'`` for the hyperbolic head.
        table_fingerprint: The table's ``matrix_fingerprint``, which a read logs as ``table``.
        checkpoint_sha256: The checkpoint's SHA-256, which a read logs as ``checkpoint``.
    '''

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        max_length: int,
        table: pl.DataFrame,
        checkpoint_sha256: str,
        batch_size: int = 32,
    ):
        codes, matrix = coordinate_matrix(table)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.distance = model.encoder.head.distance
        self.table_fingerprint = matrix_fingerprint(codes, matrix)
        self.checkpoint_sha256 = checkpoint_sha256
        self._rows = {code: row for row, code in enumerate(codes)}
        self._tangent = torch.tensor(matrix, dtype=torch.float64)

    @classmethod
    def from_files(
        cls,
        checkpoint_path: Union[str, Path],
        table_path: Union[str, Path],
        bundle: ValidatedSupervisionBundle,
        token_config: TokenizationConfig,
        *,
        device: Union[str, torch.device] = 'cpu',
        batch_size: int = 32,
    ) -> 'ArmEncoder':
        '''
        The arm of a checkpoint and the table exported from it.

        The table's provenance is checked before the model loads.

        Args:
            checkpoint_path: The arm's Lightning checkpoint.
            table_path: The table ``tools export-table`` wrote from it.
            bundle: The configured supervision bundle.
            token_config: The token cache training read (``code_token_config``): its tokenizer
                and window tokenize the queries.
            device: Where the model runs.
            batch_size: Queries per forward pass.

        Raises:
            ValueError: If the table's provenance names another checkpoint, or the table file is
                not the one it names; or as ``load_arm_model``: a curvature other than 1 (R8),
                another supervision contract, or another encoder architecture (D2).
            FileNotFoundError: If the table or its provenance is missing.
        '''

        checkpoint_path, table_path = Path(checkpoint_path), Path(table_path)
        provenance = json.loads(provenance_path(table_path).read_text())
        checkpoint_sha256 = sha256_file(checkpoint_path)
        named = provenance['checkpoint']['sha256']
        if named != checkpoint_sha256:
            raise ValueError(
                f'the table was exported from another checkpoint: its provenance names {named}, '
                f'and {checkpoint_path} is {checkpoint_sha256}'
            )
        if provenance['table_sha256'] != sha256_file(table_path):
            raise ValueError(f'{table_path} is not the table its provenance names')
        model, _ = load_arm_model(checkpoint_path, bundle, device=device)
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
            max_length=token_config.max_length,
            table=pl.read_parquet(table_path),
            checkpoint_sha256=checkpoint_sha256,
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the model, then the exp map: (Q, d + 1), float64.'''

        tokens = [tokenize_field(self.tokenizer, QUERY, text, self.max_length) for text in texts]
        rows = [{QUERY: row} for row in tokens]
        tangent = encode_token_rows(self.model, rows, fields=(QUERY, ),
                                    batch_size=self.batch_size)['tangent']
        return exp_map_origin(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' table rows through the exp map: (C, d + 1), float64.

        Raises:
            ValueError: If a code has no row in the table.
        '''

        unknown = sorted(set(codes) - set(self._rows))
        if unknown:
            raise ValueError(f'the table has no row for {unknown[:5]} ({len(unknown)} codes)')
        return exp_map_origin(self._tangent[[self._rows[code] for code in codes]])
