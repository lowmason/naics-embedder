'''
The three deterministic economic candidates (spec economic-candidates, 4.2).

Each builder reads the codebook (``code_id`` and ``code``) and, for text, the shared text-only
table, and writes a table in the export form with a provenance beside it: ``code``, ``index`` (the
codebook's ``code_id``), ``level`` (the code's length) and ``e0 … e{d-1}``, in ``code_id`` order.
No builder reads a panel or a QCEW file, and the provenance holds no timestamp, so two builds on
one machine write the same bytes.

- ``text-only``: the text-only table reduced to d by ``pca_reduce``, as a read reduces it.
- ``tree``: the classical scaling of the tree metric D* over the codebook (``scaling``).
- ``smoothed``: the text-only vectors smoothed over each code's lineage, then reduced to d
  (``smoothing``).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import pca_reduce, provenance_path, text_only_matrix
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model.scaling import classical_scaling
from naics_embedder.tree_model.smoothing import smooth_over_lineage
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
KINDS = ('text-only', 'tree', 'smoothed')
# The inputs each builder reads, as the provenance's ``inputs`` names them
BUILDER_INPUTS: Dict[str, Tuple[str, ...]] = {
    'text-only': ('codebook', 'text_only'),
    'tree': ('codebook', ),
    'smoothed': ('codebook', 'text_only'),
}
METHODS: Dict[str, str] = {
    'text-only': 'The shared text-only table reduced to the dimension by PCA, as a read reduces it.',
    'tree': (
        'Classical scaling of the tree metric D* over the codebook: the leading eigenvectors of '
        'B = -1/2 J (D* o D*) J, each scaled by the square root of its eigenvalue.'
    ),
    'smoothed': (
        'The text-only vectors, each replaced by the mean over its lineage with halving weights '
        '(the code 1, its parent 1/2, and so on, normalized), then reduced to the dimension by PCA.'
    ),
}
_COMMON_ENTRIES = frozenset(
    {
        'builder',
        'method',
        'dimension',
        'codes',
        'inputs',
        'table_sha256',
        'matrix_fingerprint',
        'library_versions',
    }
)
# What each builder's provenance holds: its keys are exactly these, and none is a timestamp
BUILDER_ENTRIES: Dict[str, FrozenSet[str]] = {
    'text-only': _COMMON_ENTRIES,
    'tree': _COMMON_ENTRIES | {'eigenvalues'},
    'smoothed': _COMMON_ENTRIES,
}
LIBRARIES = ('numpy', 'scikit-learn', 'polars')

# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

def _read_codebook(path: Path) -> Tuple[List[str], List[int]]:
    '''The codebook's codes and ``code_id`` values, in ``code_id`` order.'''

    codebook = pl.read_parquet(path).sort('code_id')
    return (
        codebook.get_column('code').cast(pl.Utf8).to_list(),
        codebook.get_column('code_id').cast(pl.Int64).to_list(),
    )

def _read_text_only(path: Path, codes: List[str]) -> Tuple[List[str], np.ndarray]:
    '''
    The text-only table's codes and vectors, as ``text_only_matrix`` reads them.

    Raises:
        ValueError: If the table's codes are not the codebook's, as ``ArmTables.from_tables``
            requires of a read: the same codes, each once.
    '''

    text_codes, vectors = text_only_matrix(pl.read_parquet(path))
    if len(text_codes) != len(codes) or set(text_codes) != set(codes):
        raise ValueError('the text-only table and the codebook cover different codes')
    return text_codes, vectors

def _in_codebook_order(rows: np.ndarray, row_codes: List[str], codes: List[str]) -> np.ndarray:
    '''The rows, which belong to ``row_codes``, in the order of ``codes``.'''

    position = {code: row for row, code in enumerate(row_codes)}
    return rows[[position[code] for code in codes]]

# -------------------------------------------------------------------------------------------------
# Building
# -------------------------------------------------------------------------------------------------

def build_candidate(
    kind: str,
    dimension: int,
    codebook: Path,
    output: Path,
    *,
    text_only: Optional[Path] = None,
) -> Path:
    '''
    Build one deterministic candidate's table and write it with its provenance beside it.

    Args:
        kind: One of ``KINDS``.
        dimension: The table's number of coordinates, d.
        codebook: The codebook parquet (``code_id`` and ``code``).
        output: The table's parquet path. The provenance is written beside it
            (``panels.text_only.provenance_path``).
        text_only: The text-only table, for the ``text-only`` and ``smoothed`` kinds; ``tree``
            reads none.

    Returns:
        The table's path.

    Raises:
        ValueError: If ``kind`` is unknown, ``text_only`` is missing for a kind that reads it or
            given for one that does not, the text-only table's codes are not the codebook's, the
            dimension is refused (``classical_scaling``, ``pca_reduce``), a code's lineage is
            incomplete (``smooth_over_lineage``) or ``table_fingerprint`` refuses the table.
        FileExistsError: If ``output`` exists.
    '''

    if kind not in KINDS:
        raise ValueError(f"unknown candidate kind '{kind}'; the kinds are {', '.join(KINDS)}")
    reads_text = 'text_only' in BUILDER_INPUTS[kind]
    if reads_text and text_only is None:
        raise ValueError(f'the {kind} builder reads the text-only table, and none is given')
    if not reads_text and text_only is not None:
        raise ValueError(f'the {kind} builder reads no text-only table, and one is given')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'a candidate table already exists: {output}')

    codebook = Path(codebook)
    codes, indices = _read_codebook(codebook)
    extra: Dict[str, Any] = {}
    if kind == 'tree':
        coordinates, eigenvalues = classical_scaling(
            tree_distance_matrix(codes).astype(np.float64), dimension
        )
        extra['eigenvalues'] = [float(value) for value in eigenvalues]
    else:
        text_codes, vectors = _read_text_only(Path(text_only), codes)
        if kind == 'text-only':
            # Reduced in the table's own row order, as ArmTables.from_tables reduces it, so a
            # read's reduced vectors are these bit for bit; then put in codebook order
            coordinates = _in_codebook_order(pca_reduce(vectors, dimension), text_codes, codes)
        else:
            ordered = _in_codebook_order(vectors, text_codes, codes)
            coordinates = pca_reduce(smooth_over_lineage(codes, ordered), dimension)

    schema = {f'{TABLE_PREFIX}{index}': pl.Float64 for index in range(coordinates.shape[1])}
    table = pl.DataFrame(
        {
            'code': codes,
            'index': indices,
            'level': [len(code) for code in codes],
        },
        schema={
            'code': pl.Utf8,
            'index': pl.Int64,
            'level': pl.Int64
        },
    ).hstack(pl.DataFrame(coordinates, schema=schema, orient='row'))
    # Fingerprinted before the write: table_fingerprint refuses a table no panel could read
    fingerprint = table_fingerprint(table)

    inputs = {'codebook': codebook}
    if reads_text:
        inputs['text_only'] = Path(text_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output)
    provenance: Dict[str, Any] = {
        'builder': kind,
        'method': METHODS[kind],
        'dimension': dimension,
        'codes': table.height,
        'inputs': {
            name: {
                'path': str(path),
                'sha256': sha256_file(path)
            }
            for name, path in inputs.items()
        },
        **extra,
        'table_sha256': sha256_file(output),
        'matrix_fingerprint': fingerprint,
        'library_versions': {
            name: version(name)
            for name in LIBRARIES
        },
    }
    provenance_path(output).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(f'Candidate {kind} ({table.height:,} codes, dimension {dimension}): {output}')
    return output
