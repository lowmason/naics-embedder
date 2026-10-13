'''
Deterministic economic candidates on the fixture growth panel (Req 23; Req 24).

A candidate is a coordinate table over the fixture codebook in the export form, with the
provenance its builder writes beside it. The candidates read the synthetic backbone and text of
``tests.fixtures.decision``, so the text-only table ``write_text_only`` writes over the codebook
is their D9 comparator.
'''

import json
from pathlib import Path

import polars as pl

from naics_embedder.decision.records import ArmSpec
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import spec

DIMENSION = 3

def candidate_spec(name: str, **overrides) -> ArmSpec:
    '''A deterministic candidate's spec: one component, flat, at the fixture dimension.'''

    fields = {'components': 1, 'dimension': DIMENSION, 'geometry': 'euclidean', **overrides}
    return spec(name, **fields)

def write_candidate(directory: Path, name: str, table: pl.DataFrame, **entries) -> Path:
    '''
    A candidate's table and the provenance its builder writes beside it, naming the table's
    sha256 and ``matrix_fingerprint``; ``entries`` join or replace the provenance's entries.
    '''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}.parquet'
    table.write_parquet(path)
    provenance = {
        'builder': 'fixture',
        'inputs': {
            'codebook': 'the fixture codebook'
        },
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': table_fingerprint(table),
        **entries,
    }
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path
