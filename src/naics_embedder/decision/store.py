'''
The content-addressed store behind decision records' artifact references (roadmap Stage 4).

A file is copied to ``objects/<sha[:2]>/<sha>/<name>`` under the store root, and its reference is
that relative path with its sha256 and size. The store never deletes or overwrites: an object is
written once and verified again on every read. Records need these files until Stage 12, so the
root belongs outside any worktree (removing a worktree deletes its ignored files), and a Lambda
instance's store must be copied off before the instance terminates
(``specs/lambda-remote-workflow.md``).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union

import polars as pl

from naics_embedder.decision.records import ArtifactRef, TableRef, TextOnlyRef
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

# -------------------------------------------------------------------------------------------------
# Store
# -------------------------------------------------------------------------------------------------

class ArtifactStore:
    '''Immutable, content-addressed files under one root.'''

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser()

    def put(self, path: Union[str, Path]) -> ArtifactRef:
        '''
        Copy a file into the store (once per content and name) and return its reference.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the stored copy does not hash to the file's sha256.
        '''

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'{path} is not a file')
        digest = sha256_file(path)
        relative = Path('objects') / digest[:2] / digest / path.name
        target = self.root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, staging = tempfile.mkstemp(dir=target.parent, prefix='.incoming-')
            os.close(handle)
            shutil.copyfile(path, staging)
            os.replace(staging, target)
        if sha256_file(target) != digest:
            raise ValueError(f'{target} does not hash to {digest}')
        return ArtifactRef(path=relative.as_posix(), sha256=digest, bytes=target.stat().st_size)

    def put_frame(self, frame: pl.DataFrame, name: str) -> ArtifactRef:
        '''Store a frame as a parquet file named ``name``.'''

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            frame.write_parquet(path)
            return self.put(path)

    def put_table(self, path: Union[str, Path]) -> TableRef:
        '''Store an arm's code table (the export form) with its ``matrix_fingerprint``.'''

        reference = self.put(path)
        fingerprint = table_fingerprint(pl.read_parquet(self.resolve(reference)))
        return TableRef(**reference.model_dump(), matrix_fingerprint=fingerprint)

    def put_text_only(self, table_path: Union[str, Path]) -> TextOnlyRef:
        '''
        Store a text-only table and the provenance beside it (``tools text-only-table``).

        Raises:
            FileNotFoundError: If the provenance is missing.
            ValueError: If the provenance describes another file, or names another
                ``matrix_fingerprint`` than the table's.
        '''

        table_path = Path(table_path)
        provenance_file = provenance_path(table_path)
        if not provenance_file.is_file():
            raise FileNotFoundError(f'{provenance_file}: the text-only table has no provenance')
        provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
        table = self.put(table_path)
        if provenance['table_sha256'] != table.sha256:
            raise ValueError(f'{provenance_file} describes another file than {table_path}')
        fingerprint = text_only_fingerprint(pl.read_parquet(self.resolve(table)))
        if provenance.get('matrix_fingerprint', fingerprint) != fingerprint:
            raise ValueError(f'{provenance_file} names another matrix_fingerprint')
        return TextOnlyRef(
            table=TableRef(**table.model_dump(), matrix_fingerprint=fingerprint),
            provenance=self.put(provenance_file),
            backbone=provenance['backbone'],
            revision=provenance['revision'],
            descriptions_sha256=provenance['descriptions']['sha256'],
            max_length=provenance['max_length'],
        )

    def resolve(self, reference: ArtifactRef) -> Path:
        '''
        The stored file, verified.

        Raises:
            FileNotFoundError: If the store has no such file.
            ValueError: If the file no longer hashes to the reference's sha256.
        '''

        path = self.root / reference.path
        if not path.is_file():
            raise FileNotFoundError(f'{path}: not in the artifact store at {self.root}')
        if sha256_file(path) != reference.sha256:
            raise ValueError(f'{path} changed since it was stored')
        return path

    def read_frame(self, reference: ArtifactRef) -> pl.DataFrame:
        '''A stored parquet file, verified.'''

        return pl.read_parquet(self.resolve(reference))
