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
from typing import Any, Dict, Mapping, Union

import polars as pl

from naics_embedder.decision.records import ArtifactRef, TableRef, TextOnlyRef
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

# -------------------------------------------------------------------------------------------------
# Table provenance (text-only and export)
# -------------------------------------------------------------------------------------------------

def provenance_fields(
    provenance: Mapping[str, Any], table_sha256: str, matrix_fingerprint: str, name: str
) -> Dict[str, Any]:
    '''
    The fields D9 checks, from a table's provenance that describes the table.

    A text-only table's provenance (``tools text-only-table``) and an exported table's
    (``tools export-table``) both carry them.

    Args:
        provenance: The provenance's JSON content.
        table_sha256: The table's sha256.
        matrix_fingerprint: The table's ``matrix_fingerprint``.
        name: Names the provenance in an error.

    Raises:
        ValueError: If the provenance is not a JSON object, describes another file, names
            another ``matrix_fingerprint``, or lacks a field D9 reads.
    '''

    if not isinstance(provenance, Mapping):
        raise ValueError(f'{name} is not a JSON object')
    if provenance.get('table_sha256') != table_sha256:
        raise ValueError(f'{name} describes another file than the table')
    if provenance.get('matrix_fingerprint', matrix_fingerprint) != matrix_fingerprint:
        raise ValueError(f'{name} names another matrix_fingerprint')
    try:
        return {
            'backbone': provenance['backbone'],
            'revision': provenance['revision'],
            'descriptions_sha256': provenance['descriptions']['sha256'],
            'summaries_sha256': provenance['summaries'],
            'max_length': provenance['max_length'],
        }
    except KeyError as exc:
        raise ValueError(f'{name} lacks the field {exc}') from None
    except TypeError:
        # The provenance is an object, so only its descriptions entry can refuse a key
        raise ValueError(f'{name}: its descriptions field is not a JSON object') from None

# -------------------------------------------------------------------------------------------------
# Store
# -------------------------------------------------------------------------------------------------

class ArtifactStore:
    '''
    Immutable, content-addressed files under one root.

    The root is made absolute, so a record names the same store from any working directory, and
    no reference may lead out of it, whether through ``..``, an absolute path or a symlink.
    '''

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()

    def put(self, path: Union[str, Path]) -> ArtifactRef:
        '''
        Copy a file into the store (once per content and name) and return its reference.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file changes while it is copied, or the stored copy does not hash
                to the file's sha256.
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
            try:
                shutil.copyfile(path, staging)
                # Only a copy that hashes to the digest is ever renamed into place
                if sha256_file(staging) != digest:
                    raise ValueError(f'{path} changed while it was copied into the store')
                os.replace(staging, target)
            finally:
                Path(staging).unlink(missing_ok=True)
        if sha256_file(target) != digest:
            raise ValueError(f'{target} does not hash to {digest}')
        return ArtifactRef(path=relative.as_posix(), sha256=digest, bytes=target.stat().st_size)

    def put_frame(self, frame: pl.DataFrame, name: str) -> ArtifactRef:
        '''
        Store a frame as a parquet file named ``name``.

        Raises:
            ValueError: If ``name`` is not a plain file name.
        '''

        if name in ('', '..') or Path(name).name != name:
            raise ValueError(f'{name!r} is not a file name')
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
        Store a text-only table and the provenance beside it (``tools text-only-table``), once
        the provenance is checked to describe the table.

        Raises:
            FileNotFoundError: If the provenance is missing.
            ValueError: As ``provenance_fields``, before anything is stored, or if either file
                changes while it is stored.
        '''

        table_path = Path(table_path)
        provenance_file = provenance_path(table_path)
        if not provenance_file.is_file():
            raise FileNotFoundError(f'{provenance_file}: the text-only table has no provenance')
        fingerprint = text_only_fingerprint(pl.read_parquet(table_path))
        provenance_fields(
            json.loads(provenance_file.read_text(encoding='utf-8')),
            sha256_file(table_path),
            fingerprint,
            str(provenance_file),
        )
        table = TableRef(**self.put(table_path).model_dump(), matrix_fingerprint=fingerprint)
        return self.text_only(table, self.put(provenance_file))

    def text_only(self, table: TableRef, provenance: ArtifactRef) -> TextOnlyRef:
        '''
        A stored text-only table's reference, with the fields D9 checks read from its stored
        provenance.

        Raises:
            FileNotFoundError: If either file is not in the store.
            ValueError: If either file changed since it was stored, or as ``provenance_fields``.
        '''

        self.resolve(table)
        content = json.loads(self.resolve(provenance).read_text(encoding='utf-8'))
        fields = provenance_fields(
            content, table.sha256, table.matrix_fingerprint, f'the stored {provenance.path}'
        )
        return TextOnlyRef(table=table, provenance=provenance, **fields)

    def resolve(self, reference: ArtifactRef) -> Path:
        '''
        The stored file, verified.

        Raises:
            FileNotFoundError: If the store has no such file.
            ValueError: If the reference leads out of the store, or the file no longer hashes to
                the reference's sha256.
        '''

        path = (self.root / reference.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'{reference.path} leaves the artifact store at {self.root}')
        if not path.is_file():
            raise FileNotFoundError(f'{path}: not in the artifact store at {self.root}')
        if sha256_file(path) != reference.sha256:
            raise ValueError(f'{path} changed since it was stored')
        return path

    def read_frame(self, reference: ArtifactRef) -> pl.DataFrame:
        '''A stored parquet file, verified.'''

        return pl.read_parquet(self.resolve(reference))
