'''
Versioned supervision artifact I/O.

Every Parquet file of a supervision bundle carries its contract version, bundle ID, and schema
version in file metadata, and every member file is recorded in the manifest with its SHA-256 hash
and row count.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import hashlib
from pathlib import Path
from typing import Iterable, Tuple

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from naics_embedder.supervision.schema import ArtifactFile

# -------------------------------------------------------------------------------------------------
# Contract metadata keys
# -------------------------------------------------------------------------------------------------

METADATA_CONTRACT = b'naics_embedder.contract_version'
METADATA_BUNDLE = b'naics_embedder.bundle_id'
METADATA_SCHEMA = b'naics_embedder.schema_version'

# -------------------------------------------------------------------------------------------------
# Hashing
# -------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    '''Streaming SHA-256 of a file's bytes.'''

    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def aggregate_fingerprint(files: Iterable[ArtifactFile]) -> str:
    '''Fingerprint of an artifact's member list (path, hash, and row count).'''

    payload = '\n'.join(f'{item.path}\t{item.sha256}\t{item.row_count}' for item in files)
    return hashlib.sha256(payload.encode()).hexdigest()

# -------------------------------------------------------------------------------------------------
# Writers
# -------------------------------------------------------------------------------------------------

def _table_with_contract_metadata(
    frame: pl.DataFrame,
    *,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> pa.Table:
    table = frame.to_arrow()
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            METADATA_CONTRACT: contract_version.encode(),
            METADATA_BUNDLE: bundle_id.encode(),
            METADATA_SCHEMA: schema_version.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)

def write_versioned_parquet(
    frame: pl.DataFrame,
    path: Path,
    *,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> ArtifactFile:
    '''Write one Parquet file stamped with contract metadata; ``path`` sits at the bundle root.'''

    path.parent.mkdir(parents=True, exist_ok=True)
    table = _table_with_contract_metadata(
        frame,
        contract_version=contract_version,
        bundle_id=bundle_id,
        schema_version=schema_version,
    )
    pq.write_table(table, path)
    return ArtifactFile(path=path.name, sha256=sha256_file(path), row_count=frame.height)

def write_versioned_dataset_batches(
    batches: Iterable[pl.DataFrame],
    root: Path,
    *,
    partition_column: str,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> Tuple[ArtifactFile, ...]:
    '''
    Write a Hive-partitioned dataset (``<root>/<column>=<value>/part-0.parquet``) from batches.

    Each partition value must occur in exactly one batch; the partition column is encoded in the
    directory name rather than stored in the files. Every member file is stamped with contract
    metadata. Member paths are relative to ``root.parent`` (the bundle root).
    '''

    root.mkdir(parents=True, exist_ok=False)
    for batch in batches:
        for (value,), part in batch.partition_by(
            partition_column, as_dict=True, maintain_order=True
        ).items():
            partition_dir = root / f'{partition_column}={value}'
            partition_dir.mkdir(exist_ok=False)
            table = _table_with_contract_metadata(
                part.drop(partition_column),
                contract_version=contract_version,
                bundle_id=bundle_id,
                schema_version=schema_version,
            )
            pq.write_table(table, partition_dir / 'part-0.parquet')
    members = []
    for path in sorted(root.glob('**/*.parquet')):
        members.append(
            ArtifactFile(
                path=path.relative_to(root.parent).as_posix(),
                sha256=sha256_file(path),
                row_count=pq.read_metadata(path).num_rows,
            )
        )
    return tuple(members)

def write_versioned_dataset(
    frame: pl.DataFrame,
    root: Path,
    *,
    partition_column: str,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> Tuple[ArtifactFile, ...]:
    '''Write one in-memory frame as a versioned, partitioned dataset.'''

    return write_versioned_dataset_batches(
        [frame],
        root,
        partition_column=partition_column,
        contract_version=contract_version,
        bundle_id=bundle_id,
        schema_version=schema_version,
    )
