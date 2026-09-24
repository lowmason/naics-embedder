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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    IndexRole,
    SemanticTarget,
    SupervisionManifest,
)

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
        for (value, ), part in batch.partition_by(
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

# -------------------------------------------------------------------------------------------------
# Shared relational validators (used at generation and at load time)
# -------------------------------------------------------------------------------------------------

REQUIRED_ARTIFACTS = (
    'codebook',
    'pair_facts',
    'distances',
    'distance_matrix',
    'relations',
    'relation_matrix',
    'training_pairs',
    'difficulty_thresholds',
)

STRUCTURAL_PAIR_COLUMNS = (
    'code_i_id',
    'code_j_id',
    'code_i',
    'code_j',
    'structural_distance',
    'structural_relation_id',
    'structural_relation_name',
)

def codebook_fingerprint(codebook: pl.DataFrame) -> str:
    '''SHA-256 over the ordered ``code_id``/``code`` rows of a codebook.'''

    payload = '\n'.join(
        f'{row["code_id"]}\t{row["code"]}' for row in codebook.iter_rows(named=True)
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def validate_structural_pairs(structural: pl.DataFrame, codebook: pl.DataFrame) -> None:
    '''Fail closed on identity, uniqueness, orientation, coverage, or sentinel violations.'''

    identities = codebook.select(code_id=pl.col('code_id'), expected=pl.col('code'))
    for side in ('i', 'j'):
        checked = structural.select(f'code_{side}_id', f'code_{side}').join(
            identities.rename({'code_id': f'code_{side}_id'}),
            on=f'code_{side}_id',
            how='left',
        )
        if checked.filter(pl.col('expected').ne_missing(pl.col(f'code_{side}'))).height:
            raise ValueError(f'pair facts code_{side} IDs disagree with the codebook')

    if structural.filter(pl.col('code_i_id').eq(pl.col('code_j_id'))).height:
        raise ValueError('pair facts must describe distinct codes')

    # Uniqueness first: a pair emitted in both orientations is diagnosed as a duplicate rather
    # than as a single non-canonical row.
    unordered = structural.select(
        low=pl.min_horizontal('code_i_id', 'code_j_id'),
        high=pl.max_horizontal('code_i_id', 'code_j_id'),
    )
    if unordered.is_duplicated().any():
        raise ValueError('pair facts contain duplicate unordered code pairs')

    level_i = pl.col('code_i').str.len_chars()
    level_j = pl.col('code_j').str.len_chars()
    non_canonical = structural.filter(
        level_i.gt(level_j) | (level_i.eq(level_j) & pl.col('code_i').ge(pl.col('code_j')))
    )
    if non_canonical.height:
        example = non_canonical.row(0, named=True)
        raise ValueError(
            'pair facts violate canonical orientation (shallower code first, code order on '
            f'ties): {non_canonical.height:,} rows, e.g. {example["code_i"]}/{example["code_j"]}'
        )

    expected_pairs = codebook.height * (codebook.height - 1) // 2
    if structural.height != expected_pairs:
        raise ValueError(
            f'pair facts cover {structural.height:,} pairs but the codebook requires '
            f'{expected_pairs:,} unordered pairs'
        )

    if structural.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError('pair facts contain null structural values')
    if structural.filter(pl.col('structural_distance').eq(0.0)).height:
        raise ValueError('distinct-code pair facts cannot contain structural distance zero')
    if structural.filter(
        pl.col('structural_relation_id').eq(0)
        | pl.col('structural_relation_name').eq('excluded')
    ).height:
        raise ValueError('structural relation fields contain an exclusion sentinel')

def validate_exclusion_derivation(pair_facts: pl.DataFrame) -> None:
    '''The symmetric exclusion flag must equal the OR of both directional flags.'''

    derived = pl.col('code_i_excludes_code_j') | pl.col('code_j_excludes_code_i')
    if pair_facts.filter(pl.col('is_explicit_exclusion').ne(derived)).height:
        raise ValueError(
            'pair facts exclusion derivation is inconsistent: is_explicit_exclusion must equal '
            'code_i_excludes_code_j OR code_j_excludes_code_i'
        )

INDEX_ROLES_ARTIFACT = 'index_roles'
INDEX_ROLE_COLUMNS = ('entry_id', 'code', 'text', 'role')

def validate_index_role_table(
    roles: pl.DataFrame,
    six_digit_codes: Collection[str],
    *,
    min_examples_per_code: int = 1,
) -> None:
    '''
    Fail closed unless every index entry holds exactly one known role for a six-digit code.

    Also requires non-empty entry text, and the examples-channel floor: every code keeps at least
    ``min_examples_per_code`` examples-role entries, or all its entries if it has fewer.
    '''

    missing = [name for name in INDEX_ROLE_COLUMNS if name not in roles.columns]
    if missing:
        raise ValueError(f'index roles lack required columns: {missing}')
    if roles.select(pl.any_horizontal(pl.col(list(INDEX_ROLE_COLUMNS)).is_null()).any()).item():
        raise ValueError('index roles contain null values')
    if roles.get_column('entry_id').is_duplicated().any():
        raise ValueError('an index entry holds more than one role')
    unknown = sorted(
        set(roles.get_column('role').unique().to_list()) - {role.value
                                                            for role in IndexRole}
    )
    if unknown:
        raise ValueError(f'index roles contain unknown roles: {unknown}')
    outside = sorted(set(roles.get_column('code').unique().to_list()) - set(six_digit_codes))
    if outside:
        raise ValueError(f'index roles name codes outside the six-digit codebook: {outside[:5]}')
    if roles.filter(pl.col('text').str.strip_chars().eq('')).height:
        raise ValueError('index roles contain an empty entry text')
    floor = pl.min_horizontal(pl.col('entries'), pl.lit(min_examples_per_code))
    short = roles.group_by('code').agg(
        examples=pl.col('role').eq(IndexRole.EXAMPLES.value).sum(), entries=pl.len()
    ).filter(pl.col('examples') < floor)
    if short.height:
        raise ValueError(
            f'{short.height:,} codes have fewer than {min_examples_per_code} examples-role entries'
        )

def validate_matrix(
    matrix: pl.DataFrame,
    pair_facts: pl.DataFrame,
    codebook: pl.DataFrame,
    value_column: str,
) -> None:
    '''A lookup matrix must follow codebook order and mirror every long-form pair value.'''

    expected_columns = [
        f'idx_{code_id}-code_{code}' for code_id, code in codebook.select('code_id', 'code').rows()
    ]
    if matrix.columns != expected_columns:
        raise ValueError('matrix columns do not follow the codebook order')
    values = matrix.to_numpy()
    code_i_id = pair_facts.get_column('code_i_id').to_numpy()
    code_j_id = pair_facts.get_column('code_j_id').to_numpy()
    expected = pair_facts.get_column(value_column).to_numpy()
    if not (
        values.shape == (codebook.height, codebook.height) and np.array_equal(
            values[code_i_id, code_j_id], expected
        ) and np.array_equal(values[code_j_id, code_i_id], expected)
        and not np.diagonal(values).any()
    ):
        raise ValueError('matrix does not reconcile with the long-form pair facts')

def _directed_pair_facts(pair_facts: pl.DataFrame) -> pl.DataFrame:
    '''Both orientations of every pair fact, with exclusion directions in the anchor's view.'''

    return pl.concat(
        [
            pair_facts.select(
                anchor=pl.col('code_i_id').cast(pl.Int32),
                other=pl.col('code_j_id').cast(pl.Int32),
                fact_anchor_excludes=pl.col('code_i_excludes_code_j'),
                fact_other_excludes=pl.col('code_j_excludes_code_i'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
            ),
            pair_facts.select(
                anchor=pl.col('code_j_id').cast(pl.Int32),
                other=pl.col('code_i_id').cast(pl.Int32),
                fact_anchor_excludes=pl.col('code_j_excludes_code_i'),
                fact_other_excludes=pl.col('code_i_excludes_code_j'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
            ),
        ]
    )

TRAINING_VALIDATION_CHUNK_FILES = 128

def validate_training_pairs_members(
    paths: List[Path],
    pair_facts: pl.DataFrame,
    n_codes: int,
) -> None:
    '''
    Validate training-pair member files against the codebook and canonical pair facts.

    Every identity must be a known code ID, no direct positive may be an explicit exclusion,
    exclusion and semantic columns must be internally consistent, and every anchor/positive and
    anchor/negative view must match the pair facts (structure and both exclusion directions).
    Members are checked in bounded chunks of files, which is exact because every row check is
    row-local and every uniqueness check is a join against the pair facts.
    '''

    if not paths:
        return
    directed = _directed_pair_facts(pair_facts)
    for start in range(0, len(paths), TRAINING_VALIDATION_CHUNK_FILES):
        _validate_training_chunk(
            paths[start:start + TRAINING_VALIDATION_CHUNK_FILES], directed, n_codes
        )

def _validate_training_chunk(paths: List[Path], directed: pl.DataFrame, n_codes: int) -> None:
    scan = pl.scan_parquet([str(path) for path in paths])
    ids = ('anchor_code_id', 'positive_code_id', 'negative_code_id')
    expected_target = pl.when(pl.col('negative_is_explicit_exclusion')).then(
        pl.lit(SemanticTarget.UNRELATED.value)
    ).otherwise(pl.lit(SemanticTarget.UNKNOWN.value))
    summary = scan.select(
        unmapped=pl.any_horizontal(
            *[
                pl.col(name).is_null() | pl.col(name).lt(0) | pl.col(name).ge(n_codes)
                for name in ids
            ]
        ).sum(),
        excluded_positives=pl.col('positive_is_explicit_exclusion').sum(),
        derivation=pl.col('negative_is_explicit_exclusion').ne(
            pl.col('anchor_excludes_negative') | pl.col('negative_excludes_anchor')
        ).sum(),
        semantic=pl.col('negative_semantic_target').ne(expected_target).sum(),
        repeats=(
            pl.col('negative_code_id').eq(pl.col('anchor_code_id'))
            | pl.col('negative_code_id').eq(pl.col('positive_code_id'))
        ).sum(),
    ).collect().row(0, named=True)
    if summary['unmapped']:
        raise ValueError(f'{summary["unmapped"]:,} rows contain an unmapped code identity')
    if summary['excluded_positives']:
        raise ValueError('a direct positive is an explicit exclusion')
    if summary['derivation']:
        raise ValueError('negative exclusion derivation is inconsistent')
    if summary['semantic']:
        raise ValueError('negative semantic target disagrees with exclusion provenance')
    if summary['repeats']:
        raise ValueError('a training negative repeats its anchor or positive code')

    positives = scan.select(
        anchor=pl.col('anchor_code_id').cast(pl.Int32),
        other=pl.col('positive_code_id').cast(pl.Int32),
        distance=pl.col('positive_structural_distance').cast(pl.Float32),
        relation=pl.col('positive_structural_relation_id').cast(pl.Int16),
    ).unique().collect().join(directed, on=['anchor', 'other'], how='left')
    if positives.filter(pl.col('fact_distance').is_null()).height:
        raise ValueError('positive identities do not join to pair facts')
    if positives.filter(
        pl.col('distance').ne(pl.col('fact_distance'))
        | pl.col('relation').ne(pl.col('fact_relation'))
    ).height:
        raise ValueError('positive structural values disagree with pair facts')

    negatives = scan.select(
        anchor=pl.col('anchor_code_id').cast(pl.Int32),
        other=pl.col('negative_code_id').cast(pl.Int32),
        anchor_excludes=pl.col('anchor_excludes_negative'),
        other_excludes=pl.col('negative_excludes_anchor'),
        distance=pl.col('negative_structural_distance').cast(pl.Float32),
        relation=pl.col('negative_structural_relation_id').cast(pl.Int16),
    ).unique().collect().join(directed, on=['anchor', 'other'], how='left')
    if negatives.filter(pl.col('fact_distance').is_null()).height:
        raise ValueError('negative identities do not join to pair facts')
    if negatives.filter(
        pl.col('anchor_excludes').ne(pl.col('fact_anchor_excludes'))
        | pl.col('other_excludes').ne(pl.col('fact_other_excludes'))
    ).height:
        raise ValueError('negative exclusion flags disagree with pair facts')
    if negatives.filter(
        pl.col('distance').ne(pl.col('fact_distance'))
        | pl.col('relation').ne(pl.col('fact_relation'))
    ).height:
        raise ValueError('negative structural values disagree with pair facts')

# -------------------------------------------------------------------------------------------------
# Validated bundle loading
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedSupervisionBundle:
    '''A supervision bundle whose manifest, members, and relations have all been verified.'''

    root: Path
    manifest_path: Path
    manifest: SupervisionManifest

    def artifact_path(self, logical_name: str) -> Path:
        try:
            record = self.manifest.artifacts[logical_name]
        except KeyError as exc:
            raise ValueError(f'bundle has no {logical_name!r} artifact') from exc
        return self.root / record.path

    def member_paths(self, logical_name: str) -> Tuple[Path, ...]:
        '''Every member file of a logical artifact, as recorded in the manifest.'''

        if logical_name not in self.manifest.artifacts:
            raise ValueError(f'bundle has no {logical_name!r} artifact')
        return tuple(
            self.root / member.path for member in self.manifest.artifacts[logical_name].files
        )

def _parquet_contract(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    metadata = pq.read_metadata(path).metadata or {}

    def decode(key: bytes) -> Optional[str]:
        value = metadata.get(key)
        return value.decode() if value is not None else None

    return decode(METADATA_CONTRACT), decode(METADATA_BUNDLE), decode(METADATA_SCHEMA)

def _in_context(logical_name: str, bundle_id: str, check: Callable[[], None]) -> None:
    try:
        check()
    except ValueError as exc:
        raise ValueError(f'{logical_name} ({bundle_id}): {exc}') from exc

def _validate_members(root: Path, manifest: SupervisionManifest) -> None:
    missing = sorted(set(REQUIRED_ARTIFACTS) - set(manifest.artifacts))
    if missing:
        raise ValueError(f'bundle {manifest.bundle_id} lacks required artifacts: {missing}')
    for logical_name, artifact in manifest.artifacts.items():
        member_rows = 0
        for member in artifact.files:
            member_path = root / member.path
            if not member_path.is_file():
                raise ValueError(f'{logical_name} artifact missing: {member_path}')
            actual_hash = sha256_file(member_path)
            if actual_hash != member.sha256:
                raise ValueError(
                    f'{logical_name} hash mismatch at {member_path}: '
                    f'expected {member.sha256}, found {actual_hash}'
                )
            member_rows += member.row_count
            if member_path.suffix == '.parquet':
                actual = _parquet_contract(member_path)
                expected = (manifest.contract_version, manifest.bundle_id, artifact.schema_version)
                if actual != expected:
                    raise ValueError(
                        f'{logical_name} metadata mismatch: expected {expected}, found {actual}'
                    )
                if pq.read_metadata(member_path).num_rows != member.row_count:
                    raise ValueError(f'{logical_name} member row count mismatch at {member_path}')
        if member_rows != artifact.row_count:
            raise ValueError(
                f'{logical_name} row count mismatch: '
                f'manifest={artifact.row_count}, members={member_rows}'
            )

def _validate_relations(root: Path, manifest: SupervisionManifest) -> None:
    bundle_id = manifest.bundle_id

    def read(logical_name: str) -> pl.DataFrame:
        return pl.read_parquet(root / manifest.artifacts[logical_name].path)

    codebook = read('codebook').sort('code_id')

    def check_codebook() -> None:
        if codebook.get_column('code_id').to_list() != list(range(codebook.height)):
            raise ValueError('code IDs are not contiguous from zero')
        if tuple(codebook.get_column('code').to_list()) != manifest.codebook_order:
            raise ValueError('codebook order differs from the manifest codebook_order')
        if codebook_fingerprint(codebook) != manifest.codebook_fingerprint:
            raise ValueError('codebook fingerprint differs from the manifest')

    _in_context('codebook', bundle_id, check_codebook)

    pair_facts = read('pair_facts')

    def check_pair_facts() -> None:
        validate_structural_pairs(pair_facts.select(STRUCTURAL_PAIR_COLUMNS), codebook)
        validate_exclusion_derivation(pair_facts)

    _in_context('pair_facts', bundle_id, check_pair_facts)

    # Positional expressions only: polars places positional expressions before named ones.
    identity = [
        pl.col('code_i_id').alias('idx_i'),
        pl.col('code_j_id').alias('idx_j'),
        pl.col('code_i'),
        pl.col('code_j'),
    ]
    flags = [
        pl.col('code_i_excludes_code_j'),
        pl.col('code_j_excludes_code_i'),
        pl.col('is_explicit_exclusion'),
    ]
    long_forms = {
        'distances': pair_facts.select(
            *identity,
            pl.col('structural_distance').alias('distance'),
            *flags,
        ),
        'relations': pair_facts.select(
            *identity,
            pl.col('structural_relation_id').alias('relation_id'),
            pl.col('structural_relation_name').alias('relation'),
            *flags,
        ),
    }
    for logical_name, expected in long_forms.items():
        frame = read(logical_name)

        def check_long_form(frame: pl.DataFrame = frame, expected: pl.DataFrame = expected) -> None:
            if not frame.equals(expected):
                raise ValueError('long-form artifact does not reconcile with the pair facts')

        _in_context(logical_name, bundle_id, check_long_form)

    for logical_name, value_column in (
        ('distance_matrix', 'structural_distance'),
        ('relation_matrix', 'structural_relation_id'),
    ):
        matrix = read(logical_name)
        _in_context(
            logical_name,
            bundle_id,
            lambda matrix=matrix, value_column=value_column: validate_matrix(
                matrix, pair_facts, codebook, value_column
            ),
        )

    training_paths = [root / member.path for member in manifest.artifacts['training_pairs'].files]
    _in_context(
        'training_pairs',
        bundle_id,
        lambda: validate_training_pairs_members(training_paths, pair_facts, codebook.height),
    )

def load_validated_bundle(
    manifest_path: str | Path,
    expected_contract: str = CONTRACT_VERSION,
) -> ValidatedSupervisionBundle:
    '''
    Load a supervision bundle, failing closed on any contract, integrity, or relational violation.

    Checks the contract version, every member's existence, hash, row count, and Parquet contract
    metadata, the recorded validation results, and then re-runs the relational checks: codebook
    order and fingerprint, pair-fact identity/orientation/coverage/sentinels/exclusion derivation,
    long-form and matrix reconciliation, and training-pair identity, exclusion, and structure.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: On any violation; the message names the artifact and bundle ID.
    '''

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'supervision manifest not found: {path}')
    manifest = SupervisionManifest.model_validate_json(path.read_text())
    if manifest.contract_version != expected_contract:
        raise ValueError(
            f'expected supervision contract {expected_contract}, '
            f'found {manifest.contract_version} in {path}'
        )

    root = path.parent
    _validate_members(root, manifest)
    if not all(manifest.validation_results.values()):
        failed = sorted(k for k, passed in manifest.validation_results.items() if not passed)
        raise ValueError(f'bundle manifest records failed validations: {failed}')
    _validate_relations(root, manifest)
    return ValidatedSupervisionBundle(root=root, manifest_path=path, manifest=manifest)
