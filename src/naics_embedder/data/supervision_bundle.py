'''
Canonical Stage-3 supervision facts.

Builds the fingerprinted codebook and the canonical pair-fact table from which every structural,
semantic, and exclusion artifact of a supervision bundle is derived. Structural facts are never
mutated by exclusion processing: exclusion provenance lives on its own directional columns.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import polars as pl

from naics_embedder.data.compute_distances import compute_structural_distances
from naics_embedder.data.compute_relations import compute_structural_relations
from naics_embedder.data.create_triplets import (
    CROSS_SECTOR_CAP_SEED,
    CROSS_SECTOR_NEGATIVE_CAP,
    iter_training_pair_batches,
)
from naics_embedder.supervision.artifacts import (
    sha256_file,
    write_versioned_dataset_batches,
    write_versioned_parquet,
)
from naics_embedder.supervision.schema import (
    CODEBOOK_SCHEMA_VERSION,
    CONTRACT_VERSION,
    DIFFICULTY_THRESHOLDS_SCHEMA_VERSION,
    DISTANCE_MATRIX_SCHEMA_VERSION,
    DISTANCES_SCHEMA_VERSION,
    PAIR_FACTS_SCHEMA_VERSION,
    RELATION_MATRIX_SCHEMA_VERSION,
    RELATIONS_SCHEMA_VERSION,
    TRAINING_PAIRS_SCHEMA_VERSION,
    ArtifactFile,
    ArtifactRecord,
    SupervisionManifest,
)
from naics_embedder.utils.config import DistancesConfig, SupervisionBuildConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Codebook
# -------------------------------------------------------------------------------------------------

def build_codebook(descriptions: pl.DataFrame) -> pl.DataFrame:
    '''
    Build the numeric codebook: contiguous ``code_id`` values tied to NAICS code strings.

    Args:
        descriptions: Descriptions frame with ``index`` and ``code`` columns.

    Returns:
        DataFrame with ``code_id`` (Int32) and ``code`` (Utf8) sorted by ``code_id``.

    Raises:
        ValueError: If IDs are not contiguous from zero or a code/ID appears twice.
    '''

    codebook = (
        descriptions.select(
            code_id=pl.col('index').cast(pl.Int32), code=pl.col('code').cast(pl.Utf8)
        ).unique().sort('code_id')
    )
    if codebook.get_column('code_id').n_unique() != codebook.height:
        raise ValueError('a description index maps to more than one NAICS code string')
    expected = list(range(codebook.height))
    if codebook.get_column('code_id').to_list() != expected:
        raise ValueError('description indices must be contiguous code IDs starting at zero')
    if codebook.get_column('code').n_unique() != codebook.height:
        raise ValueError('codebook contains duplicate NAICS code strings')
    return codebook

def codebook_fingerprint(codebook: pl.DataFrame) -> str:
    '''SHA-256 over the ordered ``code_id``/``code`` rows of a codebook.'''

    payload = '\n'.join(
        f'{row["code_id"]}\t{row["code"]}' for row in codebook.iter_rows(named=True)
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

# -------------------------------------------------------------------------------------------------
# Exclusion provenance
# -------------------------------------------------------------------------------------------------

def _directed_exclusions(descriptions: pl.DataFrame, codebook: pl.DataFrame) -> pl.DataFrame:
    '''Distinct (source excludes target) code-ID pairs whose codes both exist in the codebook.'''

    code_ids = codebook.rename({'code': 'source_code', 'code_id': 'source_code_id'})
    target_ids = codebook.rename({'code': 'target_code', 'code_id': 'target_code_id'})
    published = (
        descriptions.select(
            source_code=pl.col('code').cast(pl.Utf8),
            target_code=pl.col('excluded_codes'),
        ).explode('target_code').filter(pl.col('target_code').is_not_null())
    )
    directed = (
        published.join(code_ids, on='source_code', how='inner', validate='m:1').join(
            target_ids, on='target_code', how='inner', validate='m:1'
        ).select('source_code_id', 'target_code_id').unique()
    )
    dropped = published.height - published.join(
        target_ids, on='target_code', how='semi'
    ).height
    if dropped:
        logger.info(f'{dropped:,} published exclusion references name codes outside the codebook')
    return directed

def attach_exclusion_provenance(
    pair_facts: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> pl.DataFrame:
    '''
    Attach both directional exclusion flags and their symmetric OR to canonical pair facts.

    Raises:
        ValueError: If any published exclusion cannot be attached to exactly one pair fact.
    '''

    directed = _directed_exclusions(descriptions, codebook)
    forward = directed.rename(
        {'source_code_id': 'code_i_id', 'target_code_id': 'code_j_id'}
    ).with_columns(code_i_excludes_code_j=pl.lit(True))
    reverse = directed.rename(
        {'source_code_id': 'code_j_id', 'target_code_id': 'code_i_id'}
    ).with_columns(code_j_excludes_code_i=pl.lit(True))

    facts = (
        pair_facts.join(forward, on=['code_i_id', 'code_j_id'], how='left').join(
            reverse, on=['code_i_id', 'code_j_id'], how='left'
        ).with_columns(
            pl.col('code_i_excludes_code_j').fill_null(False),
            pl.col('code_j_excludes_code_i').fill_null(False),
        ).with_columns(
            is_explicit_exclusion=(
                pl.col('code_i_excludes_code_j') | pl.col('code_j_excludes_code_i')
            )
        ).sort('code_i_id', 'code_j_id')
    )

    attached = facts.select(
        pl.col('code_i_excludes_code_j').sum() + pl.col('code_j_excludes_code_i').sum()
    ).item()
    if attached != directed.height:
        raise ValueError(
            f'{directed.height - attached:,} of {directed.height:,} explicit exclusions could '
            'not be attached to a canonical pair fact (self-exclusion or missing pair)'
        )
    return facts

# -------------------------------------------------------------------------------------------------
# Pair facts
# -------------------------------------------------------------------------------------------------

def _validate_structural_pairs(structural: pl.DataFrame, codebook: pl.DataFrame) -> None:
    '''Fail closed on identity, orientation, uniqueness, coverage, or sentinel violations.'''

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

def build_pair_facts(
    distances: pl.DataFrame,
    relations: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> pl.DataFrame:
    '''
    Join structural distances and relations into canonical pair facts with exclusion provenance.

    Args:
        distances: Structural distances (``idx_i``, ``idx_j``, ``code_i``, ``code_j``,
            ``structural_distance``).
        relations: Structural relations keyed by the same ``idx_i``/``idx_j`` pairs.
        descriptions: Descriptions frame supplying ``code`` and ``excluded_codes``.
        codebook: Codebook from :func:`build_codebook`.

    Returns:
        One row per unordered pair of distinct codes in canonical orientation, with untouched
        structural columns plus ``code_i_excludes_code_j``, ``code_j_excludes_code_i``, and
        ``is_explicit_exclusion``.
    '''

    structural = distances.join(
        relations.select(
            'idx_i',
            'idx_j',
            'structural_relation_id',
            'structural_relation_name',
        ),
        on=['idx_i', 'idx_j'],
        how='inner',
        validate='1:1',
    ).select(
        code_i_id=pl.col('idx_i').cast(pl.Int32),
        code_j_id=pl.col('idx_j').cast(pl.Int32),
        code_i=pl.col('code_i').cast(pl.Utf8),
        code_j=pl.col('code_j').cast(pl.Utf8),
        structural_distance=pl.col('structural_distance').cast(pl.Float32),
        structural_relation_id=pl.col('structural_relation_id').cast(pl.Int16),
        structural_relation_name=pl.col('structural_relation_name').cast(pl.Utf8),
    )
    if structural.height != distances.height or structural.height != relations.height:
        raise ValueError('structural distance and relation frames describe different pairs')
    _validate_structural_pairs(structural, codebook)
    return attach_exclusion_provenance(structural, descriptions, codebook)

# -------------------------------------------------------------------------------------------------
# Matrices
# -------------------------------------------------------------------------------------------------

def _matrix(
    pair_facts: pl.DataFrame,
    codebook: pl.DataFrame,
    value_column: str,
    dtype: np.dtype,
) -> pl.DataFrame:
    size = codebook.height
    values = np.zeros((size, size), dtype=dtype)
    code_i_id = pair_facts.get_column('code_i_id').to_numpy()
    code_j_id = pair_facts.get_column('code_j_id').to_numpy()
    pair_values = pair_facts.get_column(value_column).to_numpy()
    values[code_i_id, code_j_id] = pair_values
    values[code_j_id, code_i_id] = pair_values
    columns = [
        f'idx_{code_id}-code_{code}' for code_id, code in codebook.select('code_id', 'code').rows()
    ]
    return pl.from_numpy(values, schema=columns, orient='row')

def distance_matrix_from_pair_facts(
    pair_facts: pl.DataFrame, codebook: pl.DataFrame
) -> pl.DataFrame:
    '''Symmetric structural distance matrix in codebook order (row r and column r = code r).'''

    return _matrix(pair_facts, codebook, 'structural_distance', np.float32)

def relation_matrix_from_pair_facts(
    pair_facts: pl.DataFrame, codebook: pl.DataFrame
) -> pl.DataFrame:
    '''Relation-ID lookup matrix mirroring the canonical relation ID in both directions.'''

    return _matrix(pair_facts, codebook, 'structural_relation_id', np.int16)

# -------------------------------------------------------------------------------------------------
# Bundle layout
# -------------------------------------------------------------------------------------------------

MANIFEST_FILENAME = 'manifest.json'
TRAINING_PAIRS_PARTITION_COLUMN = 'anchor'

ARTIFACT_FILENAMES = {
    'codebook': 'naics_codebook.parquet',
    'pair_facts': 'naics_pair_facts.parquet',
    'distances': 'naics_distances.parquet',
    'distance_matrix': 'naics_distance_matrix.parquet',
    'relations': 'naics_relations.parquet',
    'relation_matrix': 'naics_relation_matrix.parquet',
    'training_pairs': 'naics_training_pairs',
    'difficulty_thresholds': 'curriculum_difficulty_thresholds.json',
}

ARTIFACT_SCHEMA_VERSIONS = {
    'codebook': CODEBOOK_SCHEMA_VERSION,
    'pair_facts': PAIR_FACTS_SCHEMA_VERSION,
    'distances': DISTANCES_SCHEMA_VERSION,
    'distance_matrix': DISTANCE_MATRIX_SCHEMA_VERSION,
    'relations': RELATIONS_SCHEMA_VERSION,
    'relation_matrix': RELATION_MATRIX_SCHEMA_VERSION,
    'training_pairs': TRAINING_PAIRS_SCHEMA_VERSION,
    'difficulty_thresholds': DIFFICULTY_THRESHOLDS_SCHEMA_VERSION,
}

PAIR_FACT_SCHEMA = {
    'code_i_id': pl.Int32,
    'code_j_id': pl.Int32,
    'code_i': pl.Utf8,
    'code_j': pl.Utf8,
    'structural_distance': pl.Float32,
    'structural_relation_id': pl.Int16,
    'structural_relation_name': pl.Utf8,
    'code_i_excludes_code_j': pl.Boolean,
    'code_j_excludes_code_i': pl.Boolean,
    'is_explicit_exclusion': pl.Boolean,
}

# -------------------------------------------------------------------------------------------------
# Fingerprints and provenance
# -------------------------------------------------------------------------------------------------

def descriptions_frame_fingerprint(descriptions: pl.DataFrame) -> str:
    '''SHA-256 over the ordered rows of an in-memory descriptions frame.'''

    digest = hashlib.sha256()
    for row in descriptions.sort('index').iter_rows(named=True):
        digest.update(json.dumps(row, sort_keys=True, default=str).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def exclusion_input_fingerprint(descriptions: pl.DataFrame) -> str:
    '''SHA-256 over every published (code, excluded code) reference, sorted.'''

    published = (
        descriptions.select(
            source=pl.col('code').cast(pl.Utf8), target=pl.col('excluded_codes')
        ).explode('target').filter(pl.col('target').is_not_null()).unique().sort(
            'source', 'target'
        )
    )
    payload = '\n'.join(f'{source}\t{target}' for source, target in published.rows())
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _generator_revision() -> str:
    '''Git revision of the generating checkout, or the installed package version.'''

    repository = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ['git', 'describe', '--always', '--dirty', '--abbrev=40'],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()
        if revision:
            return revision
    except (OSError, subprocess.CalledProcessError):
        pass
    return f'naics-embedder=={importlib.metadata.version("naics-embedder")}'

# -------------------------------------------------------------------------------------------------
# Bundle validation
# -------------------------------------------------------------------------------------------------

def _normalize_pair_facts(pair_facts: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(PAIR_FACT_SCHEMA) - set(pair_facts.columns))
    if missing:
        raise ValueError(f'pair facts lack required columns: {missing}')
    return pair_facts.select(
        [pl.col(name).cast(dtype) for name, dtype in PAIR_FACT_SCHEMA.items()]
    ).sort('code_i_id', 'code_j_id')

def validate_pair_facts(
    pair_facts: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> Dict[str, bool]:
    '''
    Validate canonical pair facts against the codebook and the published exclusions.

    Returns:
        Validation results keyed by check name (all ``True``).

    Raises:
        ValueError: On the first failed check, naming the violated invariant.
    '''

    structural_columns = [
        'code_i_id',
        'code_j_id',
        'code_i',
        'code_j',
        'structural_distance',
        'structural_relation_id',
        'structural_relation_name',
    ]
    _validate_structural_pairs(pair_facts.select(structural_columns), codebook)

    derived = pl.col('code_i_excludes_code_j') | pl.col('code_j_excludes_code_i')
    if pair_facts.filter(pl.col('is_explicit_exclusion').ne(derived)).height:
        raise ValueError(
            'pair facts exclusion derivation is inconsistent: is_explicit_exclusion must equal '
            'code_i_excludes_code_j OR code_j_excludes_code_i'
        )

    published = attach_exclusion_provenance(
        pair_facts.select(structural_columns), descriptions, codebook
    )
    flags = ['code_i_excludes_code_j', 'code_j_excludes_code_i', 'is_explicit_exclusion']
    if not published.select(flags).equals(pair_facts.select(flags)):
        raise ValueError('pair facts exclusion flags disagree with the published exclusions')

    return {
        'codebook_contiguous_unique': True,
        'codebook_identity': True,
        'pair_keys_unique': True,
        'canonical_orientation': True,
        'pair_coverage': True,
        'nonzero_structural_distance': True,
        'no_structural_sentinel': True,
        'exclusion_derivation': True,
        'exclusions_match_descriptions': True,
    }

def _validate_matrices(
    pair_facts: pl.DataFrame,
    codebook: pl.DataFrame,
    distance_matrix: pl.DataFrame,
    relation_matrix: pl.DataFrame,
) -> None:
    expected_columns = [
        f'idx_{code_id}-code_{code}' for code_id, code in codebook.select('code_id', 'code').rows()
    ]
    code_i_id = pair_facts.get_column('code_i_id').to_numpy()
    code_j_id = pair_facts.get_column('code_j_id').to_numpy()
    for name, matrix, column in (
        ('distance_matrix', distance_matrix, 'structural_distance'),
        ('relation_matrix', relation_matrix, 'structural_relation_id'),
    ):
        if matrix.columns != expected_columns:
            raise ValueError(f'{name} columns do not follow the codebook order')
        values = matrix.to_numpy()
        expected = pair_facts.get_column(column).to_numpy()
        if not (
            np.array_equal(values[code_i_id, code_j_id], expected)
            and np.array_equal(values[code_j_id, code_i_id], expected)
            and not np.diagonal(values).any()
        ):
            raise ValueError(f'{name} does not reconcile with the long-form pair facts')

def _validate_training_identity(batch: pl.DataFrame, pair_keys: pl.DataFrame) -> None:
    '''Every anchor/positive and anchor/negative identity must join to a canonical pair fact.'''

    for other in ('positive_code_id', 'negative_code_id'):
        keys = batch.select(
            low=pl.min_horizontal('anchor_code_id', other).cast(pl.Int32),
            high=pl.max_horizontal('anchor_code_id', other).cast(pl.Int32),
        )
        if keys.join(pair_keys, on=['low', 'high'], how='anti').height:
            raise ValueError(f'training pair {other} identities do not join to pair facts')

# -------------------------------------------------------------------------------------------------
# Compatibility long-form artifacts
# -------------------------------------------------------------------------------------------------

def _compat_distances(pair_facts: pl.DataFrame) -> pl.DataFrame:
    return pair_facts.select(
        idx_i=pl.col('code_i_id'),
        idx_j=pl.col('code_j_id'),
        code_i=pl.col('code_i'),
        code_j=pl.col('code_j'),
        distance=pl.col('structural_distance'),
        code_i_excludes_code_j=pl.col('code_i_excludes_code_j'),
        code_j_excludes_code_i=pl.col('code_j_excludes_code_i'),
        is_explicit_exclusion=pl.col('is_explicit_exclusion'),
    )

def _compat_relations(pair_facts: pl.DataFrame) -> pl.DataFrame:
    return pair_facts.select(
        idx_i=pl.col('code_i_id'),
        idx_j=pl.col('code_j_id'),
        code_i=pl.col('code_i'),
        code_j=pl.col('code_j'),
        relation_id=pl.col('structural_relation_id'),
        relation=pl.col('structural_relation_name'),
        code_i_excludes_code_j=pl.col('code_i_excludes_code_j'),
        code_j_excludes_code_i=pl.col('code_j_excludes_code_i'),
        is_explicit_exclusion=pl.col('is_explicit_exclusion'),
    )

# -------------------------------------------------------------------------------------------------
# Bundle generation
# -------------------------------------------------------------------------------------------------

def _structural_relation_ids(pair_facts: pl.DataFrame) -> Dict[str, int]:
    mapping = pair_facts.select('structural_relation_name', 'structural_relation_id').unique()
    if mapping.get_column('structural_relation_name').is_duplicated().any():
        raise ValueError('a structural relation name maps to more than one relation ID')
    return {
        name: int(relation_id)
        for name, relation_id in mapping.sort('structural_relation_id').rows()
    }

def _checked_training_batches(
    pair_facts: pl.DataFrame,
    counts: Dict[str, int],
    *,
    cross_sector_cap: int,
    cap_seed: int,
) -> Iterator[pl.DataFrame]:
    pair_keys = pair_facts.select(
        low=pl.min_horizontal('code_i_id', 'code_j_id'),
        high=pl.max_horizontal('code_i_id', 'code_j_id'),
    )
    for batch in iter_training_pair_batches(
        pair_facts, cross_sector_cap=cross_sector_cap, cap_seed=cap_seed
    ):
        _validate_training_identity(batch, pair_keys)
        counts['rows'] += batch.height
        counts['exclusions'] += int(batch.get_column('negative_is_explicit_exclusion').sum())
        yield batch.with_columns(pl.col('anchor_code_id').alias(TRAINING_PAIRS_PARTITION_COLUMN))

def _record(
    logical_name: str,
    files: tuple[ArtifactFile, ...],
    *,
    exclusion_count: int = 0,
) -> ArtifactRecord:
    return ArtifactRecord(
        path=ARTIFACT_FILENAMES[logical_name],
        schema_version=ARTIFACT_SCHEMA_VERSIONS[logical_name],
        row_count=sum(member.row_count for member in files),
        exclusion_count=exclusion_count,
        files=files,
    )

def _write_bundle_artifacts(
    staging: Path,
    *,
    bundle_id: str,
    codebook: pl.DataFrame,
    pair_facts: pl.DataFrame,
    distance_matrix: pl.DataFrame,
    relation_matrix: pl.DataFrame,
    cross_sector_cap: int,
    cap_seed: int,
) -> Dict[str, ArtifactRecord]:
    exclusions = int(pair_facts.get_column('is_explicit_exclusion').sum())

    def parquet(logical_name: str, frame: pl.DataFrame) -> tuple[ArtifactFile, ...]:
        return (
            write_versioned_parquet(
                frame,
                staging / ARTIFACT_FILENAMES[logical_name],
                contract_version=CONTRACT_VERSION,
                bundle_id=bundle_id,
                schema_version=ARTIFACT_SCHEMA_VERSIONS[logical_name],
            ),
        )

    records = {
        'codebook': _record('codebook', parquet('codebook', codebook)),
        'pair_facts': _record(
            'pair_facts', parquet('pair_facts', pair_facts), exclusion_count=exclusions
        ),
        'distances': _record(
            'distances',
            parquet('distances', _compat_distances(pair_facts)),
            exclusion_count=exclusions,
        ),
        'distance_matrix': _record('distance_matrix', parquet('distance_matrix', distance_matrix)),
        'relations': _record(
            'relations',
            parquet('relations', _compat_relations(pair_facts)),
            exclusion_count=exclusions,
        ),
        'relation_matrix': _record('relation_matrix', parquet('relation_matrix', relation_matrix)),
    }

    counts = {'rows': 0, 'exclusions': 0}
    training_files = write_versioned_dataset_batches(
        _checked_training_batches(
            pair_facts, counts, cross_sector_cap=cross_sector_cap, cap_seed=cap_seed
        ),
        staging / ARTIFACT_FILENAMES['training_pairs'],
        partition_column=TRAINING_PAIRS_PARTITION_COLUMN,
        contract_version=CONTRACT_VERSION,
        bundle_id=bundle_id,
        schema_version=ARTIFACT_SCHEMA_VERSIONS['training_pairs'],
    )
    records['training_pairs'] = _record(
        'training_pairs', training_files, exclusion_count=counts['exclusions']
    )
    if records['training_pairs'].row_count != counts['rows']:
        raise ValueError('training_pairs member row counts disagree with generated rows')

    # Imported here: the graph curriculum pulls in torch and graph-model dependencies.
    from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
        compute_difficulty_thresholds,
    )

    thresholds_path = staging / ARTIFACT_FILENAMES['difficulty_thresholds']
    compute_difficulty_thresholds(
        str(staging / ARTIFACT_FILENAMES['distances']),
        str(staging / ARTIFACT_FILENAMES['training_pairs']),
        output_path=str(thresholds_path),
    )
    records['difficulty_thresholds'] = _record(
        'difficulty_thresholds',
        (
            ArtifactFile(
                path=thresholds_path.name,
                sha256=sha256_file(thresholds_path),
                row_count=1,
            ),
        ),
    )
    return records

def generate_supervision_bundle_from_frames(
    *,
    output_root: Path,
    bundle_id: str,
    generator_revision: str,
    naics_vintage: int,
    descriptions: pl.DataFrame,
    pair_facts: pl.DataFrame,
    description_fingerprint: Optional[str] = None,
    structural_relation_ids: Optional[Mapping[str, int]] = None,
    generation_parameters: Optional[Mapping[str, Any]] = None,
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
) -> Path:
    '''
    Validate canonical frames and publish them as one immutable supervision bundle.

    Artifacts are written to ``<output_root>/.<bundle_id>.staging``; the manifest is written only
    after every artifact validates, and the staging directory is then atomically renamed to
    ``<output_root>/<bundle_id>``. An existing bundle is never overwritten. On failure, only the
    staging directory created by this call is removed.

    Returns:
        Path to the published ``manifest.json``.
    '''

    output_root = Path(output_root)
    final = output_root / bundle_id
    staging = output_root / f'.{bundle_id}.staging'
    if final.exists():
        raise FileExistsError(f'supervision bundle {bundle_id} already exists at {final}')

    codebook = build_codebook(descriptions)
    pair_facts = _normalize_pair_facts(pair_facts)
    validation_results = validate_pair_facts(pair_facts, descriptions, codebook)
    distance_matrix = distance_matrix_from_pair_facts(pair_facts, codebook)
    relation_matrix = relation_matrix_from_pair_facts(pair_facts, codebook)
    _validate_matrices(pair_facts, codebook, distance_matrix, relation_matrix)
    validation_results['matrix_reconciliation'] = True

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f'staging directory for supervision bundle {bundle_id} already exists: {staging}'
        ) from exc

    try:
        artifacts = _write_bundle_artifacts(
            staging,
            bundle_id=bundle_id,
            codebook=codebook,
            pair_facts=pair_facts,
            distance_matrix=distance_matrix,
            relation_matrix=relation_matrix,
            cross_sector_cap=cross_sector_cap,
            cap_seed=cap_seed,
        )
        validation_results.update(
            {
                'direct_positive_safety': True,
                'training_exclusion_derivation': True,
                'training_identity_joins': True,
                'artifact_hashes_recorded': True,
            }
        )
        parameters = {
            'canonical_orientation': 'shallower code first; numeric code order on ties',
            'reversed_anchor_rows': 'same-level cross-prefix pairs seed negatives only',
            'cross_sector_negative_cap': cross_sector_cap,
            'cross_sector_cap_seed': cap_seed,
        } | dict(generation_parameters or {})
        manifest = SupervisionManifest(
            contract_version=CONTRACT_VERSION,
            bundle_id=bundle_id,
            generated_at=datetime.now(timezone.utc),
            generator_revision=generator_revision,
            naics_vintage=naics_vintage,
            codebook_order=tuple(codebook.get_column('code').to_list()),
            codebook_fingerprint=codebook_fingerprint(codebook),
            description_fingerprint=(
                description_fingerprint or descriptions_frame_fingerprint(descriptions)
            ),
            exclusion_fingerprint=exclusion_input_fingerprint(descriptions),
            generation_parameters=parameters,
            structural_relation_ids=dict(
                structural_relation_ids or _structural_relation_ids(pair_facts)
            ),
            artifacts=artifacts,
            validation_results=validation_results,
        )
        (staging / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
        if final.exists():
            raise FileExistsError(f'supervision bundle {bundle_id} already exists at {final}')
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(
        f'Published supervision bundle {bundle_id}: {artifacts["pair_facts"].row_count:,} pair '
        f'facts ({artifacts["pair_facts"].exclusion_count:,} explicit exclusions), '
        f'{artifacts["training_pairs"].row_count:,} training pairs'
    )
    return final / MANIFEST_FILENAME

def generate_supervision_bundle(cfg: SupervisionBuildConfig) -> Path:
    '''
    Build and publish a new supervision bundle from the configured descriptions.

    The bundle ID is a fresh UUID4, and the description fingerprint is the SHA-256 of the exact
    descriptions file, so training can later verify it runs against the same input.

    Returns:
        Path to the published ``manifest.json``.
    '''

    descriptions_path = Path(cfg.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path)
    codebook = build_codebook(descriptions)
    distances = compute_structural_distances(
        str(descriptions_path), DistancesConfig(input_parquet=str(descriptions_path))
    )
    relations = compute_structural_relations(str(descriptions_path), cfg.relation_id)
    pair_facts = build_pair_facts(distances, relations, descriptions, codebook)
    return generate_supervision_bundle_from_frames(
        output_root=Path(cfg.output_root),
        bundle_id=str(uuid.uuid4()),
        generator_revision=_generator_revision(),
        naics_vintage=cfg.naics_vintage,
        descriptions=descriptions,
        pair_facts=pair_facts,
        description_fingerprint=sha256_file(descriptions_path),
        structural_relation_ids=cfg.relation_id,
        generation_parameters={
            'descriptions_parquet': str(descriptions_path.resolve()),
            'output_root': str(Path(cfg.output_root).resolve()),
        },
    )
