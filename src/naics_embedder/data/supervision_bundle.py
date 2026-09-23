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
import logging

import numpy as np
import polars as pl

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
