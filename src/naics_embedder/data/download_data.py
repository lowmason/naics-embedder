# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

import polars as pl
import yaml

from naics_embedder.panels.index_roles import (
    attach_role_text,
    read_role_table,
    verify_examples_channel,
    verify_role_leakage,
)
from naics_embedder.supervision.artifacts import sha256_file, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.utils.config import DownloadConfig, load_config
from naics_embedder.utils.utilities import download_with_retry as _download_with_retry
from naics_embedder.utils.utilities import make_directories
from naics_embedder.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def _read_xlsx_bytes(
    data: bytes, sheet: str, schema: Dict[str, pl.DataType], cols: Dict[str, str]
) -> pl.DataFrame:
    '''
    Read Excel data from bytes into a Polars DataFrame.

    Args:
        data: Excel file content as bytes
        sheet: Sheet name to read
        schema: Column schema mapping
        cols: Column rename mapping

    Returns:
        pl.DataFrame: Processed DataFrame
    '''

    return pl.read_excel(
        BytesIO(data), sheet_name=sheet, columns=list(schema.keys()), schema_overrides=schema
    ).rename(mapping=cols)

def _local_source(url: str, source_dir: str) -> Path:
    '''The local copy of a source file: the URL's file name inside ``source_dir``.'''

    return Path(source_dir).expanduser() / unquote(PurePosixPath(urlparse(url).path).name)

def _read_xlsx(
    url: str,
    sheet: str,
    schema: Dict[str, pl.DataType],
    cols: Dict[str, str],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
    source_dir: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    '''
    Read an Excel file from its URL, or from its local copy in ``source_dir``.

    Raises:
        FileNotFoundError: If ``source_dir`` is set and holds no copy of the file.
        ValueError: If ``expected_sha256`` is set and the file's bytes do not match it.
    '''

    if source_dir is None:
        data = _download_with_retry(url, max_retries, initial_delay, backoff_factor, timeout)
    else:
        path = _local_source(url, source_dir)
        if not path.is_file():
            raise FileNotFoundError(f'no local copy of {url} at {path}')
        data = path.read_bytes()

    if data is None:
        return None

    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f'{url} has sha256 {digest}, not the pinned {expected_sha256}: the index-entry '
                'role table is keyed to row positions in the pinned file'
            )

    return _read_xlsx_bytes(data, sheet, schema, cols)

# -------------------------------------------------------------------------------------------------
# Download files
# -------------------------------------------------------------------------------------------------

def _download_files(cfg: DownloadConfig,
                    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Convert string schema names to Polars types
    schema_codes = {k: getattr(pl, v) for k, v in cfg.schema_codes.items()}
    schema_index = {k: getattr(pl, v) for k, v in cfg.schema_index.items()}
    schema_descriptions = {k: getattr(pl, v) for k, v in cfg.schema_descriptions.items()}
    schema_exclusions = {k: getattr(pl, v) for k, v in cfg.schema_exclusions.items()}

    # NAICS titles
    titles_df = _read_xlsx(
        url=cfg.url_codes,
        sheet=cfg.sheet_codes,
        schema=schema_codes,
        cols=cfg.rename_codes,
        source_dir=cfg.source_dir,
    )

    # NAICS descriptions
    descriptions_df = _read_xlsx(
        url=cfg.url_descriptions,
        sheet=cfg.sheet_descriptions,
        schema=schema_descriptions,
        cols=cfg.rename_descriptions,
        source_dir=cfg.source_dir,
    )

    # NAICS index file for examples, pinned: entry IDs are its row positions
    examples_df = _read_xlsx(
        url=cfg.url_index,
        sheet=cfg.sheet_index,
        schema=schema_index,
        cols=cfg.rename_index,
        source_dir=cfg.source_dir,
        expected_sha256=cfg.index_sha256,
    )

    # NAICS cross reference file for exclusions
    exclusions_df = _read_xlsx(
        url=cfg.url_exclusions,
        sheet=cfg.sheet_exclusions,
        schema=schema_exclusions,
        cols=cfg.rename_exclusions,
        source_dir=cfg.source_dir,
    )

    df_list = [
        ('Titles', titles_df),
        ('Descriptions', descriptions_df),
        ('Examples', examples_df),
        ('Exclusions', exclusions_df),
    ]

    if all(isinstance(df, pl.DataFrame) for _, df in df_list):
        logger.info('Downloaded NAICS files successfully:')

        dfs = []
        for col, df in df_list:
            # yapf: disable
            dfs.append(
                df
                .with_columns(  # type: ignore
                    code=pl.when(pl.col('code').eq('31-33'))
                           .then(pl.lit('31'))
                           .when(pl.col('code').eq('44-45'))
                           .then(pl.lit('44'))
                           .when(pl.col('code').eq('48-49'))
                           .then(pl.lit('48'))
                           .otherwise(pl.col('code'))
                )
            )
            # yapf: enable

            logger.info(f'  {col} observations: {df.height: ,}')  # type: ignore

        logger.info('')

        return dfs[0], dfs[1], dfs[2], dfs[3]

    else:
        raise ValueError('Failed to download one or more NAICS files.')

# -------------------------------------------------------------------------------------------------
# NAICS titles
# -------------------------------------------------------------------------------------------------

def _get_titles(titles_df: pl.DataFrame) -> Tuple[pl.DataFrame, Set[str]]:

    # Load NAICS titles and normalize combined sector codes (31-33, 44-45, 48-49)
    # yapf: disable
    titles = (
        titles_df
        .select(
            index=pl.col('index').sub(1),
            level=pl.col('code').str.len_chars().cast(pl.UInt8),
            code=pl.col('code'),
            title=pl.col('title'),
        )
    )
    # yapf: enable

    # Unique set of NAICS codes
    codes = set(titles.get_column('code').unique().to_list())

    logger.info('Titles:')
    logger.info(f'  Number of titles: {titles.height: ,}')
    logger.info(f'  Number of codes: {len(codes): ,}\n')

    return titles, codes

# -------------------------------------------------------------------------------------------------
# NAICS descriptions 1
# -------------------------------------------------------------------------------------------------

def _get_descriptions_1(descriptions_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:

    # descriptions: normalize combined sector codes
    descriptions_1 = (descriptions_df.select('code', 'description'))

    # Split multiline descriptions into one block per line and filter out section headers and
    # cross-references. The xlsx stores CRLF line endings; openpyxl and newer fastexcel releases
    # normalize them to LF but fastexcel 0.16 keeps them, so normalize before splitting.
    # yapf: disable
    descriptions_2 = (
        descriptions_1
        .with_columns(
            description=pl.col('description')
            .str.replace_all('\r\n', '\n', literal=True)
            .str.split('\n')
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )
        .explode('description')
        .with_columns(description_id=pl.col('description').cum_count().over('code'))
        .select('code', 'description_id', 'description')
        .filter(
            (pl.col('description').ne('The Sector as a Whole')),
            (~pl.col('description').str.contains('Cross-References.')),
            (pl.col('description').str.len_chars().gt(0)),
        )
    )
    # yapf: enable

    # Clean and normalize description text
    # yapf: disable
    descriptions_3 = descriptions_2.select(
        code=pl.col('code').str.strip_chars(),
        description_id=pl.col('description_id'),
        description=pl.col('description')
                      .str.strip_prefix(' ')
                      .str.strip_suffix(' ')
                      .str.replace_all('NULL', '', literal=True)
                      .str.replace_all(r'See industry description for \d{6}\.', '')
                      .str.replace_all(r'<.*?>', '')
                      .str.replace_all(r'\xa0', ' ')
                      .str.replace_all('.', '. ', literal=True)
                      .str.replace_all('U. S. ', 'U.S.', literal=True)
                      .str.replace_all('e. g. ,', 'e.g.,', literal=True)
                      .str.replace_all('i. e. ,', 'i.e.,', literal=True)
                      .str.replace_all(';', '; ', literal=True)
                      .str.replace_all('31-33', '31', literal=True)
                      .str.replace_all('44-45', '44', literal=True)
                      .str.replace_all('48-49', '48', literal=True)
                      .str.replace_all(r'\s{2,}', ' ')
                      .str.strip_prefix(' ')
                      .str.strip_suffix(' '),
    )
    # yapf: enable

    logger.info('Descriptions:')
    logger.info(f'  Number: {descriptions_1.height: ,}')
    logger.info(f'  Number (split on paragraphs): {descriptions_3.height: ,}\n')

    return descriptions_2, descriptions_3

# -------------------------------------------------------------------------------------------------
# NAICS exclusions
# -------------------------------------------------------------------------------------------------

def _get_exclusions(exclusions_df: pl.DataFrame, descriptions_3: pl.DataFrame,
                    codes: Set[str]) -> Tuple[pl.DataFrame, pl.DataFrame]:

    # Load descriptions from cross-reference file
    # yapf: disable
    exclusions_1 = (
        exclusions_df
        .filter(
            pl.col('excluded').str.contains(r' \d{2,6}'),
        )
    )
    # yapf: enable

    # Aggregate exclusions by code
    # yapf: disable
    exclusions_2 = (
        exclusions_1
        .group_by('code', maintain_order=True)
        .agg(excluded=pl.col('excluded'))
        .select(
            code=pl.col('code'),
            description_id=pl.lit(1, pl.UInt32),
            description=pl.col('excluded').list.join(' ')
        )
    )
    # yapf: enable

    # Extract excluded activities (typically last description block for a code)
    # yapf: disable
    exclusions_3 = (
        descriptions_3
        .filter(
            pl.col('description_id').max().over('code').eq(pl.col('description_id')),
            pl.col('description').str.contains_any(['Excluded', 'excluded', 'Exclude', 'exclude']),
            pl.col('description').str.contains(r' \d{2,6}'),
        )
        .select(
            code=pl.col('code').str.strip_chars(),
            description_id=pl.col('description_id'),
            description=pl.col('description'),
        )
    )
    # yapf: enable

    # Exclusions for cleaning descriptions
    descriptions_exclusions = exclusions_3.select('code', 'description_id')

    # Combine and extract excluded codes
    # yapf: disable
    exclusions_4 = (
        pl.concat([exclusions_2, exclusions_3])
        .filter(pl.col('description').is_not_null())
        .with_columns(
            digit=pl.col('description')
                    .str.extract_all(r' \d{2,6}')
                    .list.eval(pl.element().str.strip_prefix(' '))
                    .list.set_intersection(codes)
                    .list.drop_nulls()
                    .list.set_intersection(codes)
                    .list.drop_nulls()
        )
        .filter(pl.col('digit').list.len().gt(0))
    )
    # yapf: enable

    # Final exclusions DataFrame
    # yapf: disable
    exclusions = (
        exclusions_4
        .explode('digit')
        .select(
            level=pl.col('code').str.len_chars().cast(pl.UInt8),
            code=pl.col('code'),
            excluded=pl.col('description'),
            excluded_codes=pl.col('digit'),
        )
        .sort('level', 'code')
        .group_by('level', 'code', maintain_order=True)
        .agg(
            excluded=pl.col('excluded'),
            excluded_codes=pl.col('excluded_codes')
        )
        .with_columns(
            excluded=pl.col('excluded').list.join(' ')
        )
    )
    # yapf: enable

    # yapf: disable
    exclusions_cnt = (
        exclusions
        .with_columns(
            excluded_count=pl.col('excluded_codes').list.len()
        )
        .get_column('excluded_count')
        .sum()
    )
    # yapf: enable

    logger.info('Exclusions:')
    logger.info('  Reference codes:')
    logger.info(f'    Cross-references: {exclusions_2.height: ,}')
    logger.info(f'    Extracted from descriptions: {exclusions_3.height: ,}')
    logger.info(f'    Final: {exclusions.height: ,}')
    logger.info(f'  Excluded codes: {exclusions_cnt: ,}\n')

    return exclusions, descriptions_exclusions

# -------------------------------------------------------------------------------------------------
# NAICS examples
# -------------------------------------------------------------------------------------------------

def _get_index_entries(index_df: pl.DataFrame, codes: Set[str]) -> pl.DataFrame:
    '''
    The index file's entries for six-digit codes (``entry_id``, ``code``, ``text``).

    ``entry_id`` is the row's 0-based position in the index sheet, stable for the pinned file.
    Rows naming no six-digit code (the "see" cross-reference rows, coded ``******``) are dropped,
    and entry text is stripped of surrounding whitespace.
    '''

    six_digit = sorted(code for code in codes if len(code) == 6)
    # yapf: disable
    return (
        index_df
        .with_row_index('entry_id')
        .filter(pl.col('code').is_in(six_digit))
        .select(
            entry_id=pl.col('entry_id').cast(pl.Int64),
            code=pl.col('code'),
            text=pl.col('examples').str.strip_chars(),
        )
    )
    # yapf: enable

def _get_examples(
    index_codes: Set[str],
    examples_entries: pl.DataFrame,
    descriptions_2: pl.DataFrame,
    descriptions_3: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Each code's examples channel, and where each description's examples section starts.

    A code with index entries takes its examples-role entries only (``examples_entries``, in
    index-file order): its other entries are queries and stay out of the channel (Req 3). A code
    without index entries falls back to the bullets after its description's "Illustrative
    Examples:" marker. Either way the marker and its bullets leave the description.
    '''

    outside = sorted(set(examples_entries.get_column('code').to_list()) - index_codes)
    if outside:
        raise ValueError(f'examples entries name codes without index entries: {outside[:5]}')

    # Examples-role index entries, in index-file order
    # yapf: disable
    examples_1 = (
        examples_entries
        .sort('entry_id')
        .group_by('code', maintain_order=True)
        .agg(
            examples_1=pl.col('text')
        )
    )
    # yapf: enable

    # Identify where 'Illustrative Examples:' section begins
    # yapf: disable
    examples_2 = (
        descriptions_2
        .filter(
            pl.col('description').str.contains('Illustrative Examples:')
        )
        .select(
            code=pl.col('code'),
            example_id=pl.col('description_id')
        )
    )
    # yapf: enable

    # Extract examples that appear after 'Illustrative Examples:' marker
    # yapf: disable
    examples_3 = (
        descriptions_3.join(examples_2, how='inner', on='code')
        .filter(pl.col('example_id').lt(pl.col('description_id')))
        .group_by('code', maintain_order=True)
        .agg(examples_2=pl.col('description'), description_id_min=pl.col('example_id').min())
    )
    # yapf: enable

    # Description IDs to exclude in description dataframe, starting at the marker itself
    descriptions_examples = examples_3.select('code', 'description_id_min')

    # Codes without index entries fall back to their description's illustrative examples
    fallback = examples_3.filter(~pl.col('code').is_in(sorted(index_codes)))
    examples_4 = examples_1.join(fallback, how='full', on='code', coalesce=True).select(
        code=pl.col('code'), examples=pl.coalesce('examples_1', 'examples_2')
    )

    examples = examples_4.select(code=pl.col('code'), examples=pl.col('examples').list.join('; '))

    # yapf: disable
    examples_cnt = (
        examples_4.with_columns(example_cnt=pl.col('examples').list.len())
        .get_column('example_cnt')
        .sum()
    )
    # yapf: enable

    logger.info('Examples:')
    logger.info('  Reference codes:')
    logger.info(f'    Index entries (examples role): {examples_1.height: ,}')
    logger.info(f'    Extracted from descriptions: {fallback.height: ,}')
    logger.info(f'    Final: {examples.height: ,}')
    logger.info(f'  Number of examples: {examples_cnt: ,}\n')

    return examples, descriptions_examples

# -------------------------------------------------------------------------------------------------
# NAICS description 2 (cleaned descriptions)
# -------------------------------------------------------------------------------------------------

def _get_descriptions_2(
    descriptions_3: pl.DataFrame,
    descriptions_exclusions: pl.DataFrame,
    descriptions_examples: pl.DataFrame,
) -> pl.DataFrame:

    # descriptions: exclude exclusion and example description blocks
    # yapf: disable
    descriptions_4 = (
        descriptions_3
        .join(
            descriptions_exclusions,
            how='anti',
            on=['code', 'description_id']
        )
        .join(
            descriptions_examples,
            how='left',
            on='code'
        )
        .with_columns(
            pl.col('description_id_min').fill_null(999)
        )
        .filter(
            pl.col('description_id').lt(pl.col('description_id_min'))
        )
        .group_by('code', maintain_order=True)
        .agg(
            pl.col('description')
        )
        .with_columns(
            description=pl.col('description').list.join(' ')
        )
    )
    # yapf: enable

    # Separate complete descriptions from missing ones
    # yapf: disable
    description_complete_1 = (
        descriptions_4
        .filter(
            pl.col('description').ne('')
        )
    )
    # yapf: enable

    # Find 4-digit codes missing descriptions
    description_4_missing = descriptions_4.filter(
        pl.col('code').str.len_chars().eq(4),
        pl.col('description').eq('')
    ).select(
        code1=pl.col('code').str.pad_end(5, '1'),
        code2=pl.col('code').str.pad_end(5, '2'),
        code3=pl.col('code').str.pad_end(5, '3'),
        code4=pl.col('code').str.pad_end(5, '4'),
        code9=pl.col('code').str.pad_end(5, '9'),
    )

    # Find 5-digit codes missing descriptions
    description_5_missing = descriptions_4.filter(
        pl.col('code').str.len_chars().eq(5),
        pl.col('description').eq('')
    ).select(code=pl.col('code').str.pad_end(6, '0'))

    logger.info('NAICS missing descriptions:')
    logger.info(f'  Total: {descriptions_4.height: ,}')
    logger.info(f'  Complete: {description_complete_1.height: ,}')
    logger.info(f'  Missing (level 4): {description_4_missing.height: ,}')
    logger.info(f'  Missing (level 5): {description_5_missing.height: ,}\n')

    # Fill missing 5-digit descriptions from 6-digit children
    # yapf: disable
    description_5_complete = (
        description_5_missing.join(description_complete_1, how='inner', on='code')
        .with_columns(code=pl.col('code').str.slice(0, 5))
        .select(
            code=pl.col('code'),
            description=pl.col('description').str.replace(
                'This industry', 'This NAICS industry', literal=True
            ),
        )
    )
    # yapf: enable

    description_complete_2 = pl.concat([description_complete_1, description_5_complete])

    # Fill missing 4-digit descriptions from 5-digit children (try multiple suffixes)
    description_4_complete_1 = description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code1'
    )

    description_4_complete_2 = description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code2'
    )

    description_4_complete_3 = description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code3'
    )

    description_4_complete_4 = description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code4'
    )

    description_4_complete_9 = description_4_missing.join(
        description_complete_2, how='inner', right_on='code', left_on='code9'
    )

    # yapf: disable
    description_4_complete = (
        pl.concat(
            [
                description_4_complete_1,
                description_4_complete_2,
                description_4_complete_3,
                description_4_complete_4,
                description_4_complete_9,
            ]
        )
        .select(
            code=pl.col('code1').str.slice(0, 4),
            description=pl.col('description')
            .str.replace('This industry', 'This industry group', literal=True)
            .str.replace('This NAICS industry', 'This industry group', literal=True),
        )
        .unique(subset=['code'])
    )
    # yapf: enable

    # Combine all descriptions
    descriptions = pl.concat([description_complete_2, description_4_complete])

    logger.info('NAICS completed descriptions:')
    logger.info(f'  Missing (level 4): {description_4_missing.height: ,}')
    logger.info(f'  Filled missing (level 4): {description_4_complete.height: ,}')
    logger.info(f'  Missing (level 5): {description_5_missing.height: ,}')
    logger.info(f'  Filled missing (level 5): {description_5_complete.height: ,}')
    logger.info(f'  Complete: {descriptions.height: ,}')

    return descriptions

# -------------------------------------------------------------------------------------------------
# Sources and the combined descriptions
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NaicsSources:
    '''The four Census NAICS files as read, combined sector codes normalized.'''

    titles: pl.DataFrame
    descriptions: pl.DataFrame
    index: pl.DataFrame
    exclusions: pl.DataFrame

def load_naics_sources(cfg: DownloadConfig) -> NaicsSources:
    '''Read the four files, from ``cfg.source_dir`` when set; the index file must match its pin.'''

    return NaicsSources(*_download_files(cfg))

def naics_index_entries(sources: NaicsSources) -> pl.DataFrame:
    '''The index file's entries for six-digit codes (``entry_id``, ``code``, ``text``).'''

    entries = _get_index_entries(sources.index, set(sources.titles.get_column('code').to_list()))
    logger.info('Index entries:')
    logger.info(f'  Sheet rows: {sources.index.height: ,}')
    logger.info(f'  Entries for six-digit codes: {entries.height: ,}')
    logger.info(f'  Codes with entries: {entries.get_column("code").n_unique(): ,}\n')
    return entries

def build_descriptions(sources: NaicsSources, examples_entries: pl.DataFrame) -> pl.DataFrame:
    '''
    One row per code: title, description, examples channel and exclusions.

    Args:
        sources: The four Census files.
        examples_entries: The index entries that form examples channels (``entry_id``,
            ``code``, ``text``); every other entry of a code with index entries is a query.
    '''

    titles, codes = _get_titles(sources.titles)

    descriptions_2, descriptions_3 = _get_descriptions_1(sources.descriptions)

    exclusions, descriptions_exclusions = _get_exclusions(sources.exclusions, descriptions_3, codes)

    index_codes = set(_get_index_entries(sources.index, codes).get_column('code').to_list())
    examples, descriptions_examples = _get_examples(
        index_codes, examples_entries, descriptions_2, descriptions_3
    )

    descriptions = _get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )

    # yapf: disable
    return (
        titles.join(descriptions, how='inner', on='code')
        .join(exclusions, how='left', on='code')
        .join(examples, how='left', on='code')
        .select(
            index=pl.col('index'),
            level=pl.col('level'),
            code=pl.col('code'),
            title=pl.col('title'),
            description=pl.col('description'),
            examples=pl.col('examples'),
            excluded=pl.col('excluded'),
            excluded_codes=pl.col('excluded_codes'),
        )
        .sort('index')
    )
    # yapf: enable

# -------------------------------------------------------------------------------------------------
# Guard the descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------

# Where shipped configs name a supervision bundle: (config file, key path)
PINNING_CONFIGS: Tuple[Tuple[Path, Tuple[str, ...]], ...] = (
    (Path('conf/config.yaml'), ('supervision', 'manifest_path')),
    (Path('conf/graph.yaml'), ('supervision_manifest_path', )),
)

def pinned_description_fingerprints(
    pinning_configs: Sequence[Tuple[Path, Tuple[str, ...]]] = PINNING_CONFIGS,
) -> Dict[str, str]:
    '''
    The ``description_fingerprint`` of each bundle a config names, by manifest path.

    Configs, keys and manifests that do not exist are skipped: they pin nothing.
    '''

    fingerprints: Dict[str, str] = {}
    for config_path, keys in pinning_configs:
        if not Path(config_path).is_file():
            continue
        value = yaml.safe_load(Path(config_path).read_text())
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if not value:
            continue
        manifest_path = Path(value)
        if not manifest_path.is_file():
            logger.warning(f'{config_path} names a missing supervision manifest: {manifest_path}')
            continue
        manifest = json.loads(manifest_path.read_text())
        fingerprints[str(manifest_path)] = manifest['description_fingerprint']
    return fingerprints

def refuse_pinned_overwrite(
    output: Path,
    *,
    force: bool,
    pinning_configs: Sequence[Tuple[Path, Tuple[str, ...]]] = PINNING_CONFIGS,
) -> None:
    '''
    Refuse to overwrite the descriptions file a configured supervision bundle pins.

    Training fails closed once the descriptions file no longer matches its bundle's
    ``description_fingerprint``, so rewriting it would break every run against that bundle.

    Raises:
        FileExistsError: If ``output`` is pinned and ``force`` is False.
    '''

    if force or not Path(output).is_file():
        return
    digest = sha256_file(Path(output))
    pinned = sorted(
        path for path, fingerprint in pinned_description_fingerprints(pinning_configs).items()
        if fingerprint == digest
    )
    if pinned:
        raise FileExistsError(
            f'{output} is the descriptions file supervision bundle {pinned[0]} pins; rebuilding '
            'it would break training against that bundle. Write another output_parquet, or '
            'pass --force to overwrite it.'
        )

# -------------------------------------------------------------------------------------------------
# Combine all and write final output
# -------------------------------------------------------------------------------------------------

def download_preprocess_data(
    cfg: Optional[DownloadConfig] = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    '''
    Build the descriptions parquet and the index-roles parquet from the Census files.

    Every index entry takes its role from the frozen role table (``cfg.index_roles_csv``). A
    code's examples channel holds its examples-role entries only, and no validation or test query
    may match any training text (Req 3); both are checked before anything is written.

    Args:
        cfg: Download configuration; ``conf/data/download.yaml`` when omitted.
        force: Overwrite a descriptions file that a configured supervision bundle pins.
    '''

    # Create directories
    make_directories()

    # Load configuration from YAML
    if cfg is None:
        cfg = load_config(DownloadConfig, './data/download.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    refuse_pinned_overwrite(Path(cfg.output_parquet), force=force)

    roles_csv = Path(cfg.index_roles_csv)
    if not roles_csv.is_file():
        raise FileNotFoundError(
            f'index-entry role table not found: {roles_csv}; generate it once with '
            '`naics-embedder data roles`'
        )

    sources = load_naics_sources(cfg)

    role_rows = attach_role_text(read_role_table(roles_csv), naics_index_entries(sources))

    naics_final = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value)
    )

    six_digit_codes = naics_final.filter(pl.col('level') == 6).get_column('code').to_list()
    validate_index_role_table(role_rows, six_digit_codes)
    verify_examples_channel(naics_final, role_rows)
    leakage = verify_role_leakage(naics_final, role_rows)
    logger.info(f'Held-out queries matching training text: {leakage}\n')

    (naics_final.write_parquet(cfg.output_parquet))

    _parquet_stats(
        parquet_df=naics_final,
        message='NAICS codes (text + hierarchy) written to:',
        output_parquet=cfg.output_parquet,
        logger=logger,
    )

    Path(cfg.index_roles_parquet).parent.mkdir(parents=True, exist_ok=True)
    role_rows.write_parquet(cfg.index_roles_parquet)

    _parquet_stats(
        parquet_df=role_rows,
        message='NAICS index entries and their roles written to',
        output_parquet=cfg.index_roles_parquet,
        logger=logger,
    )

    return naics_final

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    download_preprocess_data()
