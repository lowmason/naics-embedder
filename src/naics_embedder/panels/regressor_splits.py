'''
The regressor panel's sealed outer sets (Req 2; Req 4; roadmap Stage 3; D7).

Every panel row is a code in a feature year. One partition serves both regimes, so that no
validation read touches either regime's sealed set:

- **Held-out outer set (H).** Every row of a code whose employment includes a held-out four-digit
  group: at levels 4–6 the code's own four-digit ancestor (or the code itself) is held out; at
  levels 2 and 3 a held-out group rolls into the code.
- **Seen outer set (S).** The remaining rows with feature year 2024, whose outcome is 2025 (D7).
- **Remainder (R).** The remaining rows, feature years 2022 and 2023. Validation reads only these.

The held-out groups are drawn once, stratified by sector, and committed
(``conf/data/regressor_heldout_groups.csv``). The split's fingerprint is the sha256 of that
table's canonical CSV, which equals the committed file's hash: a redraw gets a new fingerprint,
so it would not count as a reopening, and the table is therefore never redrawn.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import math
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Collection, Dict, List, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.qcew_rows import FEATURE_YEARS
from naics_embedder.utils.naics_hierarchy import naics_parent_code

# D7: the seen regime's outer set is the last feature year; the remainder keeps the earlier two
SEALED_FEATURE_YEAR = FEATURE_YEARS[-1]
REMAINDER_FEATURE_YEARS = FEATURE_YEARS[:-1]
GROUP_LEVEL = 4
SECTOR_LEVEL = 2
GROUP_TABLE_SCHEMA = {'group': pl.Utf8}

class RegressorSplit(str, Enum):
    '''Which part of the partition a panel row belongs to.'''

    REMAINDER = 'remainder'
    SEEN_OUTER = 'seen_outer'
    HELDOUT_OUTER = 'heldout_outer'

# -------------------------------------------------------------------------------------------------
# Hierarchy
# -------------------------------------------------------------------------------------------------

def ancestor_at(code: str, level: int) -> str:
    '''The code's ancestor at a level (the code itself at its own level); 31-33 is one sector.'''

    if not SECTOR_LEVEL <= level <= len(code):
        raise ValueError(f'{code} has no ancestor at level {level}')
    while len(code) > level:
        parent = naics_parent_code(code)
        if parent is None:
            raise ValueError(f'{code} has no parent')
        code = parent
    return code

def group_of(code: str) -> str:
    '''
    The row's group: its four-digit ancestor at levels 4–6, the code itself at levels 2 and 3.

    Folds and Stage 4's resampling run over groups (Req 5: four-digit-parent groups).
    '''

    return ancestor_at(code, GROUP_LEVEL) if len(code) >= GROUP_LEVEL else code

def codes_fingerprint(codes: Collection[str]) -> str:
    '''SHA-256 of the codes, sorted, one per line: the codebook's content, whatever its file.'''

    return hashlib.sha256(''.join(f'{code}\n'
                                  for code in sorted(codes)).encode('utf-8')).hexdigest()

def read_codebook_codes(path: Path, expected_sha256: str) -> Tuple[str, ...]:
    '''
    A supervision bundle codebook's codes, sorted, after checking their fingerprint.

    Raises:
        ValueError: If a code repeats or the codes' fingerprint differs from ``expected_sha256``.
    '''

    codes = pl.read_parquet(Path(path), columns=['code']).get_column('code').to_list()
    if len(set(codes)) != len(codes):
        raise ValueError(f'{path}: the codebook repeats a code')
    digest = codes_fingerprint(codes)
    if digest != expected_sha256:
        raise ValueError(f'{path}: codes fingerprint {digest} does not match {expected_sha256}')
    return tuple(sorted(codes))

# -------------------------------------------------------------------------------------------------
# The held-out draw
# -------------------------------------------------------------------------------------------------

def sector_quotas(counts: Dict[str, int], fraction: Fraction, seed: int) -> Dict[str, int]:
    '''
    Largest-remainder quotas of ``fraction`` of each sector's groups.

    The total is ``fraction`` of all groups, rounded half up; remainder ties go to the smaller of
    per-sector draws from ``np.random.default_rng([seed])``, taken in sorted sector order.
    '''

    if not 0 < fraction < 1:
        raise ValueError(f'fraction must lie in (0, 1), got {fraction}')
    sectors = sorted(counts)
    exact = {sector: fraction * counts[sector] for sector in sectors}
    quotas = {sector: math.floor(exact[sector]) for sector in sectors}
    total = math.floor(fraction * sum(counts.values()) + Fraction(1, 2))
    tie_break = dict(zip(sectors, np.random.default_rng([seed]).random(len(sectors))))
    order = sorted(
        sectors, key=lambda sector: (-(exact[sector] - quotas[sector]), tie_break[sector])
    )
    for sector in order[:total - sum(quotas.values())]:
        quotas[sector] += 1
    return quotas

def draw_heldout_groups(six_digit_codes: Collection[str], fraction: Fraction,
                        seed: int) -> Tuple[str, ...]:
    '''
    Draw the held-out four-digit groups, stratified by sector.

    Args:
        six_digit_codes: The six-digit population (Stage 1's 980 codes).
        fraction: Share of each sector's groups to hold out (largest-remainder quotas).
        seed: Base seed; sector s draws its groups with ``np.random.default_rng([seed, int(s)])``.

    Returns:
        The held-out groups, sorted.
    '''

    by_sector: Dict[str, List[str]] = {}
    for group in sorted({ancestor_at(code, GROUP_LEVEL) for code in six_digit_codes}):
        by_sector.setdefault(ancestor_at(group, SECTOR_LEVEL), []).append(group)
    quotas = sector_quotas(
        {
            sector: len(groups)
            for sector, groups in by_sector.items()
        }, fraction, seed
    )
    drawn: List[str] = []
    for sector in sorted(by_sector):
        order = np.random.default_rng([seed, int(sector)]).permutation(len(by_sector[sector]))
        drawn.extend(by_sector[sector][index] for index in order[:quotas[sector]])
    return tuple(sorted(drawn))

# -------------------------------------------------------------------------------------------------
# The committed table
# -------------------------------------------------------------------------------------------------

def _group_table_csv(groups: Collection[str]) -> bytes:
    frame = pl.DataFrame({'group': sorted(groups)}, schema=GROUP_TABLE_SCHEMA)
    return frame.write_csv().encode('utf-8')

def group_table_fingerprint(groups: Collection[str]) -> str:
    '''SHA-256 of the held-out groups' canonical CSV, which equals the committed file's hash.'''

    return hashlib.sha256(_group_table_csv(groups)).hexdigest()

def write_group_table(groups: Collection[str], path: Path) -> str:
    '''Write the canonical CSV and return its sha256 (``group_table_fingerprint``).'''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_group_table_csv(groups))
    return group_table_fingerprint(groups)

def read_group_table(path: Path) -> Tuple[str, ...]:
    '''
    Read the committed held-out groups.

    Raises:
        ValueError: If a group is not a distinct four-digit code.
    '''

    groups = pl.read_csv(Path(path), schema=GROUP_TABLE_SCHEMA).get_column('group').to_list()
    if len(set(groups)) != len(groups) or any(len(group) != GROUP_LEVEL for group in groups):
        raise ValueError(f'{path}: held-out groups must be distinct four-digit codes')
    return tuple(sorted(groups))

# -------------------------------------------------------------------------------------------------
# The partition
# -------------------------------------------------------------------------------------------------

def heldout_tainted(code: str, heldout_groups: Collection[str]) -> bool:
    '''Whether the code's employment includes a held-out group.'''

    if len(code) >= GROUP_LEVEL:
        return ancestor_at(code, GROUP_LEVEL) in set(heldout_groups)
    return any(ancestor_at(group, len(code)) == code for group in heldout_groups)

def assign_splits(rows: pl.DataFrame, heldout_groups: Collection[str]) -> pl.DataFrame:
    '''
    Add each row's ``group`` and ``split`` (``RegressorSplit``) to panel rows.

    Args:
        rows: Panel rows with ``code`` and ``feature_year``.
        heldout_groups: The committed held-out four-digit groups.
    '''

    codes = sorted(set(rows.get_column('code').to_list()))
    held = set(heldout_groups)
    info = pl.DataFrame(
        {
            'code': codes,
            'group': [group_of(code) for code in codes],
            'tainted': [heldout_tainted(code, held) for code in codes],
        },
        schema={
            'code': pl.Utf8,
            'group': pl.Utf8,
            'tainted': pl.Boolean
        },
    )
    # yapf: disable
    split = (
        pl.when(pl.col('tainted')).then(pl.lit(RegressorSplit.HELDOUT_OUTER.value))
        .when(pl.col('feature_year') == SEALED_FEATURE_YEAR)
        .then(pl.lit(RegressorSplit.SEEN_OUTER.value))
        .otherwise(pl.lit(RegressorSplit.REMAINDER.value))
    )
    # yapf: enable
    return rows.join(info, on='code', how='left').with_columns(split=split).drop('tainted')

def split_counts(rows: pl.DataFrame) -> Dict[str, int]:
    '''Rows per split, every split named (zero when empty).'''

    counts = dict(rows.group_by('split').len().iter_rows())
    return {split.value: int(counts.get(split.value, 0)) for split in RegressorSplit}

def check_partition(rows: pl.DataFrame, codes: Sequence[str]) -> None:
    '''
    Require exactly one row per code and feature year, each in exactly one split.

    Raises:
        ValueError: If a code-year is missing, repeated, or has no split.
    '''

    expected = len(set(codes)) * len(FEATURE_YEARS)
    if rows.height != expected or rows.select('code', 'feature_year').is_duplicated().any():
        raise ValueError(f'{rows.height:,} rows for {expected:,} code-years')
    if rows.get_column('split').is_null().any():
        raise ValueError('a row has no split')
