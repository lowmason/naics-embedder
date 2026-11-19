'''
Stage 10's two ITA tables on a miniature panel, and a stub arm that decodes by assignment (plan 17).

``ita_tables`` builds the pair table and the pair-code table with the exact column names, order
and dtypes of embedding-data's spec (commit ec6fb29), copied into ``ITA_PAIRS_SCHEMA`` and
``ITA_PAIR_CODES_SCHEMA``. The base fixture has eleven pairs over the eight codes of
``ITA_CANDIDATES``: copied pairs, a conflict pair, a company-fallback pair, a mojibake pair, two
employers with one description (p01 and p02), a pair that reports one node under two vintages and a
2017 crosswalked row that asserts no node and sits on a code outside the candidates (p01), and a
pair with ``flag_headline`` false (p06). A node id of another level on a reported row, and a
reported code outside the candidates, are refused by the panel, so the base holds neither on a
reported row: ``replace_codes`` builds them. The salt of ``ITA_SETTINGS`` puts p01 through p07 in
the validation part and p08 through p11 in the sealed part.

A row an extra argument adds, or a replacement ``replace_codes`` gives, is completed with neutral
values: false flags, zero counts, empty lists and null codes and years. Nothing here reads
embedding-data. ``write_ita_tables`` writes under a directory the test gives, ``tmp_path``.
``ita_run_dir`` writes them under a run directory beside an ``outputs.json``, and returns the
evidence that pins the three files (C1, plan 20). ``IdentifiedStubEncoder`` is the stub with a d16
read's identities, and ``outcome_beside_panel`` an outcome panel whose validation queries it
decodes, for a read beside an ITA read.
'''

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import polars as pl
import torch

from naics_embedder.panels.ita_evidence import OUTPUTS_FILE, ItaEvidence
from naics_embedder.panels.ita_split import ItaPanelSettings
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import sha256_file

# Stage 10's ita_text_pairs.parquet: every column, in its order, with its dtype
ITA_PAIRS_SCHEMA: Dict[str, pl.DataType] = {
    'pair_id': pl.Utf8,
    'pair_kind': pl.Utf8,
    'employer_key': pl.Utf8,
    'employer_basis': pl.Utf8,
    'description_key': pl.Utf8,
    'original_strings': pl.List(pl.Utf8),
    'representative_string': pl.Utf8,
    'n_records': pl.UInt64,
    'n_headline_records': pl.UInt64,
    'first_year': pl.Int64,
    'last_year': pl.Int64,
    'n_years': pl.UInt64,
    'flag_headline': pl.Boolean,
    'flag_headline_codes_conflict': pl.Boolean,
    'flag_empty': pl.Boolean,
    'flag_placeholder': pl.Boolean,
    'flag_copied_title_any': pl.Boolean,
    'flag_copied_index_any': pl.Boolean,
    'flag_suspected_mojibake': pl.Boolean,
    'flag_employer_unknown': pl.Boolean,
    'flag_employer_company_fallback': pl.Boolean,
    'n_records_ein_padded': pl.UInt64,
}

# Stage 10's ita_text_pair_codes.parquet: one row per pair and observed native code
ITA_PAIR_CODES_SCHEMA: Dict[str, pl.DataType] = {
    'pair_id': pl.Utf8,
    'naics_code': pl.Utf8,
    'naics_year': pl.Int64,
    'naics_mapping_status': pl.Utf8,
    'naics_2022_candidates': pl.List(pl.Utf8),
    'naics_2022_candidate_node_ids': pl.List(pl.Utf8),
    'naics_2022_node_id': pl.Utf8,
    'naics_deepest_common_node_id': pl.Utf8,
    'flag_naics_crosswalked': pl.Boolean,
    'flag_naics_ambiguous': pl.Boolean,
    'flag_naics_unresolved': pl.Boolean,
    'flag_headline_basis': pl.Boolean,
    'flag_copied_title_native': pl.Boolean,
    'flag_copied_index_native': pl.Boolean,
    'n_records': pl.UInt64,
    'n_records_year_from_dictionary': pl.UInt64,
    'years': pl.List(pl.Int64),
}

PAIRS_FILE = 'ita_text_pairs.parquet'
PAIR_CODES_FILE = 'ita_text_pair_codes.parquet'

# The protocol commit a fixture run's evidence names: any 40 hexadecimal characters
FIXTURE_PROTOCOL_COMMIT = '0' * 40

# The panel's codes: the axes of the stub encoder, in order
ITA_CANDIDATES = ('111110', '111120', '211120', '236220', '311111', '321113', '513210', '621210')

# The salt is the committed one: the fixture's parts are those of conf/data/ita_panel.yaml
ITA_SETTINGS = ItaPanelSettings(
    panel='ita_text', rule_version=1, salt='9986637c5a4468938b7bd5c4f1e8a4cd', relative_margin=0.05
)

# Each headline pair's query, and the candidate the stub decodes it to: p03's differs from its
# reported code, and p06 (flag_headline false) has no entry
ASSIGNED = {
    'We make dog food': '311111',
    'we make dog food': '311111',
    'Oil and gas wells': '111110',
    'Cat food plant': '111110',
    'Software publisher': '513210',
    'General contractor': '236220',
    'Soybean farm': '111110',
    'Canola growers': '111120',
    'Sawmill': '321113',
    'Dental office': '621210',
}

# -------------------------------------------------------------------------------------------------
# Neutral values
# -------------------------------------------------------------------------------------------------

_NEUTRAL_PAIR: Dict[str, Any] = {
    'pair_kind': 'employer',
    'employer_basis': 'ein',
    'n_records': 0,
    'n_headline_records': 0,
    'first_year': None,
    'last_year': None,
    'n_years': 0,
    'n_records_ein_padded': 0,
    **{
        name: False
        for name in ITA_PAIRS_SCHEMA if name.startswith('flag_')
    },
}
_NEUTRAL_CODE: Dict[str, Any] = {
    'naics_year': None,
    'naics_mapping_status': 'mapped',
    'naics_2022_candidates': [],
    'naics_2022_candidate_node_ids': [],
    'naics_2022_node_id': None,
    'naics_deepest_common_node_id': None,
    'n_records': 0,
    'n_records_year_from_dictionary': 0,
    'years': [],
    **{
        name: False
        for name in ITA_PAIR_CODES_SCHEMA if name.startswith('flag_')
    },
}

def _check_columns(row: Mapping[str, Any], schema: Mapping[str, pl.DataType]) -> None:
    unknown = sorted(set(row) - set(schema))
    if unknown:
        raise ValueError(f'columns Stage 10 does not have: {unknown}')

def _complete_pair(row: Mapping[str, Any]) -> Dict[str, Any]:
    _check_columns(row, ITA_PAIRS_SCHEMA)
    return {
        **_NEUTRAL_PAIR,
        'employer_key': f'employer-{row["pair_id"]}',
        'original_strings': [row['representative_string']],
        **row,
    }

def _complete_code(row: Mapping[str, Any]) -> Dict[str, Any]:
    _check_columns(row, ITA_PAIR_CODES_SCHEMA)
    completed = {**_NEUTRAL_CODE, **row}
    # Stage 10: a native copied flag is null exactly when the year is
    if completed['naics_year'] is None:
        completed['flag_copied_title_native'] = None
        completed['flag_copied_index_native'] = None
    return completed

# -------------------------------------------------------------------------------------------------
# The base fixture
# -------------------------------------------------------------------------------------------------

def _node(code: str) -> str:
    return f'naics:2022:national_industry:{code}'

def _basis_row(
    pair_id: str,
    code: str,
    years: Sequence[int],
    naics_year: Optional[int] = 2022,
) -> Dict[str, Any]:
    return _complete_code(
        {
            'pair_id': pair_id,
            'naics_code': code,
            'naics_year': naics_year,
            'naics_2022_candidates': [code],
            'naics_2022_candidate_node_ids': [_node(code)],
            'naics_2022_node_id': _node(code),
            'flag_headline_basis': True,
            'n_records': 1,
            'years': list(years),
        }
    )

# A 2017 code that crosswalks to two 2022 codes: Stage 4 asserts no node and its records are not
# headline-basis ones
_CROSSWALK_ROW = _complete_code(
    {
        'pair_id': 'p01',
        'naics_code': '311119',
        'naics_year': 2017,
        'naics_mapping_status': 'crosswalked',
        'naics_2022_candidates': ['311111', '311119'],
        'naics_2022_candidate_node_ids': [_node('311111'), _node('311119')],
        'flag_naics_crosswalked': True,
        'flag_naics_ambiguous': True,
        'n_records': 1,
        'years': [2023],
    }
)

# pair_id, description_key, representative_string, and the flags and basis that are not neutral
_BASE_PAIRS = (
    ('p01', 'we make dog food', 'We make dog food', {
        'flag_headline': True
    }),
    ('p02', 'we make dog food', 'we make dog food', {
        'flag_headline': True
    }),
    ('p03', 'oil and gas wells', 'Oil and gas wells', {
        'flag_headline': True
    }),
    (
        'p04', 'cat food plant', 'Cat food plant', {
            'flag_headline': True,
            'flag_headline_codes_conflict': True
        }
    ),
    (
        'p05', 'software publisher', 'Software publisher', {
            'flag_headline': True,
            'flag_copied_title_any': True
        }
    ),
    ('p06', 'corn farming', 'Corn farming', {}),
    (
        'p07', 'general contractor', 'General contractor', {
            'flag_headline': True,
            'flag_employer_company_fallback': True,
            'employer_basis': 'company',
        }
    ),
    ('p08', 'soybean farm', 'Soybean farm', {
        'flag_headline': True
    }),
    ('p09', 'canola growers', 'Canola growers', {
        'flag_headline': True
    }),
    ('p10', 'sawmill', 'Sawmill', {
        'flag_headline': True,
        'flag_suspected_mojibake': True
    }),
    (
        'p11', 'dental office', 'Dental office', {
            'flag_headline': True,
            'flag_copied_index_any': True
        }
    ),
)

_BASE_CODES: Dict[str, List[Dict[str, Any]]] = {
    'p01': [
        _basis_row('p01', '311111', [2023], naics_year=2022),
        _basis_row('p01', '311111', [2024], naics_year=None),
        _CROSSWALK_ROW,
    ],
    'p02': [_basis_row('p02', '311111', [2023])],
    'p03': [_basis_row('p03', '211120', [2023])],
    'p04': [_basis_row('p04', '311111', [2023]),
            _basis_row('p04', '111110', [2023])],
    'p05': [_basis_row('p05', '513210', [2023])],
    'p06': [_basis_row('p06', '111110', [2023])],
    'p07': [_basis_row('p07', '236220', [2023])],
    'p08': [_basis_row('p08', '111110', [2023])],
    'p09': [_basis_row('p09', '111120', [2023])],
    'p10': [_basis_row('p10', '321113', [2023])],
    'p11': [_basis_row('p11', '621210', [2023])],
}

def _base_pair(pair_id: str, key: str, representative: str, extra: Mapping[str, Any]) -> Dict:
    rows = _BASE_CODES[pair_id]
    years = sorted({year for row in rows for year in row['years']})
    return _complete_pair(
        {
            'pair_id': pair_id,
            'description_key': key,
            'representative_string': representative,
            'n_records': sum(row['n_records'] for row in rows),
            'n_headline_records': sum(
                row['n_records'] for row in rows if row['flag_headline_basis']
            ),
            'first_year': years[0],
            'last_year': years[-1],
            'n_years': len(years),
            **extra,
        }
    )

# -------------------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------------------

def _frame(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame({name: [row[name] for row in rows] for name in schema}, schema=dict(schema))

def ita_tables(
    extra_pairs: Sequence[Dict[str, Any]] = (),
    extra_codes: Sequence[Dict[str, Any]] = (),
    replace_codes: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    The base fixture's pair table and pair-code table, with changes.

    Args:
        extra_pairs: Rows to add to the pair table. A row names its Stage 10 columns, at least
            ``pair_id``, ``description_key`` and ``representative_string``. Neutral values fill the
            rest. A repeated ``pair_id`` is kept, so a test can build one.
        extra_codes: Rows to add to the pair-code table, at least ``pair_id`` and ``naics_code``.
        replace_codes: For a base ``pair_id``, the rows that replace all of that pair's code rows.

    Returns:
        The pair table sorted by ``pair_id``, and the pair-code table sorted by ``pair_id``,
        ``naics_code`` and ``naics_year`` (nulls first), each cast to its schema.

    Raises:
        ValueError: When a row names a column Stage 10 does not have, or ``replace_codes`` names a
            pair that is not in the base fixture.
    '''

    code_rows = {pair_id: list(rows) for pair_id, rows in _BASE_CODES.items()}
    for pair_id, rows in (replace_codes or {}).items():
        if pair_id not in code_rows:
            raise ValueError(f'replace_codes names {pair_id}, which is not a base pair')
        code_rows[pair_id] = [{'pair_id': pair_id, **row} for row in rows]
    pair_rows = [_base_pair(*base) for base in _BASE_PAIRS]
    pair_rows += [_complete_pair(row) for row in extra_pairs]
    all_codes = [row for rows in code_rows.values() for row in rows]
    all_codes += list(extra_codes)
    pairs = _frame(pair_rows, ITA_PAIRS_SCHEMA).sort('pair_id', maintain_order=True)
    codes = _frame([_complete_code(row) for row in all_codes], ITA_PAIR_CODES_SCHEMA)
    codes = codes.sort(
        ['pair_id', 'naics_code', 'naics_year'], nulls_last=False, maintain_order=True
    )
    return pairs, codes

def write_ita_tables(directory: Path, pairs: pl.DataFrame, codes: pl.DataFrame) -> Tuple[Path, Path]:
    '''
    Write the two tables under their Stage 10 file names.

    Args:
        directory: An existing directory, ``tmp_path`` in a test.
        pairs: The pair table.
        codes: The pair-code table.

    Returns:
        The pair table's path and the pair-code table's path.
    '''

    pairs_path = directory / PAIRS_FILE
    codes_path = directory / PAIR_CODES_FILE
    pairs.write_parquet(pairs_path)
    codes.write_parquet(codes_path)
    return pairs_path, codes_path

def ita_run_dir(
    directory: Path,
    pairs: pl.DataFrame,
    codes: pl.DataFrame,
    run: str = 'stage10_fixture_r1',
) -> Tuple[Path, Path, ItaEvidence]:
    '''
    Write the two tables and an ``outputs.json`` under a run directory, and pin them (C1).

    Args:
        directory: An existing directory, ``tmp_path`` in a test.
        pairs: The pair table.
        codes: The pair-code table.
        run: The run directory's name, created under ``directory``.

    Returns:
        The pair table's path, the pair-code table's path, and the evidence that pins the run's
        three files.
    '''

    run_dir = directory / run
    run_dir.mkdir(parents=True)
    pairs_path, codes_path = write_ita_tables(run_dir, pairs, codes)
    outputs = run_dir / OUTPUTS_FILE
    outputs.write_text(
        json.dumps({
            'run': run,
            'tables': [PAIRS_FILE, PAIR_CODES_FILE]
        }, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    evidence = ItaEvidence(
        protocol_commit=FIXTURE_PROTOCOL_COMMIT,
        run=run,
        outputs_sha256=sha256_file(outputs),
        pairs_sha256=sha256_file(pairs_path),
        pair_codes_sha256=sha256_file(codes_path),
    )
    return pairs_path, codes_path, evidence

# -------------------------------------------------------------------------------------------------
# The stub arm
# -------------------------------------------------------------------------------------------------

class AssignedStubEncoder:
    '''
    A stub arm: a code is one-hot on its place in ``ITA_CANDIDATES``, and a query is one-hot on the
    place of the code ``ASSIGNED`` gives it. It sets no ``distance``: a scorer reads cosine from an
    arm that has none, and a test may set one on an instance.
    '''

    @staticmethod
    def _one_hot(axes: Sequence[int]) -> torch.Tensor:
        points = torch.zeros(len(axes), len(ITA_CANDIDATES), dtype=torch.float32)
        points[torch.arange(len(axes)), torch.tensor(axes, dtype=torch.long)] = 1.0
        return points

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._one_hot([ITA_CANDIDATES.index(ASSIGNED[text]) for text in texts])

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        return self._one_hot([ITA_CANDIDATES.index(code) for code in codes])

class IdentifiedStubEncoder(AssignedStubEncoder):
    '''
    The stub arm with the two identities a d16 read logs: a checkpoint's SHA-256 and a table's
    matrix fingerprint, as ``ArmEncoder`` carries them (plan 20).
    '''

    checkpoint_sha256 = 'c' * 64
    table_fingerprint = 't' * 64

# -------------------------------------------------------------------------------------------------
# The outcome panel beside an ITA read
# -------------------------------------------------------------------------------------------------

# Index entries over ITA_CANDIDATES: every validation text is a key of ASSIGNED, so the stub
# decodes it; 'Oil and gas wells' decodes to 111110, not its 211120. 236220 and 321113 hold no
# entry, and 621210's one query is in the sealed test split
OUTCOME_BESIDE_ROWS = (
    (0, '311111', 'Dog food manufacturing', 'examples'),
    (1, '311111', 'We make dog food', 'validation'),
    (2, '111110', 'Soybean farming', 'examples'),
    (3, '111110', 'Soybean farm', 'validation'),
    (4, '111120', 'Oilseed farming', 'examples'),
    (5, '111120', 'Canola growers', 'validation'),
    (6, '513210', 'Software publishers', 'examples'),
    (7, '513210', 'Software publisher', 'validation'),
    (8, '211120', 'Crude petroleum extraction', 'examples'),
    (9, '211120', 'Oil and gas wells', 'validation'),
    (10, '621210', 'Offices of dentists', 'examples'),
    (11, '621210', 'Dental office', 'test'),
)

def outcome_beside_panel(log: SelectionLog) -> OutcomePanel:
    '''
    An outcome panel over ``ITA_CANDIDATES`` whose validation queries the stub arm decodes.

    Args:
        log: The selection log its reads go to, the ITA panel's in a read beside one.

    Returns:
        The panel, on ``OUTCOME_BESIDE_ROWS``.
    '''

    rows = pl.DataFrame(
        list(OUTCOME_BESIDE_ROWS),
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )
    return OutcomePanel(rows, ITA_CANDIDATES, log)
