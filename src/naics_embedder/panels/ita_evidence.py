'''
The ITA evidence pin and its check before a read (embedding-data's evaluation protocol, Section 1,
condition C1; plan 20).

``conf/data/ita_evidence.yaml`` copies the protocol's Evidence table: the run whose tables a read
uses, the protocol commit that names it, and the SHA-256 of the run's ``outputs.json`` (the
detached pin) and of its two pair tables. ``check_ita_run`` checks the files against the pin before
any read: both tables in one directory named for the run, ``outputs.json`` beside them, and every
SHA-256 equal to its pin. A replay of the run has equal values and other bytes, so it is refused.
Under C5 a new run is a new pin, and the owner amends the protocol before this file changes.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.supervision.artifacts import sha256_file

# The run's detached pin sits beside its tables under this name
OUTPUTS_FILE = 'outputs.json'

_COMMIT = r'^[0-9a-f]{40}$'
_SHA256 = r'^[0-9a-f]{64}$'
_RUN = r'^\S+$'

# -------------------------------------------------------------------------------------------------
# The pin
# -------------------------------------------------------------------------------------------------

class ItaEvidence(BaseModel):
    '''
    ``conf/data/ita_evidence.yaml``: the protocol's Section 1 Evidence table.

    Attributes:
        protocol_commit: The embedding-data commit whose ``specs/evaluation-protocol.md`` the pin
            copies, 40 lowercase hexadecimal characters.
        run: The run directory's name, the one a read uses.
        outputs_sha256: The SHA-256 of the run's ``outputs.json``.
        pairs_sha256: The SHA-256 of ``ita_text_pairs.parquet``.
        pair_codes_sha256: The SHA-256 of ``ita_text_pair_codes.parquet``.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True)

    protocol_commit: str = Field(pattern=_COMMIT)
    run: str = Field(pattern=_RUN)
    outputs_sha256: str = Field(pattern=_SHA256)
    pairs_sha256: str = Field(pattern=_SHA256)
    pair_codes_sha256: str = Field(pattern=_SHA256)

def load_ita_evidence(path: Union[str, Path]) -> ItaEvidence:
    '''
    Read and validate the evidence pin.

    Args:
        path: The YAML file, ``conf/data/ita_evidence.yaml``.

    Returns:
        The pin.

    Raises:
        pydantic.ValidationError: When a key is missing or unknown, or a value is out of form.
    '''

    with open(path, encoding='utf-8') as handle:
        return ItaEvidence.model_validate(yaml.safe_load(handle))

# -------------------------------------------------------------------------------------------------
# The check (C1)
# -------------------------------------------------------------------------------------------------

def _check_hash(path: Path, pin: str) -> None:
    actual = sha256_file(path)
    if actual != pin:
        raise ValueError(
            f'{path.name} at {path} has SHA-256 {actual}, not the pinned {pin} (C1): it is not '
            "the pinned run's file"
        )

def check_ita_run(
    pairs_path: Union[str, Path],
    pair_codes_path: Union[str, Path],
    evidence: ItaEvidence,
) -> None:
    '''
    Check the two tables against the pin before a read (C1).

    Args:
        pairs_path: The run's ``ita_text_pairs.parquet``.
        pair_codes_path: The run's ``ita_text_pair_codes.parquet``.
        evidence: The pin.

    Raises:
        FileNotFoundError: When a table is missing.
        ValueError: When the tables are not in one directory, that directory is not named for the
            pinned run, its ``outputs.json`` is missing, or the SHA-256 of that file or of either
            table is not its pin.
    '''

    pairs = Path(pairs_path).absolute()
    codes = Path(pair_codes_path).absolute()
    for table in (pairs, codes):
        if not table.is_file():
            raise FileNotFoundError(f'the ITA table {table.name} is missing: {table}')
    if pairs.parent != codes.parent:
        raise ValueError(
            f'the ITA tables are not in one directory: {pairs.parent} and {codes.parent} (C1)'
        )
    directory = pairs.parent
    if directory.name != evidence.run:
        raise ValueError(
            f'the ITA tables are in {directory}, whose name is not the pinned run '
            f"{evidence.run} (C1, C5): a new run waits for the protocol's amendment"
        )
    outputs = directory / OUTPUTS_FILE
    if not outputs.is_file():
        raise ValueError(f'the run has no {OUTPUTS_FILE} beside its tables: {outputs} (C1)')
    _check_hash(outputs, evidence.outputs_sha256)
    _check_hash(pairs, evidence.pairs_sha256)
    _check_hash(codes, evidence.pair_codes_sha256)
