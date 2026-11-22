'''
The relatedness panel's settings and its salted split of the six-digit codes (spec
``relatedness-artifact``, 4.3).

The unit is the anchor industry: a code's whole row of shares falls in one part. A code's part comes
from the SHA-256 of the salt, a null byte and the six-digit code, in UTF-8. The first 8 bytes of the
digest, read big-endian as an unsigned integer modulo 2, give the part: 0 is validation and 1 is
sealed. This is the ITA rule (``panels/ita_split.py``), applied to a code instead of a description
key: the module imports the rule's constants and shares no class with ITA. No assignment table is
stored, so a rebuilt table keeps each industry in its part.

``conf/data/relatedness_panel.yaml`` holds the panel name, the rule version, the salt, the relative
margin r and the NDCG cut k. The salt is chosen once and never changes. The split fingerprint is the
SHA-256 of the canonical JSON of the rule: it names the split, not the rows, so every opening of the
sealed part can be keyed on it.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.panels.ita_split import BYTE_ORDER, HASH, MODULUS, PREFIX_BYTES

# The panel's name in the selection log and in records
RELATEDNESS_PANEL = 'relatedness_co_ownership'

# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

class RelatednessPart(str, Enum):
    '''A part of the six-digit codes: where a model is judged, or the one opened once at a decision.'''

    VALIDATION = 'validation'
    SEALED = 'sealed'

# The hash's value modulo MODULUS is a position in this tuple
PARTS = (RelatednessPart.VALIDATION, RelatednessPart.SEALED)

class RelatednessPanelSettings(BaseModel):
    '''
    ``conf/data/relatedness_panel.yaml``.

    Attributes:
        panel: The panel's name.
        rule_version: The version of the split rule.
        salt: The split's salt, 32 lowercase hexadecimal characters.
        relative_margin: r, the margin as a share of the reference's validation headline NDCG@k.
        k: The headline's NDCG cut.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    panel: Literal['relatedness_co_ownership']
    rule_version: Literal[1]
    salt: str = Field(pattern=r'^[0-9a-f]{32}$')
    relative_margin: float = Field(gt=0)
    k: Literal[10]

def load_relatedness_settings(path: Union[str, Path]) -> RelatednessPanelSettings:
    '''
    Read and validate the panel's settings file.

    Args:
        path: The YAML file, ``conf/data/relatedness_panel.yaml``.

    Returns:
        The settings.

    Raises:
        pydantic.ValidationError: When a key is missing, unknown or out of range.
    '''

    with open(path, encoding='utf-8') as handle:
        return RelatednessPanelSettings.model_validate(yaml.safe_load(handle))

# -------------------------------------------------------------------------------------------------
# The split
# -------------------------------------------------------------------------------------------------

def split_part(code: str, salt: str) -> RelatednessPart:
    '''
    The part a six-digit code falls in.

    Args:
        code: A six-digit NAICS code, as the node id ends in it.
        salt: The split's salt.

    Returns:
        The part, from the code and the salt alone.
    '''

    digest = hashlib.new(HASH, f'{salt}\x00{code}'.encode('utf-8')).digest()
    return PARTS[int.from_bytes(digest[:PREFIX_BYTES], BYTE_ORDER) % MODULUS]

def split_rule(settings: RelatednessPanelSettings) -> Dict[str, Any]:
    '''
    The rule that puts a code in a part, as the split fingerprint reads it.

    Args:
        settings: The panel's settings.

    Returns:
        The salt, the version, the hash, the digest prefix and its byte order, the modulus and the
        part names.
    '''

    return {
        'salt': settings.salt,
        'rule_version': settings.rule_version,
        'hash': HASH,
        'prefix_bytes': PREFIX_BYTES,
        'byteorder': BYTE_ORDER,
        'modulus': MODULUS,
        'parts': [part.value for part in PARTS],
    }

def split_fingerprint(settings: RelatednessPanelSettings) -> str:
    '''
    The SHA-256 of the rule's canonical JSON: it identifies the split, not the rows.

    Args:
        settings: The panel's settings.

    Returns:
        The hexadecimal digest.
    '''

    canonical = json.dumps(split_rule(settings), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
