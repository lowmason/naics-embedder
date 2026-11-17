'''
The ITA panel's settings and its salted split of ``description_key`` (spec
``text-to-code-ita-panel``, 4.2).

A pair's part comes from its description key alone: the SHA-256 of the salt, a null byte and the
key, in UTF-8. The first 8 bytes of the digest, read big-endian as an unsigned integer modulo 2,
give the part: 0 is validation and 1 is sealed. No assignment table is stored, so a rebuilt pair
table puts an old key in its old part and a new key in a part of its own.

``conf/data/ita_panel.yaml`` holds the salt, the rule version, the panel name and the relative
margin r. The salt is chosen once and never changes. The split fingerprint is the SHA-256 of the
canonical JSON of the rule: it names the split, not the rows, so every opening of the sealed part
can be keyed on it.
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

# The panel's name in the selection log and in records
ITA_PANEL = 'ita_text'

HASH = 'sha256'
PREFIX_BYTES = 8
BYTE_ORDER = 'big'
MODULUS = 2

# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

class ItaPart(str, Enum):
    '''A part of the pairs: where a model is judged, or the one opened once at a decision.'''

    VALIDATION = 'validation'
    SEALED = 'sealed'

# The hash's value modulo MODULUS is a position in this tuple
PARTS = (ItaPart.VALIDATION, ItaPart.SEALED)

class ItaPanelSettings(BaseModel):
    '''
    ``conf/data/ita_panel.yaml``.

    Args:
        panel: The panel's name.
        rule_version: The version of the split rule.
        salt: The split's salt, 32 lowercase hexadecimal characters.
        relative_margin: r, the margin as a share of the reference's validation headline MRR.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    panel: Literal['ita_text']
    rule_version: Literal[1]
    salt: str = Field(pattern=r'^[0-9a-f]{32}$')
    relative_margin: float = Field(gt=0)

def load_ita_settings(path: Union[str, Path]) -> ItaPanelSettings:
    '''
    Read and validate the panel's settings file.

    Args:
        path: The YAML file, ``conf/data/ita_panel.yaml``.

    Returns:
        The settings.

    Raises:
        pydantic.ValidationError: When a key is missing, unknown or out of range.
    '''

    with open(path, encoding='utf-8') as handle:
        return ItaPanelSettings.model_validate(yaml.safe_load(handle))

# -------------------------------------------------------------------------------------------------
# The split
# -------------------------------------------------------------------------------------------------

def split_part(description_key: str, salt: str) -> ItaPart:
    '''
    The part a description key falls in.

    Args:
        description_key: A pair's normalized description.
        salt: The split's salt.

    Returns:
        The part, from the key and the salt alone.
    '''

    digest = hashlib.new(HASH, f'{salt}\x00{description_key}'.encode('utf-8')).digest()
    return PARTS[int.from_bytes(digest[:PREFIX_BYTES], BYTE_ORDER) % MODULUS]

def split_rule(settings: ItaPanelSettings) -> Dict[str, Any]:
    '''
    The rule that puts a key in a part, as the split fingerprint reads it.

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

def split_fingerprint(settings: ItaPanelSettings) -> str:
    '''
    The SHA-256 of the rule's canonical JSON: it identifies the split, not the rows.

    Args:
        settings: The panel's settings.

    Returns:
        The hexadecimal digest.
    '''

    canonical = json.dumps(split_rule(settings), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
