'''
Stage-3 supervision contract vocabulary and manifest schema.

Every rebuilt supervision artifact belongs to one immutable bundle described by a
``SupervisionManifest``. The enums here separate three independent axes of meaning: structural
facts (materialized hierarchy), semantic supervision (target and source), and the sampling role
a training pair gives each code.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -------------------------------------------------------------------------------------------------
# Contract and schema versions
# -------------------------------------------------------------------------------------------------

CONTRACT_VERSION = 'stage3-supervision-v2'
# The objective a checkpoint was trained under, which its contract records (spec 4.5): Req 11's
# three terms, the radial form and the bound. Every contract saved before Stage 7 names none, and
# reads as the legacy marker; nothing loads such a checkpoint, and nothing migrates it (D2)
OBJECTIVE = 'req11-v1'
LEGACY_OBJECTIVE = 'pre-req11'

# -------------------------------------------------------------------------------------------------
# Structural margin contract
#
# The margins the training-pair generator (``data.create_triplets``) writes. No training path reads
# them (D5): they stay in the training-pairs member until the next contract bump retires it.
# Distances are D* (Req 7): integers with no half-step and no cross-sector constant, so the
# distance axis has one special case left, the equal-distance tie.
# -------------------------------------------------------------------------------------------------

# The relation label cross-sector pairs carry: every reader finds them by it, not by a distance
CROSS_SECTOR_RELATION_ID = 99
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
# The relation axis's cross-sector margin, which only the bundle build still writes (D5)
CROSS_SECTOR_RELATION_MARGIN = 15.0
EQUAL_DISTANCE_MARGIN = 0.3333

CODEBOOK_SCHEMA_VERSION = 'codebook-v1'
# v2: the unary_pair flag (Req 9)
PAIR_FACTS_SCHEMA_VERSION = 'pair-facts-v2'
DISTANCES_SCHEMA_VERSION = 'distances-v1'
DISTANCE_MATRIX_SCHEMA_VERSION = 'distance-matrix-v1'
RELATIONS_SCHEMA_VERSION = 'relations-v1'
RELATION_MATRIX_SCHEMA_VERSION = 'relation-matrix-v1'
TRAINING_PAIRS_SCHEMA_VERSION = 'training-pairs-v1'
DIFFICULTY_THRESHOLDS_SCHEMA_VERSION = 'difficulty-thresholds-v1'
INDEX_ROLES_SCHEMA_VERSION = 'index-roles-v1'
REDIRECTIONS_SCHEMA_VERSION = 'redirections-v1'

# -------------------------------------------------------------------------------------------------
# Supervision vocabulary
# -------------------------------------------------------------------------------------------------

class SemanticTarget(str, Enum):
    '''Supervision meaning of a candidate relative to its anchor (not tree geometry).'''

    RELATED = 'related'
    UNRELATED = 'unrelated'
    UNKNOWN = 'unknown'

class SemanticSource(str, Enum):
    '''Where a semantic target came from.'''

    TRAINING_POSITIVE = 'training_positive'
    EXPLICIT_EXCLUSION = 'explicit_exclusion'
    UNLABELED = 'unlabeled'

class SamplingRole(str, Enum):
    '''Slot a candidate occupies during sampling; independent of its semantic target.'''

    POSITIVE = 'positive'
    NEGATIVE = 'negative'

# -------------------------------------------------------------------------------------------------
# Index-entry roles (outcome panel)
# -------------------------------------------------------------------------------------------------

class IndexRole(str, Enum):
    '''The one role an index entry holds: examples-channel text, or a query in one split.'''

    EXAMPLES = 'examples'
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

# -------------------------------------------------------------------------------------------------
# Manifest models
# -------------------------------------------------------------------------------------------------

class ArtifactFile(BaseModel):
    '''One physical file belonging to a logical bundle artifact.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    row_count: int = Field(ge=0)

    @field_validator('path')
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('artifact file path must be a relative bundle path')
        return value

class ArtifactRecord(BaseModel):
    '''A logical bundle artifact (a Parquet file, a partitioned dataset, or a JSON file).'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str
    schema_version: str
    row_count: int = Field(ge=0)
    exclusion_count: int = Field(
        ge=0,
        description=(
            'Informational: rows carrying an explicit exclusion at generation time. Not covered '
            'by the member hashes and not validated at load; nothing depends on it.'
        ),
    )
    files: Tuple[ArtifactFile, ...]

    @field_validator('path')
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('artifact path must be a relative bundle path')
        return value

class ChannelOverflow(BaseModel):
    '''One text channel's present texts and those longer than the input window.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    present: int = Field(ge=0)
    over: int = Field(ge=0)
    share: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_counts(self) -> 'ChannelOverflow':
        if self.over > self.present:
            raise ValueError('more texts exceed the window than are present')
        return self

class InputWindowRecord(BaseModel):
    '''
    The backbone's trained input window, and each text channel's texts beyond it (Req 9).

    The window comes from the backbone's own documentation (``utils/input_window.py``). The
    counts are the bundle's record of the texts beyond it, which readers since Stage 6b read as
    their window-fitting summaries (``panels/window_summaries.py``), never truncated. They are
    measured on text without its field marker, so they are no higher than the marked counts the
    readers summarize.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')

    backbone: str = Field(min_length=1)
    window: int = Field(gt=0)
    channels: Dict[str, ChannelOverflow]

class SupervisionManifest(BaseModel):
    '''Top-level, write-last description of one immutable supervision bundle.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    contract_version: str
    bundle_id: str = Field(min_length=1)
    generated_at: datetime
    generator_revision: str = Field(min_length=1)
    naics_vintage: int
    codebook_order: Tuple[str, ...]
    codebook_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    description_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    exclusion_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    generation_parameters: Mapping[str, Any]
    structural_relation_ids: Mapping[str, int]
    artifacts: Dict[str, ArtifactRecord]
    validation_results: Mapping[str, bool]
    input_window: InputWindowRecord
