'''
Stage-3 supervision contract vocabulary and manifest schema.

Every rebuilt supervision artifact belongs to one immutable bundle described by a
``SupervisionManifest``. The enums here separate three independent axes of meaning: structural
facts (materialized hierarchy), semantic supervision (target and source), and sampling
role/provenance.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -------------------------------------------------------------------------------------------------
# Contract and schema versions
# -------------------------------------------------------------------------------------------------

CONTRACT_VERSION = 'stage3-supervision-v2'
STRUCTURAL_PREFERENCE_LOSS_VERSION = 'structural-preference-v1'
# v2: no slot is reserved for an explicit exclusion, which is never a negative (Req 8)
MINING_CONTRACT_VERSION = 'negative-selection-v2'

# -------------------------------------------------------------------------------------------------
# Structural margin contract
#
# Shared by the training-pair generator (``data.create_triplets``) and the runtime eligibility rule
# (``supervision.margins``), so both apply the identical margins. Distances are D* (Req 7):
# integers with no half-step and no cross-sector constant, so the distance axis has one special
# case left, the equal-distance tie.
# -------------------------------------------------------------------------------------------------

# The relation label cross-sector pairs carry: every reader finds them by it, not by a distance
CROSS_SECTOR_RELATION_ID = 99
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
# The relation axis keeps its cross-sector margin until Stage 7 retires the axis (D5)
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

SAMPLING_ROLE_TO_ID = {
    SamplingRole.POSITIVE: 1,
    SamplingRole.NEGATIVE: 2,
}

class SamplingProvenance(IntEnum):
    '''How a runtime candidate entered the candidate pool.'''

    GENERATED = 1
    DIFFICULTY = 2
    LOCAL_POOL = 3
    DISTRIBUTED_POOL = 4
    BACKFILL = 5

class SelectionReason(IntEnum):
    '''Why a candidate occupies a final negative slot.'''

    EXCLUSION_QUOTA = 1
    GEOMETRIC = 2
    ROUTER = 3
    DIFFICULTY = 4
    BACKFILL = 5

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
    measured on text without its field marker, so they are lower than the marked counts the
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
