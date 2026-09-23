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

from pydantic import BaseModel, ConfigDict, Field, field_validator

# -------------------------------------------------------------------------------------------------
# Contract and schema versions
# -------------------------------------------------------------------------------------------------

CONTRACT_VERSION = 'stage3-supervision-v1'
STRUCTURAL_PREFERENCE_LOSS_VERSION = 'structural-preference-v1'
MINING_CONTRACT_VERSION = 'negative-selection-v1'

CODEBOOK_SCHEMA_VERSION = 'codebook-v1'
PAIR_FACTS_SCHEMA_VERSION = 'pair-facts-v1'
DISTANCES_SCHEMA_VERSION = 'distances-v1'
DISTANCE_MATRIX_SCHEMA_VERSION = 'distance-matrix-v1'
RELATIONS_SCHEMA_VERSION = 'relations-v1'
RELATION_MATRIX_SCHEMA_VERSION = 'relation-matrix-v1'
TRAINING_PAIRS_SCHEMA_VERSION = 'training-pairs-v1'
DIFFICULTY_THRESHOLDS_SCHEMA_VERSION = 'difficulty-thresholds-v1'

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
    exclusion_count: int = Field(ge=0)
    files: Tuple[ArtifactFile, ...]

    @field_validator('path')
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('artifact path must be a relative bundle path')
        return value

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
