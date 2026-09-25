'''
The records of Req 5's decisions (Verification "Decision records"; roadmap Stage 4).

Three kinds, each one JSON file:

- **Arm record.** One configuration's seed sweep: its spec, the text-only table it was paired
  with and that table's provenance (D9), the panels it read, and per seed the encoder checkpoint,
  the 2,125-code table, the stored scores and the selection-log records of the run's reads (the
  log itself is gitignored and dies with its worktree or Lambda instance).
- **Margin record.** Each panel's δ, a stated multiple of a reference arm's across-seed standard
  deviation, fixed before any other arm's first read.
- **Decision record.** Its arms and margins, every paired comparison with its 95 %
  non-inferiority and 98⅓ % superiority intervals (D8), the non-dominated set, the tie order,
  the chosen arm, and the reported statistics.

Artifact references are paths relative to an ``ArtifactStore`` root with their sha256, so a
record names exactly the bytes it was computed from.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

Geometry = Literal['euclidean', 'spherical', 'hyperbolic']
RecordType = TypeVar('RecordType', bound=BaseModel)

class _Record(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

# -------------------------------------------------------------------------------------------------
# Artifacts
# -------------------------------------------------------------------------------------------------

class ArtifactRef(_Record):
    '''A stored file: its path relative to the store root, its sha256 and its size.'''

    path: str
    sha256: str
    bytes: int = Field(ge=0)

class TableRef(ArtifactRef):
    '''A stored code table, with the ``matrix_fingerprint`` the selection log names it by.'''

    matrix_fingerprint: str

class TextOnlyRef(_Record):
    '''An arm's text-only table and its provenance, with the provenance fields D9 checks.'''

    table: TableRef
    provenance: ArtifactRef
    backbone: str
    revision: Optional[str]
    descriptions_sha256: str
    max_length: int

# -------------------------------------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------------------------------------

class ArmSpec(_Record):
    '''
    A configuration, with what the tie order and D9 read.

    Attributes:
        components: Stages or post-processing steps: Req 5's "fewer components".
        backbone, backbone_revision, descriptions_sha256, max_length: What the arm's encoder
            reads, which its text-only table must match (D9).
        settings: The configuration's own settings, recorded as given.
    '''

    name: str = Field(min_length=1)
    components: int = Field(ge=1)
    dimension: int = Field(ge=1)
    geometry: Geometry
    backbone: str
    backbone_revision: Optional[str]
    descriptions_sha256: str
    max_length: int = Field(ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)

class PanelSet(_Record):
    '''The panels a run read; every arm of a decision must share them (Req 5's pairing).'''

    outcome: str
    regressor: str
    fit_settings: Dict[str, Any]

class SeedRun(_Record):
    '''
    One seed of an arm.

    Attributes:
        scores: The seed's scores on all three panels (``decision.scores.SCORE_COLUMNS``).
        decoding: ``DecodingResult.per_query`` of the outcome read.
        predictions: Both regimes' level-6 validation predictions.
        statistics: Each panel's decision statistic on every unit.
        log_records: The selection-log records of this run's reads.
    '''

    seed: int
    run_id: str
    checkpoint: ArtifactRef
    table: TableRef
    scores: ArtifactRef
    decoding: ArtifactRef
    predictions: ArtifactRef
    statistics: Dict[str, float]
    log_records: List[Dict[str, Any]]

class ArmRecord(_Record):
    '''One configuration's seed sweep (``decision.sweep.run_seed_sweep``).'''

    kind: Literal['arm'] = 'arm'
    spec: ArmSpec
    text_only: TextOnlyRef
    store: str
    panels: PanelSet
    runs: List[SeedRun]
    created_at: datetime

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

class PanelMargin(_Record):
    '''One panel's δ: the multiple times the reference's across-seed standard deviation.'''

    panel: str
    statistic: str
    per_seed: List[float]
    sd: float
    margin: float

class MarginRecord(_Record):
    '''Each panel's δ, fixed from a reference arm before any other arm's first read (Req 5).'''

    kind: Literal['margins'] = 'margins'
    name: str
    multiple: float = Field(gt=0)
    reference: ArmRecord
    margins: List[PanelMargin]
    fixed_at: datetime

    def margin(self, panel: str) -> float:
        '''The panel's δ.'''

        return next(entry.margin for entry in self.margins if entry.panel == panel)

# -------------------------------------------------------------------------------------------------
# Decisions
# -------------------------------------------------------------------------------------------------

class PanelComparison(_Record):
    '''Δ on one panel, oriented so that a positive value favours A, with both intervals.'''

    panel: str
    delta: float
    noninferiority_interval: Tuple[float, float]
    superiority_interval: Tuple[float, float]
    margin: float
    non_inferior: bool
    superior: bool

class Comparison(_Record):
    '''A against B: adopted when non-inferior on every panel and superior on at least one.'''

    a: str
    b: str
    panels: List[PanelComparison]
    adopted: bool

class Estimate(_Record):
    '''A point estimate with its 95 % percentile interval.'''

    point: float
    interval: Tuple[float, float]

class ArmReport(_Record):
    '''
    What a decision reports for an arm besides the rule's statistics (D10).

    Attributes:
        statistics: Each panel's decision statistic.
        outcome_metrics: Every Req 3 metric on the outcome panel.
        comparator_mse: Each regime's mean squared error for every Req 2 comparator.
        gain: Each regime's gain over its sparse encoding (Req 1): the sparse comparator's mean
            squared error minus ``covariates+embedding``'s, on the decision's draws.
        heldout_by_feature_year: The held-out regime by feature year: ``covariates+embedding``,
            ``covariates+ancestors`` and the gain.
    '''

    arm: str
    statistics: Dict[str, float]
    outcome_metrics: Dict[str, float]
    comparator_mse: Dict[str, Dict[str, float]]
    gain: Dict[str, Estimate]
    heldout_by_feature_year: Dict[str, Dict[str, float]]

class DecisionSettings(_Record):
    '''How the decision resampled and which intervals it read.'''

    replicates: int
    bootstrap_seed: int
    min_seeds: int
    noninferiority_level: float
    superiority_level: float

class DecisionRecord(_Record):
    '''A Req 5 decision over two or more arms.'''

    kind: Literal['decision'] = 'decision'
    name: str
    question: str
    created_at: datetime
    statistics: Dict[str, str]
    settings: DecisionSettings
    margins: MarginRecord
    arms: List[ArmRecord]
    comparisons: List[Comparison]
    non_dominated: List[str]
    cycle: bool
    tie_order: List[str]
    heldout_gain: Dict[str, float]
    chosen: str
    reports: List[ArmReport]

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def write_record(record: BaseModel, path: Union[str, Path]) -> Path:
    '''Write a record as indented JSON; refuse to overwrite a file.'''

    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; a record is written once')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path

def read_record(path: Union[str, Path], kind: Type[RecordType]) -> RecordType:
    '''Read and validate a record of the given kind.'''

    return kind.model_validate_json(Path(path).read_text(encoding='utf-8'))
