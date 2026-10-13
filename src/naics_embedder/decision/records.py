'''
The records of Req 5's decisions (Verification "Decision records"; roadmap Stage 4).

Three kinds, each one JSON file:

- **Arm record.** One configuration's seed sweep: its spec, the text-only table it was paired
  with and that table's provenance (D9), the panels it read, and per seed the encoder checkpoint,
  the 2,125-code table, the stored scores and the selection-log records of the run's reads (the
  log itself is gitignored and dies with its worktree or Lambda instance). A trained seed also
  carries the monitor reads that selected its checkpoint (spec 4.4).
- **Margin record.** Each panel's δ, a stated multiple of a reference arm's across-seed standard
  deviation, fixed before any other arm's first read.
- **Decision record.** Its arms and margins, every paired comparison with its 95 %
  non-inferiority and 98⅓ % superiority intervals (D8), the non-dominated set, the tie order,
  the chosen arm, and the reported statistics.

The economic decision (Req 24) has kinds of its own:

- **Relative margin record.** Each growth panel's δ, r times the reference comparator's
  validation MSE, from a margin read that reads no candidate.
- **Economic arm record.** A deterministic candidate's single replicate: its table and that
  table's provenance (Req 23), read once on each growth panel, with no checkpoint or monitor
  reads.
- **Economic decision record.** Its candidates and margins, each candidate's comparison with the
  reference ``covariates+ancestors`` with its 95 % non-inferiority and 97.5 % superiority
  intervals, the candidates adopted over the reference, the non-dominated set and tie order among
  them, and the chosen candidate, or none when the reference stands.

Artifact references are paths relative to an ``ArtifactStore`` root with their sha256, so a
record names exactly the bytes it was computed from.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt

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
    # The window-fitting summaries the table read; null for a backbone with no pin
    summaries_sha256: Optional[str]
    max_length: int

# -------------------------------------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------------------------------------

class ArmSpec(_Record):
    '''
    A configuration, with what the tie order and D9 read.

    Attributes:
        components: Stages or post-processing steps: Req 5's "fewer components".
        backbone, backbone_revision, descriptions_sha256, summaries_sha256, max_length: What
            the arm's encoder reads, which its text-only table and each seed's table must match
            (D9). ``summaries_sha256`` is null for a backbone with no pinned summaries.
        settings: The configuration's own settings, recorded as given.
    '''

    name: str = Field(min_length=1)
    components: int = Field(ge=1)
    dimension: int = Field(ge=1)
    geometry: Geometry
    backbone: str
    backbone_revision: Optional[str]
    descriptions_sha256: str
    summaries_sha256: Optional[str]
    max_length: int = Field(ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)

class PanelSet(_Record):
    '''
    The panels a run read; every arm of a decision must share them (Req 5's pairing).

    A split's fingerprint is the name the selection log gives it, and it does not pin the data a
    read scored, so each panel is named twice.

    Attributes:
        outcome: The outcome panel's role assignment (``OutcomePanel.fingerprint``).
        outcome_data: Its validation queries, text included, and its candidates
            (``OutcomePanel.data_fingerprint``).
        regressor: The regressor panel's held-out draw (``RegressorPanel.fingerprint``).
        regressor_data: Its rows at the decision level (``RegressorPanel.data_fingerprint``).
        fit_settings: The regressor panel's tuning and resampling settings.
    '''

    outcome: str
    outcome_data: str
    regressor: str
    regressor_data: str
    fit_settings: Dict[str, Any]

class SeedRun(_Record):
    '''
    One seed of an arm.

    A trained seed names its training run and its checkpoint's epoch, and carries the monitor
    reads that selected the checkpoint (spec 4.4). A seed from a runner that trains nothing has
    none of the three.

    Attributes:
        scores: The seed's scores on all three panels (``decision.scores.SCORE_COLUMNS``).
        decoding: ``DecodingResult.per_query`` of the outcome read.
        predictions: Both regimes' level-6 validation predictions.
        statistics: Each panel's decision statistic on every unit.
        log_records: The selection-log records of this run's reads.
        training_run: The id of the training run the checkpoint is from, which its checkpoints
            save and its monitor reads name.
        checkpoint_epoch: The checkpoint's epoch: the earliest with the highest monitor MRR.
        monitor_records: The training run's ``monitor_reads.jsonl`` records, in the file's order,
            each ``{'mrr': …, 'read': …}``: an epoch's MRR, and its read as the selection log
            appended it. The checks read each record's epoch, never its position.
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
    # Each defaults, so a record written before runs carried their monitor reads still loads. An
    # imported record's identities are strict: a training run of spaces names none, and true
    # would load as epoch 1
    training_run: Optional[str] = Field(default=None, min_length=1, pattern=r'\S')
    checkpoint_epoch: Optional[StrictInt] = Field(default=None, ge=0)
    monitor_records: List[Dict[str, Any]] = Field(default_factory=list)

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
        '''
        The panel's δ.

        Raises:
            ValueError: If the record holds no δ for the panel.
        '''

        for entry in self.margins:
            if entry.panel == panel:
                return entry.margin
        raise ValueError(f'the margins {self.name!r} hold no δ for {panel}')

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
# The economic decision's margins (Req 24)
# -------------------------------------------------------------------------------------------------

class ReadEnvironment(_Record):
    '''
    Where a growth read ran. Decisions pair reads only on bit-identical rows and fits, so every
    candidate read of an economic decision runs where its margin read ran (Req 24).

    Attributes:
        platform: ``platform.platform()``.
        python: ``platform.python_version()``.
        lock_sha256: The sha256 of the ``uv.lock`` the read ran under.
    '''

    platform: str
    python: str
    lock_sha256: str

class GrowthPanelSet(_Record):
    '''
    The growth panel an economic read scored (Req 21), which every read of a decision shares.

    Attributes:
        regressor: The held-out draw (``RegressorPanel.fingerprint``).
        regressor_data: Its rows at the decision level (``RegressorPanel.data_fingerprint``).
        fit_settings: The panel's tuning and resampling settings.
    '''

    regressor: str
    regressor_data: str
    fit_settings: Dict[str, Any]

class ReferenceRead(_Record):
    '''
    The margin read (Req 24): each growth panel's validation split at the decision level, scored
    by the comparators that read no candidate.

    Attributes:
        run_id: Names the read in its selection-log records.
        panels: The growth panel it read.
        predictions: Both regimes' level-6 validation predictions.
        scores: Their per-row scores (``decision.scores.SCORE_COLUMNS``).
        log_records: The selection-log records of the read, one per growth panel.
        environment: Where the read ran.
    '''

    run_id: str = Field(min_length=1)
    panels: GrowthPanelSet
    predictions: ArtifactRef
    scores: ArtifactRef
    log_records: List[Dict[str, Any]]
    environment: ReadEnvironment

class RelativeMargin(_Record):
    '''One growth panel's δ: r times the reference comparator's validation MSE (Req 24).'''

    panel: str
    statistic: str
    reference_mse: float
    margin: float

class RelativeMarginRecord(_Record):
    '''
    Each growth panel's δ, fixed from the margin read before any candidate is read (Req 24).

    Attributes:
        r: Each δ as a multiple of the reference's MSE (U13).
        reference: The reference comparator, which reads no candidate (U9).
        read: The margin read the MSEs come from.
    '''

    kind: Literal['relative_margins'] = 'relative_margins'
    name: str
    r: float = Field(gt=0)
    reference: str
    read: ReferenceRead
    margins: List[RelativeMargin]
    fixed_at: datetime

    def margin(self, panel: str) -> float:
        '''
        The panel's δ.

        Raises:
            ValueError: If the record holds no δ for the panel.
        '''

        for entry in self.margins:
            if entry.panel == panel:
                return entry.margin
        raise ValueError(f'the margins {self.name!r} hold no δ for {panel}')

# -------------------------------------------------------------------------------------------------
# The economic decision's arms (Req 23; Req 24)
# -------------------------------------------------------------------------------------------------

class TableProvenance(_Record):
    '''
    A candidate table's provenance, as its builder wrote it beside the table (Req 23).

    Attributes:
        file: The stored provenance file.
        content: Its JSON object, verbatim: what the table was built from.
    '''

    file: ArtifactRef
    content: Dict[str, Any]

class EconomicRun(_Record):
    '''
    One replicate of an economic candidate: its table, read once on each growth panel.

    A deterministic candidate is a single replicate (Req 24), so a run has no seed, and no field
    for a checkpoint or monitor reads: an imported run that names one does not load.

    Attributes:
        seed: None: a deterministic candidate has no seed.
        run_id: Names the run in its selection-log records.
        table: The candidate's 2,125-code table, in the export form.
        provenance: The table's provenance.
        scores: Both growth panels' per-row scores (``decision.scores.SCORE_COLUMNS``).
        predictions: Both growth panels' level-6 validation predictions.
        statistics: Each growth panel's decision statistic.
        log_records: The selection-log records of the run's reads.
    '''

    seed: Optional[StrictInt] = None
    run_id: str = Field(min_length=1, pattern=r'\S')
    table: TableRef
    provenance: TableProvenance
    scores: ArtifactRef
    predictions: ArtifactRef
    statistics: Dict[str, float]
    log_records: List[Dict[str, Any]]

class EconomicArmRecord(_Record):
    '''
    An economic candidate's reads of the growth panels (``decision.economic``).

    Attributes:
        spec: The candidate, with what the tie order and D9 read.
        text_only: The text-only table its reads scored as ``covariates+text_only`` (D9).
        store: The artifact store's root.
        panels: The growth panel it read.
        environment: Where it was read.
        runs: Its replicates: one, for a deterministic candidate (Req 24).
    '''

    kind: Literal['economic_arm'] = 'economic_arm'
    spec: ArmSpec
    text_only: TextOnlyRef
    store: str
    panels: GrowthPanelSet
    environment: ReadEnvironment
    runs: List[EconomicRun]
    created_at: datetime

# -------------------------------------------------------------------------------------------------
# The economic decision (Req 24)
# -------------------------------------------------------------------------------------------------

class EconomicDecisionSettings(_Record):
    '''How the economic decision resampled and which intervals it read.'''

    replicates: int
    bootstrap_seed: int
    noninferiority_level: float
    superiority_level: float

class EconomicReport(_Record):
    '''
    What the economic decision reports for a candidate besides the rule's statistics.

    Attributes:
        statistics: Each growth panel's decision statistic, ``covariates+embedding``'s MSE.
        comparator_mse: Each growth panel's MSE for every comparator its read scored, one-hot
            included where the seen regime scores it (Req 24).
    '''

    arm: str
    statistics: Dict[str, float]
    comparator_mse: Dict[str, Dict[str, float]]

class EconomicDecisionRecord(_Record):
    '''
    The economic decision (Req 24): each candidate against the reference on both growth panels,
    and Req 5's rule among the candidates adopted over it.

    Attributes:
        reference: The reference comparator (U9).
        comparisons: Each candidate against the reference, then each eligible candidate against
            each other.
        eligible: The candidates adopted over the reference.
        non_dominated: The eligible candidates no eligible candidate is adopted over.
        cycle: Whether dominance among the eligible cycled.
        tie_order: The non-dominated candidates, from the one that stands first.
        heldout_gain: Each candidate's held-out gain over the reference: the reference's
            ``growth_heldout`` MSE minus the candidate's (D11).
        chosen: The candidate whose table is the economic deliverable; None when no candidate is
            eligible, and the reference stands.
    '''

    kind: Literal['economic_decision'] = 'economic_decision'
    name: str
    question: str
    created_at: datetime
    statistics: Dict[str, str]
    settings: EconomicDecisionSettings
    reference: str
    margins: RelativeMarginRecord
    arms: List[EconomicArmRecord]
    comparisons: List[Comparison]
    eligible: List[str]
    non_dominated: List[str]
    cycle: bool
    tie_order: List[str]
    heldout_gain: Dict[str, float]
    chosen: Optional[str]
    reports: List[EconomicReport]

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def write_record(record: BaseModel, path: Union[str, Path]) -> Path:
    '''
    Write a record as indented JSON, once.

    The file is created exclusively, so a writer racing for the same path fails instead of
    overwriting it, and a write that fails removes its file, which would otherwise block the path.
    A process killed mid-write can still leave a partial file.

    Raises:
        FileExistsError: If the path exists.
    '''

    path = Path(path)
    text = record.model_dump_json(indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise FileExistsError(f'{path} exists; a record is written once') from None
    try:
        with handle:
            handle.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path

def read_record(path: Union[str, Path], kind: Type[RecordType]) -> RecordType:
    '''Read and validate a record of the given kind.'''

    return kind.model_validate_json(Path(path).read_text(encoding='utf-8'))
