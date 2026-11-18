'''
The frozen list of the economic candidates (spec ``economic-candidates``, 4.1).

``conf/data/economic_candidates.yaml`` names the 18 candidates the economic decision reads, and
the decision requires exactly those. The list closes before Stage 3's first read, because a
candidate added after a result would make the choice unfair. It pins the inputs every candidate
shares (the codebook, the text-only table and the checkpoint contract), each trained candidate's
Stage 7/8 arm record and each deterministic candidate's built table.

A candidate is one of four kinds:

- ``text-only``, ``tree`` and ``smoothed`` are deterministic: nothing trains, and each is one
  built table.
- ``trained`` is an imported Stage 7/8 arm: its seeds' tables come from that arm's runs.

The list names no absolute path. A record's path is relative to the artifact store's root and a
run directory is relative to a runs root. Both are command options.

This module is read by ``decision.provenance`` and the importers, so it imports neither of them.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from pathlib import Path, PurePosixPath
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

import yaml
from pydantic import Field, StrictInt, model_validator

from naics_embedder.decision.decide import MIN_SEEDS
from naics_embedder.decision.records import ArmSpec, Geometry, _Record

# The frozen list, relative to the repository root
ECONOMIC_CANDIDATES = 'conf/data/economic_candidates.yaml'

Kind = Literal['text-only', 'tree', 'smoothed', 'trained']

_SHA256 = r'^[0-9a-f]{64}$'
# Where a run directory takes its seed
_SEED_FIELD = '{seed}'

# -------------------------------------------------------------------------------------------------
# The list
# -------------------------------------------------------------------------------------------------

class CheckpointContract(_Record):
    '''The checkpoint contract a trained table's export provenance must name (spec 4.5).'''

    bundle_id: str
    contract_version: str

class CandidateEntry(_Record):
    '''
    One candidate of the frozen list.

    A deterministic entry (``text-only``, ``tree`` or ``smoothed``) names no record, arm, seeds or
    run directory. A trained entry pins no table, and its record is relative, with no ``..`` part.
    It names its arm, at least ``MIN_SEEDS`` distinct seeds and a run directory with a ``{seed}``
    field and no ``/``.

    Attributes:
        name: The candidate's economic name.
        kind: Which kind of candidate it is.
        dimension: The table's coordinate count.
        components: Stages or post-processing steps, as the tie order reads them.
        geometry: The geometry the table's coordinates are read in.
        table_sha256: A deterministic candidate's built table's sha256, null until it is built.
        record: A trained candidate's Stage 7/8 arm record, relative to the artifact store's root.
        record_sha256: That record's sha256.
        arm: The name that record gives its arm.
        seeds: The seeds the decision reads, which the record holds.
        run_directory: Each seed's run directory under the runs root, with ``{seed}`` where the
            seed goes.
    '''

    name: str = Field(min_length=1)
    kind: Kind
    dimension: int = Field(ge=1)
    components: int = Field(ge=1)
    geometry: Geometry
    table_sha256: Optional[str] = Field(default=None, pattern=_SHA256)
    record: Optional[str] = None
    record_sha256: Optional[str] = Field(default=None, pattern=_SHA256)
    arm: Optional[str] = None
    seeds: List[StrictInt] = Field(default_factory=list)
    run_directory: Optional[str] = None

    @model_validator(mode='after')
    def _fits_its_kind(self) -> 'CandidateEntry':
        if self.kind == 'trained':
            self._check_trained()
        else:
            self._check_deterministic()
        return self

    def _check_deterministic(self) -> None:
        named = [
            field for field, value in (
                ('record', self.record is not None),
                ('record_sha256', self.record_sha256 is not None),
                ('arm', self.arm is not None),
                ('seeds', bool(self.seeds)),
                ('run_directory', self.run_directory is not None),
            ) if value
        ]
        if named:
            raise ValueError(
                f'{self.name}: a {self.kind} candidate has no {", ".join(named)}: only a trained '
                f'candidate names a record, arm, seeds and run directory'
            )

    def _check_trained(self) -> None:
        if self.table_sha256 is not None:
            raise ValueError(f'{self.name}: a trained candidate pins no table: its record pins it')
        missing = [
            field for field, value in (
                ('record', self.record),
                ('record_sha256', self.record_sha256),
                ('arm', self.arm),
                ('run_directory', self.run_directory),
            ) if not value
        ]
        if missing:
            raise ValueError(f'{self.name}: a trained candidate needs its {", ".join(missing)}')
        record = PurePosixPath(self.record)
        if record.is_absolute() or '..' in record.parts:
            raise ValueError(
                f'{self.name}: record {self.record!r} must be relative to the store root, with no '
                f'.. part'
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f'{self.name}: its seeds repeat: {self.seeds}')
        if len(self.seeds) < MIN_SEEDS:
            raise ValueError(
                f'{self.name}: {len(self.seeds)} seeds, but Req 5 needs at least {MIN_SEEDS}'
            )
        if _SEED_FIELD not in self.run_directory or '/' in self.run_directory:
            raise ValueError(
                f'{self.name}: run_directory {self.run_directory!r} must hold {_SEED_FIELD} and '
                f'no /: it is one directory under the runs root'
            )

class CandidateList(_Record):
    '''
    The frozen candidate list (``conf/data/economic_candidates.yaml``).

    Attributes:
        codebook_sha256: The codebook every deterministic builder reads.
        text_only_sha256: The shared text-only table that every candidate reading a backbone is
            checked against.
        text_only_provenance_sha256: The provenance file beside the shared text-only table, which
            every candidate's text-only reference must store.
        checkpoint_contract: The contract a trained table's export provenance must name.
        candidates: The candidates, in the list's order. Their names are distinct, and each
            smoothed one has exactly one text-only candidate at its dimension.
    '''

    codebook_sha256: str
    text_only_sha256: str
    text_only_provenance_sha256: str = Field(pattern=_SHA256)
    checkpoint_contract: CheckpointContract
    candidates: List[CandidateEntry]

    @model_validator(mode='after')
    def _names_are_distinct_and_smoothing_has_its_twin(self) -> 'CandidateList':
        names = [entry.name for entry in self.candidates]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f'the candidate names repeat: {repeated}')
        for entry in self.candidates:
            if entry.kind == 'smoothed':
                self._twin_of(entry)
        return self

    def _twin_of(self, entry: CandidateEntry) -> CandidateEntry:
        twins = [
            other for other in self.candidates
            if other.kind == 'text-only' and other.dimension == entry.dimension
        ]
        if len(twins) != 1:
            raise ValueError(
                f'{entry.name}: a smoothed candidate needs exactly one text-only candidate at '
                f'dimension {entry.dimension}, and the list has {len(twins)}'
            )
        return twins[0]

    @property
    def names(self) -> List[str]:
        '''The candidates' names, in the list's order.'''

        return [entry.name for entry in self.candidates]

    def entry(self, name: str) -> CandidateEntry:
        '''
        The candidate with the name.

        Args:
            name: A candidate's economic name.

        Raises:
            ValueError: If the list has no such candidate.
        '''

        for entry in self.candidates:
            if entry.name == name:
                return entry
        raise ValueError(f'{name!r} is not on the frozen candidate list')

    def propagation_pairs(self) -> List[Tuple[str, str]]:
        '''
        Each smoothed candidate with its text-only twin, in the list's order (P6).

        The twin is the one text-only candidate at the smoothed candidate's dimension. The pairs
        come from the entries' kinds and dimensions, never from their names.
        '''

        return [
            (entry.name, self._twin_of(entry).name) for entry in self.candidates
            if entry.kind == 'smoothed'
        ]

# -------------------------------------------------------------------------------------------------
# Loading and paths
# -------------------------------------------------------------------------------------------------

class CandidateListYAMLError(ValueError, yaml.YAMLError):
    '''
    A candidate list file that is not safe YAML. It is a ``ValueError`` that names the file, so a
    command's handler refuses it, and still the ``yaml.YAMLError`` it wraps.
    '''

def load_candidate_list(path: Union[str, Path]) -> CandidateList:
    '''
    Read and validate a frozen candidate list.

    Args:
        path: The YAML file, normally ``ECONOMIC_CANDIDATES`` under the repository root.

    Raises:
        CandidateListYAMLError: If the file is not safe YAML: a ``ValueError`` and a
            ``yaml.YAMLError``, whose message names the file.
        pydantic.ValidationError: If the content breaks an entry or list rule.
    '''

    try:
        content = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise CandidateListYAMLError(f'{path} is not safe YAML: {exc}') from exc
    return CandidateList.model_validate(content)

def candidate_table_path(store_root: Union[str, Path], name: str) -> Path:
    '''
    Where a deterministic candidate's table lives (spec 4.2).

    Args:
        store_root: The artifact store's root.
        name: The candidate's economic name.

    Returns:
        ``<store_root>/candidates/economic/<name>.parquet``.
    '''

    return Path(store_root) / 'candidates' / 'economic' / f'{name}.parquet'

# -------------------------------------------------------------------------------------------------
# Specs
# -------------------------------------------------------------------------------------------------

def economic_spec(
    entry: CandidateEntry,
    *,
    text_only: Optional[Mapping[str, Any]] = None,
    source: Optional[ArmSpec] = None,
) -> ArmSpec:
    '''
    The ``ArmSpec`` an economic candidate's record carries.

    A tree candidate reads no backbone, so its five D9 fields are null (P7) and its settings are
    empty. A text-only or smoothed candidate reads the shared text-only table's backbone. A
    trained candidate is its source arm's spec under its economic name.

    Args:
        entry: The candidate's list entry.
        text_only: For a text-only or smoothed candidate, the fields
            ``decision.store.provenance_fields`` returns for the text-only table's provenance.
            ``revision`` becomes the spec's ``backbone_revision``.
        source: For a trained candidate, the Stage 7/8 arm record's spec.

    Raises:
        ValueError: If a text-only or smoothed candidate has no ``text_only``, if a trained one
            has no ``source`` or its source's dimension, geometry or components differ from the
            entry's.
    '''

    shape = {
        'name': entry.name,
        'components': entry.components,
        'dimension': entry.dimension,
        'geometry': entry.geometry,
    }
    if entry.kind == 'tree':
        return ArmSpec(
            **shape,
            backbone=None,
            backbone_revision=None,
            descriptions_sha256=None,
            summaries_sha256=None,
            max_length=None,
        )
    if entry.kind == 'trained':
        return _trained_spec(entry, source)
    if text_only is None:
        raise ValueError(
            f'{entry.name}: a {entry.kind} candidate reads the text-only table, so its spec needs '
            f'the text-only fields'
        )
    return ArmSpec(
        **shape,
        backbone=text_only['backbone'],
        backbone_revision=text_only['revision'],
        descriptions_sha256=text_only['descriptions_sha256'],
        summaries_sha256=text_only['summaries_sha256'],
        max_length=text_only['max_length'],
    )

def _trained_spec(entry: CandidateEntry, source: Optional[ArmSpec]) -> ArmSpec:
    if source is None:
        raise ValueError(f'{entry.name}: a trained candidate takes its spec from its source')
    differing = [
        f'{field} {getattr(source, field)!r} (the list says {getattr(entry, field)!r})'
        for field in ('dimension', 'geometry', 'components')
        if getattr(source, field) != getattr(entry, field)
    ]
    if differing:
        raise ValueError(
            f'{entry.name}: its source spec {source.name!r} has {", ".join(differing)}'
        )
    return source.model_copy(update={'name': entry.name})
