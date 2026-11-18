'''
A trained economic candidate's import from its Stage 7/8 arm record and run directories (spec
economic-candidates, 4.4 and 4.6).

A trained candidate is an arm that Stage 7 or Stage 8 trained and read. The frozen list pins its
arm record in the artifact store by path and sha256, and each seed's export provenance is beside
the seed's table in its run directory under a runs root. The import checks them, reads no panel
and writes nothing, in this order:

1. **The source record.** The stored file the list pins loads as an ``ArmRecord`` of the list's
   arm.
2. **The source's checks.** It passes ``check_arm`` at Req 5's floor, so each seed's checkpoint is
   the earliest highest-MRR epoch of its monitor reads (spec 4.4). Its seeds are the row's, every
   run names its training run, and its text-only table is the list's.
3. **The spec.** It is the source's, under the candidate's economic name (``economic_spec``).
4. **Each seed, in seed order.** Its export provenance exists, describes the seed's stored table,
   passes the text-and-tree rule (Req 29), names the seed's checkpoint and records what the arm
   reads (D9: ``check_seed_table``). The rule reads the shape of the checkpoint, descriptions and
   summaries entries, and these last two checks read their identity.

This module reads ``decision.candidates`` and ``decision.provenance``, and neither reads it.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

from naics_embedder.decision.candidates import CandidateEntry, CandidateList, economic_spec
from naics_embedder.decision.decide import MIN_SEEDS, check_arm, check_seed_table
from naics_embedder.decision.provenance import check_candidate_provenance
from naics_embedder.decision.records import ArmRecord, ArmSpec, ArtifactRef, SeedRun, read_record
from naics_embedder.decision.store import ArtifactStore, check_provenance, provenance_fields
from naics_embedder.panels.text_only import provenance_path

# -------------------------------------------------------------------------------------------------
# The import
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainedImport:
    '''
    A trained candidate, imported and checked, to be read on the growth panels.

    Attributes:
        entry: Its frozen-list row.
        spec: Its spec: the source's, under its economic name.
        source: Its Stage 7/8 arm record.
        source_ref: That record in the store, as the list pins it.
        provenance: Each seed's export provenance file, by seed, in seed order.
    '''

    entry: CandidateEntry
    spec: ArmSpec
    source: ArmRecord
    source_ref: ArtifactRef
    provenance: Mapping[int, Path]

def import_trained_candidate(
    entry: CandidateEntry,
    candidates: CandidateList,
    store: ArtifactStore,
    runs_root: Union[str, Path],
) -> TrainedImport:
    '''
    Import a trained candidate and run every check that reads no panel (module docstring).

    Args:
        entry: The candidate's frozen-list row, of kind ``trained``.
        candidates: The frozen list, which pins the text-only table and the checkpoint contract.
        store: The artifact store that holds the source record and everything it references.
        runs_root: The directory that holds the seeds' run directories.

    Raises:
        ValueError: If the entry is not a trained candidate's, or on the first check that fails.
        FileNotFoundError: If the source record, a file it references or a seed's export
            provenance is missing.
    '''

    if entry.kind != 'trained':
        raise ValueError(
            f'{entry.name}: a {entry.kind} candidate is read from its table; only a trained '
            'candidate is imported'
        )
    source_ref, source = _source_record(entry, store)
    _check_source(entry, candidates, source, store)
    spec = economic_spec(entry, source=source.spec)
    provenance: Dict[int, Path] = {}
    for run in sorted(source.runs, key=lambda seed_run: seed_run.seed):
        provenance[run.seed] = _seed_provenance(entry, candidates, spec, run, Path(runs_root))
    return TrainedImport(
        entry=entry, spec=spec, source=source, source_ref=source_ref, provenance=provenance
    )

def _source_record(entry: CandidateEntry, store: ArtifactStore) -> Tuple[ArtifactRef, ArmRecord]:
    '''The stored record the row pins, verified, and the arm record it holds.'''

    located = store.root / entry.record
    if not located.is_file():
        raise FileNotFoundError(
            f'{entry.name}: its source record {entry.record} is not in the artifact store at '
            f'{store.root}'
        )
    reference = ArtifactRef(
        path=entry.record, sha256=entry.record_sha256, bytes=located.stat().st_size
    )
    try:
        path = store.resolve(reference)
    except ValueError as error:
        raise ValueError(
            f'{entry.name}: the stored {entry.record} is not its pin {entry.record_sha256}: '
            f'{error}'
        ) from None
    source = read_record(path, ArmRecord)
    if source.spec.name != entry.arm:
        raise ValueError(
            f'{entry.name}: its source record holds the arm {source.spec.name!r}, not '
            f'{entry.arm!r}'
        )
    return reference, source

def _check_source(
    entry: CandidateEntry, candidates: CandidateList, source: ArmRecord, store: ArtifactStore
) -> None:
    '''D8's checks, then the source's seeds, selection evidence and text-only table.'''

    check_arm(source, store, MIN_SEEDS)
    seeds = sorted(run.seed for run in source.runs)
    if seeds != entry.seeds:
        raise ValueError(
            f'{entry.name}: its source holds the seeds {seeds}, and its row names {entry.seeds}'
        )
    untrained = [run.seed for run in source.runs if run.training_run is None]
    if untrained:
        raise ValueError(
            f'{entry.name}: the source seeds {untrained} name no training run, so they carry no '
            'selection evidence for their checkpoints (spec 4.4)'
        )
    sha256 = source.text_only.table.sha256
    if sha256 != candidates.text_only_sha256:
        raise ValueError(
            f"{entry.name}: its source's text-only table {sha256[:12]} is not the shared "
            f'text-only table {candidates.text_only_sha256[:12]} that the frozen list pins'
        )
    sha256 = source.text_only.provenance.sha256
    pin = candidates.text_only_provenance_sha256
    if sha256 != pin:
        raise ValueError(
            f"{entry.name}: its source's text-only provenance {sha256[:12]} is not the shared "
            f"text-only table's provenance {pin[:12]} that the frozen list pins"
        )

def _seed_provenance(
    entry: CandidateEntry,
    candidates: CandidateList,
    spec: ArmSpec,
    run: SeedRun,
    runs_root: Path,
) -> Path:
    '''The seed's export provenance file, once it is checked to be the stored table's export.'''

    directory = runs_root / entry.run_directory.format(seed=run.seed)
    path = provenance_path(directory / Path(run.table.path).name)
    if not path.is_file():
        raise FileNotFoundError(
            f'{entry.name} seed {run.seed}: {path}: no export provenance beside its table'
        )
    content = json.loads(path.read_text(encoding='utf-8'))
    check_provenance(content, run.table.sha256, run.table.matrix_fingerprint, str(path))
    check_candidate_provenance(content, entry, candidates)
    named = content['checkpoint']['sha256']
    if named != run.checkpoint.sha256:
        raise ValueError(
            f'{entry.name} seed {run.seed}: {path} names another checkpoint, {named[:12]}, than '
            f"the seed's {run.checkpoint.sha256[:12]}"
        )
    fields = provenance_fields(content, run.table.sha256, run.table.matrix_fingerprint, str(path))
    check_seed_table(spec, run.seed, fields)
    return path
