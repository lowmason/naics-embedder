'''
Economic candidates on the fixture growth panel (Req 23; Req 24).

A deterministic candidate is a coordinate table over the fixture codebook in the export form,
with the provenance its builder writes beside it: ``write_candidate`` writes the tree builder's.
A trained candidate is a Stage 7/8 arm (``trained_source``): its record in the store, and each
seed's table with the provenance ``tools export-table`` writes (``export_provenance``) in the
seed's run directory. A candidate that reads a backbone reads the synthetic backbone and text of
``tests.fixtures.decision``, so the text-only table ``write_text_only`` writes over the codebook
is its D9 comparator; the tree candidate reads none (P7). ``fixture_list`` is the frozen list the
provenance rule (Req 29) checks them against, and ``list_for`` and ``write_candidate_list`` are
the list of exactly a decision's candidates, each pinned to its table or its source record.
'''

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import polars as pl
import yaml

from naics_embedder.decision.candidates import CandidateEntry, CandidateList, CheckpointContract
from naics_embedder.decision.economic import GROWTH_PANELS, REFERENCE_COMPARATOR
from naics_embedder.decision.records import (
    ArmRecord,
    ArmSpec,
    ArtifactRef,
    EconomicArmRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.scores import EMBEDDING_COMPARATOR, panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model.builders import LIBRARIES, METHODS
from tests.fixtures.decision import spec, synthetic_arm
from tests.fixtures.regressor_panel import CODEBOOK

DIMENSION = 3
# A trained source's monitor MRRs by epoch: its checkpoint is epoch 1's
MONITOR_MRRS = (0.1, 0.3, 0.2)
CODEBOOK_SHA256 = 'c' * 64
CONTRACT = CheckpointContract(bundle_id='fixture-bundle', contract_version='fixture-v1')
CODEBOOK_INPUT = {'path': 'naics_codebook.parquet', 'sha256': CODEBOOK_SHA256}
# A spec that reads no backbone: all five D9 fields null (P7)
NO_BACKBONE = dict.fromkeys(
    ('backbone', 'backbone_revision', 'descriptions_sha256', 'summaries_sha256', 'max_length')
)

def write_codebook(path: Path, codes: Sequence[str] = CODEBOOK) -> Path:
    '''
    A codebook parquet: ``code_id`` (Int32, 0…) and ``code``, in the given order, as bundle
    301cce28's codebook holds them.
    '''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            'code_id': list(range(len(codes))),
            'code': list(codes)
        },
        schema={
            'code_id': pl.Int32,
            'code': pl.Utf8
        },
    ).write_parquet(path)
    return path

def candidate_spec(name: str, **overrides) -> ArmSpec:
    '''
    A tree candidate's spec: one component, flat, at the fixture dimension, and reading no
    backbone (P7), as ``write_candidate``'s tree provenance reads none.
    '''

    fields = {
        'components': 1,
        'dimension': DIMENSION,
        'geometry': 'euclidean',
        **NO_BACKBONE,
        **overrides,
    }
    return spec(name, **fields)

def write_candidate(
    directory: Path,
    name: str,
    table: pl.DataFrame,
    *,
    codebook_sha256: str = CODEBOOK_SHA256,
    **entries,
) -> Path:
    '''
    A candidate's table and the provenance the tree builder writes beside it, naming the table's
    sha256 and ``matrix_fingerprint``; ``entries`` join or replace the provenance's entries.

    The provenance has exactly the tree builder's entries (``BUILDER_ENTRIES['tree']``). Its one
    input is the codebook, whose sha256 is ``codebook_sha256``, and its eigenvalues are ones.
    '''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}.parquet'
    table.write_parquet(path)
    dimension = sum(1 for column in table.columns if re.fullmatch(r'e\d+', column))
    provenance = {
        'builder': 'tree',
        'method': METHODS['tree'],
        'dimension': dimension,
        'codes': table.height,
        'inputs': {
            'codebook': {
                **CODEBOOK_INPUT, 'sha256': codebook_sha256
            }
        },
        'eigenvalues': [1.0] * (dimension + 1),
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': table_fingerprint(table),
        'library_versions': {
            library: 'fixture'
            for library in LIBRARIES
        },
        **entries,
    }
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def fixture_list(
    entries: Sequence[CandidateEntry],
    *,
    text_only_sha256: str,
    codebook_sha256: str = CODEBOOK_SHA256,
) -> CandidateList:
    '''
    A frozen candidate list over the entries, with the fixture checkpoint contract.

    A test whose candidates come from ``build_candidate`` passes the written codebook's
    ``sha256_file``, which the builder records, as ``codebook_sha256``.
    '''

    return CandidateList(
        codebook_sha256=codebook_sha256,
        text_only_sha256=text_only_sha256,
        checkpoint_contract=CONTRACT,
        candidates=list(entries),
    )

def list_for(
    arms: Sequence[EconomicArmRecord],
    store: ArtifactStore,
    *,
    text_only_sha256: str,
    codebook_sha256: str = CODEBOOK_SHA256,
) -> CandidateList:
    '''
    The frozen list of exactly the arms' candidates, with the fixture checkpoint contract.

    Each arm is one row, in order, with its spec's dimension, components and geometry. A
    deterministic arm's row has the kind its stored provenance's ``builder`` names and
    ``table_sha256`` pinned to its first run's table. A trained arm, one with a ``source``, has a
    trained row pinned to that record, with the arm and seeds the stored record holds and the run
    directories ``trained_source`` writes (an ``ArmRecord`` holds no run-directory pattern).
    '''

    entries = []
    for arm in arms:
        if arm.source is not None:
            source = read_record(store.resolve(arm.source), ArmRecord)
            entries.append(
                CandidateEntry(
                    name=arm.spec.name,
                    kind='trained',
                    dimension=arm.spec.dimension,
                    components=arm.spec.components,
                    geometry=arm.spec.geometry,
                    record=arm.source.path,
                    record_sha256=arm.source.sha256,
                    arm=source.spec.name,
                    seeds=sorted(run.seed for run in source.runs),
                    run_directory=f'{source.spec.name}-s{{seed}}',
                )
            )
            continue
        run = arm.runs[0]
        provenance = store.candidate_provenance(run.table, run.provenance.file).content
        entries.append(
            CandidateEntry(
                name=arm.spec.name,
                kind=provenance['builder'],
                dimension=arm.spec.dimension,
                components=arm.spec.components,
                geometry=arm.spec.geometry,
                table_sha256=run.table.sha256,
            )
        )
    return fixture_list(entries, text_only_sha256=text_only_sha256, codebook_sha256=codebook_sha256)

def write_candidate_list(
    path: Path,
    arms: Sequence[EconomicArmRecord],
    store: ArtifactStore,
    *,
    text_only_sha256: str,
    codebook_sha256: str = CODEBOOK_SHA256,
) -> Path:
    '''``list_for``'s list, written as the YAML that ``load_candidate_list`` reads.'''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = list_for(
        arms, store, text_only_sha256=text_only_sha256, codebook_sha256=codebook_sha256
    )
    path.write_text(yaml.safe_dump(frozen.model_dump(mode='json'), sort_keys=False))
    return path

def scaled_arm(
    arm: EconomicArmRecord, factors: Mapping[str, float], store: ArtifactStore
) -> EconomicArmRecord:
    '''
    The arm, rescored with each ``covariates+embedding`` prediction moved to the outcome plus
    ``factors[panel]`` times the reference's error on its row and repeat, in every run. Every
    row's squared error is then factor² times the reference's, so Δ on each item is
    (1 − factor²) times the reference's error, whatever the draw.

    Every other column, the binding column included, is carried through.
    '''

    seen, heldout = GROWTH_PANELS
    keys = ['panel', 'repeat', 'code', 'feature_year']
    factor = pl.when(pl.col('panel') == seen).then(factors[seen]).otherwise(factors[heldout])
    error = pl.col('reference') - pl.col('outcome')
    runs = []
    for run in arm.runs:
        predictions = store.read_frame(run.predictions)
        reference = predictions.filter(pl.col('comparator') == REFERENCE_COMPARATOR
                                       ).select(*keys, reference='prediction')
        moved = predictions.join(reference, on=keys).with_columns(
            prediction=pl.when(pl.col('comparator') == EMBEDDING_COMPARATOR).then(
                pl.col('outcome') + factor * error
            ).otherwise('prediction')
        ).select(predictions.columns)
        scores = regressor_scores(moved, arm.panels.fit_settings['repeats'], GROWTH_PANELS)
        runs.append(
            run.model_copy(
                update={
                    'predictions': store.put_frame(moved, 'predictions.parquet'),
                    'scores': store.put_frame(scores, 'scores.parquet'),
                    'statistics': {
                        panel: panel_statistic(scores, panel, EMBEDDING_COMPARATOR)
                        for panel in GROWTH_PANELS
                    },
                }
            )
        )
    return arm.model_copy(update={'runs': runs})

def export_provenance(
    table_path: Path,
    arm_spec: ArmSpec,
    checkpoint: ArtifactRef,
    *,
    geometry: bool = True,
) -> Dict[str, Any]:
    '''
    A trained table's export provenance (``tools export-table``), as a dict.

    It names ``CONTRACT`` and the checkpoint's sha256, and what ``arm_spec`` reads. It has the
    entries of Stage 8's files, or Stage 7's, which have no ``geometry`` entry, when
    ``geometry`` is false.
    '''

    table = pl.read_parquet(table_path)
    provenance: Dict[str, Any] = {
        'checkpoint': {
            'path': checkpoint.path,
            'sha256': checkpoint.sha256
        },
        'contract': CONTRACT.model_dump(mode='json'),
        'backbone': arm_spec.backbone,
        'revision': arm_spec.backbone_revision,
        'max_length': arm_spec.max_length,
        'descriptions': {
            'path': 'naics_descriptions.parquet',
            'sha256': arm_spec.descriptions_sha256
        },
        'summaries': arm_spec.summaries_sha256,
        'tokenizer': arm_spec.backbone,
        'codes': table.height,
        'dimension': arm_spec.dimension,
        'coordinates': f'the fixture {arm_spec.geometry} coordinates',
        'table_sha256': sha256_file(table_path),
        'matrix_fingerprint': table_fingerprint(table),
        'library_versions': {
            library: 'fixture'
            for library in ('torch', 'polars')
        },
        'generated_at': '2026-11-14T00:00:00+00:00',
    }
    if geometry:
        provenance['geometry'] = arm_spec.geometry
    return provenance

def trained_source(
    store: ArtifactStore,
    directory: Path,
    name: str,
    *,
    text_only: Path,
    dimension: int = DIMENSION,
    geometry: str = 'euclidean',
    codes: Sequence[str] = CODEBOOK,
) -> Tuple[CandidateEntry, Path]:
    '''
    A trained candidate's Stage 7/8 inputs, as the import finds them, and its frozen-list row.

    The arm is ``synthetic_arm``'s over ``codes``, read beside ``text_only``: five trained seeds,
    0 to 4, whose ``MONITOR_MRRS`` select epoch 1. Its record is ``<store>/records/<name>.json``.
    Seed k's stored table, with its export provenance (``export_provenance``) beside it, is in
    ``<runs root>/<name>-s<k>/`` under the stored table's own name.

    Returns:
        The trained row, pinned to the record, and the runs root, ``<directory>/runs``.
    '''

    directory = Path(directory)
    arm_spec = spec(name, dimension=dimension, geometry=geometry)
    source = synthetic_arm(
        store,
        directory,
        arm_spec,
        {},
        codes=codes,
        monitor_mrrs=MONITOR_MRRS,
        text_only_table=text_only,
    )
    record = write_record(source, store.root / 'records' / f'{name}.json')
    runs_root = directory / 'runs'
    for run in source.runs:
        table = runs_root / f'{name}-s{run.seed}' / Path(run.table.path).name
        table.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store.resolve(run.table), table)
        content = export_provenance(table, arm_spec, run.checkpoint)
        provenance_path(table).write_text(json.dumps(content, indent=2) + '\n')
    entry = CandidateEntry(
        name=name,
        kind='trained',
        dimension=dimension,
        components=arm_spec.components,
        geometry=geometry,
        record=f'records/{name}.json',
        record_sha256=sha256_file(record),
        arm=name,
        seeds=[run.seed for run in source.runs],
        run_directory=f'{name}-s{{seed}}',
    )
    return entry, runs_root
