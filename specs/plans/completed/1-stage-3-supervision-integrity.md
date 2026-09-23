# Stage-3 Supervision Integrity Implementation Plan

**Status: COMPLETE (2026-09-23)** — executed via executing-plans; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Stage-3’s implicit, contradictory supervision arrays with one versioned identity-first contract in which structural facts remain untouched, exclusions remain repulsive, selection is index-checked, and ranking gradients point in the corrective direction.

**Architecture:** Build every structural, semantic, and exclusion field from a canonical pair-fact table inside an immutable supervision bundle. At runtime, load that bundle into a dense `SupervisionIndex`, turn one collated candidate pool into a validated `NegativeCandidateBatch`, let miners return proposals over source indices, and perform one checked gather into `SelectedNegativeBatch` before false-negative handling or any candidate-based loss. Exact resume validates bundle and contract fingerprints; old artifacts and checkpoints run only through an explicitly tagged containment or weights-only path.

**Tech Stack:** Python 3.10+, Polars 1.9+, PyArrow 17+, PyTorch 2.4+, PyTorch Lightning 2.4+, Pydantic 2.12+, Typer 0.12+, pytest 8.3+, CPU `torch.distributed`/Gloo for the distributed integration test.

## Global Constraints

- Use symbolic supervision contract version `stage3-supervision-v1`.
- Use structural-preference loss version `structural-preference-v1` and mining contract version `negative-selection-v1`.
- NAICS codes remain strings at input/output boundaries; tensor identity comes only from the bundle’s fingerprinted numeric codebook.
- Do not change the NAICS hierarchy definition or perform any code-vintage conversion.
- Structural distance and canonical structural relation are never mutated by semantic or exclusion processing.
- Canonical pair orientation remains shallower-code-first with stable code ordering on ties. Relation matrices may mirror the canonical relation ID, but no consumer may interpret the canonical relation name as a bidirectional label.
- Preserve both exclusion directions; derive `is_explicit_exclusion` as their logical OR and validate the derivation every time data enters a boundary.
- Reserve exactly one final negative slot for an explicit exclusion when any exists; `K` is at least one; final negative code IDs are unique.
- Use a process- and platform-independent digest for exclusion rotation; Python `hash()` is forbidden.
- Miners return source indices, source UIDs, scores, and reasons; they never return gathered embeddings.
- Gather only candidate-intrinsic fields across ranks, then recompute all anchor-relative supervision for each local anchor.
- Contract violations are fatal and actionable; do not convert them to warnings or zero auxiliary losses.
- New and legacy artifacts coexist. Never overwrite a production bundle in place, and write its manifest only after every artifact validates.
- A repaired configuration names one supervision manifest as the authoritative artifact entry point; there is no automatic fallback to legacy paths.
- Repaired configurations reject `loss.rank_order_weight` and `data_loader.streaming.phase1_exclusion_weight` with a migration message.
- Legacy execution requires explicit `legacy_containment` mode and cannot masquerade as repaired Stage-3 training.
- Existing HGCN sampling, objectives, and semantic-retention behavior remain unchanged; compatibility projection only adapts the new training-pair schema.
- Spearman/curvature metrics, HGCN redesign, QCEW evaluation, curriculum-documentation cleanup, and unrelated refactors remain out of scope.
- Normal repaired Stage-3 training stays gated until the forced-reorder regression, gradient-sign test, bundle validation, and full training-step integration test all pass.

## Execution Baseline

The planning branch contains the approved spec but is three fetched commits behind `origin/main`. Those commits add `Phase1MapDataset`, `difficulty_sampler.py`, `all_candidates`, and the candidate-pool configuration explicitly named by the spec. Before Task 1, create the execution worktree via `using-git-worktrees` and merge that baseline while retaining this plan and `specs/completed/stage-3-supervision-integrity.md`:

```bash
git merge --no-edit origin/main
git status --short --branch
uv run pytest tests/unit/test_difficulty_sampler.py tests/unit/test_datamodule.py -q
```

Expected: the merge includes upstream commit `545ec50` (or a descendant), both spec files remain present, the worktree has no unresolved conflicts, and the two upstream suites pass.

## File Structure

### New focused supervision package

- `src/naics_embedder/supervision/__init__.py` — stable public exports only.
- `src/naics_embedder/supervision/schema.py` — contract/version constants, enums, and manifest models.
- `src/naics_embedder/supervision/artifacts.py` — versioned Parquet I/O, hashes, manifest loading, bundle validation, and resolved bundle paths.
- `src/naics_embedder/supervision/index.py` — codebook-backed dense structural/exclusion lookup and anchor-relative joins.
- `src/naics_embedder/supervision/candidates.py` — immutable entity/candidate/selection dataclasses and the one checked gather.
- `src/naics_embedder/supervision/selection.py` — stable exclusion rotation, proposal merge, code deduplication, and deterministic backfill.
- `src/naics_embedder/supervision/checkpoints.py` — checkpoint contract validation and audited weights-only migration.

### Data generation and compatibility

- `src/naics_embedder/data/supervision_bundle.py` — canonical codebook/pair-fact construction and atomic bundle orchestration.
- `src/naics_embedder/data/compute_distances.py` — pure structural distance frame; no exclusion mutation or final artifact authority.
- `src/naics_embedder/data/compute_relations.py` — pure canonical structural relation frame; `cross_sector` is structural ID `99`, never an exclusion sentinel.
- `src/naics_embedder/data/create_triplets.py` — training-pair projection with code IDs, semantic targets/sources, raw structure, provenance, and compatibility columns.
- `src/naics_embedder/cli/commands/data.py` — one `data supervision` command and `data all` orchestration.
- `conf/data/supervision.yaml` — bundle-build inputs, vintage, output root, and relation mapping.
- `src/naics_embedder/graph_model/dataloader/hgcn_streaming_dataset.py` — narrow compatibility reader for rebuilt training pairs.

### Runtime, loss, and migration

- `src/naics_embedder/text_model/dataloader/streaming_dataset.py` — bundle-backed candidate pools, quota inclusion, versioned cache envelopes.
- `src/naics_embedder/text_model/dataloader/datamodule.py` — non-mutating invalid-row collation and one canonical candidate metadata path.
- `src/naics_embedder/text_model/dataloader/difficulty_sampler.py` — difficulty proposals expressed as source indices.
- `src/naics_embedder/text_model/dataloader/tokenization_cache.py` — descriptions/codebook-fingerprint sidecar validation.
- `src/naics_embedder/text_model/hard_negative_mining.py` — geometric/router score proposals, never gathered embeddings.
- `src/naics_embedder/text_model/mixins/distributed.py` — intrinsic entity gather only.
- `src/naics_embedder/text_model/mixins/curriculum.py` — canonical candidate construction, proposals, and selection.
- `src/naics_embedder/text_model/false_negative_strategies.py` — exclusion-cleared eligibility for masking/attraction.
- `src/naics_embedder/text_model/loss.py` — padding-aware contrastive loss and `StructuralPreferenceLoss`; active `LambdaRankLoss` removed.
- `src/naics_embedder/text_model/mixins/loss.py` — selected-batch loss boundary with fatal contract failures.
- `src/naics_embedder/text_model/naics_model.py` — repaired/containment policies, canonical training-step flow, checkpoint metadata hooks.
- `src/naics_embedder/utils/config.py`, `src/naics_embedder/utils/validation.py`, `src/naics_embedder/cli/commands/training.py`, and `conf/config.yaml` — authoritative manifest config, migration validation, pre-model bundle gate, and checkpoint mode.

### Tests and documentation

- `tests/unit/test_supervision_schema.py`
- `tests/unit/test_supervision_artifacts.py`
- `tests/unit/test_supervision_index.py`
- `tests/unit/test_candidate_contract.py`
- `tests/unit/test_negative_selection.py`
- `tests/unit/test_checkpoint_contract.py`
- `tests/fixtures/supervision.py` — hand-written five-code pair facts plus reusable bundle and candidate fixtures; expectations never call production derivation code.
- `tests/conftest.py` — registers the supervision fixture plugin.
- Existing focused suites under `tests/unit/` for generation, datamodule, mining, loss, config, CLI, model, and HGCN compatibility.
- `tests/integration/test_distributed_supervision.py`
- `tests/integration/test_stage3_training_step.py`
- `docs/text_training.md` and `docs/api/config.md`

---

### Task 1: Freeze the supervision vocabulary and manifest schema

**Files:**
- Create: `src/naics_embedder/supervision/__init__.py`
- Create: `src/naics_embedder/supervision/schema.py`
- Create: `tests/unit/test_supervision_schema.py`

**Interfaces:**
- Consumes: no earlier task interfaces.
- Produces: `CONTRACT_VERSION`, `STRUCTURAL_PREFERENCE_LOSS_VERSION`, `MINING_CONTRACT_VERSION`, `SemanticTarget`, `SemanticSource`, `SamplingRole`, `SamplingProvenance`, `SelectionReason`, `ArtifactFile`, `ArtifactRecord`, and `SupervisionManifest`.

- [x] **Step 1: Write the failing schema tests**

```python
# tests/unit/test_supervision_schema.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    ArtifactRecord,
    SemanticSource,
    SemanticTarget,
    SupervisionManifest,
)


def _manifest() -> SupervisionManifest:
    artifact_file = ArtifactFile(path='naics_codebook.parquet', sha256='a' * 64, row_count=3)
    return SupervisionManifest(
        contract_version=CONTRACT_VERSION,
        bundle_id='bundle-123',
        generated_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
        generator_revision='abc123',
        naics_vintage=2022,
        codebook_order=('111111', '111112', '111113'),
        codebook_fingerprint='b' * 64,
        description_fingerprint='c' * 64,
        exclusion_fingerprint='d' * 64,
        generation_parameters={'seed': 42},
        structural_relation_ids={'child': 1, 'cross_sector': 99},
        artifacts={
            'codebook': ArtifactRecord(
                path='naics_codebook.parquet',
                schema_version='codebook-v1',
                row_count=3,
                exclusion_count=0,
                files=(artifact_file,),
            )
        },
        validation_results={'codebook_unique': True},
    )


def test_manifest_round_trip_preserves_contract_identity(tmp_path):
    manifest = _manifest()
    path = tmp_path / 'manifest.json'
    path.write_text(manifest.model_dump_json(indent=2))

    restored = SupervisionManifest.model_validate_json(path.read_text())

    assert restored.contract_version == 'stage3-supervision-v1'
    assert restored.codebook_order == ('111111', '111112', '111113')
    assert restored.artifacts['codebook'].files[0].sha256 == 'a' * 64
    assert SemanticTarget.UNRELATED.value == 'unrelated'
    assert SemanticSource.EXPLICIT_EXCLUSION.value == 'explicit_exclusion'


def test_manifest_rejects_parent_traversal():
    manifest = _manifest().model_dump()
    manifest['artifacts']['codebook']['path'] = '../outside.parquet'

    with pytest.raises(ValidationError, match='relative bundle path'):
        SupervisionManifest.model_validate(manifest)
```

- [x] **Step 2: Run the schema tests and verify the import failure**

Run: `uv run pytest tests/unit/test_supervision_schema.py -q`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'naics_embedder.supervision'`.

- [x] **Step 3: Add the complete contract vocabulary**

```python
# src/naics_embedder/supervision/schema.py
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class SemanticTarget(str, Enum):
    RELATED = 'related'
    UNRELATED = 'unrelated'
    UNKNOWN = 'unknown'


class SemanticSource(str, Enum):
    TRAINING_POSITIVE = 'training_positive'
    EXPLICIT_EXCLUSION = 'explicit_exclusion'
    UNLABELED = 'unlabeled'


class SamplingRole(str, Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


SAMPLING_ROLE_TO_ID = {
    SamplingRole.POSITIVE: 1,
    SamplingRole.NEGATIVE: 2,
}


class SamplingProvenance(IntEnum):
    GENERATED = 1
    DIFFICULTY = 2
    LOCAL_POOL = 3
    DISTRIBUTED_POOL = 4
    BACKFILL = 5


class SelectionReason(IntEnum):
    EXCLUSION_QUOTA = 1
    GEOMETRIC = 2
    ROUTER = 3
    DIFFICULTY = 4
    BACKFILL = 5


class ArtifactFile(BaseModel):
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
```

```python
# src/naics_embedder/supervision/__init__.py
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
    SemanticSource,
    SemanticTarget,
    SelectionReason,
)

__all__ = [
    'CONTRACT_VERSION',
    'MINING_CONTRACT_VERSION',
    'STRUCTURAL_PREFERENCE_LOSS_VERSION',
    'SemanticSource',
    'SemanticTarget',
    'SelectionReason',
]
```

- [x] **Step 4: Run the focused tests**

Run: `uv run pytest tests/unit/test_supervision_schema.py -q`
Expected: `2 passed`.

- [x] **Step 5: Run lint on the new package**

Run: `uv run ruff check src/naics_embedder/supervision tests/unit/test_supervision_schema.py`
Expected: `All checks passed!`.

- [x] **Step 6: Commit the schema boundary**

```bash
git add src/naics_embedder/supervision tests/unit/test_supervision_schema.py
git commit -m "feat(supervision): define stage3 contract schema"
```

---

### Task 2: Build canonical codebook and pair facts without exclusion sentinels

**Files:**
- Create: `src/naics_embedder/data/supervision_bundle.py`
- Modify: `src/naics_embedder/data/compute_distances.py:174-334`
- Modify: `src/naics_embedder/data/compute_relations.py:169-340`
- Modify: `conf/data/relations.yaml`
- Test: `tests/unit/test_data_distances.py`
- Test: `tests/unit/test_data_relations.py`
- Create: `tests/unit/test_supervision_artifacts.py`
- Create: `tests/fixtures/supervision.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: schema/version constants from Task 1.
- Produces: `build_codebook(descriptions: pl.DataFrame) -> pl.DataFrame`, `codebook_fingerprint(codebook: pl.DataFrame) -> str`, `attach_exclusion_provenance(pair_facts, descriptions, codebook) -> pl.DataFrame`, `build_pair_facts(distances, relations, descriptions, codebook) -> pl.DataFrame`, `distance_matrix_from_pair_facts(pair_facts, codebook) -> pl.DataFrame`, and `relation_matrix_from_pair_facts(pair_facts, codebook) -> pl.DataFrame`.

- [x] **Step 1: Replace sentinel expectations with invariant-focused failing tests**

Register a shared, hand-written fixture plugin. These rows deliberately include an ordinary
positive, a forward exclusion, an ordinary negative, and a reverse exclusion; later tests can
therefore assert exact values without using the production derivation as their oracle.

```python
# tests/fixtures/supervision.py
import polars as pl
import pytest


@pytest.fixture
def descriptions_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'index': [0, 1, 2, 3, 4],
            'code': ['111111', '111112', '111113', '222222', '333333'],
            'excluded_codes': [['111113'], None, None, ['111112'], None],
        }
    )


@pytest.fixture
def structural_frames_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    pair_columns = {
        'idx_i': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        'idx_j': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
        'code_i': [
            '111111',
            '111111',
            '111111',
            '111111',
            '111112',
            '111112',
            '111112',
            '111113',
            '111113',
            '222222',
        ],
        'code_j': [
            '111112',
            '111113',
            '222222',
            '333333',
            '111113',
            '222222',
            '333333',
            '222222',
            '333333',
            '333333',
        ],
    }
    distances = pl.DataFrame(
        pair_columns
        | {
            'structural_distance': [
                0.5,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ]
        }
    )
    relations = pl.DataFrame(
        pair_columns
        | {
            'structural_relation_id': [1, 2, 99, 99, 3, 99, 99, 99, 99, 99],
            'structural_relation_name': [
                'child',
                'sibling',
                'cross_sector',
                'cross_sector',
                'grandchild',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
            ],
        }
    )
    return distances, relations


@pytest.fixture
def pair_facts_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
            'code_j_id': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
            'code_i': [
                '111111',
                '111111',
                '111111',
                '111111',
                '111112',
                '111112',
                '111112',
                '111113',
                '111113',
                '222222',
            ],
            'code_j': [
                '111112',
                '111113',
                '222222',
                '333333',
                '111113',
                '222222',
                '333333',
                '222222',
                '333333',
                '333333',
            ],
            'structural_distance': [
                0.5,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ],
            'structural_relation_id': [1, 2, 99, 99, 3, 99, 99, 99, 99, 99],
            'structural_relation_name': [
                'child',
                'sibling',
                'cross_sector',
                'cross_sector',
                'grandchild',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
            ],
            'code_i_excludes_code_j': [
                False, True, False, False, False, False, False, False, False, False
            ],
            'code_j_excludes_code_i': [
                False, False, False, False, False, True, False, False, False, False
            ],
            'is_explicit_exclusion': [
                False, True, False, False, False, True, False, False, False, False
            ],
        }
    )
```

```python
# add near the imports in tests/conftest.py
pytest_plugins = ('tests.fixtures.supervision',)
```

Add these tests:

```python
# tests/unit/test_supervision_artifacts.py
import polars as pl

from naics_embedder.data.supervision_bundle import (
    build_codebook,
    build_pair_facts,
    codebook_fingerprint,
    distance_matrix_from_pair_facts,
    relation_matrix_from_pair_facts,
)


def test_exclusion_provenance_does_not_mutate_structure(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)

    facts = build_pair_facts(distances, relations, descriptions, codebook)
    row = facts.filter(
        pl.col('code_i').eq('111111') & pl.col('code_j').eq('111113')
    ).row(0, named=True)

    assert row['structural_distance'] == 2.0
    assert row['structural_relation_id'] == 2
    assert row['structural_relation_name'] == 'sibling'
    assert row['code_i_excludes_code_j'] is True
    assert row['code_j_excludes_code_i'] is False
    assert row['is_explicit_exclusion'] is True


def test_reverse_direction_survives_canonical_orientation(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)

    facts = build_pair_facts(distances, relations, descriptions, codebook)
    row = facts.filter(
        pl.col('code_i').eq('111112') & pl.col('code_j').eq('222222')
    ).row(0, named=True)

    assert row['code_i_excludes_code_j'] is False
    assert row['code_j_excludes_code_i'] is True
    assert row['is_explicit_exclusion'] is True


def test_matrices_reconcile_with_pair_facts_and_codebook_order(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    descriptions = descriptions_fixture
    codebook = build_codebook(descriptions)
    facts = build_pair_facts(distances, relations, descriptions, codebook)

    distance_matrix = distance_matrix_from_pair_facts(facts, codebook)
    relation_matrix = relation_matrix_from_pair_facts(facts, codebook)

    assert distance_matrix.row(0)[1] == 0.5
    assert distance_matrix.row(1)[0] == 0.5
    assert relation_matrix.row(0)[1] == 1
    assert relation_matrix.row(1)[0] == 1
    assert codebook_fingerprint(codebook) == codebook_fingerprint(codebook.clone())


def test_pair_rows_keep_the_generator_canonical_orientation(
    descriptions_fixture, structural_frames_fixture
):
    distances, relations = structural_frames_fixture
    facts = build_pair_facts(
        distances,
        relations,
        descriptions_fixture,
        build_codebook(descriptions_fixture),
    )
    assert facts.select(
        pl.col('code_i_id').lt(pl.col('code_j_id')).all()
    ).item()
```

In `tests/unit/test_data_distances.py` and `tests/unit/test_data_relations.py`, replace tests that expect distance `0`, relation ID `0`, or relation name `excluded` with assertions that the structural values are unchanged and no structural column contains an exclusion sentinel.

- [x] **Step 2: Run the generation tests and verify the missing-module/API failures**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_data_distances.py tests/unit/test_data_relations.py -q`
Expected: FAIL because `supervision_bundle` and the new structural column contract do not exist.

- [x] **Step 3: Make distance and relation generation structural-only**

> Deviation: Canonical orientation keeps (level, code) order: the real `index` is lexicographic, so the planned `code_i_id < code_j_id` check failed on 1.5M real rows (D1); unmapped relation names are fatal (D5); `unrelated` became `cross_sector` = 99 with consumer updates (D4).

Apply these exact semantic replacements around the final frame construction:

```diff
# src/naics_embedder/data/compute_distances.py
-    exclusions = _get_exclusions(distances_df)
-
-    distances_df = (
-        distances_df.join(exclusions, on=['code_i', 'code_j'], how='left').with_columns(
-            excluded=pl.col('excluded').fill_null(False)
-        ).select(
-            pl.col('idx_i'),
-            pl.col('idx_j'),
-            pl.col('code_i'),
-            pl.col('code_j'),
-            distance=pl.when(pl.col('excluded')).then(pl.lit(0)).otherwise(pl.col('distance')),
-        ).sort('idx_i', 'idx_j')
-    )
+    distances_df = distances_df.rename({'distance': 'structural_distance'})
```

```diff
# src/naics_embedder/data/compute_relations.py
-                     relation_id=pl.col('relation_id').fill_null(99),
-                     relation=pl.col('relation').fill_null('unrelated'),
+                     structural_relation_id=pl.col('relation_id').fill_null(99),
+                     structural_relation_name=pl.col('relation').fill_null('cross_sector'),
                  ).sort('idx_i', 'idx_j')
     )
-
-    exclusions = _get_exclusions(relations_df)
-
-    relations_df = (
-        relations_df.join(exclusions, on=['code_i', 'code_j'], how='left').with_columns(
-            excluded=pl.col('excluded').fill_null(False)
-        ).select(
-            pl.col('idx_i'),
-            pl.col('idx_j'),
-            pl.col('code_i'),
-            pl.col('code_j'),
-            relation_id=pl.when(pl.col('excluded')).then(pl.lit(0)).otherwise(
-                pl.col('relation_id')
-            ),
-            relation=pl.when(pl.col('excluded')).then(pl.lit('excluded')).otherwise(
-                pl.col('relation')
-            ),
-        ).sort('idx_i', 'idx_j')
-    )
```

Delete both now-unused `_get_exclusions` functions. Change the matrix helpers to read `structural_distance` and `structural_relation_id`. Add `cross_sector: 99` to `conf/data/relations.yaml`. Expose pure helpers `compute_structural_distances(input_parquet: str, cfg: DistancesConfig)` and `compute_structural_relations(input_parquet: str, relation_ids: Mapping[str, int])` by moving the existing computation bodies ahead of file writes; keep thin CLI-compatible wrappers until Task 3 changes orchestration.

Preserve the current generator orientation when forming pair rows: the shallower code is `code_i`, ties use stable code order, and `code_i_id < code_j_id` is validated after the codebook join. Mirroring `structural_relation_id` into a lookup matrix does not create or persist a reverse relation name.

- [x] **Step 4: Implement canonical codebook, directional provenance, and matrices**

> Deviation: `build_pair_facts` also checks identity, self-pairs, completeness, nulls, and exclusion attachment, and diagnoses duplicates before orientation (D6).

```python
# src/naics_embedder/data/supervision_bundle.py
import hashlib

import numpy as np
import polars as pl


def build_codebook(descriptions: pl.DataFrame) -> pl.DataFrame:
    codebook = (
        descriptions.select(code_id=pl.col('index').cast(pl.Int32), code=pl.col('code').cast(pl.Utf8))
        .unique()
        .sort('code_id')
    )
    expected = list(range(codebook.height))
    if codebook.get_column('code_id').to_list() != expected:
        raise ValueError('description indices must be contiguous code IDs starting at zero')
    if codebook.get_column('code').n_unique() != codebook.height:
        raise ValueError('codebook contains duplicate NAICS code strings')
    return codebook


def codebook_fingerprint(codebook: pl.DataFrame) -> str:
    payload = '\n'.join(
        f'{row["code_id"]}\t{row["code"]}' for row in codebook.iter_rows(named=True)
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _directed_exclusions(descriptions: pl.DataFrame, codebook: pl.DataFrame) -> pl.DataFrame:
    code_ids = codebook.rename({'code': 'source_code', 'code_id': 'source_code_id'})
    target_ids = codebook.rename({'code': 'target_code', 'code_id': 'target_code_id'})
    return (
        descriptions.select(
            source_code=pl.col('code').cast(pl.Utf8),
            target_code=pl.col('excluded_codes'),
        )
        .explode('target_code')
        .filter(pl.col('target_code').is_not_null())
        .join(code_ids, on='source_code', how='inner', validate='m:1')
        .join(target_ids, on='target_code', how='inner', validate='m:1')
        .select('source_code_id', 'target_code_id')
        .unique()
    )


def attach_exclusion_provenance(
    pair_facts: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> pl.DataFrame:
    directed = _directed_exclusions(descriptions, codebook)
    forward = directed.rename(
        {'source_code_id': 'code_i_id', 'target_code_id': 'code_j_id'}
    ).with_columns(code_i_excludes_code_j=pl.lit(True))
    reverse = directed.rename(
        {'source_code_id': 'code_j_id', 'target_code_id': 'code_i_id'}
    ).with_columns(code_j_excludes_code_i=pl.lit(True))

    return (
        pair_facts.join(forward, on=['code_i_id', 'code_j_id'], how='left')
        .join(reverse, on=['code_i_id', 'code_j_id'], how='left')
        .with_columns(
            pl.col('code_i_excludes_code_j').fill_null(False),
            pl.col('code_j_excludes_code_i').fill_null(False),
        )
        .with_columns(
            is_explicit_exclusion=(
                pl.col('code_i_excludes_code_j') | pl.col('code_j_excludes_code_i')
            )
        )
        .sort('code_i_id', 'code_j_id')
    )


def build_pair_facts(
    distances: pl.DataFrame,
    relations: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> pl.DataFrame:
    structural = distances.join(
        relations.select(
            'idx_i',
            'idx_j',
            'structural_relation_id',
            'structural_relation_name',
        ),
        on=['idx_i', 'idx_j'],
        how='inner',
        validate='1:1',
    ).select(
        code_i_id=pl.col('idx_i').cast(pl.Int32),
        code_j_id=pl.col('idx_j').cast(pl.Int32),
        code_i=pl.col('code_i').cast(pl.Utf8),
        code_j=pl.col('code_j').cast(pl.Utf8),
        structural_distance=pl.col('structural_distance').cast(pl.Float32),
        structural_relation_id=pl.col('structural_relation_id').cast(pl.Int16),
        structural_relation_name=pl.col('structural_relation_name').cast(pl.Utf8),
    )
    if structural.filter(pl.col('structural_distance').eq(0.0)).height:
        raise ValueError('distinct-code pair facts cannot contain structural distance zero')
    if structural.filter(pl.col('code_i_id').ge(pl.col('code_j_id'))).height:
        raise ValueError('pair facts violate canonical codebook orientation')
    if structural.filter(
        pl.col('structural_relation_id').eq(0)
        | pl.col('structural_relation_name').eq('excluded')
    ).height:
        raise ValueError('structural relation fields contain an exclusion sentinel')
    return attach_exclusion_provenance(structural, descriptions, codebook)


def _matrix(
    pair_facts: pl.DataFrame,
    codebook: pl.DataFrame,
    value_column: str,
    dtype: np.dtype,
) -> pl.DataFrame:
    size = codebook.height
    values = np.zeros((size, size), dtype=dtype)
    for row in pair_facts.select('code_i_id', 'code_j_id', value_column).iter_rows(named=True):
        code_i_id = row['code_i_id']
        code_j_id = row['code_j_id']
        values[code_i_id, code_j_id] = row[value_column]
        values[code_j_id, code_i_id] = row[value_column]
    columns = [
        f'idx_{row["code_id"]}-code_{row["code"]}' for row in codebook.iter_rows(named=True)
    ]
    return pl.from_numpy(values, schema=columns)


def distance_matrix_from_pair_facts(
    pair_facts: pl.DataFrame, codebook: pl.DataFrame
) -> pl.DataFrame:
    return _matrix(pair_facts, codebook, 'structural_distance', np.float32)


def relation_matrix_from_pair_facts(
    pair_facts: pl.DataFrame, codebook: pl.DataFrame
) -> pl.DataFrame:
    return _matrix(pair_facts, codebook, 'structural_relation_id', np.int16)
```

- [x] **Step 5: Run structural and pair-fact tests**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_data_distances.py tests/unit/test_data_relations.py -q`
Expected: PASS; no test expects an exclusion sentinel in a structural field.

- [x] **Step 6: Commit canonical structural facts**

```bash
git add conf/data/relations.yaml src/naics_embedder/data/compute_distances.py src/naics_embedder/data/compute_relations.py src/naics_embedder/data/supervision_bundle.py tests/conftest.py tests/fixtures/supervision.py tests/unit/test_data_distances.py tests/unit/test_data_relations.py tests/unit/test_supervision_artifacts.py
git commit -m "fix(data): preserve structure across exclusions"
```

---

### Task 3: Generate semantic training pairs from canonical facts

**Files:**
- Modify: `src/naics_embedder/data/create_triplets.py:22-326`
- Modify: `src/naics_embedder/data/supervision_bundle.py`
- Modify: `tests/unit/test_data_triplets.py`
- Modify: `tests/unit/test_supervision_artifacts.py`

**Interfaces:**
- Consumes: `build_pair_facts` and codebook from Task 2.
- Produces: `build_training_pairs(pair_facts: pl.DataFrame) -> pl.DataFrame` with identity, raw structure, semantic target/source, sampling role/provenance, exclusion directions, derived exclusion, and legacy compatibility columns.

- [x] **Step 1: Write failing semantic and direct-positive tests**

```python
# add to tests/unit/test_data_triplets.py
import pytest

from naics_embedder.data.create_triplets import (
    _validate_training_pairs,
    build_training_pairs,
)


def test_training_pairs_keep_semantics_separate_from_structure(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    excluded = pairs.filter(pl.col('negative_is_explicit_exclusion')).row(0, named=True)
    ordinary = pairs.filter(~pl.col('negative_is_explicit_exclusion')).row(0, named=True)

    assert excluded['negative_semantic_target'] == 'unrelated'
    assert excluded['negative_semantic_source'] == 'explicit_exclusion'
    assert excluded['negative_structural_distance'] > 0.0
    assert excluded['negative_sampling_role'] == 'negative'
    assert ordinary['negative_semantic_target'] == 'unknown'
    assert ordinary['negative_semantic_source'] == 'unlabeled'


def test_explicit_exclusion_cannot_be_a_direct_positive(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(positive_is_explicit_exclusion=pl.lit(True))

    with pytest.raises(ValueError, match='direct positive.*explicit exclusion'):
        _validate_training_pairs(bad)
```

Use the ten-row `pair_facts_fixture` from `tests/fixtures/supervision.py`. Its expected values are hand-written and independent of `build_training_pairs`.

- [x] **Step 2: Run the training-pair tests and verify failure**

Run: `uv run pytest tests/unit/test_data_triplets.py -q`
Expected: FAIL because `build_training_pairs` does not exist and current code infers exclusion from distance `0`.

- [x] **Step 3: Replace sentinel-derived triplet semantics with explicit columns**

> Deviation: Implemented the legacy combinatorics over a directed anchor view with the legacy margin special cases and a deterministic cross-sector cap of 100, since the planned snippet contradicted its own prose (D2). Real-data diff: legacy lost reverse-published exclusions and used 1,479 excluded pairs as direct positives (D7).

Keep the current positive/negative combinatorics and anti-sampling cap, but feed them from pair facts and use this final projection:

```python
# src/naics_embedder/data/create_triplets.py
from naics_embedder.supervision.schema import (
    SamplingRole,
    SemanticSource,
    SemanticTarget,
)


def _semantic_negative_columns() -> list[pl.Expr]:
    return [
        pl.when(pl.col('negative_is_explicit_exclusion'))
        .then(pl.lit(SemanticTarget.UNRELATED.value))
        .otherwise(pl.lit(SemanticTarget.UNKNOWN.value))
        .alias('negative_semantic_target'),
        pl.when(pl.col('negative_is_explicit_exclusion'))
        .then(pl.lit(SemanticSource.EXPLICIT_EXCLUSION.value))
        .otherwise(pl.lit(SemanticSource.UNLABELED.value))
        .alias('negative_semantic_source'),
        pl.lit(SamplingRole.NEGATIVE.value).alias('negative_sampling_role'),
        pl.lit('generated_candidate').alias('negative_sampling_provenance'),
    ]


def _validate_training_pairs(training_pairs: pl.DataFrame) -> None:
    if training_pairs.filter(pl.col('positive_is_explicit_exclusion')).height:
        raise ValueError('direct positive cannot be an explicit exclusion')
    inconsistent = training_pairs.filter(
        pl.col('negative_is_explicit_exclusion').ne(
            pl.col('anchor_excludes_negative') | pl.col('negative_excludes_anchor')
        )
    )
    if inconsistent.height:
        raise ValueError('negative exclusion derivation is inconsistent')
    if training_pairs.select(
        pl.any_horizontal(
            pl.col('anchor_code_id').is_null(),
            pl.col('positive_code_id').is_null(),
            pl.col('negative_code_id').is_null(),
        ).any()
    ).item():
        raise ValueError('training pair contains an unmapped code identity')


def build_training_pairs(pair_facts: pl.DataFrame) -> pl.DataFrame:
    max_distance = pair_facts.get_column('structural_distance').max()
    positives = pair_facts.filter(
        pl.col('structural_distance').gt(0.0)
        & pl.col('structural_distance').ne(max_distance)
        & ~pl.col('is_explicit_exclusion')
    ).select(
        anchor_code_id=pl.col('code_i_id'),
        positive_code_id=pl.col('code_j_id'),
        anchor_code=pl.col('code_i'),
        positive_code=pl.col('code_j'),
        positive_structural_distance=pl.col('structural_distance'),
        positive_structural_relation_id=pl.col('structural_relation_id'),
        positive_structural_relation_name=pl.col('structural_relation_name'),
        positive_is_explicit_exclusion=pl.col('is_explicit_exclusion'),
    )
    negatives = pair_facts.select(
        anchor_code_id=pl.col('code_i_id'),
        negative_code_id=pl.col('code_j_id'),
        anchor_code=pl.col('code_i'),
        negative_code=pl.col('code_j'),
        negative_structural_distance=pl.col('structural_distance'),
        negative_structural_relation_id=pl.col('structural_relation_id'),
        negative_structural_relation_name=pl.col('structural_relation_name'),
        anchor_excludes_negative=pl.col('code_i_excludes_code_j'),
        negative_excludes_anchor=pl.col('code_j_excludes_code_i'),
        negative_is_explicit_exclusion=pl.col('is_explicit_exclusion'),
    )

    training_pairs = (
        positives.join(negatives, on=['anchor_code_id', 'anchor_code'], how='inner')
        .filter(
            pl.col('negative_code_id').ne(pl.col('positive_code_id')),
            pl.col('negative_code_id').ne(pl.col('anchor_code_id')),
        )
        .with_columns(
            [
                pl.lit(SemanticTarget.RELATED.value).alias('positive_semantic_target'),
                pl.lit(SemanticSource.TRAINING_POSITIVE.value).alias(
                    'positive_semantic_source'
                ),
                pl.lit(SamplingRole.POSITIVE.value).alias('positive_sampling_role'),
                pl.lit('generated_positive').alias('positive_sampling_provenance'),
                *_semantic_negative_columns(),
            ]
        )
        .with_columns(
            relation_margin=(
                pl.col('negative_structural_relation_id')
                - pl.col('positive_structural_relation_id')
            ).cast(pl.Float32),
            distance_margin=(
                pl.col('negative_structural_distance')
                - pl.col('positive_structural_distance')
            ).cast(pl.Float32),
        )
        .filter(
            pl.col('relation_margin').gt(0),
            pl.col('distance_margin').gt(0),
        )
        .with_columns(
            margin=(
                pl.col('relation_margin').mul(1.0 / 3.0)
                + pl.col('distance_margin').mul(2.0 / 3.0)
            ).pow(-1),
            anchor_idx=pl.col('anchor_code_id'),
            positive_idx=pl.col('positive_code_id'),
            negative_idx=pl.col('negative_code_id'),
            positive_distance=pl.col('positive_structural_distance'),
            negative_distance=pl.col('negative_structural_distance'),
            positive_relation=pl.col('positive_structural_relation_id'),
            negative_relation=pl.col('negative_structural_relation_id'),
            excluded=pl.col('negative_is_explicit_exclusion'),
            unrelated=pl.col('negative_semantic_target').eq(SemanticTarget.UNRELATED.value),
        )
        .sort('anchor_code_id', 'positive_code_id', 'negative_code_id')
    )
    _validate_training_pairs(training_pairs)
    return training_pairs
```

The positive predicate is intentionally derived inside `build_training_pairs`: positive structural distance, not the cross-sector maximum, and not an explicit exclusion. Keep the legacy compatibility columns shown above because the graph loader reads them; do not use them as repaired Stage-3 authorities.

- [x] **Step 4: Add exact schema and determinism assertions**

Extend `tests/unit/test_supervision_artifacts.py` to assert:

```python
expected_identity_columns = {
    'anchor_code_id',
    'positive_code_id',
    'negative_code_id',
    'anchor_code',
    'positive_code',
    'negative_code',
}
expected_supervision_columns = {
    'positive_structural_distance',
    'negative_structural_distance',
    'positive_structural_relation_id',
    'negative_structural_relation_id',
    'positive_semantic_target',
    'negative_semantic_target',
    'positive_semantic_source',
    'negative_semantic_source',
    'anchor_excludes_negative',
    'negative_excludes_anchor',
    'negative_is_explicit_exclusion',
}
assert expected_identity_columns <= set(training_pairs.columns)
assert expected_supervision_columns <= set(training_pairs.columns)
assert training_pairs.equals(build_training_pairs(pair_facts.clone()))
```

- [x] **Step 5: Run triplet and artifact tests**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py -q`
Expected: PASS, including direct-positive rejection and deterministic frame equality.

- [x] **Step 6: Commit semantic training pairs**

```bash
git add src/naics_embedder/data/create_triplets.py src/naics_embedder/data/supervision_bundle.py tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py
git commit -m "feat(data): encode explicit supervision semantics"
```

---

### Task 4: Materialize one immutable supervision bundle and publish its manifest last

**Files:**
- Create: `src/naics_embedder/supervision/artifacts.py`
- Modify: `src/naics_embedder/data/supervision_bundle.py`
- Modify: `src/naics_embedder/utils/config.py:213-345`
- Modify: `src/naics_embedder/cli/commands/data.py`
- Create: `conf/data/supervision.yaml`
- Modify: `tests/unit/test_supervision_artifacts.py`
- Modify: `tests/unit/test_cli_commands.py`
- Modify: `tests/fixtures/supervision.py`

**Interfaces:**
- Consumes: canonical frames from Tasks 2–3 and manifest models from Task 1.
- Produces: `write_versioned_parquet`, `write_versioned_dataset`, `sha256_file`, `generate_supervision_bundle(cfg: SupervisionBuildConfig) -> Path`, and the `naics-embedder data supervision` command.

- [x] **Step 1: Write failing immutable-publication tests**

```python
# add to tests/unit/test_supervision_artifacts.py
import json

import pyarrow.parquet as pq
import pytest

from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames
from naics_embedder.supervision.schema import CONTRACT_VERSION
```

```python
# add to tests/fixtures/supervision.py after Task 4's generator exists
from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames


@pytest.fixture
def generated_bundle(tmp_path, descriptions_fixture, pair_facts_fixture):
    return generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
```

```python
# continue tests/unit/test_supervision_artifacts.py
def test_bundle_writes_manifest_last_with_matching_parquet_metadata(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifest_path = generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )

    manifest = json.loads(manifest_path.read_text())
    codebook_path = manifest_path.parent / manifest['artifacts']['codebook']['path']
    metadata = pq.read_metadata(codebook_path).metadata

    assert manifest_path.name == 'manifest.json'
    assert manifest['contract_version'] == CONTRACT_VERSION
    assert metadata[b'naics_embedder.contract_version'].decode() == CONTRACT_VERSION
    assert metadata[b'naics_embedder.bundle_id'].decode() == 'bundle-a'
    assert metadata[b'naics_embedder.schema_version'].decode() == 'codebook-v1'


def test_bundle_never_overwrites_an_existing_generation(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    kwargs = {
        'output_root': tmp_path,
        'bundle_id': 'bundle-a',
        'generator_revision': 'revision-a',
        'naics_vintage': 2022,
        'descriptions': descriptions_fixture,
        'pair_facts': pair_facts_fixture,
    }
    generate_supervision_bundle_from_frames(**kwargs)

    with pytest.raises(FileExistsError, match='bundle-a'):
        generate_supervision_bundle_from_frames(**kwargs)


def test_failed_validation_publishes_no_manifest(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    inconsistent = pair_facts_fixture.with_columns(is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='broken',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=inconsistent,
        )

    assert not (tmp_path / 'broken' / 'manifest.json').exists()


def test_two_generated_bundles_have_equal_logical_frames_but_distinct_ids(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifests = [
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id=bundle_id,
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=pair_facts_fixture,
        )
        for bundle_id in ('bundle-a', 'bundle-b')
    ]
    loaded = [json.loads(path.read_text()) for path in manifests]
    frames = [
        pl.read_parquet(
            path.parent / manifest['artifacts']['pair_facts']['path']
        )
        for path, manifest in zip(manifests, loaded)
    ]

    assert loaded[0]['bundle_id'] != loaded[1]['bundle_id']
    assert frames[0].equals(frames[1])
```

- [x] **Step 2: Run the publication tests and verify failure**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py -q`
Expected: FAIL because versioned Parquet I/O and atomic bundle publication are absent.

- [x] **Step 3: Implement versioned Parquet writers and content hashes**

```python
# src/naics_embedder/supervision/artifacts.py
import hashlib
from pathlib import Path
from typing import Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from naics_embedder.supervision.schema import ArtifactFile

METADATA_CONTRACT = b'naics_embedder.contract_version'
METADATA_BUNDLE = b'naics_embedder.bundle_id'
METADATA_SCHEMA = b'naics_embedder.schema_version'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _table_with_contract_metadata(
    frame: pl.DataFrame,
    *,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> pa.Table:
    table = frame.to_arrow()
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            METADATA_CONTRACT: contract_version.encode(),
            METADATA_BUNDLE: bundle_id.encode(),
            METADATA_SCHEMA: schema_version.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


def write_versioned_parquet(
    frame: pl.DataFrame,
    path: Path,
    *,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> ArtifactFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _table_with_contract_metadata(
        frame,
        contract_version=contract_version,
        bundle_id=bundle_id,
        schema_version=schema_version,
    )
    pq.write_table(table, path)
    return ArtifactFile(path=path.name, sha256=sha256_file(path), row_count=frame.height)


def write_versioned_dataset(
    frame: pl.DataFrame,
    root: Path,
    *,
    partition_column: str,
    contract_version: str,
    bundle_id: str,
    schema_version: str,
) -> tuple[ArtifactFile, ...]:
    root.mkdir(parents=True, exist_ok=False)
    table = _table_with_contract_metadata(
        frame,
        contract_version=contract_version,
        bundle_id=bundle_id,
        schema_version=schema_version,
    )
    pq.write_to_dataset(
        table,
        root_path=root,
        partition_cols=[partition_column],
        basename_template='part-{i}.parquet',
    )
    members = []
    for path in sorted(root.glob('**/*.parquet')):
        members.append(
            ArtifactFile(
                path=path.relative_to(root.parent).as_posix(),
                sha256=sha256_file(path),
                row_count=pq.read_metadata(path).num_rows,
            )
        )
    return tuple(members)


def aggregate_fingerprint(files: Iterable[ArtifactFile]) -> str:
    payload = '\n'.join(f'{item.path}\t{item.sha256}\t{item.row_count}' for item in files)
    return hashlib.sha256(payload.encode()).hexdigest()
```

- [x] **Step 4: Implement atomic bundle orchestration**

> Deviation: Training pairs are written per `anchor=` partition with contract metadata on every member; the production description fingerprint is the descriptions file SHA-256 (D10); difficulty-threshold input files are sorted for determinism (D9).

Add `generate_supervision_bundle_from_frames` to `data/supervision_bundle.py`. It must:

1. Validate pair keys, exclusion OR, distinct-code nonzero distances, structural relation sentinels, direct-positive safety, and long-form/matrix reconciliation in memory.
2. Write `naics_codebook.parquet`, `naics_pair_facts.parquet`, the four compatibility distance/relation artifacts, `naics_training_pairs/`, and `curriculum_difficulty_thresholds.json` into `.<bundle-id>.staging`.
3. Record every file hash/row count plus total exclusion counts in `ArtifactRecord` values.
4. Create `manifest.json` only after all validations report `True`.
5. Atomically rename the staging directory to `<bundle-id>` and reject an existing final directory.

`generate_supervision_bundle(cfg)` is the production entry point. It computes `bundle_id = str(uuid.uuid4())`, stores `codebook_order=tuple(codebook['code'])`, fingerprints the exact descriptions and exclusion inputs, records the generator revision, relation map, every material generation parameter, and all validation results, then delegates to `generate_supervision_bundle_from_frames`. Only the frame-level test helper accepts an injected bundle ID. Because the bundle ID is embedded in Parquet metadata, determinism is asserted over logical frames (and input fingerprints), not byte-for-byte file hashes across separately generated bundle IDs.

Use these fixed artifact names and schema mappings:

```python
ARTIFACT_FILENAMES = {
    'codebook': 'naics_codebook.parquet',
    'pair_facts': 'naics_pair_facts.parquet',
    'distances': 'naics_distances.parquet',
    'distance_matrix': 'naics_distance_matrix.parquet',
    'relations': 'naics_relations.parquet',
    'relation_matrix': 'naics_relation_matrix.parquet',
    'training_pairs': 'naics_training_pairs',
    'difficulty_thresholds': 'curriculum_difficulty_thresholds.json',
}

ARTIFACT_SCHEMA_VERSIONS = {
    'codebook': CODEBOOK_SCHEMA_VERSION,
    'pair_facts': PAIR_FACTS_SCHEMA_VERSION,
    'distances': DISTANCES_SCHEMA_VERSION,
    'distance_matrix': DISTANCE_MATRIX_SCHEMA_VERSION,
    'relations': RELATIONS_SCHEMA_VERSION,
    'relation_matrix': RELATION_MATRIX_SCHEMA_VERSION,
    'training_pairs': TRAINING_PAIRS_SCHEMA_VERSION,
    'difficulty_thresholds': DIFFICULTY_THRESHOLDS_SCHEMA_VERSION,
}
```

Build compatibility long-form frames directly from pair facts:

```python
distances = pair_facts.select(
    idx_i=pl.col('code_i_id'),
    idx_j=pl.col('code_j_id'),
    code_i=pl.col('code_i'),
    code_j=pl.col('code_j'),
    distance=pl.col('structural_distance'),
    code_i_excludes_code_j=pl.col('code_i_excludes_code_j'),
    code_j_excludes_code_i=pl.col('code_j_excludes_code_i'),
    is_explicit_exclusion=pl.col('is_explicit_exclusion'),
)
relations = pair_facts.select(
    idx_i=pl.col('code_i_id'),
    idx_j=pl.col('code_j_id'),
    code_i=pl.col('code_i'),
    code_j=pl.col('code_j'),
    relation_id=pl.col('structural_relation_id'),
    relation=pl.col('structural_relation_name'),
    code_i_excludes_code_j=pl.col('code_i_excludes_code_j'),
    code_j_excludes_code_i=pl.col('code_j_excludes_code_i'),
    is_explicit_exclusion=pl.col('is_explicit_exclusion'),
)
```

For failure cleanup, remove only the fully resolved staging directory created by this invocation; never remove `output_root` or a final bundle directory.

- [x] **Step 5: Add build configuration and CLI orchestration**

> Deviation: Legacy stage wrappers, matrix builders, stats PDFs, and `__main__` blocks were removed; the stage commands print a migration notice and build the complete bundle (D8).

Add this model in `utils/config.py`:

```python
class SupervisionBuildConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    descriptions_parquet: str = './data/naics_descriptions.parquet'
    output_root: str = './data/supervision/stage3-supervision-v1'
    contract_version: Literal['stage3-supervision-v1'] = CONTRACT_VERSION
    naics_vintage: int = 2022
    relation_id: Dict[str, int] = Field(default_factory=lambda: {
        'child': 1,
        'sibling': 2,
        'grandchild': 3,
        'great-grandchild': 4,
        'nephew/niece': 5,
        'great-great-grandchild': 6,
        'cousin': 7,
        'grand-nephew/niece': 8,
        'grand-grand-nephew/niece': 9,
        'cousin_1_times_removed': 10,
        'second_cousin': 11,
        'cousin_2_times_removed': 12,
        'second_cousin_1_times_removed': 13,
        'third_cousin': 14,
        'cross_sector': 99,
    })
```

Create:

```yaml
# conf/data/supervision.yaml
descriptions_parquet: ./data/naics_descriptions.parquet
output_root: ./data/supervision/stage3-supervision-v1
contract_version: stage3-supervision-v1
naics_vintage: 2022
relation_id:
  child: 1
  sibling: 2
  grandchild: 3
  great-grandchild: 4
  nephew/niece: 5
  great-great-grandchild: 6
  cousin: 7
  grand-nephew/niece: 8
  grand-grand-nephew/niece: 9
  cousin_1_times_removed: 10
  second_cousin: 11
  cousin_2_times_removed: 12
  second_cousin_1_times_removed: 13
  third_cousin: 14
  cross_sector: 99
```

Add `data supervision`, make `data all` call `preprocess()` then `supervision()`, and make old `relations`, `distances`, and `triplets` commands print a migration notice and invoke the complete bundle builder rather than publishing partial authorities. The command prints the final manifest path.

- [x] **Step 6: Run publication and CLI tests**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_cli_commands.py tests/unit/test_config.py -q`
Expected: PASS; the CLI test confirms `data all` invokes one complete supervision build after preprocessing.

- [x] **Step 7: Commit immutable bundle generation**

```bash
git add conf/data/supervision.yaml src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/utils/config.py src/naics_embedder/cli/commands/data.py tests/fixtures/supervision.py tests/unit/test_supervision_artifacts.py tests/unit/test_cli_commands.py tests/unit/test_config.py
git commit -m "feat(data): publish immutable supervision bundles"
```

---

### Task 5: Validate bundles and expose anchor-relative supervision through one index

**Files:**
- Modify: `src/naics_embedder/supervision/artifacts.py`
- Create: `src/naics_embedder/supervision/index.py`
- Modify: `src/naics_embedder/supervision/__init__.py`
- Create: `tests/unit/test_supervision_index.py`
- Modify: `tests/unit/test_supervision_artifacts.py`
- Modify: `tests/fixtures/supervision.py`

**Interfaces:**
- Consumes: bundle layout and manifest from Task 4.
- Produces: `ValidatedSupervisionBundle`, `load_validated_bundle(manifest_path, expected_contract)`, `PairSupervision`, and `SupervisionIndex.from_bundle(bundle)` with `join(anchor_code_ids, candidate_code_ids, valid_mask)` and `exclusion_code_ids(anchor_code_id)`.

- [x] **Step 1: Write failing mixed-bundle and directional-join tests**

```python
# tests/unit/test_supervision_index.py
import json

import pyarrow.parquet as pq
import pytest
import torch

from naics_embedder.supervision.artifacts import (
    METADATA_BUNDLE,
    load_validated_bundle,
    sha256_file,
)
from naics_embedder.supervision.index import SupervisionIndex


def test_loader_rejects_mixed_bundle_metadata(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    member = manifest['artifacts']['pair_facts']['files'][0]
    pair_path = generated_bundle.parent / member['path']
    table = pq.read_table(pair_path)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_BUNDLE] = b'bundle-b'
    pq.write_table(table.replace_schema_metadata(metadata), pair_path)
    member['sha256'] = sha256_file(pair_path)
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*bundle-b'):
        load_validated_bundle(
            generated_bundle,
            expected_contract='stage3-supervision-v1',
        )


def test_join_maps_canonical_directions_into_anchor_view(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)
    anchor = torch.tensor([0, 3])
    candidate = torch.tensor([[2], [1]])
    valid = torch.ones((2, 1), dtype=torch.bool)

    joined = index.join(anchor, candidate, valid)

    assert joined.anchor_excludes_candidate.tolist() == [[True], [True]]
    assert joined.candidate_excludes_anchor.tolist() == [[False], [False]]
    assert torch.equal(
        joined.is_explicit_exclusion,
        joined.anchor_excludes_candidate | joined.candidate_excludes_anchor,
    )
    assert joined.structural_distance.tolist() == [[2.0], [99.0]]


def test_join_rejects_unknown_ids_with_anchor_context(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)

    with pytest.raises(ValueError, match='anchor row 0.*candidate code ID 999'):
        index.join(
            torch.tensor([0]),
            torch.tensor([[999]]),
            torch.tensor([[True]]),
        )
```

Extend the shared fixture plugin once the validated loader exists:

```python
# add to tests/fixtures/supervision.py
from naics_embedder.supervision.artifacts import load_validated_bundle


@pytest.fixture
def validated_bundle(generated_bundle):
    return load_validated_bundle(
        generated_bundle,
        expected_contract='stage3-supervision-v1',
    )
```

- [x] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/unit/test_supervision_index.py tests/unit/test_supervision_artifacts.py -q`
Expected: FAIL because validated bundle loading and `SupervisionIndex` do not exist.

- [x] **Step 3: Implement fail-closed bundle validation**

> Deviation: Shared validators live in `supervision/artifacts.py` and run at generation and load; the loader adds member row counts, long-form reconciliation, and training-pair checks (D11); member validation runs in bounded chunks (commit ade2d5d).

Add:

```python
# src/naics_embedder/supervision/artifacts.py
from dataclasses import dataclass
import json

from naics_embedder.supervision.schema import CONTRACT_VERSION, SupervisionManifest


@dataclass(frozen=True)
class ValidatedSupervisionBundle:
    root: Path
    manifest_path: Path
    manifest: SupervisionManifest

    def artifact_path(self, logical_name: str) -> Path:
        try:
            record = self.manifest.artifacts[logical_name]
        except KeyError as exc:
            raise ValueError(f'bundle has no {logical_name!r} artifact') from exc
        return self.root / record.path


def _parquet_contract(path: Path) -> tuple[str | None, str | None, str | None]:
    metadata = pq.read_metadata(path).metadata or {}
    decode = lambda key: metadata.get(key).decode() if metadata.get(key) else None
    return decode(METADATA_CONTRACT), decode(METADATA_BUNDLE), decode(METADATA_SCHEMA)


def load_validated_bundle(
    manifest_path: str | Path,
    expected_contract: str = CONTRACT_VERSION,
) -> ValidatedSupervisionBundle:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'supervision manifest not found: {path}')
    manifest = SupervisionManifest.model_validate_json(path.read_text())
    if manifest.contract_version != expected_contract:
        raise ValueError(
            f'expected supervision contract {expected_contract}, '
            f'found {manifest.contract_version} in {path}'
        )

    root = path.parent
    for logical_name, artifact in manifest.artifacts.items():
        member_rows = 0
        for member in artifact.files:
            member_path = root / member.path
            if not member_path.is_file():
                raise ValueError(f'{logical_name} artifact missing: {member_path}')
            actual_hash = sha256_file(member_path)
            if actual_hash != member.sha256:
                raise ValueError(
                    f'{logical_name} hash mismatch at {member_path}: '
                    f'expected {member.sha256}, found {actual_hash}'
                )
            member_rows += member.row_count
            if member_path.suffix == '.parquet':
                contract, bundle_id, schema_version = _parquet_contract(member_path)
                expected = (manifest.contract_version, manifest.bundle_id, artifact.schema_version)
                actual = (contract, bundle_id, schema_version)
                if actual != expected:
                    raise ValueError(
                        f'{logical_name} metadata mismatch: expected {expected}, found {actual}'
                    )
        if member_rows != artifact.row_count:
            raise ValueError(
                f'{logical_name} row count mismatch: '
                f'manifest={artifact.row_count}, members={member_rows}'
            )
    if not all(manifest.validation_results.values()):
        failed = sorted(k for k, passed in manifest.validation_results.items() if not passed)
        raise ValueError(f'bundle manifest records failed validations: {failed}')
    return ValidatedSupervisionBundle(root=root, manifest_path=path, manifest=manifest)
```

After hash/metadata checks, read codebook, pair facts, distance matrix, relation matrix, and training pairs and rerun these relational checks before returning:

- code IDs are contiguous, the manifest order equals the ordered codebook, and the codebook fingerprint matches;
- pair keys are unique and every code ID/string pair joins to the codebook;
- symmetric flag equals the two directional flags’ OR;
- matrices equal the long-form values and codebook order;
- no direct positive is excluded;
- no `0`/`excluded` structural sentinel exists;
- every training identity joins to both codebook and pair facts.

Each raised message includes the logical artifact name and bundle ID.

- [x] **Step 4: Implement the dense runtime index**

```python
# src/naics_embedder/supervision/index.py
from dataclasses import dataclass

import polars as pl
import torch

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.schema import SemanticSource, SemanticTarget

TARGET_TO_ID = {
    SemanticTarget.UNKNOWN: 0,
    SemanticTarget.RELATED: 1,
    SemanticTarget.UNRELATED: 2,
}
SOURCE_TO_ID = {
    SemanticSource.UNLABELED: 0,
    SemanticSource.TRAINING_POSITIVE: 1,
    SemanticSource.EXPLICIT_EXCLUSION: 2,
}


@dataclass(frozen=True)
class PairSupervision:
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor


@dataclass(frozen=True)
class SupervisionIndex:
    code_to_id: dict[str, int]
    id_to_code: tuple[str, ...]
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    directed_exclusion: torch.Tensor

    @classmethod
    def from_bundle(cls, bundle: ValidatedSupervisionBundle) -> 'SupervisionIndex':
        codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
        pair_facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
        size = codebook.height
        distance = torch.zeros((size, size), dtype=torch.float32)
        relation = torch.zeros((size, size), dtype=torch.int16)
        excludes = torch.zeros((size, size), dtype=torch.bool)
        for row in pair_facts.iter_rows(named=True):
            i, j = row['code_i_id'], row['code_j_id']
            distance[i, j] = distance[j, i] = row['structural_distance']
            relation[i, j] = relation[j, i] = row['structural_relation_id']
            excludes[i, j] = row['code_i_excludes_code_j']
            excludes[j, i] = row['code_j_excludes_code_i']
        codes = tuple(codebook.get_column('code').to_list())
        return cls(
            code_to_id={code: code_id for code_id, code in enumerate(codes)},
            id_to_code=codes,
            structural_distance=distance,
            structural_relation_id=relation,
            directed_exclusion=excludes,
        )

    def exclusion_code_ids(self, anchor_code_id: int) -> tuple[int, ...]:
        if not 0 <= anchor_code_id < len(self.id_to_code):
            raise ValueError(f'unknown anchor code ID {anchor_code_id}')
        symmetric = self.directed_exclusion[anchor_code_id] | self.directed_exclusion[:, anchor_code_id]
        return tuple(torch.where(symmetric)[0].tolist())

    def join(
        self,
        anchor_code_ids: torch.Tensor,
        candidate_code_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PairSupervision:
        if candidate_code_ids.shape != valid_mask.shape:
            raise ValueError('candidate code IDs and valid mask must share shape')
        if anchor_code_ids.shape != (candidate_code_ids.shape[0],):
            raise ValueError('one anchor code ID is required per candidate row')
        invalid_anchor = anchor_code_ids.lt(0) | anchor_code_ids.ge(len(self.id_to_code))
        if invalid_anchor.any():
            row = torch.where(invalid_anchor)[0][0].item()
            raise ValueError(
                f'anchor row {row} has unknown anchor code ID '
                f'{anchor_code_ids[row].item()}'
            )
        invalid = valid_mask & (
            candidate_code_ids.lt(0) | candidate_code_ids.ge(len(self.id_to_code))
        )
        if invalid.any():
            row, column = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            bad_id = candidate_code_ids[row, column].item()
            raise ValueError(f'anchor row {row} has unknown candidate code ID {bad_id}')

        safe_ids = candidate_code_ids.masked_fill(~valid_mask, 0).cpu()
        anchor = anchor_code_ids.cpu().unsqueeze(1).expand_as(safe_ids)
        anchor_excludes = self.directed_exclusion[anchor, safe_ids] & valid_mask.cpu()
        candidate_excludes = self.directed_exclusion[safe_ids, anchor] & valid_mask.cpu()
        explicit = anchor_excludes | candidate_excludes
        target = torch.where(
            explicit,
            torch.tensor(TARGET_TO_ID[SemanticTarget.UNRELATED], dtype=torch.int8),
            torch.tensor(TARGET_TO_ID[SemanticTarget.UNKNOWN], dtype=torch.int8),
        )
        source = torch.where(
            explicit,
            torch.tensor(SOURCE_TO_ID[SemanticSource.EXPLICIT_EXCLUSION], dtype=torch.int8),
            torch.tensor(SOURCE_TO_ID[SemanticSource.UNLABELED], dtype=torch.int8),
        )
        device = candidate_code_ids.device
        return PairSupervision(
            structural_distance=self.structural_distance[anchor, safe_ids].to(device),
            structural_relation_id=self.structural_relation_id[anchor, safe_ids].to(device),
            anchor_excludes_candidate=anchor_excludes.to(device),
            candidate_excludes_anchor=candidate_excludes.to(device),
            is_explicit_exclusion=explicit.to(device),
            semantic_target_id=target.to(device),
            semantic_source_id=source.to(device),
        )
```

- [x] **Step 5: Run bundle/index tests**

Run: `uv run pytest tests/unit/test_supervision_index.py tests/unit/test_supervision_artifacts.py -q`
Expected: PASS for mixed-bundle rejection, directional mapping, and actionable unknown-ID errors.

- [x] **Step 6: Commit bundle loading and runtime joins**

```bash
git add src/naics_embedder/supervision tests/fixtures/supervision.py tests/unit/test_supervision_index.py tests/unit/test_supervision_artifacts.py
git commit -m "feat(supervision): validate bundles and join pair facts"
```

---

### Task 6: Make candidate identity and checked selection immutable runtime types

**Files:**
- Create: `src/naics_embedder/supervision/candidates.py`
- Modify: `src/naics_embedder/supervision/__init__.py`
- Create: `tests/unit/test_candidate_contract.py`
- Modify: `tests/fixtures/supervision.py`

**Interfaces:**
- Consumes: `PairSupervision` and numeric enum values from Tasks 1 and 5.
- Produces: `CandidateEntityBatch`, `CandidateProposal`, `NegativeCandidateBatch`, `NegativeSelection`, and `SelectedNegativeBatch`. `NegativeCandidateBatch.select(selection)` is the only final gather.

- [x] **Step 1: Write failing permutation, stale-pool, and invalid-index tests**

```python
# tests/unit/test_candidate_contract.py
from dataclasses import replace

import pytest
import torch

from naics_embedder.supervision.candidates import NegativeSelection
from naics_embedder.supervision.schema import SelectionReason


def test_select_gathers_every_field_by_one_source_index(candidate_batch):
    indices = torch.tensor([[2, 0, 1]])
    selection = NegativeSelection(
        source_indices=indices,
        source_candidate_uid=candidate_batch.candidate_uid.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, 3)
        ),
        scores=torch.tensor([[0.9, 0.8, 0.7]]),
        reasons=torch.full((1, 3), SelectionReason.GEOMETRIC, dtype=torch.int8),
    )

    selected = candidate_batch.select(selection)

    assert selected.code_id.tolist() == [[103, 101, 102]]
    assert selected.structural_distance.tolist() == [[3.0, 1.0, 2.0]]
    assert selected.router_gate_probs[:, :, 0].tolist() == [[0.3, 0.1, 0.2]]
    assert selected.runtime_fields['difficulty'].tolist() == [[30.0, 10.0, 20.0]]
    assert torch.equal(selected.candidate_uid, selection.source_candidate_uid)


def test_select_rejects_uid_from_a_stale_pool(candidate_batch):
    indices = torch.tensor([[0]])
    wrong_uid = candidate_batch.candidate_uid[:, :1].clone()
    wrong_uid[0, 0, 2] += 1
    selection = NegativeSelection(
        source_indices=indices,
        source_candidate_uid=wrong_uid,
        scores=torch.ones((1, 1)),
        reasons=torch.full((1, 1), SelectionReason.BACKFILL, dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='UID mismatch.*row 0.*slot 0'):
        candidate_batch.select(selection)


def test_select_rejects_invalid_source_candidate(candidate_batch):
    invalid_batch = replace(
        candidate_batch,
        valid_mask=torch.tensor([[True, False, True]]),
    )
    selection = NegativeSelection(
        source_indices=torch.tensor([[1]]),
        source_candidate_uid=candidate_batch.candidate_uid[:, 1:2],
        scores=torch.ones((1, 1)),
        reasons=torch.full((1, 1), SelectionReason.BACKFILL, dtype=torch.int8),
    )

    with pytest.raises(ValueError, match='invalid source candidate'):
        invalid_batch.select(selection)
```

Add the exact reusable candidate fixtures:

```python
# add to tests/fixtures/supervision.py
import torch

from naics_embedder.supervision.candidates import NegativeCandidateBatch


def _negative_candidate_batch(
    code_ids: list[int],
    explicit_exclusions: list[bool],
) -> NegativeCandidateBatch:
    count = len(code_ids)
    shape = (1, count)
    slots = torch.arange(count, dtype=torch.long)
    candidate_uid = torch.stack(
        [torch.zeros_like(slots), torch.zeros_like(slots), slots], dim=-1
    ).unsqueeze(0)
    ordinal = torch.arange(1, count + 1, dtype=torch.float64).unsqueeze(0)
    anchor_excludes = torch.tensor([explicit_exclusions], dtype=torch.bool)
    candidate_excludes = torch.zeros(shape, dtype=torch.bool)
    explicit = anchor_excludes | candidate_excludes
    return NegativeCandidateBatch(
        candidate_uid=candidate_uid,
        code_id=torch.tensor([code_ids], dtype=torch.long),
        embedding=torch.stack([ordinal, ordinal + 0.5], dim=-1),
        structural_distance=ordinal.clone(),
        structural_relation_id=ordinal.to(torch.int16),
        anchor_excludes_candidate=anchor_excludes,
        candidate_excludes_anchor=candidate_excludes,
        is_explicit_exclusion=explicit,
        semantic_target_id=torch.where(explicit, 2, 0).to(torch.int8),
        semantic_source_id=torch.where(explicit, 2, 0).to(torch.int8),
        sampling_role_id=torch.full(shape, 2, dtype=torch.int8),
        sampling_provenance_id=torch.full(shape, 2, dtype=torch.int8),
        relation_margin=ordinal.clone(),
        distance_margin=ordinal.clone(),
        router_gate_probs=torch.stack(
            [ordinal / 10.0, 1.0 - ordinal / 10.0], dim=-1
        ),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        runtime_fields={'difficulty': ordinal * 10.0},
    )


@pytest.fixture
def candidate_batch() -> NegativeCandidateBatch:
    return _negative_candidate_batch([101, 102, 103], [False, False, False])


@pytest.fixture
def candidate_batch_with_exclusions() -> NegativeCandidateBatch:
    return _negative_candidate_batch(
        [20, 21, 22, 30, 31, 32],
        [True, True, True, False, False, False],
    )


@pytest.fixture
def candidate_batch_with_duplicate_code() -> NegativeCandidateBatch:
    return _negative_candidate_batch([101, 101, 102], [False, False, False])
```

- [x] **Step 2: Run the candidate contract tests and verify failure**

Run: `uv run pytest tests/unit/test_candidate_contract.py -q`
Expected: FAIL because the candidate types do not exist.

- [x] **Step 3: Implement shape validation and the canonical gather**

```python
# src/naics_embedder/supervision/candidates.py
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Optional

import torch


def _gather_aligned(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError('aligned candidate tensors require [batch, candidate, ...] dimensions')
    suffix = value.shape[2:]
    gather_index = indices.view(*indices.shape, *([1] * len(suffix))).expand(
        *indices.shape, *suffix
    )
    return value.gather(1, gather_index)


@dataclass(frozen=True)
class CandidateEntityBatch:
    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        batch_candidates = self.code_id.shape
        if self.code_id.ndim != 2:
            raise ValueError('code_id must have shape [batch, candidate]')
        if self.candidate_uid.shape != (*batch_candidates, 3):
            raise ValueError('candidate_uid must have shape [batch, candidate, 3]')
        if self.embedding.shape[:2] != batch_candidates:
            raise ValueError('embedding does not align with code_id')
        if self.valid_mask.shape != batch_candidates:
            raise ValueError('valid_mask does not align with code_id')
        if self.router_gate_probs is not None and self.router_gate_probs.shape[:2] != batch_candidates:
            raise ValueError('router_gate_probs does not align with code_id')
        if (self.valid_mask & self.code_id.lt(0)).any():
            raise ValueError('valid candidate entities require nonnegative code IDs')
        valid_uid = self.valid_mask.unsqueeze(-1).expand_as(self.candidate_uid)
        if (valid_uid & self.candidate_uid.lt(0)).any():
            raise ValueError('valid candidate entities require nonnegative UID components')


@dataclass(frozen=True)
class CandidateProposal:
    source_indices: torch.Tensor
    scores: torch.Tensor
    reason: int

    def __post_init__(self) -> None:
        if self.source_indices.shape != self.scores.shape:
            raise ValueError('proposal indices and scores must share shape')


@dataclass(frozen=True)
class NegativeSelection:
    source_indices: torch.Tensor
    source_candidate_uid: torch.Tensor
    scores: torch.Tensor
    reasons: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.source_indices.shape
        if self.source_indices.ndim != 2:
            raise ValueError('selection indices must have shape [batch, selected]')
        if self.source_candidate_uid.shape != (*shape, 3):
            raise ValueError('selection UIDs must have shape [batch, selected, 3]')
        if self.scores.shape != shape or self.reasons.shape != shape:
            raise ValueError('selection scores and reasons must align with indices')


@dataclass(frozen=True)
class SelectedNegativeBatch:
    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor
    sampling_role_id: torch.Tensor
    sampling_provenance_id: torch.Tensor
    relation_margin: torch.Tensor
    distance_margin: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    selection_scores: torch.Tensor
    selection_reasons: torch.Tensor
    runtime_fields: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class NegativeCandidateBatch:
    candidate_uid: torch.Tensor
    code_id: torch.Tensor
    embedding: torch.Tensor
    structural_distance: torch.Tensor
    structural_relation_id: torch.Tensor
    anchor_excludes_candidate: torch.Tensor
    candidate_excludes_anchor: torch.Tensor
    is_explicit_exclusion: torch.Tensor
    semantic_target_id: torch.Tensor
    semantic_source_id: torch.Tensor
    sampling_role_id: torch.Tensor
    sampling_provenance_id: torch.Tensor
    relation_margin: torch.Tensor
    distance_margin: torch.Tensor
    router_gate_probs: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    runtime_fields: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        shape = self.code_id.shape
        if self.code_id.ndim != 2:
            raise ValueError('candidate code IDs must have shape [batch, candidate]')
        if self.candidate_uid.shape != (*shape, 3):
            raise ValueError('candidate UID must have shape [batch, candidate, 3]')
        aligned_names = (
            'structural_distance',
            'structural_relation_id',
            'anchor_excludes_candidate',
            'candidate_excludes_anchor',
            'is_explicit_exclusion',
            'semantic_target_id',
            'semantic_source_id',
            'sampling_role_id',
            'sampling_provenance_id',
            'relation_margin',
            'distance_margin',
            'valid_mask',
        )
        for name in aligned_names:
            if getattr(self, name).shape != shape:
                raise ValueError(f'{name} does not align with candidate code IDs')
        if self.embedding.shape[:2] != shape:
            raise ValueError('embedding does not align with candidate code IDs')
        if self.router_gate_probs is not None and self.router_gate_probs.shape[:2] != shape:
            raise ValueError('router_gate_probs does not align with candidate code IDs')
        for name, value in self.runtime_fields.items():
            if value.shape[:2] != shape:
                raise ValueError(f'runtime field {name!r} does not align with candidates')
        if (self.valid_mask & self.code_id.lt(0)).any():
            raise ValueError('valid negative candidates require nonnegative code IDs')
        valid_uid = self.valid_mask.unsqueeze(-1).expand_as(self.candidate_uid)
        if (valid_uid & self.candidate_uid.lt(0)).any():
            raise ValueError('valid negative candidates require nonnegative UID components')
        if not torch.equal(
            self.is_explicit_exclusion,
            self.anchor_excludes_candidate | self.candidate_excludes_anchor,
        ):
            raise ValueError('is_explicit_exclusion must equal the directional OR')
        object.__setattr__(self, 'runtime_fields', MappingProxyType(dict(self.runtime_fields)))

    def select(self, selection: NegativeSelection) -> SelectedNegativeBatch:
        if selection.source_indices.shape[0] != self.code_id.shape[0]:
            raise ValueError('selection batch dimension does not match candidate pool')
        if selection.source_indices.lt(0).any() or selection.source_indices.ge(
            self.code_id.shape[1]
        ).any():
            raise ValueError('selection contains an out-of-bounds source index')
        selected_valid = _gather_aligned(self.valid_mask, selection.source_indices)
        if not selected_valid.all():
            row, slot = torch.nonzero(~selected_valid, as_tuple=False)[0].tolist()
            raise ValueError(f'selection references invalid source candidate at row {row}, slot {slot}')
        actual_uid = _gather_aligned(self.candidate_uid, selection.source_indices)
        mismatch = actual_uid.ne(selection.source_candidate_uid).any(dim=-1)
        if mismatch.any():
            row, slot = torch.nonzero(mismatch, as_tuple=False)[0].tolist()
            raise ValueError(f'selection UID mismatch at row {row}, slot {slot}')

        gathered = {}
        for item in fields(self):
            if item.name in {'runtime_fields', 'router_gate_probs'}:
                continue
            gathered[item.name] = _gather_aligned(
                getattr(self, item.name), selection.source_indices
            )
        router = None
        if self.router_gate_probs is not None:
            router = _gather_aligned(self.router_gate_probs, selection.source_indices)
        runtime = {
            name: _gather_aligned(value, selection.source_indices)
            for name, value in self.runtime_fields.items()
        }
        return SelectedNegativeBatch(
            **gathered,
            router_gate_probs=router,
            selection_scores=selection.scores,
            selection_reasons=selection.reasons,
            runtime_fields=MappingProxyType(runtime),
        )
```

- [x] **Step 4: Run candidate contract tests**

Run: `uv run pytest tests/unit/test_candidate_contract.py -q`
Expected: PASS, including the forced reorder `[2, 0, 1]` across every field.

- [x] **Step 5: Commit immutable candidate types**

```bash
git add src/naics_embedder/supervision/candidates.py src/naics_embedder/supervision/__init__.py tests/fixtures/supervision.py tests/unit/test_candidate_contract.py
git commit -m "feat(supervision): add checked candidate selection types"
```

---

### Task 7: Enforce rotating one-slot exclusion quota and deterministic code deduplication

**Files:**
- Create: `src/naics_embedder/supervision/selection.py`
- Modify: `src/naics_embedder/supervision/__init__.py`
- Create: `tests/unit/test_negative_selection.py`

**Interfaces:**
- Consumes: `CandidateProposal`, `NegativeCandidateBatch`, `NegativeSelection`, and `SelectionReason`.
- Produces: `stable_hash(global_seed: int, anchor_code_id: int) -> int` and `NegativeSelectionCoordinator.select(candidates, anchor_code_ids, positive_code_ids, k, epoch, global_seed, proposals) -> NegativeSelection`.

- [x] **Step 1: Write failing quota, rotation, deduplication, and capacity tests**

> Deviation: Fixed two plan test bugs: a mask-indexed UID shape and the capacity-error message (D12).

```python
# tests/unit/test_negative_selection.py
import pytest
import torch

from naics_embedder.supervision.candidates import CandidateProposal
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.supervision.selection import NegativeSelectionCoordinator


def test_quota_selects_exactly_one_exclusion_and_rotates(candidate_batch_with_exclusions):
    coordinator = NegativeSelectionCoordinator()
    chosen = []
    for epoch in range(3):
        selection = coordinator.select(
            candidate_batch_with_exclusions,
            anchor_code_ids=torch.tensor([10]),
            positive_code_ids=torch.tensor([11]),
            k=3,
            epoch=epoch,
            global_seed=7,
            proposals=(),
        )
        selected = candidate_batch_with_exclusions.select(selection)
        assert selected.is_explicit_exclusion.sum().item() == 1
        chosen.append(
            selected.code_id[selected.is_explicit_exclusion].item()
        )

    assert len(set(chosen)) == 3


def test_rotation_is_reproducible_for_same_seed_anchor_and_epoch(candidate_batch_with_exclusions):
    coordinator = NegativeSelectionCoordinator()
    args = {
        'candidates': candidate_batch_with_exclusions,
        'anchor_code_ids': torch.tensor([10]),
        'positive_code_ids': torch.tensor([11]),
        'k': 1,
        'epoch': 4,
        'global_seed': 123,
        'proposals': (),
    }

    first = coordinator.select(**args)
    second = coordinator.select(**args)

    assert torch.equal(first.source_indices, second.source_indices)
    assert first.reasons.item() == SelectionReason.EXCLUSION_QUOTA


def test_no_exclusion_uses_all_slots_for_ordinary_candidates(candidate_batch):
    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        epoch=0,
        global_seed=7,
        proposals=(),
    )
    selected = candidate_batch.select(selection)

    assert selected.code_id.unique().numel() == 3
    assert not selected.is_explicit_exclusion.any()


def test_proposal_ties_break_by_code_then_uid(candidate_batch):
    proposal = CandidateProposal(
        source_indices=torch.tensor([[2, 1, 0]]),
        scores=torch.tensor([[1.0, 1.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )
    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        epoch=0,
        global_seed=7,
        proposals=(proposal,),
    )
    assert candidate_batch.select(selection).code_id.tolist() == [[101, 102, 103]]


def test_geometric_then_router_merge_is_deterministic_and_code_unique(candidate_batch):
    geometric = CandidateProposal(
        source_indices=torch.tensor([[2, 1]]),
        scores=torch.tensor([[0.9, 0.8]]),
        reason=SelectionReason.GEOMETRIC,
    )
    router = CandidateProposal(
        source_indices=torch.tensor([[1, 0]]),
        scores=torch.tensor([[0.95, 0.7]]),
        reason=SelectionReason.ROUTER,
    )

    selection = NegativeSelectionCoordinator().select(
        candidate_batch,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=3,
        epoch=0,
        global_seed=7,
        proposals=(geometric, router),
    )

    assert candidate_batch.select(selection).code_id.tolist() == [[103, 102, 101]]
    assert selection.reasons.tolist() == [[
        SelectionReason.GEOMETRIC,
        SelectionReason.GEOMETRIC,
        SelectionReason.ROUTER,
    ]]


def test_duplicate_codes_collapse_to_smallest_occurrence_uid(candidate_batch_with_duplicate_code):
    selection = NegativeSelectionCoordinator().select(
        candidate_batch_with_duplicate_code,
        anchor_code_ids=torch.tensor([100]),
        positive_code_ids=torch.tensor([104]),
        k=2,
        epoch=0,
        global_seed=7,
        proposals=(),
    )
    selected = candidate_batch_with_duplicate_code.select(selection)

    assert selected.code_id.unique().numel() == 2
    duplicate_slot = selected.code_id.eq(101)
    assert selected.candidate_uid[duplicate_slot].tolist() == [0, 0, 0]


def test_insufficient_unique_candidates_is_fatal(candidate_batch):
    with pytest.raises(
        ValueError,
        match='anchor code ID 100.*requested 4.*available 3',
    ):
        NegativeSelectionCoordinator().select(
            candidate_batch,
            anchor_code_ids=torch.tensor([100]),
            positive_code_ids=torch.tensor([104]),
            k=4,
            epoch=0,
            global_seed=7,
            proposals=(),
        )
```

- [x] **Step 2: Run the selection tests and verify failure**

Run: `uv run pytest tests/unit/test_negative_selection.py -q`
Expected: FAIL because the coordinator does not exist.

- [x] **Step 3: Implement stable rotation and deterministic proposal merge**

> Deviation: After review, a NaN or +inf proposal score is fatal even in a proposal the merge never reaches; -inf remains the ineligible marker.

```python
# src/naics_embedder/supervision/selection.py
import hashlib
import math
import struct
from typing import Sequence

import torch

from naics_embedder.supervision.candidates import (
    CandidateProposal,
    NegativeCandidateBatch,
    NegativeSelection,
)
from naics_embedder.supervision.schema import SelectionReason


def stable_hash(global_seed: int, anchor_code_id: int) -> int:
    payload = struct.pack('>qq', global_seed, anchor_code_id)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big', signed=False)


def _uid_tuple(uid: torch.Tensor) -> tuple[int, int, int]:
    values = uid.detach().cpu().tolist()
    return int(values[0]), int(values[1]), int(values[2])


class NegativeSelectionCoordinator:
    def select(
        self,
        candidates: NegativeCandidateBatch,
        *,
        anchor_code_ids: torch.Tensor,
        positive_code_ids: torch.Tensor,
        k: int,
        epoch: int,
        global_seed: int,
        proposals: Sequence[CandidateProposal],
    ) -> NegativeSelection:
        if k < 1:
            raise ValueError('negative selection K must be at least one')
        batch_size = candidates.code_id.shape[0]
        if anchor_code_ids.shape != (batch_size,) or positive_code_ids.shape != (batch_size,):
            raise ValueError('anchor and positive code IDs must have one value per batch row')
        for proposal in proposals:
            if proposal.source_indices.shape[0] != batch_size:
                raise ValueError('proposal batch dimension does not match candidate batch')

        selected_rows = []
        score_rows = []
        reason_rows = []
        for row in range(batch_size):
            available_by_code: dict[int, int] = {}
            for index in torch.where(candidates.valid_mask[row])[0].tolist():
                code_id = int(candidates.code_id[row, index])
                if code_id in {int(anchor_code_ids[row]), int(positive_code_ids[row])}:
                    continue
                previous = available_by_code.get(code_id)
                if previous is None or _uid_tuple(candidates.candidate_uid[row, index]) < _uid_tuple(
                    candidates.candidate_uid[row, previous]
                ):
                    available_by_code[code_id] = index

            exclusion_codes = sorted(
                code_id
                for code_id, index in available_by_code.items()
                if bool(candidates.is_explicit_exclusion[row, index])
            )
            chosen: list[int] = []
            chosen_scores: list[float] = []
            chosen_reasons: list[int] = []
            chosen_codes: set[int] = set()

            if exclusion_codes:
                rotation = (stable_hash(global_seed, int(anchor_code_ids[row])) + epoch) % len(
                    exclusion_codes
                )
                reserved_code = exclusion_codes[rotation]
                reserved_index = available_by_code[reserved_code]
                chosen.append(reserved_index)
                chosen_scores.append(float('inf'))
                chosen_reasons.append(int(SelectionReason.EXCLUSION_QUOTA))
                chosen_codes.add(reserved_code)

            ordinary_codes = {
                code_id
                for code_id, index in available_by_code.items()
                if not bool(candidates.is_explicit_exclusion[row, index])
            }
            for proposal in proposals:
                best_score_by_code: dict[int, float] = {}
                for proposal_slot, index in enumerate(proposal.source_indices[row].tolist()):
                    if index < 0 or index >= candidates.code_id.shape[1]:
                        continue
                    if not bool(candidates.valid_mask[row, index]):
                        continue
                    code_id = int(candidates.code_id[row, index])
                    if code_id not in ordinary_codes or code_id in chosen_codes:
                        continue
                    score = float(proposal.scores[row, proposal_slot])
                    if not math.isfinite(score):
                        continue
                    best_score_by_code[code_id] = max(
                        score,
                        best_score_by_code.get(code_id, -math.inf),
                    )
                entries = []
                for code_id, score in best_score_by_code.items():
                    canonical_index = available_by_code[code_id]
                    entries.append(
                        (
                            -score,
                            code_id,
                            _uid_tuple(candidates.candidate_uid[row, canonical_index]),
                            canonical_index,
                            score,
                        )
                    )
                for _, code_id, _, index, score in sorted(entries):
                    if code_id in chosen_codes:
                        continue
                    chosen.append(index)
                    chosen_scores.append(score)
                    chosen_reasons.append(int(proposal.reason))
                    chosen_codes.add(code_id)
                    if len(chosen) == k:
                        break
                if len(chosen) == k:
                    break

            if len(chosen) < k:
                remaining = sorted(
                    (
                        code_id,
                        _uid_tuple(candidates.candidate_uid[row, index]),
                        index,
                    )
                    for code_id, index in available_by_code.items()
                    if code_id in ordinary_codes and code_id not in chosen_codes
                )
                for code_id, _, index in remaining:
                    chosen.append(index)
                    chosen_scores.append(float('-inf'))
                    chosen_reasons.append(int(SelectionReason.BACKFILL))
                    chosen_codes.add(code_id)
                    if len(chosen) == k:
                        break

            if len(chosen) != k:
                raise ValueError(
                    f'anchor code ID {int(anchor_code_ids[row])} requested {k} unique '
                    f'negative codes but only {len(available_by_code)} are available'
                )
            selected_rows.append(chosen)
            score_rows.append(chosen_scores)
            reason_rows.append(chosen_reasons)

        device = candidates.code_id.device
        source_indices = torch.tensor(selected_rows, dtype=torch.long, device=device)
        uid_index = source_indices.unsqueeze(-1).expand(-1, -1, 3)
        source_uid = candidates.candidate_uid.gather(1, uid_index)
        return NegativeSelection(
            source_indices=source_indices,
            source_candidate_uid=source_uid,
            scores=torch.tensor(score_rows, dtype=candidates.embedding.dtype, device=device),
            reasons=torch.tensor(reason_rows, dtype=torch.int8, device=device),
        )
```

- [x] **Step 4: Run selection and candidate contract tests**

Run: `uv run pytest tests/unit/test_negative_selection.py tests/unit/test_candidate_contract.py -q`
Expected: PASS, including cyclic coverage, `K = 1`, tie-breaks, stale-pool checks, and fatal capacity failure.

- [x] **Step 5: Commit deterministic selection**

```bash
git add src/naics_embedder/supervision/selection.py src/naics_embedder/supervision/__init__.py tests/unit/test_negative_selection.py
git commit -m "feat(supervision): enforce exclusion selection quota"
```

---

### Task 8: Collapse `negatives` and `all_candidates` into one bundle-backed collated pool

**Files:**
- Modify: `src/naics_embedder/text_model/dataloader/streaming_dataset.py:26-879`
- Modify: `src/naics_embedder/text_model/dataloader/datamodule.py:27-409` (after the upstream merge, including `Phase1MapDataset`)
- Modify: `src/naics_embedder/text_model/dataloader/difficulty_sampler.py`
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py`
- Modify: `src/naics_embedder/utils/config.py:382-480`
- Modify: `tests/unit/test_streaming_dataset.py`
- Modify: `tests/unit/test_streaming_sampling.py`
- Modify: `tests/unit/test_datamodule.py`
- Modify: `tests/unit/test_difficulty_sampler.py`
- Modify: `tests/unit/test_tokenization_cache.py`

**Interfaces:**
- Consumes: `ValidatedSupervisionBundle`, `SupervisionIndex`, `stable_hash`, and the new training-pair schema.
- Produces: repaired dataset items with one `candidate_pool` and `difficulty_proposal_indices`; collated `candidate_inputs` plus aligned tensor metadata; versioned streaming and token-cache envelopes.

- [x] **Step 1: Write failing collation and candidate-pool invariants**

Add these tests to `tests/unit/test_datamodule.py`:

```python
import copy
from typing import Any


@pytest.fixture
def make_repaired_batch_item():
    channels = ('title', 'description', 'excluded', 'examples')

    def encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            channel: {
                'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
                'attention_mask': torch.ones(2, dtype=torch.long),
            }
            for channel in channels
        }

    def make(
        candidate_code_ids: list[int],
        anchor_code_id: int = 100,
        positive_code_id: int = 104,
        positive_structural_distance: float = 1.0,
    ) -> dict[str, Any]:
        candidates = [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_embedding': encoded(code_id),
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in candidate_code_ids
        ]
        difficulty_order = sorted(
            range(len(candidate_code_ids)),
            key=lambda index: candidate_code_ids[index],
        )
        return {
            'anchor_code_id': anchor_code_id,
            'anchor_code': str(anchor_code_id),
            'anchor_embedding': encoded(anchor_code_id),
            'positive_code_id': positive_code_id,
            'positive_code': str(positive_code_id),
            'positive_embedding': encoded(positive_code_id),
            'positive_structural_distance': positive_structural_distance,
            'positive_structural_relation_id': 1,
            'candidate_pool': candidates,
            'difficulty_proposal_indices': difficulty_order,
            'selection_k': min(3, len(candidates)),
        }

    return make


def test_collate_does_not_mutate_input_and_uses_invalid_rows(make_repaired_batch_item):
    short = make_repaired_batch_item(candidate_code_ids=[101])
    long = make_repaired_batch_item(candidate_code_ids=[201, 202, 203])
    original_short = copy.deepcopy(short)

    batch = collate_fn([short, long], supervision_mode='repaired')

    assert len(short['candidate_pool']) == 1
    assert short['candidate_pool'][0]['negative_code_id'] == 101
    assert torch.equal(
        short['candidate_pool'][0]['negative_embedding']['title']['input_ids'],
        original_short['candidate_pool'][0]['negative_embedding']['title']['input_ids'],
    )
    assert batch['candidate_code_id'].tolist()[0] == [101, -1, -1]
    assert batch['candidate_valid_mask'].tolist()[0] == [True, False, False]
    assert batch['candidate_source_slot'].tolist()[0] == [0, -1, -1]
    assert batch['candidate_inputs']['title']['attention_mask'][1].count_nonzero() == 0
    assert batch['candidate_inputs']['title']['attention_mask'][2].count_nonzero() == 0


def test_collate_carries_every_candidate_field_in_one_order(make_repaired_batch_item):
    item = make_repaired_batch_item(candidate_code_ids=[103, 101, 102])

    batch = collate_fn([item], supervision_mode='repaired')

    assert batch['candidate_code_id'].tolist() == [[103, 101, 102]]
    assert batch['candidate_sampling_provenance_id'].tolist() == [[2, 2, 2]]
    assert batch['difficulty_proposal_indices'].tolist() == [[1, 2, 0]]
    assert batch['positive_code_id'].tolist() == [104]
    assert batch['positive_structural_distance'].tolist() == [1.0]
```

Add these tests to `tests/unit/test_streaming_sampling.py`:

```python
from typing import Any

import pytest
import torch

from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.text_model.dataloader.streaming_dataset import build_candidate_pool


@pytest.fixture
def pool_builder():
    def build(
        *,
        anchor_code_id: int,
        positive_code_id: int,
        raw_candidate_code_ids: list[int],
        exclusion_code_ids: tuple[int, ...],
        n_candidates: int,
        epoch: int,
    ) -> list[dict[str, Any]]:
        size = max(
            anchor_code_id,
            positive_code_id,
            *raw_candidate_code_ids,
            *exclusion_code_ids,
        ) + 4
        directed = torch.zeros((size, size), dtype=torch.bool)
        for code_id in exclusion_code_ids:
            directed[anchor_code_id, code_id] = True
        index = SupervisionIndex(
            code_to_id={str(code_id): code_id for code_id in range(size)},
            id_to_code=tuple(str(code_id) for code_id in range(size)),
            structural_distance=torch.full((size, size), 99.0),
            structural_relation_id=torch.full((size, size), 99, dtype=torch.int16),
            directed_exclusion=directed,
        )
        raw = [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_structural_distance': 99.0,
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in raw_candidate_code_ids
        ]
        return build_candidate_pool(
            anchor_code_id=anchor_code_id,
            positive_code_id=positive_code_id,
            raw_candidates=raw,
            supervision_index=index,
            n_candidates=n_candidates,
            final_k=n_candidates,
            epoch=epoch,
            seed=7,
        )

    return build


def test_candidate_pool_contains_every_exclusion_and_unique_ordinary_codes(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 12, 13, 14],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=2,
    )

    assert {20, 21, 22} <= {item['negative_code_id'] for item in pool}
    assert len({item['negative_code_id'] for item in pool}) == len(pool)


def test_candidate_pool_backfills_to_final_selection_capacity(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12],
        exclusion_code_ids=(),
        n_candidates=3,
        epoch=0,
    )

    assert len({item['negative_code_id'] for item in pool}) >= 3
```

Add these cache-contract tests:

```python
# add to tests/unit/test_tokenization_cache.py
def test_token_cache_rebuilds_when_description_fingerprint_changes(
    tokenization_config, monkeypatch
):
    builds = []

    def fake_build(*_args):
        builds.append(object())
        return {0: {'code': '111111'}}

    monkeypatch.setattr(
        'naics_embedder.text_model.dataloader.tokenization_cache._build_tokenization_cache',
        fake_build,
    )
    tokenization_cache(
        tokenization_config,
        description_fingerprint='a' * 64,
        codebook_fingerprint='b' * 64,
    )
    tokenization_cache(
        tokenization_config,
        description_fingerprint='c' * 64,
        codebook_fingerprint='b' * 64,
    )

    assert len(builds) == 2


# add to tests/unit/test_streaming_dataset.py
@pytest.mark.parametrize(
    'missing',
    ['contract_version', 'bundle_id', 'codebook_fingerprint'],
)
def test_repaired_streaming_cache_rejects_missing_identity(missing):
    envelope = {
        'contract_version': 'stage3-supervision-v1',
        'bundle_id': 'bundle-a',
        'codebook_fingerprint': 'a' * 64,
        'cache_schema_version': 'streaming-candidates-v1',
        'payload': [],
    }
    del envelope[missing]

    with pytest.raises(ValueError, match=missing):
        _validate_streaming_cache_envelope(
            envelope,
            expected_contract='stage3-supervision-v1',
            expected_bundle_id='bundle-a',
            expected_codebook_fingerprint='a' * 64,
        )
```

- [x] **Step 2: Run the affected data-loader suites and verify failure**

Run: `uv run pytest tests/unit/test_datamodule.py tests/unit/test_streaming_dataset.py tests/unit/test_streaming_sampling.py tests/unit/test_difficulty_sampler.py tests/unit/test_tokenization_cache.py -q`
Expected: FAIL because current collation mutates samples, repeats the last negative, and maintains parallel `negatives`/`all_candidates` paths.

- [x] **Step 3: Replace high exclusion weight with explicit pool construction**

> Deviation: The pool keeps every exclusion plus at least max(n - E, K - min(E, 1)) ordinary codes, because the planned sizing failed whenever an anchor had two or more exclusions (D15). After review, backfill is restricted to structurally eligible codes (D31).

In repaired mode, remove `exclusion_weight` from `_compute_phase1_weights` and `_sample_negatives_phase1`. Retain `phase1_exclusion_weight: Optional[float] = None` only for Task 13’s containment validator. Add:

```python
def build_candidate_pool(
    *,
    anchor_code_id: int,
    positive_code_id: int,
    raw_candidates: list[dict[str, Any]],
    supervision_index: SupervisionIndex,
    n_candidates: int,
    final_k: int,
    epoch: int,
    seed: int,
) -> list[dict[str, Any]]:
    if final_k < 1:
        raise ValueError('final negative count must be at least one')
    forbidden = {anchor_code_id, positive_code_id}
    exclusion_ids = tuple(
        code_id
        for code_id in supervision_index.exclusion_code_ids(anchor_code_id)
        if code_id not in forbidden
    )
    exclusion_set = set(exclusion_ids)

    def normalized_candidate(item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        code_id = int(normalized['negative_code_id'])
        normalized['negative_is_explicit_exclusion'] = code_id in exclusion_set
        return normalized

    def backfill_candidate(code_id: int) -> dict[str, Any]:
        return {
            'negative_code_id': code_id,
            'negative_code': supervision_index.id_to_code[code_id],
            'negative_structural_distance': float(
                supervision_index.structural_distance[anchor_code_id, code_id]
            ),
            'negative_is_explicit_exclusion': code_id in exclusion_set,
            'sampling_role_id': 2,
            'sampling_provenance_id': int(SamplingProvenance.BACKFILL),
        }

    by_code: dict[int, dict[str, Any]] = {}
    for item in raw_candidates:
        code_id = int(item['negative_code_id'])
        if code_id not in forbidden:
            by_code.setdefault(code_id, normalized_candidate(item))
    for code_id in exclusion_ids:
        by_code.setdefault(code_id, backfill_candidate(code_id))

    target_pool_size = max(n_candidates, final_k, len(exclusion_ids))
    rng = np.random.default_rng(
        (stable_hash(seed, anchor_code_id) + epoch) % (2**63)
    )
    ordinary_ids = sorted(
        code_id for code_id in by_code if code_id not in exclusion_set
    )
    rng.shuffle(ordinary_ids)
    kept_ids = list(exclusion_ids) + ordinary_ids[
        : max(0, target_pool_size - len(exclusion_ids))
    ]

    if len(set(kept_ids)) < target_pool_size:
        universe = [
            code_id
            for code_id in range(len(supervision_index.id_to_code))
            if code_id not in forbidden and code_id not in by_code
        ]
        rng.shuffle(universe)
        for code_id in universe:
            by_code[code_id] = backfill_candidate(code_id)
            kept_ids.append(code_id)
            if len(set(kept_ids)) == target_pool_size:
                break
    if len(set(kept_ids)) < final_k:
        raise ValueError(
            f'anchor code ID {anchor_code_id} requires {final_k} unique candidates; '
            f'only {len(set(kept_ids))} exist'
        )
    return [by_code[code_id] for code_id in dict.fromkeys(kept_ids)]
```

Load negative rows from the bundle’s `training_pairs` directory with their code IDs, raw structure, semantic fields, directions, role, provenance, and margins. Stop reconstructing exclusions from descriptions or structural values.

- [x] **Step 4: Convert upstream difficulty selection into an indexed proposal**

Rename `select_by_difficulty` to `propose_by_difficulty` and return source positions in score order:

```python
def _bucket_indexed_candidates(
    indexed: list[tuple[int, dict[str, Any]]],
) -> tuple[
    list[tuple[int, dict[str, Any]]],
    list[tuple[int, dict[str, Any]]],
    list[tuple[int, dict[str, Any]]],
]:
    easy: list[tuple[int, dict[str, Any]]] = []
    semi_hard: list[tuple[int, dict[str, Any]]] = []
    hard: list[tuple[int, dict[str, Any]]] = []
    for entry in indexed:
        distance = float(entry[1]['negative_structural_distance'])
        if distance >= 6.0:
            easy.append(entry)
        elif distance >= 4.0:
            semi_hard.append(entry)
        elif distance >= 3.0:
            hard.append(entry)
    return easy, semi_hard, hard


def propose_by_difficulty(
    *,
    candidates: list[dict[str, Any]],
    n_propose: int,
    epoch_progress: float,
    cfg: StreamingConfig,
    rng: np.random.Generator,
) -> list[int]:
    indexed = list(enumerate(candidates))
    ordinary = [
        (index, item)
        for index, item in indexed
        if not item['negative_is_explicit_exclusion']
    ]
    easy, semi_hard, hard = _bucket_indexed_candidates(ordinary)
    easy_ratio, semi_ratio, _ = _interpolate_ratios(epoch_progress, cfg)
    counts = (
        round(n_propose * easy_ratio),
        round(n_propose * semi_ratio),
    )
    counts = (counts[0], counts[1], n_propose - counts[0] - counts[1])
    selected: list[int] = []
    shortfall = 0
    for bucket, wanted in zip((easy, semi_hard, hard), counts):
        available = [index for index, _ in bucket if index not in selected]
        take = min(wanted + shortfall, len(available))
        if take:
            chosen = rng.choice(available, size=take, replace=False).tolist()
            selected.extend(int(index) for index in chosen)
        shortfall = wanted + shortfall - take
    remaining = [index for index, _ in ordinary if index not in selected]
    if shortfall and remaining:
        selected.extend(
            int(index)
            for index in rng.choice(
                remaining, size=min(shortfall, len(remaining)), replace=False
            ).tolist()
        )
    return selected
```

`Phase1MapDataset.__getitem__` now returns one `candidate_pool` plus `difficulty_proposal_indices`. The precomputed path returns the same shape with every valid pool index as its default proposal. Do not emit repaired `negatives` or `all_candidates` keys.

- [x] **Step 5: Make repaired collation non-mutating and explicit about invalid rows**

> Deviation: The manifest is validated lazily in prepare_data/setup; the positive sampler uses the bundle codebook and drops explicit-exclusion siblings; legacy collation stays non-mutating for containment (D3, D16).

Implement `collate_fn(batch, supervision_mode='repaired')` with local copies. For each channel, flatten candidate inputs in row-major `[batch, candidate]` order, and create invalid rows with zero `input_ids` and zero `attention_mask`. Emit:

```python
result = {
    'anchor': anchor_batch,
    'positive': positive_batch,
    'candidate_inputs': candidate_inputs,
    'batch_size': len(batch),
    'k_candidates': max_candidates,
    'selection_k': selection_k,
    'anchor_code_id': torch.tensor(anchor_code_ids, dtype=torch.long),
    'positive_code_id': torch.tensor(positive_code_ids, dtype=torch.long),
    'positive_structural_distance': torch.tensor(positive_distances, dtype=torch.float32),
    'positive_structural_relation_id': torch.tensor(positive_relations, dtype=torch.int16),
    'candidate_code_id': torch.tensor(candidate_code_ids, dtype=torch.long),
    'candidate_valid_mask': torch.tensor(candidate_valid_mask, dtype=torch.bool),
    'candidate_source_slot': torch.tensor(candidate_source_slots, dtype=torch.long),
    'candidate_sampling_role_id': torch.tensor(candidate_roles, dtype=torch.int8),
    'candidate_sampling_provenance_id': torch.tensor(candidate_provenance, dtype=torch.int8),
    'difficulty_proposal_indices': torch.tensor(difficulty_indices, dtype=torch.long),
    'anchor_code': anchor_codes,
    'positive_code': positive_codes,
}
```

Pair-dependent candidate supervision is deliberately absent here; `SupervisionIndex.join` adds it after optional distributed entity gathering.

- [x] **Step 6: Version streaming and tokenization caches**

Store each pickle as:

```python
{
    'contract_version': bundle.manifest.contract_version,
    'bundle_id': bundle.manifest.bundle_id,
    'codebook_fingerprint': bundle.manifest.codebook_fingerprint,
    'cache_schema_version': 'streaming-candidates-v1',
    'payload': rows,
}
```

Validate before returning cached rows:

```python
def _validate_streaming_cache_envelope(
    envelope: dict[str, Any],
    *,
    expected_contract: str,
    expected_bundle_id: str,
    expected_codebook_fingerprint: str,
) -> list[dict[str, Any]]:
    expected = {
        'contract_version': expected_contract,
        'bundle_id': expected_bundle_id,
        'codebook_fingerprint': expected_codebook_fingerprint,
        'cache_schema_version': 'streaming-candidates-v1',
    }
    for field, expected_value in expected.items():
        if field not in envelope:
            raise ValueError(f'repaired streaming cache lacks {field}')
        if envelope[field] != expected_value:
            raise ValueError(
                f'repaired streaming cache {field} mismatch: '
                f'expected {expected_value!r}, found {envelope[field]!r}'
            )
    payload = envelope.get('payload')
    if not isinstance(payload, list):
        raise ValueError('repaired streaming cache payload must be a list')
    return payload
```

Include those four values plus every sampling/difficulty parameter in cache keys. Change the token-cache entry point to `tokenization_cache(cfg, *, description_fingerprint: str, codebook_fingerprint: str, use_locking: bool = True)`. It writes a JSON sidecar containing those fingerprints, tokenizer name, and max length and reuses the tensor cache only on an exact sidecar match. In repaired mode, stale or unversioned streaming caches are fatal and tokenization caches are regenerated because their source text is independently reproducible.

For relation-, triplet-, and curriculum-derived caches, store the relevant manifest member hashes in the envelope and invalidate whenever any recorded source hash differs. Never accept an old path merely because its filename matches a rebuilt artifact.

- [x] **Step 7: Run the repaired data-loader suites**

Run: `uv run pytest tests/unit/test_datamodule.py tests/unit/test_streaming_dataset.py tests/unit/test_streaming_sampling.py tests/unit/test_difficulty_sampler.py tests/unit/test_tokenization_cache.py -q`
Expected: PASS; no test expects repeated padding, input mutation, high exclusion weights, or dual candidate paths.

- [x] **Step 8: Commit the canonical candidate input path**

```bash
git add src/naics_embedder/text_model/dataloader src/naics_embedder/utils/config.py tests/unit/test_datamodule.py tests/unit/test_streaming_dataset.py tests/unit/test_streaming_sampling.py tests/unit/test_difficulty_sampler.py tests/unit/test_tokenization_cache.py
git commit -m "refactor(data): collate one aligned candidate pool"
```

---

### Task 9: Return indexed mining proposals and recompute supervision after distributed gather

**Files:**
- Modify: `src/naics_embedder/text_model/hard_negative_mining.py:76-413`
- Modify: `src/naics_embedder/text_model/mixins/distributed.py:20-176`
- Modify: `src/naics_embedder/text_model/mixins/curriculum.py:91-373`
- Modify: `tests/unit/test_hard_negative_mining.py`
- Modify: `tests/unit/test_naics_model.py`
- Create: `tests/integration/test_distributed_supervision.py`

**Interfaces:**
- Consumes: `CandidateEntityBatch`, `NegativeCandidateBatch`, `CandidateProposal`, `SupervisionIndex`, and selection coordinator.
- Produces: `LorentzianHardNegativeMiner.propose`, `RouterGuidedNegativeMiner.propose`, `gather_candidate_entities`, and `CurriculumMixin._select_negative_batch(...) -> SelectedNegativeBatch`.

- [x] **Step 1: Write failing miner-boundary tests**

```python
# add to tests/unit/test_hard_negative_mining.py
from naics_embedder.supervision.candidates import CandidateEntityBatch
from naics_embedder.supervision.index import SupervisionIndex


def test_geometric_miner_returns_source_indices_not_embeddings(candidate_batch):
    anchor = candidate_batch.embedding[:, 0]
    proposal = LorentzianHardNegativeMiner().propose(anchor, candidate_batch, k=2)

    assert proposal.source_indices.shape == (1, 2)
    assert proposal.reason == SelectionReason.GEOMETRIC
    assert not hasattr(proposal, 'embedding')


def test_router_miner_indices_recover_matching_gate_rows(candidate_batch):
    anchor_gate = torch.tensor([[0.9, 0.1]])
    proposal = RouterGuidedNegativeMiner().propose(
        anchor_gate_probs=anchor_gate,
        candidates=candidate_batch,
        k=2,
    )
    selected_gates = candidate_batch.router_gate_probs.gather(
        1,
        proposal.source_indices.unsqueeze(-1).expand(-1, -1, 2),
    )

    assert selected_gates.shape == (1, 2, 2)
    assert proposal.reason == SelectionReason.ROUTER


def test_gathered_entity_is_rejoined_for_each_local_anchor(validated_bundle):
    index = SupervisionIndex.from_bundle(validated_bundle)
    gathered = CandidateEntityBatch(
        candidate_uid=torch.tensor([[[1, 0, 0]]]),
        code_id=torch.tensor([[2]]),
        embedding=torch.tensor([[[0.25, 0.75]]]),
        router_gate_probs=torch.tensor([[[0.4, 0.6]]]),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    active_code_ids = gathered.code_id.expand(2, -1)
    active_valid = gathered.valid_mask.expand(2, -1)

    joined = index.join(
        anchor_code_ids=torch.tensor([0, 1]),
        candidate_code_ids=active_code_ids,
        valid_mask=active_valid,
    )

    assert joined.structural_distance.tolist() == [[2.0], [3.0]]
    assert joined.is_explicit_exclusion.tolist() == [[True], [False]]
```

- [x] **Step 2: Write the CPU two-rank integration test**

```python
# tests/integration/test_distributed_supervision.py
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from naics_embedder.supervision.candidates import CandidateEntityBatch
from naics_embedder.text_model.mixins.distributed import gather_candidate_entities


def make_entity_batch(rank: int, code_id: int) -> CandidateEntityBatch:
    return CandidateEntityBatch(
        candidate_uid=torch.tensor([[[rank, 0, 0]]], dtype=torch.long),
        code_id=torch.tensor([[code_id]], dtype=torch.long),
        embedding=torch.tensor([[[float(rank), float(code_id)]]]),
        router_gate_probs=torch.tensor([[[0.25, 0.75]]]),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )


def _worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
    )
    try:
        local = make_entity_batch(rank=rank, code_id=rank + 1)
        gathered = gather_candidate_entities(local)
        queue.put(
            (
                rank,
                gathered.code_id.squeeze(0).cpu().tolist(),
                gathered.candidate_uid.squeeze(0).cpu().tolist(),
                gathered.valid_mask.squeeze(0).cpu().tolist(),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_gather_preserves_intrinsic_identity_only(tmp_path):
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(_worker, args=(2, init_file, queue), nprocs=2, join=True)
    results = sorted(queue.get() for _ in range(2))

    assert results[0][1] == results[1][1] == [1, 2]
    assert results[0][2][0][0] == 0
    assert results[0][2][1][0] == 1
    assert results[0][3] == [True, True]
```

Do not add structural distance, relation, or exclusion fields to `make_entity_batch`: their absence is the distributed contract.

- [x] **Step 3: Run mining and distributed tests and verify failure**

Run: `uv run pytest tests/unit/test_hard_negative_mining.py tests/integration/test_distributed_supervision.py -q`
Expected: FAIL because miners return gathered tensors and distributed gathering has no entity contract.

- [x] **Step 4: Change both miners to score and propose indices**

> Deviation: The embedding-returning `mine_*` paths and `GlobalNegativeContext` were removed in Task 11 with the step rewrite rather than here (D17).

```python
# representative replacement in text_model/hard_negative_mining.py
def propose(
    self,
    anchor_emb: torch.Tensor,
    candidates: NegativeCandidateBatch,
    k: int,
) -> CandidateProposal:
    distances = self.lorentz_distance.batched_forward(anchor_emb, candidates.embedding)
    masked_scores = (-distances).masked_fill(
        ~candidates.valid_mask | candidates.is_explicit_exclusion,
        -torch.inf,
    )
    k_actual = min(k, masked_scores.shape[1])
    scores, indices = torch.topk(masked_scores, k=k_actual, dim=1, largest=True)
    return CandidateProposal(
        source_indices=indices,
        scores=scores.detach(),
        reason=SelectionReason.GEOMETRIC,
    )
```

Router `propose` computes confusion scores from `candidates.router_gate_probs`, applies the same validity/exclusion mask, and returns `CandidateProposal(reason=SelectionReason.ROUTER)`. Delete public paths that return selected candidate embeddings.

- [x] **Step 5: Gather only intrinsic entity tensors**

Replace `GlobalNegativeContext` with `gather_candidate_entities(local: CandidateEntityBatch) -> CandidateEntityBatch`. Use `torch.distributed.nn.functional.all_gather` for differentiable embeddings and fixed-shape `dist.all_gather` for code IDs, UID triples, validity, and router probabilities. Flatten rank-major results to one candidate axis. Assert equal embedding/router widths and pad variable entity counts with invalid rows before gathering.

The gathered type contains exactly:

```python
CandidateEntityBatch(
    candidate_uid=gathered_uid.unsqueeze(0),
    code_id=gathered_code_id.unsqueeze(0),
    embedding=gathered_embedding.unsqueeze(0),
    router_gate_probs=(
        None if gathered_router is None else gathered_router.unsqueeze(0)
    ),
    valid_mask=gathered_valid.unsqueeze(0),
)
```

Do not gather structural distance, relation, margins, semantic IDs, or exclusion flags.

- [x] **Step 6: Rebuild candidate supervision for each local anchor and select once**

> Deviation: The planned order put the difficulty proposal first, which crowded out mining whenever the pool held K or more candidates. After review, miners go first, `router_mix_ratio` splits their slots, miners score one canonical occurrence per code, and the difficulty proposal is the fallback (D30). Ordinary candidates must be structurally farther than the positive (D31).

In `CurriculumMixin`:

1. Build local UID triples as `[origin_rank, batch_row, source_slot]`.
2. Encode the one candidate pool.
3. Gather `CandidateEntityBatch` only when geometric or router mining is enabled under multi-rank training.
4. Expand gathered entities for each local anchor.
5. Call `SupervisionIndex.join(local_anchor_code_ids, gathered_code_ids, valid_mask)`.
6. Derive relation/distance margins against each row’s positive structural values.
7. Create one `NegativeCandidateBatch`.
8. Convert difficulty, geometric, and router results into `CandidateProposal` values.
9. Call `NegativeSelectionCoordinator.select` once and then `candidate_batch.select` once.

Use UID lookup to translate each collated local difficulty proposal into the active pool. This is
necessary because rank-major global gathering changes source positions while preserving occurrence
identity:

```python
# add to text_model/mixins/curriculum.py imports
from naics_embedder.supervision.candidates import (
    CandidateEntityBatch,
    CandidateProposal,
    NegativeCandidateBatch,
    SelectedNegativeBatch,
)
from naics_embedder.supervision.schema import SamplingProvenance, SelectionReason
from naics_embedder.text_model.mixins.distributed import gather_candidate_entities


def _proposal_from_local_uids(
    *,
    local_candidate_uid: torch.Tensor,
    active_candidates: NegativeCandidateBatch,
    local_source_indices: torch.Tensor,
) -> CandidateProposal:
    if local_source_indices.ndim != 2:
        raise ValueError('difficulty proposal indices must have shape [batch, proposal]')
    if local_source_indices.ge(local_candidate_uid.shape[1]).any():
        raise ValueError('difficulty proposal contains an out-of-bounds local source index')
    safe = local_source_indices.clamp_min(0)
    local_uids = local_candidate_uid.gather(
        1,
        safe.unsqueeze(-1).expand(-1, -1, 3),
    )
    matches = active_candidates.candidate_uid.unsqueeze(2).eq(
        local_uids.unsqueeze(1)
    ).all(dim=-1)
    expected = local_source_indices.ge(0)
    match_count = matches.sum(dim=1)
    if (match_count[expected] != 1).any():
        raise ValueError('difficulty proposal UID is missing or duplicated in active pool')
    active_indices = matches.to(torch.int64).argmax(dim=1).masked_fill(~expected, -1)
    width = local_source_indices.shape[1]
    scores = torch.arange(
        width,
        0,
        -1,
        dtype=active_candidates.embedding.dtype,
        device=active_candidates.embedding.device,
    ).unsqueeze(0).expand_as(active_indices)
    scores = scores.masked_fill(~expected, -torch.inf)
    return CandidateProposal(
        source_indices=active_indices,
        scores=scores,
        reason=SelectionReason.DIFFICULTY,
    )


def _select_negative_batch(
    self,
    *,
    batch: dict[str, Any],
    anchor_output: dict[str, torch.Tensor],
    candidate_output: dict[str, torch.Tensor],
    candidate_uid: torch.Tensor,
    batch_idx: int,
) -> SelectedNegativeBatch:
    batch_size = int(batch['batch_size'])
    candidate_count = int(batch['k_candidates'])
    embedding = candidate_output['embedding'].reshape(batch_size, candidate_count, -1)
    raw_gate_probs = candidate_output.get('gate_probs')
    gate_probs = (
        None
        if raw_gate_probs is None
        else raw_gate_probs.reshape(batch_size, candidate_count, -1)
    )
    local_entities = CandidateEntityBatch(
        candidate_uid=candidate_uid,
        code_id=batch['candidate_code_id'],
        embedding=embedding,
        router_gate_probs=gate_probs,
        valid_mask=batch['candidate_valid_mask'],
    )

    enable_geometric = self.current_curriculum_flags.get(
        'enable_hard_negative_mining', False
    )
    enable_router = self.current_curriculum_flags.get(
        'enable_router_guided_sampling', False
    )
    use_global = (
        (enable_geometric or enable_router)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    if use_global:
        gathered = gather_candidate_entities(local_entities)
        entities = CandidateEntityBatch(
            candidate_uid=gathered.candidate_uid.expand(batch_size, -1, -1),
            code_id=gathered.code_id.expand(batch_size, -1),
            embedding=gathered.embedding.expand(batch_size, -1, -1),
            router_gate_probs=(
                None
                if gathered.router_gate_probs is None
                else gathered.router_gate_probs.expand(batch_size, -1, -1)
            ),
            valid_mask=gathered.valid_mask.expand(batch_size, -1),
        )
        sampling_role_id = torch.full_like(entities.code_id, 2, dtype=torch.int8)
        sampling_provenance_id = torch.full_like(
            entities.code_id,
            int(SamplingProvenance.DISTRIBUTED_POOL),
            dtype=torch.int8,
        )
    else:
        entities = local_entities
        sampling_role_id = batch['candidate_sampling_role_id']
        sampling_provenance_id = batch['candidate_sampling_provenance_id']

    pair = self.supervision_index.join(
        batch['anchor_code_id'],
        entities.code_id,
        entities.valid_mask,
    )
    relation_margin = pair.structural_relation_id.to(torch.float32) - batch[
        'positive_structural_relation_id'
    ].to(torch.float32).unsqueeze(1)
    distance_margin = pair.structural_distance - batch[
        'positive_structural_distance'
    ].unsqueeze(1)
    candidates = NegativeCandidateBatch(
        candidate_uid=entities.candidate_uid,
        code_id=entities.code_id,
        embedding=entities.embedding,
        structural_distance=pair.structural_distance,
        structural_relation_id=pair.structural_relation_id,
        anchor_excludes_candidate=pair.anchor_excludes_candidate,
        candidate_excludes_anchor=pair.candidate_excludes_anchor,
        is_explicit_exclusion=pair.is_explicit_exclusion,
        semantic_target_id=pair.semantic_target_id,
        semantic_source_id=pair.semantic_source_id,
        sampling_role_id=sampling_role_id,
        sampling_provenance_id=sampling_provenance_id,
        relation_margin=relation_margin,
        distance_margin=distance_margin,
        router_gate_probs=entities.router_gate_probs,
        valid_mask=entities.valid_mask,
        runtime_fields={'difficulty': pair.structural_distance},
    )

    proposals: list[CandidateProposal] = [
        _proposal_from_local_uids(
            local_candidate_uid=candidate_uid,
            active_candidates=candidates,
            local_source_indices=batch['difficulty_proposal_indices'],
        )
    ]
    selection_k = int(batch['selection_k'])
    if enable_geometric:
        proposals.append(
            self.hard_negative_miner.propose(
                anchor_output['embedding'],
                candidates,
                k=selection_k,
            )
        )
    if enable_router:
        anchor_gate_probs = anchor_output.get('gate_probs')
        if anchor_gate_probs is None or candidates.router_gate_probs is None:
            raise ValueError('router-guided selection requires anchor and candidate gate probabilities')
        proposals.append(
            self.router_guided_miner.propose(
                anchor_gate_probs=anchor_gate_probs,
                candidates=candidates,
                k=selection_k,
            )
        )

    selection = self.selection_coordinator.select(
        candidates,
        anchor_code_ids=batch['anchor_code_id'],
        positive_code_ids=batch['positive_code_id'],
        k=selection_k,
        epoch=int(self.current_epoch),
        global_seed=int(getattr(self.hparams, 'seed', 0)),
        proposals=tuple(proposals),
    )
    return candidates.select(selection)
```

Remove `_perform_hard_negative_mining`, `_apply_router_guided_sampling`, and `_prepare_negative_embeddings` once their tests target the new boundary. No candidate field is independently indexed after `NegativeCandidateBatch` exists.

- [x] **Step 7: Run local and distributed selection suites**

Run: `uv run pytest tests/unit/test_hard_negative_mining.py tests/unit/test_candidate_contract.py tests/unit/test_negative_selection.py tests/integration/test_distributed_supervision.py -q`
Expected: PASS; the two-rank test proves occurrence UID preservation and unit tests prove local-anchor supervision is recomputed.

- [x] **Step 8: Commit indexed and distributed mining**

```bash
git add src/naics_embedder/text_model/hard_negative_mining.py src/naics_embedder/text_model/mixins/distributed.py src/naics_embedder/text_model/mixins/curriculum.py tests/unit/test_hard_negative_mining.py tests/unit/test_naics_model.py tests/integration/test_distributed_supervision.py
git commit -m "fix(training): mine by checked candidate indices"
```

---

### Task 10: Replace LambdaRank and make contrastive/false-negative masks exclusion-safe

**Files:**
- Modify: `src/naics_embedder/text_model/loss.py:19-601`
- Modify: `src/naics_embedder/text_model/false_negative_strategies.py`
- Modify: `src/naics_embedder/text_model/mixins/loss.py:24-473`
- Modify: `tests/unit/test_loss.py`
- Modify: `tests/unit/test_false_negative_strategy.py`

**Interfaces:**
- Consumes: `SelectedNegativeBatch` from Task 6.
- Produces: `effective_false_negative_mask`, `structural_preference_from_distances`, and `StructuralPreferenceLoss`. `HyperbolicInfoNCELoss.forward` accepts selected embeddings, validity, exclusion flags, and pseudo-related mask.

- [x] **Step 1: Replace LambdaRank scalar tests with direct gradient-contract tests**

Delete `TestLambdaRankLoss` and add:

```python
# tests/unit/test_loss.py
from naics_embedder.text_model.loss import structural_preference_from_distances


def test_structural_preference_gradient_corrects_an_inversion():
    learned = torch.tensor([[4.0, 1.0]], requires_grad=True)
    structural = torch.tensor([[1.0, 3.0]])
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=structural,
        candidate_code_ids=torch.tensor([[11, 12]]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.zeros((1, 2), dtype=torch.bool),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
    )

    loss.backward()

    assert learned.grad[0, 0] > 0
    assert learned.grad[0, 1] < 0


def test_structural_preference_is_lower_for_correct_order():
    kwargs = {
        'structural_distances': torch.tensor([[1.0, 3.0]]),
        'candidate_code_ids': torch.tensor([[11, 12]]),
        'anchor_code_ids': torch.tensor([10]),
        'is_explicit_exclusion': torch.zeros((1, 2), dtype=torch.bool),
        'valid_mask': torch.ones((1, 2), dtype=torch.bool),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    correct = structural_preference_from_distances(
        learned_distances=torch.tensor([[1.0, 4.0]]), **kwargs
    )
    inverted = structural_preference_from_distances(
        learned_distances=torch.tensor([[4.0, 1.0]]), **kwargs
    )

    assert correct < inverted


@pytest.mark.parametrize(
    ('structural', 'explicit', 'valid', 'codes'),
    [
        ([2.0, 2.0], [False, False], [True, True], [11, 12]),
        ([1.0, 3.0], [True, False], [True, True], [11, 12]),
        ([1.0, 3.0], [False, False], [True, False], [11, 12]),
        ([1.0, 3.0], [False, False], [True, True], [10, 12]),
        ([1.0, 3.0], [False, False], [True, True], [11, 11]),
    ],
)
def test_structural_preference_returns_differentiable_zero_when_fully_masked(
    structural, explicit, valid, codes
):
    learned = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=torch.tensor([structural]),
        candidate_code_ids=torch.tensor([codes]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.tensor([explicit]),
        valid_mask=torch.tensor([valid]),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.equal(learned.grad, torch.zeros_like(learned))


def test_structural_preference_detaches_importance_weights():
    learned = torch.tensor([[3.0, 1.0]], requires_grad=True)
    weights = torch.tensor([[2.0]], requires_grad=True)
    loss = structural_preference_from_distances(
        learned_distances=learned,
        structural_distances=torch.tensor([[1.0, 3.0]]),
        candidate_code_ids=torch.tensor([[11, 12]]),
        anchor_code_ids=torch.tensor([10]),
        is_explicit_exclusion=torch.zeros((1, 2), dtype=torch.bool),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        margin=0.2,
        temperature=1.0,
        tie_tolerance=1e-6,
        pair_weights=weights,
    )
    loss.backward()

    assert weights.grad is None
```

```python
def test_structural_preference_is_invariant_to_joint_candidate_permutation():
    kwargs = {
        'learned_distances': torch.tensor([[3.0, 1.0, 2.0]]),
        'structural_distances': torch.tensor([[1.0, 3.0, 5.0]]),
        'candidate_code_ids': torch.tensor([[11, 12, 13]]),
        'anchor_code_ids': torch.tensor([10]),
        'is_explicit_exclusion': torch.tensor([[False, False, False]]),
        'valid_mask': torch.tensor([[True, True, True]]),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    original = structural_preference_from_distances(**kwargs)
    permutation = torch.tensor([2, 0, 1])
    permuted = structural_preference_from_distances(
        **{
            **kwargs,
            'learned_distances': kwargs['learned_distances'][:, permutation],
            'structural_distances': kwargs['structural_distances'][:, permutation],
            'candidate_code_ids': kwargs['candidate_code_ids'][:, permutation],
            'is_explicit_exclusion': kwargs['is_explicit_exclusion'][:, permutation],
            'valid_mask': kwargs['valid_mask'][:, permutation],
        }
    )

    assert torch.allclose(original, permuted)


def test_structural_preference_normalizes_each_anchor_before_batch_mean():
    kwargs = {
        'learned_distances': torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 9.0]]),
        'structural_distances': torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 7.0]]),
        'candidate_code_ids': torch.tensor([[11, 12, 13], [21, 22, 23]]),
        'anchor_code_ids': torch.tensor([10, 20]),
        'is_explicit_exclusion': torch.zeros((2, 3), dtype=torch.bool),
        'valid_mask': torch.tensor([[True, True, True], [True, True, False]]),
        'margin': 0.2,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    combined = structural_preference_from_distances(**kwargs)
    individual = []
    for row in range(2):
        individual.append(
            structural_preference_from_distances(
                **{
                    key: value[row : row + 1]
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in kwargs.items()
                }
            )
        )

    assert torch.allclose(combined, torch.stack(individual).mean())
```

- [x] **Step 2: Write failing exclusion-precedence contrastive tests**

```python
# add to tests/unit/test_false_negative_strategy.py
from naics_embedder.text_model.loss import effective_false_negative_mask


def test_explicit_exclusion_overrides_pseudo_related_mask():
    pseudo_related = torch.tensor([[True, True, False]])
    explicit = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])

    effective = effective_false_negative_mask(pseudo_related, explicit, valid)

    assert effective.tolist() == [[False, True, False]]


def test_attraction_uses_only_valid_non_exclusion_pairs():
    anchor = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    pseudo_related = torch.tensor([[True, True, True]])
    explicit = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])

    updated_mask, attraction = apply_false_negative_strategy(
        FalseNegativeConfig(strategy='attract', attraction_metric='cosine'),
        anchor,
        negatives,
        pseudo_related,
        explicit_exclusion_mask=explicit,
        valid_mask=valid,
    )

    assert updated_mask is None
    assert attraction is not None and torch.isfinite(attraction)
```

```python
def _lorentz_point(value: float) -> torch.Tensor:
    spatial = torch.tensor([value])
    return torch.cat([torch.sqrt(1.0 + spatial.square()), spatial])


def test_explicit_pseudo_related_negative_remains_in_contrastive_denominator():
    loss_fn = HyperbolicInfoNCELoss(embedding_dim=1, temperature=0.5, curvature=1.0)
    anchor = _lorentz_point(0.0).unsqueeze(0)
    positive = _lorentz_point(0.1).unsqueeze(0)
    common = {
        'valid_mask': torch.tensor([[True]]),
        'is_explicit_exclusion': torch.tensor([[True]]),
        'pseudo_related_mask': torch.tensor([[True]]),
    }
    near_loss = loss_fn(
        anchor,
        positive,
        _lorentz_point(0.2).view(1, 1, -1),
        **common,
    )
    far_loss = loss_fn(
        anchor,
        positive,
        _lorentz_point(2.0).view(1, 1, -1),
        **common,
    )

    assert torch.isfinite(near_loss) and torch.isfinite(far_loss)
    assert not torch.allclose(near_loss, far_loss)


def test_invalid_padding_cannot_change_contrastive_loss_or_gradients():
    loss_fn = HyperbolicInfoNCELoss(embedding_dim=1, temperature=0.5, curvature=1.0)

    def run(invalid_value: float):
        anchor = _lorentz_point(0.0).unsqueeze(0).requires_grad_()
        positive = _lorentz_point(0.1).unsqueeze(0).requires_grad_()
        negatives = torch.stack(
            [_lorentz_point(0.8), _lorentz_point(invalid_value)]
        ).unsqueeze(0).requires_grad_()
        loss = loss_fn(
            anchor,
            positive,
            negatives,
            valid_mask=torch.tensor([[True, False]]),
            is_explicit_exclusion=torch.tensor([[False, False]]),
            pseudo_related_mask=None,
        )
        gradients = torch.autograd.grad(loss, (anchor, positive, negatives))
        return loss.detach(), tuple(gradient.detach() for gradient in gradients)

    first_loss, first_gradients = run(2.0)
    second_loss, second_gradients = run(20.0)

    assert torch.allclose(first_loss, second_loss)
    assert torch.allclose(first_gradients[0], second_gradients[0])
    assert torch.allclose(first_gradients[1], second_gradients[1])
    assert torch.allclose(first_gradients[2][:, 0], second_gradients[2][:, 0])
    assert torch.count_nonzero(first_gradients[2][:, 1]) == 0
    assert torch.count_nonzero(second_gradients[2][:, 1]) == 0
```

- [x] **Step 3: Run loss tests and verify the old objective fails**

Run: `uv run pytest tests/unit/test_loss.py tests/unit/test_false_negative_strategy.py -q`
Expected: FAIL because `structural_preference_from_distances` and exclusion-aware masks are absent.

- [x] **Step 4: Implement the pairwise structural-preference primitive**

> Deviation: Per-anchor normalization divides by the exact detached weight sum instead of `clamp_min(1.0)`; structural distances keep their own dtype.

```python
# src/naics_embedder/text_model/loss.py
import torch.nn.functional as F


def effective_false_negative_mask(
    pseudo_related: torch.Tensor | None,
    is_explicit_exclusion: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    if is_explicit_exclusion.shape != valid_mask.shape:
        raise ValueError('exclusion and validity masks must share shape')
    if pseudo_related is None:
        return None
    if pseudo_related.shape != valid_mask.shape:
        raise ValueError('pseudo-related mask must align with selected candidates')
    return pseudo_related & ~is_explicit_exclusion & valid_mask


def structural_preference_from_distances(
    *,
    learned_distances: torch.Tensor,
    structural_distances: torch.Tensor,
    candidate_code_ids: torch.Tensor,
    anchor_code_ids: torch.Tensor,
    is_explicit_exclusion: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
    temperature: float,
    tie_tolerance: float,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError('structural preference temperature must be positive')
    if margin < 0 or tie_tolerance < 0:
        raise ValueError('structural preference margin and tie tolerance must be nonnegative')
    shape = learned_distances.shape
    aligned = (
        structural_distances,
        candidate_code_ids,
        is_explicit_exclusion,
        valid_mask,
    )
    if learned_distances.ndim != 2 or any(value.shape != shape for value in aligned):
        raise ValueError('structural preference candidate tensors must share [batch, candidate] shape')
    if anchor_code_ids.shape != (shape[0],):
        raise ValueError('structural preference requires one anchor code ID per row')

    count = shape[1]
    left, right = torch.triu_indices(count, count, offset=1, device=learned_distances.device)
    learned_left = learned_distances[:, left]
    learned_right = learned_distances[:, right]
    structural_left = structural_distances[:, left]
    structural_right = structural_distances[:, right]
    structural_delta = structural_left - structural_right

    closer_left = structural_delta < -tie_tolerance
    closer_right = structural_delta > tie_tolerance
    learned_close = torch.where(closer_left, learned_left, learned_right)
    learned_far = torch.where(closer_left, learned_right, learned_left)

    duplicate_matrix = candidate_code_ids.unsqueeze(2).eq(candidate_code_ids.unsqueeze(1))
    seen_before = torch.tril(duplicate_matrix, diagonal=-1).any(dim=2)
    unique_occurrence = ~seen_before
    non_self = candidate_code_ids.ne(anchor_code_ids.unsqueeze(1))
    candidate_eligible = (
        valid_mask & ~is_explicit_exclusion & unique_occurrence & non_self
    )
    pair_mask = (
        candidate_eligible[:, left]
        & candidate_eligible[:, right]
        & (closer_left | closer_right)
        & candidate_code_ids[:, left].ne(candidate_code_ids[:, right])
    )

    penalties = F.softplus((learned_close - learned_far + margin) / temperature)
    if pair_weights is None:
        weights = torch.ones_like(penalties)
    else:
        if pair_weights.shape != penalties.shape:
            raise ValueError('pair importance weights must align with unordered comparisons')
        weights = pair_weights.detach().to(penalties)
    weights = weights * pair_mask
    denominator = weights.sum(dim=1)
    per_anchor = (penalties * weights).sum(dim=1) / denominator.clamp_min(1.0)
    contributing = denominator.gt(0)
    if not contributing.any():
        return learned_distances.sum() * 0.0
    return per_anchor[contributing].mean()
```

- [x] **Step 5: Add the module wrapper over selected candidates**

> Deviation: `RankOrderPreservationLoss` was deleted too (no remaining importer); the docs API page became `structural_preference_loss.md` (D18).

```python
class StructuralPreferenceLoss(nn.Module):
    def __init__(
        self,
        *,
        curvature: float,
        margin: float,
        temperature: float,
        tie_tolerance: float,
        weight: float,
    ):
        super().__init__()
        self.lorentz_distance = LorentzDistance(curvature)
        self.margin = margin
        self.temperature = temperature
        self.tie_tolerance = tie_tolerance
        self.weight = weight

    def forward(
        self,
        *,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        anchor_code_id: torch.Tensor,
        positive_code_id: torch.Tensor,
        positive_structural_distance: torch.Tensor,
        selected: SelectedNegativeBatch,
        pair_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positive_learned = self.lorentz_distance(anchor_emb, positive_emb).unsqueeze(1)
        negative_learned = self.lorentz_distance.batched_forward(
            anchor_emb, selected.embedding
        )
        learned = torch.cat([positive_learned, negative_learned], dim=1)
        structural = torch.cat(
            [positive_structural_distance.unsqueeze(1), selected.structural_distance], dim=1
        )
        code_ids = torch.cat([positive_code_id.unsqueeze(1), selected.code_id], dim=1)
        explicit = torch.cat(
            [
                torch.zeros_like(positive_code_id.unsqueeze(1), dtype=torch.bool),
                selected.is_explicit_exclusion,
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                torch.ones_like(positive_code_id.unsqueeze(1), dtype=torch.bool),
                selected.valid_mask,
            ],
            dim=1,
        )
        return self.weight * structural_preference_from_distances(
            learned_distances=learned,
            structural_distances=structural,
            candidate_code_ids=code_ids,
            anchor_code_ids=anchor_code_id,
            is_explicit_exclusion=explicit,
            valid_mask=valid,
            margin=self.margin,
            temperature=self.temperature,
            tie_tolerance=self.tie_tolerance,
            pair_weights=pair_weights,
        )
```

Delete `LambdaRankLoss` and its active imports. `RankOrderPreservationLoss` may remain only if another non-Stage-3 public API still imports it; it must not be initialized or called by the repaired path.

- [x] **Step 6: Make contrastive and attraction loss validity-aware**

> Deviation: Rows with no eligible negative are dropped before `logsumexp` (NaN-safe), and the strategy returns the effective mask.

Change `HyperbolicInfoNCELoss.forward` to accept negative embeddings shaped `[batch, selected, dim]` plus `valid_mask`, `is_explicit_exclusion`, and `pseudo_related_mask`. Compute:

```python
effective_false_negative = effective_false_negative_mask(
    pseudo_related_mask,
    is_explicit_exclusion,
    valid_mask,
)
eligible = valid_mask
if effective_false_negative is not None:
    eligible = eligible & ~effective_false_negative
neg_similarities = neg_similarities.masked_fill(~eligible, -torch.inf)
has_negative = eligible.any(dim=1)
per_anchor = -pos_similarities + torch.logsumexp(neg_similarities, dim=1)
if not has_negative.any():
    return (anchor_emb.sum() + positive_emb.sum() + negative_embs.sum()) * 0.0
return per_anchor[has_negative].mean()
```

Change `apply_false_negative_strategy` to receive explicit and validity masks and internally derive the same effective mask before elimination or attraction. Shape mismatches raise `ValueError`.

- [x] **Step 7: Replace loss-mixin LambdaRank wiring and remove warning fallbacks**

Replace the repaired-path wrappers with:

```python
def _apply_false_negative_strategy_wrapper(
    self,
    anchor_emb: torch.Tensor,
    selected: SelectedNegativeBatch,
    pseudo_related: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if self.false_negative_config is None:
        return (
            effective_false_negative_mask(
                pseudo_related,
                selected.is_explicit_exclusion,
                selected.valid_mask,
            ),
            None,
        )
    return apply_false_negative_strategy(
        self.false_negative_config,
        anchor_emb,
        selected.embedding,
        pseudo_related,
        explicit_exclusion_mask=selected.is_explicit_exclusion,
        valid_mask=selected.valid_mask,
    )


def _compute_structural_preference_loss(
    self,
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    batch: dict[str, Any],
    selected: SelectedNegativeBatch,
) -> torch.Tensor:
    return self.structural_preference_loss_fn(
        anchor_emb=anchor_emb,
        positive_emb=positive_emb,
        anchor_code_id=batch['anchor_code_id'],
        positive_code_id=batch['positive_code_id'],
        positive_structural_distance=batch['positive_structural_distance'],
        selected=selected,
    )
```

Rename the combined term and log key to `train/structural_preference_loss`. Remove broad `try/except` blocks from hierarchy, structural preference, and false-negative contract boundaries so alignment or bundle errors abort the step.

- [x] **Step 8: Run all loss tests**

Run: `uv run pytest tests/unit/test_loss.py tests/unit/test_false_negative_strategy.py -q`
Expected: PASS for gradient direction, order quality, permutation invariance, masking, detached weights, per-anchor normalization, exclusion precedence, and padding.

- [x] **Step 9: Commit the corrected loss contract**

```bash
git add src/naics_embedder/text_model/loss.py src/naics_embedder/text_model/false_negative_strategies.py src/naics_embedder/text_model/mixins/loss.py tests/unit/test_loss.py tests/unit/test_false_negative_strategy.py
git commit -m "fix(loss): replace LambdaRank with structural preference"
```

---

### Task 11: Route the Stage-3 training step through one selected batch

**Files:**
- Modify: `src/naics_embedder/text_model/naics_model.py:68-470`
- Modify: `src/naics_embedder/text_model/mixins/curriculum.py`
- Modify: `src/naics_embedder/text_model/mixins/loss.py`
- Modify: `src/naics_embedder/text_model/mixins/logging.py`
- Modify: `tests/unit/test_naics_model.py`
- Create: `tests/integration/test_stage3_training_step.py`

**Interfaces:**
- Consumes: bundle/index, collated pool, selected batch, and repaired losses from Tasks 5–10.
- Produces: `NAICSContrastiveModel._forward_candidate_pool`, post-selection pseudo-related handling, health counters, and one canonical repaired `training_step`.

- [x] **Step 1: Write the forced-reorder full-step regression**

> Deviation: `router_first_column` is compared approximately because the stub gates are float32 (D22).

```python
# tests/integration/test_stage3_training_step.py
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import naics_embedder.text_model.naics_model as model_module
from naics_embedder.supervision.candidates import NegativeSelection, SelectedNegativeBatch
from naics_embedder.supervision.schema import SelectionReason
from naics_embedder.text_model.dataloader.datamodule import collate_fn


def forced_selection(order: list[int]):
    def select(candidates, **_kwargs) -> NegativeSelection:
        indices = torch.tensor(
            [order],
            dtype=torch.long,
            device=candidates.code_id.device,
        ).expand(candidates.code_id.shape[0], -1)
        return NegativeSelection(
            source_indices=indices,
            source_candidate_uid=candidates.candidate_uid.gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 3),
            ),
            scores=torch.ones_like(indices, dtype=candidates.embedding.dtype),
            reasons=torch.full_like(
                indices,
                int(SelectionReason.GEOMETRIC),
                dtype=torch.int8,
            ),
        )

    return select


@dataclass
class SelectionSpyLoss:
    contrastive_uids: list[list[list[int]]] | None = None
    structural_uids: list[list[list[int]]] | None = None
    code_ids: list[int] | None = None
    structural_distances: list[float] | None = None
    exclusion_flags: list[bool] | None = None
    router_first_column: list[float] | None = None
    false_negative_flags: list[bool] | None = None

    def contrastive(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        selected: SelectedNegativeBatch,
        effective_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self.contrastive_uids = selected.candidate_uid.detach().cpu().tolist()
        self.false_negative_flags = (
            None if effective_mask is None else effective_mask[0].cpu().tolist()
        )
        return (
            anchor_emb.square().mean()
            + positive_emb.square().mean()
            + selected.embedding.square().mean()
        )

    def structural(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        batch: dict,
        selected: SelectedNegativeBatch,
    ) -> torch.Tensor:
        self.structural_uids = selected.candidate_uid.detach().cpu().tolist()
        self.code_ids = selected.code_id[0].cpu().tolist()
        self.structural_distances = selected.structural_distance[0].cpu().tolist()
        self.exclusion_flags = selected.is_explicit_exclusion[0].cpu().tolist()
        self.router_first_column = selected.router_gate_probs[0, :, 0].cpu().tolist()
        eligible = selected.valid_mask & ~selected.is_explicit_exclusion
        return (
            selected.embedding[eligible].square().mean()
            + anchor_emb.square().mean()
            + positive_emb.square().mean()
        ) * 0.01


class StubMultiChannelEncoder(nn.Module):
    embedding_dim = 2

    def __init__(self, **_kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, channel_inputs):
        raw = channel_inputs['title']['input_ids'][:, 0].to(torch.float32)
        value = raw * self.scale
        spatial = torch.stack([value, value / 2.0], dim=1)
        time = torch.sqrt(1.0 + spatial.square().sum(dim=1, keepdim=True))
        embedding = torch.cat([time, spatial], dim=1)
        first_gate = ((raw - 1.0) / 10.0).clamp(0.0, 1.0)
        gate_probs = torch.stack([first_gate, 1.0 - first_gate], dim=1)
        return {
            'embedding': embedding,
            'embedding_euc': spatial,
            'gate_probs': gate_probs,
            'top_k_indices': gate_probs.argmax(dim=1, keepdim=True),
        }


def _encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        channel: {
            'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
            'attention_mask': torch.ones(2, dtype=torch.long),
        }
        for channel in ('title', 'description', 'excluded', 'examples')
    }


def _repaired_item(candidate_ids: list[int]) -> dict:
    return {
        'anchor_code_id': 0,
        'anchor_code': '111111',
        'anchor_embedding': _encoded(0),
        'positive_code_id': 1,
        'positive_code': '111112',
        'positive_embedding': _encoded(1),
        'positive_structural_distance': 0.5,
        'positive_structural_relation_id': 1,
        'candidate_pool': [
            {
                'negative_code_id': code_id,
                'negative_code': str(code_id),
                'negative_embedding': _encoded(code_id),
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in candidate_ids
        ],
        'difficulty_proposal_indices': list(range(len(candidate_ids))),
        'selection_k': 3,
    }


@pytest.fixture
def repaired_training_batch():
    batch = collate_fn(
        [
            _repaired_item([2, 3, 4]),
            _repaired_item([2, 3, 4, 4]),
        ],
        supervision_mode='repaired',
    )
    assert batch['candidate_valid_mask'][0].tolist() == [True, True, True, False]
    return batch


@pytest.fixture
def tiny_repaired_model(monkeypatch, generated_bundle):
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)
    model = model_module.NAICSContrastiveModel(
        base_model_name='test-stub',
        num_experts=2,
        top_k=1,
        moe_hidden_dim=4,
        hierarchy_weight=0.0,
        radius_reg_weight=0.0,
        level_radius_weight=0.0,
        load_balancing_coef=0.0,
        supervision_manifest_path=str(generated_bundle),
        supervision_mode='repaired',
        structural_preference_weight=0.35,
    )
    model.current_curriculum_flags = {
        'enable_hard_negative_mining': True,
        'enable_router_guided_sampling': False,
        'enable_clustering': True,
    }
    model.current_schedule_scalars = {}
    model.code_to_pseudo_label = {
        '111111': 7,
        '111113': 7,
        '222222': 7,
    }
    monkeypatch.setattr(model, '_update_curriculum_state', lambda *_args: None)
    monkeypatch.setattr(model, 'log', Mock())
    return model


def test_forced_reorder_preserves_uid_across_every_loss_field(
    tiny_repaired_model,
    repaired_training_batch,
    monkeypatch,
):
    spy = SelectionSpyLoss()
    monkeypatch.setattr(
        tiny_repaired_model,
        '_compute_contrastive_loss',
        spy.contrastive,
    )
    monkeypatch.setattr(
        tiny_repaired_model,
        '_compute_structural_preference_loss',
        spy.structural,
    )
    monkeypatch.setattr(
        tiny_repaired_model.selection_coordinator,
        'select',
        forced_selection([2, 0, 1]),
    )

    loss = tiny_repaired_model.training_step(repaired_training_batch, batch_idx=0)
    loss.backward()

    assert spy.contrastive_uids == spy.structural_uids
    assert spy.code_ids == [4, 2, 3]
    assert spy.structural_distances == [99.0, 2.0, 99.0]
    assert spy.exclusion_flags == [False, True, False]
    assert spy.router_first_column == [0.3, 0.1, 0.2]
    assert spy.false_negative_flags == [False, False, True]
    assert all(
        uid[2] >= 0
        for row in spy.contrastive_uids
        for uid in row
    )
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in tiny_repaired_model.parameters()
    )
```

The fixtures above use anchor ID `0`, positive ID `1`, and original candidate IDs `[2, 3, 4]`. They mark codes `2` and `3` as pseudo-related, with `2` also explicitly excluded. Its effective false-negative flag must therefore be `False` for `2` and `True` for `3`; the structural spy computes only over `valid & ~explicit`. The shorter first item receives one invalid padded source row, and the assertion proves that UID never reaches either spy.

- [x] **Step 2: Add a boundary test that reproduces the old failure**

```python
def test_old_parallel_arrays_misalign_but_checked_selection_does_not(candidate_batch):
    order = [2, 0, 1]
    reordered_embeddings = candidate_batch.embedding[:, order]
    old_parallel_pairs = list(
        zip(
            reordered_embeddings[0, :, 0].tolist(),
            candidate_batch.code_id[0].tolist(),
        )
    )
    expected_code_for_embedding = {1.0: 101, 2.0: 102, 3.0: 103}

    assert any(
        expected_code_for_embedding[embedding_value] != code_id
        for embedding_value, code_id in old_parallel_pairs
    )

    selected = candidate_batch.select(forced_selection(order)(candidate_batch))
    checked_pairs = list(
        zip(
            selected.embedding[0, :, 0].tolist(),
            selected.code_id[0].tolist(),
        )
    )
    assert checked_pairs == [(3.0, 103), (1.0, 101), (2.0, 102)]
```

- [x] **Step 3: Run the integration test and verify failure**

Run: `uv run pytest tests/integration/test_stage3_training_step.py tests/unit/test_naics_model.py -q`
Expected: FAIL because `training_step` still builds the false-negative mask before mining and passes original-order codes beside reordered embeddings.

- [x] **Step 4: Load the validated index and initialize repaired losses**

> Deviation: The bundle is validated before the encoder is built; evaluation ground truth and the hierarchy come from the bundle, and legacy distance/relations paths are rejected (D19). `rank_order_weight` was removed and `selection_seed` added.

Add constructor parameters:

```python
supervision_manifest_path: str | None = None,
supervision_contract_version: str = CONTRACT_VERSION,
supervision_mode: str = 'repaired',
structural_preference_weight: float = 0.35,
structural_preference_margin: float = 0.1,
structural_preference_temperature: float = 1.0,
structural_preference_tie_tolerance: float = 1e-6,
```

In repaired mode, call `load_validated_bundle`, create `SupervisionIndex`, load the validated structural matrix for `HierarchyPreservationLoss`, initialize `StructuralPreferenceLoss`, and construct `NegativeSelectionCoordinator`. Save paths/identifiers in hyperparameters, not the index tensors or manifest object.

- [x] **Step 5: Forward the candidate pool once and build UID triples**

> Deviation: Only valid candidate rows are encoded; padding rows receive zero outputs (D20).

```python
def _forward_candidate_pool(
    self, batch: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    candidate_output = self(batch['candidate_inputs'])
    embedding = candidate_output['embedding'].view(
        batch['batch_size'], batch['k_candidates'], -1
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    rank_component = torch.full_like(batch['candidate_source_slot'], rank)
    batch_component = torch.arange(
        batch['batch_size'], device=self.device
    ).unsqueeze(1).expand_as(batch['candidate_source_slot'])
    candidate_uid = torch.stack(
        [rank_component, batch_component, batch['candidate_source_slot']], dim=-1
    )
    return candidate_output, candidate_uid
```

Invalid rows keep source slot `-1` and `valid_mask = False`; their UIDs are never selectable.

- [x] **Step 6: Build pseudo-related masks only after checked selection**

Replace `_build_false_negative_mask(batch, batch_size)` with:

```python
def _build_selected_pseudo_related_mask(
    self,
    anchor_code_ids: torch.Tensor,
    selected: SelectedNegativeBatch,
) -> torch.Tensor | None:
    if not self.current_curriculum_flags.get('enable_clustering', False):
        return None
    if not self.code_to_pseudo_label:
        return None
    anchor_labels = torch.tensor(
        [
            self.code_to_pseudo_label.get(self.supervision_index.id_to_code[int(code_id)], -1)
            for code_id in anchor_code_ids
        ],
        device=self.device,
    )
    candidate_labels = torch.tensor(
        [
            [
                self.code_to_pseudo_label.get(
                    self.supervision_index.id_to_code[int(code_id)], -2
                )
                for code_id in row
            ]
            for row in selected.code_id
        ],
        device=self.device,
    )
    pseudo_related = (
        anchor_labels.unsqueeze(1).eq(candidate_labels)
        & anchor_labels.unsqueeze(1).ge(0)
        & candidate_labels.ge(0)
    )
    return pseudo_related & ~selected.is_explicit_exclusion & selected.valid_mask
```

Do not catch identity or shape errors in this method.

- [x] **Step 7: Rewrite the repaired training-step sequence**

> Deviation: Repaired validation scores the whole eligible pool without selection (D13); `test_naics_model.py` was triaged per D14.

The repaired branch of `training_step` executes in this exact order:

```python
def _compute_contrastive_loss(
    self,
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    selected: SelectedNegativeBatch,
    effective_mask: torch.Tensor | None,
) -> torch.Tensor:
    return self.loss_fn(
        anchor_emb,
        positive_emb,
        selected.embedding,
        valid_mask=selected.valid_mask,
        is_explicit_exclusion=selected.is_explicit_exclusion,
        pseudo_related_mask=effective_mask,
    )


anchor_output = self(batch['anchor'])
positive_output = self(batch['positive'])
candidate_output, candidate_uid = self._forward_candidate_pool(batch)

selected = self._select_negative_batch(
    batch=batch,
    anchor_output=anchor_output,
    candidate_output=candidate_output,
    candidate_uid=candidate_uid,
    batch_idx=batch_idx,
)
pseudo_related = self._build_selected_pseudo_related_mask(
    batch['anchor_code_id'], selected
)
effective_mask, auxiliary_fn_loss = self._apply_false_negative_strategy_wrapper(
    anchor_output['embedding'],
    selected,
    pseudo_related,
)
contrastive_loss = self._compute_contrastive_loss(
    anchor_output['embedding'],
    positive_output['embedding'],
    selected,
    effective_mask,
)
structural_preference_loss = self._compute_structural_preference_loss(
    anchor_output['embedding'],
    positive_output['embedding'],
    batch,
    selected,
)
```

Then compute hierarchy, radius, level-radius, and load-balancing terms. Candidate router outputs used by load balancing are masked by `candidate_valid_mask`; selected-candidate regularizers use `selected.valid_mask`. No loss reads `negative_codes`, pre-mining masks, or separately reordered router arrays.

- [x] **Step 8: Add low-cardinality selection health counters**

> Deviation: Negative-distribution and hard-negative diagnostics now read the selected batch; the global_batch/* and router_confusion_* metrics were removed (D21). Per-reason and eligibility counters were added (D32).

Add `_log_selection_health(candidates, selected, batch_size)` to `LoggingMixin`, call it from
`_select_negative_batch` after the checked gather, and log these epoch aggregates:

```python
selection_metrics = {
    'train/integrity/anchors_with_exclusions': candidates.is_explicit_exclusion.any(dim=1).sum(),
    'train/integrity/quota_selections': selected.selection_reasons.eq(
        SelectionReason.EXCLUSION_QUOTA
    ).sum(),
    'train/integrity/invalid_candidates_ignored': (~candidates.valid_mask).sum(),
    'train/integrity/deterministic_backfills': selected.selection_reasons.eq(
        SelectionReason.BACKFILL
    ).sum(),
    'train/integrity/duplicate_candidates_removed': torch.tensor(
        sum(
            int(valid.sum())
            - len(set(codes[valid].detach().cpu().tolist()))
            for codes, valid in zip(candidates.code_id, candidates.valid_mask)
        ),
        device=candidates.code_id.device,
    ),
}
```

Log each value with `on_step=False`, `on_epoch=True`, `reduce_fx='sum'`, and the current batch size. Keep candidate IDs out of metric names and values.

- [x] **Step 9: Run model and full-step integration tests**

Run: `uv run pytest tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py -q`
Expected: PASS; the old-failure fixture demonstrates the prior mismatch and the canonical selector keeps every spy field on the same UID.

- [x] **Step 10: Commit the canonical Stage-3 step**

```bash
git add src/naics_embedder/text_model/naics_model.py src/naics_embedder/text_model/mixins/curriculum.py src/naics_embedder/text_model/mixins/loss.py src/naics_embedder/text_model/mixins/logging.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py
git commit -m "fix(training): align stage3 supervision by candidate identity"
```

---

### Task 12: Make repaired configuration and checkpoint resume fail closed

**Files:**
- Create: `src/naics_embedder/supervision/checkpoints.py`
- Modify: `src/naics_embedder/utils/config.py:382-1041`
- Modify: `src/naics_embedder/utils/validation.py`
- Modify: `src/naics_embedder/cli/commands/training.py:237-668`
- Modify: `src/naics_embedder/text_model/naics_model.py`
- Modify: `conf/config.yaml`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_utils_validation.py`
- Modify: `tests/unit/test_cli_training.py`
- Create: `tests/unit/test_checkpoint_contract.py`

**Interfaces:**
- Consumes: validated bundle identity and model/loss/mining version constants.
- Produces: `SupervisionRuntimeConfig`, `StructuralPreferenceConfig`, `CheckpointLoadMode`, `CheckpointContract`, `validate_exact_resume`, `load_weights_only`, and mandatory pre-model supervision validation.

- [x] **Step 1: Write failing repaired-config migration tests**

```python
# add to tests/unit/test_config.py
from pathlib import Path

import yaml


@pytest.fixture
def valid_config_dict():
    return yaml.safe_load(Path('conf/config.yaml').read_text())


def test_repaired_config_rejects_legacy_rank_key(valid_config_dict):
    valid_config_dict['supervision'] = {
        'mode': 'repaired',
        'manifest_path': '/tmp/bundle/manifest.json',
    }
    valid_config_dict['loss']['rank_order_weight'] = 0.35

    with pytest.raises(
        ValidationError,
        match='rank_order_weight.*structural_preference',
    ):
        Config.model_validate(valid_config_dict)


def test_repaired_config_rejects_high_exclusion_weight(valid_config_dict):
    valid_config_dict['supervision'] = {
        'mode': 'repaired',
        'manifest_path': '/tmp/bundle/manifest.json',
    }
    valid_config_dict['data_loader']['streaming']['phase1_exclusion_weight'] = 100.0

    with pytest.raises(
        ValidationError,
        match='phase1_exclusion_weight.*one-slot exclusion quota',
    ):
        Config.model_validate(valid_config_dict)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('temperature', 0.0, 'temperature'),
        ('margin', -0.1, 'margin'),
        ('tie_tolerance', -0.1, 'tie_tolerance'),
    ],
)
def test_structural_preference_config_bounds(field, value, message):
    data = {
        'weight': 0.35,
        'margin': 0.1,
        'temperature': 1.0,
        'tie_tolerance': 1e-6,
    }
    data[field] = value

    with pytest.raises(ValidationError, match=message):
        StructuralPreferenceConfig(**data)
```

- [x] **Step 2: Write failing checkpoint-contract tests**

```python
# tests/unit/test_checkpoint_contract.py
import pytest
import torch
from torch import nn

from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    load_weights_only,
    validate_exact_resume,
)


@pytest.fixture
def runtime_contract() -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )


@pytest.fixture
def tiny_repaired_model() -> nn.Module:
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    model.current_curriculum_flags = {}
    return model


def test_matching_new_checkpoint_can_exact_resume(tmp_path, runtime_contract):
    path = tmp_path / 'new.ckpt'
    torch.save({'stage3_supervision': runtime_contract.model_dump()}, path)

    validate_exact_resume(path, runtime_contract)


@pytest.mark.parametrize(
    'checkpoint_metadata',
    [
        None,
        {'contract_version': 'legacy'},
        {'bundle_id': 'other-bundle'},
        {'codebook_fingerprint': 'f' * 64},
        {'structural_preference_loss_version': 'other-loss'},
        {'mining_contract_version': 'other-mining'},
    ],
)
def test_legacy_or_mismatched_checkpoint_cannot_exact_resume(
    tmp_path, runtime_contract, checkpoint_metadata
):
    path = tmp_path / 'checkpoint.ckpt'
    payload = {}
    if checkpoint_metadata is not None:
        payload['stage3_supervision'] = {
            **runtime_contract.model_dump(),
            **checkpoint_metadata,
        }
    torch.save(payload, path)

    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)


def test_weights_only_loads_allowlisted_encoder_and_resets_training_state(
    tmp_path, tiny_repaired_model
):
    path = tmp_path / 'legacy.ckpt'
    encoder_key = next(
        name
        for name in tiny_repaired_model.state_dict()
        if name.startswith('encoder.')
    )
    state = {
        encoder_key: torch.ones_like(tiny_repaired_model.state_dict()[encoder_key]),
        'lambdarank_loss_fn.tree_distances': torch.ones((3, 3)),
        'unexpected.weight': torch.ones(1),
    }
    torch.save(
        {
            'state_dict': state,
            'optimizer_states': [{'state': {'x': 1}}],
            'epoch': 9,
            'global_step': 123,
        },
        path,
    )

    with pytest.raises(ValueError, match='unexpected.weight'):
        load_weights_only(tiny_repaired_model, path)


def test_weights_only_reports_loaded_skipped_and_missing_without_restoring_state(
    tmp_path, tiny_repaired_model
):
    path = tmp_path / 'legacy.ckpt'
    target = tiny_repaired_model.state_dict()
    encoder_keys = sorted(name for name in target if name.startswith('encoder.'))
    loaded_key = encoder_keys[0]
    initial_flags = dict(tiny_repaired_model.current_curriculum_flags)
    torch.save(
        {
            'state_dict': {
                loaded_key: torch.full_like(target[loaded_key], 0.25),
                'loss_fn.legacy_buffer': torch.ones(1),
            },
            'optimizer_states': [{'state': {'legacy': 1}}],
            'epoch': 9,
            'global_step': 123,
        },
        path,
    )

    report = load_weights_only(tiny_repaired_model, path)

    assert report.loaded == (loaded_key,)
    assert report.skipped == ('loss_fn.legacy_buffer',)
    assert report.missing == tuple(encoder_keys[1:])
    assert report.unexpected == ()
    assert tiny_repaired_model.current_curriculum_flags == initial_flags
    assert torch.equal(
        tiny_repaired_model.state_dict()[loaded_key],
        torch.full_like(target[loaded_key], 0.25),
    )
```

Add this CLI boundary test; it proves optimizer, epoch/global-step, curriculum, and sampler state remain freshly initialized rather than being restored by Lightning:

```python
# add to tests/unit/test_cli_training.py
from naics_embedder.supervision.checkpoints import MigrationReport
from naics_embedder.utils.config import CheckpointLoadMode


def test_weights_only_never_passes_checkpoint_to_trainer(
    training_env, monkeypatch
):
    training_env.checkpoint_info = CheckpointInfo(
        path='legacy.ckpt',
        is_same_stage=False,
        exists=True,
    )
    reports = []

    def fake_load_weights_only(model, path):
        assert path == 'legacy.ckpt'
        report = MigrationReport(
            loaded=('encoder.weight',),
            skipped=('loss_fn.buffer',),
            missing=(),
            unexpected=(),
        )
        reports.append(report)
        return report

    monkeypatch.setattr(training, 'load_weights_only', fake_load_weights_only)

    training.train(
        ckpt_path='last',
        checkpoint_load_mode=CheckpointLoadMode.WEIGHTS_ONLY,
        skip_validation=True,
    )

    assert reports
    assert training_env.trainer.fit_calls[0]['ckpt_path'] is None
```

- [x] **Step 3: Run config/checkpoint suites and verify failure**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_checkpoint_contract.py tests/unit/test_utils_validation.py tests/unit/test_cli_training.py -q`
Expected: FAIL because repaired migration keys are accepted and checkpoint contract identifiers are absent.

- [x] **Step 4: Add explicit runtime and loss configuration**

```python
# src/naics_embedder/utils/config.py
class StructuralPreferenceConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    weight: float = Field(default=0.35, ge=0.0, le=1.0)
    margin: float = Field(default=0.1, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
    tie_tolerance: float = Field(default=1e-6, ge=0.0)


class SupervisionRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal['repaired', 'legacy_containment'] = 'repaired'
    manifest_path: Optional[str] = None
    contract_version: Literal['stage3-supervision-v1'] = CONTRACT_VERSION


class CheckpointLoadMode(str, Enum):
    EXACT = 'exact'
    WEIGHTS_ONLY = 'weights_only'
```

Change `LossConfig` to contain `structural_preference: StructuralPreferenceConfig`. Retain `rank_order_weight: Optional[float] = None` solely so the top-level validator can emit a specific migration message. Retain `phase1_exclusion_weight: Optional[float] = None` for the same reason. Add `supervision: SupervisionRuntimeConfig` to `Config` and this validator:

```python
@model_validator(mode='after')
def validate_supervision_contract(self) -> 'Config':
    if self.supervision.mode == 'repaired':
        if self.loss.rank_order_weight is not None:
            raise ValueError(
                'loss.rank_order_weight is a legacy LambdaRank setting; '
                'configure loss.structural_preference instead'
            )
        if self.data_loader.streaming.phase1_exclusion_weight is not None:
            raise ValueError(
                'data_loader.streaming.phase1_exclusion_weight is invalid in repaired mode; '
                'the one-slot exclusion quota owns representation'
            )
    return self
```

In `conf/config.yaml` add:

```yaml
supervision:
  mode: repaired
  contract_version: stage3-supervision-v1
  manifest_path: null  # data supervision prints the exact immutable path to set before training

loss:
  temperature: 0.07
  curvature: 1.0
  base_margin: 0.5
  hierarchy_weight: 0.45
  structural_preference:
    weight: 0.35
    margin: 0.1
    temperature: 1.0
    tie_tolerance: 0.000001
  radius_reg_weight: 0.10
  level_radius_weight: 0.15
```

Remove the repaired defaults for `rank_order_weight` and `phase1_exclusion_weight`. `manifest_path: null` is an intentional pre-generation state: config parsing succeeds, while the mandatory training gate below emits the exact command to set the generated path.

- [x] **Step 5: Make bundle validation mandatory before model or checkpoint work**

> Deviation: Advisory data-path checks are mode-aware, so repaired runs do not require legacy files (D23).

Add:

```python
# src/naics_embedder/utils/validation.py
def require_valid_supervision_bundle(
    cfg: Config,
) -> ValidatedSupervisionBundle | None:
    if cfg.supervision.mode == 'legacy_containment':
        return None
    if not cfg.supervision.manifest_path:
        raise ValidationError(
            'Repaired Stage-3 training requires supervision.manifest_path',
            remediation=[
                'Run: uv run naics-embedder data supervision',
                'Set supervision.manifest_path to the printed immutable manifest path',
            ],
        )
    bundle = load_validated_bundle(
        cfg.supervision.manifest_path,
        expected_contract=cfg.supervision.contract_version,
    )
    descriptions_hash = sha256_file(Path(cfg.data_loader.streaming.descriptions_parquet))
    if descriptions_hash != bundle.manifest.description_fingerprint:
        raise ValidationError(
            'Descriptions input does not match the supervision bundle',
            details=(
                f'expected {bundle.manifest.description_fingerprint}, '
                f'found {descriptions_hash}'
            ),
        )
    return bundle
```

Call this immediately after configuration/overrides in `train`, before `validate_training_config`, DataModule construction, checkpoint resolution, or model construction. `--skip-validation` skips only the older advisory/path/cache checks; update its help text accordingly.

Update the CLI unit fixture so this mandatory gate is testable without filesystem artifacts:

```python
# add inside tests/unit/test_cli_training.py::training_env
context.bundle = SimpleNamespace(
    manifest=SimpleNamespace(
        contract_version='stage3-supervision-v1',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )
)
monkeypatch.setattr(
    training,
    'require_valid_supervision_bundle',
    lambda cfg: (
        None
        if cfg.supervision.mode == 'legacy_containment'
        else context.bundle
    ),
)
```

- [x] **Step 6: Implement checkpoint contract and model hooks**

> Deviation: The model accepts a pre-validated bundle and a runtime contract, both excluded from hyperparameters (D24). After review, a weights-only checkpoint with no allowlisted encoder tensor is fatal.

```python
# src/naics_embedder/supervision/checkpoints.py
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
)


class CheckpointContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    supervision_mode: str
    contract_version: str = CONTRACT_VERSION
    bundle_id: str
    codebook_fingerprint: str
    structural_preference_loss_version: str = STRUCTURAL_PREFERENCE_LOSS_VERSION
    mining_contract_version: str = MINING_CONTRACT_VERSION


@dataclass(frozen=True)
class MigrationReport:
    loaded: tuple[str, ...]
    skipped: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def _load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), map_location='cpu', weights_only=False)


def validate_exact_resume(path: str | Path, runtime: CheckpointContract) -> None:
    checkpoint = _load_checkpoint(path)
    raw = checkpoint.get('stage3_supervision')
    if raw is None:
        raise ValueError(
            'legacy checkpoint has no Stage-3 contract and cannot exact resume; '
            'use weights_only explicitly'
        )
    saved = CheckpointContract.model_validate(raw)
    if saved != runtime:
        differences = {
            name: (getattr(saved, name), getattr(runtime, name))
            for name in runtime.model_fields
            if getattr(saved, name) != getattr(runtime, name)
        }
        raise ValueError(f'exact resume contract mismatch: {differences}')


def load_weights_only(model: torch.nn.Module, path: str | Path) -> MigrationReport:
    checkpoint = _load_checkpoint(path)
    source = checkpoint.get('state_dict')
    if not isinstance(source, dict):
        raise ValueError('weights-only checkpoint has no state_dict')
    allowed_prefixes = ('encoder.',)
    excluded_prefixes = (
        'loss_fn.',
        'hierarchy_loss_fn.',
        'lambdarank_loss_fn.',
        'structural_preference_loss_fn.',
        'ground_truth_distances',
        'norm_adaptive_margin.',
    )
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped = []
    unexpected = []
    for name, value in source.items():
        if name.startswith(allowed_prefixes):
            if name not in target or target[name].shape != value.shape:
                unexpected.append(name)
            else:
                loaded[name] = value
        elif name.startswith(excluded_prefixes):
            skipped.append(name)
        else:
            unexpected.append(name)
    if unexpected:
        raise ValueError(
            f'weights-only checkpoint has unexpected parameter groups: {sorted(unexpected)}'
        )
    model.load_state_dict(loaded, strict=False)
    missing = tuple(
        sorted(
            name
            for name in target
            if name.startswith(allowed_prefixes) and name not in loaded
        )
    )
    return MigrationReport(
        loaded=tuple(sorted(loaded)),
        skipped=tuple(sorted(skipped)),
        missing=missing,
        unexpected=(),
    )
```

`NAICSContrastiveModel.on_save_checkpoint` writes `self.checkpoint_contract.model_dump()` under `stage3_supervision`. `on_load_checkpoint` calls `validate_exact_resume` semantics against the in-memory checkpoint dictionary for repaired exact resume. The structural matrices remain bundle-loaded buffers and are excluded from weights-only loading.

- [x] **Step 7: Make CLI checkpoint modes explicit**

> Deviation: An explicit load mode replaces the is_same_stage heuristic, and exact resume is validated before the model is built (D25). Embedding generation requires the contract and fixes the tokenization-cache call (D26). `weights_only` without a checkpoint is fatal (review M7).

Add `--checkpoint-load-mode [exact|weights_only]` defaulting to `exact`. The flow is:

```python
runtime_contract = CheckpointContract(
    supervision_mode=cfg.supervision.mode,
    contract_version=bundle.manifest.contract_version,
    bundle_id=bundle.manifest.bundle_id,
    codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    structural_preference_loss_version=STRUCTURAL_PREFERENCE_LOSS_VERSION,
    mining_contract_version=MINING_CONTRACT_VERSION,
)
model = build_model_from_config(cfg, runtime_contract)
trainer_ckpt_path = None
if checkpoint_path and checkpoint_load_mode is CheckpointLoadMode.EXACT:
    validate_exact_resume(checkpoint_path, runtime_contract)
    trainer_ckpt_path = checkpoint_path
elif checkpoint_path:
    report = load_weights_only(model, checkpoint_path)
    log_migration_report(report)
```

Do not call `load_from_checkpoint` for weights-only migration. Do not pass its path to `trainer.fit`. Update embedding generation to require a matching repaired contract or an explicitly documented legacy-containment checkpoint.

- [x] **Step 8: Run config, validation, checkpoint, and CLI tests**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_utils_validation.py tests/unit/test_checkpoint_contract.py tests/unit/test_cli_training.py tests/unit/test_naics_model.py -q`
Expected: PASS; the CLI spy proves bundle validation occurs before model construction and checkpoint restoration.

- [x] **Step 9: Commit fail-closed migration**

```bash
git add conf/config.yaml src/naics_embedder/supervision/checkpoints.py src/naics_embedder/utils/config.py src/naics_embedder/utils/validation.py src/naics_embedder/cli/commands/training.py src/naics_embedder/text_model/naics_model.py tests/unit/test_config.py tests/unit/test_utils_validation.py tests/unit/test_cli_training.py tests/unit/test_checkpoint_contract.py
git commit -m "feat(training): enforce supervision checkpoint contracts"
```

---

### Task 13: Add explicit legacy containment without weakening repaired training

**Files:**
- Create: `src/naics_embedder/supervision/mode.py`
- Modify: `src/naics_embedder/text_model/dataloader/datamodule.py`
- Modify: `src/naics_embedder/text_model/mixins/curriculum.py`
- Modify: `src/naics_embedder/text_model/mixins/loss.py`
- Modify: `src/naics_embedder/text_model/naics_model.py`
- Modify: `src/naics_embedder/cli/commands/training.py`
- Modify: `tests/unit/test_datamodule.py`
- Modify: `tests/unit/test_naics_model.py`
- Modify: `tests/unit/test_cli_training.py`

**Interfaces:**
- Consumes: `supervision.mode` and checkpoint tags.
- Produces: `SupervisionModePolicy.from_name` and a contained legacy branch that permits only local unmined contrastive learning plus supervision-independent regularizers.

- [x] **Step 1: Write failing containment-policy tests**

```python
# add to tests/unit/test_naics_model.py
from naics_embedder.supervision.checkpoints import CheckpointContract


@pytest.fixture
def runtime_contract():
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )


@pytest.fixture
def legacy_model(model_config):
    return NAICSContrastiveModel(
        **model_config,
        supervision_mode='legacy_containment',
    )


@pytest.fixture
def legacy_batch(sample_training_batch):
    return sample_training_batch


def test_legacy_containment_disables_contaminated_and_reordering_paths(
    legacy_model, legacy_batch, monkeypatch
):
    monkeypatch.setattr(legacy_model, 'log', Mock())
    forbidden = [
        ('hard_negative_miner', 'propose'),
        ('router_guided_miner', 'propose'),
        ('structural_preference_loss_fn', 'forward'),
        ('hierarchy_loss_fn', 'forward'),
    ]
    for name, method in forbidden:
        value = getattr(legacy_model, name, None)
        if value is not None:
            monkeypatch.setattr(value, method, Mock(side_effect=AssertionError(name)))

    monkeypatch.setattr(
        legacy_model,
        '_build_selected_pseudo_related_mask',
        Mock(side_effect=AssertionError('pseudo-related handling')),
    )

    loss = legacy_model.training_step(legacy_batch, 0)

    assert torch.isfinite(loss)
    assert legacy_model.checkpoint_contract.supervision_mode == 'legacy_containment'


def test_containment_checkpoint_cannot_resume_repaired(runtime_contract):
    containment = runtime_contract.model_copy(
        update={
            'supervision_mode': 'legacy_containment',
            'bundle_id': 'legacy-containment',
            'codebook_fingerprint': 'unversioned',
        }
    )

    assert containment != runtime_contract
```

Add the explicit-mode CLI and datamodule boundaries:

```python
# add to tests/unit/test_cli_training.py
def test_cli_legacy_containment_is_prominently_tagged(
    cli_runner, training_env, monkeypatch, caplog
):
    cfg = training.Config.from_yaml('unused.yaml')
    cfg.supervision.mode = 'legacy_containment'
    cfg.supervision.manifest_path = None
    cfg.loss.rank_order_weight = 0.35
    cfg.data_loader.streaming.phase1_exclusion_weight = 100.0
    monkeypatch.setattr(
        training.Config,
        'from_yaml',
        classmethod(lambda cls, path: cfg),
    )

    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    assert 'LEGACY CONTAINMENT' in caplog.text or 'LEGACY CONTAINMENT' in result.output
    model = training_env.trainer.fit_calls[0]['model']
    assert model.kwargs['supervision_mode'] == 'legacy_containment'
    assert model.kwargs['checkpoint_contract'].bundle_id == 'legacy-containment'


# add to tests/unit/test_datamodule.py
def test_legacy_containment_ignores_all_candidates(make_batch_item):
    item = make_batch_item('111', '11', ['222', '333'])
    item['all_candidates'] = [
        {
            'negative_code': 'SHOULD-NOT-ENTER',
            'negative_idx': 999,
            'negative_embedding': item['negatives'][0]['negative_embedding'],
        }
    ]

    batch = collate_fn([item], supervision_mode='legacy_containment')

    assert batch['negative_codes'] == [['222', '333']]
    assert 'all_candidates' not in batch
    assert 'candidate_inputs' not in batch
```

The repaired-mode config tests from Task 12 are the negative half of this boundary: the same legacy keys fail unless `supervision.mode='legacy_containment'` is explicit.

- [x] **Step 2: Run containment tests and verify failure**

Run: `uv run pytest tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_datamodule.py -q`
Expected: FAIL because there is no explicit policy and legacy flags can still activate contaminated losses or reordering.

- [x] **Step 3: Define an immutable mode policy**

```python
# src/naics_embedder/supervision/mode.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisionModePolicy:
    name: str
    require_bundle: bool
    enable_structural_losses: bool
    enable_candidate_reordering: bool
    enable_pseudo_related: bool
    checkpoint_tag: str

    @classmethod
    def from_name(cls, name: str) -> 'SupervisionModePolicy':
        if name == 'repaired':
            return cls(
                name='repaired',
                require_bundle=True,
                enable_structural_losses=True,
                enable_candidate_reordering=True,
                enable_pseudo_related=True,
                checkpoint_tag='stage3-supervision-v1',
            )
        if name == 'legacy_containment':
            return cls(
                name='legacy_containment',
                require_bundle=False,
                enable_structural_losses=False,
                enable_candidate_reordering=False,
                enable_pseudo_related=False,
                checkpoint_tag='LEGACY-CONTAINMENT',
            )
        raise ValueError(f'unknown supervision mode {name!r}')
```

- [x] **Step 4: Enforce the policy at construction and execution boundaries**

> Deviation: The curriculum and loss mixins are unchanged: containment never enters them and their collaborators are None (D27).

Dispatch before repaired candidate construction and keep the contained branch explicit:

```python
def _legacy_containment_training_step(
    self,
    batch: dict[str, Any],
    batch_idx: int,
) -> torch.Tensor:
    anchor_output = self(batch['anchor'])
    positive_output = self(batch['positive'])
    negative_output = self(batch['negatives'])
    batch_size = int(batch['batch_size'])
    k_negatives = int(batch['k_negatives'])
    negative_emb = negative_output['embedding'].reshape(batch_size, k_negatives, -1)
    valid = torch.ones(
        (batch_size, k_negatives),
        dtype=torch.bool,
        device=negative_emb.device,
    )
    explicit = torch.zeros_like(valid)
    contrastive = self.loss_fn(
        anchor_output['embedding'],
        positive_output['embedding'],
        negative_emb,
        valid_mask=valid,
        is_explicit_exclusion=explicit,
        pseudo_related_mask=None,
    )
    gate_probs, topk_indices = self._collect_gate_outputs(
        [anchor_output, positive_output, negative_output]
    )
    load_balancing = self._compute_load_balancing_loss(
        gate_probs,
        topk_indices,
        batch_size,
    )
    radius = self._compute_radius_regularization(
        anchor_output['embedding'],
        positive_output['embedding'],
        negative_output['embedding'],
        batch_size,
    )
    level_radius = self._compute_level_radius_alignment_loss(
        anchor_output['embedding'],
        positive_output['embedding'],
        batch,
        batch_size,
    )
    total = (
        contrastive
        + self.load_balancing_coef * load_balancing
        + radius
        + level_radius
    )
    self.log(
        'train/integrity/legacy_containment',
        1.0,
        on_step=False,
        on_epoch=True,
        batch_size=batch_size,
    )
    return total


# insert as the first executable lines of Task 11's training_step
if self.supervision_policy.name == 'legacy_containment':
    return self._legacy_containment_training_step(batch, batch_idx)
```

In containment mode:

- keep the legacy file paths and legacy collation adapter so old artifacts remain readable;
- disable hierarchy and every structural-ranking loss even when old weights are nonzero;
- bypass hard-negative and router-based selection;
- return local unmined contrastive negatives in their collated order;
- return `None` before pseudo-related classification, elimination, or attraction;
- keep normal MoE routing, load balancing, radius, and other supervision-independent regularizers;
- log a prominent `LEGACY CONTAINMENT` message at startup and an epoch metric `train/integrity/legacy_containment = 1`;
- save checkpoint contract values `supervision_mode='legacy_containment'`, `bundle_id='legacy-containment'`, and `codebook_fingerprint='unversioned'`.

Use explicit policy predicates; do not rewrite user configuration values or silently fall back from repaired mode.

- [x] **Step 5: Run containment and repaired regression suites**

Run: `uv run pytest tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_datamodule.py tests/integration/test_stage3_training_step.py -q`
Expected: PASS for both modes; containment never calls forbidden paths and repaired training still uses the canonical selector.

- [x] **Step 6: Commit legacy containment**

```bash
git add src/naics_embedder/supervision/mode.py src/naics_embedder/text_model/dataloader/datamodule.py src/naics_embedder/text_model/mixins/curriculum.py src/naics_embedder/text_model/mixins/loss.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py tests/unit/test_datamodule.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py
git commit -m "feat(training): contain legacy supervision explicitly"
```

---

### Task 14: Preserve graph compatibility, document rollout, and run the acceptance gate

**Files:**
- Modify: `src/naics_embedder/graph_model/dataloader/hgcn_streaming_dataset.py:140-215`
- Modify: `src/naics_embedder/graph_model/dataloader/hgcn_datamodule.py`
- Modify: `src/naics_embedder/graph_model/curriculum/preprocess_curriculum.py:415-579`
- Modify: `src/naics_embedder/utils/config.py:771-953`
- Modify: `conf/graph.yaml`
- Modify: `tests/unit/test_hgcn_streaming_dataset.py`
- Modify: `tests/unit/test_hgcn_datamodule.py`
- Modify: `tests/unit/test_graph_preprocessing.py`
- Modify: `docs/text_training.md`
- Modify: `docs/api/config.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: validated bundle paths and legacy compatibility columns.
- Produces: a narrow graph projection that reads rebuilt artifacts without changing graph sampling/objectives, plus operator-facing generation/migration/rollout documentation.

- [x] **Step 1: Write the graph compatibility regression**

```python
# add to tests/unit/test_hgcn_streaming_dataset.py
def test_graph_loader_reads_rebuilt_training_pairs_without_new_semantics(
    generated_bundle,
):
    bundle = load_validated_bundle(generated_bundle)
    result = streaming._load_negative_candidates(
        str(bundle.artifact_path('training_pairs')),
        required_pairs={(0, 1)},
    )

    negative = result[(0, 1)][0]
    assert set(negative) == {
        'negative_idx',
        'negative_code',
        'relation_margin',
        'distance_margin',
    }
    assert negative == {
        'negative_idx': 2,
        'negative_code': '111113',
        'relation_margin': 1.0,
        'distance_margin': 1.5,
    }
```

```python
# add to tests/unit/test_graph_preprocessing.py
from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    resolve_graph_supervision_paths,
)
from naics_embedder.supervision.artifacts import load_validated_bundle


def test_graph_preprocessing_resolves_one_bundle_and_rejects_mixed_paths(
    tmp_path,
    generated_bundle,
    descriptions_fixture,
    pair_facts_fixture,
):
    first = resolve_graph_supervision_paths(generated_bundle)
    other_manifest = generate_supervision_bundle_from_frames(
        output_root=tmp_path / 'other',
        bundle_id='bundle-b',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
    other_bundle = load_validated_bundle(other_manifest)

    assert first.distances == load_validated_bundle(generated_bundle).artifact_path(
        'distances'
    )
    assert first.training_pairs == load_validated_bundle(
        generated_bundle
    ).artifact_path('training_pairs')
    with pytest.raises(ValueError, match='distances.*bundle-a'):
        resolve_graph_supervision_paths(
            generated_bundle,
            distances_path=other_bundle.artifact_path('distances'),
        )
```

- [x] **Step 2: Run graph compatibility tests and verify failure**

Run: `uv run pytest tests/unit/test_hgcn_streaming_dataset.py tests/unit/test_hgcn_datamodule.py tests/unit/test_graph_preprocessing.py -q`
Expected: FAIL because graph configuration still accepts independent artifact paths and cannot resolve the immutable bundle.

- [x] **Step 3: Add a narrow bundle-to-graph adapter**

> Deviation: The adapter also resolves `distance_matrix` for HGCN evaluation, and `conf/graph.yaml` drops its explicit legacy paths so a manifest alone suffices (D28).

Add `supervision_manifest_path: Optional[str]` to `GraphConfig`. When set, validate one bundle and resolve its `relations`, `distances`, and `training_pairs` paths. Project only these existing fields at the graph loader boundary:

```python
from dataclasses import dataclass
from pathlib import Path

from naics_embedder.supervision.artifacts import load_validated_bundle


@dataclass(frozen=True)
class GraphSupervisionPaths:
    relations: Path
    distances: Path
    training_pairs: Path


def resolve_graph_supervision_paths(
    manifest_path: str | Path,
    *,
    relations_path: str | Path | None = None,
    distances_path: str | Path | None = None,
    training_pairs_path: str | Path | None = None,
) -> GraphSupervisionPaths:
    bundle = load_validated_bundle(manifest_path)
    resolved = GraphSupervisionPaths(
        relations=bundle.artifact_path('relations').resolve(),
        distances=bundle.artifact_path('distances').resolve(),
        training_pairs=bundle.artifact_path('training_pairs').resolve(),
    )
    supplied = {
        'relations': relations_path,
        'distances': distances_path,
        'training_pairs': training_pairs_path,
    }
    for logical_name, candidate in supplied.items():
        if candidate is None:
            continue
        expected = getattr(resolved, logical_name)
        if Path(candidate).resolve() != expected:
            raise ValueError(
                f'{logical_name} path does not belong to supervision bundle '
                f'{bundle.manifest.bundle_id}: {candidate}'
            )
    return resolved


GRAPH_NEGATIVE_COLUMNS = (
    'negative_idx',
    'negative_code',
    'relation_margin',
    'distance_margin',
)


def _project_graph_negative(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in GRAPH_NEGATIVE_COLUMNS}
```

Do not forward semantic target/source, exclusion flags, candidate UIDs, router fields, or repaired Stage-3 selection policy into HGCN. Keep its samplers, loss construction, and semantic-retention behavior unchanged. `compute_difficulty_thresholds` resolves both frames from the same bundle and writes the threshold artifact during bundle generation; it never combines independent legacy paths in repaired mode.

- [x] **Step 4: Document the exact operator workflow**

> Deviation: Also refreshed overview, index, usage, quickstart, hgcn_training, tests/README, and CLAUDE.md, which referenced deleted classes and the old pipeline (D29).

Add this workflow, with prose explaining each gate, to `docs/text_training.md` and link it from `README.md`:

```bash
uv run naics-embedder data supervision
uv run naics-embedder train \
  --config conf/config.yaml \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json

# Exact resume: every identifier must match.
uv run naics-embedder train \
  --ckpt-path checkpoints/run/last.ckpt \
  --checkpoint-load-mode exact \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json

# Explicit legacy initialization: weights only, all training state resets.
uv run naics-embedder train \
  --ckpt-path checkpoints/legacy.ckpt \
  --checkpoint-load-mode weights_only \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Document:

- the three independent axes (structure, semantic target/source, exclusion provenance);
- candidate UID versus code ID;
- exactly-one exclusion rotation;
- `StructuralPreferenceLoss` formula and correction-direction gradient;
- immutable bundle contents and validation failure messages;
- cache regeneration;
- exact resume versus weights-only migration;
- `legacy_containment` limitations and checkpoint tag;
- the four rollout gates required before normal repaired Stage-3 training.

Update `docs/api/config.md` with the exact YAML from Task 12 and remove `rank_order_weight`/high exclusion-weight guidance.

- [x] **Step 5: Run the four mandatory rollout gates directly**

Run:

```bash
uv run pytest tests/unit/test_candidate_contract.py -q
uv run pytest tests/unit/test_loss.py::test_structural_preference_gradient_corrects_an_inversion -q
uv run pytest tests/unit/test_supervision_artifacts.py -q
uv run pytest tests/integration/test_stage3_training_step.py -q
```

Expected: every command passes independently. These are the forced-reorder, gradient-sign, bundle-validation, and full-step rollout gates.

- [x] **Step 6: Run all focused supervision, data, runtime, migration, and graph suites**

Run:

```bash
uv run pytest \
  tests/unit/test_supervision_schema.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_supervision_index.py \
  tests/unit/test_candidate_contract.py \
  tests/unit/test_negative_selection.py \
  tests/unit/test_data_distances.py \
  tests/unit/test_data_relations.py \
  tests/unit/test_data_triplets.py \
  tests/unit/test_streaming_dataset.py \
  tests/unit/test_streaming_sampling.py \
  tests/unit/test_difficulty_sampler.py \
  tests/unit/test_datamodule.py \
  tests/unit/test_tokenization_cache.py \
  tests/unit/test_hard_negative_mining.py \
  tests/unit/test_loss.py \
  tests/unit/test_false_negative_strategy.py \
  tests/unit/test_naics_model.py \
  tests/unit/test_config.py \
  tests/unit/test_utils_validation.py \
  tests/unit/test_checkpoint_contract.py \
  tests/unit/test_cli_training.py \
  tests/unit/test_hgcn_streaming_dataset.py \
  tests/unit/test_hgcn_datamodule.py \
  tests/unit/test_graph_preprocessing.py \
  tests/integration/test_distributed_supervision.py \
  tests/integration/test_stage3_training_step.py \
  -q
```

Expected: all listed tests pass with no warnings converted from supervision-contract failures.

- [x] **Step 7: Run the complete repository verification**

> Deviation: `ruff check src tests` still reports 27 pre-existing errors, all in files this branch never touched (baseline 86), so that part is deferred. MkDocs was built to a temporary site directory.

Run:

```bash
uv run ruff check src tests
uv run pytest
uv run mkdocs build --strict
git diff --check
```

Expected: ruff reports `All checks passed!`, the full pytest suite passes, MkDocs completes without warnings/errors, and `git diff --check` prints nothing.

- [x] **Step 8: Commit compatibility and rollout documentation**

```bash
git add conf/graph.yaml src/naics_embedder/graph_model src/naics_embedder/utils/config.py tests/unit/test_hgcn_streaming_dataset.py tests/unit/test_hgcn_datamodule.py tests/unit/test_graph_preprocessing.py docs/text_training.md docs/api/config.md README.md
git commit -m "docs(stage3): document supervision integrity rollout"
```

After the final whole-branch review is resolved, run the Writing Plans `Plan Completion Protocol`: resolve or defer every leftover, mark this plan complete, update `specs/deferred_items.md`, report deferred-backlog health, and retire the plan/spec only when the protocol’s shared-spec check permits it.
