# Stage-3 Supervision Integrity

**Status:** Approved design; awaiting written-spec review

**Next skill after approval:** `writing-plans` in a fresh session

## 1. Purpose

Repair the Stage-3 supervision contract identified by findings #3, #4, and #8 in
`reports/two-stage-methodology-review-2026-09-21.md`:

1. Explicit NAICS exclusions are currently encoded by overwriting structural distance and
   relation values, even though training treats those pairs as negatives.
2. The active LambdaRank formulation can produce gradients that move an inverted ranking farther
   in the wrong direction.
3. Hard-negative mining reorders or selects embeddings without applying the same selection to
   candidate codes, masks, router outputs, structural targets, and other supervision metadata.

These are one problem: candidate identity and supervision meaning are implicit. The repair makes
both explicit so that every selected negative retains its identity and every loss receives one
unambiguous target with a verifiably correct gradient direction.

## 2. Scope

### 2.1 In scope

- Structural distance and relation generation.
- Directional exclusion provenance and its symmetric Stage-3 interpretation.
- Semantic supervision labels and sampling roles.
- Training-pair generation and cached candidate data.
- Exclusion-aware negative sampling.
- Replacement of LambdaRank with a simple structural-preference loss.
- Hard-negative, router-based, and distributed candidate selection.
- Collation and training-step metadata alignment.
- Generated-artifact versioning and validation.
- Configuration and checkpoint migration.
- Temporary containment for legacy data, configurations, and checkpoints.
- Invariant-focused tests at generation, loss, mining, and training-step boundaries.

### 2.2 Explicitly out of scope

- Spearman or curvature metrics.
- HGCN semantic-retention behavior or graph-model redesign.
- QCEW evaluation.
- Curriculum-documentation cleanup.
- Changes to the NAICS hierarchy definition or code-vintage conversion.
- General MoE/router architecture changes.
- Unrelated model, data, or evaluation refactors.

Compatibility changes required to keep an existing consumer working are in scope, but must not
expand into redesigning that consumer.

## 3. Success criteria

The repair is complete when all of the following hold:

- Exclusion handling never mutates a structural distance or structural relation.
- Both directional exclusion flags survive generation; Stage-3 derives a symmetric boundary from
  their logical OR.
- Every runtime negative has stable occurrence identity, canonical code identity, and aligned
  metadata.
- Candidate selection is expressed as checked source indices, and all fields are gathered once.
- Explicit exclusions receive guaranteed but bounded sampling exposure.
- Explicit exclusions cannot be masked or attracted as false negatives.
- Structural ranking never compares an explicit exclusion.
- The structural-ranking replacement passes direct gradient-sign tests.
- Distributed mining recomputes supervision relative to the local anchor.
- Training rejects mixed, stale, or unversioned supervision artifacts.
- Legacy checkpoints cannot exact-resume into the repaired contract.
- Existing graph consumers can read the rebuilt data through compatibility columns or an adapter,
  without changing HGCN behavior.

## 4. Chosen approach and rejected alternatives

### 4.1 Chosen: identity-first supervision contract

A typed candidate batch owns all aligned fields. Mining returns a typed selection of source
indices, scores, and provenance. A single batch operation validates and gathers every field.
Structural facts, semantic supervision, and explicit exclusion provenance are represented on
separate axes. LambdaRank is replaced rather than repaired.

This boundary directly encodes the invariants implicated by all three findings and makes future
candidate metadata additions safe by default.

### 4.2 Rejected: patch the existing parallel arrays

Returning indices from the miner and manually applying them to each existing tensor would be a
smaller initial diff. It would leave alignment as a repeated call-site convention, however, so the
next metadata field could reproduce finding #8.

### 4.3 Rejected as a final design: disable ranking and mining

Local unmined contrastive training is a safe temporary containment option, but it abandons intended
structural and hard-negative behavior. It is retained only as an explicitly tagged legacy mode.

## 5. Supervision semantics

The contract has three independent axes.

### 5.1 Structural facts

Structural fields describe the materialized NAICS hierarchy only:

- `structural_distance: float`
- `structural_relation_id: int`
- `structural_relation_name: str`

Structural distance is symmetric. Structural relation values follow a documented canonical pair
orientation; the current generator places the shallower code first and uses stable code ordering for
ties. A lookup matrix may mirror the resulting relation ID in both directions, but the relation name
itself must not be assumed to express both directions. Neither distance nor relation is changed by
semantic or exclusion processing. Distance zero is reserved for a genuine zero structural distance.
No structural relation ID or name is reserved to mean exclusion.

Cross-sector or otherwise special structural relationships must use an explicit structural value
or enum member, not an exclusion sentinel.

### 5.2 Semantic supervision

Semantic supervision uses a typed target:

- `RELATED`
- `UNRELATED`
- `UNKNOWN`

It also carries a source such as `TRAINING_POSITIVE`, `EXPLICIT_EXCLUSION`, or `UNLABELED`.
This field records supervision meaning, not tree geometry.

- A generated positive is `RELATED`.
- An explicit exclusion is `UNRELATED`.
- An ordinary sampled negative may remain `UNKNOWN` even though it occupies a negative sampling
  slot.
- Runtime pseudo-relatedness is model-derived metadata and is not persisted as ground truth.

Sampling role is separate from semantic target. Selecting an `UNKNOWN` candidate as a
noise-contrastive negative does not relabel it as authoritative `UNRELATED` data.

### 5.3 Explicit exclusion provenance

Each anchor-candidate view exposes:

- `anchor_excludes_candidate: bool`
- `candidate_excludes_anchor: bool`
- `is_explicit_exclusion: bool`

The first two fields are mapped from the canonical pair's two source directions and preserve which
code published the exclusion. Stage-3 defines:

```text
is_explicit_exclusion = anchor_excludes_candidate OR candidate_excludes_anchor
```

The symmetric field may be materialized for efficient loading, but validation must recompute it.
The directional fields remain available for audit and future consumers.

### 5.4 Required pair identity

Every pair-fact record contains canonical `code_i_id` and `code_j_id` values tied to a versioned
codebook, plus human-readable code strings for inspection. Training records and runtime joins map
those fields into explicit `anchor_code_id` and `candidate_code_id` views. Runtime logic uses IDs,
not string position, as identity.

NAICS codes remain strings at input/output boundaries. The codebook provides a stable numeric index
for tensor operations and is fingerprinted as part of the supervision bundle.

## 6. Generated data contract

### 6.1 Pair facts

The long-form distance and relation data must expose, directly or through a validated join:

- both code IDs and code strings;
- untouched structural distance;
- untouched structural relation ID and name;
- both directional exclusion flags;
- the derived symmetric exclusion flag.

The distance and relation matrices are derived from the same pair-fact source. They are not
independent authorities.

### 6.2 Training pairs

Every generated positive/negative record retains:

- anchor, positive, and negative code identities;
- raw anchor-positive and anchor-negative structural values;
- semantic target and source for each candidate role;
- an explicit sampling role such as positive or negative;
- both directional exclusion flags for the anchor-negative pair;
- the derived symmetric exclusion flag;
- sampling provenance and any relation/distance margins required by current consumers.

An exclusion may not occupy the direct positive slot. Generation must fail if such a row would be
created.

No loader may reconstruct exclusion status from distance zero, relation ID zero, the string
`excluded` in a structural field, or candidate position.

### 6.3 Graph compatibility

The existing graph loader may continue to consume legacy-named positive/negative distance,
relation, and margin columns. Those columns now contain honest structural values. New exclusion
fields carry exclusion meaning.

If direct schema compatibility is impractical, a narrow adapter may project the new training-pair
schema into the existing graph-loader input shape. This repair must not change graph sampling,
HGCN objectives, or semantic-retention behavior.

## 7. Runtime candidate types

### 7.1 Candidate occurrence and code identity

Two identities are required:

- `candidate_uid` is a collision-free occurrence identity within a training step. It is represented
  by fixed-width numeric components suitable for distributed gathering and includes origin rank,
  source sample, and source slot (or an equivalent collision-free encoding).
- `code_id` is the canonical NAICS identity used for semantic joins and deduplication.

Different occurrences of one code may have different UIDs. They are still duplicates for final
negative selection and are resolved deterministically by `code_id`.

### 7.2 `NegativeCandidateBatch`

An immutable candidate batch owns every aligned field with leading shape `[batch, candidate]`:

- candidate UID;
- code ID;
- embedding;
- structural distance and canonical pair relation ID joined for the current anchor-candidate pair;
- both directional exclusion flags and the symmetric flag;
- semantic target/source, sampling role, and sampling provenance;
- relation/distance margins and difficulty fields still used by existing interfaces;
- router gate probabilities or other candidate-level router outputs;
- validity mask;
- any runtime score or flag that must survive selection.

Code strings are recovered through the shared codebook rather than gathered as Python objects.

The type validates common leading dimensions, legal IDs, exclusion derivation, and validity before
selection. Candidate fields may not be passed around as unrelated parallel arrays after this
boundary.

### 7.3 `NegativeSelection`

Mining returns a typed selection containing:

- `source_indices[batch, selected]`;
- the corresponding source candidate UIDs;
- selection scores;
- a reason enum such as `EXCLUSION_QUOTA`, `GEOMETRIC`, `ROUTER`, or `BACKFILL`.

Miners do not return gathered or reordered embeddings.

`NegativeCandidateBatch.select(selection)` is the only final gather operation. It must:

1. verify shape and bounds;
2. reject any index whose source candidate is invalid;
3. verify the selected UID against the UID recorded by the selection;
4. gather every aligned field with the same indices;
5. return an immutable `SelectedNegativeBatch`.

This check prevents a valid set of indices from being accidentally applied to a stale or different
candidate pool.

## 8. Exclusion sampling policy

For an anchor with one or more explicit exclusions, final negative selection reserves exactly one
slot for an exclusion.

### 8.1 Deterministic rotation

Exclusions are sorted by canonical code identity. The selected exclusion index is:

```text
(stable_hash(global_seed, anchor_code_id) + epoch) modulo exclusion_count
```

`stable_hash` must be process- and platform-independent; Python's randomized built-in `hash()` is
not acceptable. An equivalent deterministic formula is acceptable only if it guarantees
reproducibility and cyclic coverage. A resumed run at the same epoch must choose the same exclusion.

### 8.2 Bounded representation

- The reserved exclusion cannot be removed by mining.
- Other exclusions are excluded from ordinary mining for that anchor during that selection, so the
  result contains exactly one exclusion rather than merely at least one.
- Mining fills the remaining `K - 1` slots from valid non-exclusion candidates.
- The reserved code is deduplicated against every proposal.
- If no exclusion exists, all `K` slots are filled normally.
- `K` must be at least one.

Candidate construction must supply at least `K` distinct valid codes under this policy. It may
expand or backfill from the valid candidate universe, but it may not repeat the final candidate as
padding. Insufficient unique candidates are a fatal data/configuration error with anchor identity
and counts in the message.

The quota replaces the current very large exclusion sampling weight in the repaired path. Exclusion
representation is therefore controlled once, rather than multiplied by both a sampling weight and
a special loss weight.

## 9. Loss contract

### 9.1 Contrastive loss

An explicit exclusion is always a valid repulsive negative:

- it remains in the contrastive denominator;
- it cannot be eliminated as a false negative;
- it cannot be used by an auxiliary attraction loss;
- it uses the ordinary negative loss weight;
- invalid padding never contributes.

False-negative handling obeys:

```text
effective_false_negative = pseudo_related AND NOT is_explicit_exclusion
```

Pseudo-related handling may still mask or attract eligible `UNKNOWN` candidates according to the
configured strategy.

### 9.2 Replace LambdaRank with `StructuralPreferenceLoss`

For an anchor, construct the candidate set from its positive plus its selected non-exclusion
negatives. For each unordered pair with unequal raw structural distance, orient the pair so that
candidate `i` is structurally closer than candidate `j` and apply:

```text
softplus((embedding_distance_i - embedding_distance_j + margin) / temperature)
```

This produces a positive derivative for `embedding_distance_i` and a negative derivative for
`embedding_distance_j`. Gradient descent therefore reduces the learned distance of the target-close
candidate and increases that of the target-far candidate.

The loss must:

- mask every comparison involving an explicit exclusion;
- mask invalid padding, self-candidates, and duplicate code identities;
- ignore structural ties within a configured numerical tolerance;
- include each unordered pair at most once;
- detach any optional importance weight from autograd;
- normalize valid comparisons per anchor and then average over anchors with at least one valid
  comparison;
- return a finite differentiable zero when an anchor or batch has no valid comparisons.

There is no NDCG computation and no learned-distance-dependent lambda. `LambdaRankLoss` is removed
from the active Stage-3 path rather than silently retained behind its old name.

### 9.3 Other losses

- Hierarchy preservation remains structural-only and consumes the rebuilt structural matrix.
  Exclusion never overwrites or reinterprets its target. Direct positive records are guaranteed not
  to be exclusions.
- Radius and router/load-balancing regularizers do not receive semantic targets. Where they operate
  over selected candidates, they respect candidate validity.
- A false-negative auxiliary attraction loss receives the same exclusion-cleared mask as the
  contrastive loss.

Contract violations in candidate identity, targets, or artifact metadata are fatal. They must not
be caught and converted into a logged warning plus a zero auxiliary loss.

## 10. Mining and training-step data flow

The required flow is:

```text
collated candidate entities
    -> optional distributed entity gather
    -> anchor-relative supervision join
    -> NegativeCandidateBatch
    -> exclusion reservation and miner scoring
    -> deterministic proposal merge and code deduplication
    -> one canonical selection
    -> false-negative handling
    -> all candidate-based losses
```

### 10.1 Collation

- Collation carries every candidate identity and supervision field.
- Variable candidate pools use an explicit invalid row with `valid_mask = false`.
- Collation never mutates an input sample in place and never pads by repeating its last candidate.
- Recent `all_candidates`/difficulty-sampling inputs become sources for the canonical candidate
  batch; they do not create a second parallel metadata path.

### 10.2 Local mining

Internal geometric and router scoring strategies operate on the same canonical pool. They may
produce score grids or typed indexed proposals, but may not gather fields independently. The public
hard-negative mining boundary returns the final `NegativeSelection`. Its selection coordinator:

1. installs the protected exclusion index;
2. merges strategy proposals;
3. deduplicates by code ID;
4. breaks ties by code ID and then candidate UID;
5. deterministically backfills from remaining valid unique non-exclusion candidates;
6. emits one final `NegativeSelection`.

Router gate probabilities are candidate fields and therefore follow the same final indices as
embeddings and codes.

### 10.3 False-negative handling

False-negative classification runs after final selection. If an implementation computes candidate
scores before selection for efficiency, those scores must be fields on `NegativeCandidateBatch`
and pass through the canonical gather. A pre-mining mask may never be paired with post-mining
embeddings.

### 10.4 Distributed mining

Only candidate-intrinsic data are gathered across ranks:

- embedding;
- code ID;
- occurrence UID;
- validity;
- candidate router outputs.

Structural distance, canonical structural relation, and exclusion direction are pair-dependent;
they cannot travel as properties of a candidate occurrence. After the global entity gather, each
local anchor joins against every gathered candidate code, canonicalizes the pair consistently with
the generated facts, and maps the two exclusion directions into the local anchor-candidate view.
Metadata computed relative to a remote candidate's original anchor may not be reused.

Duplicate global code IDs are collapsed before final selection with a deterministic occurrence-UID
tie-break.

## 11. Generated supervision bundle

### 11.1 Bundle identity

All rebuilt supervision artifacts belong to one immutable bundle identified by:

- a symbolic contract version, initially `stage3-supervision-v1`;
- a unique bundle ID for that generation event.

A top-level manifest records:

- contract and per-artifact schema versions;
- NAICS vintage;
- codebook order and fingerprint;
- input description and exclusion fingerprints;
- generator revision and material generation parameters;
- structural relation-ID mapping;
- artifact paths, hashes, row counts, and exclusion counts;
- bundle validation results.

Each Parquet artifact also carries its bundle ID and schema version in metadata. Generation writes
the manifest only after all artifacts pass validation. A configuration identifies one manifest as
the authoritative bundle entry point; loaders resolve member artifacts from it.

Artifacts from separate bundle IDs may not be mixed even if their schemas match.

### 11.2 Required rebuilds

The following are rebuilt or invalidated:

| Artifact | Action |
|---|---|
| `naics_distances.parquet` | Rebuild with untouched structural values and explicit exclusion fields |
| `naics_distance_matrix.parquet` | Rebuild from and reconcile against long-form structural facts |
| `naics_relations.parquet` | Rebuild without an exclusion structural relation |
| `naics_relation_matrix.parquet` | Rebuild from and reconcile against long-form relations |
| `naics_training_pairs/` | Rebuild with identities, semantic fields, exclusion provenance, and raw structure |
| Streaming and multi-epoch caches | Invalidate and regenerate |
| Curriculum difficulty thresholds | Regenerate from the rebuilt source distributions |
| Relation/triplet-derived downstream caches | Invalidate when their input fingerprint references an old artifact |

Description inputs, the canonical code registry, and tokenization caches may be retained when their
fingerprints match. Graph artifacts derived from changed relations or triplets are regenerated only
for input consistency; graph behavior remains out of scope.

Large generated artifacts remain ignored data products. Code, schema definitions, manifest logic,
and tests are committed; production bundles are generated by the data pipeline.

### 11.3 Validation before training

Training fails before model construction or checkpoint restoration when:

- an artifact lacks a contract version or bundle ID;
- bundle IDs differ;
- the codebook fingerprint or ordering differs;
- a matrix does not reconcile with its long-form source;
- a training row cannot join to all code identities and pair facts;
- exclusion derivation is inconsistent;
- a direct positive is an exclusion;
- an exclusion sentinel remains in a structural field.

There is no automatic fallback from the repaired path to legacy files.

## 12. Configuration contract

The repaired configuration introduces an authoritative supervision-manifest path and explicit
contract version. It replaces `loss.rank_order_weight` with a structurally named preference-loss
configuration, including weight, margin, temperature, and tie tolerance.

Configuration validation requires a positive temperature and nonnegative margin and tie tolerance.

The old high `phase1_exclusion_weight` is invalid in the repaired path because the one-slot quota
owns exclusion representation. The quota is fixed at one for this repair rather than exposed as a
new tuning dimension.

The repaired path rejects legacy `rank_order_weight` and unversioned supervision settings with a
migration message. It does not reinterpret the old key as the new loss automatically, because the
objectives have different semantics.

Recent candidate-pool and difficulty-sampler configuration remains supported, but its output must
enter the canonical candidate type before mining.

## 13. Checkpoint migration

### 13.1 New checkpoints

New checkpoints record:

- supervision contract version;
- artifact bundle ID;
- codebook fingerprint;
- structural-preference-loss version;
- mining-contract version.

Generated supervision matrices are loaded from the validated bundle rather than treated as
authoritative persistent checkpoint state.

Exact resume is allowed only when all contract identifiers and fingerprints match the runtime
configuration.

### 13.2 Legacy weights-only warm start

A checkpoint without the new contract identifiers cannot exact-resume. It may be used only through
an explicit weights-only migration mode that:

- loads compatible encoder, projection, MoE/router, and other model parameters from an allowlist;
- excludes loss modules and data-derived supervision buffers;
- discards optimizer and scheduler state;
- discards global step, epoch, curriculum, sampler, and mining state;
- reports loaded, skipped, missing, and unexpected parameter groups;
- starts a new run at epoch zero against a validated new bundle.

Unexpected incompatibilities outside the approved excluded groups are fatal. The mode must not
silently degrade into a broad `strict = false` load.

This path does not claim to reverse learning caused by the old objective. It is an explicit
initialization choice.

## 14. Legacy containment

An old configuration may run only through an explicit `legacy_containment` mode. It is not the
default and is not contract-compliant Stage-3 training.

Containment:

- disables LambdaRank and every structural-ranking term;
- disables hierarchy loss backed by contaminated matrices;
- disables hard-negative and router-based negative reordering;
- disables pseudo-related elimination or attraction that cannot protect exclusions;
- uses local, unmined contrastive negatives;
- preserves ordinary model MoE routing and supervision-independent regularizers;
- prominently tags logs and checkpoints as legacy containment.

Containment checkpoints cannot exact-resume into the repaired path. They may only be considered
through the same explicit weights-only migration process.

## 15. Error handling and observability

Contract failures are fatal and include actionable context such as artifact path, bundle ID, anchor
code ID, candidate counts, or mismatched UID. The system must not continue with partially aligned
metadata or silently disable a required loss.

Low-cardinality health counters may report:

- anchors with available exclusions;
- quota selections;
- duplicate candidates removed;
- invalid candidates ignored;
- deterministic backfills;
- artifact/checkpoint validation failures.

These are integrity diagnostics, not new model-evaluation metrics.

## 16. Verification strategy

### 16.1 Data generation and artifacts

A synthetic hierarchy fixture includes a pair that is structurally close but directionally
excluded. Tests assert:

- exclusion processing leaves its structural distance and relation unchanged;
- forward and reverse directional flags are correct;
- the symmetric flag is exactly their OR;
- long-form and matrix values reconcile;
- no distance/relation sentinel carries exclusion meaning;
- direct positives cannot be exclusions;
- semantic targets and sources are correct;
- every candidate identity joins to the codebook and pair facts;
- bundle/version mismatches fail before loading;
- fixed inputs and configuration produce deterministic outputs.

Sampling tests cover deterministic rotation, full cyclic coverage, exactly one selected exclusion
when available, none when unavailable, `K = 1`, uniqueness, and insufficient-candidate failure.

### 16.2 Losses

The primary structural-preference regression test sets a structurally closer candidate farther away
in learned space. Backpropagation must yield:

- positive gradient with respect to the closer candidate's learned distance;
- negative gradient with respect to the farther candidate's learned distance.

Gradient descent consequently corrects the inversion.

Additional tests cover:

- lower loss for correct order than inverted order;
- joint candidate/metadata permutation invariance;
- finite differentiable zero for ties or fully masked comparisons;
- exclusion, padding, self, and duplicate masking;
- no gradient through detached weights;
- per-anchor normalization;
- exclusion precedence over a pseudo-related mask;
- padding exclusion from contrastive and auxiliary losses.

The existing LambdaRank tests are replaced with behavioral tests for the new objective.

### 16.3 Mining and selection

Fixtures give each candidate distinguishable embedding, code, structural, exclusion, and gate values.
A forced reorder such as `[2, 0, 1]` must preserve one source UID across every selected field.

Tests cover:

- stale/wrong-pool UID rejection;
- invalid-candidate rejection;
- code-level deduplication;
- deterministic geometry/router merge;
- protected quota survival;
- no duplicate reserved exclusion;
- deterministic score ties and backfill;
- anchor-relative supervision after a distributed gather.

The distributed rule receives both a pure join unit test and a small CPU multi-process integration
test.

### 16.4 Training-step boundary

A spy loss records selected UIDs and metadata. A Stage-3 forward/backward step with mining enabled
asserts that:

- embeddings, codes, structural targets, exclusion flags, false-negative flags, and router fields
  share the same selected UIDs;
- false-negative handling uses selected order;
- a pseudo-related exclusion remains repulsive;
- structural preference never sees that exclusion;
- padding reaches no loss;
- all loss values and gradients are finite.

One regression fixture must reproduce the present failure: embeddings reorder while codes and masks
remain in original order. It must fail under the old parallel-array behavior and pass through the
canonical selector.

### 16.5 Configuration, checkpoint, and containment

Tests assert:

- a matching new checkpoint resumes exactly;
- a legacy or fingerprint-mismatched checkpoint cannot exact-resume;
- weights-only migration loads only allowed parameter groups and resets all training state;
- repaired configurations reject legacy ranking/exclusion-weight semantics;
- containment never invokes ranking, contaminated hierarchy supervision, negative reordering, or
  pseudo-related attraction;
- containment output is tagged and cannot resume into the repaired path.

## 17. Expected implementation surface

Planning should expect focused changes in these existing areas:

- `src/naics_embedder/data/compute_distances.py`
- `src/naics_embedder/data/compute_relations.py`
- `src/naics_embedder/data/create_triplets.py`
- `src/naics_embedder/text_model/dataloader/streaming_dataset.py`
- `src/naics_embedder/text_model/hard_negative_mining.py`
- `src/naics_embedder/text_model/loss.py`
- `src/naics_embedder/text_model/mixins/curriculum.py`
- `src/naics_embedder/text_model/mixins/loss.py`
- `src/naics_embedder/text_model/naics_model.py`
- `src/naics_embedder/utils/config.py`
- `src/naics_embedder/cli/commands/training.py`
- `conf/config.yaml` and relevant data configuration files
- graph-loader compatibility at its existing training-pair boundary
- corresponding unit and integration tests under `tests/`

A new focused supervision-contract module is appropriate for candidate types, selection validation,
and artifact manifest types. It should not become a general dumping ground for training utilities.

Recent upstream changes that introduce `all_candidates` and difficulty-based candidate pools must be
integrated as candidate sources rather than removed or allowed to bypass the typed contract.

## 18. Rollout constraints

1. New and legacy artifacts coexist; no production artifact is overwritten in place.
2. The repaired loader is enabled only with a fully validated new bundle.
3. Legacy execution requires explicit containment mode.
4. New checkpoints are never written with legacy or mixed supervision identifiers.
5. Before normal Stage-3 training is enabled, the forced-reorder regression, gradient-sign test,
   bundle validation, and one full training-step integration test must pass.

## 19. Final acceptance checklist

- [ ] Structural data contains no exclusion sentinel.
- [ ] Directional provenance and symmetric boundary are both present and validated.
- [ ] Exclusion sampling selects exactly one protected rotating candidate when available.
- [ ] Explicit exclusions cannot be false-negative masked or attracted.
- [ ] Structural preference excludes exclusions and has proven correction-direction gradients.
- [ ] Mining returns checked indices/UIDs, not gathered embeddings.
- [ ] Every selected metadata field is gathered exactly once from the canonical batch.
- [ ] Distributed joins use each local anchor.
- [ ] Rebuilt artifacts share one valid bundle ID.
- [ ] Exact resume enforces matching contract fingerprints.
- [ ] Legacy migration is explicit and weights-only.
- [ ] Containment is narrow, tagged, and cannot masquerade as repaired training.
- [ ] HGCN behavior, requested evaluation metrics, and unrelated curriculum work remain unchanged.
