# Structural Spearman Metric Integrity

**Status:** DESIGN APPROVED — awaiting written-spec review

**Next skill after written-spec approval:** `writing-plans` in a fresh session

## 1. Purpose

Repair the structural Spearman metric identified in finding #5 of
`reports/two-stage-methodology-review-2026-09-21.md`.

The current implementation assigns distinct ordinal ranks to tied values. For predicted pair
distances `[1, 2, 3, 4, 5, 6]` and tied structural targets `[1, 1, 1, 2, 2, 2]`, it returns `1.0`
instead of SciPy's `0.87831006565368`. Simultaneously permuting the same four nodes changes the
repository result as low as `0.5428571701049805`, while SciPy remains invariant.

Taxonomy distances are tie-heavy, so this is a metric-definition defect rather than a minor
numerical discrepancy. The repair establishes one canonical observation population, correct
average ranks, explicit invalid and undefined behavior, and versioned reporting at every public
call boundary.

## 2. Repository context and evidence

At design time:

- `main` is at `9fa0140`.
- The Stage-3 supervision-integrity repair is complete. Its completed spec and plan explicitly
  exclude Spearman and curvature work, so this repair does not alter that contract.
- `specs/deferred_items.md` contains no existing Spearman repair.
- The working tree has an unrelated user change in `conf/config.yaml`; implementation must
  preserve it.
- Active production callers are:
  - text validation in `text_model/mixins/validation.py`;
  - HGCN full evaluation in `graph_model/hgcn.py`;
  - the general runner in `metrics/runner.py`.
- `tools/embeddings_verification.py` does not currently report Spearman and will gain a
  report-only corrected value.
- Every inspected production caller supplies square pairwise matrices. No caller depends on
  rectangular input or a raw pair-list API.
- The current repaired Stage-3 bundle contains 2,125 codes and 2,256,750 unique unordered pairs.
  Its target matrix has only 12 distinct off-diagonal distances; 1,984,647 pairs have distance
  `99.0`.
- SciPy is already a production dependency. A local probe computed Spearman over all 2,256,750
  observations in approximately 0.22 seconds. The two raw `float32` observation vectors occupy
  approximately 18.05 MB before promotion and SciPy work buffers.

## 3. Goals

- Define one versioned structural Spearman contract.
- Count every unordered, non-self node pair at most once.
- Use average ranks for ties.
- Make the result invariant to simultaneous row and column permutations.
- Make behavior deterministic across supported tensor dtypes and devices.
- Distinguish malformed inputs from statistically undefined correlations.
- Route every caller through the same extraction and calculation boundary.
- Prevent corrected values from being confused with historical defective values.
- Add the corrected metric to Stage-4 verification without changing its acceptance gate.

## 4. Non-goals

This repair does not include:

- non-unit-curvature Lorentz distance corrections;
- propagation of curvature through the Stage-4 verifier;
- learnable-curvature gradient repair;
- HGCN architecture, objective, curriculum, or semantic-retention changes;
- QCEW or other external evaluation design;
- Stage-3 supervision changes;
- changes to cophenetic correlation, NDCG, distortion, or retrieval definitions;
- unrelated metric or visualization refactoring;
- rewriting historical metric artifacts.

## 5. Metric definition

### 5.1 Identity and public entry point

The symbolic definition identifier is:

```text
structural-spearman-v1
```

External numeric fields use:

```text
structural_spearman_v1
```

`HierarchyMetrics.spearman_correlation()` remains the public Python entry point for source
compatibility. It delegates to a focused structural-Spearman component and no longer performs
ranking itself.

### 5.2 Accepted inputs

The public method accepts two PyTorch tensors:

- predicted embedding distances;
- target structural distances.

Both tensors must:

- have the same shape;
- be two-dimensional and square;
- use a real, non-boolean, non-complex numeric dtype;
- contain finite off-diagonal observations;
- be symmetric within the tolerance for their source dtype.

Supported source dtypes are real floating and integer tensor dtypes. Each input matrix is checked
against its own transpose using its own source dtype tolerance:

| Source dtype | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| `float64` | `1e-7` | `1e-9` |
| `float32` | `1e-5` | `1e-7` |
| `float16`, `bfloat16` | `1e-3` | `1e-3` |
| Integer | exact | exact |

Inputs may originate on CPU, CUDA, or MPS. They are detached and moved to CPU before canonical
validation and calculation. The method is evaluation-only and has no gradient contract.
Cross-dtype determinism means that the same values representable in each supported dtype follow
the same algorithm and agree after the public `float32` result conversion. The metric does not
undo quantization that already changed values or ties in a low-precision source tensor.

A square submatrix over sampled nodes is supported. Rectangular matrices and raw pair vectors are
not public input forms. Empty, one-node, and two-node square matrices are structurally valid; they
may produce a statistically undefined result after extraction.

The diagonal is outside the observation population. Its values, including non-finite sentinels,
are ignored. A non-finite value in either orientation of an off-diagonal pair is invalid.

`min_distance` remains in the public signature for compatibility. It must be finite.

### 5.3 Canonical observation extraction

For each matrix independently:

1. Validate shape, dtype, off-diagonal finiteness, and mirrored symmetry.
2. Promote values to CPU `float64`.
3. Canonicalize each unordered pair as the arithmetic mean of its two mirrored entries.
4. Extract the strict upper triangle, `i < j`.

This yields exactly `N(N-1)/2` candidate observations for an `N x N` matrix. Mirrored pairs are
not separate samples and therefore receive no unintended double weight. Averaging after symmetry
validation ensures that harmless directional roundoff cannot make the score depend on node order.

Apply the existing target filter after extraction:

```text
canonical_target >= min_distance
```

`n_total` is the number of unique non-self observations before this filter. `n_pairs` is the
number remaining after it. Invalid values are checked before filtering so a threshold cannot hide
malformed data.

### 5.4 Tie handling and calculation

Canonical predicted and target observations are passed to `scipy.stats.spearmanr` as one-
dimensional CPU `float64` arrays. SciPy's average-rank tie convention is the production
implementation. Only exactly equal canonical `float64` observations form a tie; the symmetry
tolerance validates mirrored entries but does not merge merely near-equal observations into one
rank group. The SciPy p-value is not part of this metric contract and is discarded.

The wrapper detects known undefined cases before calling SciPy. For finite, non-constant vectors
with at least two observations, a non-finite SciPy result is an unexpected computation failure and
raises `RuntimeError` with the definition identifier and observation count.

The public compatibility wrapper returns correlation as a detached `torch.float32` scalar on
`HierarchyMetrics.device`. The calculation itself remains CPU `float64`, so input device does not
select another algorithm.

## 6. Result and error contract

The public result retains existing fields and adds explicit metadata:

| Field | Type | Meaning |
| --- | --- | --- |
| `correlation` | scalar tensor | Defined value or `NaN` when statistically undefined |
| `n_pairs` | integer | Observations after target filtering |
| `n_total` | integer | Unique non-self observations before filtering |
| `definition` | string | Always `structural-spearman-v1` |
| `status` | string | `defined` or `undefined` |
| `reason` | optional string | `null` when defined; exact undefined reason otherwise |

Malformed inputs raise `StructuralMetricInputError`, a public `ValueError` subclass. This includes:

- shape mismatch;
- non-square input;
- unsupported dtype;
- non-finite `min_distance`;
- non-finite off-diagonal values;
- asymmetry beyond the documented tolerance.

Valid but statistically undefined inputs return `status='undefined'` and a `NaN` correlation.
Classify them in this order:

1. Fewer than two filtered observations: `fewer_than_two_observations`.
2. At least two observations and both vectors constant: `constant_prediction_and_target`.
3. Only the prediction constant: `constant_prediction`.
4. Only the target constant: `constant_target`.

Malformed input is fatal at every public caller. Undefined correlation is non-fatal and is
reported explicitly.

## 7. Architecture and code boundaries

### 7.1 Focused metric component

Add `src/naics_embedder/metrics/structural_spearman.py` to own:

- the definition and external-field constants;
- `StructuralMetricInputError`;
- input validation;
- mirrored-pair canonicalization;
- strict-upper-triangle extraction and filtering;
- undefined-case classification;
- the SciPy calculation and structured result.

The component is independent of Lightning, HGCN, and artifact serialization.

### 7.2 Compatibility wrapper

Update `src/naics_embedder/metrics/core.py` so
`HierarchyMetrics.spearman_correlation()` delegates to the focused component and converts its
numeric result to the existing tensor-oriented API. Remove `_rank_tensor`; no other inspected code
uses it.

Export the definition constant and `StructuralMetricInputError` from
`src/naics_embedder/metrics/__init__.py`.

### 7.3 File-backed distances

`src/naics_embedder/utils/distance_matrix.py` currently replaces NaNs with zero. Remove that
sanitization so malformed file-backed observations reach the shared metric boundary unchanged.
Affected callers compute structural Spearman before other hierarchy metrics, ensuring invalid
matrices fail before partial hierarchy results are logged.

This is the only shared distance-loader behavior changed by this repair.

## 8. Caller and reporting contract

Every caller invokes `HierarchyMetrics.spearman_correlation()` directly. No caller ranks,
flattens, filters, or deduplicates pairs independently.

### 8.1 Text validation

`text_model/mixins/validation.py` computes structural Spearman immediately after embedding
distances and before other hierarchy metrics.

For a defined result, Lightning logs:

- `val/structural_spearman_v1`;
- `val/structural_spearman_v1_n_pairs`;
- `val/structural_spearman_v1_n_total`.

For an undefined result, it omits the numeric correlation scalar, logs the counts, and emits one
warning with the exact reason. `evaluation_metrics.json` uses these fields:

- `structural_spearman_v1`: number or JSON `null`;
- `structural_spearman_v1_n_pairs`;
- `structural_spearman_v1_n_total`;
- `structural_spearman_v1_status`;
- `structural_spearman_v1_reason`: string or `null`;
- `structural_spearman_v1_definition`: `structural-spearman-v1`.

The existing broad evaluation exception handler must re-raise `StructuralMetricInputError` rather
than logging and continuing.

### 8.2 HGCN full evaluation

`graph_model/hgcn.py` uses the same versioned scalar and count names. Lightning logs numeric
correlation only when defined. `training_log.json` stores the value as a number or JSON `null`, plus
status, reason, definition, `n_pairs`, and `n_total` under the existing `val_` history prefix.

`StructuralMetricInputError` propagates and fails validation. It is not converted into a skipped
optional metric.

### 8.3 General evaluation runner

`metrics/runner.py` replaces the top-level result key `spearman_correlation` with
`structural_spearman_v1`. Its value is the complete public result record. Malformed-input errors
propagate.

### 8.4 Stage-4 verifier

`tools/embeddings_verification.py` adds `structural_spearman_v1` to `pre`, `post`, and `delta`:

- `pre` and `post` contain a number or `null`;
- `delta` is computed only when both values are defined, otherwise it is `null`;
- `metric_metadata['structural_spearman_v1']` records the definition and the pre/post status,
  reason, `n_pairs`, and `n_total`.

The CLI renders an undefined value or delta as `N/A`. The existing cophenetic, NDCG, and local
retrieval checks remain the complete pass/fail gate. No Spearman threshold or CLI threshold option
is added.

### 8.5 Historical artifacts

Historical files are not rewritten. Documentation labels any unversioned fields as
`legacy-ordinal-rank-v0`, including:

- `spearman`;
- `spearman_correlation`;
- `val/spearman_correlation`;
- `val_spearman_correlation`.

Those values used order-sensitive ordinal ranks, are not valid tied-rank Spearman coefficients,
and are not directly comparable with `structural_spearman_v1`. New code does not dual-write a
legacy key, and no automatic numerical conversion is possible.

## 9. Implementation approach decision

### 9.1 Chosen: SciPy production calculation

Use `scipy.stats.spearmanr` after repository-owned validation and canonical observation
extraction.

Reasons:

- correct, established average-rank handling;
- SciPy is already a required dependency;
- approximately 0.22 seconds for the complete current NAICS observation population;
- one deterministic CPU implementation across CPU, CUDA, and MPS callers;
- substantially less custom statistical code to maintain and test.

### 9.2 Rejected: focused internal average-rank implementation

An internal CPU implementation could be compared independently with SciPy, but would duplicate
tie grouping, average-rank assignment, constant handling, and correlation numerics without
removing a dependency or improving the current workload materially.

### 9.3 Rejected: PyTorch-native device implementation

A device-native implementation could avoid the observation transfer, but would require custom
tie grouping and sort behavior across CPU, CUDA, and MPS while consuming accelerator memory for
temporary rank buffers. Differentiability is not required by any inspected caller.

## 10. Verification

### 10.1 Focused metric tests

Add focused tests that:

- compare matrix results with direct `scipy.stats.spearmanr` calls;
- lock the motivating result to approximately `0.87831006565368`;
- test all 24 simultaneous row/column permutations of the four-node example;
- repeat calls to prove deterministic output;
- verify `N(N-1)/2` total observations and no mirrored double weight;
- prove diagonal changes, including non-finite sentinels, do not affect the result;
- verify `min_distance`, `n_total`, and `n_pairs` semantics;
- verify near-symmetric averaging and permutation invariance;
- reject excessive asymmetry, shape mismatch, rectangular matrices, complex/bool tensors,
  non-finite thresholds, and non-finite off-diagonal values;
- cover each undefined reason and require a `NaN` scalar;
- parameterize supported floating and integer dtypes;
- compare conditional CUDA and MPS execution with the CPU oracle;
- assert the result has no autograd connection.

Tests that compare with SciPy primarily verify repository-owned extraction and adaptation. The
fixed numeric regression separately protects the motivating scientific result.

### 10.2 Public-boundary regressions

Add tie-heavy regression coverage for:

- `NAICSEvaluationRunner`;
- text validation logging and JSON serialization;
- HGCN full evaluation and training history;
- Stage-4 verifier pre/post/delta output and undefined handling;
- Stage-4 CLI rendering;
- file-backed non-finite distances reaching and failing at the metric boundary.

Each boundary test asserts the versioned key is present and no new legacy key is emitted.

### 10.3 Repository verification

Run:

- focused structural metric and caller suites;
- the complete pytest suite;
- formatting and ruff checks on every touched Python file;
- repository-wide ruff to confirm no new failures beyond the explicitly deferred baseline in
  `specs/deferred_items.md`;
- strict MkDocs build;
- `git diff --check`.

Do not add a wall-clock pytest assertion. Re-run the full-scale benchmark manually during
implementation and record any material regression.

## 11. Documentation and migration

Update `README.md`, `docs/overview.md`, `docs/text_training.md`, and `docs/hgcn_training.md` to:

- define `structural-spearman-v1`;
- state the strict-upper-triangle, mirrored-average, diagonal-exclusion, and average-rank rules;
- document undefined and invalid behavior;
- show the new artifact fields;
- label unversioned historical fields as `legacy-ordinal-rank-v0`;
- state that old and corrected values are not directly comparable;
- state that Stage-4 verification reports but does not gate on structural Spearman.

Generated API pages continue to derive from source docstrings. No dependency migration is needed.

## 12. Curvature boundary

Structural Spearman consumes distance matrices; it does not establish whether those distances were
computed with the correct Lorentz formula.

Near-term model comparisons covered by this repair remain fixed at curvature `1.0`:

- Stage-4 verification remains explicitly fixed at `1.0`;
- HGCN full evaluation remains explicitly fixed at `1.0`;
- text-stage comparison configurations explicitly retain `loss.curvature: 1.0`.

The implementation neither repairs nor generalizes non-unit-curvature evaluation. A
`structural_spearman_v1` value computed from defective non-unit-curvature distances is not made
reliable by this rank-correlation repair.

## 13. Acceptance criteria

The repair is complete when:

1. The motivating tied-target example matches SciPy and remains unchanged under every simultaneous
   node permutation.
2. Every unordered non-self pair contributes exactly once.
3. Supported dtypes and devices use one deterministic CPU-`float64` calculation contract.
4. Malformed inputs raise `StructuralMetricInputError` at every public boundary.
5. Valid but undefined inputs return explicit status/reason metadata and serialize as JSON `null`.
6. Every active caller emits `structural_spearman_v1` and no new legacy key.
7. Stage-4 verification reports pre/post/delta values without changing its release gate.
8. Historical unversioned values are documented as `legacy-ordinal-rank-v0` and non-comparable.
9. Focused tests, the full suite, touched-file quality checks, strict docs build, and diff checks
   pass, with no new repo-wide lint failures.
10. No curvature, HGCN-design, QCEW, Stage-3 supervision, or unrelated metric behavior changes.
