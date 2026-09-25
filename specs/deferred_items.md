# Deferred items

## 1-stage-3-supervision-integrity — 2026-09-23
- [x] Repo-wide ruff baseline (plan Task 14 Step 7 gate): `uv run ruff check src tests` reports
      27 errors (13 I001, 7 E501, 7 E741; 14 in src/naics_embedder/metrics/qcew.py, 2 in other
      source modules, 11 in test modules), all in files the Stage-3 branch never touched; the
      branch cleaned every file it touched (baseline was 86). Deferred to keep unrelated churn out
      of the branch. See specs/plans/completed/1-stage-3-supervision-integrity.md.
      Size: quick-fix. Done when: `uv run ruff check src tests` prints "All checks passed!".
      → retired 2026-09-23 during plan 2: already fixed before execution; initial and final
      repository-wide Ruff checks passed, as did the final full-repository YAPF check.
- [ ] Review I4: distributed selection cost. `NegativeSelectionCoordinator.select`
      (src/naics_embedder/supervision/selection.py) walks every global-pool entry per anchor in
      Python: about 105 ms/step at world size 8 (pool 12,288, batch 32), 21 ms at world 2, and
      under 1 ms single-GPU (the shipped `devices: 1`). The behavioural half of the finding
      (duplicate codes crowding the miners) is fixed; only cost remains. Size: plan. Done when: a
      vectorized coordinator (reusing `canonical_occurrence_mask`), checked by a randomized
      equivalence test against the current coordinator, brings world-8 selection under 20 ms.
      Revisit if: multi-GPU training with mining is run.
- [ ] Review M6: in repaired mode, explicitly set legacy
      `data_loader.streaming.{distances,distance_matrix,relations,triplets}_parquet` values are
      ignored rather than rejected (spec §12). Full enforcement needs `Config.override`
      (src/naics_embedder/utils/config.py) to preserve fields-set, because it re-validates a full
      `model_dump`. A cheap interim: reject (or warn) when one of those paths differs from its
      default in repaired mode. Size: quick-fix. Done when: a repaired config that sets any of
      those paths to a non-default value fails validation with a migration message.
- [ ] Review M8: `data relations`, `data distances`, and `data triplets` each build a complete
      bundle (src/naics_embedder/cli/commands/data.py), so the old three-step sequence leaves
      three bundles; the per-stage stats PDFs were removed with the legacy wrappers (D8).
      Size: quick-fix. Revisit if: the stats reports are still wanted (restore as one
      bundle-level report) or users keep running the legacy sequence (make those commands print
      the notice and exit without building).
- [ ] Review M9 (pre-existing): `_update_pseudo_labels`
      (src/naics_embedder/text_model/mixins/curriculum.py) wraps clustering in a broad
      `except Exception` that logs and continues with stale pseudo-labels. Size: quick-fix.
      Revisit if: pseudo-label clustering errors appear in training logs.
- [ ] Review M10: the distributed path relabels every gathered candidate's provenance as
      DISTRIBUTED_POOL, dropping GENERATED/BACKFILL provenance
      (src/naics_embedder/text_model/mixins/curriculum.py). Size: quick-fix. Revisit if:
      candidate provenance feeds a loss, sampler, or diagnostic.
- [x] Pre-existing: `NAICSDataModule.on_train_epoch_start`
      (src/naics_embedder/text_model/dataloader/datamodule.py) is not a Lightning datamodule
      hook, so `set_epoch` never runs; on-the-fly training pools and the Phase 1 difficulty mix
      stay at epoch 0 (the shipped precomputed mode is unaffected). Size: quick-fix. Done when:
      a Trainer-driven test shows training-dataset epochs advancing (persistent workers
      included) while validation pools stay epoch-independent.
      → retired 2026-09-23 during plan 2: already fixed by `0a13b65` using
      `TrainDatasetEpochCallback`; Trainer-driven epoch and persistent-worker regressions passed
      in both final full suites.
- [ ] Pre-existing: graph curriculum difficulty thresholds are degenerate on real data
      (phase1–3 max_distance all 99.0, because about 97% of pairs are cross-sector), as in
      legacy. See `compute_difficulty_thresholds` in
      src/naics_embedder/graph_model/curriculum/preprocess_curriculum.py. Size: design.
      Revisit if: HGCN curriculum phases are tuned or thresholds gate training.
- [x] Pre-existing: tests/unit/test_data_download.py::test_get_descriptions_filters_cross_references
      fails on main and on this branch (cause not investigated; outside the Stage-3 scope).
      Size: quick-fix. Done when: the test passes.
      → retired 2026-09-23: fixed by PR #75 (line-ending-independent split); test passes on main.

## 3-employment-statistics-coverage — 2026-09-24
- [ ] Review: the national-total employment comparison in `check_invariants`
      (scripts/employment_statistics_coverage.py) has no rounding allowance, so
      annual-average rounding (+46 in 2022, +27 in 2023) makes `run` exit 2 on the
      2022–2025 files. Kept as-is under the 2026-09-24 ruling (no re-run, no code
      change); evidence in specs/findings/employment-statistics-coverage.md,
      section 6. Fix: give that comparison the `cells/2 + 1` employment allowance
      that `_nested_excess` already applies, plus a fixture test. Size: quick-fix.
      Done when: `run` on the 2022–2025 files reports no invariant failure and a
      test pins the allowance.

## 4-outcome-panel-sealed-splits — 2026-09-24
- [ ] Review Minor: the index-roles checks outside `data roles` hardcode the near-duplicate
      threshold (9/10) and the examples floor (1): `download_preprocess_data`
      (src/naics_embedder/data/download_data.py), the bundle build
      (src/naics_embedder/data/supervision_bundle.py), `load_validated_bundle`
      (src/naics_embedder/supervision/artifacts.py) and `OutcomePanel.__init__`
      (src/naics_embedder/panels/outcome.py). `OutcomePanelConfig.near_duplicate_min_jaccard`
      and `examples_floor` (conf/data/outcome_panel.yaml) reach only the draw, so a redraw with
      other values would be checked against the defaults. Deferred because the committed table
      (conf/data/index_roles.csv) is frozen: the values cannot change without a redraw.
      Fix: read both from the draw's provenance (conf/data/index_roles_provenance.json), or
      document the two keys as frozen with the table. Size: quick-fix. Done when: every
      index-roles check uses the threshold and floor the table was drawn with, or the config
      documents them as frozen.
- [ ] Review Minor: when a bundle carries the `index_roles` member, `load_validated_bundle`
      (src/naics_embedder/supervision/artifacts.py) re-validates the table but does not require
      the build's `index_roles_one_role_per_entry`, `index_roles_examples_channel` and
      `index_roles_no_leakage` entries in `validation_results`, and it cannot re-run the leakage
      check itself. Deferred to roadmap Stage 5, whose contract version makes the member
      required. Size: quick-fix. Done when: the Stage 5 contract rejects a bundle whose
      `index_roles` member lacks those three validation results.
- [ ] Review Minor: nothing pins the index-entry text until a bundle carries `index_roles`.
      conf/data/index_roles.csv pins (entry_id, code, role) and `DownloadConfig.index_sha256`
      pins the index file, but `entry_id` and the text come from `pl.read_excel` (calamine via
      fastexcel) in `_read_xlsx_bytes` (src/naics_embedder/data/download_data.py). A parser
      change that altered text within a row would pass `attach_role_text`, which catches only
      entries that move to another code. Stage 5's bundle member carries the text under its
      artifact hash. Size: quick-fix. Revisit if: polars or fastexcel is upgraded in uv.lock
      before Stage 5 builds the first bundle with the `index_roles` member.
- [ ] Review Minor: the scorer's `lorentz` distance (`lorentz_distances` in
      src/naics_embedder/panels/decoding.py) assumes curvature −1, while the text model's
      curvature is learnable. Roadmap Stage 6 must pass a curvature-aware callable to
      `OutcomePanel.score(..., distance=...)` or add a curvature parameter. Size: quick-fix.
      Done when: Stage 6 scores its arm under the trained curvature.

## 5-regressor-panel — 2026-09-24
- [ ] Review Minor: nothing checks the held-out group table's hash when the regressor panel
      loads. `load_regressor_panel` and `RegressorPanel.from_sources`
      (src/naics_embedder/panels/regressor.py) read conf/data/regressor_heldout_groups.csv
      through `read_group_table` and log every read and opening under whatever hash the file
      has. Only the CI test (tests/unit/test_committed_regressor_groups.py) pins deddfd4c…, and
      `data regressor-groups` replaces the file only with `--force`. A hand edit gets a new
      fingerprint, which `SelectionLog.openings` counts as a first opening. Deferred because
      Stage 2's role table (conf/data/index_roles.csv) has the same guards and no load-time pin
      either. Fix: pin the sha256 in conf/data/regressor_panel.yaml and refuse a mismatch in
      `load_regressor_panel`. Size: quick-fix. Done when: loading the panel refuses a group
      table whose sha256 differs from the configured pin. It must land before Stage 12 opens
      either outer set, because the one-opening rule counts openings under this fingerprint.
- [ ] Review Minor: a regressor read's log record names the text-only table by
      `ArmTables.text_only_fingerprint`, the `matrix_fingerprint` of the table's codes and
      values (src/naics_embedder/panels/regressor.py), while `tools text-only-table` records
      the parquet's file hash as `table_sha256` in `<stem>_provenance.json`
      (src/naics_embedder/panels/text_only.py). Matching a logged read to its table file means
      recomputing `matrix_fingerprint` from the parquet. Deferred because nothing matches them
      yet. Fix: record `table_sha256` in the read's detail too, or record the matrix fingerprint
      in the provenance. Size: quick-fix. Revisit if: Stage 4's tooling or a later stage has to
      match logged reads to text-only table files.
- [ ] Review Minor: `run_plan` (src/naics_embedder/panels/regressor.py) calls
      `standardized_ridge_path` (src/naics_embedder/panels/ridge.py), one SVD per call, for
      every tuning split and final fit. In the seen regime every task fits the same 2022 rows,
      so a validation read repeats one SVD 2 × repeats × folds times per comparator and level.
      The real stub run took about 30 s for levels 2–6. Deferred as performance only. Fix:
      cache the SVD per distinct fit row set within a read. Size: quick-fix. Revisit if: a
      Stage 4 seed sweep or a Stage 8–11 run is slowed by panel reads.
- [ ] Review note, for Stage 4's decision statistic: a validation read scores each row once
      per repeat (five), and in the seen regime every repeat's predictions come from the same
      2022 fit with only the penalty's folds redrawn, so the repeats are not independent draws;
      group resampling should aggregate the repeats per row first. The held-out outer set also
      holds 2024 rows, which validation never scores, so the held-out statistic should be
      reported by feature_year as well. See specs/findings/regressor-panel-splits.md,
      section 6. Deferred because Open questions leaves the statistic to Stage 4. Size: design.
      Done when: Stage 4's plan aggregates repeats per row before resampling by group and
      reports the held-out regime by feature year, or records why not.
