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
