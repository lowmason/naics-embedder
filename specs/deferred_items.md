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
- [x] Review M6: in repaired mode, explicitly set legacy
      `data_loader.streaming.{distances,distance_matrix,relations,triplets}_parquet` values are
      ignored rather than rejected (spec §12). Full enforcement needs `Config.override`
      (src/naics_embedder/utils/config.py) to preserve fields-set, because it re-validates a full
      `model_dump`. A cheap interim: reject (or warn) when one of those paths differs from its
      default in repaired mode. Size: quick-fix. Done when: a repaired config that sets any of
      those paths to a non-default value fails validation with a migration message.
      → done in plan 7 (Task 12: `validate_supervision_contract` rejects a legacy streaming path
      set to anything but its default, with a migration message, also after `Config.override`).
- [x] Review M8: `data relations`, `data distances`, and `data triplets` each build a complete
      bundle (src/naics_embedder/cli/commands/data.py), so the old three-step sequence leaves
      three bundles; the per-stage stats PDFs were removed with the legacy wrappers (D8).
      Size: quick-fix. Revisit if: the stats reports are still wanted (restore as one
      bundle-level report) or users keep running the legacy sequence (make those commands print
      the notice and exit without building).
      → done in plan 7 (Task 12: each prints the migration notice and exits with status 1
      without building).
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
      Note, plan 7: under D* the thresholds are 7, 9 and 10, so they now gate HGCN's phases 1
      and 2. Still open: roadmap Stage 11 removes them.
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
- [x] Review Minor: when a bundle carries the `index_roles` member, `load_validated_bundle`
      (src/naics_embedder/supervision/artifacts.py) re-validates the table but does not require
      the build's `index_roles_one_role_per_entry`, `index_roles_examples_channel` and
      `index_roles_no_leakage` entries in `validation_results`, and it cannot re-run the leakage
      check itself. Deferred to roadmap Stage 5, whose contract version makes the member
      required. Size: quick-fix. Done when: the Stage 5 contract rejects a bundle whose
      `index_roles` member lacks those three validation results.
      → done in plan 7 (Task 10: every bundle carries the member, and the loader requires all 27
      validation results a build records, the three `index_roles_*` among them).
- [x] Review Minor: nothing pins the index-entry text until a bundle carries `index_roles`.
      conf/data/index_roles.csv pins (entry_id, code, role) and `DownloadConfig.index_sha256`
      pins the index file, but `entry_id` and the text come from `pl.read_excel` (calamine via
      fastexcel) in `_read_xlsx_bytes` (src/naics_embedder/data/download_data.py). A parser
      change that altered text within a row would pass `attach_role_text`, which catches only
      entries that move to another code. Stage 5's bundle member carries the text under its
      artifact hash. Size: quick-fix. Revisit if: polars or fastexcel is upgraded in uv.lock
      before Stage 5 builds the first bundle with the `index_roles` member.
      → done in plan 7 (Task 14's bundle `301cce28-539c-42ea-8781-496bbdcf511c` carries the
      member, entry text included, under its artifact hash).
- [ ] Review Minor: the scorer's `lorentz` distance (`lorentz_distances` in
      src/naics_embedder/panels/decoding.py) assumes curvature −1, while the text model's
      curvature is learnable. Roadmap Stage 6 must pass a curvature-aware callable to
      `OutcomePanel.score(..., distance=...)` or add a curvature parameter. Size: quick-fix.
      Done when: Stage 6 scores its arm under the trained curvature.

## 5-regressor-panel — 2026-09-24
- [x] Review Minor: nothing checks the held-out group table's hash when the regressor panel
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
      → done in plan 5 (a53d6e4, `heldout_groups_sha256`), on PR #114 after Codex's review
      raised it as P1.
- [x] Review Minor: a regressor read's log record names the text-only table by
      `ArmTables.text_only_fingerprint`, the `matrix_fingerprint` of the table's codes and
      values (src/naics_embedder/panels/regressor.py), while `tools text-only-table` records
      the parquet's file hash as `table_sha256` in `<stem>_provenance.json`
      (src/naics_embedder/panels/text_only.py). Matching a logged read to its table file means
      recomputing `matrix_fingerprint` from the parquet. Deferred because nothing matches them
      yet. Fix: record `table_sha256` in the read's detail too, or record the matrix fingerprint
      in the provenance. Size: quick-fix. Revisit if: Stage 4's tooling or a later stage has to
      match logged reads to text-only table files.
      → done in plan 6 (Task 1 records `matrix_fingerprint` in the provenance beside
      `table_sha256`; Task 5's store refuses a provenance naming another).
- [ ] Review Minor: `run_plan` (src/naics_embedder/panels/regressor.py) calls
      `standardized_ridge_path` (src/naics_embedder/panels/ridge.py), one SVD per call, for
      every tuning split and final fit. In the seen regime every task fits the same 2022 rows,
      so a validation read repeats one SVD 2 × repeats × folds times per comparator and level.
      The real stub run took about 30 s for levels 2–6. Deferred as performance only. Fix:
      cache the SVD per distinct fit row set within a read. Size: quick-fix. Revisit if: a
      Stage 4 seed sweep or a Stage 8–11 run is slowed by panel reads.
- [x] Review note, for Stage 4's decision statistic: a validation read scores each row once
      per repeat (five), and in the seen regime every repeat's predictions come from the same
      2022 fit with only the penalty's folds redrawn, so the repeats are not independent draws;
      group resampling should aggregate the repeats per row first. The held-out outer set also
      holds 2024 rows, which validation never scores, so the held-out statistic should be
      reported by feature_year as well. See specs/findings/regressor-panel-splits.md,
      section 6. Deferred because Open questions leaves the statistic to Stage 4. Size: design.
      Done when: Stage 4's plan aggregates repeats per row before resampling by group and
      reports the held-out regime by feature year, or records why not.
      → done in plan 6 (Task 2 averages each row's repeats, requiring every repeat, before Task
      3's group resampling; Task 6's record reports the held-out regime by feature year).

## 6-decision-rule-and-diagnostics — 2026-09-25
- [x] Review Important (final review, group A; deferred by the user): guards and statistic
      wrappers that no test exercises, each correct by reading. In
      src/naics_embedder/decision/decide.py: the repeated seed or run id refusal, the margin
      reference's pairing entry, the regressor `fingerprint` and `arm` log keys, the repeated
      arm names refusal and the two-arm minimum. In src/naics_embedder/decision/sweep.py: the
      repeated-seed refusal, and `detail.seed`, which no test asserts. `PanelItems.values`
      (decision/resampling.py) has no success-path test (a shuffled frame's item order and
      sums). tests/unit/test_diagnostics.py cannot tell `mean_over_sectors` from
      `mean_over_queries`, and does not check the MAP and NDCG wrappers' values. Deferred
      because nothing depends on them before real arms exist. Fix: one `pytest.raises` or
      assert per guard in tests/unit/test_decision.py, test_decision_sweep.py,
      test_decision_resampling.py and test_diagnostics.py. Size: quick-fix. Done when:
      deleting any listed guard fails a test, before Stage 7's first real `tools decide`.
      → done 2026-09-26 (/deferred quick fix): each guard's test fails when the guard is
      deleted, 13 of 13 checked; the MAP and NDCG values are worked by hand on a five-code tree.
- [x] Review Minor (final review, group B): the decision records and the artifact store trust
      their inputs more than a Lambda run can. (1) `decide` never checks that the margins
      record covers every panel with its `DECISION_STATISTIC`, and `MarginRecord.margin` raises
      a bare StopIteration for a missing panel (src/naics_embedder/decision/decide.py,
      records.py). (2) `write_record` checks that the path is free and then writes, so a racing
      writer overwrites and a crash leaves truncated JSON that blocks the path (records.py).
      (3) `ArtifactStore.put` verifies the copy only after `os.replace`, so a source changed
      mid-copy leaves a bad object that no later `put` replaces, and a failed copy leaves its
      staging file (decision/store.py). (4) `put_text_only` stores the table before it
      validates the provenance, and `decide`'s D9 check compares the record's copied fields
      instead of parsing the stored provenance (store.py, decide.py). (5) `resolve` and
      `put_frame` join paths unchecked, and the root is never made absolute, so
      `ArmRecord.store` is relative to the caller's working directory (store.py). (6) `tools
      margins` and `tools decide` refuse an existing `--output` only after every replicate
      (src/naics_embedder/cli/commands/tools.py). Every object's hash is checked again on
      `resolve`, so none of these corrupts a decision silently. Deferred as hardening. Fix: a
      margins check beside `check_pairing`; `path.open('x')`; hash the staging file before the
      rename and unlink it on failure; parse the stored provenance; resolve the root and refuse
      references that leave it; check `--output` first. Size: quick-fix. Done when: each is
      fixed or recorded as accepted, before Stage 7's reference sweep on Lambda.
      → done 2026-09-26 (/deferred quick fix): all six fixed, (2) only in part: a write that
      raises removes its file, but a process killed mid-write can still leave a partial
      record, to be deleted by hand (accepted). Beyond the proposals: (1) runs before any arm
      is checked; (4) stores nothing until the provenance checks out, and `decide` refuses a
      record whose text-only fields differ from its stored provenance, naming them; (5) also
      refuses a symlink inside the store that leads out of it.
- [x] Review Minor, for Stage 6: (1) `regressor_scores` (src/naics_embedder/decision/scores.py)
      counts each row's predictions with `pl.len()`, not its distinct `repeat` values, so a
      duplicated repeat beside a missing one passes; the panel's own `_predict` is the only
      producer today. (2) `coordinate_matrix`'s refusal of Lorentz points
      (src/naics_embedder/panels/regressor.py) says "the regressor panel takes the export
      form", which misleads when `tools diagnostics` raises it. Deferred because Stage 6 owns
      the export form and next touches both. Fix: count `pl.col('repeat').n_unique()`; word the
      refusal around the export form alone. Size: quick-fix. Done when: Stage 6's export lands
      with both changed.
      → done in plan 8 (Task 1: `regressor_scores` checks each row's distinct repeats as well as
      its count, and `coordinate_matrix`'s refusal names the export form alone).
- [ ] Review Minor, for Stage 10: the diagnostics report's interfaces
      (src/naics_embedder/metrics/diagnostics.py). (1) A codebook mismatch gives counts only,
      which confuses when the counts match, and `tools diagnostics` does not cast the
      codebook's `code` column to Utf8 (src/naics_embedder/cli/commands/tools.py). (2) The JSON
      keys differ in style: parent retrieval uses `1` and `5`, NDCG `@5`, `@10` and `@20`.
      (3) `codebook_codes` is optional, so only the CLI enforces "exactly the codebook's
      codes". Deferred because plan 6's Task 13 verified the current JSON and only the CLI
      calls the report today. Fix: name the first missing or extra code and cast; settle one
      key style; make `codebook_codes` required. Size: quick-fix. Done when: settled before
      Stage 10's keep-or-drop record reads the report.
- [ ] Review Minor: the table-reading `tools` commands (src/naics_embedder/cli/commands/tools.py)
      catch `OSError` and `ValueError` but not polars' errors, so a parquet without a `code`
      column ends in a `ColumnNotFoundError` traceback rather than a formatted refusal.
      Deferred because the command still refuses. Fix: add `pl.exceptions.PolarsError` to each
      such `except`. Size: quick-fix. Revisit if: a `tools` command shows a traceback on a
      malformed table.
- [ ] Review Minor (plan-mandated; deferred by the user): `tie_order`'s `TieUnresolvedError`
      (src/naics_embedder/decision/rule.py) fires on a tie anywhere among the surviving arms,
      even when first place is clear, so two identical arms entered under different names block
      a decision whose simpler third arm is obvious. Kept strict so a recorded order is never
      arbitrary. Fix: raise only when the first two survivors tie. Size: quick-fix. Revisit if:
      a tie below first place blocks a decision (Stage 8's nine cells are the first multi-arm
      decision).
- [ ] Pre-existing, found during plan 6: every Lorentz distance in
      src/naics_embedder/text_model/hyperbolic.py (`_lorentz_distance_compiled`,
      `LorentzDistance.batched_forward` and `_lorentz_distance_ops_compiled`, behind
      `LorentzOps.lorentz_distance`) computes √c·acosh(−⟨u,v⟩) on points the module's exp maps
      place on the hyperboloid ⟨x,x⟩ = −1/c; the geodesic there is acosh(−c⟨u,v⟩)/√c, so the
      distance is right only at c = 1 (at c = 4, a point at distance 1 from the origin reads
      0). Latent today: the text model runs at curvature 1.0 (conf/config.yaml), and HGCN,
      which calls these ops (src/naics_embedder/graph_model/hgcn.py,
      graph_model/curriculum/adaptive_loss.py) and makes its layer curvature a parameter by
      default, reads it through `.item()`, so it gets no gradient and stays at 1.0. Stage 7
      fixes the text stage's curvature at 1 with no parameter. Fix: acosh(clamp(−c⟨u,v⟩, 1))/√c,
      with a test at c ≠ 1. Size: quick-fix. Revisit if: any run sets curvature ≠ 1, or HGCN's
      layer curvature starts receiving gradient (Stages 10–11).

## 7-supervision-target-and-text — 2026-09-26
- [ ] Review Minor: the bundle build loads its tokenizer last. `generate_supervision_bundle`
      (src/naics_embedder/data/supervision_bundle.py:885-890) loads the backbone's tokenizer
      (`local_files_only=True`) only after distances, relations and pair facts are built, so a
      missing Hugging Face cache fails late, with transformers' `OSError`. Deferred from plan 7's
      final review: no bundle change, and bundle 301cce28 built. Fix: resolve the tokenizer
      before any artifact and name the cache fix in the error. Size: quick-fix. Done when: a
      build without the cached backbone fails before building its first artifact.
- [ ] Review Minor: the bundle build does not re-check the text channels.
      `verify_text_channels` (src/naics_embedder/data/download_data.py:636: no blanks, no
      `[EMPTY]`, provenance present) runs only in preprocess, not on the descriptions a bundle
      pins. Deferred from plan 7's final review. Fix: call it in
      `generate_supervision_bundle_from_frames` as a raising check only, since a new key in
      `REQUIRED_VALIDATION_RESULTS` would make bundle 301cce28 unloadable; the five-code fixtures
      in tests/fixtures/supervision.py then need `description_source`. Size: quick-fix. Done
      when: a build refuses descriptions that fail `verify_text_channels`.
- [ ] Review Minor: nothing ties the bundle's backbone to training at runtime.
      `manifest.input_window.backbone` (src/naics_embedder/supervision/schema.py:178) is never
      compared with `model.base_model_name` or `data_loader.tokenization.tokenizer_name`; only
      the static tests/unit/test_config.py::test_backbone_is_the_training_backbone links them.
      Deferred from plan 7's final review: whether a bundle's recorded window binds the training
      backbone is Stage 9's question, since Stage 9 trains several backbones. Size: design. Done
      when: Stage 9's plan decides whether training refuses a backbone other than the bundle's,
      and lands the check it chooses, verified against bundle 301cce28.
- [ ] Review Minor: the loader checks where each activity phrase sits, not its value.
      `validate_redirection_table` (src/naics_embedder/supervision/artifacts.py:439) would accept
      a rehashed table with a wrong phrase. Deferred from plan 7's final review. Fix: move
      `activity_phrase` (src/naics_embedder/data/redirections.py:62) to a torch-free module, since
      `supervision/` must not import `data/`, and recompute each row's phrase, skipping withheld
      rows. Size: quick-fix. Done when: the loader refuses a table whose phrase differs from its
      text's, verified against bundle 301cce28, before Stage 7 trains on the phrases as queries.
- [ ] Review Minor: a forced redraw of the role table skips the activity phrases.
      src/naics_embedder/data/index_role_table.py:103 calls `verify_role_leakage` without
      `extra_texts`, so `data roles --force` checks less than preprocess does. Nothing leaks:
      preprocess withholds a leaking row or fails closed. Deferred from plan 7's final review.
      Fix: pass the redirection table's activity phrases as `extra_texts`. Size: quick-fix.
      Revisit if: the role table is redrawn with `data roles --force`.
- [ ] Review Minor: the manifest omits the tokenizer revision behind its overflow counts.
      `InputWindowRecord` (src/naics_embedder/supervision/schema.py:178) records the backbone
      and window only. Bundle 301cce28's counts came from revision 1110a243, which
      specs/findings/supervision-target-and-text.md records. Deferred from plan 7's final review
      because the field changes bundle output. Size: quick-fix. Revisit if: a later stage
      rebuilds the bundle or bumps its contract.

## 8-shared-encoder-and-projection — 2026-10-03
- [ ] Stage 7: one training epoch reads every pre-sampled epoch.
      `NAICSDataModule` pre-samples `data_loader.n_epochs` epochs (default 100,
      src/naics_embedder/utils/config.py:853), and `RepairedMapDataset`
      (src/naics_embedder/text_model/dataloader/datamodule.py:656) serves all of them in one
      Lightning epoch, so `training.trainer.max_epochs=1` runs about 19,883 batches on bundle
      301cce28. Plan 8's Exit set `data_loader.n_epochs=1`, for 199 batches
      (specs/findings/shared-encoder-first-reading.md, section 1). Deferred by the user's ruling
      at plan 8's gate, as its final review recommended: Stage 7 owns the Lambda workflow and its
      training schedule. Size: design. Done when: Stage 7's spec fixes what one training epoch
      reads, and its Lambda config sets `data_loader.n_epochs` to match.
- [ ] Review Minor: the export and the arm encoder's reads have untested branches.
      In src/naics_embedder/text_model/export.py: the `batch_size < 1` refusal (:95), an absent
      curvature reading as 1 (:131), the claim that a refused table is never written (the
      curvature test in tests/unit/test_export.py asserts nothing about it), the exact provenance
      key set (`coordinates` and `generated_at` go unchecked), and a cap check that cannot tell a
      capped table from an uncapped one. In src/naics_embedder/cli/commands/tools.py:
      `export-table`'s ValidationError and OSError branches (:903), and `outcome-panel`'s
      bare-override and legacy refusals, its failure on a bad `--output` before the read is
      logged, and its payload's `fingerprint` key (:958-984). `exp_map_origin`'s `.cpu()`
      (src/naics_embedder/text_model/arm_encoder.py:57) never meets an MPS tensor in the tests:
      removing it fails nothing. Deferred from plan 8's final review as coverage gaps, not
      defects. Size: plan. Done when: each branch has a test, or a recorded ruling that it needs
      none.
- [ ] Review Minor: the export and the outcome read handle a few failures untidily.
      src/naics_embedder/text_model/export.py writes the table (:246) before it hashes the
      checkpoint and descriptions and writes the provenance (:273), so a failure between them
      leaves a table without provenance, which `ArmEncoder.from_files` then refuses. It records
      the checkpoint and descriptions paths as given, unresolved (:250, :258).
      src/naics_embedder/cli/commands/tools.py puts exception text and paths into Rich markup
      unescaped (:904, :972), and lets an `UnpicklingError` or `RuntimeError` from a corrupt
      checkpoint through as a traceback (:903, :971). `tools outcome-panel` refuses a blank
      `--purpose` only after the model loads. The provenance records no tokenizer name, so
      `from_files` checks the token window but not the tokenizer. docs/usage.md's outcome-panel
      paragraph does not say that a read refuses a table exported under another
      `data_loader.streaming.max_length`. Deferred from plan 8's final review and its re-review:
      none affects the Exit, whose export and reads shared one config. Size: plan. Done when:
      each case is fixed or ruled no-action.
- [ ] Review Minor: the encoder, fusion and cache tests leave gaps.
      No test changes only the field markers in a tokenization cache's sidecar
      (`_cache_identity`, src/naics_embedder/text_model/dataloader/tokenization_cache.py).
      `AttentionFusion`'s autocast case, where the scores are narrower than the input
      (src/naics_embedder/text_model/fusion.py:92-93), is untested, and
      tests/unit/test_fusion.py:88 runs its finite-gradient test in eval mode, which drops the
      train-mode (dropout) case. The tiny fixtures share one width: `WIDTH`, `TINY_HIDDEN` and
      the default dimension are all 8 (tests/unit/test_encoder.py:37,
      tests/fixtures/shared_encoder.py:31), which can hide a width and hidden-size mix-up.
      tests/unit/test_naics_model.py:941 builds its other contract without an encoder record and
      matches only 'bundle', so it passes for a reason other than the one it names. The HGCN
      feeder's encoder-mismatch refusal (src/naics_embedder/cli/commands/training.py) has no
      direct test, and no test runs two backward passes through one graph, the re-entry that
      `_MpsStateReplay` (src/naics_embedder/text_model/shared_encoder.py) exists for. Deferred
      from plan 8's reviews as coverage gaps, not defects. Size: plan. Done when: each gap has a
      test, or a recorded ruling that it needs none.
- [ ] Review Minor: `_pool_present` breaks under true half precision.
      src/naics_embedder/text_model/shared_encoder.py allocates the pooled rows in the
      projection weight's dtype, but `_pool_chunk` returns float32 (its mask is cast with
      `.float()`), so a `bf16-true` or `16-true` model raises "Index put requires the source and
      destination dtypes match". It is unreachable today: `create_trainer` picks `16-mixed` or
      `32-true`. Deferred from plan 8's final review, which kept the training path still during
      the Exit run. Fix: cast each chunk to the pooled dtype before the write. Size: quick-fix.
      Revisit if: a run can select `bf16-true` or `16-true` precision.
- [ ] Review Minor: a test needs a newer transformers than pyproject's floor.
      tests/unit/test_encoder.py:21 imports `GradientCheckpointingLayer` from
      `transformers.modeling_layers`, which the floor `transformers[torch]>=4.46` (pyproject.toml)
      lacks. The lock's 4.57.1 has it, and CI installs from `uv.lock`, so no environment the
      project builds is affected; the source itself runs at the floor. Deferred from plan 8's
      reviews. Fix: raise the floor in a deliberate re-lock, or guard the import. Size:
      quick-fix. Done when: pyproject's floor includes `transformers.modeling_layers`, or the
      test no longer imports it.
- [ ] Review Minor: code and docs polish left by plan 8's reviews.
      Code: the tokenization cache's load messages still name only the fingerprints, though the
      sidecar identity also covers format, markers and summaries
      (src/naics_embedder/text_model/dataloader/tokenization_cache.py:256-262); its cache
      annotations read `Dict[int, Dict[str, torch.Tensor]]`, but rows nest channel dicts (:254,
      :282); `LoggingMixin`'s docstring (src/naics_embedder/text_model/mixins/logging.py:28-37)
      omits its `fusion` dependency; three bare `'moe'` literals (mixins/curriculum.py:162,
      naics_model.py:606, :676) could use a constant beside `FUSIONS`; and the HGCN feeder
      (cli/commands/training.py) and the export (text_model/export.py) repeat a three-line
      code-row flow. Docs: docs/text_training.md could say near :69-73 that channels stay
      tail-truncated until Stage 6b (R15); its Cache Regeneration list (:414) omits format,
      markers and summaries; its "Exact Resume versus Weights-Only Migration" heading (:435)
      sits beside "nothing migrates it" (:92); "buffers" sits alone on :451; its "MoE gating"
      compiled op (:602, and CLAUDE.md:862) applies under `moe` only; tests/README.md omits the
      P8 pooler exemption; five statements of c = 1 (CLAUDE.md:41, README.md:212,
      docs/overview.md:59 and :193, docs/text_training.md:83) could say "by default"; and
      docs/overview.md:587 cites docs/sampling_architecture.md, deleted in a7517dd. Deferred
      from plan 8's reviews: each is true or harmless as written. Size: quick-fix. Done when:
      each is edited or ruled no-action.
