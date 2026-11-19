# Deferred items

## 1-stage-3-supervision-integrity — 2026-09-23
- [ ] Pre-existing: graph curriculum difficulty thresholds are degenerate on real data
      (phase1–3 max_distance all 99.0, because about 97% of pairs are cross-sector), as in
      legacy. See `compute_difficulty_thresholds` in
      src/naics_embedder/graph_model/curriculum/preprocess_curriculum.py. Size: design.
      Revisit if: HGCN curriculum phases are tuned or thresholds gate training.
      Note, plan 7: under D* the thresholds are 7, 9 and 10, so they now gate HGCN's phases 1
      and 2. Still open: roadmap Stage 11 removes them.
      → triggered 2026-11-18: plan 7's note above records that under D* the thresholds (7, 9, 10) gate HGCN's phases 1 and 2.
      → held 2026-11-18: the graph stage is parked; redesign Stages 6–7 (HELD on embedding-data gates) take it if they reuse `graph_model/`.

## 3-employment-statistics-coverage — 2026-09-24
- [x] Review: the national-total employment comparison in `check_invariants`
      (scripts/employment_statistics_coverage.py) has no rounding allowance, so
      annual-average rounding (+46 in 2022, +27 in 2023) makes `run` exit 2 on the
      2022–2025 files. Kept as-is under the 2026-09-24 ruling (no re-run, no code
      change); evidence in specs/findings/employment-statistics-coverage.md,
      section 6. Fix: give that comparison the `cells/2 + 1` employment allowance
      that `_nested_excess` already applies, plus a fixture test. Size: quick-fix.
      Done when: `run` on the 2022–2025 files reports no invariant failure and a
      test pins the allowance.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → done 2026-11-18 (/deferred quick fix): `check_invariants` gives the national-total employment comparison the `cells/2 + 1` allowance (wages none), tests pin both, and `run` on the 2022–2025 files exits 0 with no invariant failure and the same decision block.

## 4-outcome-panel-sealed-splits — 2026-09-24
- [x] Review Minor: the index-roles checks outside `data roles` hardcode the near-duplicate
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
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → done 2026-11-18 (/deferred quick fix): conf/data/outcome_panel.yaml and `OutcomePanelConfig` document both keys as frozen with the table, and tests/unit/test_committed_index_roles.py pins the config, the draw's provenance and the checks' defaults to agree.
- [x] Review Minor: the scorer's `lorentz` distance (`lorentz_distances` in
      src/naics_embedder/panels/decoding.py) assumes curvature −1, while the text model's
      curvature is learnable. Roadmap Stage 6 must pass a curvature-aware callable to
      `OutcomePanel.score(..., distance=...)` or add a curvature parameter. Size: quick-fix.
      Done when: Stage 6 scores its arm under the trained curvature.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → retired 2026-11-18: plan 10 removed the text model's curvature (conf/config.yaml has no curvature key; see the plan 10 note under the Lorentz-distance item), so curvature −1 is the trained geometry.

## 5-regressor-panel — 2026-09-24
- [ ] Review Minor: `run_plan` (src/naics_embedder/panels/regressor.py) calls
      `standardized_ridge_path` (src/naics_embedder/panels/ridge.py), one SVD per call, for
      every tuning split and final fit. In the seen regime every task fits the same 2022 rows,
      so a validation read repeats one SVD 2 × repeats × folds times per comparator and level.
      The real stub run took about 30 s for levels 2–6. Deferred as performance only. Fix:
      cache the SVD per distinct fit row set within a read. Size: quick-fix. Revisit if: a
      Stage 4 seed sweep or a Stage 8–11 run is slowed by panel reads.

## 6-decision-rule-and-diagnostics — 2026-09-25
- [x] Review Minor, for Stage 10: the diagnostics report's interfaces
      (src/naics_embedder/metrics/diagnostics.py). (1) A codebook mismatch gives counts only,
      which confuses when the counts match, and `tools diagnostics` does not cast the
      codebook's `code` column to Utf8 (src/naics_embedder/cli/commands/tools.py). (2) The JSON
      keys differ in style: parent retrieval uses `1` and `5`, NDCG `@5`, `@10` and `@20`.
      (3) `codebook_codes` is optional, so only the CLI enforces "exactly the codebook's
      codes". Deferred because plan 6's Task 13 verified the current JSON and only the CLI
      calls the report today. Fix: name the first missing or extra code and cast; settle one
      key style; make `codebook_codes` required. Size: quick-fix. Done when: settled before
      Stage 10's keep-or-drop record reads the report.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → done 2026-11-18 (/deferred quick fix): a mismatch names the first missing and the first extra code, `tools diagnostics` casts the codebook's `code` to Utf8, parent retrieval's keys are `@1` and `@5` like NDCG's, and `codebook_codes` is required.
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
      → plan 12: the nine-arm decision raised no TieUnresolvedError; the rule stays strict.
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
      → plan 10: the text stage has no curvature (Tasks 4, 9 and 11), so its half cannot fire; the
      HGCN half stays open behind `LorentzOps.lorentz_distance`.

## 7-supervision-target-and-text — 2026-09-26
- [x] Review Minor: the bundle build loads its tokenizer last. `generate_supervision_bundle`
      (src/naics_embedder/data/supervision_bundle.py:885-890) loads the backbone's tokenizer
      (`local_files_only=True`) only after distances, relations and pair facts are built, so a
      missing Hugging Face cache fails late, with transformers' `OSError`. Deferred from plan 7's
      final review: no bundle change, and bundle 301cce28 built. Fix: resolve the tokenizer
      before any artifact and name the cache fix in the error. Size: quick-fix. Done when: a
      build without the cached backbone fails before building its first artifact.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → done 2026-11-18 (/deferred quick fix): the build resolves the tokenizer before it reads any input, and a missing cache raises an OSError that names the command which caches it.
- [x] Review Minor: the bundle build does not re-check the text channels.
      `verify_text_channels` (src/naics_embedder/data/download_data.py:636: no blanks, no
      `[EMPTY]`, provenance present) runs only in preprocess, not on the descriptions a bundle
      pins. Deferred from plan 7's final review. Fix: call it in
      `generate_supervision_bundle_from_frames` as a raising check only, since a new key in
      `REQUIRED_VALIDATION_RESULTS` would make bundle 301cce28 unloadable; the five-code fixtures
      in tests/fixtures/supervision.py then need `description_source`. Size: quick-fix. Done
      when: a build refuses descriptions that fail `verify_text_channels`.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → done 2026-11-18 (/deferred quick fix): `generate_supervision_bundle_from_frames` calls `verify_text_channels` first, as a raising check with no new validation key, and the three fixture description frames carry `description_source`.
- [ ] Review Minor: nothing ties the bundle's backbone to training at runtime.
      `manifest.input_window.backbone` (src/naics_embedder/supervision/schema.py:178) is never
      compared with `model.base_model_name` or `data_loader.tokenization.tokenizer_name`; only
      the static tests/unit/test_config.py::test_backbone_is_the_training_backbone links them.
      Deferred from plan 7's final review: whether a bundle's recorded window binds the training
      backbone is Stage 9's question, since Stage 9 trains several backbones. Size: design. Done
      when: Stage 9's plan decides whether training refuses a backbone other than the bundle's,
      and lands the check it chooses, verified against bundle 301cce28.
      → acknowledged 2026-11-15 (claude/naics-embedder-economic-candidates-a9d499): they wait for the next /deferred pass, which the owner will run after this PR.
      → held 2026-11-18: embedding roadmap Stage 9's questions moved to redesign roadmap Stage 5, which is HELD on embedding-data gates G2–G5; its plan decides this.
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
      removing it fails nothing, and the same holds for the flat heads' `read_points`
      (src/naics_embedder/text_model/heads.py:173, :222; plan 12's review M4). Deferred from plan
      8's final review as coverage gaps, not
      defects. Size: plan. Done when: each branch has a test, or a recorded ruling that it needs
      none.
      → partly mooted by plan 10: the curvature branches left with `require_unit_curvature`
      (Task 11), and the cap check is a check that the table holds the head's bounded tangent
      (Task 4). The other branches stay open.
      → partly done in plan 12: each geometry's coordinates are checked in the provenance (Task 4);
      `generated_at`, the exact key set and the other branches stay open.
      → planned 2026-11-18: specs/plans/18-export-and-encoder-hardening.md
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
      → partly done in plan 9 (Task 7): the provenance records the tokenizer, and `from_files`
      refuses a table exported with another. The other cases stay open.
      → planned 2026-11-18: specs/plans/18-export-and-encoder-hardening.md
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
      → partly done in plan 9 (Task 1): `test_a_cache_built_under_other_markers_is_rebuilt`
      (tests/unit/test_tokenization_cache.py) changes only the markers. The other gaps stay open.
      → planned 2026-11-18: specs/plans/18-export-and-encoder-hardening.md
- [ ] Review Minor: `_pool_present` breaks under true half precision.
      src/naics_embedder/text_model/shared_encoder.py allocates the pooled rows in the
      projection weight's dtype, but `_pool_chunk` returns float32 (its mask is cast with
      `.float()`), so a `bf16-true` or `16-true` model raises "Index put requires the source and
      destination dtypes match". It is unreachable today: `create_trainer` picks `16-mixed` or
      `32-true`. Deferred from plan 8's final review, which kept the training path still during
      the Exit run. Fix: cast each chunk to the pooled dtype before the write. Size: quick-fix.
      Revisit if: a run can select `bf16-true` or `16-true` precision.
- [x] Review Minor: a test needs a newer transformers than pyproject's floor.
      tests/unit/test_encoder.py:21 imports `GradientCheckpointingLayer` from
      `transformers.modeling_layers`, which the floor `transformers[torch]>=4.46` (pyproject.toml)
      lacks. The lock's 4.57.1 has it, and CI installs from `uv.lock`, so no environment the
      project builds is affected; the source itself runs at the floor. Deferred from plan 8's
      reviews. Fix: raise the floor in a deliberate re-lock, or guard the import. Size:
      quick-fix. Done when: pyproject's floor includes `transformers.modeling_layers`, or the
      test no longer imports it.
      → done 2026-11-18 (/deferred quick fix): the test guards the import and skips its one checkpointing test without the class, with no re-lock, since redesign Stage 3 reads under uv.lock 6de4fbdc.
- [x] Review Minor: code and docs polish left by plan 8's reviews.
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
      → partly done in plan 9: the cache's load messages name each sidecar entry that differs
      (Task 1), and docs/text_training.md describes the summaries where the R15 note would have gone
      and lists format, markers and summaries under Cache Regeneration (Task 9). The rest stays
      open.
      → partly mooted by plan 10: no curvature is configurable, so the five c = 1 statements are
      exact (Tasks 9–11), and the weights-only heading left with D2 (Tasks 2 and 16). The rest stays
      open.
      → retired 2026-11-18: the owner ruled the rest no-action, since the item records each part as true or harmless as written; the `sampling_architecture` link, the "MoE gating" line and mixins/curriculum.py are already gone.

## 9-window-fitting-summaries — 2026-10-04
- [ ] Review Minor: building and parsing the summaries leave untested and untidy branches.
      In src/naics_embedder/panels/window_summaries.py, no test pins P5's level-2 guards in
      `text_units` (`_NO_BREAK`, and parenthesis balance when an over-long sentence is cut at
      clauses). Two hand-traced cases would, with budgets in words as the tests count them: 'Farms
      in the U.S. grow corn; ranches raise cattle. Others are not.' at budget 7, and 'Farms grow
      (corn; wheat) here; ranches raise cattle. Others are not.' at budget 6.
      `_parse_window_summaries` and `_is_extract` handle a malformed artifact thinly: a null summary
      raises TypeError rather than ValueError, an empty pinned file raises polars' NoDataError, the
      header goes unchecked because the schema applies by position, and an empty summary passes
      every check when its source splits with a trailing empty piece, so the channel reads as
      absent. Only a hand-edited, re-pinned artifact reaches these, since the build cannot emit
      them; one guard (header names equal to `SUMMARIES_SCHEMA`, no nulls, no blank summary,
      NoDataError raised as ValueError) would close them. In
      src/naics_embedder/data/window_summaries.py, `select_units`' per-candidate fit filter is
      untested: vectors [[1, 0], [0, 1], [1, 1]], weights [1, 1, 2], sizes [1, 1, 3] and budget 2
      should select [0, 1] with no plateau. Every build test patches `backbone_embedder`, so it has
      no test; a tiny BERT reading one text alone and in a padded batch would do. Its `embed`
      neither calls `model.eval()` nor moves tokens to the model's device, relying on
      `load_backbone`, while the provenance writes device cpu and dtype float32 as constants
      (:389-390). Nothing pins that each unit is embedded once (P8). `summary_rows`' Raises omits
      `select_units`' "no unit fits" ValueError. `generate_window_summaries` annotates its tokenizer
      `Any` with a None default rather than Optional, silently replaces a caller's tokenizer and
      revision when `model` is None, and writes the provenance after the move, outside its
      try/finally (:316, :331). The `data summaries` command
      (src/naics_embedder/cli/commands/data.py) lets `PackageNotFoundError` past its except clause
      for OSError and ValueError (:283) and prints the error into Rich markup unescaped (:284). Its
      test in tests/unit/test_cli_commands.py cannot fail on the default backbone, since MiniLM is
      every default, and no test passes `--force`, `--descriptions` or `--output` or reaches that
      clause. tests/unit/test_window_summaries_build.py checks that a failed build leaves no
      artifact only from an empty conf/, never that a failed `force=True` rebuild leaves a good
      artifact unchanged. The text-only builder (src/naics_embedder/panels/text_only.py) has no
      no-pin refusal test, though the cache has one through the same resolver call, and a
      descriptions frame without a channel column now raises ColumnNotFoundError in the cache build
      instead of encoding the channel absent. Deferred at plan 9's gate: the build ran once, for the
      committed artifact that every reader re-checks, and none of these changes what a reader reads.
      Size: plan. Done when: each case is tested, fixed or ruled no-action.
- [ ] Review Minor: the summaries' readers, tests and docs leave polish and coverage gaps.
      Tests: `_identity_mismatch`
      (src/naics_embedder/text_model/dataloader/tokenization_cache.py:249) has untested message
      branches: no sidecar, an unreadable or non-object sidecar, an '<absent>' text, and several
      differing keys. `ArmEncoder.from_files` (src/naics_embedder/text_model/arm_encoder.py) checks
      in P11's order, but no test has both the tokenizer and the summaries wrong (the tokenizer's
      message should win), or a pre-6b table with a wrong checkpoint ("before Stage 6b" should win).
      No `run_seed_sweep` test gives a seed a provenance that describes another table, such as a
      `table_sha256` of 64 zeros, which should be refused as 'describes another file'.
      tests/unit/test_arm_encoder.py repeats a read-mutate-write of the provenance about seven
      times, which a helper would shorten; tests/unit/test_decision_store.py:166's parametrize has
      no `ids=`; the MiniLM name is defined in eight test modules, tests/conftest.py:51 among them;
      and the `real_window_summaries` marker is registered in tests/conftest.py's
      `pytest_configure`, while pyproject.toml registers the others. Code: the tokenizer refusal in
      `from_files` has no remedy sentence; `check_seed_table` and `check_text_only`
      (src/naics_embedder/decision/decide.py) print two unnamed 5-tuples, and `D9_FIELDS` and
      `_arm_reads` are parallel orderings; and the Raises of `_build_tokenization_cache`
      (tokenization_cache.py:72-77) and `build_text_only_table`
      (src/naics_embedder/panels/text_only.py:183-186) give `check_window`'s refusal only as a
      max_length over the window, though `trained_window` (src/naics_embedder/utils/input_window.py)
      also raises for a backbone with no recorded window. Docs: "a channel text over the window is
      read as its summary" remains at docs/usage.md:241, docs/api/input_window.md:4,
      src/naics_embedder/cli/commands/data.py:257, and the docstrings at data/window_summaries.py:4,
      panels/window_summaries.py:4, panels/text_only.py:175, tokenization_cache.py:67 and
      utils/input_window.py:9. A title over the window is refused, not summarized, so none is wrong
      under MiniLM, where no title is over it. Deferred at plan 9's gate: each is true or harmless
      as written, or a coverage gap rather than a defect. Size: plan. Done when: each is tested,
      edited or ruled no-action.
- [ ] Watch: an export keys the summaries on its tokenizer, a text-only table on its backbone.
      src/naics_embedder/text_model/export.py records
      `summaries_identity(token_config.tokenizer_name)`, and src/naics_embedder/panels/text_only.py
      records `summaries_identity(backbone)`; one `ArmSpec.summaries_sha256` is checked against
      both, by `check_seed_table` and `check_text_only` (src/naics_embedder/decision/decide.py). The
      two agree while the tokenizer's name is the backbone's, as for MiniLM. Where they differ, a
      mismatch is refused, but the refusal shows two 5-tuples whose summaries differ instead of
      naming the tokenizer and backbone split. The sweep's per-seed check reads the export through
      `provenance_fields` (src/naics_embedder/decision/store.py), which skips its `tokenizer`, so it
      compares tokenizers only through the summaries' sha256, and two unpinned names compare equal,
      null with null. Deferred at plan 9's gate, keeping the code as built: no backbone the roadmap
      runs has a tokenizer of another name. Size: plan. Revisit if: Stage 9 admits a backbone whose
      tokenizer name differs from its own.

## 10-objective-anchors-and-live-radius — 2026-10-07
- [ ] HGCN feeder precision (completion gate candidate): `generate_embeddings_from_checkpoint` in
      src/naics_embedder/cli/commands/training.py still writes float32 head Lorentz points. The text
      campaign uses bounded tangent export and CPU float64 reconstruction, so this did not affect
      its panels. Stage 11 owns the feeder. Size: plan. Done when: the feeder writes the float64
      origin map of the bounded tangent and fixture tests check manifold residuals at large observed
      radii, without changing the text objective.
      → held 2026-11-18: the graph stage is parked; redesign Stages 6–7 (HELD on embedding-data gates) take it if they reuse `graph_model/`.
- [ ] Large-radius overflow watch (completion gate candidate): `polar_distance` in
      src/naics_embedder/text_model/hyperbolic.py multiplies float32 sinh terms, while
      `model.radius_bound` in src/naics_embedder/utils/config.py accepts any finite positive value.
      Overflow becomes possible around R = 44; this campaign used R = 8 and every all-pairs report
      passed. Size: quick-fix. Revisit if: an arm proposes a bound near or above that range; then
      restrict the supported bound or add a numerically stable form and a meaningful boundary test
      before that arm trains.
- [ ] Older-kept-checkpoint continuation (completion gate candidate): CLI exact resume can accept a
      kept epoch older than `last.ckpt` through src/naics_embedder/utils/training.py and
      cli/commands/training.py, rewinding monitor and summary histories. The approved operator path
      resumes only last, and no campaign run used this rewind. Size: plan. Done when: exact
      continuation refuses a checkpoint below the experiment last epoch before pruning or model/data
      construction, with tests for unchanged last continuation and preserved histories on refusal.
- [ ] Monitor/cache test coverage (Task 8, completion gate candidate): tests/conftest.py
      selection-log guard wiring lacks a pytester-level test; tests/unit/test_monitor.py does not
      exercise a non-default `batch_size` in `refresh_code_cache` or `LiveEncoder`, and
      device-placement assertions need MPS. Deferred as fixture coverage, not an observed campaign
      defect. Size: plan. Done when: subprocess/session guard tests prove unauthorized reads fail
      and allowed reads pass, non-default batch chunks are checked, and a CPU placement test runs in
      CI.
- [ ] HGCN metric mutation coverage (Task 10, completion gate candidate): pre-existing mutants of
      `LorentzOps.lorentz_distance` in src/naics_embedder/text_model/hyperbolic.py, taxonomy code
      slicing in data/positive_sampling.py and metrics/core.py statistics survived the suite in
      Phase 1. The new text training path uses polar distances and does not select on these
      statistics. Stage 11 owns graph verification. Size: plan. Done when: independently worked
      geometry/taxonomy/metric examples fail the recorded mutants, or each mutant has a recorded
      no-action ruling.
      → held 2026-11-18: the graph stage is parked; redesign Stages 6–7 (HELD on embedding-data gates) take it if they reuse `graph_model/`.
- [x] Imported monitor-record strictness (Task 12, completion gate candidate): decision/records.py
      permits boolean `checkpoint_epoch` and whitespace-only `training_run`; decision/decide.py
      compares `detail.seed` by value, and the `SeedRun.monitor_records` docstring says oldest first
      rather than file order. Producer records in all ten audited runs are well formed. Size: plan.
      Done when: src/naics_embedder/decision/records.py and decide.py reject these malformed
      imported identities before panel reads, retain valid producer records, and document file
      order, with focused refusal tests.
      → done in plan 14 (Task 3: strict SeedRun identities; monitor reads matched by type; D8's
      decision-read identities still compare by value, see the plan 14 section below).
- [x] Margin-timing error shape (completion gate candidate): `check_margins_first` in
      src/naics_embedder/decision/decide.py fails closed with KeyError for a missing read time and
      TypeError for a naive time. No campaign record has either shape. Deferred as imported-record
      error handling. Size: quick-fix. Done when: it validates presence and timezone awareness and
      raises a named ValueError before comparison, with tests covering both decision and monitor
      reads and valid aware times.
      → done in plan 14 (Task 2: read_time's named ValueError, before any comparison).
- [ ] Fixture warning cleanup (Phase 1 Task 16 review): preserved suite output includes
      benchmark/xdist plugin notices, platform/worker warnings and graph integer-log/tensor-scalar
      warnings. The final finding-branch suite passed 3000 tests with one skip and 171 warnings;
      nothing was suppressed. Deferred to keep unrelated fixture changes out of the evidence branch.
      Size: plan. Done when: warning sources in tests/conftest.py and graph fixtures are classified,
      actionable fixture warnings are corrected with targeted tests, and remaining intentional
      warnings are documented without broad suppression.
- [ ] Remote stop-target race (private lifecycle correction review):
      src/naics_embedder/remote/worker.py inspects the owned tmux session and later sends C-c by
      session/current-window target. A retarget between those operations remains a pre-existing
      limitation; the private PID/exec and wrapper fixes did not widen it. Size: plan. Revisit if:
      the remote stop lifecycle is hardened or this race is reproduced; pin the verified pane ID and
      recheck ownership at the send boundary with refusal tests before claiming race-free shutdown.
      → the private PID/exec and wrapper fixes (b94ecf3) are public as `bd546ea` (2026-10-16);
      the race itself stays open.
- [ ] Public native-qualification reconciliation (Phase 2 source boundary): the actual campaign
      source b94ecf3 and Mac-read source 2d1f045 include approved local-only corrections to
      remote/model_cache.py, canonical.py, loop.py, worker.py, launch.py and workflow.py. Public
      base a3e82f1 lacks them. The finding explicitly qualifies only the recorded private source.
      Held configuration, these private commits and their descendants must never be pushed. Size:
      plan. Revisit if: native readiness is claimed for a public release or later campaign; use a
      separately authorized fresh public-base change and review/qualification, preserving the
      private configuration and campaign artifacts.
      → plan 12 (Stage 8): the campaign source also carries `4a1a3f2`, a further local-only
      correction to remote/sync.py (a final pull verifies its baseline once, not once per file);
      the public base lacks it too. The finding qualifies only the recorded private sources.
      → partly done (2026-10-16): every local-only code correction is now public. Three are
      `git cherry-pick -x` copies on origin/main 83f9b27: d7927a5 as `6406c72` (model_cache.py,
      launch.py, workflow.py, worker.py), 5ce248f as `b4dc15e` (canonical.py) and b94ecf3 as
      `bd546ea` (loop.py, worker.py). `4a1a3f2` landed as `1643b67` (lowmason/naics-embedder#135).
      The held originals stay unpushed, and 2d1f045 is a configuration pin, not a code fix. The
      finding still qualifies only the recorded private sources; native readiness for a public
      release still needs its own qualification.
      → held configuration moved (2026-10-16, audit row 1): the manifest pins aa8ebd6
      (conf/config.yaml) and 3fc580a (conf/graph.yaml) now live in the gitignored
      `conf/local.yaml`. It may set only those two manifests, each repo-relative under `data/`,
      and is never pushed. 2d1f045's `qcew_dir` pin is dropped, because the public default
      `~/Downloads/Data/QCEW` holds slices identical under `qcew_sha256`. After this lands, local
      main follows origin/main, and these and the other held SHAs that completed plans and
      findings cite stay reachable locally under `refs/held/*` in the primary clone, never pushed.

## 12-geometry-by-dimension — 2026-10-11
- [ ] Review M2 (Phase 1 final review): `build_head` and `head_of`
      (src/naics_embedder/text_model/heads.py:232, :255) are annotated `nn.Module`, so nothing
      types the surface the three heads share and a type checker cannot see what a caller may use
      on a head. Deferred because the finding branch carries no source change. Size: quick-fix.
      Done when: a `typing.Protocol` names that surface, `build_head` and `head_of` use it, and the
      suite, Ruff and the format check pass. Revisit if: a fourth head is added, or a caller
      reaches for a hyperbolic-only member on a flat head.
- [ ] Review m2 (Phase 1 final review): flat arms still log radius per level.
      `LoggingMixin` (src/naics_embedder/text_model/mixins/logging.py:77-82) writes
      `radius/mean/level_<k>` and `radius/sd/level_<k>` from the code cache for every geometry, and
      the epoch summary admits those keys (text_model/epoch_summary.py:14), so the spherical arm
      plots a flat line at 1 with SD 0, and nothing documents it. Nothing selects on these values.
      Size: quick-fix. Done when: either a flat arm's summary and plots omit the radius panels, or
      docs/text_training.md says what each geometry's radius means. Revisit if: a reader takes a
      flat arm's radius for a learned quantity.
- [ ] `remote finish` overhead still grows with the runs on one instance (Phase 2 campaign):
      after the private fix `4a1a3f2` (src/naics_embedder/remote/sync.py,
      `recover_pending_promotion` verifies the cumulative baseline once, not once per
      replacement), the time from a run's training end to the driver's DONE line still rose about
      0.2 min per run on a fresh instance (1.55 min for the first run to 5.67 min for the 22nd;
      about 1.2-1.4 min per run before the fix). The residual cause is undiagnosed. `4a1a3f2` is
      local on `main` and never pushed, and the public base lacks it. Size: plan. Done when: the
      residual is profiled (a CPU-bound `remote finish` python; macOS `sample`) and either removed
      or recorded as the cost of one cumulative-manifest hash per pull, and a public PR branched
      from origin/main carries the `4a1a3f2` fix with its regression test
      (`test_a_final_pull_verifies_the_cumulative_baseline_a_constant_number_of_times`). Revisit
      if: a campaign runs more than about 15 runs on one instance.
      → partly done: the public fix and its regression test merged as
      lowmason/naics-embedder#135 (2026-10-11), so the public base now has the fix. Profiling
      the residual growth stays open and undiagnosed.
      → correction (2026-10-16): the note above that `4a1a3f2` is unpushed and missing from the
      public base is stale. The fix is public as `1643b67`, a copy of `4a1a3f2`; only the held
      original stays local.
      → held 2026-11-18: profiling needs a Lambda campaign of many runs on one instance; the next one is redesign Stage 5's, HELD on embedding-data gates G2–G5.

## 14-economic-decision-tooling — 2026-10-13
- [x] Trained economic candidates (plan 14, deferred by the owner to the candidates' spec):
      src/naics_embedder/decision/economic.py records and decides deterministic candidates
      only. `EconomicRun` (decision/records.py) has no checkpoint, training run, epoch or
      monitor-record field, and `check_economic_arm` requires one run with no seed. Req 24 gives
      a trained candidate at least 5 seeds, and Req 23 has its checkpoint selected on validation
      splits only. The command that reads a candidate (only `read_deterministic_arm` exists) and
      the text-only comparator of a candidate with no backbone of its own, such as a tree
      embedding, wait with it. See specs/plans/completed/14-economic-decision-tooling.md (P7,
      P9). Size: plan. Done when: an economic arm record holds a trained candidate's seeds with
      their checkpoints' selection evidence, `decide_economic` resamples those seeds nested
      inside the groups under the 5-seed floor, and a command reads a candidate, with tests.
      → done 2026-11-15 (plan 16)
- [x] Economic stored-scores check (plan 14 final review, Minor, triaged defer):
      `check_economic_arm` resolves `EconomicRun.scores`, and `check_relative_margins` never
      resolves `ReferenceRead.scores` (src/naics_embedder/decision/economic.py). Only the stored
      predictions are rescored, so a record can name a stale or missing scores file and still
      validate; the decision's outcome is unaffected. Size: quick-fix. Done when: both checks
      resolve the stored scores and compare them with the rescore within `STATISTIC_TOLERANCE`,
      or their docstrings say stored scores are informational, with a test.
      → done 2026-11-15 (plan 16)
- [x] Economic predictions-to-table binding (plan 14 final review, Minor, triaged defer):
      the stored predictions carry no run id or table fingerprint (`PREDICTION_COLUMNS`,
      src/naics_embedder/panels/regressor.py), so an imported `EconomicArmRecord` could pair one
      candidate's log records, table and provenance with another's predictions and recomputed
      statistics, and still pass `check_economic_arm`, the read-identity check and
      `check_reference_pairing`. D8's `SeedRun` has the same gap. Size: design. Revisit if: the
      candidates' spec builds candidates from tables it does not control, or when trained
      candidates rewrite `EconomicRun`. Done when: a record's predictions are bound to its table
      (the arm's matrix fingerprint stored with them, or a re-read spot check) and a mismatched
      pair is refused, with a test.
      → done 2026-11-15 (plan 16)
- [x] Economic decision test gaps (plan 14 final review, Minors, triaged defer): enforced but
      untested: `_same` on non-seed identities, `StrictInt` refusing `1.0` and `'1'`, a logged
      `dimension` of 3.0 or `level` of true on the economic path, `RelativeMarginRecord.margin()`
      with no delta, the `GROWTH_PANELS` literal, `check_economic_arm`'s statistics-edit and
      key-set refusals, duplicate margins panels, a margin-read log record whose run id differs,
      the CLI's adopted path ("Chosen: <candidate>") and `_margin_kind` on a non-margin file, and
      mixed D8 and relative inputs to `tools decide`. D8's decision-read identities still compare
      by value (`_check_log_records`, src/naics_embedder/decision/decide.py), outside the
      strictness item's scope. Size: quick-fix. Done when: each has a focused test, and a ruling
      is recorded on `_check_log_records`' by-value comparison.
      → done 2026-11-15 (plan 16): the gaps in the functions plan 16 rewrote (the statistics-edit and key-set refusals, the logged read identities, the CLI's adopted path and `_margin_kind`) → item: 16-economic-candidates (the remaining gaps)

## tech-debt-audit — 2026-10-16
- [ ] HGCN legacy-input fallback (audit row 1): with no graph supervision manifest,
      `resolve_graph_config`
      (src/naics_embedder/graph_model/curriculum/preprocess_curriculum.py:117-118) returns the
      config unchanged. HGCN then reads the legacy `./data/naics_relations.parquet`,
      `./data/naics_training_pairs.parquet` and `./data/naics_distance_matrix.parquet` defaults,
      which `data all` no longer writes, and says nothing, where text training refuses at its gate.
      The manifest now lives only in the gitignored `conf/local.yaml`, so a checkout without one
      (a fresh clone, or `NAICS_EMBEDDER_LOCAL_CONFIG=off`) falls back silently. Deferred to keep
      HGCN behaviour out of the override's change; the graph stage is parked. Size: quick-fix.
      Done when: HGCN with no manifest either refuses, naming `conf/local.yaml`, or logs that it
      reads legacy inputs, with a test, and docs/hgcn_training.md says which. Revisit if: the
      graph stage resumes or HGCN is run again.

## 16-economic-candidates — 2026-11-15
- [ ] The `--dry-run` text of `tools economic-read` overstates the dry run (final review, Declined
      to judge; both seats judged it harmless to an operator, the controller ruled defer): the
      option's help text and the command's docstring in `src/naics_embedder/cli/commands/tools.py`
      say every check runs before the panel loads, and the printed line says `Dry run: every check
      passed for <name>`. The dry run stops before the panel loads, so the panel's identity against
      the margins, the table's panel-code coverage and width, and a deterministic candidate's D9
      comparison run only in the real read (`docs/usage.md` and `CLAUDE.md` say so). A Task 8 test
      pins the printed line. See specs/plans/completed/16-economic-candidates.md. Size: quick-fix.
      Done when: the help text, the docstring and the printed line say which checks the dry run
      runs, and the test that pins the line is updated.
- [x] Pin the shared text-only table's provenance file in the frozen list (plan 16 execution,
      controller ruling): `conf/data/economic_candidates.yaml` pins the text-only table's bytes
      (`text_only_sha256`) but not its provenance file, so a hand-built deterministic record could
      store another provenance for the same bytes and pass the D9 comparison when its spec names
      the same values (`_check_shared_text_only`, src/naics_embedder/decision/economic.py). No tool
      writes such a record. Changing the list changes its sha256, which `decide_economic` records,
      so this goes in before Stage 3's first real read. See
      specs/plans/completed/16-economic-candidates.md (Handoff). Size: plan. Done when: the list
      pins the provenance file's sha256, the gate refuses a record whose stored text-only
      provenance differs from it, with a test, and the list's sha256 is recorded again before the
      first real candidate read.
      → planned 2026-11-18: specs/plans/19-economic-decision-read.md
      → done 2026-11-18 (plan 19): the list pins the provenance file's sha256 (`text_only_provenance_sha256`), three places refuse another provenance, and the list's sha256 f5a9223b was recorded again before the first real read.
- [ ] Remaining economic decision test gaps (the rest of plan 14's item "Economic decision test
      gaps", which plan 16 closed in part): plan 16 tests the statistics-edit and key-set
      refusals, the logged read identities, the CLI's adopted path and `_margin_kind` on a
      non-margin file. The tests it added do not show these: `RelativeMarginRecord.margin()` with
      no delta, the `GROWTH_PANELS` literal, duplicate margins panels, a margin-read log record
      whose run id differs, mixed D8 and relative inputs to `tools decide`, `StrictInt` refusing
      `'1'`, and a ruling on `_check_log_records`' by-value comparison
      (src/naics_embedder/decision/decide.py). See specs/plans/completed/14-economic-decision-tooling.md.
      Size: quick-fix. Done when: each gap that tests/unit/test_decision_economic.py and
      tests/unit/test_cli_commands.py still do not cover has a focused test, and the ruling on
      `_check_log_records` is recorded.
- [ ] The front-door docs lag the code (plan 16 Task 9 review, outside the plan's file list):
      `README.md:174` and `tests/README.md:5` and `:22` still state 3,342 tests and 105 unit
      files, and `CLAUDE.md` states 4,086 tests and 107 unit files. `README.md`,
      `docs/quickstart.md` and `docs/overview.md` do not name `tools economic-build`,
      `tools economic-read` or `tools decide --candidates`. Size: quick-fix. Done when: the counts
      match `CLAUDE.md`'s and each page that lists the `tools` commands names the three.
      → progress 2026-11-17 (plan 17): the counts now match in `README.md`, `CLAUDE.md` and
      `tests/README.md`. The `tools` command names remain.

## 17-text-to-code-ita-panel — 2026-11-17
- [ ] Margin δ against the candidate's table build (plan 17 final review, controller ruling):
      `compare_to_reference` (src/naics_embedder/decision/text_to_code.py) refuses seeds read on
      another ITA table build through `ItaResult.data_fingerprint`, but it takes δ as a float, so
      it cannot check that `ItaMarginRecord.data_fingerprint` matches the results' build. Stage 5's
      decision caller must compare them. See specs/plans/completed/17-text-to-code-ita-panel.md
      (Handoff). Size: quick-fix. Done when: Stage 5's decision path refuses a margin record whose
      data fingerprint differs from the compared results', with a test.
- [ ] Margin-first across selection logs (plan 17 final review, owner deferred to Stage 5):
      `check_ita_margin_first` reads only the one `SelectionLog` it is given, so training-monitor
      ITA validation reads logged in a remote instance's separate log are not seen. Size: plan.
      Done when: Stage 5's margin-first check reads every log that holds ITA validation reads
      (the Mac's and each pulled run's), with a test.
- [ ] A candidate read before any margin blocks every later margin (plan 17 final review, owner
      deferred to Stage 5): the log is append-only and `check_ita_margin_first` counts any earlier
      `ita_text` validation candidate or no-role read whatever its split fingerprint, so one early
      read refuses every margin record. This follows spec §4.5. Size: design. Revisit if: Stage 5
      needs a margin record after an `ita_text` validation candidate read is logged.
- [ ] Margin reads name no checkpoint (plan 17 final review, owner deferred to Stage 5): the
      margin record's reads log no table or checkpoint fingerprint for `euclidean-d16`'s seeds, so
      the record cannot show the outcome-selected checkpoints of spec §4.5 were the ones read.
      Size: quick-fix. Done when: each reference read in Stage 5's margin record names its seed's
      checkpoint or table sha256, with a test.
- [ ] Margin record self-check gaps (plan 17 gate-fix review, controller deferred):
      `check_ita_margin` does not require reads of both reference arms nor refuse a repeated seed
      among `log_records`, so a hand-edited record can pass it. Size: quick-fix. Done when: both are
      refused, with tests, before Stage 5 writes its margin record.

## 19-economic-decision-read — 2026-11-18
- [ ] The level-mode `CAPTURED_HERE` pin fails on macOS 27.2.1 (plan 19 Pre-flight Step 3; the owner
      ruled "PR A first" and the plan went on): `test_level_predictions_are_the_captured_ones_bit_for_bit`
      in `tests/unit/test_regressor_panel.py` compares level-6 predictions to literals captured in an
      earlier environment. It fails on this Mac, serially and with the BLAS thread variables set to
      1, while the d61f49b oracle and the float-free digest pins pass. The test's own note says that
      this means the environment changed, and CI on Linux skips it. Every full-suite run on this Mac
      shows one failure, and a plan that reuses plan 15's level-pin pre-flight stops on it. Plan 19
      did not re-capture the literals: that needs the owner's review of the change first. See
      specs/plans/completed/19-economic-decision-read.md and specs/findings/economic-decision.md.
      Size: quick-fix. Done when: the owner has reviewed the environment change and the literals
      are re-captured on this macOS, or the pin takes a tolerance, and a full-suite run on the Mac
      passes.

## 20-text-to-code-reference-read — 2026-11-19
- [ ] `tools ita-read` accepts `--outcome` on a `euclidean-d16` read and leaves it off by default
      for the stub (plan 20 final review, Declined to judge; owner deferred to Stage 5). A slip logs
      a one-time read that `tools ita-gap` refuses later; plan 20 guarded it only by its Gate B
      commands and `check_read.py`. See `ita_read_command` in
      src/naics_embedder/cli/commands/tools.py and
      specs/plans/completed/20-text-to-code-reference-read.md (Handoff). Size: quick-fix.
      Done when: the read path that Stage 5 uses refuses `--outcome` on `euclidean-d16` and logs the
      stub's outcome read whenever the stub's record must carry C, with a test for each.
- [ ] `tools ita-read` does not check a d16 seed's `--checkpoint` and `--table` against the Stage 8
      record before it logs (plan 20 final review, Declined to judge; owner deferred to Stage 5).
      `gap_seeds` (src/naics_embedder/decision/ita_gap.py) and plan 20's `check_read.py` catch a
      mismatch only after the one-time read is in the selection log. Size: quick-fix. Done when: a
      d16 read refuses a checkpoint whose SHA-256, or a table whose matrix fingerprint, is not the
      Stage 8 record's for that seed, before any log record, with a test.
