# Development history

## 16-economic-candidates — 2026-11-15
Plan: specs/plans/completed/16-economic-candidates.md · Spec: specs/completed/economic-candidates.md (retired) · Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 1

Differences from the design:
- Ruling (Task 4): Task 4 updates one existing assertion (tests/unit/test_decision_economic.py:764-767, inputs == {'codebook': 'the fixture codebook'} -> {'codebook': CODEBOOK_INPUT}) — the brief's write_candidate interface mandates the new inputs value and the brief's "change no existing assertion" (lines 141-142) is wrong for this one; the file is not one of the six D8 files — cost if wrong: none, a fixture value is pinned
- Ruling (Task 4): Task 4's export rule stays shape-only for checkpoint/descriptions/summaries (the brief's rule) — spec :226-229 and :235 hold by composition: Task 6 calls check_seed_table (backbone, revision, descriptions sha256, summaries and window must equal the source spec's, which the list pins through the record sha256) and requires the export's checkpoint sha256 to equal the run's; a QCEW path in those slots can pass only with the real file's bytes — cost if wrong: a gap if Task 6's checks are skipped; Task 6 must add refusal tests for another descriptions sha256, another summaries entry and a QCEW path in those slots
- Ruling (Task 5): Task 5 changes the last assertion of tests/unit/test_decision_economic.py::test_a_single_replicate_record_validates_and_pairs_with_the_reference to select the margin read's columns first (store.read_frame(predictions).select(store.read_frame(margins.read.predictions).columns)) — spec :296 "The reference comparison still reads only the margin read's columns, so a binding stored with the predictions cannot break the pairing"; the brief's Step 4 "PASS" and its Rewrites list missed this test; the binding stays on the arm side — cost if wrong: none, the compared columns are the margin read's own, so every shared value is still compared bit for bit
- Ruling (Task 5): yaml.YAMLError from load_candidate_list (malformed --candidates file) is not caught by decide_command's handler — the committed list is the normal input; Task 8's dispatch adds a ValueError wrap in load_candidate_list (with a test) because all three commands call it — cost if wrong: a traceback instead of "Decision failed" for a hand-broken list file
- Ruling (Task 6): plan's test_d8_still_times_a_trained_runs_monitor_reads says check_margins_first refuses with `monitor read of epoch`; that text appears in decide.py only for a naive/missing time (read_time), a valid early read raises `read at ..., before the margins were fixed` (decide.py:452); changing D8's messages is forbidden by the global constraints, so the implementer's two tests pin both D8 paths and the plan's claim stands corrected — cost if wrong: none
- Ruling (Task 6): land the implementer's proposed fix — check_economic_arm's trained branch also requires the arm's seeds to equal its list row's seeds (patch task-6-fix1-row-seeds-proposed.patch with its red test task-6-fix1-row-seeds-test.patch) — spec :217 "refuses a candidate that is not on [the list]" and the §4.1 table lists each trained row's seeds (1-10, 1-5); the plan's trained-branch list omits it (a deviation to surface at completion); in the real flow the import already enforces equality, so no valid record is newly refused — cost if wrong: one extra refusal for a record that matches its source but not its list
- Ruling (final review): the final review's Declined line "/Users/ paths in docs/usage.md's economic-read example" is made a finding and fixed by hand (docs-only, 5 occurrences -> /path/to/naics-embedder, commit below) — the base docs had none and the page is published to GitHub Pages; the global constraint names config and source files only, but the same reason applies — cost if wrong: none
- Ruling (final review): REVERSES the Task 6 ruling that check_economic_arm does not re-run check_seed_table on a trained run's stored provenance — the final review's Codex seat reproduced it as a P2 (a stored provenance with another revision passes the decision gate, though the import refuses it), the fix is small and only tightens the economic path (D8's gate stays as is), and CLAUDE.md requires the arm's backbone revision to match — cost if wrong: one extra check in the economic gate; goes to a fix subagent (final-fix wave)
- Deviation (Task 4): one existing assertion changes: `inputs` is now `{'codebook': CODEBOOK_INPUT}`. Source: plan, Task 4.
- Deviation (Task 5): one existing assertion selects the margin read's columns first. Source: plan, Task 5.
- Deviation (Task 6): the gate also checks a trained row's seeds, and after the final review it runs D9 on each stored export provenance. Source: plan, Task 6.
- Deviation (Task 8): a YAML error in the list becomes a `ValueError`, and `--dry-run` skips the checks that need the loaded panel. Source: plan, Task 8.
- Deviation (Task 9): the usage example names `/path/to/naics-embedder`, not a home path. Source: plan, Task 9.
- Deferred: 4 items. Source: specs/deferred_items.md, § 16-economic-candidates.

Closed earlier deferred items:
- "Trained economic candidates (plan 14, deferred by the owner to the candidates' spec):" (deferred by 14-economic-decision-tooling).
- "Economic stored-scores check (plan 14 final review, Minor, triaged defer):" (deferred by 14-economic-decision-tooling).
- "Economic predictions-to-table binding (plan 14 final review, Minor, triaged defer):" (deferred by 14-economic-decision-tooling).
- "Economic decision test gaps (plan 14 final review, Minors, triaged defer): enforced but" (deferred by 14-economic-decision-tooling).

## 17-text-to-code-ita-panel — 2026-11-17
Plan: specs/plans/completed/17-text-to-code-ita-panel.md · Spec: specs/completed/text-to-code-ita-panel.md (retired) · Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 2

Differences from the design:
- Ruling, contradicts spec (Task 1): split_rule keeps the plan's seven keys (prefix_bytes, byteorder included); P1 pins them, O1 approved, spec 134 makes byte detail part of the rule.
- Ruling (Task 2): keep the extra null guards and the table-wide repeated pair_id refusal (Global Constraint "Refuse, never drop"); accept Withdrawn 28 (.value equivalent under polars); empty headline set returns an empty frame, Task 4 refuses an empty part — cost if wrong: a Stage 10 table with nulls in unread non-headline rows would be refused at Stage 4, loudly.
- Ruling (Task 4): fsum-based _mean replaces maintain_order (root cause is chunked Polars mean, not group order); null score refusal kept (refuse, never drop) — cost if wrong: Task 6's point-statistic equality may need approx if resampling sums differently (watch in Task 6).
- Ruling (Task 5): public ItaPanel.pairs accepted (row-order plan test needs it); role=None refused (stricter than P5, fails closed); row-order test owned by Task 2 code; three commits fine — cost if wrong: a caller passing role=None gets a refusal instead of a candidate read.
- Ruling (Task 7): no-seed refusal kept; margin-first check counts every earlier ita_text validation candidate/no-role read regardless of split fingerprint (conservative) — cost if wrong: a stale read under an old split would block a margin, loudly.
- Ruling (final review): None-vs-None fingerprints still pair (raw summarize_ita results, fixture-only); Stage 5's caller must compare ItaMarginRecord.data_fingerprint with the results' (handoff bullet) — cost if wrong: a delta fixed on an old build used against a new-build candidate.
- Ruling (final review): test_earlier_reference_reads_and_later_candidate_reads_pass logs its earlier reference read with arm euclidean-d16 instead of c; assertions unchanged — owner's gate decision "role label trusted" makes a reference-role read of a non-reference arm a candidate read — cost if wrong: none, the test still pins earlier-reference/later-candidate passing.
- Decision (owner, completion gate, 2026-11-17): fix now — repeated seeds, the margin record self-check and reference reads of a non-reference arm; three other review lines deferred to Stage 5. Source: specs/deferred_items.md, § 17-text-to-code-ita-panel.
- Deviation (Task 1): fixture-only tests are proven by a hand harness (`mutant-run` refuses `tests/` patches), plus process-seed and as-given key tests. Source: plan, Task 1.
- Deviation (Task 2): `panel_pairs` also refuses nulls in read columns and a repeated `pair_id` anywhere in the pair table. Source: plan, Task 2.
- Deviation (Task 4): every report mean is `math.fsum(values) / len(values)` (Polars chunked means were not bit-reproducible), and null scores are refused. Source: plan, Task 4.
- Deviation (Task 5): `ItaPanel.pairs` is public, a present `role=None` is refused, and after the final review the result carries `data_fingerprint`. Source: plan, Task 5.
- Deviation (Task 6): an arm with no seed is refused, and after the final review `compare_to_reference` refuses seeds read on another table build. Source: plan, Task 6.
- Deviation (Task 7): owner gate added refusals of repeated seeds, `check_ita_margin` (record self-check) and reference reads of a non-reference arm. Source: plan, Task 7.
- Deviation (Task 8): `README.md` and `tests/README.md` counts also updated, and `ItaPanelSettings` documents its fields as `Attributes:` for the strict build. Source: plan, Task 8.
- Deferred: 5 items. Source: specs/deferred_items.md, § 17-text-to-code-ita-panel.

## /deferred — 2026-11-18

Closed deferred items:
- "Review: the national-total employment comparison in `check_invariants`" (deferred by 3-employment-statistics-coverage): the comparison gets the `cells/2 + 1` employment allowance, and the real 2022–2025 run exits 0.
- "Review Minor: the index-roles checks outside `data roles` hardcode the near-duplicate" (deferred by 4-outcome-panel-sealed-splits): both keys are documented as frozen with the table, and a test pins the config, the provenance and the checks' defaults together.
- "Review Minor, for Stage 10: the diagnostics report's interfaces" (deferred by 6-decision-rule-and-diagnostics): a mismatch names the first missing and extra code, the codebook is cast to Utf8, every key uses `@k`, and `codebook_codes` is required.
- "Review Minor: the bundle build loads its tokenizer last. `generate_supervision_bundle`" (deferred by 7-supervision-target-and-text): the build resolves the cached tokenizer first and names the command that caches it.
- "Review Minor: the bundle build does not re-check the text channels." (deferred by 7-supervision-target-and-text): the build runs `verify_text_channels` before any artifact, as a raising check only.
- "Review Minor: a test needs a newer transformers than pyproject's floor." (deferred by 8-shared-encoder-and-projection): the test guards its import and skips its one checkpointing test without the class.

## 19-economic-decision-read — 2026-11-18
Plan: specs/plans/completed/19-economic-decision-read.md · Spec: specs/completed/economic-candidates.md (retired) · Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 3

Differences from the design:
- Ruling (Task 1): `tests/unit/test_economic_candidates.py::_list` (the raw YAML-dict builder) gains `'text_only_provenance_sha256': '3' * 64` - the plan lists the fixture helpers but not this builder, and the field is required (P1/Review Focus 4) - a list without it cannot load - if wrong, nothing: the key is a fixture literal.
- Ruling (Task 1): `test_a_tree_candidate_is_not_d9_checked` (test_decision_economic.py) now asserts the gate refuses the other revision's provenance (`^tree: its text-only provenance`) and accepts it once the list pins it (`frozen.model_copy(update=text_only_pins(other))`) - P1 says the gate runs for both kinds, so the old assertion (passes against the shared list) is superseded; the test's point (a tree candidate is not D9-checked) still holds - if wrong, the test still covers the gate through Review Focus 3's tests.
- Ruling (Task 1): `test_economic_read_refuses_a_text_only_table_whose_provenance_does_not_describe_it` (test_cli_commands.py) now expects the pin refusal - the plan's Interfaces put the new `_require_pin` "right after the table's pin, before the JSON is parsed", so an edited provenance is refused at the pin before `provenance_fields` runs; `provenance_fields`' description check stays as defense in depth but no input that passes both pins reaches it from the read - if wrong: the CLI loses one unreachable-path assertion.
- Ruling, contradicts spec (Task 4): the tolerance is an absolute `PAIRING_TOLERANCE = 1e-13` on `prediction` only. Every other column of the reference rows (panel, repeat, fold, code, group, years, alpha, outcome) stays exact. A NaN never pairs. Why: the measured largest difference is 6.9e-16, so 1e-13 is about 145 times that. A relative tolerance fails on near-zero predictions (relative max 5.2e-11 at an absolute 6.9e-16). 1e-13 is below the existing tests' +1e-12 shifts, so those tests stay valid. A prediction error of 1e-13 moves the reference MSE by at most 2 x RMSE x 1e-13, about 1.1e-14 on an MSE of 3.2e-3, which is 7e-11 of delta. What it costs if wrong: too tight and a later OS update refuses a real read (fails closed); too loose and a genuinely different reference pairs, but a different fit differs by far more than 1e-13.
- Ruling (Task 4): `text-only-d8.json` was read at 2f15310; every later read and the decision run at 37b47c4. Why: `git diff 2f15310..37b47c4 -- src/` touches only `check_reference_pairing`, `_reference_difference`, `PAIRING_TOLERANCE`, docstrings and the refusal text in `decision/economic.py`, `decision/records.py` and `cli/commands/tools.py`; `uv.lock` and `conf/` are unchanged (list sha f5a9223b, lock sha 6de4fbdc). The read path is the same bytes. What it costs if wrong: the record names a read made before the tolerance existed; re-read it to a new path (+2 log records) if the owner objects.
- Deviation (Pre-flight Step 3): the level pin `test_level_predictions_are_the_captured_ones_bit_for_bit` failed on macOS 27.2.1 and the owner ruled "PR A first"; it is a deferred item. Source: plan, Pre-flight Step 3.
- Deviation (Task 4): the first read, `text-only-d8`, failed the pairing check, so PR #172 paired within 1e-13 and the other 17 reads ran at 37b47c4. Source: plan, Task 4.
- Deviation (Task 6): MAIN's HEAD is 37b47c4, not READ_SHA 2f15310, because PR #172 merged first. Source: plan, Task 6.
- Decision D2 (spec amendment): the pairing check pairs the reference to within an absolute 1e-13, and economic-candidates §4.8 is amended. Source: plan, `## Decisions`.
- Deferred: 1 item. Source: specs/deferred_items.md, § 19-economic-decision-read.

Closed earlier deferred items:
- "Pin the shared text-only table's provenance file in the frozen list (plan 16 execution," (deferred by 16-economic-candidates).
