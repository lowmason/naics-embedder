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

## /deferred — 2026-11-18

Closed deferred items:
- "Review: the national-total employment comparison in `check_invariants`" (deferred by 3-employment-statistics-coverage): the comparison gets the `cells/2 + 1` employment allowance, and the real 2022–2025 run exits 0.
- "Review Minor: the index-roles checks outside `data roles` hardcode the near-duplicate" (deferred by 4-outcome-panel-sealed-splits): both keys are documented as frozen with the table, and a test pins the config, the provenance and the checks' defaults together.
- "Review Minor, for Stage 10: the diagnostics report's interfaces" (deferred by 6-decision-rule-and-diagnostics): a mismatch names the first missing and extra code, the codebook is cast to Utf8, every key uses `@k`, and `codebook_codes` is required.
- "Review Minor: the bundle build loads its tokenizer last. `generate_supervision_bundle`" (deferred by 7-supervision-target-and-text): the build resolves the cached tokenizer first and names the command that caches it.
- "Review Minor: the bundle build does not re-check the text channels." (deferred by 7-supervision-target-and-text): the build runs `verify_text_channels` before any artifact, as a raising check only.
- "Review Minor: a test needs a newer transformers than pyproject's floor." (deferred by 8-shared-encoder-and-projection): the test guards its import and skips its one checkpointing test without the class.
