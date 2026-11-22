---
paths:
  - "src/naics_embedder/panels/relatedness*.py"
  - "src/naics_embedder/decision/relatedness.py"
  - "tests/unit/test_relatedness_*.py"
  - "tests/unit/test_decision_relatedness.py"
  - "tests/fixtures/relatedness.py"
  - "conf/data/relatedness_panel.yaml"
---

# Relatedness Artifact

The root `CLAUDE.md` keeps the relatedness panel's guardrails: the sealed part opens only through
`RelatednessPanel.open_sealed`, once, no co-ownership row enters training, and no code or test
reads a real table. This file holds the scores, the status and the traps.

## Scores and Status

The headline is NDCG@10, and NDCG@20 is reported.
Stage 6 (2026-11-22) built the panel and `decision/relatedness.py`. No `tools` command reads the
panel yet. Stage 6b adds that read.

## Traps

These come from the plan 21 Handoff (2026-11-22). Check `specs/deferred_items.md` before relying
on one: a later stage may have fixed it.

- `VectorArm`'s cosine sums depend on the memory layout of its input. `text_only_matrix` returns
  a column-major array, so a `from_text_only` arm and a `VectorArm` over the same values in C
  order differ in the last bits. Build every arm of one run through the same constructor.
- `fix_relatedness_margin` takes r as an argument and never compares it with
  `panel.settings.relative_margin`. The Stage 6b read command must bind r from the settings.
- A similarity that is not finite refuses the read and names the arm and the anchor. A read that
  fails while it scores stays logged. A pair table that disagrees with spec §4.2 is refused at
  construction, before any log record.
- `RelatednessResult` has no arm field. For each family, the caller must load the validation read
  of the arm that `margins[key].reference` names. A wrong read gives a mislabelled verdict.
- n = `len(deltas)` sets the test's level, so pass every judging family. Take `replicates`,
  `bootstrap_seed` and `min_seeds` from `conf/data/decision.yaml`.
- The margin's headline is a `math.fsum` mean and a verdict's Δ is a numpy sum, so they differ by
  one ulp. The verdict reads the interval, not Δ.
- `check_relatedness_provenance` reads no file and inspects names and shapes only. The loader that
  reads a candidate must bind the area-mix file, the checkpoint's `path` and `sha256`, and
  `table_sha256`. A co-ownership file under the `ita_area_mix` name passes.
- `open_sealed` under a settings file with another salt gives another split fingerprint, so it
  finds no earlier opening and logs a fresh `open`.

## Tests

- `test_relatedness_split.py`, `test_relatedness_inputs.py`, `test_relatedness_scoring.py`,
  `test_relatedness_arms.py`, `test_relatedness_panel.py` and `test_decision_relatedness.py`:
  the relatedness panel's salted anchor split, input refusals, tie-aware scores, reference arms,
  logged reads and sealed opening, and relatedness's margin, test and provenance rule. All read
  `tests/fixtures/relatedness.py` tables only.
