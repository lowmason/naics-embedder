---
paths:
  - "src/naics_embedder/decision/economic.py"
  - "src/naics_embedder/decision/candidates.py"
  - "src/naics_embedder/decision/provenance.py"
  - "src/naics_embedder/decision/trained.py"
  - "src/naics_embedder/tree_model/**"
  - "src/naics_embedder/panels/regressor*.py"
  - "src/naics_embedder/panels/qcew_rows.py"
  - "src/naics_embedder/panels/ridge.py"
  - "tests/unit/test_decision_economic.py"
  - "tests/unit/test_economic_candidates.py"
  - "tests/unit/test_tree_model.py"
  - "tests/unit/test_regressor_*.py"
  - "tests/unit/test_committed_regressor_groups.py"
  - "tests/fixtures/economic.py"
  - "tests/fixtures/regressor_panel.py"
  - "conf/data/economic_candidates.yaml"
  - "conf/data/regressor_panel.yaml"
  - "conf/data/decision.yaml"
---

# Economic Artifact

The root `CLAUDE.md` keeps the economic guardrails: one opening of a regime's outer set, level
mode only before the margin, margin-first reads, one `uv.lock`, the fixed growth margins and the
recorded decision. This file holds the mechanics.

## Regressor Outcome Modes

The regressor panel's `outcome` (`conf/data/regressor_panel.yaml`, Req 21) is `level` by default,
the levels history read as `regressor_seen` and `regressor_heldout`. `growth` reads
log E(t+1) − log E(t) on the same rows, partition and folds, as `growth_seen` and
`growth_heldout`. The fixture panel pins level mode's rows, data fingerprints, folds and scores at
d61f49b (`test_regressor_panel.py`).

## Economic Decision

The economic decision (Req 24, `decision/economic.py`) reads `growth_seen` and `growth_heldout`
against the reference `covariates+ancestors`. `tools growth-margins` reads growth whatever the
config says, scoring only the comparators that read no candidate, and records δ = r × the
reference's validation MSE, with r = `relative_margin` (0.05) in `conf/data/decision.yaml`.
A deterministic candidate is one replicate, read by `read_deterministic_arm` with its table's
provenance and no checkpoint or monitor reads. `tools economic-build` builds the nine
deterministic tables from the frozen list. `tools economic-read` reads one frozen-list candidate,
deterministic or trained, after the checks that run before the panel loads. Under `--dry-run` it
stops after those checks: it loads no panel and logs nothing. The table's width, its panel-code
coverage and a deterministic candidate's D9 comparison wait for the real read, and a deterministic
row must pin its built table's sha256 before the read accepts it. A trained candidate is imported
from its Stage 7/8 arm record and run directories (`--runs-root`) and read once per seed.
`tools decide` makes the economic decision from a relative margin record and takes `--candidates`,
the frozen list (default `conf/data/economic_candidates.yaml`). It requires exactly the list's
candidates and records the list's sha256. It also reports whether each smoothed candidate is
adopted over its text-only twin, and chooses nothing by it. It refuses a candidate whose reference
rows differ from the margin read's, or whose reference predictions differ from its by more than
`PAIRING_TOLERANCE` (1e-13, absolute; the prediction alone has a tolerance), or that read before
the margins.
Superiority is 1 − 0.05/n: 97.5 % over the two growth panels, 98⅓ % over D8's three. Only a
candidate adopted over the reference is eligible; with none, the reference stands.

## Pairing and the Lock

The `uv.lock` freeze held from Plan 10's first campaign launch through Stage 8's decision and was
released when `specs/naics-representation-split.md` merged (Req 19). `decide` pairs reads only on
identical regressor rows and reference predictions equal to within `PAIRING_TOLERANCE`, so the
growth margin read and every economic candidate read share one `uv.lock` and run on macOS arm64.
The tolerance exists because plan 19's first real read, on macOS 27.2.1 against the margin read's
26.6.2, differed in the last bits (at most 6.9e-16) and moved the reference's validation MSE by
one ulp.

## Tests

- `test_regressor_panel.py`, `test_regressor_qcew_rows.py`: level mode pinned at d61f49b, the
  growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22).
- `test_decision_economic.py`: the reference-only margin read and δ, deterministic candidates'
  records and pairing, eligibility, the tie order, margin-first refusals and the economic
  decision record (Req 24).
- `test_tree_model.py`: classical scaling and its cut refusals, lineage smoothing, and the three
  builders' tables, codebook order and byte-for-byte determinism.
- `test_economic_candidates.py`: the frozen list and its pins, the text-and-tree-only provenance
  rule and trained imports. `test_cli_commands.py` covers `tools economic-build`,
  `tools economic-read` (with its dry run) and `tools decide --candidates`.
