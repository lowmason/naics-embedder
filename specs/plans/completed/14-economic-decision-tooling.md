# Economic Decision Tooling Implementation Plan

**Status: COMPLETE (2026-10-13)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-representation-split-roadmap.md, Stage 2 — on plan completion, tick the
> stage and re-validate later stages against what shipped.

**Goal:** Build Stage 2 of the representation-split roadmap: Req 24's decision tooling for the
economic artifact, exercised on fixtures. A reference-only growth read fixes each growth panel's
margin, δ = r × the reference's validation MSE (r = 0.05, U13). A deterministic candidate is read
as one replicate, recorded with its table's provenance and no checkpoint or monitor reads. The
economic decision judges candidates against the comparator `covariates+ancestors` on
`growth_seen` and `growth_heldout`, at a superiority level set by the panel count. Only a
candidate adopted over the reference is eligible; with none, the reference stands. The two
deferred items on the margin-first check are discharged on the way.

**Architecture:**

- **The superiority level by panel count (Task 1).** `superiority_level(n)`, 1 − 0.05/n, replaces
  the constant `SUPERIORITY_LEVEL`. `compare_panel` takes the level as a keyword, and D8's
  decision passes its own, so a three-panel decision still reads 98⅓ % (P1).
- **Read times refused by name (Task 2).** `read_time` refuses a missing, malformed or naive
  read time with a named `ValueError`, and reads pydantic's `Z` under Python 3.10.
  `check_margins_first` takes the margins' time rather than a D8 margin record, so the economic
  decision shares it (P2).
- **Imported monitor-record strictness (Task 3).** A trained run no longer loads a training run
  of spaces or a boolean epoch, and a monitor read's identities match by type as well as value
  (P3).
- **The reference-only read (Task 4).** `RegressorPanel.reference_validation` reads and logs only
  the comparators that read no arm's tables, on the rows, folds and penalties every arm's read
  fits. Each of them predicts exactly as in any arm's read (P4).
- **The relative margins (Task 5).** A new module, `decision/economic.py`. `fix_relative_margins`
  runs the reference-only read on both growth panels and records a `RelativeMarginRecord`. The
  record names r, the reference, the read, the panel and where the read ran: platform, Python
  and `uv.lock` sha256 (P5, P6).
- **A deterministic candidate (Task 6).** `read_deterministic_arm` reads a candidate's table once
  on each growth panel and records an `EconomicArmRecord`. Its one `EconomicRun` has no seed,
  checkpoint or monitor reads, and carries the provenance the table's builder wrote.
  `check_economic_arm` validates a record. `check_reference_pairing` requires the reference's
  predictions to equal the margin read's bit for bit (P6–P9).
- **The economic decision (Task 7).** `decide_economic` first checks the margins, each candidate,
  its pairing and the order of the reads. It then compares each candidate with the reference at
  95 % and 97.5 %. Among the candidates adopted over it, Req 5's non-dominated set and tie order
  pick one; the higher held-out gain over the reference breaks a last tie (D11). It rescores
  every read's stored predictions (P10–P12).
- **The CLI (Task 8).** `tools growth-margins` reads growth in memory, whatever the committed
  config says. `tools decide` makes the economic decision from a relative margin record.
  `relative_margin: 0.05` joins `conf/data/decision.yaml` (P13).
- **Documentation (Task 9).** The usage guide, the API page, the overview pages, CLAUDE.md, the
  README and the test README (P14).

Nothing reads real data. Stage 3 runs `tools growth-margins` on real data, with the user's
approval at the time. The economic candidates' spec builds candidates and decides through these
interfaces; a trained candidate's economic record waits for it (O6).

**Tech Stack:**

- Python 3.10 and 3.12 (CI runs both; `.python-version` pins 3.12).
- polars 1.35.1, numpy (2.3.4 under 3.12, 2.2.6 under 3.10) and pydantic 2.12.4; typer and rich.
- pytest with xdist; ruff and yapf; mkdocs with mkdocstrings (strict build).
- No model, backbone or GPU: the fixture panel, codebook and tables are synthetic.

**Spec:** `specs/naics-representation-split.md` (last changed 0a39042), Reqs 18, 23 and 24. This
plan implements Stage 2 of its roadmap, `specs/naics-representation-split-roadmap.md` (last
changed 32b102b). Retirement leaves the spec in `specs/`: Stage 3 and the candidates' and
text→code specs cite it.

## Global Constraints

Every task's requirements include this section.

**Locators.** Bare D1–D11 are the embedding roadmap's decisions
(`specs/naics-embedding-roadmap.md`), as the codebase's comments and this plan's code cite them:
D8 the three-panel decision, D9 the text-only table, D10 each panel's statistic, D11 the final
tie-break. U1–U16 are the split spec's user rulings. P1–P15 are this plan's decisions, and O1–O6
the owner's (`## Decisions`).

### The spec (`specs/naics-representation-split.md` at 0a39042), verbatim

> **Req 18 — Three artifacts (user adjudication Q3, 2026-10-12).** Q3 supersedes Q1's clause that one
> embedding "must serve both as a regressor and as an outcome so it is not specialized" (U1, U3).
>
> Q1's purpose carries over to the economic artifact: low-dimensional coordinates in place of sparse
> encodings, "16 coordinates, for example, instead of 1,000+ indicators". All three artifacts share
> the code universe, the bundle's 2,125 NAICS 2022 codes.
>
> | Artifact | What it is | Judged on |
> |---|---|---|
> | Hierarchy | The bundle's codebook: codes, parents, ancestors and the tree metric (naics-embedding Req 7). Exact, with nothing trained | Nothing; it is exact by construction |
> | Text→code | A retrieval system, free of the low-dimension requirement | The outcome panel only (naics-embedding Req 3) |
> | Economic | Low-dimensional coordinates for the 2,125 codes | Economic panels only; the first is Req 21 |
>
> - The hierarchy needs no new work. Its ancestor indicators are the economic reference (Req 24).
> - Each artifact has its own reference, margins, deliverable and sealed opening. No artifact's
>   adoption decision reads another artifact's panel.
> - The text→code artifact's design belongs to its own spec (Out of scope).

> **Req 23 — Candidates read text and the tree only (U7).** naics-embedding Req 11's independence
> holds for the economic artifact. No candidate's construction reads QCEW or any economic panel's
> data.
>
> - **What a candidate is.** A coordinate table for all 2,125 codes. Examples are a trained arm's
>   export, the text-only table at the candidate's dimension (roadmap D9), an embedding of the tree,
>   or a combination of these.
> - **Selection.** No training term of a candidate reads any panel (naics-embedding Req 11). Every
>   selection reads validation splits only (naics-embedding Req 4). That covers a trained
>   candidate's checkpoint, which Stages 7–8 selected on the outcome panel, and the choice among
>   candidates, which reads growth validation splits.
> - **Candidates trained on employment data.** Such a candidate needs a time window that the panel
>   never scores. With three growth years there is none, so it waits for a later reference year,
>   once that year is final.

> **Req 24 — The economic decision rule (U8, U9, U13).** naics-embedding Req 5 carries over:
>
> - the paired two-stage bootstrap: 10,000 replicates with seed 20260924, resampled by four-digit
>   group, with seeds nested inside;
> - the non-dominated set;
> - the tie order.
>
> It changes as follows for the economic artifact.
>
> - **Panels.** `growth_seen` and `growth_heldout`, which give two chances to adopt.
>   - Non-inferior: the lower bound of the 95% interval on Δ exceeds −δ.
>   - Superior: the lower bound of the 97.5% interval exceeds 0.
> - **Statistic.** roadmap D10's, applied to growth: the level-6 MSE of `covariates+candidate`, with
>   each row's squared error averaged over its repeats. Δ is the reference's error minus the
>   candidate's, so a positive Δ favours the candidate.
> - **Reference.** `covariates+ancestors` (U9). It reads no candidate, and the folds depend only on
>   the regime, level, repeat and group. So its row errors are the same in every read, and they pair
>   exactly with each candidate's.
> - **Replicates.**
>   - A trained candidate brings at least 5 seeds (naics-embedding Req 5).
>   - A deterministic candidate, such as the text-only table or a tree embedding, is a single
>     replicate.
> - **Margin.** δ = 0.05 × the reference's validation MSE on that panel (U13), where that MSE is
>   the statistic above computed for `covariates+ancestors`. δ is fixed and recorded from a
>   reference-only read before any candidate is read.
>   - The margin read scores only comparators that read no candidate table: `covariates`, `one_hot`,
>     `ancestors` and their covariate pairs. It computes neither `embedding` nor `text_only`, so no
>     candidate's growth error exists before δ is recorded.
>   - Rejected: δ = k × the reference's SD across fold repeats. That is near 0 in the seen regime,
>     where repeats only redraw the tuning folds, so non-inferiority would become superiority.
>   - Rejected: δ = k × the bootstrap SD of the reference's MSE. It double-counts the resampling the
>     interval already covers, and it is loose.
> - **Eligibility and ties.**
>   - Only a candidate adopted over the reference is eligible. Among those, the non-dominated set
>     and naics-embedding Req 5's order pick one: fewer components, then lower dimension, then
>     non-hyperbolic.
>   - The candidates' spec counts each candidate's components.
>   - roadmap D11's held-out estimate breaks any tie that remains. For growth, that is the higher
>     held-out gain over the reference.
> - **The economic deliverable** is the chosen candidate's 2,125-code table. If no candidate is
>   eligible, the reference stands. There is then no economic artifact, and downstream models use
>   the hierarchy's indicators.
> - **Tooling.**
>   - **Superiority level.** It follows the decision's panel count, 1 − 0.05/n; today it is
>     hard-coded as `SUPERIORITY_LEVEL = 1.0 - 0.05 / 3` at `src/naics_embedder/decision/rule.py:27`.
>     A three-panel decision still reads 98⅓%.
>   - **Arm records** accept a deterministic single-replicate table, recorded with its provenance
>     and without checkpoint or monitor reads.
>   - **Margin records** gain the relative rule, naming r and the reference-only read.
>   - **`decide`** accepts a comparator as the reference, and its margin-first check covers the
>     reference-only read.

From Supersessions:

> | Superseded | Replaced by | Applies to |
> |---|---|---|
> | naics-embedding Req 5's reference configuration and margin | `covariates+ancestors`, and δ = 0.05 × its MSE (Req 24) | Economic; text→code fixes its own in its spec |
> | Roadmap D8, three panels at 98⅓% | Two growth panels at 97.5% (Req 24); the outcome panel goes to text→code | Economic |
> | Roadmap D10's sparse baselines: one-hot in the seen regime, ancestors in the held-out | `covariates+ancestors` in both regimes, with one-hot still reported (Req 24) | Economic |

> ### Deferred items this touches
>
> `specs/deferred_items.md` was read, not edited.
>
> - **Waiting with the parked graph stage:**
>   - the graph curriculum's degenerate difficulty thresholds;
>   - the diagnostics report's interfaces "for Stage 10";
>   - HGCN feeder precision;
>   - HGCN metric mutation coverage.
> - **Moving with Stage 9 to the text→code spec:**
>   - nothing tying the bundle's backbone to training at runtime;
>   - the tokenizer-name check deferred at plan 9's gate.
> - **Bearing on Req 24's tooling:**
>   - the margin-timing error shape in `check_margins_first`;
>   - imported monitor-record strictness. A deterministic arm has no monitor records.

From Verification, the items this stage carries:

> - [ ] **Margin first, reference only.**
>   - The margin read logs only comparators that read no candidate.
>   - The margin record's time precedes every candidate read.
>   - `decide` refuses a margin record that does not.
> - [ ] **Rule by panel count.** A fixture decision with two panels reads 97.5%, and one with three
>       still reads 98⅓%.
> - [ ] **Deterministic arms.** A single-replicate arm record with no checkpoint validates and pairs
>       with the reference.
> - [ ] **Economic decision record.** It shows:
>   - two panels with their 95% non-inferiority and 97.5% superiority intervals;
>   - δ = 0.05 × the reference's MSE per panel;
>   - the reference `covariates+ancestors`;
>   - each candidate's provenance, none naming an employment-data input;
>   - the chosen arm, or that the reference stands.
> - [ ] **Checks.** The full suite, Ruff, the read-only YAPF check and the strict docs build pass.

From the Rollout note:

> 3. **The decision tooling (Req 24),** tested on fixtures.

### The roadmap (`specs/naics-representation-split-roadmap.md` at 32b102b), verbatim

> **Deferred items.** `specs/deferred_items.md` was read, not edited. Stage 2 discharges the
> margin-timing error shape, since `check_margins_first` is the check Req 24 extends. Its plan
> discharges imported monitor-record strictness or records why not: a deterministic arm carries no
> monitor records. The spec's other named entries went with embedding roadmap Stages 9–11 at that
> roadmap's resume. The ridge path's SVD cache stays standalone, because one reference-only read does
> not trigger it, though the candidates' sweeps may. The tie order's strictness keeps its trigger
> for the candidates' decision.

> - [ ] Stage 2: Economic decision tooling
>       Objective: Build Req 24's rule for the economic artifact and exercise it on fixtures: the
>       reference-only margin read and relative margin, deterministic single-replicate arms, a
>       comparator as the reference, and a superiority level set by the panel count.
>       Spec: Req 24 (all but the real margin read); Req 23 (what a candidate is; selection on
>       validation splits; provenance in the record); Req 18 (the economic decision reads its own
>       panels only); Verification "Margin first, reference only", "Rule by panel count",
>       "Deterministic arms", "Economic decision record" (on fixtures), "Checks"; Supersessions (D8,
>       D10's sparse baselines); Deferred items this touches; D10; D11.
>       Gap closed: Req 24 (the tooling); Req 23 (the record); Req 18 (the economic decision).
>       Consumes: Stage 1's growth reads, read names, data fingerprint and seal, through the
>       interfaces and traps plan 13's Handoff names (plan 13, Handoff:
>       `specs/plans/completed/13-growth-outcome-and-seals.md:3020`; Resolved at resume,
>       2026-10-12). The panel's read path, which takes an arm and scores `embedding` and
>       `text_only` in every read (`panels/regressor.py:163-177`, `:670-695`, `:799-813` at
>       d228132; plan 13, Handoff): the reference-only read is this stage's.
>       `tools regressor-panel`, `tools sweep` and `run_seed_sweep` refuse a growth panel, so this
>       stage lifts or routes around them (plan 13, Handoff; Resolved at resume, 2026-10-12). The
>       fixture panel's growth rows (`regressor_growth_rows` in `tests/fixtures/regressor_panel.py`;
>       found at resume; Resolved at resume, 2026-10-12). The decision package, wired to D8's three
>       panels: `PANELS` (`decision/scores.py:30`), the per-run read check and `check_margins`
>       (`decision/decide.py:196-232`, `:344-358`), `PanelSet`'s required outcome fields and
>       `SeedRun`'s required checkpoint (`decision/records.py:92-148`), a margin reference that must
>       be an arm (`:174-182`), the 5-seed floor (`decision/decide.py:85`; `utils/config.py:599` at
>       d228132, found at resume; Resolved at resume, 2026-10-12), and `tools margins` and
>       `tools decide`. Pairing only on bit-identical regressor rows (embedding roadmap Stage 4,
>       Rollout note). The deferred entries on margin timing and imported monitor records
>       (`specs/deferred_items.md:529-541`). The docs that describe today's three-panel decision
>       (`docs/usage.md`, `docs/overview.md`, `docs/quickstart.md`, `docs/index.md`, `README.md`),
>       and the level-only growth refusals that `docs/usage.md`, `CLAUDE.md` and the comment in
>       `conf/data/regressor_panel.yaml` state (found at resume; Resolved at resume, 2026-10-12).
>       Produces: A reference-only growth read that computes neither `embedding` nor `text_only`; a
>       margin record of the relative rule, naming r and that read; arm records for deterministic
>       single-replicate tables, with provenance and no checkpoint or monitor reads; an economic
>       decision over `growth_seen` and `growth_heldout` against `covariates+ancestors`, in which
>       only candidates adopted over the reference are eligible; a superiority level of
>       1 − 0.05/n, which keeps the three-panel decision at 98⅓ %. The candidates' spec decides
>       through these.
>       Exit: On fixtures: the margin read logs only comparators that read no candidate, its
>       record's time precedes every candidate read, and `decide` refuses a margin record that does
>       not (tests); a two-panel decision reads 97.5 % and a three-panel one 98⅓ % (test); a
>       single-replicate arm record with no checkpoint validates and pairs with the reference
>       (test); a fixture economic decision record shows every field Verification "Economic
>       decision record" lists (test); `check_margins_first` refuses a missing or naive read time
>       with a named error (test); Verification "Checks" passes.
>       ROUTING: writing-plans

### The embedding roadmap's D8–D11 (`specs/naics-embedding-roadmap.md` at dc7b9ea), verbatim

> - **D8 — Regressor regimes under Req 5 (Stages 4 and 7; every later decision).** Req 2 reports
>   the seen-code and held-out-code regimes separately, but Req 5 counts two panels, each with its
>   own δ and half the error rate, and ends its tie order on "the higher regressor-panel estimate".
>   The finding runs both regimes. Decision: each regime counts as a panel under Req 5, which then
>   has three: the outcome panel and the two regressor regimes. Adoption needs non-inferiority on
>   all three (the 95 % interval, unchanged) and superiority on at least one; each panel gets its
>   own δ and a third of the error rate, so superiority reads the 98⅓ % interval. This supersedes
>   the 97.5 % that Req 5 and Verification "Decision records" name; the Completion audit reads it
>   as this recorded deviation, not an unmet requirement. The final tie-break stayed open; D11
>   settles it.

> - **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
>   representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
>   the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
>   dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
>   backbone the arm uses after. The comparator measures what taxonomy training adds over the same
>   encoder reading the same text.

> - **D10 — Each panel's decision statistic (Req 5; Stage 4; Stages 7–12 read it).** Req 5 fixes a
>   δ and an interval per panel but names no statistic. Decision: the outcome panel's is per-query
>   MRR, resampled by code with its queries, the statistic D6 already selects checkpoints on. Each
>   regressor regime's is the out-of-sample mean squared error of the `covariates+embedding`
>   comparator on log employment at level 6, each row's squared error averaged over its repeats
>   (five on a validation read) before resampling by group; its Δ is oriented so that a positive
>   value favours A (B's error minus A's). The sparse comparators never read the arm, and their
>   folds depend only on the regime, level, repeat and group, so the gain over a sparse encoding
>   (over one-hot in the seen regime, over ancestors in the held-out regime) gives the same paired
>   Δ; that gain is reported for Req 1. The text-only comparator is no baseline, since it changes
>   with each arm's dimension and backbone. Every other Req 2 comparator and Req 3 metric is still
>   reported.
> - **D11 — D8's final tie-break (Req 5; Stage 4).** Req 5's tie order ends on "the higher
>   regressor-panel estimate", and D8 gives two. Decision: the held-out regime's, read as its gain
>   over ancestors, so the higher estimate is the lower error. For a held-out code, one-hot
>   predicts only the intercept and ancestors help only through levels 2–3, so that regime is where
>   the embedding's value over sparse encodings is tested.

For the economic artifact, Req 24 supersedes D8's three panels and D10's sparse baselines. D8's
decision, its records and `tools margins` stay as they are, for the levels history. D9 and D11
are unchanged, and D11 reads the gain over the reference.

### Deferred items this plan touches

`specs/deferred_items.md` at 1af1af9, verbatim:

> - [ ] Imported monitor-record strictness (Task 12, completion gate candidate): decision/records.py
>       permits boolean `checkpoint_epoch` and whitespace-only `training_run`; decision/decide.py
>       compares `detail.seed` by value, and the `SeedRun.monitor_records` docstring says oldest first
>       rather than file order. Producer records in all ten audited runs are well formed. Size: plan.
>       Done when: src/naics_embedder/decision/records.py and decide.py reject these malformed
>       imported identities before panel reads, retain valid producer records, and document file
>       order, with focused refusal tests.
> - [ ] Margin-timing error shape (completion gate candidate): `check_margins_first` in
>       src/naics_embedder/decision/decide.py fails closed with KeyError for a missing read time and
>       TypeError for a naive time. No campaign record has either shape. Deferred as imported-record
>       error handling. Size: quick-fix. Done when: it validates presence and timezone awareness and
>       raises a named ValueError before comparison, with tests covering both decision and monitor
>       reads and valid aware times.

- Task 2 discharges the margin-timing error shape (P2), and Task 3 imported monitor-record
  strictness (P3; O2). **Plan completion** ticks both.
- The ridge path's SVD cache and the tie order's strictness stay open, as the roadmap's Deferred
  items paragraph says.
- Plan completion appends one new item: trained economic candidates (P7; O6).

### This plan's decisions

Each goes beyond a spec line, or reads one. Do not reopen them during execution. Task numbers name
where each lands.

- **P1. The superiority level follows the panel count (Task 1).**
  - `superiority_level(panels)` returns `1.0 - SUPERIORITY_ERROR / panels`, where
    `SUPERIORITY_ERROR = 0.05`, and refuses fewer than one panel.
  - It replaces the constant `SUPERIORITY_LEVEL` at `decision/rule.py:27`, which is removed. Left
    in place, it would let a caller read the three-panel level for another panel count.
  - `compare_panel(..., *, superiority)` takes the level as a required keyword. D8's `decide`
    records `superiority_level(len(PANELS))` in its settings and passes it to every comparison.
    Its records still hold `1.0 - 0.05 / 3` exactly.
  - **The printed level (Task 8).** `tools decide` prints the record's own level as
    `f'{100 * level:.4g} %'`. D8's decisions now print "98.33 %" where they printed the literal
    "98⅓ %", and the economic decision prints "97.5 %".
- **P2. `read_time`, and one margin-first check for both decisions (Task 2).**
  - `read_time(record, name)` requires a selection-log record's time to be an ISO 8601 string
    that names its timezone. Otherwise its `ValueError` names the read: "has no time", "has the
    time 'yesterday', which is not ISO 8601", or "has the time …, with no timezone".
  - Pydantic writes a UTC time with `Z`. Python 3.10's `datetime.fromisoformat` refuses that
    suffix, and 3.11 reads it, so `read_time` rewrites a final `Z` as `+00:00` before parsing.
  - `check_margins_first(arms, fixed_at, *, exempt=())` replaces
    `check_margins_first(arms, margins)`.
    - It refuses margins fixed at a naive time.
    - It reads every decision read's time, and every monitor read's, through `read_time`.
    - It skips the runs whose ids are in `exempt`.
  - D8's `decide` passes its reference runs as `exempt`, as before. Task 7 passes economic arm
    records, with none exempt.
- **P3. Imported monitor-record strictness (Task 3; O2).**
  - `SeedRun.training_run` gains `pattern=r'\S'`, and `SeedRun.checkpoint_epoch` becomes
    `Optional[StrictInt]`. A training run of spaces names none, and `true` would load as epoch 1.
  - `_same(logged, named)` holds a logged identity to the named one by type as well as value,
    since `True == 1` and `3.0 == 3`. `_check_monitor_records` uses it for every identity, the
    seed included, and Task 6 reuses it for the economic reads.
  - The docstrings of `SeedRun.monitor_records` and `run_seed_sweep` say the records keep the
    file's order, and that the checks read each record's epoch, never its position.
  - Valid producer records load and check as before; the existing decision tests carry that.
- **P4. The reference-only read (Task 4).**
  - `ARM_REPRESENTATIONS = ('embedding', 'text_only')` names what a comparator can take from an
    arm's tables. `reference_comparators(regime, level)` keeps the regime's comparators that
    take neither. At level 6 they are:
    - in the seen regime, `covariates`, `one_hot`, `covariates+one_hot`, `ancestors` and
      `covariates+ancestors`;
    - in the held-out regime, `covariates`, `ancestors` and `covariates+ancestors`.
  - `RegressorPanel.reference_validation(regime, level, purpose, detail=None)` fits the remainder
    rows under the plan `validation` uses, and predicts only those comparators. It logs
    `{'level', 'comparators'}` plus `detail`, with no arm, text-only table or dimension.
  - `feature_matrix` accepts `arm=None`, and refuses a comparator that reads an arm's table
    without one.
  - `validation` and `test` now pass their comparators to `_predict`. Their outputs are unchanged,
    so plan 13's level-mode pins keep passing.
  - **Why the pairing is exact.** A comparator's fit depends only on the rows, the fold plan and
    the penalty grid, and none of them reads an arm. Task 4's test requires the reference read's
    frame to equal the same comparators' rows of an arm's read (`DataFrame.equals`).
- **P5. New record kinds beside D8's (Tasks 5–7; O1).**
  - Three kinds join `decision/records.py`: `relative_margins` (`RelativeMarginRecord`),
    `economic_arm` (`EconomicArmRecord`) and `economic_decision` (`EconomicDecisionRecord`).
    D8's kinds (`margins`, `arm`, `decision`) and their checks are unchanged.
  - `decision/economic.py` holds the economic decision's reads, checks and decision. It imports
    `decide.py`'s checks rather than widening them.
  - **Not chosen:** a reference field on `MarginRecord`, or an optional seed and checkpoint on
    `SeedRun`. Each would loosen what a D8 record guarantees.
- **P6. Where a read ran, and pairing judged on predictions (Tasks 5–6).**
  - `ReadEnvironment(platform, python, lock_sha256)` holds `platform.platform()`,
    `platform.python_version()` and the sha256 of the `uv.lock` passed as `lock`. The margin
    record and every economic arm record carry one.
  - `check_reference_pairing(margins, arm, store)` requires two things:
    - the arm read the growth panel the margins read: its draw, data fingerprint and fit
      settings;
    - in each run, the reference's predictions equal the margin read's, as rows sorted by
      panel, repeat, code and feature year, compared with `DataFrame.equals`.
  - The environment strings decide nothing; a refusal quotes both, to say where each read ran.
    Bit identity is what Req 24's exact pairing needs, and equal platform strings neither
    establish it nor are required for it. Stage 1's Handoff names the trap: reads pair only
    within one platform and one lock.
  - The bit identity was verified at plan time within each platform: on macOS (Accelerate), and
    on Linux CI under Python 3.12 and 3.10 (PR #142), each comparing reads on one machine. It was
    not checked across platforms: a margin read on the Mac need not pair with a candidate read on
    Linux or Lambda, and `check_reference_pairing` refuses one that does not.
- **P7. Economic arms are deterministic (Tasks 6–7; O6).**
  - `EconomicRun.seed` is `Optional[StrictInt] = None`, and `check_economic_arm` requires exactly
    one run, with no seed.
  - `EconomicRun` has no field for a checkpoint, training run, epoch, decoding or monitor records.
    Under `extra='forbid'`, an imported run that names one does not load.
  - Deferred to the candidates' spec, by **Plan completion**'s new item:
    - a trained candidate's economic record, with its seeds and its checkpoint's selection
      evidence, and the 5-seed floor;
    - a command that reads a candidate (O3);
    - the text-only comparator of a candidate with no backbone of its own, such as a tree
      embedding (P9).
- **P8. A candidate's provenance, verbatim (Task 6).**
  - A candidate table carries its builder's provenance beside it, at
    `panels.text_only.provenance_path(table)`, as the text-only table and the export do.
  - `check_provenance(provenance, table_sha256, matrix_fingerprint, name)` is extracted from
    `provenance_fields`. It requires a JSON object naming the table's sha256 and, if it names
    one, the table's `matrix_fingerprint`.
  - `ArtifactStore.put_candidate(table_path)` runs that check before it stores anything, then
    stores both files. `candidate_provenance(table, provenance)` reads a stored pair back as a
    `TableProvenance(file, content)`.
  - The record keeps the provenance's content verbatim. Verification's "each candidate's
    provenance, none naming an employment-data input" is read from the record.
  - The tooling checks that a provenance describes its table, not what it names. Enforcing
    Req 23's limit on how a candidate is built is the candidates' spec's (roadmap, Completion).
- **P9. D9's text-only table for every economic arm (Task 6).**
  - An economic read scores every comparator of its regime, `covariates+text_only` included. So
    each candidate names a text-only table, checked against its spec by `check_text_only`, as a
    D8 arm's is.
  - `check_recorded_text_only(spec, text_only, store)` is extracted from `check_arm`, so both
    kinds of arm share it.
  - The candidates' spec decides which text-only table suits a candidate with no backbone of its
    own (P7).
- **P10. The decision rescores stored predictions (Tasks 6–7).**
  - `decide_economic` recomputes every side's scores from its stored predictions (`_rescored`),
    rather than reading stored scores. The values it resamples are those that
    `check_reference_pairing` compared bit for bit.
  - `check_economic_arm` and `check_relative_margins` require each recorded statistic and
    reference MSE to agree with the value recomputed from the stored predictions, within
    `math.isclose(rel_tol=1e-12)` (`STATISTIC_TOLERANCE`).
  - **Why a tolerance.** A parquet round trip can change a float sum's last bit (about 1 ulp,
    measured at plan time), so exact equality would refuse intact records.
  - Each δ must equal r × its recorded reference MSE exactly, as `fix_relative_margins` computes
    it.
- **P11. Eligibility before dominance, and the reference standing (Task 7).**
  - Each candidate is compared with the reference on both growth panels. Adoption needs
    non-inferiority on both at 95 % and superiority on at least one at 97.5 %
    (`superiority_level(2)`).
  - Only the adopted are eligible. Among them, every ordered pair is compared, then
    `non_dominated` and `tie_order` pick one. `heldout_gain` is the reference's `growth_heldout`
    MSE minus the candidate's (D11, for growth).
  - With none eligible, the non-dominated set and the tie order are empty, `cycle` is false and
    `chosen` is None: the reference stands, and there is no economic artifact.
  - The record holds every comparison: each candidate against the reference first, then each
    ordered pair of eligible candidates.
  - **The bootstrap.** The reference's side is the margin read, a single replicate.
    `unit_draws(panel, …)` draws each panel's four-digit groups once for every side, and a
    single replicate's `seed_draws` is trivial.
- **P12. Margin first, both ways (Tasks 5 and 7).**
  - `fix_relative_margins` stamps `fixed_at` after its read.
  - `check_economic_margins_first` requires every record of the margin read to precede
    `fixed_at`, and every candidate read to follow it (through `check_margins_first`, P2).
  - So margins fixed again after a candidate read cannot judge that candidate: its read precedes
    them.
- **P13. The CLI (Task 8; O3).**
  - `tools growth-margins` takes `--codebook`, `--name`, `--store` and `--output`, and optionally
    `--purpose`, `--log` and `--lock`.
    - It loads the regressor panel at level 6 with `outcome` set to `growth` in memory
      (`model_copy`), whatever `conf/data/regressor_panel.yaml` says.
    - The committed config keeps `outcome: level`. So `tools regressor-panel`, `tools sweep` and
      `run_seed_sweep` keep refusing growth: this stage routes around Stage 1's refusals rather
      than lifting them, since each scores `embedding` and `text_only` on every read.
    - It refuses an existing `--output` before it loads anything. The purpose defaults to
      "economic margin read", and `--lock` to `uv.lock`.
  - r is `DecisionConfig.relative_margin`, which defaults to 0.05 and must be positive, set in
    `conf/data/decision.yaml`.
  - `tools decide` reads the margin record's `kind`.
    - `relative_margins` reads economic arm records and calls `decide_economic`; `margins`
      keeps D8's path.
    - It prints the record's own superiority level, the candidates adopted over the reference,
      and "Chosen: none; the reference covariates+ancestors stands" when none is eligible.
  - No command reads a candidate. `read_deterministic_arm` is the Python API (O3).
- **P14. Documentation and counts (Task 9).**
  - `docs/usage.md` gains a `tools growth-margins` section, and its `tools decide` section
    describes the economic decision.
  - `docs/api/decision.md` documents `decision.economic`.
  - `docs/overview.md`, `docs/quickstart.md`, `docs/index.md`, `docs/text_training.md` and the
    README describe the economic decision beside D8's, with links to `tools growth-margins`.
  - CLAUDE.md and `tests/README.md` add the economic decision and the new test file.
  - The stated counts become 119 source files, 101 unit files and 3,187 collected tests, as
    collected after Task 8.
- **P15. Effects constructed on real reads, in the decision's tests (Task 7).**
  - The fixture panel's signal is too weak to give decisive outcomes from candidate tables
    alone.
  - So `_scaled(arm, factors, store)` rewrites a candidate's `covariates+embedding` predictions
    as the outcome plus f × (the reference's prediction − the outcome), on each row and repeat.
    Each row's squared error is then f² times the reference's, so each item's Δ is (1 − f²)
    times the reference's error.
  - At f = 0.5 a candidate is adopted; at 1.0 it is non-inferior only; at 1.5 it is neither.
  - The scores and statistics are recomputed after the rewrite, so the records stay intact.
  - One test checks the intervals against an independent computation of the single-replicate
    bootstrap.

### Recorded deviations

- **Trained candidates deferred (P7; O6).** Stage 2's Spec line reads "Req 24 (all but the real
  margin read)", and its Gap closed reads "Req 24 (the tooling)". Req 24's Replicates bullet
  gives a trained candidate at least 5 seeds.
  - This plan records deterministic candidates only.
  - A trained candidate's economic record waits for the candidates' spec, with its checkpoint's
    selection evidence and the 5-seed floor.
  - **Plan completion** appends that item to `specs/deferred_items.md` (Size: plan).
- **No command reads a candidate (O3).** Req 24's tooling names records and `decide`, not a
  command that reads a candidate. `read_deterministic_arm` is the Python API, and its command
  goes to the candidates' spec, with what it reads.
- **D8's printed superiority level (P1).** `tools decide` prints "98.33 %" for D8's decisions,
  where it printed "98⅓ %". The record's value is unchanged, `1.0 - 0.05 / 3`.
- **Task 2's Python 3.10 red.** Python 3.11 and later read a `Z` suffix.
  `test_a_utc_time_written_with_z_reads_on_every_supported_python` therefore fails without the
  rewrite under 3.10 only. Task 2's red step is the collection error both versions share, and
  its Step 4 also runs under 3.10.
- **O-numbers in `## Decisions`.** writing-plans numbers decisions D1, D2, … . Here they are
  O1–O6, because bare D-numbers are the embedding roadmap's (Locators). Their source is
  "planning": the owner answered them while this plan was written, before plan review.
- **A plan-time Linux check (O5).** The dry run's commits were pushed by name and opened as
  draft, do-not-merge PR #142, for one Linux CI run before this plan was written. Its result is
  Pre-flight Step 3's CI expectation. It is not this stage's PR; **Integration** opens that,
  and closes #142 with the owner's go-ahead.
- **The file name.** This plan is named for its stage, as plans 12 and 13 were, not for its spec.
  The spec is shared with Stage 3, so retirement leaves it in place (Plan completion).

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - Code, comments and docstrings cite the spec (`Req 24`), its rulings (`U13`) or the embedding
    roadmap's decisions (`D11`), never a task number, a P-number or an O-number, which mean
    nothing once the plan retires.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. The plan's code
  was formatted at plan time, so the script should change nothing. If it changes a file, keep its
  layout and say so in the ledger. At the end, `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push without the user's go-ahead, and never push to `main`.
  - Local `main` carries private commits that must never be pushed: `config`, `graph config` and
    the others the main checkout's pre-push hook guards. Never push them or a descendant, never
    cherry-pick or merge them into a public branch, and never bypass the hook with `--no-verify`.
  - Push a branch by name: `git push -u origin <branch>`, never a bare `git push`.
  - Never run bare `git stash`.
  - Add files by explicit path, never `git add -A` or `git add .`.
  - End each commit message with the session's attribution trailer.
- **Data safety.**
  - No task reads real data: no QCEW slice, real panel, bundle or `data/` file.
  - No task builds or reads a real candidate table.
  - Tests use the fixture panel (`tests/fixtures/regressor_panel.py`) and the fixture codebook,
    and write only under `tmp_path`.
  - Growth is never read on real data before Stage 3, which needs the user's approval at the
    time.
- **The selection log.** The main checkout's `logs/selection_log.jsonl` is append-only (Req 4).
  Nothing in this plan reads or appends to it.
- **Sealed splits.** No step opens a real split, and no fixture test opens one either: every
  economic read is a validation read.
- **The lock.** `uv.lock` has sha256
  `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21`. The freeze is released
  (Req 19), but this plan needs no dependency change. If a step would change the lock, stop and
  ask.
- **Configs.**
  - `conf/config.yaml` is unchanged and keeps `supervision.manifest_path: null`.
  - `conf/data/decision.yaml` gains `relative_margin: 0.05`, its comment and a two-line header.
  - `conf/data/regressor_panel.yaml` keeps `outcome: level`. One comment line becomes two; no pin
    changes.
- **Downloads.** None.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-representation-split-roadmap.md` and `specs/deferred_items.md`, and hand the user
  the exact edit instead.
- **Docs.**
  - `uv run mkdocs build --strict` must pass locally. PR CI also builds the docs strictly.
  - Griffe is strict: every documented parameter gets its own `Args:` entry.
- **Tests.**
  - Rich wraps at 80 columns on CI, so CLI tests match on whitespace-collapsed output
    (`' '.join(result.output.split())`), or on `result.output.replace('\n', '')` for a phrase
    without spaces at the fold.
  - pyproject's `addopts` has `-v`, so a count-only listing needs `--collect-only -q -q`.
- **Bash tool.** It runs zsh.
  - Brace variables (`${FILE}`): `$FILE:t` is a zsh modifier.
  - Quote `=`-leading words (`echo '====='`) and globs (`--include='*.py'`); an unquoted one
    aborts the command.
  - If the tool refuses a heredoc or a compound command, run one plain command per call and
    write files with the Write tool.

## Decisions

The owner answered these on 2026-10-12, while this plan was written. Do not re-triage them. Each
entry carries an O-number rather than a D-number, because bare D-numbers are the embedding
roadmap's (Locators).

- **O1** (owner, planning, 2026-10-12): the economic decision gets new record kinds beside D8's,
  rather than widening D8's records. Applied in: Tasks 5–8 (P5).
- **O2** (owner, planning, 2026-10-12): discharge the imported monitor-record strictness item in
  this plan. Applied in: Task 3 (P3).
- **O3** (owner, planning, 2026-10-12): reading a deterministic candidate is Python API only; its
  command goes to the candidates' spec. Applied in: Tasks 6 and 8 (P13).
- **O4** (owner, planning, 2026-10-12): dry-run every task at plan time, and never push or open a
  PR from the dry run without the owner's go-ahead. Applied in: Workspace; every task's measured
  steps.
- **O5** (owner, planning, 2026-10-12): push the plan-ordered dry-run branch by name and open it
  as a draft, do-not-merge PR, for one Linux CI run (PR #142). Applied in: Pre-flight Step 3;
  Recorded deviations; Integration Step 3.
- **O6** (owner, planning, 2026-10-12): defer trained economic candidates to the candidates'
  spec. Applied in: Tasks 6–7 (P7); Plan completion Step 4.

---

## Workspace

- **Execution ledger:** `logs/plan14_execution.md` in the execution worktree, ignored by git.
  Write each task's commands, outputs and deviations there. Integration's last step copies it to
  the main checkout before the worktree is removed.
- **Worktree:** the app worktree the execution session runs in, or
  `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-14-economic`. Run every command
  from its root.
- **Branch:** `claude/stage-2-economic-decision`, cut at `origin/main` once this plan has merged
  (**Pre-flight** Step 1).
  - An app worktree is cut from local `main`, which carries the private commits. So the branch
    is always cut from `origin/main`, never from local `main`.
  - Without an app worktree, run, from the main checkout,
    `git worktree add --detach .claude/worktrees/plan-14-economic origin/main`, then cut the
    branch there.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder`. Nothing in this plan writes there
  but Integration's ledger copy. At plan time, local `main` was 5d19bc3: origin/main 0fdfc26 plus
  the private commits.
- **The plan-time dry run (O4).** Every task ran at plan time, in this plan's order, on a branch
  cut from 0fdfc26. Its commits are the head of draft PR #142, which GitHub keeps at
  `refs/pull/142/head` even after the PR is closed. Each task's last step compares the task's
  files with its dry-run commit:

  | Task | Dry-run commit |
  |---|---|
  | 1 | `2d7b60f55bcd2218fc1b920eb6810c3b217d8963` |
  | 2 | `93e9d9154e346c3386eea66dd79ed7fd2a411c40` |
  | 3 | `0d12b8fb5be78c16605af37993cd897856a1d766` |
  | 4 | `6d2eebd0813122f84dcced7e392f94d4aac829d6` |
  | 5 | `fd93b81ab61ea034b47a343e83ec35123affebfd` |
  | 6 | `5876ddfb70ff202d4997ab2b6c16a1c003c7cf93` |
  | 7 | `dc64266e554f713cbee43ba8f848e51ac164eae1` |
  | 8 | `92412753e1dda493084972e8bb64b333478393cf` |
  | 9 | `62b19d4e0a991b040a14eeb6666e9277906d4501` |

- **Python 3.10 environment:** `/tmp/naics_plan14_py310`. Task 2's first 3.10 run builds it, Final
  verification reuses it, and Final verification deletes it.
- **Docs build output:** `/tmp/naics_plan14_site`, deleted after each build.
- **Working directory.** The Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before every commit. If it is not this worktree, `cd` back first, or
  prefix the command with `cd <worktree> &&`.

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- Pre-flight's lock hash, platform or baseline differs from its Expected line.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A red step passes, or fails other than as its Expected line says, and the cause is not a
  transcription slip.
- After a task's commit, its files differ from its dry-run commit (Step 6), and the difference is
  neither a review fix nor a formatting change that the ledger records. Fix a transcription slip
  and compare again.
- On CI, a pairing or decision test fails while it passes here: the reference read's equality
  with an arm's read (Task 4), `check_reference_pairing`, or an interval check (Task 7).
  - Suspect BLAS before logic, and report the failing assertion.
  - Never loosen `STATISTIC_TOLERANCE` or a bit-identity check without the user's ruling.
- A step would do any of these:
  - read real QCEW slices, a real panel, the bundle, `data/` or the main checkout's selection
    log;
  - open a real split, read growth on real data, or build or read a real candidate table;
  - change `uv.lock` or `conf/config.yaml`;
  - write the main checkout beyond Integration's ledger copy;
  - push a private commit or a descendant, or push with `--no-verify`.
- `origin/main` gains a commit that touches a file in **File structure**, the split roadmap or
  `specs/deferred_items.md`, or an open PR other than #142 does.

## File structure

The complete file map. Plan completion also edits `specs/naics-representation-split-roadmap.md`
and `specs/deferred_items.md`, and moves this plan to `specs/plans/completed/`. `AGENTS.md` is a
symlink to `CLAUDE.md` and needs no edit of its own.

| Action | Path | Tasks | Responsibility |
|---|---|---|---|
| Modify | `src/naics_embedder/decision/rule.py` | 1 | `SUPERIORITY_ERROR`, `superiority_level`; `compare_panel(..., superiority=)` |
| Modify | `src/naics_embedder/decision/decide.py` | 1, 2, 3, 6, 7 | D8's level by its panel count (1); `read_time` and `check_margins_first(arms, fixed_at, exempt=)` (2); `_same` (3); `check_recorded_text_only` (6); economic runs in the margin-first check (7) |
| Modify | `tests/unit/test_decision_rule.py` | 1 | The level by panel count |
| Modify | `tests/unit/test_decision.py` | 1, 2, 3 | D8's record level (1); read-time refusals (2); strict imported identities (3) |
| Modify | `src/naics_embedder/decision/records.py` | 3, 5, 6, 7 | Strict `SeedRun` identities (3); the relative margin record (5); the economic arm record (6); the economic decision record (7) |
| Modify | `src/naics_embedder/decision/sweep.py` | 3 | Its docstring: monitor records in the file's order |
| Modify | `src/naics_embedder/panels/regressor.py` | 4 | `ARM_REPRESENTATIONS`, `reference_comparators`, `RegressorPanel.reference_validation`; `feature_matrix` without an arm |
| Modify | `tests/unit/test_regressor_panel.py` | 4 | The reference-only read |
| Create | `src/naics_embedder/decision/economic.py` | 5, 6, 7 | The margin read and relative margins (5); reading and checking a deterministic candidate (6); the economic decision (7) |
| Modify | `src/naics_embedder/decision/scores.py` | 5 | `regressor_scores(..., panels=)` |
| Create | `tests/unit/test_decision_economic.py` | 5, 6, 7 | The margins (5); deterministic candidates (6); the decision (7) |
| Modify | `tests/unit/test_decision_scores.py` | 5 | Growth panels scored only where named |
| Modify | `src/naics_embedder/decision/store.py` | 6 | `check_provenance`; `ArtifactStore.put_candidate` and `candidate_provenance` |
| Create | `tests/fixtures/economic.py` | 6 | `candidate_spec`, `write_candidate` |
| Modify | `src/naics_embedder/cli/commands/tools.py` | 8 | `tools growth-margins`; `tools decide` by margin kind |
| Modify | `src/naics_embedder/utils/config.py` | 8 | `DecisionConfig.relative_margin` |
| Modify | `conf/data/decision.yaml` | 8 | `relative_margin: 0.05` |
| Modify | `conf/data/regressor_panel.yaml` | 8 | Its comment names `tools growth-margins` |
| Modify | `tests/unit/test_cli_commands.py` | 8 | Both commands |
| Modify | `tests/unit/test_config.py` | 8 | The relative margin's default and bound |
| Modify | `docs/usage.md`, `docs/api/decision.md` | 9 | `tools growth-margins`, the economic decision and its API |
| Modify | `docs/overview.md`, `docs/quickstart.md`, `docs/index.md`, `docs/text_training.md` | 9 | The economic decision beside D8's, with links to `tools growth-margins` |
| Modify | `CLAUDE.md`, `README.md`, `tests/README.md` | 9 | The economic decision, the counts and the test seams |

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

> Deviation: #143 was still open when execution began; the user merged it first (2026-10-13), and the branch was cut from origin/main 765f43e.

Run: `git fetch origin`, then:

```bash
git log --oneline --reverse origin/main -- specs/plans/14-economic-decision-tooling.md
```
Expected: this plan's commits, oldest first: `docs(plans): add plan 14, economic decision
tooling`, then any review follow-ups (`docs(plans): …`). If there is no output, this plan has not
merged: stop and ask.

Run: `git checkout --no-track -B claude/stage-2-economic-decision origin/main`
Expected: `Switched to a new branch 'claude/stage-2-economic-decision'` (or `Reset branch`).

Run: `git status --short --branch`, then `git log --oneline origin/main..HEAD`
Expected: `## claude/stage-2-economic-decision` and nothing else, then no output. No private
commit is on this branch.

Run: `git log --oneline 0fdfc26..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: this plan's commits, and possibly their merge commit; no other commit.
- If anything else landed, read it.
- If it touches a file in **File structure**, the split roadmap or `specs/deferred_items.md`,
  stop and ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**, but draft PR #142, the plan-time dry
run, unless it has been closed. If another does, stop and ask.

Run: `git fetch origin pull/142/head`, then
`git cat-file -t 62b19d4e0a991b040a14eeb6666e9277906d4501`
Expected: `commit`. The dry run's commits are now local, for each task's Step 6. If the fetch
fails, record that in the ledger and skip each Step 6 comparison.

- [x] **Step 2: Build the environment**

Run: `uv sync --locked`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number (3.12.12 at plan time).

Run:

```bash
uv run python -c "import numpy, polars, pydantic; print(numpy.__version__, polars.__version__, pydantic.__version__)"
```
Expected: `2.3.4 1.35.1 2.12.4`.

Run: `shasum -a 256 uv.lock`
Expected: `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21  uv.lock`. If it
differs, stop and ask.

Run: `uname -sm`
Expected: `Darwin arm64`. The plan's counts were measured there; two level-mode pins run only
there. Elsewhere, stop and ask.

- [x] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `3121 passed, 2 skipped`, measured at 0fdfc26 on 2026-10-12 in a worktree without
`data/naics_descriptions.parquet`. One skip needs CUDA. The other needs that file, which only the
main checkout has.
- Each later full-suite run must pass with those skips, and each task gives its count.
- The warnings count varies under xdist; ignore it.

The full suite after each task, measured at plan time on this Mac:

| After | Passed | Skipped | Collected |
|---|---:|---:|---:|
| Pre-flight (0fdfc26) | 3121 | 2 | 3123 |
| Task 1 | 3122 | 2 | 3124 |
| Task 2 | 3133 | 2 | 3135 |
| Task 3 | 3137 | 2 | 3139 |
| Task 4 | 3143 | 2 | 3145 |
| Task 5 | 3151 | 2 | 3153 |
| Task 6 | 3165 | 2 | 3167 |
| Task 7 | 3180 | 2 | 3182 |
| Task 8 | 3185 | 2 | 3187 |
| Task 9 | 3185 | 2 | 3187 |

CI runs Linux, with no MPS and no `data/`. Each test job runs pytest twice: a 47-test
remote-transport step, then the full suite with coverage.
- At 0fdfc26, main's full-suite line read `3110 passed, 13 skipped` under both Pythons.
- On PR #142, the dry run's last commit read `3174 passed, 13 skipped` under both Pythons (O5).
  This stage's PR should read the same: every new test runs on Linux.

- [x] **Step 4: Route the tasks**

> Deviation: implementers ran on Sonnet; task-reviewers ran on Sonnet for Tasks 1-3 and 9 and on Opus for Tasks 4-8.

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–8 each get a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**.
- Every code block in a task is exact, so an implementer copies it. Each "Replace" text occurs
  exactly once in its file when its edit is made, and one file's edits are made in the order
  given.
- At plan time, a script applied every block of this plan, in order, to 0fdfc26's files and
  reproduced each task's dry-run commit exactly.
- Task 9 (documentation) may go to the docs-writer agent. Its acceptance checks are the gate.
- **Final verification**, **Plan completion** and **Integration** run inline in the controller
  session.
- Model routing: execution runs on Sonnet, the task-reviewer floors at Sonnet, and the final
  code-reviewer is pinned to Opus.

## Tasks

Each task's red step was measured at plan time on the task's parent commit with the task's test
edits applied, and each green step on the task's own commit. The counts are for the named test
files only. A **Files** list's line ranges locate its Replace texts in the file as the task finds
it, before its edits.

### Task 1: The superiority level by panel count

Req 24's tooling: the superiority level "follows the decision's panel count, 1 − 0.05/n", where
`decision/rule.py:27` hard-codes three panels. A three-panel decision still reads 98⅓ % (P1).

The tests cover:
- the level for two and three panels, and the refusal of zero panels;
- a set of replicates whose lower bound clears zero at 97.5 % and not at 98⅓ %;
- D8's decision record, which still holds `1.0 - 0.05 / 3`.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 60-63, 503 and 537
- Modify: `src/naics_embedder/decision/rule.py`, lines 2-6, 27, 36-37, 43-49 and 54-57
- Test: `tests/unit/test_decision.py`, lines 25-26 and 240
- Test: `tests/unit/test_decision_rule.py`, lines 2-3, 12-13, 17, 21, 39-43 and 61

**Interfaces:**
- Consumes: nothing from an earlier task.
- Produces, in `naics_embedder.decision.rule`:
  - `SUPERIORITY_ERROR = 0.05`;
  - `superiority_level(panels: int) -> float`: `1.0 - SUPERIORITY_ERROR / panels`. Below one
    panel it raises `ValueError` with "a decision reads at least one panel, not 0";
  - `compare_panel(panel, delta, replicates, margin, *, superiority: float) -> PanelComparison`.
  - `SUPERIORITY_LEVEL` is removed.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_decision_rule.py`, make six edits, in this order.

Replace:

```python
Req 5's rule over D8's three panels: 95 % non-inferiority, 98⅓ % superiority, adoption, the
non-dominated set and the tie order (D11 breaks the last tie).
```

with:

```python
Req 5's rule: 95 % non-inferiority, superiority at 1 − 0.05/n over a decision's n panels (98⅓ %
over D8's three, Req 24), adoption, the non-dominated set and the tie order (D11 breaks the last
tie).
```

Replace:

```python
    NONINFERIORITY_LEVEL,
    SUPERIORITY_LEVEL,
```

with:

```python
    NONINFERIORITY_LEVEL,
```

Replace:

```python
    non_dominated,
```

with:

```python
    non_dominated,
    superiority_level,
```

Replace:

```python
pytestmark = pytest.mark.unit
```

with:

```python
pytestmark = pytest.mark.unit

# D8's three panels
THREE_PANELS = superiority_level(3)
```

Replace:

```python
    return compare_panel(panel, (low + high) / 2, replicates, margin)

def test_the_levels_are_d8s():
    assert NONINFERIORITY_LEVEL == 0.95
    assert SUPERIORITY_LEVEL == pytest.approx(0.98333333)
```

with:

```python
    return compare_panel(panel, (low + high) / 2, replicates, margin, superiority=THREE_PANELS)

def test_superiority_splits_the_error_rate_evenly_over_a_decisions_panels():
    '''Req 24: 1 − 0.05/n, so D8's three panels still read 98⅓ % and two growth panels 97.5 %.'''

    assert NONINFERIORITY_LEVEL == 0.95
    assert superiority_level(3) == 1.0 - 0.05 / 3
    assert superiority_level(3) == pytest.approx(0.98333333)
    assert superiority_level(2) == 0.975
    with pytest.raises(ValueError, match='at least one panel, not 0'):
        superiority_level(0)

def test_superiority_reads_the_level_the_decision_gives():
    # Replicates spread evenly over [-0.04, 3.96]: the lower bound is 0.01 at 97.5 % and
    # -0.0067 at 98⅓ %
    replicates = np.linspace(-0.04, 3.96, 12001)

    two = compare_panel('growth_seen', 1.96, replicates, 1.0, superiority=superiority_level(2))
    three = compare_panel('growth_seen', 1.96, replicates, 1.0, superiority=THREE_PANELS)

    assert two.superiority_interval[0] == pytest.approx(0.01)
    assert two.superior
    assert three.superiority_interval[0] == pytest.approx(-0.04 + 4 / 120)
    assert not three.superior
```

Replace:

```python
        compare_panel('outcome', float('nan'), np.linspace(-1.0, 3.0, 12001), 0.95)
```

with:

```python
        compare_panel(
            'outcome', float('nan'), np.linspace(-1.0, 3.0, 12001), 0.95, superiority=THREE_PANELS
        )
```

In `tests/unit/test_decision.py`, make two edits, in this order.

Replace:

```python
)
from naics_embedder.decision.rule import SUPERIORITY_LEVEL
```

with:

```python
)
```

Replace:

```python
    assert record.settings.superiority_level == SUPERIORITY_LEVEL
```

with:

```python
    # Three panels still read 98⅓ % (Req 24)
    assert record.settings.superiority_level == 1.0 - 0.05 / 3
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision_rule.py tests/unit/test_decision.py -q`
Expected: FAIL, `1 error`. Collecting `tests/unit/test_decision_rule.py` raises
`ImportError: cannot import name 'superiority_level' from 'naics_embedder.decision.rule'`.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/decision/rule.py`, make five edits, in this order.

Replace:

```python
Req 5's rule on paired resamples, over D8's three panels.

- **Non-inferior** on a panel: the lower bound of the 95 % interval on Δ exceeds −δ.
- **Superior** on a panel: the 98⅓ % interval lies above zero. D8 gives each of the three panels
  a third of the 5 % error rate; this supersedes the 97.5 % that Req 5 names for two panels.
```

with:

```python
Req 5's rule on paired resamples, over a decision's panels.

- **Non-inferior** on a panel: the lower bound of the 95 % interval on Δ exceeds −δ.
- **Superior** on a panel: the interval at 1 − 0.05/n lies above zero, for a decision over n
  panels, each of which gets an even share of the 5 % error rate (Req 24). D8's three panels read
  98⅓ %, which supersedes the 97.5 % that Req 5 names for two; the economic decision's two growth
  panels read 97.5 %.
```

Replace:

```python
SUPERIORITY_LEVEL = 1.0 - 0.05 / 3
```

with:

```python
# The error rate superiority splits evenly over a decision's panels (Req 24)
SUPERIORITY_ERROR = 0.05
```

Replace:

```python
def compare_panel(
    panel: str, delta: float, replicates: np.ndarray, margin: float
```

with:

```python
def superiority_level(panels: int) -> float:
    '''
    The superiority interval's level for a decision over ``panels`` panels: 1 − 0.05/n (Req 24).

    Raises:
        ValueError: If ``panels`` is below 1.
    '''

    if panels < 1:
        raise ValueError(f'a decision reads at least one panel, not {panels}')
    return 1.0 - SUPERIORITY_ERROR / panels

def compare_panel(
    panel: str,
    delta: float,
    replicates: np.ndarray,
    margin: float,
    *,
    superiority: float,
```

Replace:

```python
        delta: The point estimate of Δ, positive when it favours A.
        replicates: Δ on each paired resample, oriented the same way.
        margin: The panel's δ.
    '''

    noninferiority = percentile_interval(replicates, NONINFERIORITY_LEVEL)
    superiority = percentile_interval(replicates, SUPERIORITY_LEVEL)
```

with:

```python
        panel: The panel's name.
        delta: The point estimate of Δ, positive when it favours A.
        replicates: Δ on each paired resample, oriented the same way.
        margin: The panel's δ.
        superiority: The superiority interval's level (``superiority_level``).
    '''

    noninferiority = percentile_interval(replicates, NONINFERIORITY_LEVEL)
    upper = percentile_interval(replicates, superiority)
```

Replace:

```python
        superiority_interval=superiority,
        margin=float(margin),
        non_inferior=noninferiority[0] > -margin,
        superior=superiority[0] > 0.0,
```

with:

```python
        superiority_interval=upper,
        margin=float(margin),
        non_inferior=noninferiority[0] > -margin,
        superior=upper[0] > 0.0,
```

In `src/naics_embedder/decision/decide.py`, make three edits, in this order.

Replace:

```python
    SUPERIORITY_LEVEL,
    compare,
    compare_panel,
    non_dominated,
```

with:

```python
    compare,
    compare_panel,
    non_dominated,
    superiority_level,
```

Replace:

```python
        superiority_level=SUPERIORITY_LEVEL,
```

with:

```python
        superiority_level=superiority_level(len(PANELS)),
```

Replace:

```python
                    margins.margin(panel),
```

with:

```python
                    margins.margin(panel),
                    superiority=settings.superiority_level,
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_rule.py tests/unit/test_decision.py -q`
Expected: PASS, `83 passed`.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py src/naics_embedder/decision/rule.py tests/unit/test_decision.py tests/unit/test_decision_rule.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3122 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/decide.py src/naics_embedder/decision/rule.py tests/unit/test_decision.py tests/unit/test_decision_rule.py
git commit -m "feat(decision): set the superiority level by the decision's panel count (Req 24)"
```

Run:

```bash
git diff --stat 2d7b60f55bcd2218fc1b920eb6810c3b217d8963 HEAD -- src/naics_embedder/decision/decide.py src/naics_embedder/decision/rule.py tests/unit/test_decision.py tests/unit/test_decision_rule.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 2: Refuse a missing or naive read time by name

The margin-timing deferred item: `check_margins_first` fails with `KeyError` for a missing read
time and `TypeError` for a naive one. Each now raises a named `ValueError` before any comparison.
The check takes the margins' time, so Task 7's economic decision can share it (P2).

The tests cover:
- four malformed read times (missing, null, naive, not ISO 8601), on a decision read and on a
  monitor read;
- margins fixed at a time with no timezone;
- aware times that compare across offsets: a read a minute after the margins, written at UTC+5,
  is after them;
- a UTC time written with `Z`, which only Python 3.10 refuses (Recorded deviations).

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 22, 32, 361-380 and 497
- Test: `tests/unit/test_decision.py`, lines 13, 18 and 703

**Interfaces:**
- Consumes: Task 1's `compare_panel(..., superiority=)`, unchanged here.
- Produces, in `naics_embedder.decision.decide`:
  - `read_time(record: Any, name: str) -> datetime`;
  - `check_margins_first(arms: Sequence[Any], fixed_at: datetime, *,
    exempt: Collection[str] = ()) -> None`. Task 7 passes it economic arm records.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_decision.py`, make three edits, in this order.

Replace:

```python
from datetime import datetime, timedelta
```

with:

```python
from datetime import datetime, timedelta, timezone
```

Replace:

```python
from naics_embedder.decision.decide import check_arm, check_seed_distance, decide, fix_margins
```

with:

```python
from naics_embedder.decision.decide import (
    check_arm,
    check_seed_distance,
    decide,
    fix_margins,
    read_time,
)
```

Replace:

```python
def test_a_decision_record_carries_every_trained_runs_monitor_reads(store, tmp_path):
```

with:

```python
# Each malformed read time, and what follows the read's name in its refusal
TIMELESS = [
    (lambda read: read.pop('time'), 'has no time'),
    (lambda read: read.update(time=None), 'has no time'),
    (lambda read: read.update(time='2026-10-12T12:00:00'), 'has the time .*, with no timezone'),
    (lambda read: read.update(time='yesterday'), "has the time 'yesterday', which is not ISO 8601"),
]
TIMELESS_IDS = ['missing', 'null', 'naive', 'not-iso-8601']

@pytest.mark.parametrize('edit, message', TIMELESS, ids=TIMELESS_IDS)
def test_a_decision_read_without_an_aware_time_is_refused_by_name(
    store, tmp_path, reference, margins, edit, message
):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('timeless', dimension=32), {}),
        lambda data: edit(_run(data)['log_records'][1]),
    )

    with pytest.raises(ValueError, match=f'^timeless seed 3: its regressor_seen read {message}'):
        _decide([arm, reference], margins, store)

@pytest.mark.parametrize('edit, message', TIMELESS, ids=TIMELESS_IDS)
def test_a_monitor_read_without_an_aware_time_is_refused_by_name(
    store, tmp_path, reference, margins, edit, message
):
    arm = _edited(
        synthetic_arm(
            store, tmp_path, spec('timeless', dimension=32), {}, monitor_mrrs=MONITOR_MRRS
        ),
        lambda data: edit(_monitor_read(data)),
    )

    with pytest.raises(
        ValueError, match=f'^timeless seed 3: its monitor read of epoch 2 {message}'
    ):
        _decide([arm, reference], margins, store)

def test_margins_fixed_at_a_time_without_a_timezone_are_refused(
    store, tmp_path, reference, margins
):
    naive = MarginRecord.model_validate(
        _json_edited(
            margins, lambda data: data.update(fixed_at=data['fixed_at'].removesuffix('Z'))
        )
    )
    candidate = synthetic_arm(store, tmp_path, spec('candidate', dimension=32), {})

    assert naive.fixed_at.utcoffset() is None
    with pytest.raises(ValueError, match='^the margins were fixed at .*, with no timezone'):
        _decide([candidate, reference], naive, store)

def test_aware_read_times_compare_whatever_their_offset(store, tmp_path, reference, margins):
    '''A minute after the margins, written at UTC+5, is after them; a minute before is not.'''

    plus_five = timezone(timedelta(hours=5))

    def read_at(name, minutes):
        '''An arm whose every decision read is ``minutes`` after the margins, written at UTC+5.'''

        time = (margins.fixed_at + timedelta(minutes=minutes)).astimezone(plus_five).isoformat()
        arm = synthetic_arm(store, tmp_path, spec(name, dimension=32), {})
        return time, _edited(
            arm, lambda data: [
                read.update(time=time) for run in data['runs'] for read in run['log_records']
            ]
        )

    _, later = read_at('later', 1)
    before, early = read_at('early', -1)

    _decide([later, reference], margins, store)
    with pytest.raises(ValueError, match=f'^early seed 0 read at {re.escape(before)}, before'):
        _decide([early, reference], margins, store)

def test_a_utc_time_written_with_z_reads_on_every_supported_python():
    '''Pydantic writes a UTC time with Z, which Python 3.10's ``fromisoformat`` cannot read.'''

    read = {'time': '2026-10-12T12:00:00Z'}

    assert read_time(read, 'a read') == datetime(2026, 10, 12, 12, tzinfo=timezone.utc)

def test_a_decision_record_carries_every_trained_runs_monitor_reads(store, tmp_path):
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: FAIL, `1 error`. Collecting the file raises
`ImportError: cannot import name 'read_time' from 'naics_embedder.decision.decide'`.

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan14_py310 uv run --locked pytest tests/unit/test_decision.py -q
```
Expected: the same `1 error`. This first run builds the 3.10 environment.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/decision/decide.py`, make four edits, in this order.

Replace:

```python
reads included, before the margins were fixed.
```

with:

```python
reads included, before the margins were fixed. Every read's time, and the margins', must name its
timezone.
```

Replace:

```python
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
```

with:

```python
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple
```

Replace:

```python
def check_margins_first(arms: Sequence[ArmRecord], margins: MarginRecord) -> None:
    '''
    Require every run but the margin record's reference runs to read after the margins were fixed:
    its decision reads and, for a trained run, the monitor reads that selected its checkpoint.

    Raises:
        ValueError: If a run read before ``margins.fixed_at``.
    '''

    reference = {run.run_id for run in margins.reference.runs}
    for arm in arms:
        for run in arm.runs:
            if run.run_id in reference:
                continue
            reads = [*run.log_records, *(record['read'] for record in run.monitor_records)]
            first = min(datetime.fromisoformat(record['time']) for record in reads)
            if first < margins.fixed_at:
                raise ValueError(
                    f'{arm.spec.name} seed {run.seed} read at {first.isoformat()}, before the '
                    f'margins were fixed at {margins.fixed_at.isoformat()} (Req 5)'
```

with:

```python
def read_time(record: Any, name: str) -> datetime:
    '''
    A selection-log record's time: an ISO 8601 string that names its timezone.

    Args:
        record: The record, as the selection log appended it.
        name: Names the read in a refusal.

    Raises:
        ValueError: If the record has no time, or its time is not ISO 8601 or names no timezone.
    '''

    time = record.get('time') if isinstance(record, dict) else None
    if not isinstance(time, str):
        raise ValueError(f'{name} has no time')
    # Python 3.10's fromisoformat cannot read the Z that pydantic writes for UTC
    utc = time[:-1] + '+00:00' if time.endswith('Z') else time
    try:
        parsed = datetime.fromisoformat(utc)
    except ValueError:
        raise ValueError(f'{name} has the time {time!r}, which is not ISO 8601') from None
    if parsed.utcoffset() is None:
        raise ValueError(f'{name} has the time {time}, with no timezone')
    return parsed

def check_margins_first(
    arms: Sequence[Any], fixed_at: datetime, *, exempt: Collection[str] = ()
) -> None:
    '''
    Require every run but the ``exempt`` ones to read after the margins were fixed at ``fixed_at``:
    its decision reads and, for a trained run, the monitor reads that selected its checkpoint.

    Run it after ``check_arm``, which requires each monitor record to name its epoch.

    Raises:
        ValueError: If ``fixed_at`` or a read's time is missing or names no timezone, so that
            the two cannot be compared, or if a run read before ``fixed_at``.
    '''

    if fixed_at.utcoffset() is None:
        raise ValueError(f'the margins were fixed at {fixed_at.isoformat()}, with no timezone')
    for arm in arms:
        for run in arm.runs:
            if run.run_id in exempt:
                continue
            name = f'{arm.spec.name} seed {run.seed}'
            reads = [
                read_time(record, f'{name}: its {record.get("panel")} read')
                for record in run.log_records
            ]
            reads.extend(
                read_time(
                    record['read'],
                    f'{name}: its monitor read of epoch {record["read"]["detail"]["epoch"]}'
                ) for record in run.monitor_records
            )
            first = min(reads)
            if first < fixed_at:
                raise ValueError(
                    f'{name} read at {first.isoformat()}, before the margins were fixed at '
                    f'{fixed_at.isoformat()} (Req 5)'
```

Replace:

```python
    check_margins_first(arms, margins)
```

with:

```python
    reference = {run.run_id for run in margins.reference.runs}
    check_margins_first(arms, margins.fixed_at, exempt=reference)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: PASS, `86 passed`.

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan14_py310 uv run --locked pytest tests/unit/test_decision.py -q
```
Expected: PASS, `86 passed`. Under 3.10, the `Z` test passes only through `read_time`'s rewrite.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py tests/unit/test_decision.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3133 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/decide.py tests/unit/test_decision.py
git commit -m "fix(decision): refuse a missing or naive read time by name before comparing it" \
  -m "Discharges the margin-timing deferred item: check_margins_first validated
neither a read's time nor the margins', and failed with KeyError or TypeError."
```

Run:

```bash
git diff --stat 93e9d9154e346c3386eea66dd79ed7fd2a411c40 HEAD -- src/naics_embedder/decision/decide.py tests/unit/test_decision.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 3: Hold imported monitor-record identities to their types

The imported monitor-record strictness item, discharged by the owner's ruling (O2; P3). The
records, the seed comparison and the docstrings change; valid producer records still load and
check.

The tests cover:
- an imported run whose training run is spaces, or whose checkpoint epoch is `true`, does not
  load;
- a monitor read that names its run's seed as `True` or `3.0` is refused.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 250, 256 and 302-303
- Modify: `src/naics_embedder/decision/records.py`, lines 29, 131-133 and 145-147
- Modify: `src/naics_embedder/decision/sweep.py`, lines 69-70
- Test: `tests/unit/test_decision.py`, line 691

**Interfaces:**
- Consumes: Task 2's `check_margins_first`, unchanged here.
- Produces:
  - `naics_embedder.decision.decide._same(logged: Any, named: Any) -> bool`, which Task 6
    imports;
  - `SeedRun.training_run: Optional[str]`, with `min_length=1` and `pattern=r'\S'`;
  - `SeedRun.checkpoint_epoch: Optional[StrictInt]`, with `ge=0`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_decision.py`, make one edit.

Replace:

```python
def test_a_monitor_read_before_the_margins_were_fixed_is_refused(
```

with:

```python
@pytest.mark.parametrize(
    'field, value, error',
    [('training_run', '   ', 'string_pattern_mismatch'), ('checkpoint_epoch', True, 'int_type')],
    ids=['a-whitespace-training-run', 'a-boolean-checkpoint-epoch'],
)
def test_an_imported_run_with_a_whitespace_training_run_or_a_boolean_epoch_does_not_load(
    tmp_path, trained, field, value, error
):
    '''A training run of spaces names none, and true would load as epoch 1.'''

    path = tmp_path / 'imported.json'
    path.write_text(
        json.dumps(_json_edited(trained, lambda data: _run(data).update({field: value})))
    )

    with pytest.raises(ValidationError, match=f'runs\\.3\\.{field}\\n.*{error}'):
        read_record(path, ArmRecord)

@pytest.mark.parametrize(
    'seed, logged', [(1, True), (3, 3.0)], ids=['a-boolean-seed', 'a-float-seed']
)
def test_a_monitor_read_names_its_runs_seed_as_an_integer(store, trained, seed, logged):
    '''True == 1 and 3.0 == 3: only the type tells these from the run's seed.'''

    arm = _edited(
        trained, lambda data: _monitor_read(data, seed=seed)['detail'].update(seed=logged)
    )

    with pytest.raises(
        ValueError, match=f"^trained seed {seed}: a monitor read names another \\['seed'\\]"
    ):
        check_arm(arm, store, min_seeds=5)

def test_a_monitor_read_before_the_margins_were_fixed_is_refused(
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: FAIL, `4 failed, 86 passed`. Each new case fails with `DID NOT RAISE`: the two
`ValidationError` cases and the two `ValueError` cases.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/decision/records.py`, make three edits, in this order.

Replace:

```python
from pydantic import BaseModel, ConfigDict, Field
```

with:

```python
from pydantic import BaseModel, ConfigDict, Field, StrictInt
```

Replace:

```python
        monitor_records: The training run's ``monitor_reads.jsonl`` records, oldest first, each
            ``{'mrr': …, 'read': …}``: an epoch's MRR, and its read as the selection log
            appended it.
```

with:

```python
        monitor_records: The training run's ``monitor_reads.jsonl`` records, in the file's order,
            each ``{'mrr': …, 'read': …}``: an epoch's MRR, and its read as the selection log
            appended it. The checks read each record's epoch, never its position.
```

Replace:

```python
    # Each defaults, so a record written before runs carried their monitor reads still loads
    training_run: Optional[str] = Field(default=None, min_length=1)
    checkpoint_epoch: Optional[int] = Field(default=None, ge=0)
```

with:

```python
    # Each defaults, so a record written before runs carried their monitor reads still loads. An
    # imported record's identities are strict: a training run of spaces names none, and true
    # would load as epoch 1
    training_run: Optional[str] = Field(default=None, min_length=1, pattern=r'\S')
    checkpoint_epoch: Optional[StrictInt] = Field(default=None, ge=0)
```

In `src/naics_embedder/decision/decide.py`, make three edits, in this order.

Replace:

```python
def _check_monitor_records(arm: ArmRecord, run: SeedRun) -> None:
```

with:

```python
def _same(logged: Any, named: Any) -> bool:
    '''Whether a logged identity is the named one: equal, and of its type (True == 1, 3.0 == 3).'''

    return type(logged) is type(named) and logged == named

def _check_monitor_records(arm: ArmRecord, run: SeedRun) -> None:
```

Replace:

```python
    epoch.
```

with:

```python
    epoch. Each read names the run's identities by type as well as value.
```

Replace:

```python
        }
        wrong = sorted(key for key in named if logged[key] != named[key])
```

with:

```python
        }
        wrong = sorted(key for key in named if not _same(logged[key], named[key]))
```

In `src/naics_embedder/decision/sweep.py`, make one edit.

Replace:

```python
        monitor_records: A trained seed's ``monitor_reads.jsonl`` records, oldest first: the
            reads that selected its checkpoint.
```

with:

```python
        monitor_records: A trained seed's ``monitor_reads.jsonl`` records, in the file's order:
            the reads that selected its checkpoint.
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: PASS, `90 passed`.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py src/naics_embedder/decision/records.py src/naics_embedder/decision/sweep.py tests/unit/test_decision.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3137 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

> Deviation: the implementer's git add was refused by the auto-mode classifier, so the controller committed its verified tree as b3ddfbe.

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/decide.py src/naics_embedder/decision/records.py src/naics_embedder/decision/sweep.py tests/unit/test_decision.py
git commit -m "fix(decision): hold imported monitor-record identities to their types" \
  -m "Discharges the imported monitor-record strictness deferred item: a record no
longer loads a boolean checkpoint epoch or a training run of spaces, a monitor
read's seed must match by type as well as value, and the docstrings say the
records keep the file's order."
```

Run:

```bash
git diff --stat 0d12b8fb5be78c16605af37993cd897856a1d766 HEAD -- src/naics_embedder/decision/decide.py src/naics_embedder/decision/records.py src/naics_embedder/decision/sweep.py tests/unit/test_decision.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 4: The reference-only read

Req 24's margin read "scores only comparators that read no candidate table … It computes neither
`embedding` nor `text_only`". Every read the panel offers takes an arm and scores both
(`panels/regressor.py:163-177`, `:670-695`, `:799-813` at d228132, as the roadmap's Consumes
cites). This task adds a read that takes none (P4).

The tests cover:
- the reference comparators by regime and level;
- the reference read's log record and predictions, which name and score only those comparators,
  with no arm in the read's detail;
- each reference comparator's predictions, equal to those of an arm's read;
- the refusal of a comparator that reads an arm's table when no arm is read.

**Files:**
- Modify: `src/naics_embedder/panels/regressor.py`, lines 20, 110, 178, 443, 447, 452, 689-695, 784,
  792, 834-835 and 840
- Test: `tests/unit/test_regressor_panel.py`, lines 10, 41 and 1244-1245

**Interfaces:**
- Consumes: Stage 1's growth panel, `RegressorPanel(..., outcome=RegressorOutcome.GROWTH)`, and
  the fixtures `growth_panel`, `regressor_arm` and `regressor_rows` in
  `tests/unit/test_regressor_panel.py`.
- Produces, in `naics_embedder.panels.regressor`:
  - `ARM_REPRESENTATIONS = ('embedding', 'text_only')`;
  - `reference_comparators(regime: Regime, level: int) -> Tuple[str, ...]`;
  - `feature_matrix(comparator, frame, level, arm: Optional[ArmTables]) -> np.ndarray`. With no
    arm, a comparator that reads an arm's table raises `ValueError` with "the comparator
    'embedding' reads an arm's table, and no arm is read";
  - `RegressorPanel.reference_validation(regime, level, purpose, detail=None) -> pl.DataFrame`.
    It returns `PREDICTION_COLUMNS` and logs one validation read, whose detail is
    `{'level': level, 'comparators': [...]}` plus `detail`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_regressor_panel.py`, make three edits, in this order.

Replace:

```python
import platform
```

with:

```python
import platform
import re
```

Replace:

```python
    outer_plan,
```

with:

```python
    outer_plan,
    reference_comparators,
```

Replace:

```python
        assert built.data_fingerprint(level) == expected.data_fingerprint(level)

```

with:

```python
        assert built.data_fingerprint(level) == expected.data_fingerprint(level)

# -------------------------------------------------------------------------------------------------
# The reference-only read (Req 24): the margin read reads no arm's tables
# -------------------------------------------------------------------------------------------------

def test_the_reference_comparators_are_those_that_read_no_arms_tables():
    assert reference_comparators(Regime.SEEN, 6) == (
        'covariates', 'one_hot', 'covariates+one_hot', 'ancestors', 'covariates+ancestors'
    )
    assert reference_comparators(Regime.HELDOUT, 6) == (
        'covariates', 'ancestors', 'covariates+ancestors'
    )
    assert reference_comparators(Regime.SEEN, 2) == ('covariates', 'one_hot', 'covariates+one_hot')

@pytest.mark.parametrize('regime', list(Regime))
def test_the_reference_read_logs_and_scores_only_comparators_that_read_no_arm(
    growth_panel, log, regime
):
    predictions = growth_panel.reference_validation(regime, 6, PURPOSE, detail={'run': 'margins'})

    names = reference_comparators(regime, 6)
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert tuple(predictions.get_column('comparator').unique(maintain_order=True)) == names
    (record, ) = log.records()
    name = READ_NAMES[RegressorOutcome.GROWTH][regime]
    assert (record['event'], record['panel'], record['split']) == ('read', name, 'validation')
    assert record['fingerprint'] == growth_panel.fingerprint
    # No arm, text-only table or dimension: the read names no candidate
    assert record['detail'] == {'level': 6, 'comparators': list(names), 'run': 'margins'}

@pytest.mark.parametrize('regime', list(Regime))
def test_each_reference_comparator_predicts_as_in_any_arms_read(
    growth_panel, regressor_arm, regime
):
    '''Req 24: the reference reads no candidate, so its errors pair exactly with each one's.'''

    reference = growth_panel.reference_validation(regime, 6, PURPOSE)
    read = growth_panel.validation(regime, 6, regressor_arm, PURPOSE)

    names = list(reference_comparators(regime, 6))
    assert reference.equals(read.filter(pl.col('comparator').is_in(names)))

def test_a_comparator_that_reads_an_arms_table_needs_an_arm(regressor_rows):
    for comparator in ('embedding', 'covariates+text_only'):
        with pytest.raises(ValueError, match=f"{re.escape(comparator)}' reads an arm's table"):
            feature_matrix(comparator, regressor_rows[6], 6, None)

```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_regressor_panel.py -q`
Expected: FAIL, `1 error`. Collecting the file raises
`ImportError: cannot import name 'reference_comparators' from 'naics_embedder.panels.regressor'`.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/panels/regressor.py`, make eleven edits, in this order.

Replace:

```python
  and the text-only table reduced by PCA to the arm's dimension (D9).
```

with:

```python
  and the text-only table reduced by PCA to the arm's dimension (D9). The margin read (Req 24)
  scores only the comparators that read no arm's tables, and reads no arm.
```

Replace:

```python
REPRESENTATIONS = ('embedding', 'one_hot', 'ancestors', 'text_only')
```

with:

```python
REPRESENTATIONS = ('embedding', 'one_hot', 'ancestors', 'text_only')
# What a read takes from an arm's tables (ArmTables): a comparator with either reads a candidate
ARM_REPRESENTATIONS = ('embedding', 'text_only')
```

Replace:

```python
    return tuple(names)
```

with:

```python
    return tuple(names)

def reference_comparators(regime: Regime, level: int) -> Tuple[str, ...]:
    '''
    The regime's comparators at a level that read no arm's tables: covariates, one-hot and
    ancestor indicators, each alone and with the covariates (Req 24's margin read).
    '''

    return tuple(
        name for name in comparators(regime, level)
        if not set(name.split('+')) & set(ARM_REPRESENTATIONS)
    )
```

Replace:

```python
def feature_matrix(comparator: str, frame: pl.DataFrame, level: int, arm: ArmTables) -> np.ndarray:
```

with:

```python
def feature_matrix(
    comparator: str, frame: pl.DataFrame, level: int, arm: Optional[ArmTables]
) -> np.ndarray:
```

Replace:

```python
    Indicator columns come from the codes in ``frame``, never from outcomes.
```

with:

```python
    Indicator columns come from the codes in ``frame``, never from outcomes.

    Raises:
        ValueError: If the comparator reads an arm's table and ``arm`` is None.
```

Replace:

```python
    for part in comparator.split('+'):
```

with:

```python
    for part in comparator.split('+'):
        if part in ARM_REPRESENTATIONS and arm is None:
            raise ValueError(
                f"the comparator '{comparator}' reads an arm's table, and no arm is read"
            )
```

Replace:

```python
        if regime is Regime.SEEN:
            plan = seen_validation_plan(frame, level, self.settings)
        else:
            plan = heldout_validation_plan(frame, level, self.settings)
        _require_codes(arm, frame)
        self._log_read(regime, VALIDATION, purpose, frame.height, logged)
        return self._predict(regime, VALIDATION, level, frame, plan, arm)
```

with:

```python
        plan = self._validation_plan(regime, level, frame)
        _require_codes(arm, frame)
        self._log_read(regime, VALIDATION, purpose, frame.height, logged)
        return self._predict(
            regime, VALIDATION, level, frame, plan, arm, comparators(regime, level)
        )

    def reference_validation(
        self,
        regime: Regime,
        level: int,
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> pl.DataFrame:
        '''
        Out-of-sample predictions for the remainder rows from ``reference_comparators`` alone,
        logging the read: Req 24's margin read, which reads no arm and computes neither
        ``embedding`` nor ``text_only``.

        It fits the rows, folds and penalties ``validation`` fits, so each of its comparators
        predicts exactly as in any arm's read. ``detail`` joins the read's logged detail, as in
        ``validation``.
        '''

        regime = Regime(regime)
        names = reference_comparators(regime, level)
        logged = merge_read_detail({'level': level, 'comparators': list(names)}, detail)
        self._require_defined(regime, level)
        frame = self._remainder(level)
        plan = self._validation_plan(regime, level, frame)
        self._log_read(regime, VALIDATION, purpose, frame.height, logged)
        return self._predict(regime, VALIDATION, level, frame, plan, None, names)
```

Replace:

```python
        return self._predict(regime, TEST, level, frame, plan, arm)
```

with:

```python
        return self._predict(regime, TEST, level, frame, plan, arm, comparators(regime, level))
```

Replace:

```python
        return self._frame(level).filter(pl.col('split') == RegressorSplit.REMAINDER.value)
```

with:

```python
        return self._frame(level).filter(pl.col('split') == RegressorSplit.REMAINDER.value)

    def _validation_plan(self, regime: Regime, level: int, frame: pl.DataFrame) -> Plan:
        if regime is Regime.SEEN:
            return seen_validation_plan(frame, level, self.settings)
        return heldout_validation_plan(frame, level, self.settings)
```

Replace:

```python
        plan: Sequence[FitTask],
        arm: ArmTables,
```

with:

```python
        plan: Sequence[FitTask],
        arm: Optional[ArmTables],
        names: Sequence[str],
```

Replace:

```python
        for comparator in comparators(regime, level):
```

with:

```python
        for comparator in names:
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_panel.py -q`
Expected: PASS, `100 passed`. Plan 13's level-mode pins pass unchanged.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/panels/regressor.py tests/unit/test_regressor_panel.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3143 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/panels/regressor.py tests/unit/test_regressor_panel.py
git commit -m "feat(panels): read the regressor panel's reference comparators alone (Req 24)" \
  -m "The margin read scores covariates, one-hot and ancestor indicators, alone and
with the covariates, on validation folds and penalties shared with every arm's
read, and logs no arm, text-only table or dimension."
```

Run:

```bash
git diff --stat 6d2eebd0813122f84dcced7e392f94d4aac829d6 HEAD -- src/naics_embedder/panels/regressor.py tests/unit/test_regressor_panel.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 5: The relative margins

Req 24's margin: "δ = 0.05 × the reference's validation MSE on that panel (U13) … fixed and
recorded from a reference-only read before any candidate is read." This task creates
`decision/economic.py` and the relative margin record (P5, P6, P12).

The tests cover:
- each δ, against the reference's MSE recomputed from the stored predictions, each row's squared
  error averaged over its repeats first;
- the margin read's log records, which hold only the reference comparators and precede
  `fixed_at`;
- the record's read, panel and environment, and its round trip through a file;
- the refusals, before anything is read, of a level panel, a non-positive r and a missing lock;
- `regressor_scores`, which scores growth reads only where the caller names their panels.

**Files:**
- Create: `src/naics_embedder/decision/economic.py`
- Modify: `src/naics_embedder/decision/records.py`, lines 15 and 278
- Modify: `src/naics_embedder/decision/scores.py`, lines 19, 105, 110-113, 118 and 123
- Create: `tests/unit/test_decision_economic.py`
- Test: `tests/unit/test_decision_scores.py`, line 149

**Interfaces:**
- Consumes: Task 4's `RegressorPanel.reference_validation` and `reference_comparators`; Stage 1's
  `READ_NAMES[RegressorOutcome.GROWTH]` and the fixture `regressor_growth_rows`.
- Produces:
  - in `naics_embedder.decision.records`:
    - `ReadEnvironment(platform: str, python: str, lock_sha256: str)`;
    - `GrowthPanelSet(regressor: str, regressor_data: str, fit_settings: Dict[str, Any])`;
    - `ReferenceRead(run_id, panels: GrowthPanelSet, predictions: ArtifactRef,
      scores: ArtifactRef, log_records, environment: ReadEnvironment)`;
    - `RelativeMargin(panel, statistic, reference_mse: float, margin: float)`;
    - `RelativeMarginRecord(kind='relative_margins', name, r: float, reference,
      read: ReferenceRead, margins: List[RelativeMargin], fixed_at)`, where `r` must be
      positive, with `margin(panel) -> float`;
  - in `naics_embedder.decision.scores`:
    `regressor_scores(predictions, repeats, panels: Sequence[str] = REGRESSOR_PANELS)`;
  - in `naics_embedder.decision.economic`:
    - `GROWTH_PANELS == ('growth_seen', 'growth_heldout')`;
    - `REFERENCE_COMPARATOR = 'covariates+ancestors'`;
    - `read_environment(lock) -> ReadEnvironment`;
    - `growth_panel_set(panel) -> GrowthPanelSet`;
    - `_require_growth(panel, reader) -> None`;
    - `fix_relative_margins(panel, r, name, store, *, purpose, lock) -> RelativeMarginRecord`.
      Its run id is `margins/<32 hex digits>`.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_decision_economic.py`:

```python
'''
The economic decision on the fixture growth panel (Req 24; roadmap Stage 2 Exit).

The margin read scores the reference, ``covariates+ancestors``, beside the other comparators that
read no candidate. Every candidate's read fits the same rows, folds and penalties, so the
reference's errors pair exactly with each candidate's.
'''

import hashlib
import platform
from datetime import datetime

import polars as pl
import pytest

from naics_embedder.decision.economic import (
    GROWTH_PANELS,
    REFERENCE_COMPARATOR,
    fix_relative_margins,
)
from naics_embedder.decision.records import RelativeMarginRecord, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import Regime, RegressorPanel, reference_comparators
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.regressor_panel import HELDOUT_GROUPS, SETTINGS

pytestmark = pytest.mark.unit

PURPOSE = 'fixture economic decision'
# U13
R = 0.05

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'selection_log.jsonl')

@pytest.fixture
def growth_panel(regressor_growth_rows, log):
    return RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def lock(tmp_path):
    path = tmp_path / 'uv.lock'
    path.write_text('a fixture lock\n')
    return path

@pytest.fixture
def margins(growth_panel, store, lock):
    return fix_relative_margins(
        growth_panel, R, 'fixture margins', store, purpose=PURPOSE, lock=lock
    )

# -------------------------------------------------------------------------------------------------
# The margin read and the relative margins (Verification "Margin first, reference only")
# -------------------------------------------------------------------------------------------------

def _reference_mse(predictions, panel):
    '''The reference's MSE on a panel, each row's squared error averaged over its repeats first.'''

    # yapf: disable
    rows = (
        predictions
        .filter((pl.col('panel') == panel) & (pl.col('comparator') == REFERENCE_COMPARATOR))
        .group_by('code', 'feature_year')
        .agg(error=((pl.col('prediction') - pl.col('outcome'))**2).mean())
    )
    # yapf: enable
    return rows.get_column('error').mean()

def test_each_margin_is_r_times_the_references_validation_mse(margins, store):
    predictions = store.read_frame(margins.read.predictions)

    assert (margins.r, margins.reference) == (R, 'covariates+ancestors')
    assert [entry.panel for entry in margins.margins] == list(GROWTH_PANELS)
    for entry in margins.margins:
        assert entry.statistic == REFERENCE_COMPARATOR
        assert entry.reference_mse == pytest.approx(
            _reference_mse(predictions, entry.panel), rel=1e-12
        )
        assert entry.margin == R * entry.reference_mse
        assert margins.margin(entry.panel) == entry.margin

def test_the_margin_read_logs_only_comparators_that_read_no_candidate(margins, store, log):
    records = log.records()
    predictions = store.read_frame(margins.read.predictions)

    assert margins.read.log_records == records
    assert [(record['event'], record['panel'], record['split']) for record in records] == [
        ('read', 'growth_seen', 'validation'),
        ('read', 'growth_heldout', 'validation'),
    ]
    for record, regime in zip(records, Regime):
        names = list(reference_comparators(regime, 6))
        assert record['detail'] == {'level': 6, 'comparators': names, 'run': margins.read.run_id}
        assert datetime.fromisoformat(record['time']) <= margins.fixed_at
        read = predictions.filter(pl.col('panel') == record['panel'])
        assert read.get_column('comparator').unique(maintain_order=True).to_list() == names

def test_the_margins_name_the_read_its_panel_and_where_it_ran(
    margins, growth_panel, store, lock, tmp_path
):
    read = margins.read

    assert read.panels.regressor == growth_panel.fingerprint
    assert read.panels.regressor_data == growth_panel.data_fingerprint(6)
    assert read.panels.fit_settings['repeats'] == SETTINGS.repeats
    assert read.environment.platform == platform.platform()
    assert read.environment.python == platform.python_version()
    assert read.environment.lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    store.resolve(read.scores)
    path = write_record(margins, tmp_path / 'margins.json')
    assert read_record(path, RelativeMarginRecord) == margins

def test_the_margins_read_growth_only(regressor_rows, log, store, lock):
    level = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    with pytest.raises(ValueError, match='read growth, not level'):
        fix_relative_margins(level, R, 'level margins', store, purpose=PURPOSE, lock=lock)
    assert log.records() == []

@pytest.mark.parametrize('r', [0.0, -0.05])
def test_r_must_be_positive(growth_panel, log, store, lock, r):
    with pytest.raises(ValueError, match='r must be positive'):
        fix_relative_margins(growth_panel, r, 'none', store, purpose=PURPOSE, lock=lock)
    assert log.records() == []

def test_the_margins_name_a_lock_that_exists(growth_panel, log, store, tmp_path):
    with pytest.raises(FileNotFoundError, match='no uv.lock'):
        fix_relative_margins(
            growth_panel, R, 'unlocked', store, purpose=PURPOSE, lock=tmp_path / 'missing'
        )
    assert log.records() == []
```

In `tests/unit/test_decision_scores.py`, make one edit.

Replace:

```python
@pytest.mark.parametrize(
```

with:

```python
def test_the_growth_panels_are_scored_only_where_the_caller_names_them():
    '''Req 24: the economic decision scores growth reads by D10's statistic; D8's never does.'''

    row = _prediction('growth_seen', 'covariates+ancestors', 0, '111111', 2023, 1.5)

    with pytest.raises(ValueError, match="rows of \\['regressor_seen', 'regressor_heldout'\\]"):
        regressor_scores(_predictions([row]), repeats=1)
    scores = regressor_scores(
        _predictions([row]), repeats=1, panels=('growth_seen', 'growth_heldout')
    )
    assert scores.select('panel', 'statistic', 'item', 'value').rows() == [
        ('growth_seen', 'covariates+ancestors', '111111/2023', 0.25)
    ]

@pytest.mark.parametrize(
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision_scores.py -q`
Expected: FAIL, `1 error`. Collecting `tests/unit/test_decision_economic.py` raises
`ModuleNotFoundError: No module named 'naics_embedder.decision.economic'`.

- [x] **Step 3: Write the implementation**

> Deviation: fix_relative_margins refuses r unless math.isfinite(r) and r > 0, by the owner's ruling (11641b0).

In `src/naics_embedder/decision/records.py`, make two edits, in this order.

Replace:

```python
  the chosen arm, and the reported statistics.
```

with:

```python
  the chosen arm, and the reported statistics.

The economic decision (Req 24) has kinds of its own:

- **Relative margin record.** Each growth panel's δ, r times the reference comparator's
  validation MSE, from a margin read that reads no candidate.
```

Replace:

```python
# Files
```

with:

```python
# The economic decision's margins (Req 24)
# -------------------------------------------------------------------------------------------------

class ReadEnvironment(_Record):
    '''
    Where a growth read ran. Decisions pair reads only on bit-identical rows and fits, so every
    candidate read of an economic decision runs where its margin read ran (Req 24).

    Attributes:
        platform: ``platform.platform()``.
        python: ``platform.python_version()``.
        lock_sha256: The sha256 of the ``uv.lock`` the read ran under.
    '''

    platform: str
    python: str
    lock_sha256: str

class GrowthPanelSet(_Record):
    '''
    The growth panel an economic read scored (Req 21), which every read of a decision shares.

    Attributes:
        regressor: The held-out draw (``RegressorPanel.fingerprint``).
        regressor_data: Its rows at the decision level (``RegressorPanel.data_fingerprint``).
        fit_settings: The panel's tuning and resampling settings.
    '''

    regressor: str
    regressor_data: str
    fit_settings: Dict[str, Any]

class ReferenceRead(_Record):
    '''
    The margin read (Req 24): each growth panel's validation split at the decision level, scored
    by the comparators that read no candidate.

    Attributes:
        run_id: Names the read in its selection-log records.
        panels: The growth panel it read.
        predictions: Both regimes' level-6 validation predictions.
        scores: Their per-row scores (``decision.scores.SCORE_COLUMNS``).
        log_records: The selection-log records of the read, one per growth panel.
        environment: Where the read ran.
    '''

    run_id: str = Field(min_length=1)
    panels: GrowthPanelSet
    predictions: ArtifactRef
    scores: ArtifactRef
    log_records: List[Dict[str, Any]]
    environment: ReadEnvironment

class RelativeMargin(_Record):
    '''One growth panel's δ: r times the reference comparator's validation MSE (Req 24).'''

    panel: str
    statistic: str
    reference_mse: float
    margin: float

class RelativeMarginRecord(_Record):
    '''
    Each growth panel's δ, fixed from the margin read before any candidate is read (Req 24).

    Attributes:
        r: Each δ as a multiple of the reference's MSE (U13).
        reference: The reference comparator, which reads no candidate (U9).
        read: The margin read the MSEs come from.
    '''

    kind: Literal['relative_margins'] = 'relative_margins'
    name: str
    r: float = Field(gt=0)
    reference: str
    read: ReferenceRead
    margins: List[RelativeMargin]
    fixed_at: datetime

    def margin(self, panel: str) -> float:
        '''
        The panel's δ.

        Raises:
            ValueError: If the record holds no δ for the panel.
        '''

        for entry in self.margins:
            if entry.panel == panel:
                return entry.margin
        raise ValueError(f'the margins {self.name!r} hold no δ for {panel}')

# -------------------------------------------------------------------------------------------------
# Files
```

In `src/naics_embedder/decision/scores.py`, make five edits, in this order.

Replace:

```python
from typing import Dict, Tuple
```

with:

```python
from typing import Dict, Sequence, Tuple
```

Replace:

```python
def regressor_scores(predictions: pl.DataFrame, repeats: int) -> pl.DataFrame:
```

with:

```python
def regressor_scores(
    predictions: pl.DataFrame,
    repeats: int,
    panels: Sequence[str] = REGRESSOR_PANELS,
) -> pl.DataFrame:
```

Replace:

```python
    Raises:
        ValueError: If a row is not a level-6 validation row of a regressor panel, or a row does not
            have exactly ``repeats`` predictions, one per repeat, under every comparator of its
            panel.
```

with:

```python
    Args:
        predictions: The reads' predictions (``panels.regressor.PREDICTION_COLUMNS``).
        repeats: The validation repeats each row was scored under.
        panels: The read names the predictions may carry: D8's regressor regimes, or the growth
            panels of the economic decision (Req 24).

    Raises:
        ValueError: If a row is not a level-6 validation row of ``panels``, or a row does not have
            exactly ``repeats`` predictions, one per repeat, under every comparator of its panel.
```

Replace:

```python
        | ~pl.col('panel').is_in(list(REGRESSOR_PANELS))
```

with:

```python
        | ~pl.col('panel').is_in(list(panels))
```

Replace:

```python
            f'{list(REGRESSOR_PANELS)}'
```

with:

```python
            f'{list(panels)}'
```

Create `src/naics_embedder/decision/economic.py`:

```python
'''
The economic artifact's decision (Req 24; Req 23; Req 18): the margin read, the candidates' reads
and the decision over the two growth panels.

- **Panels.** ``growth_seen`` and ``growth_heldout``, the growth outcome's regimes at level 6
  (Req 21), and no other: no artifact's decision reads another artifact's panel (Req 18).
- **Statistic.** D10's, applied to growth: each row's squared error averaged over its repeats
  (``decision.scores.regressor_scores``); a candidate's is ``covariates+embedding``'s MSE.
- **Reference.** The comparator ``covariates+ancestors`` (U9). It reads no candidate, and the
  folds depend on the regime, level, repeat and group alone, so the margin read's row errors are
  those every candidate's read computes, and they pair exactly.
- **Margins.** δ = r × the reference's validation MSE on each panel (U13), from a margin read that
  reads no candidate, fixed before any candidate is read.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

import polars as pl

from naics_embedder.decision.records import (
    GrowthPanelSet,
    ReadEnvironment,
    ReferenceRead,
    RelativeMargin,
    RelativeMarginRecord,
)
from naics_embedder.decision.scores import panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import _fit_settings, _log_records
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import DECISION_LEVEL, READ_NAMES, Regime, RegressorPanel
from naics_embedder.supervision.artifacts import sha256_file

# The growth outcome's two regimes: the economic decision's panels (Req 24)
GROWTH_PANELS = tuple(READ_NAMES[RegressorOutcome.GROWTH][regime] for regime in Regime)
# The hierarchy's ancestor indicators beside the covariates (U9)
REFERENCE_COMPARATOR = 'covariates+ancestors'

# -------------------------------------------------------------------------------------------------
# Reads
# -------------------------------------------------------------------------------------------------

def read_environment(lock: Union[str, Path]) -> ReadEnvironment:
    '''
    This process's platform and Python, and the sha256 of ``lock``, the ``uv.lock`` it runs under.

    Raises:
        FileNotFoundError: If ``lock`` is not a file.
    '''

    path = Path(lock)
    if not path.is_file():
        raise FileNotFoundError(f'{path}: no uv.lock to name the read by')
    return ReadEnvironment(
        platform=platform.platform(),
        python=platform.python_version(),
        lock_sha256=sha256_file(path),
    )

def growth_panel_set(panel: RegressorPanel) -> GrowthPanelSet:
    '''The growth panel's identity at the decision level, which every read of a decision shares.'''

    return GrowthPanelSet(
        regressor=panel.fingerprint,
        regressor_data=panel.data_fingerprint(DECISION_LEVEL),
        fit_settings=_fit_settings(panel),
    )

def _require_growth(panel: RegressorPanel, reader: str) -> None:
    '''
    Require the growth panel at the decision level: the economic decision's (Req 24; U8).

    Raises:
        ValueError: If the panel reads another outcome or lacks the decision level.
    '''

    if panel.outcome is not RegressorOutcome.GROWTH:
        raise ValueError(f'{reader} read growth, not {panel.outcome.value} (Req 24; U8)')
    if DECISION_LEVEL not in panel.levels:
        raise ValueError(f'{reader} read level {DECISION_LEVEL}, which the panel has not loaded')

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def fix_relative_margins(
    panel: RegressorPanel,
    r: float,
    name: str,
    store: ArtifactStore,
    *,
    purpose: str,
    lock: Union[str, Path],
) -> RelativeMarginRecord:
    '''
    Read the reference alone on each growth panel's validation split, and fix each panel's δ as
    ``r`` times the reference comparator's MSE there (Req 24).

    The read logs and scores only the comparators that read no candidate
    (``reference_comparators``), so no candidate's growth error exists before δ is recorded.

    Args:
        panel: The growth panel.
        r: Each δ as a multiple of the reference's MSE (U13).
        name: Names the margins in the decisions that use them.
        store: Where the read's predictions and scores go.
        purpose: Why the read happens, which the selection log records.
        lock: The ``uv.lock`` the read runs under, which the record names.

    Raises:
        ValueError: If ``r`` is not positive, or the panel is not the growth panel at the decision
            level, before anything is read.
        FileNotFoundError: If ``lock`` is not a file, before anything is read.
    '''

    if r <= 0:
        raise ValueError(f'the margin multiple r must be positive, got {r}')
    _require_growth(panel, 'the margins')
    environment = read_environment(lock)
    run_id = f'margins/{uuid.uuid4().hex}'
    predictions = pl.concat(
        [
            panel.reference_validation(regime, DECISION_LEVEL, purpose, detail={'run': run_id})
            for regime in Regime
        ]
    )
    scores = regressor_scores(predictions, panel.settings.repeats, GROWTH_PANELS)
    read = ReferenceRead(
        run_id=run_id,
        panels=growth_panel_set(panel),
        predictions=store.put_frame(predictions, 'predictions.parquet'),
        scores=store.put_frame(scores, 'scores.parquet'),
        log_records=_log_records([panel.log], run_id),
        environment=environment,
    )
    margins: List[RelativeMargin] = []
    for growth in GROWTH_PANELS:
        mse = panel_statistic(scores, growth, REFERENCE_COMPARATOR)
        margins.append(
            RelativeMargin(
                panel=growth, statistic=REFERENCE_COMPARATOR, reference_mse=mse, margin=r * mse
            )
        )
    return RelativeMarginRecord(
        name=name,
        r=r,
        reference=REFERENCE_COMPARATOR,
        read=read,
        margins=margins,
        fixed_at=datetime.now(timezone.utc),
    )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision_scores.py -q`
Expected: PASS, `17 passed`.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/scores.py tests/unit/test_decision_economic.py tests/unit/test_decision_scores.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3151 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/scores.py tests/unit/test_decision_economic.py tests/unit/test_decision_scores.py
git commit -m "feat(decision): fix the economic margins from a reference-only growth read (Req 24)" \
  -m "Each growth panel's delta is r times covariates+ancestors' validation MSE, read
by the comparators that read no candidate; the record names r, the reference,
the read, its panel and the platform and lock it ran under."
```

Run:

```bash
git diff --stat fd93b81ab61ea034b47a343e83ec35123affebfd HEAD -- src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/scores.py tests/unit/test_decision_economic.py tests/unit/test_decision_scores.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 6: A deterministic candidate

Req 24's replicates and tooling: a deterministic candidate "is a single replicate", and arm
records accept one "recorded with its provenance and without checkpoint or monitor reads". Req 23
says what a candidate is: a table for all 2,125 codes (P6–P9).

The tests cover:
- a candidate read once on each growth panel, as one run with no seed, whose reads name its
  table, its text-only table and its dimension;
- the provenance kept verbatim, and the record's round trip through a file;
- a record that validates and pairs with the reference (Verification "Deterministic arms");
- the refusals before any read:
  - a panel other than the margins';
  - a missing provenance, or one that describes another file;
  - a text-only table built from other inputs;
  - a table of another dimension;
- the refusals of a record:
  - an imported run naming a checkpoint, training run or monitor records, which does not load;
  - a seed, or a second run;
  - a read that names another table;
  - a provenance edited after it was stored;
- reference predictions 1e-12 off the margin read's, which do not pair, whatever the
  environments say.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 122 and 183-190
- Modify: `src/naics_embedder/decision/economic.py`, lines 13, 28, 35-39, 86-88 and 158-160
- Modify: `src/naics_embedder/decision/records.py`, lines 20 and 376
- Modify: `src/naics_embedder/decision/store.py`, lines 21-25, 31-32, 50-59 and 185
- Create: `tests/fixtures/economic.py`
- Test: `tests/unit/test_decision_economic.py`, lines 10, 14, 19-26, 124 and 138-140

**Interfaces:**
- Consumes: Task 3's `_same`; Task 5's records and `decision.economic` names; `check_text_only`
  (`decision/decide.py`); `ArmTables.from_tables` and `RegressorPanel.require_arm`
  (`panels/regressor.py`); `provenance_path` (`panels/text_only.py`); `write_text_only`,
  `spec` (`tests/fixtures/decision.py`); `CODEBOOK` and `coordinate_table`
  (`tests/fixtures/regressor_panel.py`).
- Produces:
  - in `naics_embedder.decision.records`:
    - `TableProvenance(file: ArtifactRef, content: Dict[str, Any])`;
    - `EconomicRun(seed: Optional[StrictInt] = None, run_id, table: TableRef,
      provenance: TableProvenance, scores, predictions, statistics: Dict[str, float],
      log_records)`;
    - `EconomicArmRecord(kind='economic_arm', spec: ArmSpec, text_only: TextOnlyRef,
      store: str, panels: GrowthPanelSet, environment: ReadEnvironment,
      runs: List[EconomicRun], created_at)`;
  - in `naics_embedder.decision.store`:
    - `check_provenance(provenance, table_sha256, matrix_fingerprint, name) -> None`;
    - `ArtifactStore.put_candidate(table_path) -> Tuple[TableRef, TableProvenance]`;
    - `ArtifactStore.candidate_provenance(table: TableRef, provenance: ArtifactRef) ->
      TableProvenance`;
  - in `naics_embedder.decision.decide`:
    `check_recorded_text_only(spec, text_only, store) -> None`;
  - in `naics_embedder.decision.economic`:
    - `read_deterministic_arm(spec, table, *, regressor_panel, text_only_table, margins,
      store, purpose, lock) -> EconomicArmRecord`. Its run id is `<name>/<32 hex digits>`,
      and each read's detail adds `{'run': run_id, 'arm_name': name}`;
    - `check_economic_arm(arm, store) -> None`;
    - `check_reference_pairing(margins, arm, store) -> None`;
  - in `tests/fixtures/economic.py`: `DIMENSION = 3`,
    `candidate_spec(name, **overrides) -> ArmSpec` (one component, Euclidean, dimension 3) and
    `write_candidate(directory, name, table, **entries) -> Path`.

- [x] **Step 1: Write the failing tests**

Create `tests/fixtures/economic.py`:

```python
'''
Deterministic economic candidates on the fixture growth panel (Req 23; Req 24).

A candidate is a coordinate table over the fixture codebook in the export form, with the
provenance its builder writes beside it. The candidates read the synthetic backbone and text of
``tests.fixtures.decision``, so the text-only table ``write_text_only`` writes over the codebook
is their D9 comparator.
'''

import json
from pathlib import Path

import polars as pl

from naics_embedder.decision.records import ArmSpec
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import spec

DIMENSION = 3

def candidate_spec(name: str, **overrides) -> ArmSpec:
    '''A deterministic candidate's spec: one component, flat, at the fixture dimension.'''

    fields = {'components': 1, 'dimension': DIMENSION, 'geometry': 'euclidean', **overrides}
    return spec(name, **fields)

def write_candidate(directory: Path, name: str, table: pl.DataFrame, **entries) -> Path:
    '''
    A candidate's table and the provenance its builder writes beside it, naming the table's
    sha256 and ``matrix_fingerprint``; ``entries`` join or replace the provenance's entries.
    '''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}.parquet'
    table.write_parquet(path)
    provenance = {
        'builder': 'fixture',
        'inputs': {
            'codebook': 'the fixture codebook'
        },
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': table_fingerprint(table),
        **entries,
    }
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path
```

In `tests/unit/test_decision_economic.py`, make five edits, in this order.

Replace:

```python
import platform
```

with:

```python
import json
import platform
from dataclasses import replace
```

Replace:

```python
import pytest
```

with:

```python
import pytest
from pydantic import ValidationError
```

Replace:

```python
    fix_relative_margins,
)
from naics_embedder.decision.records import RelativeMarginRecord, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import Regime, RegressorPanel, reference_comparators
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.regressor_panel import HELDOUT_GROUPS, SETTINGS
```

with:

```python
    check_economic_arm,
    check_reference_pairing,
    fix_relative_margins,
    read_deterministic_arm,
)
from naics_embedder.decision.records import (
    EconomicArmRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.scores import panel_statistic
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
    Regime,
    RegressorPanel,
    comparators,
    reference_comparators,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from tests.fixtures.decision import write_text_only
from tests.fixtures.economic import candidate_spec, write_candidate
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, coordinate_table
```

Replace:

```python
    with pytest.raises(ValueError, match='read growth, not level'):
```

with:

```python
    with pytest.raises(
        ValueError, match='^the margins: the economic decision reads growth, not level'
    ):
```

Replace:

```python
        )
    assert log.records() == []

```

with:

```python
        )
    assert log.records() == []

# -------------------------------------------------------------------------------------------------
# A deterministic candidate (Verification "Deterministic arms")
# -------------------------------------------------------------------------------------------------

TREE = candidate_spec('tree')

@pytest.fixture
def text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

@pytest.fixture
def candidate(tmp_path):
    return write_candidate(tmp_path / 'candidates', 'tree', coordinate_table(CODEBOOK))

def _read(table, panel, text_only, margins, store, lock, arm_spec=TREE):
    return read_deterministic_arm(
        arm_spec,
        table,
        regressor_panel=panel,
        text_only_table=text_only,
        margins=margins,
        store=store,
        purpose=PURPOSE,
        lock=lock,
    )

@pytest.fixture
def arm(candidate, growth_panel, text_only, margins, store, lock):
    return _read(candidate, growth_panel, text_only, margins, store, lock)

def test_a_deterministic_candidate_is_one_replicate_read_once_on_each_growth_panel(
    arm, margins, store, log
):
    '''Req 24: a single replicate, with no seed, checkpoint or monitor reads.'''

    (run, ) = arm.runs
    reads = log.records()[len(margins.read.log_records):]

    assert run.seed is None
    assert run.log_records == reads
    assert [(read['panel'], read['split']) for read in reads] == [
        ('growth_seen', 'validation'),
        ('growth_heldout', 'validation'),
    ]
    for read, regime in zip(reads, Regime):
        assert read['detail'] == {
            'level': 6,
            'comparators': list(comparators(regime, 6)),
            'arm': run.table.matrix_fingerprint,
            'text_only': arm.text_only.table.matrix_fingerprint,
            'dimension': 3,
            'run': run.run_id,
            'arm_name': 'tree',
        }
    scores = store.read_frame(run.scores)
    assert run.statistics == {
        panel: panel_statistic(scores, panel, 'covariates+embedding')
        for panel in GROWTH_PANELS
    }
    assert (arm.panels, arm.environment) == (margins.read.panels, margins.read.environment)

def test_the_record_keeps_the_tables_provenance_verbatim(arm, candidate, store, tmp_path):
    '''Req 23: the record shows what the candidate was built from, as its builder wrote it.'''

    (run, ) = arm.runs
    written = provenance_path(candidate)

    assert run.provenance.content == json.loads(written.read_text())
    assert store.resolve(run.provenance.file).read_bytes() == written.read_bytes()
    path = write_record(arm, tmp_path / 'arm.json')
    assert read_record(path, EconomicArmRecord) == arm

def test_a_single_replicate_record_validates_and_pairs_with_the_reference(arm, margins, store):
    '''Verification "Deterministic arms": the reference predicts as in the margin read.'''

    check_economic_arm(arm, store)
    check_reference_pairing(margins, arm, store)

    def reference(predictions):
        return store.read_frame(predictions).filter(pl.col('comparator') == REFERENCE_COMPARATOR)

    assert reference(arm.runs[0].predictions).equals(reference(margins.read.predictions))

def test_a_candidate_is_read_on_the_growth_panel_the_margins_read(
    candidate, regressor_rows, regressor_growth_rows, log, text_only, margins, store, lock
):
    level = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    refit = RegressorPanel(
        regressor_growth_rows,
        HELDOUT_GROUPS,
        log,
        replace(SETTINGS, alphas=(0.1, 1.0)),
        outcome=RegressorOutcome.GROWTH,
    )

    with pytest.raises(ValueError, match='^tree: the economic decision reads growth, not level'):
        _read(candidate, level, text_only, margins, store, lock)
    with pytest.raises(
        ValueError,
        match="^tree read other panels or fit settings than the margins 'fixture margins', "
        "differing in \\['fit_settings'\\]",
    ):
        _read(candidate, refit, text_only, margins, store, lock)
    assert log.records() == margins.read.log_records

@pytest.mark.parametrize(
    'edit, error, message',
    [
        (lambda table: provenance_path(table).unlink(), FileNotFoundError, 'has no provenance'),
        (
            lambda table: coordinate_table(CODEBOOK, seed=8).write_parquet(table),
            ValueError,
            'describes another file than the table',
        ),
    ],
    ids=['missing', 'another-file'],
)
def test_a_candidate_is_read_only_with_provenance_that_describes_it(
    candidate, growth_panel, text_only, margins, store, lock, log, edit, error, message
):
    edit(candidate)

    with pytest.raises(error, match=message):
        _read(candidate, growth_panel, text_only, margins, store, lock)
    assert log.records() == margins.read.log_records

def test_a_candidate_reads_its_own_text_only_table_at_its_dimension(
    candidate, growth_panel, text_only, margins, store, lock, log, tmp_path
):
    other = write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK)
    wide = candidate_spec('tree', dimension=4)

    with pytest.raises(ValueError, match='^tree: the text-only table was built from .*D9'):
        _read(candidate, growth_panel, other, margins, store, lock)
    with pytest.raises(ValueError, match="^tree: the table has dimension 3, not the spec's 4"):
        _read(candidate, growth_panel, text_only, margins, store, lock, arm_spec=wide)
    assert log.records() == margins.read.log_records

@pytest.mark.parametrize(
    'field, value',
    [
        ('checkpoint', {
            'path': 'objects/a',
            'sha256': 'a' * 64,
            'bytes': 1
        }),
        ('training_run', 'tree-training-0'),
        ('monitor_records', []),
    ],
)
def test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load(
    arm, tmp_path, field, value
):
    '''Req 24: a deterministic candidate is recorded without checkpoint or monitor reads.'''

    data = json.loads(arm.model_dump_json())
    data['runs'][0][field] = value
    path = tmp_path / 'imported.json'
    path.write_text(json.dumps(data))

    with pytest.raises(ValidationError, match=f'runs\\.0\\.{field}\\n  Extra inputs'):
        read_record(path, EconomicArmRecord)

def test_an_economic_arm_is_one_replicate_with_no_seed(arm, store):
    (run, ) = arm.runs
    seeded = arm.model_copy(update={'runs': [run.model_copy(update={'seed': 0})]})
    twice = arm.model_copy(update={'runs': [run, run]})

    for record in (seeded, twice):
        with pytest.raises(ValueError, match='^tree: an economic arm is one deterministic'):
            check_economic_arm(record, store)

def test_a_run_whose_reads_name_another_table_is_refused(arm, store):
    (run, ) = arm.runs
    reads = json.loads(json.dumps(run.log_records))
    reads[0]['detail']['arm'] = 'another table'
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'log_records': reads})]})

    with pytest.raises(ValueError, match="^tree: a growth_seen read names another \\['arm'\\]"):
        check_economic_arm(edited, store)

def test_a_provenance_edited_after_it_was_stored_is_refused(arm, store):
    (run, ) = arm.runs
    content = {**run.provenance.content, 'inputs': {'qcew': 'national slices'}}
    provenance = run.provenance.model_copy(update={'content': content})
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'provenance': provenance})]})

    with pytest.raises(ValueError, match="^tree: the record's provenance differs from its stored"):
        check_economic_arm(edited, store)

def test_reference_predictions_that_differ_from_the_margin_reads_do_not_pair(arm, margins, store):
    '''Req 24: only the predictions decide; the environments name where each read ran.'''

    (run, ) = arm.runs
    moved = arm.environment.model_copy(update={'platform': 'another platform'})
    reference = pl.col('comparator') == REFERENCE_COMPARATOR
    predictions = store.read_frame(run.predictions).with_columns(
        prediction=pl.when(reference).then(pl.col('prediction') + 1e-12).otherwise('prediction')
    )
    refit = run.model_copy(
        update={'predictions': store.put_frame(predictions, 'predictions.parquet')}
    )

    check_reference_pairing(margins, arm.model_copy(update={'environment': moved}), store)
    with pytest.raises(
        ValueError,
        match='^tree: its covariates\\+ancestors predictions differ .* another platform'
    ):
        check_reference_pairing(
            margins, arm.model_copy(update={
                'environment': moved,
                'runs': [refit]
            }), store
        )

```

- [x] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision.py tests/unit/test_decision_store.py -q
```
Expected: FAIL, `1 error`. Collecting `tests/unit/test_decision_economic.py` raises
`ImportError: cannot import name 'check_economic_arm' from 'naics_embedder.decision.economic'`.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/decision/records.py`, make two edits, in this order.

Replace:

```python
  validation MSE, from a margin read that reads no candidate.
```

with:

```python
  validation MSE, from a margin read that reads no candidate.
- **Economic arm record.** A deterministic candidate's single replicate: its table and that
  table's provenance (Req 23), read once on each growth panel, with no checkpoint or monitor
  reads.
```

Replace:

```python
# Files
```

with:

```python
# The economic decision's arms (Req 23; Req 24)
# -------------------------------------------------------------------------------------------------

class TableProvenance(_Record):
    '''
    A candidate table's provenance, as its builder wrote it beside the table (Req 23).

    Attributes:
        file: The stored provenance file.
        content: Its JSON object, verbatim: what the table was built from.
    '''

    file: ArtifactRef
    content: Dict[str, Any]

class EconomicRun(_Record):
    '''
    One replicate of an economic candidate: its table, read once on each growth panel.

    A deterministic candidate is a single replicate (Req 24), so a run has no seed, and no field
    for a checkpoint or monitor reads: an imported run that names one does not load.

    Attributes:
        seed: None: a deterministic candidate has no seed.
        run_id: Names the run in its selection-log records.
        table: The candidate's 2,125-code table, in the export form.
        provenance: The table's provenance.
        scores: Both growth panels' per-row scores (``decision.scores.SCORE_COLUMNS``).
        predictions: Both growth panels' level-6 validation predictions.
        statistics: Each growth panel's decision statistic.
        log_records: The selection-log records of the run's reads.
    '''

    seed: Optional[StrictInt] = None
    run_id: str = Field(min_length=1)
    table: TableRef
    provenance: TableProvenance
    scores: ArtifactRef
    predictions: ArtifactRef
    statistics: Dict[str, float]
    log_records: List[Dict[str, Any]]

class EconomicArmRecord(_Record):
    '''
    An economic candidate's reads of the growth panels (``decision.economic``).

    Attributes:
        spec: The candidate, with what the tie order and D9 read.
        text_only: The text-only table its reads scored as ``covariates+text_only`` (D9).
        store: The artifact store's root.
        panels: The growth panel it read.
        environment: Where it was read.
        runs: Its replicates: one, for a deterministic candidate (Req 24).
    '''

    kind: Literal['economic_arm'] = 'economic_arm'
    spec: ArmSpec
    text_only: TextOnlyRef
    store: str
    panels: GrowthPanelSet
    environment: ReadEnvironment
    runs: List[EconomicRun]
    created_at: datetime

# -------------------------------------------------------------------------------------------------
# Files
```

In `src/naics_embedder/decision/store.py`, make four edits, in this order.

Replace:

```python
from typing import Any, Dict, Mapping, Union

import polars as pl

from naics_embedder.decision.records import ArtifactRef, TableRef, TextOnlyRef
```

with:

```python
from typing import Any, Dict, Mapping, Tuple, Union

import polars as pl

from naics_embedder.decision.records import ArtifactRef, TableProvenance, TableRef, TextOnlyRef
```

Replace:

```python
# Table provenance (text-only and export)
# -------------------------------------------------------------------------------------------------
```

with:

```python
# Table provenance (text-only and export)
# -------------------------------------------------------------------------------------------------

def check_provenance(
    provenance: Any, table_sha256: str, matrix_fingerprint: str, name: str
) -> None:
    '''
    Require a table's provenance to describe the table.

    Args:
        provenance: The provenance's JSON content.
        table_sha256: The table's sha256, which the provenance must name.
        matrix_fingerprint: The table's ``matrix_fingerprint``, which the provenance must name if
            it names one.
        name: Names the provenance in an error.

    Raises:
        ValueError: If the provenance is not a JSON object, describes another file or names
            another ``matrix_fingerprint``.
    '''

    if not isinstance(provenance, Mapping):
        raise ValueError(f'{name} is not a JSON object')
    if provenance.get('table_sha256') != table_sha256:
        raise ValueError(f'{name} describes another file than the table')
    if provenance.get('matrix_fingerprint', matrix_fingerprint) != matrix_fingerprint:
        raise ValueError(f'{name} names another matrix_fingerprint')
```

Replace:

```python
        ValueError: If the provenance is not a JSON object, describes another file, names
            another ``matrix_fingerprint``, or lacks a field D9 reads.
    '''

    if not isinstance(provenance, Mapping):
        raise ValueError(f'{name} is not a JSON object')
    if provenance.get('table_sha256') != table_sha256:
        raise ValueError(f'{name} describes another file than the table')
    if provenance.get('matrix_fingerprint', matrix_fingerprint) != matrix_fingerprint:
        raise ValueError(f'{name} names another matrix_fingerprint')
```

with:

```python
        ValueError: As ``check_provenance``, or if the provenance lacks a field D9 reads.
    '''

    check_provenance(provenance, table_sha256, matrix_fingerprint, name)
```

Replace:

```python
    def resolve(self, reference: ArtifactRef) -> Path:
```

with:

```python
    def put_candidate(self, table_path: Union[str, Path]) -> Tuple[TableRef, TableProvenance]:
        '''
        Store an economic candidate's code table (the export form) and the provenance beside it,
        once the provenance is checked to describe the table (Req 23).

        Raises:
            FileNotFoundError: If the provenance is missing.
            ValueError: As ``check_provenance``, before anything is stored, or if either file
                changes while it is stored.
        '''

        table_path = Path(table_path)
        provenance_file = provenance_path(table_path)
        if not provenance_file.is_file():
            raise FileNotFoundError(f'{provenance_file}: the table has no provenance beside it')
        check_provenance(
            json.loads(provenance_file.read_text(encoding='utf-8')),
            sha256_file(table_path),
            table_fingerprint(pl.read_parquet(table_path)),
            str(provenance_file),
        )
        table = self.put_table(table_path)
        return table, self.candidate_provenance(table, self.put(provenance_file))

    def candidate_provenance(self, table: TableRef, provenance: ArtifactRef) -> TableProvenance:
        '''
        A stored candidate table's provenance, read from the store.

        Raises:
            FileNotFoundError: If either file is not in the store.
            ValueError: If either file changed since it was stored, or as ``check_provenance``.
        '''

        self.resolve(table)
        content = json.loads(self.resolve(provenance).read_text(encoding='utf-8'))
        check_provenance(
            content, table.sha256, table.matrix_fingerprint, f'the stored {provenance.path}'
        )
        return TableProvenance(file=provenance, content=content)

    def resolve(self, reference: ArtifactRef) -> Path:
```

In `src/naics_embedder/decision/decide.py`, make two edits, in this order.

Replace:

```python
def check_seed_table(spec: ArmSpec, seed: int, fields: Mapping[str, Any]) -> None:
```

with:

```python
def check_recorded_text_only(spec: ArmSpec, text_only: TextOnlyRef, store: ArtifactStore) -> None:
    '''
    Require a record's text-only fields to be those of its stored provenance, and that
    provenance to be the arm's (D9).

    Raises:
        ValueError: If a recorded field differs from the stored provenance, or as
            ``check_text_only``.
    '''

    stored = store.text_only(text_only.table, text_only.provenance)
    recorded = text_only.model_dump()
    wrong = sorted(key for key, value in stored.model_dump().items() if value != recorded[key])
    if wrong:
        raise ValueError(
            f"{spec.name}: the record's text-only fields {wrong} differ from its stored "
            'provenance'
        )
    check_text_only(spec, stored)

def check_seed_table(spec: ArmSpec, seed: int, fields: Mapping[str, Any]) -> None:
```

Replace:

```python
    stored = store.text_only(arm.text_only.table, arm.text_only.provenance)
    recorded = arm.text_only.model_dump()
    wrong = sorted(key for key, value in stored.model_dump().items() if value != recorded[key])
    if wrong:
        raise ValueError(
            f"{name}: the record's text-only fields {wrong} differ from its stored provenance"
        )
    check_text_only(arm.spec, stored)
```

with:

```python
    check_recorded_text_only(arm.spec, arm.text_only, store)
```

In `src/naics_embedder/decision/economic.py`, make five edits, in this order.

Replace:

```python
  reads no candidate, fixed before any candidate is read.
```

with:

```python
  reads no candidate, fixed before any candidate is read.
- **Candidates.** A deterministic candidate, such as the text-only table or a tree embedding, is
  one replicate: its table, with the provenance its builder wrote, read once on each growth panel
  the margins read, with no checkpoint or monitor reads (Req 24; Req 23). Its reads pair with
  the margin read only where the reference predicts bit for bit as it did there.
```

Replace:

```python
from naics_embedder.decision.records import (
```

with:

```python
from naics_embedder.decision.decide import _same, check_recorded_text_only, check_text_only
from naics_embedder.decision.records import (
    ArmSpec,
    EconomicArmRecord,
    EconomicRun,
```

Replace:

```python
from naics_embedder.decision.scores import panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import _fit_settings, _log_records
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import DECISION_LEVEL, READ_NAMES, Regime, RegressorPanel
```

with:

```python
from naics_embedder.decision.scores import EMBEDDING_COMPARATOR, panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import _fit_settings, _log_records
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    READ_NAMES,
    VALIDATION,
    ArmTables,
    Regime,
    RegressorPanel,
)
```

Replace:

```python
        raise ValueError(f'{reader} read growth, not {panel.outcome.value} (Req 24; U8)')
    if DECISION_LEVEL not in panel.levels:
        raise ValueError(f'{reader} read level {DECISION_LEVEL}, which the panel has not loaded')
```

with:

```python
        raise ValueError(
            f'{reader}: the economic decision reads growth, not {panel.outcome.value} '
            '(Req 24; U8)'
        )
    if DECISION_LEVEL not in panel.levels:
        raise ValueError(
            f'{reader}: the economic decision reads level {DECISION_LEVEL}, which the panel has '
            'not loaded'
        )

def _require_same_panels(name: str, panels: GrowthPanelSet, margins: RelativeMarginRecord) -> None:
    '''
    Require a read of the growth panel the margins read, with its data and fit settings.

    Raises:
        ValueError: Naming what differs.
    '''

    expected = margins.read.panels.model_dump()
    wrong = sorted(key for key, value in panels.model_dump().items() if value != expected[key])
    if wrong:
        raise ValueError(
            f'{name} read other panels or fit settings than the margins {margins.name!r}, '
            f'differing in {wrong}: paired reads must score the same rows (Req 24)'
        )
```

Replace:

```python
        fixed_at=datetime.now(timezone.utc),
    )

```

with:

```python
        fixed_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Candidates
# -------------------------------------------------------------------------------------------------

def read_deterministic_arm(
    spec: ArmSpec,
    table: Union[str, Path],
    *,
    regressor_panel: RegressorPanel,
    text_only_table: Union[str, Path],
    margins: RelativeMarginRecord,
    store: ArtifactStore,
    purpose: str,
    lock: Union[str, Path],
) -> EconomicArmRecord:
    '''
    Read a deterministic candidate once on each growth panel's validation split: a single
    replicate (Req 24), recorded with its table's provenance (Req 23).

    Every check runs before the first read.

    Args:
        spec: The candidate, with what the tie order and D9 read.
        table: Its 2,125-code table in the export form, with its builder's provenance beside it
            (``panels.text_only.provenance_path``).
        regressor_panel: The growth panel the margins read.
        text_only_table: The text-only table its reads score as ``covariates+text_only`` (D9).
        margins: The margins the candidate will be judged against.
        store: Where the tables, the provenance and the read's outputs go.
        purpose: Why the read happens, which the selection log records.
        lock: The ``uv.lock`` the read runs under, which the record names.

    Raises:
        ValueError: If the panel is not the growth panel the margins read, with their data and
            fit settings; if the provenance does not describe the table; if the text-only table
            was not built from what the candidate reads (D9); or if the table's width is not the
            spec's dimension.
        FileNotFoundError: If the table has no provenance beside it, or ``lock`` is not a file.
    '''

    _require_growth(regressor_panel, spec.name)
    panels = growth_panel_set(regressor_panel)
    _require_same_panels(spec.name, panels, margins)
    environment = read_environment(lock)
    stored, provenance = store.put_candidate(table)
    text_only = store.put_text_only(text_only_table)
    check_text_only(spec, text_only)
    arm = ArmTables.from_tables(
        pl.read_parquet(store.resolve(stored)), pl.read_parquet(store.resolve(text_only.table))
    )
    if arm.dimension != spec.dimension:
        raise ValueError(
            f"{spec.name}: the table has dimension {arm.dimension}, not the spec's "
            f'{spec.dimension}'
        )
    regressor_panel.require_arm(arm)

    run_id = f'{spec.name}/{uuid.uuid4().hex}'
    detail = {'run': run_id, 'arm_name': spec.name}
    predictions = pl.concat(
        [
            regressor_panel.validation(regime, DECISION_LEVEL, arm, purpose, detail=detail)
            for regime in Regime
        ]
    )
    scores = regressor_scores(predictions, regressor_panel.settings.repeats, GROWTH_PANELS)
    run = EconomicRun(
        run_id=run_id,
        table=stored,
        provenance=provenance,
        scores=store.put_frame(scores, 'scores.parquet'),
        predictions=store.put_frame(predictions, 'predictions.parquet'),
        statistics={
            panel: panel_statistic(scores, panel, EMBEDDING_COMPARATOR)
            for panel in GROWTH_PANELS
        },
        log_records=_log_records([regressor_panel.log], run_id),
    )
    return EconomicArmRecord(
        spec=spec,
        text_only=text_only,
        store=str(store.root),
        panels=panels,
        environment=environment,
        runs=[run],
        created_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def check_economic_arm(arm: EconomicArmRecord, store: ArtifactStore) -> None:
    '''
    Require an economic arm record to be a deterministic candidate's single replicate, intact,
    with its provenance, read once on each growth panel as the decision reads it (Req 24; Req 23).

    Raises:
        ValueError: On the first problem found.
    '''

    name = arm.spec.name
    if len(arm.runs) != 1 or arm.runs[0].seed is not None:
        raise ValueError(
            f'{name}: an economic arm is one deterministic replicate with no seed (Req 24), not '
            f'{len(arm.runs)} runs with seeds {[run.seed for run in arm.runs]}'
        )
    check_recorded_text_only(arm.spec, arm.text_only, store)
    (run, ) = arm.runs
    for reference in (run.table, run.scores, run.predictions):
        store.resolve(reference)
    if store.candidate_provenance(run.table, run.provenance.file) != run.provenance:
        raise ValueError(f"{name}: the record's provenance differs from its stored provenance")
    _check_growth_reads(arm, run)

def _check_growth_reads(arm: EconomicArmRecord, run: EconomicRun) -> None:
    '''Require the run's reads to be its table's validation reads, once on each growth panel.'''

    name = arm.spec.name
    panels = sorted(str(record.get('panel')) for record in run.log_records)
    if panels != sorted(GROWTH_PANELS):
        raise ValueError(f'{name}: the run read {panels}, not each of {sorted(GROWTH_PANELS)} once')
    named = {
        'fingerprint': arm.panels.regressor,
        'run': run.run_id,
        'level': DECISION_LEVEL,
        'arm': run.table.matrix_fingerprint,
        'text_only': arm.text_only.table.matrix_fingerprint,
        'dimension': arm.spec.dimension,
    }
    for record in run.log_records:
        if record.get('event') != 'read' or record.get('split') != VALIDATION:
            raise ValueError(
                f'{name}: a {record.get("event")} of the {record.get("split")} split; a decision '
                'reads validation splits only (Req 4)'
            )
        detail = record.get('detail') or {}
        logged = {key: detail.get(key) for key in named}
        logged['fingerprint'] = record.get('fingerprint')
        wrong = sorted(key for key in named if not _same(logged[key], named[key]))
        if wrong:
            raise ValueError(f'{name}: a {record["panel"]} read names another {wrong}')

def _reference_rows(predictions: pl.DataFrame) -> pl.DataFrame:
    '''The reference's predictions, one per panel, repeat and row.'''

    return predictions.filter(pl.col('comparator') == REFERENCE_COMPARATOR
                              ).sort('panel', 'repeat', 'code', 'feature_year')

def _where(environment: ReadEnvironment) -> str:
    return (
        f'{environment.platform}, Python {environment.python}, uv.lock '
        f'{environment.lock_sha256[:12]}'
    )

def check_reference_pairing(
    margins: RelativeMarginRecord, arm: EconomicArmRecord, store: ArtifactStore
) -> None:
    '''
    Require the arm to have read the growth panel the margins read, and the reference to predict
    in each of its runs exactly as in the margin read, so its errors pair with δ's (Req 24).

    Only the predictions decide; the environments name where each read ran.

    Raises:
        ValueError: If the panels, data or fit settings differ, or the reference's predictions
            in a run differ from the margin read's.
    '''

    name = arm.spec.name
    _require_same_panels(name, arm.panels, margins)
    reference = _reference_rows(store.read_frame(margins.read.predictions))
    for run in arm.runs:
        if not _reference_rows(store.read_frame(run.predictions)).equals(reference):
            raise ValueError(
                f"{name}: its {REFERENCE_COMPARATOR} predictions differ from the margin read's, "
                f'so its errors do not pair with the reference (Req 24). It was read on '
                f'{_where(arm.environment)}; the margins on {_where(margins.read.environment)}'
            )

```

- [x] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision.py tests/unit/test_decision_store.py -q
```
Expected: PASS, `131 passed`.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/store.py tests/fixtures/economic.py tests/unit/test_decision_economic.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3165 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

> Deviation: the implementer's git add was refused by the auto-mode classifier, so the controller committed its verified tree as ebba560.

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/store.py tests/fixtures/economic.py tests/unit/test_decision_economic.py
git commit -m "feat(decision): read a deterministic economic candidate as one replicate (Req 24; Req 23)" \
  -m "An economic arm record holds a deterministic candidate's single
replicate: its table with the provenance its builder wrote, read once
on each growth panel the margins read, with no field for a checkpoint
or monitor reads. check_economic_arm validates a record, and
check_reference_pairing requires the reference to predict bit for bit
as in the margin read; the environments only name where each ran."
```

Run:

```bash
git diff --stat 5876ddfb70ff202d4997ab2b6c16a1c003c7cf93 HEAD -- src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py src/naics_embedder/decision/store.py tests/fixtures/economic.py tests/unit/test_decision_economic.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 7: The economic decision

Req 24's rule for the economic artifact: two growth panels at 95 % and 97.5 %, the reference
`covariates+ancestors`, eligibility before the rule, D11's held-out gain over the reference, and
the reference standing when no candidate is eligible (P10–P12, P15).

The tests cover:
- a candidate adopted over the reference, which is chosen;
- an equal and a worse candidate, with which the reference stands (`chosen` is None);
- three candidates, of which two are eligible and the better of those is chosen;
- the tie order's fewer components, and the higher held-out gain over the reference;
- the record against Verification "Economic decision record", item by item;
- the intervals against an independent computation of the single-replicate bootstrap;
- the refusals:
  - a candidate read before the margins were fixed;
  - margins fixed before their own read;
  - a margin read that read a candidate, in its log or its predictions;
  - a δ that is not r × the reference's MSE, and another reference;
  - reference predictions that differ from the margin read's;
  - recorded statistics that are not the predictions';
  - no candidate, a repeated name, or a candidate named for the reference.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 410, 425 and 434
- Modify: `src/naics_embedder/decision/economic.py`, lines 17, 24, 28-35, 43, 53, 59-60, 310 and
  375-377
- Modify: `src/naics_embedder/decision/records.py`, lines 23 and 444
- Test: `tests/unit/test_decision_economic.py`, lines 12-15, 23, 28, 33 and 375-377

**Interfaces:**
- Consumes:
  - Task 2's `read_time` and `check_margins_first`;
  - Task 6's records and checks;
  - `PanelItems`, `unit_draws`, `seed_draws`, `point_statistic` and `replicate_statistics`
    (`decision/resampling.py`);
  - `compare`, `compare_panel`, `non_dominated`, `tie_order`, `superiority_level` and
    `NONINFERIORITY_LEVEL` (`decision/rule.py`);
  - `statistic_values` and `statistic_means` (`decision/scores.py`).
- Produces:
  - in `naics_embedder.decision.records`:
    - `EconomicDecisionSettings(replicates, bootstrap_seed, noninferiority_level,
      superiority_level)`;
    - `EconomicReport(arm, statistics: Dict[str, float],
      comparator_mse: Dict[str, Dict[str, float]])`;
    - `EconomicDecisionRecord(kind='economic_decision', name, question, created_at,
      statistics, settings, reference, margins, arms, comparisons, eligible, non_dominated,
      cycle, tie_order, heldout_gain, chosen: Optional[str], reports)`;
  - in `naics_embedder.decision.economic`:
    - `GROWTH_HELDOUT`, `GROWTH_STATISTIC_DEFINITIONS` and `STATISTIC_TOLERANCE = 1e-12`;
    - `check_relative_margins(margins, store) -> None`;
    - `check_economic_margins_first(arms, margins) -> None`;
    - `decide_economic(name, question, arms, margins, store, *, replicates,
      bootstrap_seed) -> EconomicDecisionRecord`;
  - in `naics_embedder.decision.decide`: `check_margins_first` accepts economic arm records. A
    run whose seed is None is named by its arm alone, and a run without `monitor_records` has
    none.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_decision_economic.py`, make five edits, in this order.

Replace:

```python
from dataclasses import replace
from datetime import datetime

import polars as pl
```

with:

```python
import re
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
```

Replace:

```python
    check_reference_pairing,
```

with:

```python
    check_reference_pairing,
    decide_economic,
```

Replace:

```python
    EconomicArmRecord,
```

with:

```python
    EconomicArmRecord,
    EconomicDecisionRecord,
```

Replace:

```python
from naics_embedder.decision.scores import panel_statistic
```

with:

```python
from naics_embedder.decision.resampling import percentile_interval, stream
from naics_embedder.decision.scores import panel_statistic, regressor_scores, statistic_values
```

Replace:

```python
            }), store
        )

```

with:

```python
            }), store
        )

# -------------------------------------------------------------------------------------------------
# The economic decision (Verification "Economic decision record", "Margin first, reference only")
# -------------------------------------------------------------------------------------------------

REPLICATES = 2000
BOOTSTRAP_SEED = 20260924
GROWTH_SEEN, GROWTH_HELDOUT = GROWTH_PANELS

def _both(factor):
    return dict.fromkeys(GROWTH_PANELS, factor)

def _scaled(arm, factors, store):
    '''
    The arm, rescored with each ``covariates+embedding`` prediction moved to the outcome plus
    ``factors[panel]`` times the reference's error on its row and repeat. Every row's squared
    error is then factor² times the reference's, so Δ on each item is (1 − factor²) times the
    reference's error, whatever the draw.
    '''

    (run, ) = arm.runs
    predictions = store.read_frame(run.predictions)
    keys = ['panel', 'repeat', 'code', 'feature_year']
    reference = predictions.filter(pl.col('comparator') == REFERENCE_COMPARATOR
                                   ).select(*keys, reference='prediction')
    factor = pl.when(pl.col('panel') == GROWTH_SEEN).then(factors[GROWTH_SEEN]).otherwise(
        factors[GROWTH_HELDOUT]
    )
    error = pl.col('reference') - pl.col('outcome')
    moved = predictions.join(reference, on=keys).with_columns(
        prediction=pl.when(pl.col('comparator') == 'covariates+embedding').then(
            pl.col('outcome') + factor * error
        ).otherwise('prediction')
    ).select(predictions.columns)
    scores = regressor_scores(moved, SETTINGS.repeats, GROWTH_PANELS)
    scaled = run.model_copy(
        update={
            'predictions': store.put_frame(moved, 'predictions.parquet'),
            'scores': store.put_frame(scores, 'scores.parquet'),
            'statistics': {
                panel: panel_statistic(scores, panel, 'covariates+embedding')
                for panel in GROWTH_PANELS
            },
        }
    )
    return arm.model_copy(update={'runs': [scaled]})

@pytest.fixture
def candidates(growth_panel, text_only, margins, store, lock, tmp_path):
    '''Reads a new candidate, after the margins, and scales its errors (``_scaled``).'''

    def read(name, factors, **overrides):
        table = write_candidate(tmp_path / 'candidates', name, coordinate_table(CODEBOOK))
        arm_spec = candidate_spec(name, **overrides)
        arm = _read(table, growth_panel, text_only, margins, store, lock, arm_spec)
        return _scaled(arm, factors, store)

    return read

def _decide(arms, margins, store):
    return decide_economic(
        'fixture economic decision',
        'which candidate, if any, is the economic artifact?',
        arms,
        margins,
        store,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

def test_a_candidate_adopted_over_the_reference_is_chosen(candidates, margins, store):
    record = _decide([candidates('better', _both(0.5))], margins, store)

    (comparison, ) = record.comparisons
    assert (comparison.a, comparison.b, comparison.adopted) == (
        'better', REFERENCE_COMPARATOR, True
    )
    assert [(panel.non_inferior, panel.superior)
            for panel in comparison.panels] == [(True, True)] * 2
    assert (record.eligible, record.non_dominated, record.cycle) == (['better'], ['better'], False)
    assert (record.tie_order, record.chosen) == (['better'], 'better')

@pytest.mark.parametrize(
    'factor, verdicts', [(1.0, (True, False)), (1.5, (False, False))], ids=['equal', 'worse']
)
def test_with_no_candidate_adopted_the_reference_stands(
    candidates, margins, store, factor, verdicts
):
    '''Req 24: with none eligible the reference stands, and there is no economic artifact.'''

    record = _decide([candidates('candidate', _both(factor))], margins, store)

    (comparison, ) = record.comparisons
    assert [(panel.non_inferior, panel.superior) for panel in comparison.panels] == [verdicts] * 2
    assert (record.eligible, record.non_dominated, record.cycle) == ([], [], False)
    assert (record.tie_order, record.chosen) == ([], None)

def test_only_candidates_adopted_over_the_reference_are_eligible(candidates, margins, store):
    '''Req 24: the rule picks among the eligible, compared with each other only.'''

    arms = [
        candidates('better', _both(0.5)),
        candidates('best', _both(0.3)),
        candidates('equal', _both(1.0)),
    ]

    record = _decide(arms, margins, store)

    assert [(c.a, c.b, c.adopted) for c in record.comparisons] == [
        ('better', REFERENCE_COMPARATOR, True),
        ('best', REFERENCE_COMPARATOR, True),
        ('equal', REFERENCE_COMPARATOR, False),
        ('better', 'best', False),
        ('best', 'better', True),
    ]
    assert record.eligible == ['better', 'best']
    assert (record.non_dominated, record.tie_order, record.chosen) == (['best'], ['best'], 'best')

def test_the_tie_order_prefers_fewer_components_among_the_eligible(candidates, margins, store):
    stacked = candidates('stacked', _both(0.5), components=2)
    simple = candidates('simple', _both(0.5))

    record = _decide([stacked, simple], margins, store)

    # Equal errors: neither is adopted over the other
    assert record.non_dominated == ['stacked', 'simple']
    assert (record.tie_order, record.chosen) == (['simple', 'stacked'], 'simple')

def test_the_higher_heldout_gain_over_the_reference_breaks_a_remaining_tie(
    candidates, margins, store
):
    '''D11, for growth: each candidate is superior on one panel and inferior on the other.'''

    seen = candidates('seen', {GROWTH_SEEN: 0.5, GROWTH_HELDOUT: 0.7})
    heldout = candidates('heldout', {GROWTH_SEEN: 0.7, GROWTH_HELDOUT: 0.5})

    record = _decide([seen, heldout], margins, store)

    assert record.eligible == record.non_dominated == ['seen', 'heldout']
    assert record.heldout_gain['heldout'] > record.heldout_gain['seen'] > 0
    assert (record.tie_order, record.chosen) == (['heldout', 'seen'], 'heldout')

def test_the_economic_decision_record_shows_what_verification_lists(
    candidates, margins, store, tmp_path
):
    record = _decide([candidates('better', _both(0.5))], margins, store)
    path = write_record(record, tmp_path / 'decision.json')

    assert read_record(path, EconomicDecisionRecord) == record
    # Two panels, with their 95 % non-inferiority and 97.5 % superiority intervals
    assert (record.settings.noninferiority_level, record.settings.superiority_level) == (
        0.95, 0.975
    )
    assert [panel.panel for panel in record.comparisons[0].panels] == list(GROWTH_PANELS)
    # δ = 0.05 × the reference's MSE per panel, and the reference
    scores = store.read_frame(margins.read.scores)
    for panel, entry in zip(GROWTH_PANELS, record.margins.margins):
        assert (entry.panel, entry.margin) == (panel, 0.05 * entry.reference_mse)
        assert entry.reference_mse == pytest.approx(
            panel_statistic(scores, panel, REFERENCE_COMPARATOR), rel=1e-12
        )
    assert record.reference == record.margins.reference == 'covariates+ancestors'
    # Each candidate's provenance, which names no employment-data input
    assert record.arms[0].runs[0].provenance.content['inputs'] == {
        'codebook': 'the fixture codebook'
    }
    # The chosen arm
    assert record.chosen == 'better'
    # One-hot is still reported, where the seen regime scores it
    (report, ) = record.reports
    assert report.statistics == pytest.approx(record.arms[0].runs[0].statistics, rel=1e-12)
    assert 'covariates+one_hot' in report.comparator_mse[GROWTH_SEEN]
    assert 'covariates+one_hot' not in report.comparator_mse[GROWTH_HELDOUT]

def test_a_single_replicate_is_resampled_by_its_units_alone(candidates, margins, store):
    '''
    Req 24: a deterministic candidate is one replicate, so each bootstrap replicate's Δ is the
    reference's MSE minus the candidate's on a draw of four-digit groups alone, shared by both.
    '''

    arm = candidates('better', _both(0.5))

    record = _decide([arm], margins, store)

    for panel, comparison in zip(GROWTH_PANELS, record.comparisons[0].panels):
        reference = statistic_values(
            store.read_frame(margins.read.scores), panel, REFERENCE_COMPARATOR
        )
        candidate = statistic_values(
            store.read_frame(arm.runs[0].scores), panel, 'covariates+embedding'
        )
        # yapf: disable
        by_unit = (
            reference
            .join(candidate, on=['unit', 'item'], suffix='_candidate')
            .group_by('unit')
            .agg(delta=(pl.col('value') - pl.col('value_candidate')).sum(), n=pl.len())
            .sort('unit')
        )
        # yapf: enable
        units = by_unit.height
        counts = np.random.default_rng([BOOTSTRAP_SEED, stream(panel)]).multinomial(
            units, np.full(units, 1.0 / units), size=REPLICATES
        )
        replicates = counts @ by_unit['delta'].to_numpy() / (counts @ by_unit['n'].to_numpy())
        assert comparison.noninferiority_interval == pytest.approx(
            percentile_interval(replicates, 0.95), rel=1e-12
        )
        assert comparison.superiority_interval == pytest.approx(
            percentile_interval(replicates, 0.975), rel=1e-12
        )

def test_a_candidate_read_before_the_margins_were_fixed_is_refused(
    candidates, growth_panel, store, lock
):
    '''Verification "Margin first": margins fixed after a candidate's read cannot judge it.'''

    arm = candidates('tree', _both(0.5))
    later = fix_relative_margins(
        growth_panel, R, 'later margins', store, purpose=PURPOSE, lock=lock
    )
    first = min(datetime.fromisoformat(read['time']) for read in arm.runs[0].log_records)

    with pytest.raises(
        ValueError,
        match=f'^tree read at {re.escape(first.isoformat())}, before the margins were fixed',
    ):
        _decide([arm], later, store)

def test_margins_fixed_before_their_own_read_are_refused(candidates, margins, store):
    '''Verification "Margin first": δ is recorded from a reference-only read that precedes it.'''

    arm = candidates('tree', _both(0.5))
    early = margins.model_copy(update={'fixed_at': margins.fixed_at - timedelta(hours=1)})

    with pytest.raises(
        ValueError,
        match="^the margins 'fixture margins' were fixed at .*, before their growth_seen read",
    ):
        _decide([arm], early, store)

def _margin_read(margins, **update):
    return margins.model_copy(update={'read': margins.read.model_copy(update=update)})

def test_margins_whose_read_was_not_reference_only_are_refused(candidates, margins, store):
    '''Verification "Margin first, reference only": decide refuses a margin read of a candidate.'''

    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    reads = json.loads(json.dumps(margins.read.log_records))
    reads[0]['detail']['arm'] = run.table.matrix_fingerprint
    candidate_read = _margin_read(margins, predictions=run.predictions, scores=run.scores)

    with pytest.raises(
        ValueError,
        match="^the margins 'fixture margins': the growth_seen read is not a validation read of "
        'the reference comparators alone',
    ):
        _decide([arm], _margin_read(margins, log_records=reads), store)
    with pytest.raises(
        ValueError,
        match=
        "^the margins 'fixture margins': the growth_seen read scored .*'covariates\\+embedding'",
    ):
        _decide([arm], candidate_read, store)

def test_margins_whose_delta_is_not_r_times_the_reference_mse_are_refused(
    candidates, margins, store
):
    arm = candidates('tree', _both(0.5))
    first = margins.margins[0]
    doubled = first.model_copy(update={'margin': 2 * first.margin})
    rescaled = first.model_copy(
        update={
            'reference_mse': 2 * first.reference_mse,
            'margin': R * 2 * first.reference_mse
        }
    )

    for entry in (doubled, rescaled):
        edited = margins.model_copy(update={'margins': [entry, *margins.margins[1:]]})
        with pytest.raises(
            ValueError,
            match="^the margins 'fixture margins': the growth_seen δ is not r × the reference's",
        ):
            _decide([arm], edited, store)
    with pytest.raises(ValueError, match="name the reference 'covariates\\+one_hot'"):
        _decide([arm], margins.model_copy(update={'reference': 'covariates+one_hot'}), store)

def test_the_decision_refuses_a_candidate_whose_reference_predicts_otherwise(
    candidates, margins, store
):
    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    reference = pl.col('comparator') == REFERENCE_COMPARATOR
    predictions = store.read_frame(run.predictions).with_columns(
        prediction=pl.when(reference).then(pl.col('prediction') + 1e-12).otherwise('prediction')
    )
    scores = regressor_scores(predictions, SETTINGS.repeats, GROWTH_PANELS)
    refit = run.model_copy(
        update={
            'predictions': store.put_frame(predictions, 'predictions.parquet'),
            'scores': store.put_frame(scores, 'scores.parquet'),
        }
    )

    with pytest.raises(ValueError, match='^tree: its covariates\\+ancestors predictions differ'):
        _decide([arm.model_copy(update={'runs': [refit]})], margins, store)

def test_a_candidate_whose_statistics_are_not_its_predictions_is_refused(candidates, store):
    '''The decision scores the predictions the pairing guard compared, and the record must agree.'''

    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    shifted = store.read_frame(run.predictions).with_columns(prediction=pl.col('prediction') + 1.0)
    moved = run.model_copy(update={'predictions': store.put_frame(shifted, 'predictions.parquet')})

    with pytest.raises(ValueError, match="^tree: the record's statistics are not those of its"):
        check_economic_arm(arm.model_copy(update={'runs': [moved]}), store)

def test_a_decision_reads_distinct_candidates_none_named_for_the_reference(
    candidates, margins, store
):
    better = candidates('better', _both(0.5))
    named = candidates(REFERENCE_COMPARATOR, _both(0.5))

    with pytest.raises(ValueError, match='at least one candidate'):
        _decide([], margins, store)
    for arms in ([better, better], [named]):
        with pytest.raises(ValueError, match='^candidate names must be distinct'):
            _decide(arms, margins, store)

```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision.py -q`
Expected: FAIL, `1 error`. Collecting `tests/unit/test_decision_economic.py` raises
`ImportError: cannot import name 'decide_economic' from 'naics_embedder.decision.economic'`.

- [x] **Step 3: Write the implementation**

In `src/naics_embedder/decision/records.py`, make two edits, in this order.

Replace:

```python
  reads.
```

with:

```python
  reads.
- **Economic decision record.** Its candidates and margins, each candidate's comparison with the
  reference ``covariates+ancestors`` with its 95 % non-inferiority and 97.5 % superiority
  intervals, the candidates adopted over the reference, the non-dominated set and tie order among
  them, and the chosen candidate, or none when the reference stands.
```

Replace:

```python
# Files
```

with:

```python
# The economic decision (Req 24)
# -------------------------------------------------------------------------------------------------

class EconomicDecisionSettings(_Record):
    '''How the economic decision resampled and which intervals it read.'''

    replicates: int
    bootstrap_seed: int
    noninferiority_level: float
    superiority_level: float

class EconomicReport(_Record):
    '''
    What the economic decision reports for a candidate besides the rule's statistics.

    Attributes:
        statistics: Each growth panel's decision statistic, ``covariates+embedding``'s MSE.
        comparator_mse: Each growth panel's MSE for every comparator its read scored, one-hot
            included where the seen regime scores it (Req 24).
    '''

    arm: str
    statistics: Dict[str, float]
    comparator_mse: Dict[str, Dict[str, float]]

class EconomicDecisionRecord(_Record):
    '''
    The economic decision (Req 24): each candidate against the reference on both growth panels,
    and Req 5's rule among the candidates adopted over it.

    Attributes:
        reference: The reference comparator (U9).
        comparisons: Each candidate against the reference, then each eligible candidate against
            each other.
        eligible: The candidates adopted over the reference.
        non_dominated: The eligible candidates no eligible candidate is adopted over.
        cycle: Whether dominance among the eligible cycled.
        tie_order: The non-dominated candidates, from the one that stands first.
        heldout_gain: Each candidate's held-out gain over the reference: the reference's
            ``growth_heldout`` MSE minus the candidate's (D11).
        chosen: The candidate whose table is the economic deliverable; None when no candidate is
            eligible, and the reference stands.
    '''

    kind: Literal['economic_decision'] = 'economic_decision'
    name: str
    question: str
    created_at: datetime
    statistics: Dict[str, str]
    settings: EconomicDecisionSettings
    reference: str
    margins: RelativeMarginRecord
    arms: List[EconomicArmRecord]
    comparisons: List[Comparison]
    eligible: List[str]
    non_dominated: List[str]
    cycle: bool
    tie_order: List[str]
    heldout_gain: Dict[str, float]
    chosen: Optional[str]
    reports: List[EconomicReport]

# -------------------------------------------------------------------------------------------------
# Files
```

In `src/naics_embedder/decision/decide.py`, make three edits, in this order.

Replace:

```python
    its decision reads and, for a trained run, the monitor reads that selected its checkpoint.
```

with:

```python
    its decision reads and, for a trained run, the monitor reads that selected its checkpoint.
    ``arms`` are arm records or economic arm records.
```

Replace:

```python
            name = f'{arm.spec.name} seed {run.seed}'
```

with:

```python
            # A deterministic economic run has no seed and no monitor reads (Req 24)
            name = arm.spec.name if run.seed is None else f'{arm.spec.name} seed {run.seed}'
```

Replace:

```python
                ) for record in run.monitor_records
```

with:

```python
                ) for record in getattr(run, 'monitor_records', ())
```

In `src/naics_embedder/decision/economic.py`, make eight edits, in this order.

Replace:

```python
  the margin read only where the reference predicts bit for bit as it did there.
```

with:

```python
  the margin read only where the reference predicts bit for bit as it did there.
- **Decision.** Each candidate against the reference on both panels, at 95 % non-inferiority and
  97.5 % superiority. Only a candidate adopted over the reference is eligible; among the eligible,
  Req 5's non-dominated set and tie order pick one, the higher held-out gain over the reference
  breaking a last tie (D11). With none eligible, the reference stands. The decision scores each
  read's stored predictions itself, so the values it resamples are those the pairing compared.
```

Replace:

```python
import platform
```

with:

```python
import math
import platform
```

Replace:

```python
from typing import List, Union

import polars as pl

from naics_embedder.decision.decide import _same, check_recorded_text_only, check_text_only
from naics_embedder.decision.records import (
    ArmSpec,
    EconomicArmRecord,
```

with:

```python
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import polars as pl

from naics_embedder.decision.decide import (
    _same,
    check_margins_first,
    check_recorded_text_only,
    check_text_only,
    read_time,
)
from naics_embedder.decision.records import (
    ArmSpec,
    ArtifactRef,
    Comparison,
    EconomicArmRecord,
    EconomicDecisionRecord,
    EconomicDecisionSettings,
    EconomicReport,
```

Replace:

```python
from naics_embedder.decision.scores import EMBEDDING_COMPARATOR, panel_statistic, regressor_scores
```

with:

```python
from naics_embedder.decision.resampling import (
    PanelItems,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import (
    NONINFERIORITY_LEVEL,
    compare,
    compare_panel,
    non_dominated,
    superiority_level,
    tie_order,
)
from naics_embedder.decision.scores import (
    EMBEDDING_COMPARATOR,
    panel_statistic,
    regressor_scores,
    statistic_means,
    statistic_values,
)
```

Replace:

```python
    RegressorPanel,
```

with:

```python
    RegressorPanel,
    reference_comparators,
```

Replace:

```python
# The hierarchy's ancestor indicators beside the covariates (U9)
REFERENCE_COMPARATOR = 'covariates+ancestors'
```

with:

```python
GROWTH_HELDOUT = READ_NAMES[RegressorOutcome.GROWTH][Regime.HELDOUT]
# The hierarchy's ancestor indicators beside the covariates (U9)
REFERENCE_COMPARATOR = 'covariates+ancestors'
GROWTH_STATISTIC_DEFINITIONS = dict.fromkeys(
    GROWTH_PANELS,
    'mean squared error of covariates+embedding on level-6 log employment growth, each row '
    'averaged over its repeats; unit: a four-digit group',
)
# A recomputed statistic agrees with a recorded one to within float summation order
STATISTIC_TOLERANCE = 1e-12
```

Replace:

```python
        raise ValueError(f"{name}: the record's provenance differs from its stored provenance")
```

with:

```python
        raise ValueError(f"{name}: the record's provenance differs from its stored provenance")
    scores = _rescored(store, run.predictions, arm.panels.fit_settings['repeats'])
    statistics = {
        panel: panel_statistic(scores, panel, EMBEDDING_COMPARATOR)
        for panel in GROWTH_PANELS
    }
    if set(run.statistics) != set(statistics) or not all(
        math.isclose(run.statistics[panel], value, rel_tol=STATISTIC_TOLERANCE)
        for panel, value in statistics.items()
    ):
        raise ValueError(
            f"{name}: the record's statistics are not those of its stored predictions, "
            f'{statistics}'
        )
```

Replace:

```python
                f'{_where(arm.environment)}; the margins on {_where(margins.read.environment)}'
            )

```

with:

```python
                f'{_where(arm.environment)}; the margins on {_where(margins.read.environment)}'
            )

def _rescored(store: ArtifactStore, predictions: ArtifactRef, repeats: int) -> pl.DataFrame:
    '''A read's scores, from its stored predictions (D10's statistic on the growth panels).'''

    return regressor_scores(store.read_frame(predictions), repeats, GROWTH_PANELS)

def check_relative_margins(margins: RelativeMarginRecord, store: ArtifactStore) -> None:
    '''
    Require the margins to hold, for each growth panel, δ = r × the reference's validation MSE,
    from a margin read that read no candidate (Req 24; U9; U13).

    Raises:
        ValueError: If the margins name another reference, hold another δ than one per growth
            panel for the reference's statistic, or their read logged or scored a comparator that
            reads a candidate; or if a δ is not r times the MSE the read's predictions give.
    '''

    name = f'the margins {margins.name!r}'
    if margins.reference != REFERENCE_COMPARATOR:
        raise ValueError(
            f'{name} name the reference {margins.reference!r}, not {REFERENCE_COMPARATOR} (U9)'
        )
    held = sorted((entry.panel, entry.statistic) for entry in margins.margins)
    needed = sorted((panel, REFERENCE_COMPARATOR) for panel in GROWTH_PANELS)
    if held != needed:
        raise ValueError(f'{name} hold {held}, not one δ per growth panel: {needed} (Req 24)')
    predictions = store.read_frame(margins.read.predictions)
    _check_reference_read(name, margins.read, predictions)
    scores = regressor_scores(
        predictions, margins.read.panels.fit_settings['repeats'], GROWTH_PANELS
    )
    for entry in margins.margins:
        mse = panel_statistic(scores, entry.panel, REFERENCE_COMPARATOR)
        scored = math.isclose(entry.reference_mse, mse, rel_tol=STATISTIC_TOLERANCE)
        if not scored or entry.margin != margins.r * entry.reference_mse:
            raise ValueError(
                f"{name}: the {entry.panel} δ is not r × the reference's MSE its read scored, "
                f'{margins.r} × {mse} (Req 24)'
            )

def _check_reference_read(name: str, read: ReferenceRead, predictions: pl.DataFrame) -> None:
    '''Require the margin read to be each growth panel's validation read of the reference alone.'''

    regimes = {READ_NAMES[RegressorOutcome.GROWTH][regime]: regime for regime in Regime}
    panels = sorted(str(record.get('panel')) for record in read.log_records)
    if panels != sorted(GROWTH_PANELS):
        raise ValueError(f'{name}: the margin read read {panels}, not each growth panel once')
    for record in read.log_records:
        panel = record['panel']
        names = list(reference_comparators(regimes[panel], DECISION_LEVEL))
        logged = (record.get('event'), record.get('split'), record.get('fingerprint'))
        detail = {'level': DECISION_LEVEL, 'comparators': names, 'run': read.run_id}
        if logged != ('read', VALIDATION, read.panels.regressor) or record.get('detail') != detail:
            raise ValueError(
                f'{name}: the {panel} read is not a validation read of the reference comparators '
                f"alone: {logged}, {record.get('detail')} (Req 24)"
            )
        scored = predictions.filter(pl.col('panel') == panel).get_column('comparator')
        if sorted(scored.unique().to_list()) != sorted(names):
            raise ValueError(
                f'{name}: the {panel} read scored {sorted(scored.unique().to_list())}, not the '
                'reference comparators alone (Req 24)'
            )

def check_economic_margins_first(
    arms: Sequence[EconomicArmRecord], margins: RelativeMarginRecord
) -> None:
    '''
    Require the margin read to precede the margins, and every candidate read to follow them
    (Req 24: δ is fixed from a reference-only read before any candidate is read).

    Raises:
        ValueError: If a read's time or the margins' is missing or names no timezone, or if
            either order fails.
    '''

    check_margins_first(arms, margins.fixed_at)
    name = f'the margins {margins.name!r}'
    for record in margins.read.log_records:
        time = read_time(record, f'{name}: its {record.get("panel")} read')
        if time > margins.fixed_at:
            raise ValueError(
                f'{name} were fixed at {margins.fixed_at.isoformat()}, before their '
                f'{record["panel"]} read at {time.isoformat()} (Req 24)'
            )

# -------------------------------------------------------------------------------------------------
# Decision
# -------------------------------------------------------------------------------------------------

def decide_economic(
    name: str,
    question: str,
    arms: Sequence[EconomicArmRecord],
    margins: RelativeMarginRecord,
    store: ArtifactStore,
    *,
    replicates: int,
    bootstrap_seed: int,
) -> EconomicDecisionRecord:
    '''
    Req 24's decision: each candidate against the reference on both growth panels; among those
    adopted over it, Req 5's non-dominated set and tie order pick one. With none adopted, the
    reference stands and ``chosen`` is None.

    Every check runs before any number is computed: the margins (``check_relative_margins``),
    each candidate (``check_economic_arm``) and its pairing with the margin read
    (``check_reference_pairing``), then the order of the reads
    (``check_economic_margins_first``).

    Raises:
        ValueError: If no candidate is given, two share a name or one is named for the
            reference, or a check fails.
        TieUnresolvedError: If the tie order cannot separate the surviving candidates.
    '''

    names = [arm.spec.name for arm in arms]
    if not names:
        raise ValueError('an economic decision reads at least one candidate')
    if len(set(names)) != len(names) or REFERENCE_COMPARATOR in names:
        raise ValueError(
            f"candidate names must be distinct, and none the reference's {REFERENCE_COMPARATOR}: "
            f'{names}'
        )
    check_relative_margins(margins, store)
    for arm in arms:
        check_economic_arm(arm, store)
        check_reference_pairing(margins, arm, store)
    check_economic_margins_first(arms, margins)
    settings = EconomicDecisionSettings(
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        noninferiority_level=NONINFERIORITY_LEVEL,
        superiority_level=superiority_level(len(GROWTH_PANELS)),
    )

    repeats = margins.read.panels.fit_settings['repeats']
    # Each side's per-run scores and the statistic the rule reads from them: the reference is
    # the margin read, a single replicate
    sides: Dict[str, Tuple[str, List[pl.DataFrame]]] = {
        REFERENCE_COMPARATOR: (
            REFERENCE_COMPARATOR, [_rescored(store, margins.read.predictions, repeats)]
        )
    }
    for arm in arms:
        sides[arm.spec.name] = (
            EMBEDDING_COMPARATOR, [_rescored(store, run.predictions, repeats) for run in arm.runs]
        )
    points: Dict[Tuple[str, str], float] = {}
    draws: Dict[Tuple[str, str], np.ndarray] = {}
    for panel in GROWTH_PANELS:
        reference = sides[REFERENCE_COMPARATOR][1][0]
        items = PanelItems.from_values(statistic_values(reference, panel, REFERENCE_COMPARATOR))
        units = unit_draws(panel, len(items.units), replicates, bootstrap_seed)
        for side, (statistic, frames) in sides.items():
            sums = items.sums(
                np.stack(
                    [items.values(statistic_values(frame, panel, statistic)) for frame in frames]
                )
            )
            seeds = seed_draws(panel, side, len(frames), replicates, bootstrap_seed)
            points[panel, side] = point_statistic(sums, items.sizes)
            draws[panel, side] = replicate_statistics(sums, items.sizes, units, seeds)

    def against(a: str, b: str) -> Comparison:
        # A lower MSE is better, so Δ = B's MSE − A's favours A when positive
        return compare(
            a,
            b,
            [
                compare_panel(
                    panel,
                    points[panel, b] - points[panel, a],
                    draws[panel, b] - draws[panel, a],
                    margins.margin(panel),
                    superiority=settings.superiority_level,
                ) for panel in GROWTH_PANELS
            ],
        )

    comparisons = [against(arm, REFERENCE_COMPARATOR) for arm in names]
    eligible = [comparison.a for comparison in comparisons if comparison.adopted]
    among = [against(a, b) for a in eligible for b in eligible if a != b]
    heldout_gain = {
        arm: points[GROWTH_HELDOUT, REFERENCE_COMPARATOR] - points[GROWTH_HELDOUT, arm]
        for arm in names
    }
    survivors: List[str] = []
    cycle = False
    order: List[str] = []
    if eligible:
        survivors, cycle = non_dominated(eligible, among)
        order = tie_order([arm.spec for arm in arms if arm.spec.name in survivors], heldout_gain)
    return EconomicDecisionRecord(
        name=name,
        question=question,
        created_at=datetime.now(timezone.utc),
        statistics=dict(GROWTH_STATISTIC_DEFINITIONS),
        settings=settings,
        reference=REFERENCE_COMPARATOR,
        margins=margins,
        arms=list(arms),
        comparisons=[*comparisons, *among],
        eligible=eligible,
        non_dominated=survivors,
        cycle=cycle,
        tie_order=order,
        heldout_gain=heldout_gain,
        chosen=order[0] if order else None,
        reports=[_economic_report(arm.spec.name, sides[arm.spec.name][1]) for arm in arms],
    )

def _economic_report(arm: str, frames: Sequence[pl.DataFrame]) -> EconomicReport:
    '''Every comparator's MSE on each growth panel, a mean over the arm's runs.'''

    means = [statistic_means(frame) for frame in frames]
    comparator_mse = {
        panel: {
            statistic: float(np.mean([run[panel][statistic] for run in means]))
            for statistic in means[0][panel]
        }
        for panel in GROWTH_PANELS
    }
    return EconomicReport(
        arm=arm,
        statistics={panel: comparator_mse[panel][EMBEDDING_COMPARATOR]
                    for panel in GROWTH_PANELS},
        comparator_mse=comparator_mse,
    )

```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_decision.py -q`
Expected: PASS, `126 passed`.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py tests/unit/test_decision_economic.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3180 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py tests/unit/test_decision_economic.py
git commit -m "feat(decision): decide the economic artifact against covariates+ancestors (Req 24)" \
  -m "Each candidate is compared with the reference on growth_seen and
growth_heldout at 95 % non-inferiority and 97.5 % superiority. Only a
candidate adopted over the reference is eligible; among the eligible,
the non-dominated set and the tie order pick one, the higher held-out
gain over the reference breaking a last tie (D11). With none eligible,
the reference stands." \
  -m "Before any number is computed, the decision checks that the margins
hold r times the reference's MSE from a reference-only read that
preceded them, that each candidate is intact and pairs with the margin
read, and that every candidate read followed the margins. It rescores
each read's stored predictions, so the values it resamples are those
the pairing compared bit for bit."
```

Run:

```bash
git diff --stat dc64266e554f713cbee43ba8f848e51ac164eae1 HEAD -- src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/records.py tests/unit/test_decision_economic.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 8: The CLI

`tools growth-margins` fixes the margins from the reference-only read, and `tools decide` makes
the economic decision from a relative margin record (P13). The decision config gains r.

The tests cover:
- `tools growth-margins`:
  - it loads growth in memory at level 6, logs only the reference comparators under the default
    purpose, and writes the record;
  - it refuses an existing record before it loads anything;
- `tools decide`:
  - it makes the economic decision from relative margins, and prints 97.5 % and that the
    reference stands;
  - it still prints D8's level, now as "98.33 %";
- `relative_margin`'s default, 0.05, and its positive bound.

**Files:**
- Modify: `conf/data/decision.yaml`, lines 1 and 11-12
- Modify: `conf/data/regressor_panel.yaml`, line 17
- Modify: `src/naics_embedder/cli/commands/tools.py`, lines 17, 39, 94, 548, 556, 576-583, 599-608,
  614 and 623-628
- Modify: `src/naics_embedder/utils/config.py`, lines 589 and 599
- Test: `tests/unit/test_cli_commands.py`, lines 13-16, 21, 378 and 393
- Test: `tests/unit/test_config.py`, lines 621 and 626

**Interfaces:**
- Consumes: Task 5's `fix_relative_margins`, Task 6's `read_deterministic_arm` (in the test),
  Task 7's `decide_economic`, and the records of Tasks 5–7.
- Produces:
  - the command `tools growth-margins`, with `--codebook`, `--name`, `--store`, `--output`,
    `--purpose` (default "economic margin read"), `--log` and `--lock` (default `uv.lock`);
  - `tools decide`, which dispatches on the margin record's kind (`_margin_kind(path) -> str`);
  - `LOCK_FILE = 'uv.lock'` in `naics_embedder.cli.commands.tools`;
  - `DecisionConfig.relative_margin: float`, default 0.05, greater than 0, and
    `relative_margin: 0.05` in `conf/data/decision.yaml`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_cli_commands.py`, make four edits, in this order.

Replace:

```python
from naics_embedder.decision.records import DecisionRecord, MarginRecord, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.regressor import RegressorPanel
```

with:

```python
from naics_embedder.decision.economic import read_deterministic_arm
from naics_embedder.decision.records import (
    DecisionRecord,
    EconomicDecisionRecord,
    MarginRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import Regime, RegressorPanel, reference_comparators
```

Replace:

```python
from tests.fixtures.decision import spec, synthetic_arm
```

with:

```python
from tests.fixtures.decision import spec, synthetic_arm, write_text_only
from tests.fixtures.economic import candidate_spec, write_candidate
```

Replace:

```python
    assert 'better over reference: adopted' in output
```

with:

```python
    assert 'better over reference: adopted' in output
    # Three panels still read 98⅓ % (Req 24); Rich may wrap the line at any space
    assert '98.33 % [' in ' '.join(result.output.split())
```

Replace:

```python
    assert not (tmp_path / 'decision.json').exists()
```

with:

```python
    assert not (tmp_path / 'decision.json').exists()

@pytest.fixture
def growth_panel(monkeypatch, tmp_path, regressor_growth_rows):
    '''The fixture growth panel in place of the real one, with the outcome each load asked for.'''

    log = SelectionLog(tmp_path / 'selection_log.jsonl')
    panel = RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )
    loaded = []

    def fake_load(cfg, codebook, *, log_path, levels):
        loaded.append((cfg.outcome, codebook, log_path, tuple(levels)))
        return panel

    monkeypatch.setattr(tools_cli, 'load_regressor_panel', fake_load)
    return panel, loaded

def _growth_margins(runner, tmp_path, store):
    lock = tmp_path / 'uv.lock'
    lock.write_text('a fixture lock\n')
    return runner.invoke(
        tools_cli.app,
        [
            'growth-margins',
            '--codebook',
            'naics_codebook.parquet',
            '--name',
            'growth margins',
            '--store',
            str(store.root),
            '--lock',
            str(lock),
            '--output',
            str(tmp_path / 'margins.json'),
        ],
    )

@pytest.mark.unit
def test_growth_margins_reads_the_reference_alone_and_writes_the_margins(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel

    result = _growth_margins(runner, tmp_path, ArtifactStore(tmp_path / 'store'))

    assert result.exit_code == 0, result.output
    # The committed config reads level mode; the margin read reads growth in memory
    assert loaded == [('growth', 'naics_codebook.parquet', None, (6, ))]
    record = read_record(tmp_path / 'margins.json', RelativeMarginRecord)
    assert record.r == 0.05
    assert [entry.panel for entry in record.margins] == ['growth_seen', 'growth_heldout']
    assert [read['detail']['comparators'] for read in panel.log.records()] == [
        list(reference_comparators(regime, 6)) for regime in Regime
    ]
    assert {read['purpose'] for read in panel.log.records()} == {'economic margin read'}
    output = result.output.replace('\n', '')
    assert 'growth_heldout: δ' in output
    assert record.read.environment.lock_sha256 in output

@pytest.mark.unit
def test_growth_margins_refuses_an_existing_record_before_anything_is_read(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel
    (tmp_path / 'margins.json').write_text('an earlier record')

    result = _growth_margins(runner, tmp_path, ArtifactStore(tmp_path / 'store'))

    assert result.exit_code == 1
    assert 'exists; a record is written once' in ' '.join(result.output.split())
    assert (loaded, panel.log.records()) == ([], [])
    assert (tmp_path / 'margins.json').read_text() == 'an earlier record'

@pytest.mark.unit
def test_decide_reads_an_economic_decision_from_relative_margins(runner, tmp_path, growth_panel):
    panel, _ = growth_panel
    store = ArtifactStore(tmp_path / 'store')
    assert _growth_margins(runner, tmp_path, store).exit_code == 0
    table = write_candidate(tmp_path / 'candidates', 'tree', coordinate_table(CODEBOOK))
    candidate = read_deterministic_arm(
        candidate_spec('tree'),
        table,
        regressor_panel=panel,
        text_only_table=write_text_only(tmp_path / 'text', codes=CODEBOOK),
        margins=read_record(tmp_path / 'margins.json', RelativeMarginRecord),
        store=store,
        purpose='fixture candidate read',
        lock=tmp_path / 'uv.lock',
    )

    result = _decide(runner, tmp_path, store, [write_record(candidate, tmp_path / 'tree.json')])

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    assert 'tree over covariates+ancestors: not adopted' in output
    assert '97.5 % [' in ' '.join(result.output.split())
    assert 'Chosen: none; the reference covariates+ancestors stands' in output
    assert read_record(tmp_path / 'decision.json', EconomicDecisionRecord).chosen is None
```

In `tests/unit/test_config.py`, make two edits, in this order.

Replace:

```python
        assert (cfg.replicates, cfg.bootstrap_seed, cfg.min_seeds) == (10000, 20260924, 5)
```

with:

```python
        assert (cfg.replicates, cfg.bootstrap_seed, cfg.min_seeds) == (10000, 20260924, 5)
        # U13: the economic δ is 0.05 × the reference's MSE
        assert cfg.relative_margin == 0.05
```

Replace:

```python
            DecisionConfig(min_seeds=4)
```

with:

```python
            DecisionConfig(min_seeds=4)

    @pytest.mark.parametrize('r', [0.0, -0.05])
    def test_the_relative_margin_is_positive(self, r):
        with pytest.raises(
            ValidationError, match='relative_margin\n  Input should be greater than 0'
        ):
            DecisionConfig(relative_margin=r)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_commands.py tests/unit/test_config.py -q`
Expected: FAIL, `7 failed, 267 passed`:

- in `tests/unit/test_cli_commands.py`:
  - `test_decide_adopts_the_better_arm_and_writes_the_record`: the output has no "98.33 % [";
  - `test_growth_margins_reads_the_reference_alone_and_writes_the_margins`: "No such command
    'growth-margins'";
  - `test_growth_margins_refuses_an_existing_record_before_anything_is_read` and
    `test_decide_reads_an_economic_decision_from_relative_margins`: exit code 2, the missing
    command's;
- in `tests/unit/test_config.py`, class `TestDecisionConfig`:
  - `test_yaml_matches_defaults`: `AttributeError: 'DecisionConfig' object has no attribute
    'relative_margin'`;
  - `test_the_relative_margin_is_positive[0.0]` and `[-0.05]`: "Regex pattern did not match".
    The `ValidationError` names the key as an extra input, not as below the bound.

- [x] **Step 3: Write the implementation**

> Deviation: relative_margin adds allow_inf_nan=False, by the owner's ruling (fix commit 11641b0, with Task 5's guard).

In `src/naics_embedder/utils/config.py`, make two edits, in this order.

Replace:

```python
    '''Req 5's decision procedure (roadmap Stage 4): the paired bootstrap and the seed floor.'''
```

with:

```python
    '''
    Req 5's decision procedure (roadmap Stage 4): the paired bootstrap and the seed floor, and the
    economic decision's relative margin (Req 24).
    '''
```

Replace:

```python
    min_seeds: int = Field(default=5, ge=5, description='Req 5: each arm runs at least 5 seeds')
```

with:

```python
    min_seeds: int = Field(default=5, ge=5, description='Req 5: each arm runs at least 5 seeds')
    relative_margin: float = Field(
        default=0.05,
        gt=0.0,
        description="Req 24 (U13): each growth panel's δ as r times the reference's MSE",
    )
```

In `conf/data/decision.yaml`, make two edits, in this order.

Replace:

```yaml
# Req 5's decision procedure (roadmap Stage 4; D8, D10, D11): the paired bootstrap and the seed floor
```

with:

```yaml
# Req 5's decision procedure (roadmap Stage 4; D8, D10, D11) and the economic decision's (Req 24):
# the paired bootstrap, the seed floor and the relative margin
```

Replace:

```yaml
min_seeds: 5

```

with:

```yaml
min_seeds: 5

# Req 24 (U13): the economic decision's δ on each growth panel is r times the reference
# covariates+ancestors's validation MSE there, fixed by tools growth-margins before any candidate
# is read
relative_margin: 0.05

```

In `conf/data/regressor_panel.yaml`, make one edit.

Replace:

```yaml
# growth_heldout. tools regressor-panel and tools sweep read level mode only (Req 24)
```

with:

```yaml
# growth_heldout. tools regressor-panel and tools sweep read level mode only, and tools
# growth-margins reads growth whatever this says (Req 24)
```

In `src/naics_embedder/cli/commands/tools.py`, make nine edits, in this order.

Replace:

```python
    decide: Decide among arms under Req 5's rule over D8's three panels.
```

with:

```python
    growth-margins: Fix each growth panel's margin from a reference-only read (Req 24).
    decide: Decide among arms over D8's three panels (Req 5), or among economic candidates
        against the reference on the two growth panels (Req 24).
```

Replace:

```python
from naics_embedder.decision.records import ArmRecord, ArmSpec, MarginRecord, read_record, write_record
```

with:

```python
from naics_embedder.decision.economic import decide_economic, fix_relative_margins
from naics_embedder.decision.records import (
    ArmRecord,
    ArmSpec,
    EconomicArmRecord,
    MarginRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
```

Replace:

```python
DECISION_CONFIG = 'data/decision.yaml'
```

with:

```python
DECISION_CONFIG = 'data/decision.yaml'
# The lock a growth read runs under, which its record names (Req 24)
LOCK_FILE = 'uv.lock'
```

Replace:

```python
@app.command('decide')
```

with:

```python
@app.command('growth-margins')
def growth_margins_command(
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    name: Annotated[
        str,
        typer.Option('--name', help='Names the margins in the decision records that use them'),
    ],
    store: Annotated[
        str,
        typer.Option('--store', help="The artifact store for the read's predictions and scores"),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the margin record (JSON)'),
    ],
    purpose: Annotated[
        Optional[str],
        typer.Option(
            '--purpose',
            help='Why the read happens; recorded in the selection log '
            '(default: economic margin read)',
        ),
    ] = None,
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the regressor config)'),
    ] = None,
    lock: Annotated[
        str,
        typer.Option('--lock', help='The uv.lock the read runs under, which the record names'),
    ] = LOCK_FILE,
):
    '''
    Read the reference alone on both growth panels and fix each one's margin δ (Req 24).

    The read scores, on each growth regime's level-6 validation split, only the comparators that
    read no candidate: the covariates, one-hot and ancestor indicators, each alone and with the
    covariates. δ is the decision config's ``relative_margin`` r (0.05, U13) times the MSE of the
    reference ``covariates+ancestors`` there. The panel reads growth whatever the regressor
    config's ``outcome`` says; it stays ``level`` for tools regressor-panel and tools sweep.

    Fix the margins once, before any candidate is read on growth: a decision refuses a candidate
    read before its margins were fixed. The record names the platform, Python and ``uv.lock``
    the read ran under, and every candidate read must run there too, since a decision pairs only
    bit-identical reference predictions.

    Example:
        Fix the growth margins::

            $ uv run naics-embedder tools growth-margins --codebook PATH/naics_codebook.parquet \\
                --name growth-margins --store ~/naics-artifacts --output growth_margins.json
    '''

    configure_logging('tools_growth_margins.log')

    regressor_cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    cfg = load_config(DecisionConfig, DECISION_CONFIG)
    read_purpose = (purpose or '').strip() or 'economic margin read'
    try:
        # Before the panel is loaded or read
        _require_new_record(Path(output))
        growth = regressor_cfg.model_copy(update={'outcome': RegressorOutcome.GROWTH.value})
        panel = load_regressor_panel(growth, codebook, log_path=log, levels=[DECISION_LEVEL])
        record = fix_relative_margins(
            panel,
            cfg.relative_margin,
            name,
            ArtifactStore(store),
            purpose=read_purpose,
            lock=lock,
        )
        path = write_record(record, output)
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]Growth margins failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(
        f'\n[bold cyan]Growth margins {name!r}, fixed {record.fixed_at.isoformat()}[/bold cyan]\n'
    )
    for entry in record.margins:
        console.print(
            f'  • {entry.panel}: δ {entry.margin:.6g} = {record.r:g} × MSE '
            f'{entry.reference_mse:.6g} of {entry.statistic}'
        )
    environment = record.read.environment
    console.print(
        f'\nRead on {environment.platform}, Python {environment.python}, uv.lock '
        f'{environment.lock_sha256}: every candidate read must run there too'
    )
    console.print(f'Reads logged to {panel.log.path} (fingerprint {panel.fingerprint})')
    console.print(f'\nMargin record: {path}\n')

def _margin_kind(path: str) -> str:
    '''
    A margin record's kind: ``margins`` (Req 5, from a reference arm) or ``relative_margins``
    (Req 24, from the growth margin read).

    Raises:
        ValueError: If the file is not a margin record.
    '''

    data = json.loads(Path(path).read_text(encoding='utf-8'))
    kind = data.get('kind') if isinstance(data, dict) else None
    if kind not in ('margins', 'relative_margins'):
        raise ValueError(f'{path} is not a margin record: its kind is {kind!r}')
    return kind

@app.command('decide')
```

Replace:

```python
        typer.Option('--margins', help='The margin record (tools margins)'),
```

with:

```python
        typer.Option('--margins', help='The margin record (tools margins or tools growth-margins)'),
```

Replace:

```python
    Decide among two or more arms under Req 5's rule over D8's three panels.

    Each A-against-B comparison reads Δ on paired resamples of each panel's units, seeds nested.
    A is adopted over B when it is non-inferior on all three panels (the 95 % interval's lower
    bound above −δ) and superior on at least one (the 98⅓ % interval above zero). The survivors
    are the arms no other arm is adopted over, and the tie order picks among them. The record
    carries the arms with their selection-log records and artifact references, the margins, every
    comparison, the non-dominated set, the tie order and the chosen arm.
```

with:

```python
    Decide among arms under Req 5's rule: over D8's three panels with margins from tools margins,
    or over the two growth panels (Req 24) with margins from tools growth-margins.

    Each A-against-B comparison reads Δ on paired resamples of each panel's units, seeds nested.
    A is adopted over B when it is non-inferior on every panel (the 95 % interval's lower bound
    above −δ) and superior on at least one (the 1 − 0.05/n interval above zero: 98⅓ % over three
    panels, 97.5 % over two). The survivors are the arms no other arm is adopted over, and the
    tie order picks among them. The record carries the arms with their selection-log records and
    artifact references, the margins, every comparison, the non-dominated set, the tie order and
    the chosen arm.

    The economic decision reads economic arm records, one or more, and compares each candidate
    with the reference ``covariates+ancestors``. Only a candidate adopted over the reference is
    eligible for the rule among candidates; with none, the reference stands.
```

Replace:

```python
        record = decide(
            name,
            question,
            [read_record(path, ArmRecord) for path in arm],
            read_record(margins, MarginRecord),
            ArtifactStore(store),
            replicates=cfg.replicates,
            bootstrap_seed=cfg.bootstrap_seed,
            min_seeds=cfg.min_seeds,
        )
```

with:

```python
        if _margin_kind(margins) == 'relative_margins':
            record = decide_economic(
                name,
                question,
                [read_record(path, EconomicArmRecord) for path in arm],
                read_record(margins, RelativeMarginRecord),
                ArtifactStore(store),
                replicates=cfg.replicates,
                bootstrap_seed=cfg.bootstrap_seed,
            )
        else:
            record = decide(
                name,
                question,
                [read_record(path, ArmRecord) for path in arm],
                read_record(margins, MarginRecord),
                ArtifactStore(store),
                replicates=cfg.replicates,
                bootstrap_seed=cfg.bootstrap_seed,
                min_seeds=cfg.min_seeds,
            )
```

Replace:

```python
    console.print(f'\n[bold cyan]Decision {name!r}[/bold cyan]\n')
```

with:

```python
    console.print(f'\n[bold cyan]Decision {name!r}[/bold cyan]\n')
    superiority = f'{100 * record.settings.superiority_level:.4g} %'
```

Replace:

```python
                f'against −δ {-panel.margin:.4g}; 98⅓ % [{upper_low:+.4g}, {upper_high:+.4g}]'
            )
    cycle = ' (dominance cycled)' if record.cycle else ''
    console.print(f'\nNon-dominated: {", ".join(record.non_dominated)}{cycle}')
    console.print(f'Tie order: {", ".join(record.tie_order)}')
    console.print(f'[bold]Chosen: {record.chosen}[/bold]')
```

with:

```python
                f'against −δ {-panel.margin:.4g}; {superiority} [{upper_low:+.4g}, '
                f'{upper_high:+.4g}]'
            )
    reference = getattr(record, 'reference', None)
    if reference is not None:
        console.print(f'\nAdopted over the reference: {", ".join(record.eligible) or "none"}')
    cycle = ' (dominance cycled)' if record.cycle else ''
    console.print(f'\nNon-dominated: {", ".join(record.non_dominated)}{cycle}')
    console.print(f'Tie order: {", ".join(record.tie_order)}')
    chosen = record.chosen or f'none; the reference {reference} stands'
    console.print(f'[bold]Chosen: {chosen}[/bold]')
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_commands.py tests/unit/test_config.py -q`
Expected: PASS, `274 passed`.

Run: `COLUMNS=80 uv run pytest tests/unit/test_cli_commands.py -q`
Expected: PASS, `53 passed`. Rich wraps at 80 columns on CI, and the CLI tests match on
whitespace-collapsed output.

- [x] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/cli/commands/tools.py src/naics_embedder/utils/config.py tests/unit/test_cli_commands.py tests/unit/test_config.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3185 passed, 2 skipped`.

- [x] **Step 6: Commit, and compare with the dry run**

> Deviation: review round 1 added fix commit 11641b0 on top of 4baae4a; the suite then read 3189 passed, 2 skipped.

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add conf/data/decision.yaml conf/data/regressor_panel.yaml src/naics_embedder/cli/commands/tools.py src/naics_embedder/utils/config.py tests/unit/test_cli_commands.py tests/unit/test_config.py
git commit -m "feat(cli): fix the growth margins and decide the economic artifact (Req 24)" \
  -m "tools growth-margins reads the reference alone on both growth panels,
whatever the regressor config's outcome says, and writes a relative
margin record with r from the decision config (relative_margin, 0.05).
tools decide reads an economic decision when its margin record is
relative, and prints each decision's own superiority level and, when no
candidate is adopted, that the reference stands. tools regressor-panel
and tools sweep still read level mode only."
```

Run:

```bash
git diff --stat 92412753e1dda493084972e8bb64b333478393cf HEAD -- conf/data/decision.yaml conf/data/regressor_panel.yaml src/naics_embedder/cli/commands/tools.py src/naics_embedder/utils/config.py tests/unit/test_cli_commands.py tests/unit/test_config.py
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

### Task 9: Documentation

The usage guide, the API page, the overview pages, CLAUDE.md, the README and the test README
describe the growth margins and the economic decision, and state the new counts (P14).

**Files:**
- Modify: `CLAUDE.md`, lines 21, 83, 90, 238, 308, 481 and 493
- Modify: `README.md`, lines 131 and 164
- Modify: `docs/api/decision.md`, lines 4 and 30-31
- Modify: `docs/index.md`, lines 29-30
- Modify: `docs/overview.md`, line 171
- Modify: `docs/quickstart.md`, line 92
- Modify: `docs/text_training.md`, lines 324-325
- Modify: `docs/usage.md`, lines 219, 372-386 and 397-399
- Modify: `tests/README.md`, lines 5, 17 and 36

**Interfaces:**
- Consumes: the names Tasks 1–8 produce.
- Produces: documentation only. `docs/usage.md` gains the anchor `#tools-growth-margins`, and
  `docs/api/decision.md` renders `naics_embedder.decision.economic`.

- [x] **Step 1: Write the documentation**

In `docs/usage.md`, make three edits, in this order.

Replace:

```markdown
`growth` is refused before anything is read.
```

with:

```markdown
`growth` is refused before anything is read. The growth margin read is
[`tools growth-margins`](#tools-growth-margins)'s.
```

Replace:

````markdown
### `tools decide`

Decide among two or more arms under Req 5's rule over the three panels of roadmap D8: the outcome
panel and the regressor panel's seen and held-out regimes at level 6. Each comparison reads the
difference Δ on paired resamples of each panel's units (codes with their queries; four-digit
groups), with each arm's seeds resampled inside every replicate. A is adopted over B when it is
non-inferior on all three panels (the 95 % interval's lower bound above −δ) and superior on at
least one (the 98⅓ % interval above zero). The survivors are the arms no other arm is adopted
over, and the tie order picks among them: fewer components, then lower dimension, then
non-hyperbolic geometry, then the higher held-out gain over ancestors (roadmap D11). The number
of replicates, the bootstrap seed and the seed floor come from `conf/data/decision.yaml`.

**Generates:** the decision record (JSON): the arms with their selection-log records and artifact
references, the margins, every comparison with both intervals, the non-dominated set, the tie
order, the chosen arm, and each arm's reported gains over the sparse comparators
````

with:

````markdown
### `tools growth-margins`

Fix each growth panel's non-inferiority margin δ from a reference-only read (Req 24). The read
scores each growth regime's level-6 validation split, `growth_seen` and `growth_heldout`, with
only the comparators that read no candidate: the covariates, one-hot and ancestor indicators,
each alone and with the covariates. It computes neither `embedding` nor `text_only`, so no
candidate's growth error exists before δ is recorded. δ is `relative_margin` in
`conf/data/decision.yaml` (r = 0.05) times the validation MSE of the reference
`covariates+ancestors`. The panel reads growth whatever the regressor config's `outcome` says,
which stays `level` for `tools regressor-panel` and `tools sweep`.

Run it once, before any candidate is read on growth: a decision refuses a candidate read before
its margins were fixed. The record names the platform, Python and `uv.lock` sha256 the read ran
under. A decision pairs only bit-identical reference predictions, so every candidate read must
run on the same platform and lock.

**Generates:** the relative margin record (JSON), with the read's selection-log records, and its
predictions and scores in the store

```bash
uv run naics-embedder tools growth-margins --codebook PATH/naics_codebook.parquet \
  --name growth-margins --store ~/naics-artifacts \
  --output ~/naics-artifacts/records/economic/growth_margins.json
```

**Options:**
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`
- `--name TEXT` - Names the margins in the decision records that use them
- `--store PATH` - The artifact store for the read's predictions and scores
- `--output PATH` - Where to write the margin record; an existing file is refused before anything
  is read, never overwritten
- `--purpose TEXT` - Why the read happens, recorded in the selection log (default: economic
  margin read)
- `--log PATH` - Selection log (default: the regressor config's)
- `--lock PATH` - The `uv.lock` the read runs under, which the record names (default: `uv.lock`)

### `tools decide`

Decide among arms under Req 5's rule. With a margin record from `tools margins`, it decides among
two or more arms over the three panels of roadmap D8: the outcome panel and the regressor panel's
seen and held-out regimes at level 6. With one from `tools growth-margins`, it makes the economic
decision over the two growth panels (Req 24), below. Each comparison reads the difference Δ on
paired resamples of each panel's units (codes with their queries; four-digit groups), with each
arm's seeds resampled inside every replicate. A is adopted over B when it is non-inferior on
every panel (the 95 % interval's lower bound above −δ) and superior on at least one (the
1 − 0.05/n interval above zero, for n panels: 98⅓ % over D8's three, 97.5 % over the two growth
panels). The survivors are the arms no other arm is adopted over, and the tie order picks among
them: fewer components, then lower dimension, then non-hyperbolic geometry, then the higher
held-out gain over ancestors (roadmap D11). The number of replicates, the bootstrap seed and the
seed floor come from `conf/data/decision.yaml`.

**The economic decision.** Its arms are economic arm records: a deterministic candidate's table,
with the provenance its builder wrote, read once on each growth panel by
`naics_embedder.decision.economic.read_deterministic_arm`, which has no command yet. Each
candidate is a single replicate. A trained candidate's seeds have no economic record yet. Each
candidate is compared with the reference `covariates+ancestors`, whose predictions in the
margin read its read must reproduce bit for bit. Only a candidate adopted over the reference is
eligible, and the rule above picks among the eligible; its last tie goes to the higher held-out
gain over the reference. With none eligible, the reference stands, and there is no economic
artifact.

**Generates:** the decision record (JSON): the arms with their selection-log records and artifact
references, the margins, every comparison with both intervals, the non-dominated set, the tie
order, the chosen arm, and each arm's reported gains over the sparse comparators. The economic
decision record names its reference, lists the candidates adopted over it and each candidate's
MSE for every comparator, and chooses none when the reference stands
````

Replace:

````markdown
**Options:**
- `--arm PATH` - An arm record; repeat for each arm
- `--margins PATH` - The margin record (`tools margins`)
````

with:

````markdown
The economic decision, over one candidate:

```bash
uv run naics-embedder tools decide \
  --arm ~/naics-artifacts/records/economic/tree.json \
  --margins ~/naics-artifacts/records/economic/growth_margins.json \
  --name economic --question "Which candidate, if any, is the economic artifact?" \
  --store ~/naics-artifacts --output ~/naics-artifacts/records/economic/decision.json
```

**Options:**
- `--arm PATH` - An arm record, or for the economic decision an economic arm record; repeat for
  each arm
- `--margins PATH` - The margin record (`tools margins`, or `tools growth-margins` for the
  economic decision)
````

In `docs/api/decision.md`, make two edits, in this order.

Replace:

```markdown
(roadmap Stage 4).
```

with:

```markdown
(roadmap Stage 4); and the economic artifact's decision over the two growth panels, against the
reference `covariates+ancestors` (Req 24).
```

Replace:

```markdown
::: naics_embedder.decision.sweep

```

with:

```markdown
::: naics_embedder.decision.sweep

## The economic decision

::: naics_embedder.decision.economic

```

In `docs/overview.md`, make one edit.

Replace:

```markdown
## Implementation Reference
```

with:

```markdown
The economic artifact has a decision of its own on the two growth panels (Req 24).
[`tools growth-margins`](usage.md#tools-growth-margins) fixes each panel's δ as 0.05 times the
reference `covariates+ancestors`'s validation MSE, from a read that scores no candidate. Then
`tools decide` compares each candidate with that reference at 95 % non-inferiority and 97.5 %
superiority. Only a candidate adopted over the reference is eligible; with none, the reference
stands.

## Implementation Reference
```

In `docs/quickstart.md`, make one edit.

Replace:

```markdown
margins with `tools margins --multiple 3` before candidate reads. Panel test splits stay sealed.
```

with:

```markdown
margins with `tools margins --multiple 3` before candidate reads. Panel test splits stay sealed.
The economic artifact is decided on its own growth panels: fix their margins with
[`tools growth-margins`](usage.md#tools-growth-margins) before any candidate is read, then
compare candidates with the reference `covariates+ancestors` through `tools decide`.
```

In `docs/index.md`, make one edit.

Replace:

```markdown
and decisions use the three panels under Req 5. Structural metrics describe an arm through
`tools diagnostics`; they do not select the text checkpoint or schedule its learning rate.
```

with:

```markdown
and decisions use the three panels under Req 5. The economic artifact's decision reads its own
two growth panels against the reference `covariates+ancestors` (Req 24). Structural metrics
describe an arm through `tools diagnostics`; they do not select the text checkpoint or schedule
its learning rate.
```

In `docs/text_training.md`, make one edit.

Replace:

```markdown
bit-identical regressor rows, so the growth margin read and every economic candidate read share
one platform and one `uv.lock`.
```

with:

```markdown
bit-identical regressor rows, so the growth margin read
([`tools growth-margins`](usage.md#tools-growth-margins)) and every economic candidate read share
one platform and one `uv.lock`; the margin record names both.
```

In `CLAUDE.md`, make seven edits, in this order.

Replace:

```markdown
Python 3.12. The source tree has **118 Python files**; tests have **100 unit** files and two
```

with:

```markdown
Python 3.12. The source tree has **119 Python files**; tests have **101 unit** files and two
```

Replace:

```markdown
├── decision/                  # paired resampling, margins, arm records and decisions
```

with:

```markdown
├── decision/                  # paired resampling, margins, arm records and decisions;
│                              # economic.py: the growth margin read and economic decision
```

Replace:

```markdown
├── unit/                      # 100 unit test files
```

with:

```markdown
├── unit/                      # 101 unit test files
```

Replace:

```markdown
### Remote Workflow and Phase 2 Boundary
```

with:

```markdown
The economic decision (Req 24, `decision/economic.py`) reads `growth_seen` and `growth_heldout`
against the reference `covariates+ancestors`. `tools growth-margins` reads growth whatever the
config says, scoring only the comparators that read no candidate, and records δ = r × the
reference's validation MSE, with r = `relative_margin` (0.05) in `conf/data/decision.yaml`.
A deterministic candidate is one replicate, read by `read_deterministic_arm` (no command yet)
with its table's provenance and no checkpoint or monitor reads. `tools decide` makes the economic
decision from a relative margin record. It refuses a candidate whose reference predictions
differ from the margin read's bit for bit, or that read before the margins. Superiority is
1 − 0.05/n: 97.5 % over the two growth panels, 98⅓ % over D8's three. Only a candidate adopted
over the reference is eligible; with none, the reference stands. A trained candidate's seeds
have no economic record yet.

### Remote Workflow and Phase 2 Boundary
```

Replace:

```markdown
The current suite collects **3,123 tests**. Actual skip counts depend on local data and hardware
```

with:

```markdown
The current suite collects **3,187 tests**. Actual skip counts depend on local data and hardware
```

Replace:

```markdown
There are 100 unit files and two integration files. Important current seams include:
```

with:

```markdown
There are 101 unit files and two integration files. Important current seams include:
```

Replace:

```markdown
  growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22).
```

with:

```markdown
  growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22).
- `test_decision_economic.py`: the reference-only margin read and δ, deterministic candidates'
  records and pairing, eligibility, the tie order, margin-first refusals and the economic
  decision record (Req 24).
```

In `README.md`, make two edits, in this order.

Replace:

```markdown
## Graph Refinement and Structural Diagnostics
```

with:

```markdown
The economic artifact is decided on its own panels, `growth_seen` and `growth_heldout`, against
the reference `covariates+ancestors` (Req 24). `tools growth-margins` fixes each panel's δ from
a read that scores no candidate, before any candidate is read. `tools decide` then reads 97.5 %
superiority over the two panels, and only a candidate adopted over the reference is eligible.
See [`tools growth-margins`](docs/usage.md#tools-growth-margins).

## Graph Refinement and Structural Diagnostics
```

Replace:

```markdown
The suite collects 3,123 tests in 100 unit files and two integration files. Actual skip counts
```

with:

```markdown
The suite collects 3,187 tests in 101 unit files and two integration files. Actual skip counts
```

In `tests/README.md`, make three edits, in this order.

Replace:

```markdown
The suite contains **100 unit** test files and **2 integration** files, with **3,123 collected
```

with:

```markdown
The suite contains **101 unit** test files and **2 integration** files, with **3,187 collected
```

Replace:

```markdown
├── unit/                         # 100 unit test files
```

with:

```markdown
├── unit/                         # 101 unit test files
```

Replace:

```markdown
| Regressor panel | `test_regressor_panel.py`, `test_regressor_qcew_rows.py`: level mode pinned at d61f49b, the growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22) |
```

with:

```markdown
| Regressor panel | `test_regressor_panel.py`, `test_regressor_qcew_rows.py`: level mode pinned at d61f49b, the growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22) |
| Economic decision | `test_decision_economic.py`: the reference-only margin read and δ, deterministic candidates' records and bit-identical pairing, eligibility, the tie order, margin-first refusals and the decision record (Req 24) |
```

- [x] **Step 2: Check the stated counts and build the docs**

> Deviation: the collected total read 3,191, not 3,187, after 11641b0; the docs now state 3,194 after 0081110 and 08892d1.

Run: `uv run pytest --collect-only -q -q | tail -1`
Expected: `3187 tests collected`, followed by a time, matching the count the three files now
state.

Run: `git ls-files 'src/*.py' | wc -l`, then `git ls-files 'tests/unit/test_*.py' | wc -l`
Expected: 119, then 101, matching CLAUDE.md, whose counts exclude generated and ignored files.
At 0fdfc26 they were 118 and 100.

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

Run: `grep -c 'id="tools-growth-margins"' /tmp/naics_plan14_site/usage/index.html`, then
`grep -c 'decide_economic' /tmp/naics_plan14_site/api/decision/index.html`
Expected: each prints a count of at least 1: the anchor `docs/usage.md` links to renders, and so
does the economic API.

Run: `rm -rf /tmp/naics_plan14_site`

- [x] **Step 3: Commit, and compare with the dry run**

Commit, with the session's attribution trailer as the message's last `-m` paragraph:

```bash
git add CLAUDE.md README.md docs/api/decision.md docs/index.md docs/overview.md docs/quickstart.md docs/text_training.md docs/usage.md tests/README.md
git commit -m "docs: describe the growth margins and the economic decision (Req 24)"
```

Run:

```bash
git diff --stat 62b19d4e0a991b040a14eeb6666e9277906d4501 HEAD -- CLAUDE.md README.md docs/api/decision.md docs/index.md docs/overview.md docs/quickstart.md docs/text_training.md docs/usage.md tests/README.md
```
Expected: no output. The task's files are those of its dry-run commit (**Workspace**).

## Final verification (controller, inline)

Run every check before the final review, and paste each output into the ledger.

- [x] **Step 1: The suite under both CI versions**

> Deviation: 3189 passed, 2 skipped after 11641b0 and 3192 after the final-review fixes, under both Pythons; the 3.10 environment was built twice.

Run: `uv run pytest -n auto -q`
Expected: `3185 passed, 2 skipped`.

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan14_py310 uv run --locked pytest -n auto -q
```
Expected:
- `3185 passed, 2 skipped`, the same counts as under 3.12 (measured at plan time);
- a few hundred extra "encountered in matmul" RuntimeWarnings, from numpy 2.2 with Accelerate on
  3.10, which are not a failure.

Run: `rm -rf /tmp/naics_plan14_py310`

- [x] **Step 2: Lint, format and docs**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict -d /tmp/naics_plan14_site`, then `rm -rf /tmp/naics_plan14_site`
Expected: the build succeeds with no warning.

- [x] **Step 3: The branch**

> Deviation: twelve commits (nine tasks, 11641b0, 0081110, 08892d1), 29 files, 2658 insertions, 142 deletions; twelve files differ from the dry run, all in the ledger.

Run: `git log --oneline origin/main..HEAD`
Expected: the nine task commits, newest first, and no private commit:

```text
docs: describe the growth margins and the economic decision (Req 24)
feat(cli): fix the growth margins and decide the economic artifact (Req 24)
feat(decision): decide the economic artifact against covariates+ancestors (Req 24)
feat(decision): read a deterministic economic candidate as one replicate (Req 24; Req 23)
feat(decision): fix the economic margins from a reference-only growth read (Req 24)
feat(panels): read the regressor panel's reference comparators alone (Req 24)
fix(decision): hold imported monitor-record identities to their types
fix(decision): refuse a missing or naive read time by name before comparing it
feat(decision): set the superiority level by the decision's panel count (Req 24)
```

Run: `git diff --stat origin/main...HEAD | tail -1`
Expected: `29 files changed, 2596 insertions(+), 133 deletions(-)`.

Run: `git diff --name-only origin/main...HEAD -- uv.lock conf/config.yaml specs`
Expected: no output.

Run:

```bash
git diff --stat 62b19d4e0a991b040a14eeb6666e9277906d4501 HEAD -- src tests conf docs CLAUDE.md README.md
```
Expected: no output: the branch's code, tests, configs and docs are the plan-time dry run's.
A file no task touches can appear only if `origin/main` changed it after 0fdfc26; confirm with
`git log --oneline 0fdfc26..origin/main -- <file>`.

Run: `git diff origin/main...HEAD -- conf/data/regressor_panel.yaml`
Expected: one comment line replaced by two, and no other change. `outcome: level` and every pin
are unchanged.

- [x] **Step 4: The spec's Verification items**

Record in the ledger which test carries each of this stage's items:
- **Margin first, reference only:**
  - the margin read logs only comparators that read no candidate:
    `test_the_reference_read_logs_and_scores_only_comparators_that_read_no_arm`,
    `test_the_margin_read_logs_only_comparators_that_read_no_candidate` and
    `test_growth_margins_reads_the_reference_alone_and_writes_the_margins`;
  - the margin record's time precedes every candidate read, and `decide` refuses a margin record
    that does not: `test_a_candidate_read_before_the_margins_were_fixed_is_refused`,
    `test_margins_fixed_before_their_own_read_are_refused` and
    `test_margins_whose_read_was_not_reference_only_are_refused`.
- **Rule by panel count:**
  `test_superiority_splits_the_error_rate_evenly_over_a_decisions_panels`,
  `test_superiority_reads_the_level_the_decision_gives` and
  `test_the_record_carries_every_field_verification_lists` (D8's record); the CLI's "98.33 %" and
  "97.5 %" in `test_decide_adopts_the_better_arm_and_writes_the_record` and
  `test_decide_reads_an_economic_decision_from_relative_margins`.
- **Deterministic arms:** `test_a_single_replicate_record_validates_and_pairs_with_the_reference`,
  `test_a_deterministic_candidate_is_one_replicate_read_once_on_each_growth_panel` and
  `test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load`.
- **Economic decision record (on fixtures):**
  `test_the_economic_decision_record_shows_what_verification_lists`, with
  `test_with_no_candidate_adopted_the_reference_stands` for "or that the reference stands".
- **The roadmap Exit's named read-time error:** Task 2's
  `test_a_decision_read_without_an_aware_time_is_refused_by_name` and
  `test_a_monitor_read_without_an_aware_time_is_refused_by_name`.
- **Checks:** Steps 1 and 2.

- [x] **Step 5: The final review**

> Deviation: both seats found no Critical or Important; fixes 0081110 and 08892d1 landed after 'Codex reviewed ffa16af', and a scoped re-review covered them.

Dispatch the code-reviewer agent, which is pinned to Opus, on the whole branch, with:
- `git diff origin/main...HEAD`;
- this plan;
- the spec's Reqs 18, 23 and 24, and the roadmap's Stage 2 entry.

Also run Codex beside it, from the worktree root:

```bash
codex exec review --base origin/main -m gpt-6-astra
```

- `--base origin/main` makes it review the branch's commits. Without it, Codex reviews only
  uncommitted changes, and there are none at this point.
- `--base` takes no custom prompt: Codex refuses the combination.
- The config's default model fails for this account, so always pass `-m gpt-6-astra`, and never
  edit `~/.codex/config.toml`.

Then, for each finding:
- fix it, with a test where it is behavior, and rerun Steps 1–3; or
- triage it as deferred, which **Plan completion**'s gate handles.

A fix makes the branch differ from the dry run, so Step 3's comparison with `62b19d4` then
lists the fixed files; record each in the ledger.

## Plan completion (controller, inline)

Run this after **Final verification**, once the review findings are resolved, and before
**Integration**. It is writing-plans' Plan Completion Protocol, with this plan's edits written
out. Its commits land on `claude/stage-2-economic-decision`, so the PR carries them.

- [x] **Step 1: Check for parallel sessions**

`specs/naics-representation-split-roadmap.md` and `specs/deferred_items.md` are shared by every
session.

Run: `git worktree list`
- Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.
- If another worktree belongs to a running session, or the user has mentioned one, hold the
  shared edits (Steps 4 and 6) and give the user the exact text of each.

- [x] **Step 2: The resolve-before-defer gate**

> Deviation: the owner ruled four batched questions on 2026-10-13: fix the wording and two hardening one-liners now, defer the rest, and push and open the PR.

Collect the leftovers: plan steps skipped or descoped, and review findings not fixed. Partition
them:
- those that need the user's input: ask now, as one batched set of questions;
- those an answer unblocks: implement them now, then restart this protocol;
- everything else: defer.

The partition is final only once every question is answered. The trained-candidate item (O6) is
already decided: it is deferred, and Step 4 appends it.

- [x] **Step 3: Mark up this plan**

- Tick every completed step (`- [x]`).
- Under each step that deviated, add a one-line `> Deviation: …` note.
- Annotate each skipped step with `> Skipped: <why> → deferred`.
- Add the status line under the title: `**Status: COMPLETE (<date>)** — executed via <skill>;
  deferred items in specs/deferred_items.md`. O6 defers one item, so the line never reads
  "nothing deferred".

- [x] **Step 4: The deferred items**

> Deviation: the section also defers the stored-scores check, the predictions-to-table binding and a test-gap item (review Minors), beside the trained candidates.

In `specs/deferred_items.md`, tick the two items this plan implemented, in their `## 10-…`
section. Change each `- [ ]` to `- [x]`, and add a last line, indented like the item's other
lines:
- under "Imported monitor-record strictness":
  `→ done in plan 14 (Task 3: strict SeedRun identities; monitor reads matched by type).`
- under "Margin-timing error shape":
  `→ done in plan 14 (Task 2: read_time's named ValueError, before any comparison).`

Then append this section at the end of the file, with the gate's other deferred items, if any,
after its item:

```markdown
## 14-economic-decision-tooling — <date>
- [ ] Trained economic candidates (plan 14, deferred by the owner to the candidates' spec):
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
```

- [x] **Step 5: The handoff to later stages**

> Deviation: the Handoff adds the selection-log check before Stage 3, the BLAS suspect and D8's printed level, and says the owner's fixes changed the plan's code.

Add this section at the end of this plan, after **Integration**. It was drafted at plan time.
Check each bullet against what shipped: the deviation notes, the ledger and the final code.
Rewrite any bullet that execution changed, and keep a bullet for any interim name that execution
reverted, saying it is gone.

```markdown
## Handoff to later stages

Nothing was deferred into Stage 3. The trained-candidate item (O6) goes to the candidates' spec,
under `## 14-economic-decision-tooling` in `specs/deferred_items.md`.

- **Stage 3 (the growth margin read)** consumes, as built:
  - **The command.** `uv run naics-embedder tools growth-margins --codebook
    <bundle>/naics_codebook.parquet --name <name> --store <store> --output <record.json>`, with
    `--purpose`, `--log` and `--lock` optional (`cli/commands/tools.py`). It loads
    `conf/data/regressor_panel.yaml` at level 6 with `outcome` set to growth in memory. The
    committed config keeps `outcome: level`. r is `relative_margin` in `conf/data/decision.yaml`,
    0.05.
  - **Relative paths.** `--lock` defaults to `uv.lock`, and the regressor config's
    `selection_log` to `./logs/selection_log.jsonl`. Both resolve against the working
    directory, as `conf/` does.
    - Run from the main checkout's root, or pass `--lock` and `--log` as absolute paths.
    - Run from a worktree, the margin reads would be appended to that worktree's ignored log,
      which goes with the worktree, not to the real append-only log.
    - A missing lock is refused before any panel read.
  - **The record.** A `RelativeMarginRecord` (`decision/records.py`, kind `relative_margins`):
    `name`, `r`, `reference` (`covariates+ancestors`), `read`, `margins` (one `RelativeMargin`
    per growth panel: `panel`, `statistic`, `reference_mse`, `margin`) and `fixed_at`.
    - `read` is a `ReferenceRead`: `run_id` (`margins/<hex>`), `panels` (the draw, level 6's
      data fingerprint and the fit settings), `predictions`, `scores`, `log_records` (one per
      growth panel) and `environment` (platform, Python and the lock's sha256).
    - **Differs from Stage 3's Produces.** The command writes the record to `--output`, and the
      read's predictions and scores to the store. It neither stores the record itself nor prints
      its sha256, so Stage 3 records that (`shasum -a 256`).
  - **The selection log.** The read appends two validation reads, `growth_seen` and
    `growth_heldout`. Each has the detail `{'level': 6, 'comparators': [...], 'run':
    'margins/<hex>'}`, and the default purpose "economic margin read". They go to the log named
    by `--log`, or by the regressor config. The read opens no outer set.
  - **Traps.**
    - Nothing refuses a second `tools growth-margins` run. Each run appends two more reads, and
      a candidate read before a record's `fixed_at` is refused under that record (P12). Fix the
      margins once.
    - Every candidate read must run on the platform and lock of the margin read. `decide`
      pairs only bit-identical reference predictions (`check_reference_pairing`), and its
      refusal names both environments.
    - `tools regressor-panel`, `tools sweep` and `run_seed_sweep` still refuse growth.
  - **Merging into local main.** At plan time, the dry run's last commit merged cleanly into
    local main (5d19bc3), checked by `git merge-tree`. The held `chore(local)` commit edits
    `conf/data/regressor_panel.yaml`, as Task 8 does, in hunks that do not overlap.
  - No real data was read, no real split opened, and the main checkout's selection log was not
    touched in Stage 2.
```

- [x] **Step 6: The roadmap stamp**

In `specs/naics-representation-split-roadmap.md`, Stage 2's entry:
- tick it: `- [ ] Stage 2: Economic decision tooling` becomes
  `- [x] Stage 2: Economic decision tooling`;
- after its `ROUTING: writing-plans` line, add the stamp, in the roadmap's own form (its
  "Stage-spec stamp" section), indented like the entry's other lines:

```text
      > Stage 2: COMPLETE (<date>) — implemented by plan 14
      > (specs/plans/completed/14-economic-decision-tooling.md). Its Handoff section names what
      > Stage 3 consumes.
      > Next: resume the roadmap.
```

The stamp asks completion to re-validate later stages. That re-validation is the Handoff section
(Step 5); amending Stage 3's entry is the next roadmap resume's job.

- [x] **Step 7: Backlog triage**

Run:

```bash
uv run --no-project --python 3.13 python /Users/lowell/.claude/skills/writing-plans/scripts/deferred_stats.py
```
Report its summary line: the open count, the closure rate and the aged tail. With 20 or more open
items or any aged tail, run steps 1–4 of the skill's Triage rubric and present the proposal. It
is read-only; `/deferred` acts on a selection.

- [x] **Step 8: Retire the plan**

Run:

```bash
git mv specs/plans/14-economic-decision-tooling.md specs/plans/completed/14-economic-decision-tooling.md
```

This plan has no relative links to re-point. The spec stays in `specs/`: Stage 3 and the
candidates' and text→code specs cite it, and this plan implements one stage of it.

```bash
git add specs/plans/completed/14-economic-decision-tooling.md specs/naics-representation-split-roadmap.md specs/deferred_items.md
git commit -m "chore(specs): retire plan 14"
```

## Integration (controller, inline)

> Integration runs after this retirement commit; its outcome is recorded in the PR and the ledger.

This is finishing-a-development-branch, with this plan's PR written out.

- [ ] **Step 1: Open the PR, with the user's go-ahead**

Run: `git log --oneline origin/main..HEAD`
Expected: **Final verification** Step 3's nine commits, any review fixes, and
`chore(specs): retire plan 14`; no private commit.

Ask the user before pushing. Then run `git push -u origin claude/stage-2-economic-decision`,
never a bare `git push`, and open the PR against `main`. Its description carries:
- what the stage changes: the plan's Goal and Architecture, shortened;
- the recorded deviations, one line each, with the deferred trained candidates (O6) and D8's
  printed level flagged for the reviewer;
- the suite counts: `3185 passed, 2 skipped` locally under both Pythons, and the full suite's
  expected summary line in each CI test job, `3174 passed, 13 skipped`.

End the description with the session's PR attribution line.

- [ ] **Step 2: Wait for CI, then merge**

GitHub auto-merge is off, and main has no required checks.
- Watch `gh pr checks <number>` until the lint, docs and both test jobs pass.
- In each test job, the full suite's summary line should read `3174 passed, 13 skipped`, as on
  PR #142. The 47-test remote-transport step before it is unchanged. If a pairing or decision
  test fails there, stop and ask (**Stop-and-ask conditions**).
- The user merges. If the user asks you to merge, run
  `gh pr merge <number> --merge --match-head-commit <sha>`.

Before pushing a follow-up, check that the PR is still open (`gh pr view <number> --json state`).
If the user merged meanwhile, land the follow-up as a new PR from `origin/main`.

- [ ] **Step 3: Close the dry run's PR, with the user's go-ahead**

Run: `gh pr view 142 --json state,isDraft,title`
Expected: `OPEN`, a draft, titled "DO NOT MERGE: plan 14 dry run, Linux CI check (economic
decision tooling)". If it is closed or merged, record that in the ledger and skip this step.

Ask the user, then run `gh pr close 142 --comment "Superseded by #<number>."`. Delete its branch,
`claude/plan14-economic-decision-dryrun`, only if the user asks.

- [ ] **Step 4: Keep the ledger, then remove the worktree**

Removing the worktree deletes every ignored file in it. Keep the execution ledger first.

Run: `ls /Users/lowell/Projects/naics-embedder/logs/plan14_worktree`
Expected: "No such file or directory". If it exists, stop and ask.

Run: `mkdir /Users/lowell/Projects/naics-embedder/logs/plan14_worktree`, then
`cp -c logs/plan14_execution.md /Users/lowell/Projects/naics-embedder/logs/plan14_worktree/`

Then finish the branch: detach onto `origin/main`, delete the local branch, and remove the
worktree, or archive the app session that owns it.

## Handoff to later stages

Nothing was deferred into Stage 3. The trained-candidate item (O6) and the review Minors go to the
candidates' spec, under `## 14-economic-decision-tooling` in `specs/deferred_items.md`.

- **Stage 3 (the growth margin read)** consumes, as built:
  - **The command.** `uv run naics-embedder tools growth-margins --codebook
    <bundle>/naics_codebook.parquet --name <name> --store <store> --output <record.json>`, with
    `--purpose`, `--log` and `--lock` optional (`cli/commands/tools.py`). It loads
    `conf/data/regressor_panel.yaml` at level 6 with `outcome` set to growth in memory. The
    committed config keeps `outcome: level`. r is `relative_margin` in `conf/data/decision.yaml`,
    0.05; a non-positive or non-finite r is refused before anything is read.
  - **Relative paths.** `--lock` defaults to `uv.lock`, and the regressor config's
    `selection_log` to `./logs/selection_log.jsonl`. Both resolve against the working
    directory, as `conf/` does.
    - Run from the main checkout's root, or pass `--lock` and `--log` as absolute paths.
    - Run from a worktree, the margin reads would be appended to that worktree's ignored log,
      which goes with the worktree, not to the real append-only log.
    - A missing lock is refused before any panel read.
  - **The record.** A `RelativeMarginRecord` (`decision/records.py`, kind `relative_margins`):
    `name`, `r`, `reference` (`covariates+ancestors`), `read`, `margins` (one `RelativeMargin`
    per growth panel: `panel`, `statistic`, `reference_mse`, `margin`) and `fixed_at`.
    - `read` is a `ReferenceRead`: `run_id` (`margins/<hex>`), `panels` (the draw, level 6's
      data fingerprint and the fit settings), `predictions`, `scores`, `log_records` (one per
      growth panel) and `environment` (platform, Python and the lock's sha256).
    - **Differs from Stage 3's Produces.** The command writes the record to `--output`, and the
      read's predictions and scores to the store. It neither stores the record itself nor prints
      its sha256, so Stage 3 records that (`shasum -a 256`).
  - **The selection log.** The read appends two validation reads, `growth_seen` and
    `growth_heldout`. Each has the detail `{'level': 6, 'comparators': [...], 'run':
    'margins/<hex>'}`, and the default purpose "economic margin read". They go to the log named
    by `--log`, or by the regressor config. The read opens no outer set.
    - The seen regime logs `covariates`, `one_hot`, `covariates+one_hot`, `ancestors` and
      `covariates+ancestors`. The held-out regime has no one-hot comparator at level 6, so it
      logs `covariates`, `ancestors` and `covariates+ancestors`.
  - **Traps.**
    - Nothing refuses a second `tools growth-margins` run. Each run appends two more reads, and
      a candidate read before a record's `fixed_at` is refused under that record. Fix the
      margins once.
    - Margin first is enforced only for the reads inside the records `decide` is given; it does
      not scan the selection log. Before the first real growth read, check that the log holds no
      `growth_seen` or `growth_heldout` read. `tools regressor-panel` and `tools sweep` still
      refuse growth, and `read_deterministic_arm` needs a margin record.
    - Every candidate read must run on the platform and lock of the margin read. `decide`
      pairs only bit-identical reference predictions (`check_reference_pairing`), and its
      refusal names both environments. Bit identity was verified within one machine on fixtures.
      If a margin read and a candidate read on one machine fail to pair on real data, suspect
      BLAS threading on the larger matrices first. The check fails closed.
    - `tools regressor-panel`, `tools sweep` and `run_seed_sweep` still refuse growth.
  - **Merging into local main.** At the final tip (08892d1), the branch merges cleanly into local
    main (5d19bc3), checked by `git merge-tree`. The held `chore(local)` commit edits
    `conf/data/regressor_panel.yaml`, as Task 8 does, in hunks that do not overlap.
  - No real data was read, no real split opened, and the main checkout's selection log was not
    touched in Stage 2.
- **The candidates' spec** (not a roadmap stage) builds candidates and decides through these
  interfaces, and consumes, as built:
  - `read_deterministic_arm` is the Python API only, and `check_economic_arm` requires exactly
    one run with no seed. `check_reference_pairing` refuses an arm without exactly one run.
  - D8's printed level changed: `tools decide` prints the record's own level as
    `f'{100 * level:.4g} %'`, "98.33 %" where it printed "98⅓ %". The record's value is
    unchanged, `1.0 - 0.05 / 3`.
  - D8's decision-read identities still compare by value (`_check_log_records`, `decide.py`);
    the economic reads and the monitor records compare by type as well (`_same`).
  - The deferred items under `## 14-economic-decision-tooling`: trained economic candidates,
    the stored-scores check, the predictions-to-table binding and a test-gap item.
  - After the plan was written, the owner ruled two fixes that changed its code: a non-finite r
    is refused before any read (`11641b0`), and the last review's wording, a blank-run-id
    pattern and the run-less-arm refusal landed as `0081110` and `08892d1`.
