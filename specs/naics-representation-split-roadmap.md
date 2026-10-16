# NAICS representation split — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

**Status: APPROVED (2026-10-12); resumed twice, on 2026-10-12 and 2026-10-16.** Derived in an
interactive session. The user answered the derivation's two questions (Decisions, below) and
approved the three-stage partition at the human checkpoint. Stages 1 and 2 are complete, and the
resume after each re-validated the stages after it (below). Stage 3 is next, via writing-plans,
with the user's approval when it starts.

**Basis.** Source spec `specs/naics-representation-split.md` at 0a39042, unchanged through
origin/main 5ef7e7b (PR #137). Evidence was read at origin/main 5ef7e7b, and line citations are
to that commit unless a resume re-pointed them (below). Paths are under `src/naics_embedder/`
unless they start with `conf/`, `tests/`, `specs/` or `docs/`, or name a top-level file. "The
embedding roadmap" is `specs/naics-embedding-roadmap.md`: its stages are cited as "embedding
roadmap Stage N", and a bare "Stage N" is this roadmap's. U1–U16 are the spec's rulings and
D1–D11 the embedding roadmap's decisions.

**Decisions (2026-10-12).** Two questions the derivation asked, answered by the user.

- **Recording the split (Rollout note, item 1).** The embedding roadmap still routed its Stage 8
  (`:22`), whose completion no resume had reconciled, and plan 12 handed its notes on later stages
  to "the next roadmap resume". Decision: record the split as that roadmap's resume, made at this
  derivation and merged with this file, not as a stage here. Its "Resume after Stage 8, and the
  split" applies Req 19.
- **Where the release is recorded (Req 19).** The spec names CLAUDE.md. Decision: every live copy
  of the `uv.lock` freeze and of the one-shared-space overview, in `CLAUDE.md`, `README.md`,
  `docs/index.md`, `docs/text_training.md`, `docs/remote_workflow.md` and the embedding roadmap.

**Resume after Stage 1 (2026-10-12).** Stage 1's stamp is authoritative: plan 13, merged as PR
#139 (c2b381b) with its completion in PR #140 (d228132). Stages 2 and 3 were re-validated at
origin/main d228132, where the spec is still 0a39042's. Under `src/`, Stage 1 edited
`panels/qcew_rows.py`, `panels/regressor.py`, `panels/selection_log.py`, `decision/sweep.py`,
`cli/commands/tools.py` and `utils/config.py`; the other `decision` modules cited here,
`conf/data/decision.yaml` and `specs/deferred_items.md` are unchanged. Plan 13's Handoff
(`specs/plans/completed/13-growth-outcome-and-seals.md:3020`, the shipped section, not the draft
at `:2893` inside its completion step) names what both stages consume, and their entries now cite
it: Stage 2 lifts or routes around three paths that
refuse a growth panel, and Stage 3 lists the real log's existing openings. The read path's three
citations moved; they were re-pointed to d228132 in Stage 2's Consumes and the Req 24 row. Three
shipped differences that no Handoff bullet names were put to the user at the resume's Ask and
approved; each is marked "found at resume" in Stage 2's Consumes. They are `min_seeds`'s move to
`utils/config.py:599` (also re-pointed in the Req 23 row), the growth refusals that Stage 1 wrote
into `CLAUDE.md` and the panel config's comment, and the fixture panel's growth rows. Every
verdict stands, and the rows of Reqs 20–22, which Stage 1 closed, stay as entry-time evidence. No
panel read, selection-log read or opening ran during this resume. Stage 2 routes to
writing-plans; the next plan number is 14.

**Resume after Stage 2 (2026-10-16).** Stage 2's stamp is authoritative: plan 14, merged as PR
#144 (a7aa0c3). Stage 3 was re-validated at origin/main d4f8ddc, where the spec is still 0a39042's;
the line citations this resume adds are to d4f8ddc. Plan 14's Handoff
(`specs/plans/completed/14-economic-decision-tooling.md:6483`) is the shipped section, not the
draft at `:6334` inside its completion step, and Stage 3's entry now cites it.
From a7aa0c3 to d4f8ddc, nothing changed under `panels/` or `decision/`, nor in
`conf/data/regressor_panel.yaml` or `conf/data/decision.yaml`; `utils/config.py` changed, but in
no field those two configs declare. `cli/commands/tools.py` now applies `conf/local.yaml` to the
run config, which `tools growth-margins` never loads: it takes the codebook by `--codebook`. The
Handoff's "Merging into local main" bullet is superseded: local main has followed origin/main
since 2026-10-16, no held commit edits the panel config, and a66549c already re-pointed Stage 3's
Consumes at `qcew_dir`'s public default. Two shipped facts that no Handoff bullet names were put
to the user at the resume's Ask and approved; each is marked "found at resume" in Stage 3's
entry. They are `uv.lock`'s move after Stage 2 (639ce07, dev dependencies only) and the codebook
pin, whose comment in `conf/data/regressor_panel.yaml` still names bundle 18403d29. The rows of
Reqs 18, 23 and 24, which Stage 2 closed in part, stay as entry-time evidence. No panel read,
selection-log read or opening ran during this resume. Stage 3 routes to writing-plans; the next
plan number is 15.

## Gap analysis

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 18 | implemented-differently | `decision/scores.py:30` (`PANELS`: the outcome panel and both regressor regimes, D8); `decision/decide.py:196-200` (every run reads all three); `decision/records.py:92-112` (`PanelSet` requires the outcome panel); `specs/naics-embedding-roadmap.md:1001-1050` (embedding roadmap Stage 12: one final configuration and one table) | The spec has three artifacts, each with its own reference, margins, deliverable and opening; the code judges one shared table by one three-panel decision. The hierarchy exists as the bundle's codebook. The record was made at derivation (Decisions); the economic decision is Stage 2's, and the text→code half is its own spec's (Out of scope). |
| 19 | missing | `specs/naics-embedding-roadmap.md:22` (routes its Stage 8, stamped COMPLETE at `:902`), `:905-1050` (Stages 9–12 live), `:208-211` (the freeze through Stages 8–10); `CLAUDE.md:5`, `:243`; `README.md:8`; `docs/index.md:3`; `docs/text_training.md:316`; `docs/remote_workflow.md:196` | The freeze ended when PR #137 merged, but every live statement still held it; `uv.lock` still hashes `4167042e…`. Discharged at derivation (Decisions). |
| 20 | missing | `panels/qcew_rows.py:218-219` (the only outcome, log E(t+1)) | Its one build is Req 21's panel. The ranking, U5's answer and the NAICS 2022 basis need no code: the panel reads the bundle's NAICS 2022 codebook. |
| 21 | missing | `utils/config.py:532-583` (`RegressorPanelConfig` has no `outcome` setting); `panels/qcew_rows.py:202-225` (rows carry log E(t+1) only); `panels/regressor.py:78` (read names `regressor_seen` and `regressor_heldout` only) | What it keeps exists: the committed rows (finding: regressor-panel-splits §1), the draw `deddfd4c…`, the regimes and folds (`panels/regressor.py:146-160`, `:635-660`) and ridge with a nested penalty (`:484-499`). The population already requires positive employment in all four years (`panels/qcew_rows.py:186-200`). No test pins level mode's values (`tests/unit/test_regressor_panel.py`). |
| 22 | missing | `panels/regressor.py:662-681` (`require_openable` counts the regime's level read name only), `:531` (`fingerprint`, the draw); `panels/selection_log.py:84-91` (`openings` keys on read name and fingerprint) | An opening under one outcome's name would not count against the other's. The key's fingerprint is unchanged, so existing records keep matching. |
| 23 | missing | `decision/records.py:136-139` (every seed names a checkpoint); `decision/decide.py:85`, `:172-179`, `utils/config.py:599` at d228132 (at least 5 seeds, with no lower setting); `decision/records.py:69-90` (`ArmSpec` names a trained arm's text inputs) | It holds for trained arms: `decide` admits validation reads only (`decision/decide.py:196-232`). Missing: a record that shows a deterministic candidate's provenance. Building candidates is the next spec's (Out of scope). |
| 24 | missing | `decision/rule.py:26-27` (98⅓ %, fixed); `decision/scores.py:30-44` (D8's panels; one-hot and ancestors as the sparse baselines); `decision/records.py:165-195`, `decision/decide.py:387-429` (δ from a reference arm's across-seed SD); `decision/decide.py:344-384` (margins and the margin-first check over D8's panels and arms' reads); `panels/regressor.py:163-177`, `:670-695`, `:799-813` at d228132 (every read takes an arm and scores `embedding` and `text_only`) | No reference-only read exists. Unchanged: the paired two-stage bootstrap (`decision/resampling.py`; `conf/data/decision.yaml`), the non-dominated set and the tie order (`decision/rule.py:60-105`). Two deferred entries bear on it (`specs/deferred_items.md:529-541`). |

No in-code-but-not-in-spec row: none found in `panels/regressor.py`, `panels/qcew_rows.py`,
`panels/selection_log.py` or `decision/*` beyond what Reqs 21–24, and the naics-embedding
requirements they keep, cover.

## Stages

**Sequencing.** The Rollout note's items 2, 3 and 4 are Stages 1, 2 and 3; item 1 was done at
derivation (Decisions). The order follows the spec's own, which is also dependency order: Stage 2's
reference-only read and decision read Stage 1's growth reads, and Stage 3 runs Stage 2's read on
real data. It also agrees with the spec's priority, measurement before models (U3), since the
margin is on record before any candidate exists. After Stage 3, the economic candidates (Req 23)
and the text→code artifact each get their own spec (Rollout note); neither is a stage here. Plan
numbers start at 13.

**Deferred items.** `specs/deferred_items.md` was read, not edited. Stage 2 discharges the
margin-timing error shape, since `check_margins_first` is the check Req 24 extends. Its plan
discharges imported monitor-record strictness or records why not: a deterministic arm carries no
monitor records. The spec's other named entries went with embedding roadmap Stages 9–11 at that
roadmap's resume. The ridge path's SVD cache stays standalone, because one reference-only read does
not trigger it, though the candidates' sweeps may. The tie order's strictness keeps its trigger
for the candidates' decision.

- [x] Stage 1: Growth outcome and one seal per split
      Objective: Give the regressor panel a growth outcome on the committed rows, read under its
      own names, and make an opening of a regime's split count for both outcomes, with level
      mode pinned before any change.
      Spec: Req 21; Req 22; Req 20 (the first panel); Verification "Levels unchanged", "Growth
      rows", "One seal per split", "Checks"; Rollout note, item 2; Supersessions (D7); D1.
      Gap closed: Req 21; Req 22; Req 20.
      Consumes: Nothing from a prior stage. The regressor panel as embedding roadmap Stage 3 built
      it (`panels/regressor.py`, `panels/qcew_rows.py`, `panels/regressor_splits.py`; finding:
      regressor-panel-splits §§1, 3 and 6), with its committed draw (`deddfd4c…`) and fixture
      panel (`tests/fixtures/regressor_panel.py`); `SelectionLog.openings`, which keys an opening
      on its read name and fingerprint (`panels/selection_log.py:84-91`). A config key is declared
      in `RegressorPanelConfig` (`utils/config.py`), as every stage's is.
      Produces: The `outcome` setting, `level` by default; growth reads that return today's
      per-row prediction columns for every comparator, logged as `growth_seen` and
      `growth_heldout` under the draw's fingerprint; the growth rows' data fingerprint;
      `require_openable` counting a regime's openings under both outcomes' read names; level
      mode's values pinned by test.
      Exit: On the fixture panel, level mode's rows, data fingerprints, folds and scores equal the
      values pinned before the change (test); each growth row's outcome is log E(t+1) − log E(t)
      from the same cells, its features are dated year t, and its partition and draw fingerprint
      equal level mode's (test); after a regime opens under one outcome, opening it under the
      other is refused without a reason and logged as a reopening with one (test); Verification
      "Checks" passes.
      ROUTING: writing-plans
      > Stage 1: COMPLETE (2026-10-12) — implemented by plan 13
      > (specs/plans/completed/13-growth-outcome-and-seals.md). Its Handoff section names what
      > Stages 2 and 3 consume.
      > Next: resume the roadmap.

- [x] Stage 2: Economic decision tooling
      Objective: Build Req 24's rule for the economic artifact and exercise it on fixtures: the
      reference-only margin read and relative margin, deterministic single-replicate arms, a
      comparator as the reference, and a superiority level set by the panel count.
      Spec: Req 24 (all but the real margin read); Req 23 (what a candidate is; selection on
      validation splits; provenance in the record); Req 18 (the economic decision reads its own
      panels only); Verification "Margin first, reference only", "Rule by panel count",
      "Deterministic arms", "Economic decision record" (on fixtures), "Checks"; Supersessions (D8,
      D10's sparse baselines); Deferred items this touches; D10; D11.
      Gap closed: Req 24 (the tooling); Req 23 (the record); Req 18 (the economic decision).
      Consumes: Stage 1's growth reads, read names, data fingerprint and seal, through the
      interfaces and traps plan 13's Handoff names (plan 13, Handoff:
      `specs/plans/completed/13-growth-outcome-and-seals.md:3020`; Resolved at resume,
      2026-10-12). The panel's read path, which takes an arm and scores `embedding` and
      `text_only` in every read (`panels/regressor.py:163-177`, `:670-695`, `:799-813` at
      d228132; plan 13, Handoff): the reference-only read is this stage's.
      `tools regressor-panel`, `tools sweep` and `run_seed_sweep` refuse a growth panel, so this
      stage lifts or routes around them (plan 13, Handoff; Resolved at resume, 2026-10-12). The
      fixture panel's growth rows (`regressor_growth_rows` in `tests/fixtures/regressor_panel.py`;
      found at resume; Resolved at resume, 2026-10-12). The decision package, wired to D8's three
      panels: `PANELS` (`decision/scores.py:30`), the per-run read check and `check_margins`
      (`decision/decide.py:196-232`, `:344-358`), `PanelSet`'s required outcome fields and
      `SeedRun`'s required checkpoint (`decision/records.py:92-148`), a margin reference that must
      be an arm (`:174-182`), the 5-seed floor (`decision/decide.py:85`; `utils/config.py:599` at
      d228132, found at resume; Resolved at resume, 2026-10-12), and `tools margins` and
      `tools decide`. Pairing only on bit-identical regressor rows (embedding roadmap Stage 4,
      Rollout note). The deferred entries on margin timing and imported monitor records
      (`specs/deferred_items.md:529-541`). The docs that describe today's three-panel decision
      (`docs/usage.md`, `docs/overview.md`, `docs/quickstart.md`, `docs/index.md`, `README.md`),
      and the level-only growth refusals that `docs/usage.md`, `CLAUDE.md` and the comment in
      `conf/data/regressor_panel.yaml` state (found at resume; Resolved at resume, 2026-10-12).
      Produces: A reference-only growth read that computes neither `embedding` nor `text_only`; a
      margin record of the relative rule, naming r and that read; arm records for deterministic
      single-replicate tables, with provenance and no checkpoint or monitor reads; an economic
      decision over `growth_seen` and `growth_heldout` against `covariates+ancestors`, in which
      only candidates adopted over the reference are eligible; a superiority level of
      1 − 0.05/n, which keeps the three-panel decision at 98⅓ %. The candidates' spec decides
      through these.
      Exit: On fixtures: the margin read logs only comparators that read no candidate, its
      record's time precedes every candidate read, and `decide` refuses a margin record that does
      not (tests); a two-panel decision reads 97.5 % and a three-panel one 98⅓ % (test); a
      single-replicate arm record with no checkpoint validates and pairs with the reference
      (test); a fixture economic decision record shows every field Verification "Economic
      decision record" lists (test); `check_margins_first` refuses a missing or naive read time
      with a named error (test); Verification "Checks" passes.
      ROUTING: writing-plans
      > Stage 2: COMPLETE (2026-10-13) — implemented by plan 14
      > (specs/plans/completed/14-economic-decision-tooling.md). Its Handoff section names what
      > Stage 3 consumes.
      > Next: resume the roadmap.

- [ ] Stage 3: The growth margin read
      Objective: With the user's approval at the time, run the reference-only growth read on real
      data on the Mac and record δ for both growth panels before any candidate exists.
      Spec: Req 24 (Margin); Verification "Margin first, reference only", "Outer sets closed";
      Rollout note, item 4; U13.
      Gap closed: Req 24 (the recorded margin).
      Consumes: Stage 2's reference-only read and margin record, and Stage 1's growth panel, merged
      to origin/main and synced into local main, where the read runs. `qcew_dir` keeps its public
      default `~/Downloads/Data/QCEW`, where the read finds the QCEW slices and verifies each
      against `qcew_sha256` in `conf/data/regressor_panel.yaml`; no held commit points it there
      (audit row 1, 2026-10-16). That file keeps `outcome: level`, and no real growth read or
      opening has happened (plan 13, Handoff:
      `specs/plans/completed/13-growth-outcome-and-seals.md:3020`; Resolved at resume,
      2026-10-12). The read is `tools growth-margins`, run from the main checkout's root or with
      absolute `--lock` and `--log`: both resolve against the working directory, and a worktree's
      log goes with the worktree (plan 14, Handoff:
      `specs/plans/completed/14-economic-decision-tooling.md:6483`; Resolved at resume,
      2026-10-16). Its `--codebook` is bundle 301cce28's, whose codes keep the pinned
      `codebook_codes_sha256` (`specs/naics-embedding-roadmap.md:116-118`), though the pin's
      comment still names bundle 18403d29 (found at resume; Resolved at resume, 2026-10-16). The
      main checkout's selection log (`logs/selection_log.jsonl`, gitignored and append-only), which
      this stage is the first to read: an opening of either level regime under `deddfd4c…` already
      seals that regime's growth split, and `SelectionLog.openings(seal_names(regime), fingerprint)`
      lists any (plan 13, Handoff; Resolved at resume, 2026-10-12); margin first is not checked
      against the log, so it must hold no `growth_seen` or `growth_heldout` read before this one
      (plan 14, Handoff; Resolved at resume, 2026-10-16). An artifact store outside any worktree,
      as embedding roadmap Stages 7–8 kept theirs; the user's approval, asked when the stage starts.
      Produces: The economic margin record, holding δ for `growth_seen` and `growth_heldout` and,
      in its read's environment, the platform, Python and `uv.lock` sha256 it was read under.
      Every candidate read must share them, since `decide` pairs only bit-identical reference
      predictions; bit identity was verified on fixtures only, and BLAS threading is the first
      suspect if real reads fail to pair (plan 14, Handoff; Resolved at resume, 2026-10-16). The
      command writes the record once, to `--output`, and the read's predictions and scores to
      `--store`; it neither stores the record nor prints its sha256, so this stage stores it
      outside any worktree and records its sha256 with `shasum -a 256` (plan 14, Handoff;
      Resolved at resume, 2026-10-16). A written finding that records it with the read's
      selection-log records. The candidates' spec reads both.
      Exit: The margin record exists with r = 0.05 and one δ per growth panel; the selection log
      gained only the two growth validation reads of one run, which log the comparators plan
      14's Handoff lists per regime and neither `embedding` nor `text_only`; nothing refuses a
      second run, so the margins are fixed once (plan 14, Handoff; Resolved at resume,
      2026-10-16); no outer set was opened; the finding cites the record's sha256, the platform
      and the lock, and the origin/main commit the read ran at, from which a candidate can
      rebuild that lock after `uv.lock` moves, as it did after Stage 2 (639ce07; found at resume;
      Resolved at resume, 2026-10-16), states that the log held no growth read before the margin
      read (plan 14, Handoff; Resolved at resume, 2026-10-16), and lists the existing openings
      that check found, or says it found none (plan 13, Handoff; Resolved at resume, 2026-10-12).
      ROUTING: writing-plans

## Stage-spec stamp

No stage here has a stage spec, so each plan's header carries this line, copied verbatim:

> Roadmap: specs/naics-representation-split-roadmap.md, Stage N — on plan completion, tick the
> stage and re-validate later stages against what shipped.

On completion the stamp goes under the stage's entry, and it is authoritative:

> Stage N: COMPLETE (YYYY-MM-DD) — implemented by plan <id> (path). Its Handoff section names
> what <stages> consume.
> Next: resume the roadmap.

## Completion

Retirement is gated behind a conformance audit of the accumulated system: re-run the gap rubric
over Reqs 18–24 with evidence per verdict (implementing stage and plan, Deviation notes,
deferred-items entries), and re-check the spec's Verification items. Three are observable only
once candidates exist, so the candidates' spec carries them and they do not keep this roadmap
live: "Economic decision record" on real candidates, Req 23's limit on how a candidate is built,
and "Outer sets closed" up to the economic artifact's opening (embedding roadmap Stage 12). An
unmet requirement exits exactly two ways: a new stage, with the roadmap staying live, or a
conscious deferral with a written why. This is not a whole-roadmap code review; each stage was
reviewed when it merged. Parking is a first-class exit: remaining stages append to
`specs/deferred_items.md` as self-contained items, and this file moves to `specs/completed/`
marked `PARKED (date) — N of 3 stages complete`. The spec stays in `specs/` while the candidates'
and text→code specs cite it.
