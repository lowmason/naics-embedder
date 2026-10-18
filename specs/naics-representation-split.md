# NAICS representation split — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: derive-roadmap — do not plan
> this spec directly and do not split it into per-subsystem plans.

**Status (2026-10-18).** Its roadmap is complete and retired
(`specs/completed/naics-representation-split-roadmap.md`, Completion audit). The economic
candidates and the text→code artifact come next, each in its own spec (Rollout note). The
candidates' spec carries four items: Verification's "Economic decision record" on real candidates
and "Outer sets closed" through the economic opening, Req 23's limit on how a candidate is built,
and Req 19's graph-stage question, due now that the first economic panel exists. Trained
candidates wait for it in `specs/deferred_items.md`, under `14-economic-decision-tooling`.

The single shared NAICS embedding is replaced by three artifacts over the same 2,125 NAICS 2022
codes. Each is judged only on evidence for its own use:

- the **hierarchy** is the tree itself, exact and untrained;
- **text→code** is a retrieval system, judged on the outcome panel alone and free of the
  low-dimension requirement;
- the **economic** artifact keeps the original purpose: low-dimensional coordinates that replace
  one-hot and other sparse encodings in downstream models, judged on economic panels.

This spec records the split as user adjudication Q3 and sets out how the roadmap's stages change.
It also specifies the economic artifact's first panel: national six-digit employment growth on the
committed regressor rows. A candidate must beat the tree's own ancestor indicators, under a margin
fixed before any candidate is read. `naics-embedding.md` is not edited; Supersessions lists every
requirement and decision this spec replaces.

**Design provenance.** A brainstorming session with the user on 2026-10-12, after a review of
Stage 8's adoption of euclidean-d16. The inputs read were:

- `naics-embedding.md` and its roadmap's decisions D1–D11;
- the findings `employment-statistics-coverage.md`, `outcome-panel-splits.md`,
  `regressor-panel-splits.md`, `reference-configuration.md` and `geometry-by-dimension.md`;
- the panel and decision code;

all at origin/main b5a0e8f.

- **Locators.**
  - (naics-embedding Req n) cites the original spec, and (roadmap Dn) a decision in its roadmap.
  - (finding: name §n) cites a finding's section, and (Un) a user ruling in the table below.
  - Checks run while writing are marked (verified locally: …).
  - Figures from the session's own re-analysis, held only in scratch files outside the
    repository, are marked (session analysis, not a recorded finding). No requirement depends on
    them.
- **Numbering.** Requirements continue the original spec's numbering from Req 18, so a requirement
  number is unique across both specs.

**User rulings (2026-10-12).** Free-text rulings are quoted as given. For a multiple-choice
question, the ruling is the chosen option's label.

| Id | Question | Ruling |
|---|---|---|
| U1 | (the user's own proposal) | "I wonder if we are trying to do too much with one set of embeddings. Would it make more sense to have separate codes for text-to-code, NAICS hierarchy, and downstream economic measurement?" |
| U2 | Which downstream economic analyses must the economic representation serve? | "1. Employment growth; 2. wages; 3. shock exposure; 4. nowcasting; 5. joint non-response + wage growth modeling; Order in terms of importance (1, 5, 2, 3, 4)" |
| U3 | Commit to the split, and which sub-project first? | "Split; panels first (Recommended)" |
| U4 | Does the umbrella design look right, and what happens to the graph stage? | "Approve; park graph stage (Recommended)" |
| U5 | Which data would the joint non-response + wage growth model use? | "No data exists..." |
| U6 | How far back should the employment-growth panel's history reach? | "Let's focus on employment growth initially, and just use NAICS 2022" |
| U7 | May the economic artifact learn from employment data, or only from text and the tree? | "Text and tree only (Recommended)" |
| U8 | Should the economic adoption rule also guard the existing next-year-levels regimes? | "Growth only (Recommended)" |
| U9 | What should the economic artifact have to beat? | "Ancestor indicators (Recommended)" |
| U10 | Which way should the growth panel be built? | "Retarget in place (Recommended)" |
| U11 | Does the records-and-scope section look right: one spec or two? | "Yes, but note there will be additional datasets. I am still putting them together but they are: OSHA ITA, IPUMS ACS, and IPUMS CPS. They will live at ~/Projects/estab-data." |
| U12 | Does the panel section look right: which covariates? | "Keep D1 covariates (Recommended)" |
| U13 | Does the decision section look right: which margin r? | "r = 0.05 (Recommended)" |
| U14 | Does the verification section look right? | "yes, write the spec" |
| U15 | (the user's correction, after reviewing the written spec) | "In the spec, Req 20.2 should read: 2. joint non-response + employment growth;" |
| U16 | Approve the written spec, including five additions made after the section approvals? They are: Req 24's eligibility rule; Req 24's reference-only margin read; Req 23's allowance for checkpoints selected on the outcome panel; δ's named statistic, component counting and Req 12's status; and Motivation's labeled session-analysis figures | "Spec + five additions approved" |

## Motivation — one space, three purposes

Stage 8 adopted euclidean-d16 under naics-embedding Req 5 (finding: geometry-by-dimension §3).
Reviewing that decision showed one space asked to serve three purposes, and specialized to none:

- **One panel carried the decision.**
  - euclidean-d16's mean validation outcome MRR is 0.592, against the reference's 0.413 (finding:
    geometry-by-dimension §2).
  - On the regressor regimes, Stage 7's margins (δ = 3 × the across-seed SD) were about 39% of the
    reference's mean validation MSE in both regimes: 0.0333 against 0.0848, and 0.0368 against
    0.0950 (verified locally: arithmetic on finding: reference-configuration §2 and finding:
    geometry-by-dimension §2). Non-inferiority there constrained little.
  - Against its strongest rivals, euclidean-d16's adoption rested on the outcome panel alone
    (session analysis, not a recorded finding).
- **Structure never entered the decision.** By design, agreement with the taxonomy is only a
  diagnostic (naics-embedding Reqs 1 and 6). The diagnostics split between the arms, and nothing
  selects on them (finding: geometry-by-dimension §6). A text-free embedding of the tree itself
  beat every trained arm on 6 of the 8 diagnostics (session analysis, not a recorded finding).
- **Each use has a stronger specialist.**
  - For decoding, the lexical stub's validation MRR of 0.6165 (finding: outcome-panel-splits §5)
    exceeds euclidean-d16's 0.592.
  - As regressors, the frozen text-only comparator (roadmap D9) had a lower point estimate of
    error than the trained embedding in all 18 arm × regime cases of Stage 8 (session analysis,
    not a recorded finding).

The user's conclusion is U1. Splitting lets each purpose be served, and judged, on its own terms.

## Core principle

Each artifact is judged only on evidence for its own use, and nothing is built from an outcome it
is judged on. The hierarchy needs no judging, because it is exact by construction.

Measurement comes before models (U3). The economic panel and its decision rule are fixed, and the
margin is recorded from the reference alone, before any economic candidate is built or read.

## Requirements

### The split

**Req 18 — Three artifacts (user adjudication Q3, 2026-10-12).** Q3 supersedes Q1's clause that one
embedding "must serve both as a regressor and as an outcome so it is not specialized" (U1, U3).

Q1's purpose carries over to the economic artifact: low-dimensional coordinates in place of sparse
encodings, "16 coordinates, for example, instead of 1,000+ indicators". All three artifacts share
the code universe, the bundle's 2,125 NAICS 2022 codes.

| Artifact | What it is | Judged on |
|---|---|---|
| Hierarchy | The bundle's codebook: codes, parents, ancestors and the tree metric (naics-embedding Req 7). Exact, with nothing trained | Nothing; it is exact by construction |
| Text→code | A retrieval system, free of the low-dimension requirement | The outcome panel only (naics-embedding Req 3) |
| Economic | Low-dimensional coordinates for the 2,125 codes | Economic panels only; the first is Req 21 |

- The hierarchy needs no new work. Its ancestor indicators are the economic reference (Req 24).
- Each artifact has its own reference, margins, deliverable and sealed opening. No artifact's
  adoption decision reads another artifact's panel.
- The text→code artifact's design belongs to its own spec (Out of scope).

**Req 19 — The roadmap's stages under the split (U3, U4).**

- **Stage 8** stays COMPLETE as a record. Its decision stands as history, but its winner no longer
  defines a deliverable.
- **Stage 9 is held.** Its backbone and fusion ablations (naics-embedding Req 14) move to the
  text→code artifact's spec.
- **The graph stage, Stages 10–11, is parked, not retired.** User adjudication Q2 stands.
  - Once the economic panels exist, its question is asked again: does propagating features over
    the tree, by smoothing or by the graph stage, help the economic artifact?
  - naics-embedding Reqs 15 and 17 wait with it.
- **Stage 12** becomes one sealed opening per artifact. Each opening is planned in writing before
  anything is opened.
- **Releases.** Stage 7's margins and the `uv.lock` freeze bounded decisions against the shared
  reference.
  - Both are released when this spec merges to main. Until then, the freeze holds.
  - Stage 7's margin record stays valid for the decisions that used it.

### The economic artifact

**Req 20 — Uses, and the first panel.** The economic artifact's uses, ranked by importance (U2,
with use 2 as corrected by U15), are:

1. employment growth;
2. joint non-response + employment growth;
3. wages;
4. shock exposure;
5. nowcasting.

How they are handled for now:

- The first panel, and for now the only one, is employment growth (U6). Each later use joins
  through its own panel spec, once its data are verified.
- Joint non-response + employment growth has no panel. Asked for its data source under its earlier
  name, joint non-response + wage growth, the user answered "No data exists..." (U5).
- Data stay on the NAICS 2022 basis (U6). Cross-vintage history stays Out of scope, as in
  naics-embedding.

**Req 21 — The growth panel retargets the regressor panel (U10, U12).** The regressor panel gains
one setting, `outcome`, either `level` or `growth`. `level` is the default and is today's panel.

- **Rows.** The committed ones (finding: regressor-panel-splits §1).
  - These are national, private, annual QCEW cells: 980 six-digit codes, each with feature years
    2022–2024, giving 2,940 rows. Levels 2–5 are kept for the multi-level variant.
  - They come from the same sha256-pinned files, and nothing new is downloaded.
- **Outcome.** log E(t+1) − log E(t), where E is annual-average employment. Every code in the
  population has positive employment in all four window years, so the difference is always
  defined.
- **Covariates.** roadmap D1, unchanged: log establishments and log total annual wages in year t
  (U12).
- **Unchanged:**
  - the partition and the committed held-out draw (60 of 300 four-digit groups);
  - both regimes and their folds;
  - the comparators (naics-embedding Req 2, with roadmap D9's text-only table);
  - ridge fitting with a nested-cross-validated penalty, and decision level 6.
- **The regimes, as they apply to growth.**
  - Seen regime: fit on the remainder's 2022→2023 growth and score its 2023→2024 growth. Its outer
    set is the 2024→2025 rows.
  - Held-out regime: repeated grouped folds with nested tuning inside the remainder. Its outer set
    is every row of the held-out groups' codes.
- **Read names.** Growth reads are logged as `growth_seen` and `growth_heldout`. That keeps them
  apart from the levels history, `regressor_seen` and `regressor_heldout`.
- **Levels unchanged.** With `outcome: level`, the rows, data fingerprints, folds and scores are
  bit-identical to today's, pinned on the fixture panel before any change. Real Stage 7–8 records
  keep verifying through their stored hashes; nothing re-reads real data to check them.

**Req 22 — One seal per split.** The outer sets are rows, and 2025 employment underlies both
outcomes' outer rows. An opening is therefore recorded against the split, not the outcome.

- **The key** is the regime plus `RegressorPanel.fingerprint`. That fingerprint names the held-out
  draw alone (`src/naics_embedder/panels/regressor.py:531`, `:594`). The row partition follows
  from the draw and the committed rows, which the data fingerprint pins.
- **The check.** `require_openable` consults both outcomes' read names under that key.
  - An opening of `regressor_seen` or `growth_seen` counts as an opening of the seen regime for
    both outcomes; likewise for the held-out regime.
  - A second opening needs a stated reason, as today.
- **Existing records.** Level mode's fingerprint inputs are unchanged, so existing selection-log
  records still match.
- **The levels regimes' outer sets** stay unopened (U8).

**Req 23 — Candidates read text and the tree only (U7).** naics-embedding Req 11's independence
holds for the economic artifact. No candidate's construction reads QCEW or any economic panel's
data.

- **What a candidate is.** A coordinate table for all 2,125 codes. Examples are a trained arm's
  export, the text-only table at the candidate's dimension (roadmap D9), an embedding of the tree,
  or a combination of these.
- **Selection.** No training term of a candidate reads any panel (naics-embedding Req 11). Every
  selection reads validation splits only (naics-embedding Req 4). That covers a trained
  candidate's checkpoint, which Stages 7–8 selected on the outcome panel, and the choice among
  candidates, which reads growth validation splits.
- **Candidates trained on employment data.** Such a candidate needs a time window that the panel
  never scores. With three growth years there is none, so it waits for a later reference year,
  once that year is final.

**Req 24 — The economic decision rule (U8, U9, U13).** naics-embedding Req 5 carries over:

- the paired two-stage bootstrap: 10,000 replicates with seed 20260924, resampled by four-digit
  group, with seeds nested inside;
- the non-dominated set;
- the tie order.

It changes as follows for the economic artifact.

- **Panels.** `growth_seen` and `growth_heldout`, which give two chances to adopt.
  - Non-inferior: the lower bound of the 95% interval on Δ exceeds −δ.
  - Superior: the lower bound of the 97.5% interval exceeds 0.
- **Statistic.** roadmap D10's, applied to growth: the level-6 MSE of `covariates+candidate`, with
  each row's squared error averaged over its repeats. Δ is the reference's error minus the
  candidate's, so a positive Δ favours the candidate.
- **Reference.** `covariates+ancestors` (U9). It reads no candidate, and the folds depend only on
  the regime, level, repeat and group. So its row errors are the same in every read, and they pair
  exactly with each candidate's.
- **Replicates.**
  - A trained candidate brings at least 5 seeds (naics-embedding Req 5).
  - A deterministic candidate, such as the text-only table or a tree embedding, is a single
    replicate.
- **Margin.** δ = 0.05 × the reference's validation MSE on that panel (U13), where that MSE is
  the statistic above computed for `covariates+ancestors`. δ is fixed and recorded from a
  reference-only read before any candidate is read.
  - The margin read scores only comparators that read no candidate table: `covariates`, `one_hot`,
    `ancestors` and their covariate pairs. It computes neither `embedding` nor `text_only`, so no
    candidate's growth error exists before δ is recorded.
  - Rejected: δ = k × the reference's SD across fold repeats. That is near 0 in the seen regime,
    where repeats only redraw the tuning folds, so non-inferiority would become superiority.
  - Rejected: δ = k × the bootstrap SD of the reference's MSE. It double-counts the resampling the
    interval already covers, and it is loose.
- **Eligibility and ties.**
  - Only a candidate adopted over the reference is eligible. Among those, the non-dominated set
    and naics-embedding Req 5's order pick one: fewer components, then lower dimension, then
    non-hyperbolic.
  - The candidates' spec counts each candidate's components.
  - roadmap D11's held-out estimate breaks any tie that remains. For growth, that is the higher
    held-out gain over the reference.
- **The economic deliverable** is the chosen candidate's 2,125-code table. If no candidate is
  eligible, the reference stands. There is then no economic artifact, and downstream models use
  the hierarchy's indicators.
- **Tooling.**
  - **Superiority level.** It follows the decision's panel count, 1 − 0.05/n; today it is
    hard-coded as `SUPERIORITY_LEVEL = 1.0 - 0.05 / 3` at `src/naics_embedder/decision/rule.py:27`.
    A three-panel decision still reads 98⅓%.
  - **Arm records** accept a deterministic single-replicate table, recorded with its provenance
    and without checkpoint or monitor reads.
  - **Margin records** gain the relative rule, naming r and the reference-only read.
  - **`decide`** accepts a comparator as the reference, and its margin-first check covers the
    reference-only read.

## Supersessions

| Superseded | Replaced by | Applies to |
|---|---|---|
| User adjudication Q1: one embedding "must serve both as a regressor and as an outcome so it is not specialized" | Q3: three artifacts (Req 18) | All |
| naics-embedding Req 1, two co-primary estimands | Each artifact judged on its own panels (Req 18) | All |
| naics-embedding Req 16, one 2,125-code table as the deliverable | One deliverable per artifact (Req 18) | All |
| naics-embedding Req 5's reference configuration and margin | `covariates+ancestors`, and δ = 0.05 × its MSE (Req 24) | Economic; text→code fixes its own in its spec |
| Roadmap D7, next-year log employment | Log growth from t to t+1 (Req 21) | Economic; `level` stays the default mode |
| Roadmap D8, three panels at 98⅓% | Two growth panels at 97.5% (Req 24); the outcome panel goes to text→code | Economic |
| Roadmap D10's sparse baselines: one-hot in the seen regime, ancestors in the held-out | `covariates+ancestors` in both regimes, with one-hot still reported (Req 24) | Economic |
| CLAUDE.md's `uv.lock` freeze through the last decision | Released when this spec merges (Req 19) | All |

These are held or parked, not superseded:

- naics-embedding Req 14 (Stage 9) is held, and its questions move to the text→code spec;
- Reqs 15 and 17 are parked with Stages 10–11, and user adjudication Q2 stands;
- Req 12 was carried out by Stage 8, and its record stands. Each artifact's own spec decides
  whether geometry and dimension remain factors for it.

Unchanged:

- naics-embedding Reqs 2–4 and 6–11, as they apply;
- the outcome panel and its sealed split;
- roadmap D1, D9 and D11;
- the selection log.

### Deferred items this touches

`specs/deferred_items.md` was read, not edited.

- **Waiting with the parked graph stage:**
  - the graph curriculum's degenerate difficulty thresholds;
  - the diagnostics report's interfaces "for Stage 10";
  - HGCN feeder precision;
  - HGCN metric mutation coverage.
- **Moving with Stage 9 to the text→code spec:**
  - nothing tying the bundle's backbone to training at runtime;
  - the tokenizer-name check deferred at plan 9's gate.
- **Bearing on Req 24's tooling:**
  - the margin-timing error shape in `check_margins_first`;
  - imported monitor-record strictness. A deterministic arm has no monitor records.

## Verification — observable outcomes

- [ ] **Split recorded.** The roadmap and CLAUDE.md name Q3, mark Stage 9 held and Stages 10–11
      parked, and describe Stage 12 as one opening per artifact. `naics-embedding.md` is
      unchanged.
- [ ] **Freeze released on merge.** This spec's branch does not change `uv.lock`. After the merge,
      CLAUDE.md's freeze instruction records its release.
- [ ] **Levels unchanged.** On the fixture panel, level mode's rows, data fingerprints, folds and
      scores equal the values pinned before the change.
- [ ] **Growth rows.** On the fixture panel:
  - each growth row's outcome equals log E(t+1) − log E(t) from the same cells;
  - its features are dated year t;
  - the partition and the held-out draw's fingerprint equal level mode's.
- [ ] **One seal per split.** After a regime is opened under one outcome, opening it under the
      other is refused without a stated reason. With a reason, the log records a reopening.
- [ ] **Margin first, reference only.**
  - The margin read logs only comparators that read no candidate.
  - The margin record's time precedes every candidate read.
  - `decide` refuses a margin record that does not.
- [ ] **Rule by panel count.** A fixture decision with two panels reads 97.5%, and one with three
      still reads 98⅓%.
- [ ] **Deterministic arms.** A single-replicate arm record with no checkpoint validates and pairs
      with the reference.
- [ ] **Economic decision record.** It shows:
  - two panels with their 95% non-inferiority and 97.5% superiority intervals;
  - δ = 0.05 × the reference's MSE per panel;
  - the reference `covariates+ancestors`;
  - each candidate's provenance, none naming an employment-data input;
  - the chosen arm, or that the reference stands.
- [ ] **Outer sets closed.** The selection log shows validation reads only. No outer set is opened
      before the economic artifact's own planned Stage 12 opening.
- [ ] **Checks.** The full suite, Ruff, the read-only YAPF check and the strict docs build pass.

## Planned sources (U11)

The user is assembling OSHA ITA, IPUMS ACS and IPUMS CPS in `~/Projects/estab-data`.

- This spec reads none of them.
- Each joins later through its own spec, once its coverage and coding are verified.
- Each is read in place under a sha256 pin, as QCEW is.
- IPUMS extracts are never copied into this public repository.

What is known about them so far:

- **OSHA ITA.** The case-detail file has, for each establishment, `naics_code`, `naics_year`, a
  free-text `industry_description`, `annual_average_employees` and `total_hours_worked`. For each
  case it has `job_description` and `SOC_code` (verified locally: the header of
  `~/Projects/estab-data/data/raw/osha/ITA_Case_Detail_Data_2025_through_08-31-2026.csv`).
  Independently labeled establishment descriptions, Out of scope in naics-embedding, could test
  text→code on real text.
- **IPUMS ACS.** The extract keeps the Census allocation flags (`QIND`, `QOCC`), and its core years
  are coded on 2022 NAICS (verified locally: the docstring of
  `~/Projects/estab-data/src/estab_data/acs_benchmark_extract.py`).
- **Census industry codes.** ACS and CPS code industry with Census industry codes, which are
  derived from NAICS but less detailed. A panel on them needs a mapping from each Census code to
  its set of NAICS codes.
- **CPS timing.** CPS has used the 2022 Census industry classification since January 2025
  (BLS CPS documentation as summarized in the bls-data-context reference; recheck it before
  building on it).

## Out of scope

- **The text→code artifact's design:** its reference, margins, backbone (Stage 9's questions) and
  any frozen or fused baselines. Its spec also takes up Stage 8's open training questions:
  - The winner's seeds were selected at epochs 38–39 of a 40-epoch budget.
  - Five arms stopped early: euclidean-d32, hyperbolic-d32 and spherical at every dimension. Each
    seed selected an epoch between 1 and 5 (finding: geometry-by-dimension §1).
- **Building the economic candidates.** That is the next spec. Reqs 23 and 24 state what any
  candidate must satisfy.
- **Uses 2–5 (U2).** Each joins through its own panel spec, and use 2 has no data (U5).
- **History and frequency.** Cross-vintage history and concordances (U6), and QCEW's monthly
  employment from its quarterly files.
- **Candidates trained on employment data (U7).**
- **The levels regimes as economic decision panels (U8).** They stay sealed.
- **The planned sources above.**

## Rollout note

Dependencies run in this order. The staging itself belongs to derive-roadmap.

1. **Record the split (Reqs 18–19).** Update the roadmap's status, mark Stage 9 held and Stages
   10–11 parked, make Stage 12 one opening per artifact, and update CLAUDE.md's overview and its
   freeze instruction. Docs only, and `naics-embedding.md` stays unedited.
2. **The growth outcome and the shared seal (Reqs 21–22),** tested on fixtures, with the levels
   values pinned first.
3. **The decision tooling (Req 24),** tested on fixtures.
4. **The margin read on real data (Req 24).** It runs on the Mac, is logged to the selection log,
   and needs the user's approval at the time.

Then the economic candidates (Req 23) and the text→code artifact each get their own spec.

This spec is not a stage of an existing roadmap, so it carries no roadmap stamp line.
