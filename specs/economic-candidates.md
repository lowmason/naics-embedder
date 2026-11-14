# Economic candidates — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: writing-plans — plan this spec as plan 16 in a
> fresh session. Do not read a real growth panel while planning or executing it.

**Status: DRAFT (2026-11-14).** Written in a brainstorming session and awaiting the user's
approval. A builder check found nothing left for the user. Its four notes are folded into §4.4,
§4.5, §4.8 and §4.1.

**Roadmap:** `specs/naics-artifact-redesign-roadmap.md`, Stage 1 (ROUTING: brainstorming). Source
specs: `specs/naics-artifact-redesign.md` (Q4) Req 29, and `specs/naics-representation-split.md`
(Q3) Reqs 19, 23 and 24. Evidence was read at origin/main b4f8c9d. Paths are under
`src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/` or `docs/`, or name a
top-level file. `~/naics-artifacts` is the Mac's artifact store.

**Next skill after approval:** `writing-plans` (plan 16) in a fresh session. Stage 3 then reads
the candidates on the Mac with the user's approval at the time.

**Design provenance.** A brainstorming session with the user on 2026-11-14. The inputs read were
the two source specs, ADRs 0001 and 0002, the finding `growth-margins.md`, plan 14's Deviations
P6–P10 and its Handoff, `specs/deferred_items.md` under `14-economic-decision-tooling`,
`decision/economic.py`, `decision/records.py`, `decision/decide.py`, `decision/store.py`,
`panels/regressor.py`, `panels/text_only.py`, `utils/naics_hierarchy.py`, and the Stage 7 and
Stage 8 records in `~/naics-artifacts/records/`.

- **Locators.** (Req n) cites Q4's spec for n ≥ 25. (split Req n) cites Q3's. (Un) cites a
  ruling below or in those specs. (finding: growth-margins §n) cites that finding. Checks run
  while writing are marked (verified locally: …).
- **My calls.** Choices I made while writing, which the user has not ruled on, are marked
  (my call).

**User rulings (2026-11-14).** Rulings continue from U31, so each number is unique across the
three specs.

| Id | Question | Ruling |
|---|---|---|
| U32 | Which candidates should Stage 1 cover: deterministic only, or deterministic plus the 50 existing trained tables? | "Deterministic + trained" |
| U33 | Does this candidate list look right? It was played back as five proposals. (1) 9 deterministic candidates: text-only, tree and tree-smoothed text-only, each at d = 8, 16 and 32, with smoothing weights fixed in the spec and the graph stage left parked. (2) 9 trained arms read as they are, with their Stage 7/8 selection evidence. (3) The list frozen in the spec. (4) The tree candidate naming MiniLM's text-only table for the reported comparator. (5) A provenance rule that refuses QCEW or any panel | "Looks right" |
| U34 | How should the trained arms enter: import the Stage 7/8 records, or read the run directories again? | "Import the records" |

## 1. Purpose

The economic artifact replaces one-hot and other sparse codes in downstream models with a few
coordinates per code. The user's test is whether such coordinates "give a better prediction on
economic panels" than one-hot encoding (U22). The split spec implements that test on the growth
panel against the tree's ancestor indicators, `covariates+ancestors`, under margins fixed on
2026-10-17 (split Req 24; finding: growth-margins §2).

No candidate exists yet. This stage builds the candidates, freezes their list, adds a command that
reads one candidate on the growth validation splits, and extends the economic decision to trained
candidates. Everything is tested on fixtures. Stage 3 runs the real reads and the decision.

Every candidate reads only the codebook's text and its tree (U23; split Req 23). The stage also
answers split Req 19's question for this artifact: does propagating features over the tree help?

## 2. Scope

### 2.1 In scope

- The 18 candidates of §4.1: their builders, their tables and their frozen list.
- Trained candidates in the economic records and the decision (§4.4), which closes the deferred
  trained-candidate item.
- A provenance rule that enforces text-and-tree inputs (§4.5).
- One read command for both kinds of candidate, with a dry run (§4.6).
- The propagation answer's rule (§4.7).
- Two more deferred items whose functions this stage rewrites (§7).

### 2.2 Out of scope

- **Any real growth read, and the decision itself.** Both are Stage 3's.
- **New training.** It would need a Lambda campaign and the public native qualification
  (`specs/deferred_items.md:560`). The trained candidates are read as they are.
- **The graph stage.** HGCN stays parked with embedding roadmap Stages 10–11 (Req 27). Smoothing
  measures propagation here (§4.7).
- **The relatedness artifact as a candidate** (U23; Req 29, Later).
- **Candidates trained on employment data** (split Req 23).
- **The margins.** They stand. `tools growth-margins` never runs again (finding: growth-margins
  §4).
- **The economic sealed opening.** It is embedding roadmap Stage 12's.
- **Later panels** (P2; Req 29).

## 3. Approaches

Two choices shaped this design. The user made both.

**Which candidates (U32).**

- *Deterministic only.* The text-only table, a tree embedding and a tree-smoothed text-only table,
  each at three dimensions. This is the smallest build: it wraps `read_deterministic_arm` in a
  command. It leaves the 50 trained tables unread on growth, and so it leaves the project's main
  models out of the economic decision.
- *Deterministic plus trained (chosen).* It adds the trained-candidate records and checks, which
  plan 14 deferred. The trained tables need no new training. The decision loop already resamples
  seeds nested inside groups (`decide_economic`, `seed_draws`), so most of the new code is in the
  records and checks.
- One reason against trained candidates was weighed and not taken. Q3's session analysis had the
  frozen text-only table beat every trained arm as a regressor. That analysis is not a recorded
  finding, and it read levels, not growth.

**How the trained arms enter (U34).**

- *Import the Stage 7 and Stage 8 records (chosen).* The evidence for each seed was verified when
  those records were written. The tables and checkpoints are in the store under their sha256. The
  limit is that a trained candidate must first have such a record. No new training is planned,
  so the limit costs nothing now.
- *Read the run directories again, as `tools sweep` does.* This is the general path. It needs more
  code, it repeats Stage 8's checks, and it must not log new outcome-panel reads.
- *Rejected: each seed as its own deterministic candidate.* It breaks the 5-seed floor and the
  nested seed resampling (split Req 24), and it would make 50 candidates of 9.

## 4. Design

### 4.1 The frozen candidate list (U33)

The 18 candidates are fixed here. The list is committed as `conf/data/economic_candidates.yaml`
(my call), and the decision requires exactly its candidates (§4.4). The choice among candidates
and the decision read the same growth validation splits. A candidate added after Stage 3 saw a
result would make that choice unfair, so the list closes before the first read.

| Name | Kind | d | Components | Geometry | Source |
|---|---|---|---|---|---|
| `text-only-d8`, `-d16`, `-d32` | text-only | 8, 16, 32 | 1 | euclidean | §4.2 |
| `tree-d8`, `-d16`, `-d32` | tree | 8, 16, 32 | 1 | euclidean | §4.2 |
| `smoothed-d8`, `-d16`, `-d32` | smoothed | 8, 16, 32 | 2 | euclidean | §4.2 |
| `hyperbolic-d16` | trained | 16 | 1 | hyperbolic | `records/stage7/reference.json`, seeds 1–10 |
| `euclidean-d8`, `-d16`, `-d32` | trained | 8, 16, 32 | 1 | euclidean | `records/stage8/<name>.json`, seeds 1–5 |
| `hyperbolic-d8`, `-d32` | trained | 8, 32 | 1 | hyperbolic | `records/stage8/<name>.json`, seeds 1–5 |
| `spherical-d8`, `-d16`, `-d32` | trained | 8, 16, 32 | 1 | spherical | `records/stage8/<name>.json`, seeds 1–5 |

- **Components** are stages or post-processing steps, as the tie order reads them (split Req 24).
  The trained arms keep the count that Stage 8 recorded, 1 (verified locally: every Stage 7/8
  record's `spec.components`).
- **Names.** Stage 7's record names its arm `reference`. Its economic name is `hyperbolic-d16`,
  so it cannot be confused with the economic reference `covariates+ancestors` (my call).
- **Record paths** in the list are relative to the store root, which is a command option. The
  tracked file names no `/Users/` path.
- **Pins.** The list pins each source record's sha256, and each built table's sha256 once Stage 1
  has built it (my call). A rebuilt table with other bytes is then refused.
- **Multiplicity.** With 18 candidates, one could pass the reference by chance. The decision rule
  does not correct for that (split Req 24, unchanged). The economic sealed opening is the check
  against it.

### 4.2 The deterministic builders

All three builders read the codebook of bundle 301cce28 and, for text, the shared text-only table
(§4.3). They read no panel and no QCEW file. Each writes a 2,125-code table in the export form
(`panels.regressor.coordinate_matrix`), with its provenance beside it
(`panels.text_only.provenance_path`). The code goes in a new model layer, `tree_model/`, under
ADR 0001's rule for a method that fits neither `text_model/` nor `graph_model/` (my call).

- **`text-only-dN`.** The shared text-only table reduced to d by `panels.text_only.pca_reduce`,
  the same reduction that D9 applies inside a read. Its read's `covariates+embedding` therefore
  predicts as its `covariates+text_only` does (§6).
- **`tree-dN`.** Classical scaling of the tree metric D* (`utils.naics_hierarchy`,
  `tree_distance_matrix`) over the codebook's 2,125 codes, in float64 (my call for the method):
  - B = −½ J (D* ∘ D*) J, with J the centring matrix. `numpy.linalg.eigh` gives its
    eigenpairs.
  - The coordinates are the d leading eigenvectors, each scaled by the square root of its
    eigenvalue. Each column's sign is fixed so that its entry of largest magnitude is positive.
  - The builder refuses a non-positive d-th eigenvalue, or a d-th and (d+1)-th eigenvalue whose
    relative gap is below 1e-6. A tie would make the truncation arbitrary.
  - On the real codebook, the gaps at the cuts are 1,779.1 against 1,631.5 (d = 8), 970.08
    against 961.94 (d = 16) and 398.63 against 396.16 (d = 32). D* is not Euclidean: 424
    eigenvalues are negative (verified locally: the bundle's codebook and `tree_distance_matrix`
    on the Mac).
- **`smoothed-dN`.** Each code's text-only vector, at the backbone's width, is replaced by a
  weighted mean over its lineage (`code_lineage`, sector down to the code). The code has weight 1,
  its parent ½, its grandparent ¼, and so on, normalized to sum 1 (my call for the weights). The
  smoothed vectors are then reduced to d by `pca_reduce`. The weights are fixed here and never
  tuned. Every code's lineage is in the codebook (verified locally: no gaps over the 2,125 codes).
- **Determinism.** Two builds on one machine write the same bytes (test). The table's sha256 is
  its identity, and every read reads the stored table.
- **Where the real tables go.** Stage 1 builds the nine real tables on the Mac, outside the
  repository, under the store root in `candidates/economic/` (my call). Building reads no panel.

### 4.3 The shared text-only comparator (plan 14, P9)

Every economic read also scores `covariates+text_only` (D9). Every candidate names the same
text-only table, the one that all nine Stage 7/8 records name: sha256 `f4fb4f574f…`, from
MiniLM at revision `1110a243…`, descriptions `fe8c54e3…` and summaries `dd425eb5…` (verified
locally: the ten records' `text_only` fields). So the comparator is the same in every read, and
it decides nothing.

- A candidate that reads the backbone (text-only, smoothed and trained) is checked against that
  table by `check_text_only`, as today.
- The tree candidate reads no backbone. Its record says so and names no backbone, revision,
  descriptions, summaries or window. Its check requires only that its text-only table is the
  shared one. The field's shape is the builder's.
- `ArmSpec` is shared with Stage 8's records. Any change to it keeps those records loading and
  validating unchanged (test).

### 4.4 Trained candidates in the records and the decision

The deferred item's "Done when" is the target (`specs/deferred_items.md:622`).

- **The record.** An economic run of a trained candidate carries its seed and its checkpoint's
  selection evidence: the training run, the checkpoint and its epoch, and the monitor reads. They
  are copied from the imported Stage 7/8 seed run. A deterministic run keeps none of them, and its
  seed stays null.
- **The checks.**
  - An economic arm is one seedless run, or at least 5 runs with distinct seeds and their
    selection evidence. Anything else is refused.
  - Each trained run's selection evidence passes the same checks that Stage 8's records pass
    (`check_arm`'s monitor checks): the checkpoint is the earliest highest-MRR epoch, read on
    the outcome validation split only.
  - `check_reference_pairing` checks every run: the reference's predictions in each run equal the
    margin read's bit for bit.
- **Margin first.** It covers each run's growth reads. A trained seed's monitor reads are outcome
  reads from training, and they are not growth reads (split Req 24: "no candidate's growth error
  exists before δ is recorded"). Today's `check_margins_first` also times monitor reads, as D8
  needs. The economic path must not refuse a trained seed for its monitor reads. U32 requires
  this: the margins were fixed at 2026-10-18T00:43Z, after every monitor read of the 50 trained
  tables, so a check that timed them would refuse every trained candidate.
- **The decision.** `decide_economic` already draws seeds nested inside groups for a side with
  several runs. It reads the trained candidates through that loop, under the 5-seed floor.
- **The frozen list.** The decision takes the committed list, records its sha256, and refuses a
  candidate that is not on it or a list candidate that is missing.
- **The propagation comparisons** (§4.7) go in the decision record. They choose nothing.

### 4.5 Text and tree only, enforced (Req 29; split Req 23)

Plan 14's tooling checks that a provenance describes its table, not what it names (P8). This
stage adds the rule. A candidate's provenance must come from one of four builders: the export
(`tools export-table`), and the three builders of §4.2. Each builder writes a fixed set of
entries. The check refuses an entry that its builder does not write, and an input file other than:

- the bundle's codebook, descriptions or window-fitting summaries
- the backbone and its revision
- a text-model checkpoint whose contract has `bundle_id` 301cce28-539c-42ea-8781-496bbdcf511c
  and `contract_version` stage3-supervision-v2 (verified locally: the contracts in one Stage 7
  and one Stage 8 export provenance)
- the shared text-only table

A provenance that names a QCEW file, a panel, the selection log or an unknown entry is refused
(test). The export provenance of the 50 trained tables names a checkpoint, the descriptions, the
summaries and the backbone, with the contract. It has two known versions, and the rule accepts
both (verified locally: the 50 files in the main checkout's `checkpoints/`):

- Stage 7's ten files have no `geometry` entry. They were written before 48ff710 added it.
- Stage 8's forty files have the same entries plus `geometry`.

How the check knows each builder is left to the builder of this stage.

### 4.6 The read command and its dry run

One command, `tools economic-read` (my call for the name), reads one candidate from the frozen
list on both growth validation splits and writes its economic arm record.

- **Deterministic candidates** go through `read_deterministic_arm`, with the built table and its
  provenance.
- **Trained candidates** are imported. The command loads the source record from the store and
  runs `check_arm` on it. It finds each seed's export provenance in the main checkout's
  gitignored run directory (`checkpoints/stage8-<arm>-s<seed>/` or
  `checkpoints/stage7-reference-s<seed>/`), and requires it to name the stored table's sha256
  (`check_provenance`). It then reads each seed's stored table once on each growth panel.
  All 50 provenance files exist (verified locally).
- **Every check runs before the first read.** That covers the list, the provenance rule, the
  source record, the text-only table and the margin record's own checks.
- **`--dry-run`** runs every check that needs no panel and stops. It reads no panel, loads no
  QCEW file and logs nothing. Stage 1 runs it on all 18 real candidates, so that Stage 3 is a
  read and nothing else.
- **Where it runs.** From the main checkout's root, or with `--log` and `--lock` as absolute
  paths, as `tools growth-margins` does (plan 14, Handoff). Stage 3 owns the real reads.

### 4.7 The propagation answer (split Req 19)

Split Req 19 asks whether propagating features over the tree, by smoothing or by the graph stage,
helps the economic artifact. This stage answers it with smoothing, and the graph stage stays
parked (U33).

- **The rule** (my call). At each d, the decision compares `smoothed-dN` with `text-only-dN`,
  using `compare` on the decision's own draws and the panels' δ. Propagation helps at d if the
  smoothed candidate is adopted over its twin: non-inferior on both panels and superior on at
  least one.
- **The answer** is Stage 3's finding. If propagation helps at some d, a later spec may revisit
  the parked graph stage for this artifact. If not, the finding records no evidence that it
  helps, and the graph stage stays parked.
- The comparison is a report. It does not change which candidate is chosen.

### 4.8 The pinned economic path

Stage 3's reads pair only where the reference predicts bit for bit as in the margin read, under
`uv.lock` 6de4fbdc, on macOS arm64 with Python 3.12.12 (finding: growth-margins §4).

- **`uv.lock` does not change.** Every method here uses packages that are already locked: numpy,
  scipy, scikit-learn and torch (verified locally: `uv.lock` at b4f8c9d, sha256 `6de4fbdc…`). If a
  change to the lock seems needed, stop and ask the user. No change keeps the reads pairing.
- **The reference's code path does not change.** The reference's predictions run through
  `panels/regressor.py`, `panels/qcew_rows.py`, `panels/regressor_splits.py` and
  `panels/ridge.py`. This stage edits none of them. If it must, it first pins the fixture growth
  reference's predictions and shows them unchanged. Nothing in `src/` or `uv.lock` has changed
  since the margin read (verified locally: `git diff 30aeff6..b4f8c9d -- src/ uv.lock` is empty).
- **The predictions are bound to their table** (`specs/deferred_items.md:640`). Its trigger is
  met: this stage reads tables it does not build, and it rewrites the economic run. A record
  whose predictions belong to another table is refused (test). The reference comparison still
  reads only the margin read's columns, so a binding stored with the predictions cannot break
  the pairing.

## 5. One role per table or part (Req 26)

| Table or part | Role for the economic artifact | Read in this stage |
|---|---|---|
| Bundle 301cce28: codebook, descriptions, summaries | Training (a candidate's input) | Yes, by the builders |
| MiniLM text-only table `f4fb4f574f…` | Training (input to text-only and smoothed). As `covariates+text_only`, it is only reported | Yes, by the builders |
| Stage 7/8 checkpoints and their tables | Training (the trained candidates) | Yes, from the store |
| Outcome panel, validation split | Selection: it chose the trained checkpoints at Stages 7–8 | No: the evidence is imported, not re-read |
| Outcome panel, test split | Not the economic artifact's. It stays sealed | No |
| Growth panels, validation splits | Selection: the choice among candidates, and §4.7's report | No: Stage 3 reads them |
| Growth panels, outer sets | Judging: the economic sealed opening | No: embedding roadmap Stage 12 |
| Level-mode validation splits | None: Stage 7–8 history | No |
| Level-mode outer sets | None: sealed, one seal per split with growth (split Req 22) | No |
| QCEW files | Panel data only. Never a candidate's input (§4.5) | No |
| The margin record of 2026-10-17 | The decision's margins, never refixed | Only its own checks, from the store |
| Stage 7/8 arm records | The trained candidates' evidence | Yes, by the dry run |
| embedding-data tables | None | No |

## 6. Verification — observable outcomes

- [ ] **One role per table or part.** §5 lists every table or part this stage and Stage 3 read,
      with its one role.
- [ ] **Text and tree only.** Each of the four builders' provenance passes the rule. A
      provenance naming a QCEW file, a panel, the selection log or an unknown entry is refused.
- [ ] **Builders.** On a fixture tree, `tree-dN` matches a hand-computed classical scaling, keeps
      its sign rule and refuses a tied cut. `smoothed-dN`'s weights match a hand-computed toy
      lineage. Each builder writes the same bytes twice.
- [ ] **D9 consistency.** On the fixture panel, a `text-only-dN` read's `covariates+embedding`
      predictions equal its `covariates+text_only` predictions.
- [ ] **Trained records.** On fixtures, an imported trained arm with 5 seeds is read, and its
      record validates. A trained arm with 4 seeds, a seedless run among several, a seeded single
      run and a trained run with no selection evidence are each refused.
- [ ] **Margin first.** A trained seed whose monitor reads precede the margins is accepted. A
      growth read before the margins is refused.
- [ ] **Decision.** On fixtures, the command reads a deterministic and a trained candidate, and
      `decide_economic` records a decision over `growth_seen` and `growth_heldout`, with the
      trained seeds resampled nested inside the groups. It records the propagation comparisons.
      It refuses a candidate off the frozen list and a missing list candidate.
- [ ] **Binding and stored scores.** A record whose predictions belong to another table is
      refused. A record whose stored scores differ from its rescored predictions is refused.
- [ ] **Stage 8 unchanged.** The existing D8 tests pass unchanged, and Stage 8-style records
      still load and validate.
- [ ] **Real tables and dry run.** The nine deterministic tables exist under the store root, each
      with its provenance. The dry run passes on all 18 real candidates. The main checkout's
      selection log has the same sha256 before and after.
- [ ] **Nothing real was read.** No growth read ran. No margin record was written, and
      `tools growth-margins` did not run.
- [ ] **Pinned path.** `uv.lock` is unchanged (sha256 `6de4fbdc…`). `panels/regressor.py`,
      `panels/qcew_rows.py`, `panels/regressor_splits.py` and `panels/ridge.py` are unchanged
      against b4f8c9d.
- [ ] **Checks.** The full suite, Ruff, the read-only style gate and the strict docs build pass.

## 7. Deferred items this touches

`specs/deferred_items.md` was read, not edited. The plan's completion protocol records each
outcome.

- **Trained economic candidates** (`:622`): done by §4.4 and §4.6.
- **Economic stored-scores check** (`:633`): done here, because §4.4 rewrites
  `check_economic_arm` (my call). Both checks resolve the stored scores and compare them with the
  rescore within `STATISTIC_TOLERANCE`.
- **Economic predictions-to-table binding** (`:640`): done by §4.8, because its trigger is met.
- **Economic decision test gaps** (`:650`): the gaps in functions this stage rewrites get their
  tests: `check_economic_arm`'s statistics-edit and key-set refusals, `_same` on the economic
  read identities, the CLI's adopted path and `_margin_kind` (my call). The rest stays open.
- **Public native qualification** (`:560`): not needed, because nothing trains.
- **The graph-stage items:** untouched, because HGCN stays parked.

## 8. Left to the builder

- The record fields' names and shapes, including how the tree candidate's record says it reads no
  backbone.
- How the provenance check identifies each builder.
- The command's options, beyond those this spec names.
- The frozen list file's schema.
- Where the binding is stored, within §4.8's limit.
- The docs: an API page for `tree_model/` in `docs/.nav.yml`, and the command in
  `docs/usage.md`.

## Rollout note

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 1 — on plan completion, tick the stage
> and re-validate later stages against what shipped.

Stage 3 reads what this stage produces, on the Mac, with the user's approval at the time:

- **The command** and the frozen list, with the nine built tables and the 50 imported tables.
- **The environment.** A commit whose `uv.lock` is 6de4fbdc, with Python 3.12.12 and numpy 2.3.4
  on Accelerate, and no BLAS thread variable set (finding: growth-margins §1).
- **The log.** The main checkout's `logs/selection_log.jsonl`, checked for growth reads beyond
  the margin read's two before the first candidate read.
- **The margins.** `~/naics-artifacts/records/economic/growth_margins.json`, sha256 `4f86a0b2…`,
  read and never refixed.
- **The propagation rule** of §4.7, whose result Stage 3's finding states.
