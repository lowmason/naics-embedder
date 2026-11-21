# NAICS artifact redesign — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

**Status: APPROVED (2026-11-14).** Derived in an interactive session. The user answered the
derivation's three questions (Decisions, below) and approved the seven-stage partition at the
human checkpoint ("Approve, hand off Stage 1"). Stages 1–4 are COMPLETE. The resume of
2026-11-17 amended Stage 3 from plan 16's Handoff, re-read the gates at embedding-data ec6fb29 and,
on the owner's answer ("Yes, re-pin"), re-pinned Stage 2 there. The resume of 2026-11-18 amended
Stages 4 and 5 from plan 17's Handoff, re-read the gates at embedding-data fd9e495 and lifted the
HELD stamps of Stages 4 and 5. On the owner's answer ("Approve split (Recommended)"), it split
Stage 6: Stage 6 writes the relatedness spec and builds its panel on fixtures, with no gate, and
the new Stage 6b reads the reference and records the margins. The resume of 2026-11-19 read plan
19's Handoff, which amends no stage here, and took the edits that Stage 6's approved spec gives a
resume (`specs/relatedness-artifact.md` §8). It re-read the gates at embedding-data 6be42f2 and
added G6. On the owner's answers (Decisions, 2026-11-19), it added Stage 7b, ended the economic
chain at Stage 3 with no sealed opening, and recorded the tenth resume of the embedding roadmap.
The resume of 2026-11-21 read plan 20's Handoff and amended Stage 5 from it. It re-read the gates
at embedding-data 4942e1d, where no gate changed, and routed Stage 5. Stage 6 is executing as plan
21 (PR #175). Stages 6b, 7 and 7b are HELD on gates in embedding-data (Gates, under Stages).
`uv.lock` is still 6de4fbdc at 1f19b1c.

**Basis.** Source spec `specs/naics-artifact-redesign.md` (user adjudication Q4, approved as U31),
merged as PR #156 (beaee8f), under ADRs 0001 and 0002 (`specs/adr/`, accepted 2026-11-14).
Evidence in this repository was read at origin/main beaee8f. Evidence in embedding-data
(`~/Projects/embedding-data`) was read at its main 60068eb, seven commits past the spec's 147802f.
Its Stage 10 spec, `specs/ita-establishment-targets.md`, is on that main and identical to the
49b394c copy that the spec cites. Paths are under `src/naics_embedder/` unless they start with
`conf/`, `tests/`, `specs/` or `docs/`, or name a top-level file. A bare "Stage N" is this
roadmap's. "Embedding roadmap Stage N" cites `specs/naics-embedding-roadmap.md`, and
"embedding-data Stage N" cites that repository's `specs/embedding-evaluation-data-spec-roadmap.md`.
U17–U31 and P1–P7 are the spec's rulings and proposals, and "split Req n" cites
`specs/naics-representation-split.md`. The refresh of the methodology description to Q4 (P7) is
not a stage here.

**Decisions (2026-11-14).** Three questions the derivation asked, answered by the user.

- **Recording Q4 (Rollout note, item 1).** Decision: a ninth resume of the embedding roadmap, made
  at this derivation and merged with this file, not a stage here. Q3's reason does not carry over,
  because that roadmap routes nothing now. The reason now is that the record edits that roadmap's
  pointer and its Stage 12, and a resume is how that roadmap is edited. The record updates every
  live copy of the overview: `CLAUDE.md`, `README.md` and `docs/index.md`.
- **Where the embedding-data gates live (ADR 0002, Negative).** Decision: as evidence, not as
  `Consumes`. The Req 26 row and the Gates paragraph (under Stages) record each gate's state with
  the embedding-data SHA read. Each gated stage carries a HELD stamp that names its gates. A
  resume lifts a stamp only after it re-checks those gates in embedding-data, and it records the
  SHA it read.
- **A fourth embedding-data gate, G4.** embedding-data requires a successor evaluation-protocol
  spec "before any embedder is scored" (spec `:634`). Its Stage 10 spec gives that successor spec
  the data splits and the use of the copied-text flags (`ita-establishment-targets.md:72-73`,
  `:83-84`). Req 28 and U28 give both to text→code's own spec. Decision: record G4. Before any ITA
  read or relatedness judging read, the successor spec exists and assigns ITA scoring and
  splitting to text→code's spec. G4 does not block writing or building the scoring on fixtures.

**Text→code's reference (owner ruling, 2026-11-16).** No spec named the comparator that text→code
must beat on ITA text, so the user was asked which bar the README should state. The ruling is the
chosen option's label, "Stub or d16, better (Recommended)". The option's description, quoted as
given:

> Proposed ruling: text→code must beat the better of the lexical stub and euclidean-d16 on the ITA
> validation headline MRR, by a relative margin (r × the reference's MRR) fixed from reference
> reads only, before any candidate read. The margin is relative because the stub has no seeds, so
> a seed-spread margin is undefined. I would record the ruling in the roadmap's Stage 2 entry so
> that the README can link to it.

The lexical stub is finding: outcome-panel-splits §5, and `euclidean-d16` is finding:
geometry-by-dimension §3. The ruling leaves r open. Stage 2's spec sets it with the rest of the
margin rule.

**Decisions (2026-11-19).** The resume after Stage 3 asked one batched question with four parts.
The user chose the recommended option in each. Each ruling is the chosen option's label.

- **Stages: "Approve both (Recommended)".** Stage 7b, relatedness's sealed opening, is added in
  its own chain (U45, relatedness spec §4.8). Spec §4.7 leaves one choice to a resume: whether Stage
  7 builds its model on fixtures before G1 opens. The resume after Stage 6 makes that choice, when
  such a stage could first start. Spec §8 gives its edits to "the resume after Stage 6". They need
  nothing from Stage 6's plan, so this resume made them.
- **The economic opening and the lock: "(a) No opening, lock lapses (Recommended)".** Stage 3
  adopted no candidate, so the reference `covariates+ancestors` stands and there is no economic
  artifact (finding: economic-decision). The economic chain ends at Stage 3. Embedding roadmap
  Stage 12 opens neither growth regime's outer set, and both stay sealed for a later economic
  round. The `uv.lock` constraint no longer binds another chain (Across chains, under Stages).
  Five texts state or assume the economic opening. This decision supersedes each of them:
  - Embedding roadmap Stage 12 (`specs/naics-embedding-roadmap.md:1148-1153`, `:1163`): corrected
    by the tenth resume of that roadmap (`:1164-1174`), in the same PR as this resume.
  - Plan 19's Handoff (`specs/plans/completed/19-economic-decision-read.md:941-942`) and the
    finding (`specs/findings/economic-decision.md:254-255`, `:261`): records, not edited.
  - The Q4 spec's Unchanged list (`specs/naics-artifact-redesign.md:265`) and split Verification
    (`specs/naics-representation-split.md:346-347`): approved specs, so the user's to edit. They
    are follow-ups (next bullet).
- **Texts: "Tenth resume in this PR (Recommended)".** This PR edits this roadmap and the
  embedding roadmap (its tenth resume), and adds dated amendment notes to ADRs 0001 and 0002.
  Follow-ups, named in the PR body: the relatedness spec's lock constraint (`:105`,
  `:347-349`, `:428-429`) and its released code-PR hold (`:430-434`), `CLAUDE.md:334-340`, and
  the two spec lines above.
- **Hold: "Hold both until merge (Recommended)".** Stage 4's execution (plan 20) and Stage 6's
  planning start only after this resume's PR merges.

## Gap analysis

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 25 | implemented-differently | `CLAUDE.md:5-7`, `README.md:8-11` and `docs/index.md:3-6` name Q3's hierarchy artifact, and `CLAUDE.md:7` judges text→code on the outcome panel. Embedding roadmap Stage 12 (`specs/naics-embedding-roadmap.md:1093-1094`) gives the hierarchy no opening and opens the outcome test split for text→code. The codebook is the bundle's (`supervision/`). | The codebook, and the economic artifact's own reference and margins, exist. No relatedness artifact exists. The derivation discharges the docs half (Decisions). Stages 1–7 build the rest. |
| 26 | out-of-repo | embedding-data at 60068eb: R8.4 is unamended (spec `:550`). V9 runs "after the first full build" (`:606-609`, `:669-670`), at embedding-data Stage 13, which is unticked (roadmap `:123`, `:299`). Stage 10's anti-goal is unchanged (`ita-establishment-targets.md:124`, `:92-93`). The successor protocol spec does not exist (spec `:634`). In this repository, no `src/` module reads an embedding-data table. Re-read at ec6fb29 on 2026-11-17: no gate opened (Gates, under Stages). Re-read at fd9e495 on 2026-11-18: G2 and G3 opened, and G4 and G5 opened for the ITA text–code pairs only. Re-read at 6be42f2 on 2026-11-19: no gate changed. G6 was added from Stage 6's spec, and it is closed. Re-read at 4942e1d on 2026-11-21: no gate changed. | The owner gates live in embedding-data (Gates, under Stages). For the existing panels, the one-role rule already holds through the selection log and `decide`'s validation-only reads (`decision/decide.py`). Each stage spec lists its tables and parts with one role each. |
| 27 | missing | `graph_model/` holds the parked HGCN only, and nothing reads a relation family. embedding-data: the ITA area mix is built (embedding-data Stage 6) with `size_gate_pending`. Co-ownership (embedding-data Stage 10) is not built, and none of plan 11's modules is in its `src/`. ACS (Stage 9) and CPS (Stage 11) are unticked there. | Every family waits for its build and its count gate. Training also waits for R8.4. Stages 6 and 7. |
| 28 | implemented-differently | Training reads the codebook's own text only (`supervision/queries.py:9-19`). Judging uses the outcome panel (`panels/outcome.py:159`, `:217`). No ITA reader exists, and embedding-data Stage 10's pairs are not built. | Spec: judged on the ITA validation and sealed parts, with the outcome panel reported beside them. Code: judged on the outcome panel. Stages 2, 4 and 5. |
| 29 | implemented-differently | The test is built: `fix_relative_margins` and `decide_economic` (`decision/economic.py:179`, `:537`) and `tools growth-margins` (`cli/commands/tools.py:562`). The margins were fixed once (finding: growth-margins), and `uv.lock` is still 6de4fbdc. `read_deterministic_arm` (`decision/economic.py:250`) has no command. Trained candidates are deferred (`specs/deferred_items.md:622`). | No candidate exists. The Q3 audit gave its candidates row the same verdict. Stages 1 and 3. |

No in-code-but-not-in-spec row: the parked HGCN stays with naics-embedding Reqs 15 and 17, and the
outcome panel stays under Req 28.

## Stages

**Sequencing.** The stages form one chain per artifact (ADR 0002): economic (Stages 1 and 3),
text→code (Stages 2, 4 and 5) and relatedness (Stages 6, 6b, 7 and 7b). They follow the Rollout
note's items 2–4, and item 1 was done at derivation (Decisions). The spec ranks nothing beyond that
order, so this order follows the spec's. The stages with no gate come first: Stages 1 and 2 need
nothing from embedding-data, and Stage 3 needs only Stage 1. Each chain measures before it builds
models (split Core principle). The economic margins exist already. Stage 2 fixes text→code's
scoring and margin rule, and Stage 5 records the margins before its first candidate read. Stage
4's gap read selects nothing, so G2 does not hold it (Rollout note, item 3). Stage 6b fixes
relatedness's margins before Stage 7 trains. No stage's `Consumes` names a stage of another
chain. Text→code's sealed opening is not a stage here: it stays in embedding roadmap Stage 12
(Req 28, ninth resume). The economic chain has no sealed opening, because Stage 3 adopted no
candidate (Decisions, 2026-11-19). Relatedness's opening is Stage 7b, in its own chain (U45,
relatedness spec §4.8; added at resume, 2026-11-19). Stage 6's spec chose Stage 7's method,
GraphSAGE (U46, spec §4.7), so Stage 7 routes to writing-plans (Resolved at resume, 2026-11-19).
The next stage to hand off is the first unticked one whose `Consumes` are met and that carries
no HELD stamp. Plans start at 16.

**Across chains.** Economic reads paired only where the reference predictions matched the
2026-10-17 margin read to within 1e-13 (economic-candidates §4.8, amended 2026-11-18 from "bit for
bit"), under `uv.lock` 6de4fbdc (finding: growth-margins). Stage 3's decision exists, and no
economic read is planned (Decisions, 2026-11-19). So a stage spec in another chain may change
`uv.lock` and need not say how the economic reads pair (ADR 0002, Negative, amended 2026-11-19).
A later economic read needs its own spec, which says how it pairs, for example from a checkout at
lock 6de4fbdc. The economic chain trains nothing more, so a change to `text_model/` (ADR 0001)
now reaches text→code's training runs only.

**Gates in embedding-data, read at 4942e1d** (resume, 2026-11-21; earlier reads at 60068eb,
ec6fb29, fd9e495 and 6be42f2). Local main and `origin/main` there were both 4942e1d. Since
6be42f2, embedding-data completed its Stage 17, the ACS wage microdata. That file is a
person-level input for a later wage panel here, not a target family. The owner's notes for it
under R7 and R8 (spec `:529-530`, `:547-548`, `:558-563`, `:576-577`) cover that file only. The
protocol spec and Stage 10's spec did not change, so no gate changed. Since fd9e495, the protocol
spec gained only copied-pair counts for Section 1. The new specs there (TRI parent links and
USAspending vendor links) open no gate. The gates are not `Consumes`
(Decisions). Each HELD stamp names the gates it waits on. On 2026-11-16 embedding-data parked its
roadmap (now `specs/completed/embedding-evaluation-data-spec-roadmap.md`) and replaced it with
`specs/naics-evaluation-data-roadmap.md`. Old Stage 10 continues as new Stages 1 and 4. The NAICS
families' count gate is new Stage 3. Old Stages 9 and 11 returned as new Stages 9 and 11 on
2026-11-18, and old Stages 12 and 13 are deferred items there. "Embedding-data Stage N" below
cites the new roadmap. Its Stage 10 amendment and its protocol spec
cite text→code's spec as `specs/text-to-code-ita-panel.md`, which is now
`specs/completed/text-to-code-ita-panel.md` here.

- **G1, R8.4.** No embedding-data table trains an artifact until the owner amends it (Req 26).
  State: unamended, "Benchmark tables are never training data" (spec `:570`). The protocol spec's
  Handoff puts the owner's R8.4 decision after this repository's relatedness spec names the
  families that train (`specs/evaluation-protocol.md`, Handoff). That spec is approved (PR #170).
  It names one family to train, the 2023 ITA area mix (relatedness spec §6, §7 item 3), so the
  owner's decision can now be made. Holds Stage 7.
- **G2, Stage 10's anti-goal.** It must allow selection on the ITA validation part (Req 26, U27).
  State: open. The owner's amendment of 2026-11-18 (e03ec67, lowmason/embedding-data#21) allows
  checkpoint and candidate selection on the validation part of the headline text–code pairs only
  (`specs/completed/ita-establishment-targets.md:11-27`).
- **G3, the V9 Rollout sentence.** It must allow a per-family count gate (Req 26, U29). State:
  open. V9 sets each family's gate after that family's first full build (spec `:629-635`,
  `:698-699`). The V9 ITA NAICS decision (`specs/findings/v9-ita-naics-minimum-counts.md`,
  2026-11-17) sets a minimum of 1 for ITA co-ownership, the 2023 ITA area mix and the ITA
  text–code pairs. These are the three families that a stage here reads, so G3 holds no stage.
- **G4, the successor evaluation-protocol spec** (Decisions). For text→code, it assigns ITA
  scoring and splitting to text→code's spec. For relatedness, it covers the families that Stage
  6's spec names, because the protocol for tables other than the ITA pairs stays with
  embedding-data (the spec's Out of scope). State: `specs/evaluation-protocol.md`, APPROVED
  2026-11-18. Its Section 1 opens G4 for the ITA text–code pairs. It assigns their split and
  scoring to text→code's spec, pins the run and table hashes that a read uses, and keeps read
  conditions C1–C6. No other family has a section, so G4 stays closed for relatedness.
  Embedding-data Stage 16 writes the sections for co-ownership and the 2023 area mix
  (`specs/naics-evaluation-data-roadmap.md:201`, unticked at 4942e1d), as the relatedness spec
  asks (§7, item 1). Holds Stages 6b, 7 and 7b.
- **G5, each family's build and count gate** (Req 26, U29). A family is read only after it is
  built and has passed its own V9 count gate under G3. State per family:
  - ITA text–code pairs: open. Built (embedding-data Stage 1, complete 2026-11-16), count gate
    set (Stage 3, complete 2026-11-17) and hardened (Stage 4, complete 2026-11-18). Every
    headline pair meets the minimum of 1 (protocol, Section 1).
  - ITA co-ownership (Stage 1) and ITA area mix (old Stage 6): open. Built, with a count gate of
    1, and no protocol section (G4).
  - ACS families and CPS job-change flows: not built. Embedding-data re-staged them on 2026-11-18
    as its Stages 9 and 11. The relatedness spec gives them no role (§4.1), nor TRI parent links,
    which are not built (embedding-data Stage 5).
  - ACS wage microdata: built (embedding-data Stage 17, complete 2026-11-19), with no count gate
    yet (its O23) and no protocol section (its O24). No stage here reads them.

  Both families that Stage 6's spec names are open, so G5 holds no stage.

- **G6, Stage 10's exclusion for co-ownership** (relatedness spec §6; added at resume,
  2026-11-19). Selection on co-ownership's validation part (U43) needs the owner to extend Stage
  10's selection exception to co-ownership, as G2 did for the text pairs (relatedness spec §7,
  item 2). State: closed. The amendment of 2026-11-18 says "Co-ownership and the link table keep
  the full exclusion" (`specs/completed/ita-establishment-targets.md:24-25`). That exclusion
  covers training, curriculum, hard-negative mining and model selection (`:109`), not judging, so
  G6 does not hold a sealed read.
  Holds Stage 6b's reference choice and margin record, and Stage 7's selection.

**Deferred items.** `specs/deferred_items.md` was read, not edited. The entries that the spec
names (Deferred items this touches) go with their chains. Stage 1 takes trained economic
candidates (`:622`) and the predictions-to-table binding (`:640`), whose trigger the candidates'
spec may meet. The other plan-14 entries (`:633`, `:650`) touch the same module and may go with
it. Stage 5 takes the entries that moved with embedding roadmap Stage 9: nothing ties the bundle's
backbone to training at runtime (`:276`), and the summaries-key watch (`:481`). Stage 5 also takes
the five entries of § 17-text-to-code-ita-panel (plan 17, Handoff). Stages 6 and 7
read the graph entries if they reuse `graph_model/`: the curriculum thresholds (`:60`), the
diagnostics report's interfaces (`:220`), the HGCN half of the Lorentz-distance item (`:244`),
feeder precision (`:496`), metric mutation coverage (`:522`) and the legacy-input fallback
(`:662`). A multi-seed Lambda campaign in Stage 1 or 5 first needs the public native
qualification (`:560`).

- [x] Stage 1: Economic — candidates
      Objective: Build the economic candidates from text and the tree, with a command that reads
      a candidate, and take each one through the economic decision on fixtures.
      Spec: Req 29 (all but the real read), split Req 23 (what a candidate is, and selection on
      validation splits), split Req 19 (the propagation question), Req 26, Verification "One role
      per table or part" and "Economic candidates", Rollout note item 2, Deferred items this
      touches (first bullet).
      Gap closed: Req 29 (the candidates).
      Consumes: The economic tooling on main, through the interfaces that plan 14's Handoff names
      (`specs/plans/completed/14-economic-decision-tooling.md:6483`): `read_deterministic_arm`
      (`decision/economic.py:250`), which no command calls, `decide_economic` (`:537`) and
      `EconomicRun`, which has no seed, checkpoint or monitor field. The margin record of
      2026-10-17 and its read environment (finding: growth-margins §§2, 4). The text-only table
      builder (roadmap D9) and the bundle's codebook. The deferred entries that Deferred items
      names.
      Produces: The candidates' stage spec. Each candidate's 2,125-code table, with a provenance
      that names only text and tree inputs. A command that reads a candidate on the growth
      validation splits. Trained candidates' records, if the spec keeps trained candidates. The
      spec's answer to split Req 19's propagation question for this artifact. Stage 3 reads them.
      Exit: The stage spec lists every table or part it reads, with its one role. Each
      candidate's provenance names only text and tree inputs (test). On fixtures, the command
      reads a candidate and `decide_economic` records a decision over `growth_seen` and
      `growth_heldout` (test). No real growth read ran, and no margin record was written. The
      suite, Ruff, the style gate and the strict docs build pass.
      ROUTING: brainstorming
      > Stage 1: COMPLETE (2026-11-15) — implemented by plan 16 (specs/plans/completed/16-economic-candidates.md). Its Handoff section names what Stage 3 consumes.
      > Next: resume the roadmap.

- [x] Stage 2: Text→code — ITA scoring and split
      Objective: Write text→code's stage spec and build its ITA panel on a fixture: the grouped
      split into a validation and a sealed part, the shortlist scoring, the reference and margin
      rule, and the one logged opening of the sealed part.
      Spec: Req 28 (Output, Judge, What the ITA text is like, The outcome panel stays and Owner
      gate), Req 26, Req 25 (own reference, margins and opening), U24, U25, U27, U28, P3–P5,
      Supersessions (the text→code rows), Verification "One role per table or part", Rollout note
      item 3, Stage 10 (the pair table's schema and copied-text flags, at embedding-data ec6fb29,
      where index-entry copies are flagged apart from title copies; re-pinned from 60068eb,
      owner, resume 2026-11-17).
      Text→code's reference (owner ruling, 2026-11-16, above Gap analysis).
      Gap closed: Req 28 (the judging panel, on fixtures).
      Consumes: The codebook (bundle 301cce28), from which a six-digit code's ancestors follow.
      The outcome panel (`panels/outcome.py`), reported beside the ITA scores. The selection
      log, which keys an opening on its panel name and fingerprint (`panels/selection_log.py`).
      The query path, `ArmEncoder.from_files` (`text_model/arm_encoder.py`). ADR 0001, which puts
      the ITA panel under `panels/`.
      Produces: Text→code's stage spec. The ITA panel: its split under a committed fingerprint,
      its validation reads and scores, its headline and copied-pair reports, and its one logged
      sealed opening. Text→code's reference and margin rule. The spec's ruling on whether the
      outcome test split also opens at embedding roadmap Stage 12, for the report. Stages 4 and
      5 read them.
      Exit: On a fixture with Stage 10's pair schema, each normalized description falls in one
      part only (test). The headline score excludes copied pairs and counts each distinct pair
      once (test). The score reads the reported code's shortlist rank and the coarser-level
      matches (test). A second opening of the sealed part is refused without a reason (test). The
      stage spec lists every table or part it reads, with its one role. No real ITA pair was
      read. The suite, Ruff, the style gate and the strict docs build pass.
      ROUTING: brainstorming
      > Stage 2: COMPLETE (2026-11-17) — implemented by plan 17 (specs/plans/completed/17-text-to-code-ita-panel.md). Its Handoff section names what Stages 4 and 5 consume.
      > Next: resume the roadmap.

- [x] Stage 3: Economic — decision read
      Objective: With the user's approval at the time, read every Stage 1 candidate on the growth
      validation splits on the Mac, and record the economic decision against the fixed margins.
      Spec: Req 29 (The test, and the margins of 2026-10-17), split Req 24, Supersessions
      (Unchanged), Verification "Economic candidates".
      Gap closed: Req 29 (the decision).
      Consumes: Stage 1's candidates, read command and Handoff. The margin record, read at
      origin/main 30aeff6 on macOS arm64 with Python 3.12.12 under `uv.lock` 6de4fbdc (finding:
      growth-margins, Sources and §1). Every candidate read runs under the same three, because
      `decide` pairs only reference predictions equal to the margin read's to within 1e-13
      (amended 2026-11-18 from bit-identical; `check_reference_pairing` in
      `decision/economic.py`; cite names, not lines: plan 16, Handoff, Resolved at resume,
      2026-11-17). The lock on main is still 6de4fbdc at 6e3ff00. The main checkout's selection
      log (`logs/selection_log.jsonl`), in which the margin read found no opening of either growth
      regime (finding: growth-margins §3). The user's approval, asked when the stage starts.
      As built (plan 16, Handoff, Resolved at resume, 2026-11-17): the frozen list
      `conf/data/economic_candidates.yaml` with its nine pinned deterministic tables, the read
      command `tools economic-read`, and `tools decide --candidates`. The nine trained candidates
      have no Stage 1 record: each is imported at its read from its Stage 7/8 arm record and
      `--runs-root`.
      Produces: The economic decision record over `growth_seen` and `growth_heldout`, which
      names the adopted candidate or says that the reference stands, and a written finding.
      The finding also answers split Req 19's propagation question from `decide`'s
      `Propagation:` lines (plan 16, Handoff, Resolved at resume, 2026-11-17). Embedding roadmap
      Stage 12's economic opening reads both.
      Exit: Before the first real candidate read, the frozen list pins the text-only table's
      provenance file and the list's sha256 is recorded again (`specs/deferred_items.md`,
      § 16-economic-candidates, second item; plan 16, Handoff, Resolved at resume, 2026-11-17).
      The decision record exists and names each candidate's text-and-tree provenance.
      `tools growth-margins` did not run, and the decision reads the 2026-10-17 record. The
      selection log gained only validation reads, and no outer set opened. The finding cites the
      record's sha256 and the platform, Python and lock that it was read under.
      ROUTING: writing-plans
      > Stage 3: COMPLETE (2026-11-18) — implemented by plan 19 (specs/plans/completed/19-economic-decision-read.md). Its Handoff section names what embedding roadmap Stage 12 consumes.
      > Next: resume the roadmap.

- [x] Stage 4: Text→code — reference read and gap
      Objective: With the user's approval at the time, read the reference on the real ITA
      validation part and measure the gap between Census phrases and owner text.
      Spec: Req 28 (Judge, What the ITA text is like, and Later), Req 26 (Size gate), U25, U27,
      Verification "Gates held".
      Gap closed: Req 28 (the reference read and the measured gap).
      Consumes: Stage 2's ITA panel, split, reference and margin rule, through its Handoff.
      As built (plan 17, Handoff, Resolved at resume, 2026-11-18): `ItaPanel.from_files` with
      `conf/data/ita_panel.yaml`, `score_logged` on the validation part with role `reference`,
      and `ItaResult`'s reports and `data_fingerprint`. A real build that the panel refuses at
      construction logs nothing.
      Produces: The reference's validation scores. A finding that reports the gap, on headline
      and copied pairs apart, and says whether a later spec should add an ITA training part
      (Req 28, Later). Stage 5 reads both.
      Exit: The selection log holds the reference's ITA validation reads and no opening of the
      sealed part. No margin record was written, and nothing was selected. The finding reports the
      gap with its intervals, and it names the embedding-data commit and the pair table that it
      read.
      ROUTING: writing-plans
      Stage 4: HELD lifted (2026-11-18), gates read at embedding-data fd9e495 (Gates, above). G3
      is open. G4 is open for the ITA pairs through `specs/evaluation-protocol.md` Section 1,
      whose pinned run and conditions C1–C6 bind every read. G5 is open for the ITA pairs.
      Planned as plan 20 (`specs/plans/20-text-to-code-reference-read.md`, PR #169, 4ba1f27),
      approved and not executed. Its pre-flight's Stage 3 condition (D3, the COMPLETE stamp) is
      met. Its execution starts after the resume of 2026-11-19 merges (Decisions, 2026-11-19).
      > Stage 4: COMPLETE (2026-11-19) — implemented by plan 20 (specs/plans/completed/20-text-to-code-reference-read.md). Its Handoff section names what Stage 5 and embedding roadmap Stage 12 consume.
      > Next: resume the roadmap.

- [ ] Stage 5: Text→code — candidates and decision
      Objective: Record text→code's margins before any candidate read, then build the candidates,
      among them embedding roadmap Stage 9's backbone and one-factor ablations, select them on
      the ITA validation part, and record text→code's adoption decision.
      Spec: Req 28 (Training, and its last bullet), split Req 19 (Stage 9 held), U25, U27,
      Verification "One role per table or part" and "Gates held".
      Gap closed: Req 28 (the candidates and the decision).
      Consumes: Stage 2's ITA panel and margin rule, and Stage 4's reference scores and finding,
      through their Handoffs. Embedding roadmap Stage 9's entry (HELD), with its reference cell
      `euclidean-d16` and Plan 10 R7's term-weight decision (finding: geometry-by-dimension §7).
      The remote workflow, for any multi-seed campaign. The deferred entries that Deferred items
      names. As built (plan 17, Handoff, Resolved at resume, 2026-11-18): the margin rule in
      `decision/text_to_code.py` (`fix_ita_margin`, `check_ita_margin`, `check_ita_margin_first`
      and `compare_to_reference`), and the five entries of `specs/deferred_items.md`
      § 17-text-to-code-ita-panel. As read (plan 20, Handoff, Resolved at resume, 2026-11-21):
      Stage 4's six read records and `gap.json`, with RUN and DATA_FP. DATA_FP is the table build,
      under Stage 2's split, that every Stage 5 read must match. `decision/ita_reads.py`'s
      `read_reference` and `tools ita-read`, with the d16 run directories, are a reusable read
      path. Stage 4's reads are reference reads, so they leave `check_ita_margin_first` open, and
      the margin record makes its own fresh reads (text-to-code spec §4.5). The two entries of
      § 20-text-to-code-reference-read. A known trap: the level-mode `CAPTURED_HERE` pin fails on
      this Mac (§ 19-economic-decision-read), so a local full-suite run shows one failure until
      that entry is done. D2's answer is yes, and it changes nothing here: an ITA training part is
      Req 28's Later, which needs a later spec and G1.
      Produces: Text→code's margin record. Each candidate's checkpoint, table and query path, and
      text→code's decision record, which names the adopted candidate or says that the reference
      stands. Embedding roadmap Stage 12's text→code opening reads them.
      Exit: The stage spec lists every table or part it reads, with its one role. The margin
      record exists, from reads of the reference only, and precedes every candidate read. No ITA
      pair entered training. Every selection read is an ITA validation read, logged after the
      margin record. Every ITA read is of Stage 4's table build, DATA_FP (plan 20, Handoff). The
      decision record exists. Each entry of § 17-text-to-code-ita-panel and
      § 20-text-to-code-reference-read is done by its own condition, or carries a written why
      (plans 17 and 20, Handoffs, Resolved at resume, 2026-11-18 and 2026-11-21). The suite,
      Ruff, the style gate and the strict docs build pass.
      ROUTING: brainstorming
      Stage 5: HELD lifted (2026-11-18), gates read at embedding-data fd9e495 (Gates, above). G2
      is open for selection on the ITA validation part. G3 is open. G4 is open for the ITA pairs
      through `specs/evaluation-protocol.md` Section 1, whose pinned run and conditions C1–C6 bind
      every read. G5 is open for the ITA pairs. Re-read 2026-11-21 at 4942e1d: unchanged.
      Its `Consumes` are met. The resume of 2026-11-21 routed it to brainstorming, which starts
      after that resume's PR merges.

- [ ] Stage 6: Relatedness — stage spec and judging panel
      Objective: Write the relatedness stage spec and build its judging panel on fixtures of
      whole held-out embedding-data families, with no real read.
      Spec: Req 27 (all but Training input), Req 26, Req 25, U18–U22, U29, U30, Verification "One
      role per table or part", Rollout note item 4.
      Gap closed: Req 27 (the judging panel, on fixtures).
      Consumes: The codebook and its tree metric. The frozen text-only table (roadmap D9,
      `panels/text_only.py`). The decision tooling (`decision/resampling.py`,
      `decision/rule.py`) and the selection log. `graph_model/`, which the relatedness stages may
      reuse (Req 27, ADR 0001). The deferred entries that Deferred items names.
      Produces: The relatedness stage spec: the families that train and those that judge, the
      split unit where one family is split, the scores, the margin rule, the method and the place
      of the sealed opening. The judging panel on fixtures. Stages 6b and 7 read them. The
      families that the spec names to train are the input to the owner's R8.4 decision (G1).
      Exit: The stage spec names every family it reads, with its one role. The panel reads only
      the families that the spec names as judging, under a committed fingerprint (test). No real
      embedding-data table was read. The suite, Ruff, the style gate and the strict docs build
      pass.
      ROUTING: brainstorming
      Split at resume (2026-11-18, owner: "Approve split (Recommended)"). The real reference read
      and the margin record moved to Stage 6b. Stage 6 reads no embedding-data table, so no gate
      holds it, and its spec gives the input that G1 and the relatedness half of G4 wait on.
      Stage spec approved 2026-11-18 (`specs/relatedness-artifact.md`, U48, PR #170, ec02cf9). It
      names co-ownership to judge and the 2023 ITA area mix to train (§5), and its header routes
      the plan to writing-plans. Its planning starts after the resume of 2026-11-19 merges
      (Decisions, 2026-11-19).

- [ ] Stage 6b: Relatedness — reference read and margins
      Objective: With the user's approval at the time, read the reference (the better of the
      tree's distance and the text-only table) on the judging families, and record the margins
      before any model exists.
      Spec: Req 27 (Judging), Req 26, Req 25, U29, U30, Verification "One role per table or part"
      and "Gates held".
      Gap closed: Req 27 (the reference and the margins).
      Consumes: Stage 6's stage spec and judging panel, through its Handoff. The codebook and its
      tree metric, the frozen text-only table, the decision tooling and the selection log, as for
      Stage 6.
      Produces: The relatedness margin record, from reference reads only. Stage 7 reads it.
      Exit: The margin record exists, from reads of the tree's distance and the text-only table
      only. Each family that was read had passed its count gate and has a section in
      embedding-data's protocol spec. The record names the embedding-data commit and each table
      that it read. No model was read.
      ROUTING: writing-plans
      Stage 6b: HELD (2026-11-18; gates re-read 2026-11-21 at embedding-data 4942e1d) — waits on
      G4 and G6 for co-ownership, the judging family that Stage 6's spec names. G3 and G5 are open
      for it (Gates, above).

- [ ] Stage 7: Relatedness — model and decision
      Objective: Train the relatedness embedding on the tree's edges and the families that Stage
      6 named for training, and record its adoption decision on the held-out families.
      Spec: Req 27 (Training input, Judging and Output), Req 26, U17, U22, U30, Verification
      "Gates held".
      Gap closed: Req 27 (the artifact and its validation test; the sealed decision is Stage 7b's).
      Consumes: Stage 6's stage spec and panel, and Stage 6b's margin record, through their
      Handoffs. `graph_model/`, as for Stage 6.
      Produces: The flat relatedness embedding of the 2,125 codes, each seed's table with its
      provenance, and the forward candidate's validation test record, which names the forward
      candidate or says that the reference stands. Stage 7b reads both (relatedness spec §8;
      Resolved at resume, 2026-11-19).
      Exit: No embedding-data table entered training before G1 opened, and each training family
      had passed its count gate. No co-ownership row entered training, and only co-ownership's
      validation part was read for selection (U43; Resolved at resume, 2026-11-19). The
      validation test's record exists, against Stage 6b's margins. The suite, Ruff, the style gate
      and the strict docs build pass.
      ROUTING: writing-plans
      Stage 6's spec chose the method, GraphSAGE (U46, spec §4.7), so this routing stands
      (Resolved at resume, 2026-11-19). Whether a fixture build runs before G1 opens is the resume
      after Stage 6's choice (Decisions, 2026-11-19).
      Stage 7: HELD (2026-11-14; gates re-read 2026-11-21 at embedding-data 4942e1d) — waits on G1
      and G4 for the 2023 ITA area mix, its one training family, and on G6 for selection on
      co-ownership's validation part. G3 and G5 are open for both families (Gates, above). If G1
      never opens, this stage parks (Req 27).

- [ ] Stage 7b: Relatedness — sealed opening
      Objective: With the user's approval at the time, open co-ownership's sealed part once, score
      the forward candidate's seeds and the chosen reference on it, and record relatedness's
      adoption decision.
      Spec: Req 27 (Judging), Req 26, Req 25, U30, U45, relatedness spec §4.6 (the test, with the
      validation δ) and §4.8, Verification "One role per table or part" and "Gates held".
      Gap closed: Req 27 (the sealed decision).
      Consumes: Stage 6's judging panel, with its one logged opening of the sealed part. Stage 6b's
      margin record. Stage 7's forward candidate, which beat the reference by δ in the validation
      test, with its seeds' tables and that test's record. All through their Handoffs. If Stage
      7's record says that the reference stands, this input is unmet: the relatedness chain ends at
      Stage 7, and the sealed part stays sealed for a later candidate (relatedness spec §4.6,
      "Which read decides"; Resolved at resume, 2026-11-19).
      Produces: The logged opening of co-ownership's sealed part, the sealed scores, and
      relatedness's decision record, which names the adopted embedding or says that the reference
      stands. A written finding.
      Exit: The selection log shows exactly one opening of co-ownership's sealed part, after every
      validation read that selected anything. The test used the δ fixed on validation. The
      finding names the embedding-data commit and the SHA-256 of each table that it read.
      ROUTING: writing-plans
      Added at resume (2026-11-19, owner: "Approve both (Recommended)"), from relatedness spec
      §4.8 (U45). Embedding roadmap Stage 12 keeps text→code's opening only.
      Stage 7b: HELD (2026-11-19; gates re-read 2026-11-21 at embedding-data 4942e1d) — waits on
      G4 for co-ownership's sealed part (Gates, above). G6 does not hold it, because the opening
      judges and selects nothing. Stage 7's gates hold it
      through its `Consumes`.

## Stage-spec stamp

Every stage spec's Rollout note carries this line, which writing-plans copies verbatim into the
stage plan's header. A stage with no stage spec puts it in its plan's header:

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage N — on plan completion, tick the stage
> and re-validate later stages against what shipped.

On completion, a stamp goes under the stage's entry, and it is authoritative:

> Stage N: COMPLETE (YYYY-MM-DD) — implemented by plan <id> (path). Its Handoff section names
> what <stages> consume.
> Next: resume the roadmap.

A resume that lifts a HELD stamp replaces it with a line that gives each named gate's state and
the embedding-data SHA that it read (Decisions).

## Completion

Retirement is gated behind a conformance audit of the accumulated system. Re-run the gap rubric
over Reqs 25–29 with evidence per verdict (implementing stage and plan, Deviation notes and
deferred-items entries), and re-check the spec's Verification items. The audit also re-reads each
gate in embedding-data at its current SHA. An unmet requirement exits exactly two ways: a new
stage, with the roadmap staying live, or a conscious deferral with a written why. This is not a
whole-roadmap code review, because each stage is reviewed when it merges. Text→code's sealed
opening in embedding roadmap Stage 12 is not part of this audit. Relatedness's is Stage 7b, and
the economic artifact has none (Decisions, 2026-11-19).

Parking is a first-class exit. If G1 never opens, the relatedness chain's unticked stages park
alone (Req 27, ADR 0002). Each appends to `specs/deferred_items.md` as a self-contained item, and
the other chains stay live. If the whole roadmap parks, its unticked stages append in the same
way, and this file moves to `specs/completed/` marked `PARKED (date) — N of 9 stages complete`.
The spec stays in `specs/` while a stage spec or embedding roadmap Stage 12 cites it.
