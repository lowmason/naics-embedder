# NAICS artifact redesign — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

**Status: APPROVED (2026-11-14).** Derived in an interactive session. The user answered the
derivation's three questions (Decisions, below) and approved the seven-stage partition at the
human checkpoint ("Approve, hand off Stage 1"). Stage 1 is handed off. Stage 2 waits for the next
resume. Stages 4–7 are HELD on gates in embedding-data (Gates, under Stages).

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

## Gap analysis

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 25 | implemented-differently | `CLAUDE.md:5-7`, `README.md:8-11` and `docs/index.md:3-6` name Q3's hierarchy artifact, and `CLAUDE.md:7` judges text→code on the outcome panel. Embedding roadmap Stage 12 (`specs/naics-embedding-roadmap.md:1093-1094`) gives the hierarchy no opening and opens the outcome test split for text→code. The codebook is the bundle's (`supervision/`). | The codebook, and the economic artifact's own reference and margins, exist. No relatedness artifact exists. The derivation discharges the docs half (Decisions). Stages 1–7 build the rest. |
| 26 | out-of-repo | embedding-data at 60068eb: R8.4 is unamended (spec `:550`). V9 runs "after the first full build" (`:606-609`, `:669-670`), at embedding-data Stage 13, which is unticked (roadmap `:123`, `:299`). Stage 10's anti-goal is unchanged (`ita-establishment-targets.md:124`, `:92-93`). The successor protocol spec does not exist (spec `:634`). In this repository, no `src/` module reads an embedding-data table. | The owner gates live in embedding-data (Gates, under Stages). For the existing panels, the one-role rule already holds through the selection log and `decide`'s validation-only reads (`decision/decide.py`). Each stage spec lists its tables and parts with one role each. |
| 27 | missing | `graph_model/` holds the parked HGCN only, and nothing reads a relation family. embedding-data: the ITA area mix is built (embedding-data Stage 6) with `size_gate_pending`. Co-ownership (embedding-data Stage 10) is not built, and none of plan 11's modules is in its `src/`. ACS (Stage 9) and CPS (Stage 11) are unticked there. | Every family waits for its build and its count gate. Training also waits for R8.4. Stages 6 and 7. |
| 28 | implemented-differently | Training reads the codebook's own text only (`supervision/queries.py:9-19`). Judging uses the outcome panel (`panels/outcome.py:159`, `:217`). No ITA reader exists, and embedding-data Stage 10's pairs are not built. | Spec: judged on the ITA validation and sealed parts, with the outcome panel reported beside them. Code: judged on the outcome panel. Stages 2, 4 and 5. |
| 29 | implemented-differently | The test is built: `fix_relative_margins` and `decide_economic` (`decision/economic.py:179`, `:537`) and `tools growth-margins` (`cli/commands/tools.py:562`). The margins were fixed once (finding: growth-margins), and `uv.lock` is still 6de4fbdc. `read_deterministic_arm` (`decision/economic.py:250`) has no command. Trained candidates are deferred (`specs/deferred_items.md:622`). | No candidate exists. The Q3 audit gave its candidates row the same verdict. Stages 1 and 3. |

No in-code-but-not-in-spec row: the parked HGCN stays with naics-embedding Reqs 15 and 17, and the
outcome panel stays under Req 28.

## Stages

**Sequencing.** The stages form one chain per artifact (ADR 0002): economic (Stages 1 and 3),
text→code (Stages 2, 4 and 5) and relatedness (Stages 6 and 7). They follow the Rollout note's
items 2–4, and item 1 was done at derivation (Decisions). The spec ranks nothing beyond that
order, so this order follows the spec's. The stages with no gate come first: Stages 1 and 2 need
nothing from embedding-data, and Stage 3 needs only Stage 1. Each chain measures before it builds
models (split Core principle). The economic margins exist already. Stage 2 fixes text→code's
scoring and margin rule, and Stage 5 records the margins before its first candidate read. Stage
4's gap read selects nothing, so G2 does not hold it (Rollout note, item 3). Stage 6 fixes
relatedness's margins before Stage 7 trains. No stage's `Consumes` names a stage of another
chain. The sealed openings are not stages here. Text→code's and the economic artifact's stay in
embedding roadmap Stage 12 (Req 28, ninth resume), and relatedness's goes where Stage 6's spec
puts it (Req 27). Stage 7 routes to writing-plans because Stage 6's spec chooses the method
(Req 27). If that spec leaves the method open, the resume after Stage 6 asks to route Stage 7 to
brainstorming. The next stage to hand off is the first unticked one whose `Consumes` are met and
that carries no HELD stamp. Plans start at 16.

**Across chains.** Economic reads pair only where the reference predictions match the 2026-10-17
margin read bit for bit, under `uv.lock` 6de4fbdc (finding: growth-margins). Until Stage 3's
decision exists, a stage spec that changes `uv.lock` says how those reads still pair (ADR 0002,
Negative). The text→code and economic chains both build on `text_model/` (ADR 0001), so a change
that one chain makes there reaches the other chain's next training run.

**Gates in embedding-data, read at 60068eb.** They are not `Consumes` (Decisions). Each HELD stamp
names the gates it waits on.

- **G1, R8.4.** No embedding-data table trains an artifact until the owner amends it (Req 26).
  State: unamended, "Benchmark tables are never training data" (spec `:550`). Holds Stage 7.
- **G2, Stage 10's anti-goal.** It must allow selection on the ITA validation part (Req 26, U27).
  State: unchanged (`ita-establishment-targets.md:124`, and `:92-93` under Out of scope). Holds
  Stage 5.
- **G3, the V9 Rollout sentence.** It must allow a per-family count gate (Req 26, U29). State:
  unchanged. V9 runs "after the first full build" (spec `:606-609`, `:669-670`), at embedding-data
  Stage 13 (roadmap `:123`, `:299`), which is unticked. Holds Stages 4–7.
- **G4, the successor evaluation-protocol spec** (Decisions). For text→code, it assigns ITA
  scoring and splitting to text→code's spec. For relatedness, it covers the families that Stage
  6's spec names, because the protocol for tables other than the ITA pairs stays with
  embedding-data (the spec's Out of scope). State: none exists (spec `:634`). Holds Stages 4–7.
- **G5, each family's build and count gate** (Req 26, U29). A family is read only after it is
  built and has passed its own V9 count gate under G3. State per family:
  - ITA text–code pairs and co-ownership (embedding-data Stage 10, unticked): not built. Plan 11
    is on that main, and none of its modules (`ita_text_pairs.py`, `ita_co_ownership.py` and
    others) is in `src/`.
  - ITA area mix (embedding-data Stage 6, complete): built, with `size_gate_pending`.
  - ACS families (embedding-data Stage 9, unticked, its acceptance on hold) and CPS job-change
    flows (embedding-data Stage 11, unticked): not built.

  The ITA pairs hold Stages 4 and 5. The families that Stage 6's spec names hold Stages 6 and 7.

**Deferred items.** `specs/deferred_items.md` was read, not edited. The entries that the spec
names (Deferred items this touches) go with their chains. Stage 1 takes trained economic
candidates (`:622`) and the predictions-to-table binding (`:640`), whose trigger the candidates'
spec may meet. The other plan-14 entries (`:633`, `:650`) touch the same module and may go with
it. Stage 5 takes the entries that moved with embedding roadmap Stage 9: nothing ties the bundle's
backbone to training at runtime (`:276`), and the summaries-key watch (`:481`). Stages 6 and 7
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

- [ ] Stage 2: Text→code — ITA scoring and split
      Objective: Write text→code's stage spec and build its ITA panel on a fixture: the grouped
      split into a validation and a sealed part, the shortlist scoring, the reference and margin
      rule, and the one logged opening of the sealed part.
      Spec: Req 28 (Output, Judge, What the ITA text is like, The outcome panel stays and Owner
      gate), Req 26, Req 25 (own reference, margins and opening), U24, U25, U27, U28, P3–P5,
      Supersessions (the text→code rows), Verification "One role per table or part", Rollout note
      item 3, Stage 10 (the pair table's schema and copied-text flags, at embedding-data 60068eb).
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

- [ ] Stage 3: Economic — decision read
      Objective: With the user's approval at the time, read every Stage 1 candidate on the growth
      validation splits on the Mac, and record the economic decision against the fixed margins.
      Spec: Req 29 (The test, and the margins of 2026-10-17), split Req 24, Supersessions
      (Unchanged), Verification "Economic candidates".
      Gap closed: Req 29 (the decision).
      Consumes: Stage 1's candidates, read command and Handoff. The margin record, read at
      origin/main 30aeff6 on macOS arm64 with Python 3.12.12 under `uv.lock` 6de4fbdc (finding:
      growth-margins, Sources and §1). Every candidate read runs under the same three, because
      `decide` pairs only bit-identical reference predictions (`decision/economic.py:414`). The
      lock on main is still 6de4fbdc at beaee8f. The main checkout's selection log
      (`logs/selection_log.jsonl`), in which the margin read found no opening of either growth
      regime (finding: growth-margins §3). The user's approval, asked when the stage starts.
      Produces: The economic decision record over `growth_seen` and `growth_heldout`, which
      names the adopted candidate or says that the reference stands, and a written finding.
      Embedding roadmap Stage 12's economic opening reads both.
      Exit: The decision record exists and names each candidate's text-and-tree provenance.
      `tools growth-margins` did not run, and the decision reads the 2026-10-17 record. The
      selection log gained only validation reads, and no outer set opened. The finding cites the
      record's sha256 and the platform, Python and lock that it was read under.
      ROUTING: writing-plans

- [ ] Stage 4: Text→code — reference read and gap
      Objective: With the user's approval at the time, read the reference on the real ITA
      validation part and measure the gap between Census phrases and owner text.
      Spec: Req 28 (Judge, What the ITA text is like, and Later), Req 26 (Size gate), U25, U27,
      Verification "Gates held".
      Gap closed: Req 28 (the reference read and the measured gap).
      Consumes: Stage 2's ITA panel, split, reference and margin rule, through its Handoff.
      Produces: The reference's validation scores. A finding that reports the gap, on headline
      and copied pairs apart, and says whether a later spec should add an ITA training part
      (Req 28, Later). Stage 5 reads both.
      Exit: The selection log holds the reference's ITA validation reads and no opening of the
      sealed part. No margin record was written, and nothing was selected. The finding reports the
      gap with its intervals, and it names the embedding-data commit and the pair table that it
      read.
      ROUTING: writing-plans
      Stage 4: HELD (2026-11-14) — waits on G3, G4 and G5 for the ITA text–code pairs (Gates,
      above).

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
      names.
      Produces: Text→code's margin record. Each candidate's checkpoint, table and query path, and
      text→code's decision record, which names the adopted candidate or says that the reference
      stands. Embedding roadmap Stage 12's text→code opening reads them.
      Exit: The stage spec lists every table or part it reads, with its one role. The margin
      record exists, from reads of the reference only, and precedes every candidate read. No ITA
      pair entered training. Every selection read is an ITA validation read, logged after the
      margin record. The decision record exists. The suite, Ruff, the style gate and the strict
      docs build pass.
      ROUTING: brainstorming
      Stage 5: HELD (2026-11-14) — waits on G2, G3, G4 and G5 for the ITA text–code pairs (Gates,
      above).

- [ ] Stage 6: Relatedness — judging panel and margins
      Objective: Write the relatedness stage spec and build its judging panel on whole held-out
      embedding-data families. Then, with the user's approval at the time, read the reference
      (the better of the tree's distance and the text-only table) and record the margins before
      any model exists.
      Spec: Req 27 (all but Training input), Req 26, Req 25, U18–U22, U29, U30, Verification "One
      role per table or part" and "Gates held", Rollout note item 4.
      Gap closed: Req 27 (the judging panel, the reference and the margins).
      Consumes: The codebook and its tree metric. The frozen text-only table (roadmap D9,
      `panels/text_only.py`). The decision tooling (`decision/resampling.py`,
      `decision/rule.py`) and the selection log. `graph_model/`, which the relatedness stages may
      reuse (Req 27, ADR 0001). The deferred entries that Deferred items names.
      Produces: The relatedness stage spec: the families that train and those that judge, the
      split unit where one family is split, the scores, the margins, the method and the place of
      the sealed opening. The judging panel and its margin record. Stage 7 reads them.
      Exit: The stage spec names every family it reads, with its one role. The panel reads only
      the families that the spec names as judging, under a committed fingerprint (test). The
      margin record exists, from reads of the tree's distance and the text-only table only. Each
      family that was read had passed its count gate. The suite, Ruff, the style gate and the
      strict docs build pass.
      ROUTING: brainstorming
      Stage 6: HELD (2026-11-14) — waits on G3, G4 and G5 for the families its spec would name
      (Gates, above). Req 27 lets this spec be written before G1 opens.

- [ ] Stage 7: Relatedness — model and decision
      Objective: Train the relatedness embedding on the tree's edges and the families that Stage
      6 named for training, and record its adoption decision on the held-out families.
      Spec: Req 27 (Training input, Judging and Output), Req 26, U17, U22, U30, Verification
      "Gates held".
      Gap closed: Req 27 (the artifact and its decision).
      Consumes: Stage 6's stage spec, panel and margin record, through its Handoff.
      `graph_model/`, as for Stage 6.
      Produces: The flat relatedness embedding of the 2,125 codes, with its provenance, and its
      decision record, which names the adopted embedding or says that the reference stands. The
      sealed opening that Stage 6's spec places reads both.
      Exit: No embedding-data table entered training before G1 opened, and each training family
      had passed its count gate. No judging family was read for training or selection. The
      decision record exists, against Stage 6's margins. The suite, Ruff, the style gate and the
      strict docs build pass.
      ROUTING: writing-plans
      Stage 7: HELD (2026-11-14) — waits on G1, G3, G4 and G5 for its training families (Gates,
      above). If G1 never opens, this stage parks (Req 27).

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
whole-roadmap code review, because each stage is reviewed when it merges. The sealed openings in
embedding roadmap Stage 12 are not part of this audit.

Parking is a first-class exit. If G1 never opens, the relatedness chain's unticked stages park
alone (Req 27, ADR 0002). Each appends to `specs/deferred_items.md` as a self-contained item, and
the other chains stay live. If the whole roadmap parks, its unticked stages append in the same
way, and this file moves to `specs/completed/` marked `PARKED (date) — N of 7 stages complete`.
The spec stays in `specs/` while a stage spec or embedding roadmap Stage 12 cites it.
