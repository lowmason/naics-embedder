# Text→code candidates and decision — Design Spec

**Status:** APPROVED (2026-11-22, U54). Written in a brainstorming session. A
builder check raised three points. The user ruled on one (U53, the input length). The other two
are folded into §4.4 as my calls: the outcome-best epoch is selectable, and no checkpoint is
deleted.

**Roadmap:** `specs/naics-artifact-redesign-roadmap.md`, Stage 5 (ROUTING: brainstorming). Source
spec: `specs/naics-artifact-redesign.md` (Q4) Req 28 (Training, and its last bullet), Req 26,
split Req 19, U25 and U27. Stage 2's approved spec, `specs/completed/text-to-code-ita-panel.md`
(called "Stage 2's spec" here), gives the margin rule (§4.5) and the hand-off to this stage (§7).
Evidence was read at origin/main 1b62c17, and in embedding-data (`~/Projects/embedding-data`) at
3382744. Paths are under `src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/`
or `docs/`, or name a top-level file.

**Next skill after approval:** `writing-plans` in a fresh Opus session, with this file as the
handoff. This spec asks for two plans (§7). Execution then runs on Sonnet in its own fresh session.

**Design provenance.** A brainstorming session with the user on 2026-11-22. The inputs read were:

- the roadmap's Stage 5 entry, its Gates paragraph and its Decisions of 2026-11-19 to 2026-11-22
- Q4's Req 28, Stage 2's spec, and embedding roadmap Stage 9's entry
  (`specs/naics-embedding-roadmap.md`, HELD, whose questions moved here)
- the findings text-to-code-gap (Stage 4) and geometry-by-dimension (§3, §7)
- `decision/text_to_code.py`, `decision/ita_reads.py`, `panels/ita.py`, `utils/training.py`
  (`ModelCheckpoint`), `text_model/shared_encoder.py`, `remote/model_cache.py`,
  `utils/input_window.py`, `conf/config.yaml`, and the Stage 8 d16 runs' `epoch_summary.jsonl`
- `specs/deferred_items.md`, the entries that the roadmap's Deferred items paragraph gives Stage 5
- embedding-data's `specs/evaluation-protocol.md` (Section 1, C1–C6),
  `specs/completed/ita-establishment-targets.md` and `specs/embedding-evaluation-data-spec.md`.

- **Locators.** (Req n) and (Pn) cite Q4's spec. (Un) cites a ruling below or in an earlier spec.
  "Stage N" alone is the redesign roadmap's. "Stage 9" with no qualifier is embedding roadmap
  Stage 9.
- **My calls.** Choices I made while writing, which the user has not ruled on, are marked
  (my call).

**User rulings (2026-11-22).** Rulings continue from U48, so each number is unique across the
specs. Each ruling is the user's words as given.

| Id | Question | Ruling |
|---|---|---|
| U49 | Where do the ITA checkpoint-selection reads run? | "Mac, after training" |
| U50 | Which candidate set does this round train? | "Backbone first" |
| U51 | Which two backbones does this round train? | "gte-small + bge-small" |
| U52 | How does the native qualification run? | "First campaign run" |
| U53 | Which input length do gte-small and bge-small use? | "128 for every arm" |
| U54 | Do you approve the committed Stage 5 spec? | "Approve, push PR" |

## 1. Intent

Text→code returns a ranked shortlist of six-digit codes for an owner's own description of an
establishment (Req 28). Stage 4 measured how much harder owners' words are than held-out Census
phrases. For the reference `euclidean-d16`, the code-matched gap C − H is 0.1631 [0.1363, 0.1909]
(text-to-code-gap §5). This stage asks one question: does a different backbone, or the same model
selected on owner text, close part of that gap by more than the margin δ?

The stage does three things, in this order:

1. It records text→code's margin from fresh reads of the two reference arms only.
2. It trains three candidates and one control, and selects each seed's checkpoint on the ITA
   validation part.
3. It records text→code's adoption decision: one candidate goes forward to the sealed opening, or
   the reference stands.

## 2. Goals, non-goals and anti-goals

**Goals.**

- A margin record that comes before every candidate read in the selection log.
- Each candidate's checkpoints, tables and query path, with five seeds per arm.
- A decision record that names the candidate that goes forward, or says that the reference stands.
- A finding that reports the reads and the decision.
- The seven deferred entries that the roadmap gives this stage, each done or with a written why
  (§4.9).

**Non-goals.**

- **The one-factor ablations.** MoE fusion, the term weight, the information-content (IC) target
  and the channel-presence indicator wait for a later stage on the backbone that wins (U50). §7
  asks the next roadmap resume to add that stage.
- **An ITA training part.** The finding recommends one (text-to-code-gap §6). It needs a later spec
  and G1 (roadmap Decisions, 2026-11-21, D2).
- **The sealed opening.** It stays in embedding roadmap Stage 12.
- **A change to the text objective.** §4.2 keeps it as it is.
- **A text-only comparator for each backbone** (embedding roadmap D9). The text→code rule compares
  a candidate with the reference on ITA, not with a text-only table (my call).

**Anti-goals.** These would make the stage a failure even if it ran:

- an ITA pair or ITA description that enters training, a curriculum or negative mining (C3)
- an ITA candidate read logged before the margin record, which refuses every later margin record
  (Stage 2's spec §4.5)
- an ITA table, or a part of one, that leaves the Mac (U49)
- a read of the sealed part, or of the outcome test split
- a candidate that changes more than one factor against the reference, so that a win cannot be
  read.

## 3. Approaches

The session settled four choices. Each one had two or three real options.

**Where selection reads run (U49).**

- **On the Mac, after training (chosen).** Training keeps more checkpoints. The Mac reads them on
  ITA validation into the main checkout's log. The tables stay on the Mac, and the margin-first
  check reads one log. The cost is disk space and hours of reads.
- **In the training monitor on the Lambda instance.** C1 checks the SHA-256 of each whole table, so
  the sealed rows would go to the cloud. A validation-only extract fails C1 as written. ITA would
  also steer the learning-rate plateau and early stopping, and the margin check would have to merge
  logs.
- **Keep outcome selection in each run, and use ITA only to choose among candidates.** This is the
  cheapest. It changes approved text (Req 28 "checkpoint selection", Stage 2's spec §7).

**How wide the round is (U50).**

- **Backbone first (chosen).** Three candidates and a control, about 20 runs. The factor ablations
  wait for the winning backbone, because a MoE result on MiniLM need not carry over.
- **The full Stage 9 list.** About 40 runs, and two factors (IC, channel-presence) that are not
  built yet.
- **Only the factors that exist today.** MoE and the term weight, with no new backbone. This is the
  least likely to help on owner text, because those factors shape code geometry learned from Census
  text only.

**Which backbones (U51).**

- **`thenlper/gte-small` and `BAAI/bge-small-en-v1.5` (chosen).** Each has 33.4M parameters,
  the BERT architecture and an MIT license. A run costs about twice a MiniLM run (12 layers against
  6). Two strong models of one size compare cleanly.
- **`gte-small` and `BAAI/bge-base-en-v1.5`.** The base model has 109.5M parameters, at about five
  times the cost. It tests size, but it mixes size with model family.
- **`bge-base-en-v1.5` and `sentence-transformers/all-mpnet-base-v2`.** The most capacity and the
  largest campaign.

**The native qualification (U52).**

- **The first campaign run (chosen).** Seed 1 of `euclidean-d16-ita` runs first, on a fresh
  experiment from public main. It qualifies the public code, and its outcome MRR checks training
  against Stage 8's d16 seed 1.
- **A separate throwaway run.** A cleaner record, at the cost of one more run.

## 4. Design

### 4.1 The arms

| Arm | Backbone | Changes against `euclidean-d16` | Role |
|---|---|---|---|
| `euclidean-d16-ita` | `sentence-transformers/all-MiniLM-L6-v2` | The selector only: ITA validation (§4.6) | Candidate |
| `gte-small-d16` | `thenlper/gte-small` | The backbone | Candidate |
| `bge-small-d16` | `BAAI/bge-small-en-v1.5` | The backbone | Candidate |
| `frozen-minilm-d16` | `sentence-transformers/all-MiniLM-L6-v2` | LoRA off: the backbone's weights do not train | Control |

- **Every other setting is `euclidean-d16`'s.** Geometry Euclidean, dimension 16, masked-mean
  fusion, term weights 1, the bundle 301cce28, the input length 128 (U53), the 40-epoch budget and
  the schedule. The run settings differ only in the backbone, its tokenizer and summaries, and the
  frozen switch (my call).
- **The budget stays at 40 epochs (my call).** d16's outcome MRR was still rising at epoch 39 on all
  five Stage 8 seeds, and four of the five selected epoch 39. A longer budget would mix a training
  effect into every comparison. The finding reports this limit. A later stage may change the budget
  as its own factor.
- **Five seeds per arm, 1 to 5** (Stage 9 Exit).
- **The frozen control (my call).** Its projection and head train, and the backbone's weights do
  not. It shows how much of d16's score comes from fine-tuning the backbone on Census text. It is
  read and reported, and it never goes forward (§4.7).
- **Names.** `euclidean-d16-ita` is a candidate arm name, not `euclidean-d16`. So its reads never
  pass as reads of a reference arm under `REFERENCE_ARMS` (`decision/text_to_code.py`).

### 4.2 Training targets above six digits (Req 28)

No change (my call). Each query keeps its own level's candidates and named targets, as
`supervision/queries.py` builds them today, and the code-code term stays over all codes. The
shortlist is read at six digits, and the coarser levels are reported (Stage 2's spec §4.3). A change
to the objective in this round would mix with the backbone effect. A later stage may take it as its
own factor.

### 4.3 Backbones in training and on the instance

A new backbone needs four things before it trains. Each comes from that backbone's own model card
and files.

1. **A trained window** in `TRAINED_WINDOWS` (`utils/input_window.py`). It comes from the model's
   documentation, as Stage 9's Produces asks. The plan records the card's value and the
   `max_position_embeddings` and tokenizer `model_max_length` that it checked. Both cards are
   expected to give 512.
2. **The run's length is 128 for every arm (U53).** `data_loader.tokenization.max_length` stays
   128, as MiniLM's is. `check_window` allows a length below the trained window. So the window is
   not a second factor, and the cost estimates in §3 and §4.6 hold. The finding notes that the two
   new backbones never read a text longer than 128 tokens, which may understate them.
3. **Summaries at 128 for each new backbone.** `resolve_channel_texts` tests a text against the
   run's length, so the texts that overflow 128 for MiniLM (753 today) likely overflow for gte and
   bge too. Each new backbone gets its own artifact from `naics-embedder data summaries --backbone
   <name>` at window 128, pinned in `WINDOW_SUMMARIES` (`panels/window_summaries.py`). All three
   backbones appear to share BERT's 30,522-token uncased vocabulary. The plan checks this, and
   reports whether the new summaries equal MiniLM's. If they differ, each arm reads its own, as
   Stage 9 allows.
4. **A revision pin in the offline model cache.** `remote/model_cache.py` pins MiniLM's revision.
   Each new backbone gets its own pinned 40-character revision, checked the same way before launch.
5. **The tokenizer.** `data_loader.tokenization.tokenizer_name` follows the backbone. The token
   cache is a preprocessing artifact for each backbone.

- **Pooling.** The encoder mean-pools every backbone. The bge card recommends its CLS token. The
  spec keeps mean pooling for all arms, so the pooling is not a second factor (my call). The finding
  reports each card's recommended pooling beside the scores.
- **The backbone check (deferred: "nothing ties the bundle's backbone to training at runtime").**
  Training refuses a backbone that has no `TRAINED_WINDOWS` entry. It also refuses one with an
  over-window channel text and no `WINDOW_SUMMARIES` pin. The bundle's
  `manifest.input_window.backbone` records the window under which the bundle was built. It does not
  bind the training backbone, because the bundle's targets and texts do not depend on it (my call).
  The check is verified against bundle 301cce28 with each of the three backbones.
- **The summaries-key watch (deferred: "an export keys the summaries on its tokenizer").** For
  gte-small and bge-small the tokenizer name is the backbone name, as for MiniLM. The plan checks
  this. If it holds, the entry stays a watch with a written why. If not, the plan fixes the refusal
  message that the entry names.

### 4.4 Checkpoint retention

Today `outcome_checkpoint` keeps one checkpoint, the outcome-best epoch, plus `last.ckpt`
(`utils/training.py`, `save_top_k=1`). For Stage 5's arms, the outcome-best checkpoint and
`last.ckpt` stay as they are. Training also keeps the checkpoint of every 4th epoch, epochs 3, 7,
…, 39, and the last epoch (my call: every 4th plus the last halves the reads again against every
2nd).

- **The ITA-selectable set** of a seed is those grid epochs, its last epoch and its outcome-best
  epoch, each once. That is at most 12 checkpoints per seed, at about 91 MB each for MiniLM. The
  outcome-best epoch is in the set so that `euclidean-d16-ita` can select the epoch that the
  reference's own rule selected (my call).

- **Outcome MRR still drives the run.** It drives `ReduceLROnPlateau`, early stopping and the
  outcome-best checkpoint. The outcome panel is report-only for text→code's decision (Stage 2's spec
  §4.6). Its role in the run's schedule is unchanged.
- **An early-stopped run** keeps its kept epochs up to the stop and its last epoch.
- **The kept set is authoritative.** `remote/canonical.py` and `text_model/checkpoint_runner.py`
  check a kept set and an earliest-best monitor epoch today. The plan extends those checks to the
  retained epochs, so a pull or a preflight refuses a run with a missing or extra kept file.
- **Exact resume.** The retention setting enters the run settings that exact resume compares, so a
  resume cannot change it.
- **Disk.** This stage deletes no checkpoint (my call). The kept set is authoritative, so a deleted
  file would make every later pull and preflight of the run fail, embedding roadmap Stage 12's read
  and the ablation stage's reuse of the runs among them. The round needs about 30 GB on the Mac,
  which has over 400 GB free. Plan B files a deferred entry for a later cleanup that keeps each
  seed's selected epoch, its outcome-best epoch and `last.ckpt`, and records the deletion in the
  kept-set record.

### 4.5 The margin record

With the user's approval at the time, `fix_ita_margin` reads the lexical stub and d16's five
Stage 8 seeds on the ITA validation part, under one fresh run id with role `reference`. It chooses
the reference and fixes δ = 0.05 × that arm's headline MRR (Stage 2's spec §4.5).

- **First.** It runs before any candidate read and before any candidate arm trains. No training
  monitor reads ITA under U49, so training can start before the record, but no ITA candidate read
  may (my call: the record still comes first in time, to keep the order plain).
- **The log** is the main checkout's `logs/selection_log.jsonl`, passed by absolute path. A read
  from a worktree with the default relative path would write that worktree's log.
- **Its reads name their inputs.** Each d16 read names its seed's checkpoint SHA-256 and table
  matrix fingerprint, and both must equal the Stage 8 record's
  (`records/stage8/euclidean-d16.json`) before anything is logged.
- **Its table build** is Stage 4's, DATA_FP `8980e358…` (text-to-code-gap §2). A read on another
  build is refused before it logs.
- **Stage 4's reads are not reused.** They stay gap evidence.

### 4.6 Selection reads on the Mac

After the margin record and after each arm's runs are pulled and verified, the Mac reads each kept
epoch of each candidate seed, and each of the control's, on the ITA validation part.

- **Each read is logged** to the main checkout's log, with role `candidate`, its arm, seed, epoch, a
  fresh run id, the checkpoint's SHA-256 and the table's matrix fingerprint. The control's reads
  also have role `candidate`, with `control: true` in the detail (my call: a new role would let its
  reads pass the margin-first check unseen).
- **Each kept epoch is exported first** with `tools export-table`, so that `ArmEncoder.from_files`
  reads a table with its provenance.
- **The seed's checkpoint** is the earliest kept epoch with the highest ITA validation headline
  MRR. The reads of the other epochs stay in the log and count as candidate reads.
- **Order.** All kept epochs of a seed are read before that seed's checkpoint is chosen. The
  reads of one arm run as one batch, so the log shows a contiguous block per arm (my call).
- **Time.** About 140 s per MiniLM read and about twice that per gte or bge read (Stage 4's read
  times). With up to 12 epochs and 5 seeds per arm, the round is about 13 to 14 hours of reads.
  The plan says how it runs them, for example in batches with a drift check before each.

### 4.7 The decision

- **The test** is Stage 2's spec §4.5: Δ = candidate − reference on the headline MRR, from the
  paired two-stage bootstrap. A candidate beats the reference when the lower bound of the 95 %
  interval on Δ exceeds +δ.
- **Which candidate goes forward (my call).** Among the candidates that beat the reference, the one
  with the highest point headline MRR. A tie goes to the cheaper backbone, MiniLM before the two
  small models, and gte before bge by name. If none beats the reference, the reference stands.
- **The control** is reported beside the candidates and is never tested for adoption.
- **Several candidates.** Each candidate's test is one comparison on one panel, as Stage 2's spec
  §4.5 reads it (n = 1). The sealed read at embedding roadmap Stage 12 guards against a candidate
  that won on validation by chance (my call: the spec adds no correction for three candidates).
- **The decision path** refuses a margin record whose data fingerprint differs from the compared
  results' (§4.9).
- **Report-only.** Phrasings, copied pairs, the coarser levels, conflict counts and the outcome
  validation scores never decide.

### 4.8 The campaign

- **The native qualification (U52).** Seed 1 of `euclidean-d16-ita` runs first, on a fresh
  experiment from public `origin/main` with no private commit. This qualifies the public code for
  this campaign, and the deferred entry "Public native-qualification reconciliation" records it. The
  run's outcome MRR trajectory is compared with Stage 8's d16 seed 1. CUDA training is not bitwise
  reproducible, so the plan states a tolerance before the run. The other runs wait for this one to
  pass.
- **Order (my call).** Then the other four seeds of `euclidean-d16-ita`, then `frozen-minilm-d16`,
  `gte-small-d16` and `bge-small-d16`, with seed 1 of each gated before its other seeds, as Stage 8
  ran.
- **Transport.** The remote workflow's `up`, `train`, `sync`, `finish` and `status`, as
  `.claude/rules/remote.md` gives them. The canonical bundle 301cce28 is uploaded, never rebuilt.
  No ITA table, store or Mac log goes to an instance.
- **`remote finish`'s overhead** grows with the number of runs (deferred: "remote finish"). The plan
  says whether it pulls one run at a time to bound it.
- **Each real step needs the user's approval at the time:** the margin read, the qualification run,
  the campaign, the selection reads and the decision read.

### 4.9 Deferred entries this stage takes

| Entry (`specs/deferred_items.md`) | Disposition |
|---|---|
| § 7: nothing ties the bundle's backbone to training at runtime | Done by §4.3's check, with a test against bundle 301cce28 |
| § 9: an export keys the summaries on its tokenizer | §4.3: checked for each backbone. It stays a watch with a written why, or is fixed |
| § 17: margin δ against the candidate's table build | Done: the decision path refuses a mismatched data fingerprint, with a test |
| § 17: margin-first across selection logs | Written why: under U49 no ITA read is logged outside the Mac's log. The check also refuses an `ita_text` record in a pulled run's log, with a test (my call) |
| § 17: a candidate read before any margin blocks every later margin | Written why: §4.5 puts the margin record first. The entry stays open as a design item |
| § 17: margin reads name no checkpoint | Done by §4.5, with a test |
| § 17: margin record self-check gaps | Done: `check_ita_margin` requires reads of both reference arms and refuses a repeated seed, with tests, before the margin record |
| § 20: `tools ita-read` accepts `--outcome` on a d16 read | Done: the read path refuses `--outcome` on `euclidean-d16` and logs the stub's outcome read when its record must carry C, with a test for each |
| § 20: `tools ita-read` does not check a d16 seed's checkpoint and table | Done by §4.5's check before any log record, with a test |
| § 10: Public native-qualification reconciliation | §4.8: the first campaign run qualifies this campaign's public source |

The roadmap's Deferred items paragraph counts the § 17 and § 20 entries as seven. The table also
lists the two embedding-roadmap entries and the qualification entry that the roadmap gives Stage 5.

## 5. One role per table or part (Req 26)

| Table or part | Read by | Role for text→code |
|---|---|---|
| Codebook text in the bundle | Training | Training (U25) |
| Codebook codes and lineage | Panel | The codebook: candidates and ancestors, no role |
| `ita_text_pairs`, validation part | The margin record, the selection reads and the decision | Selection |
| `ita_text_pairs`, sealed part | Nothing in this stage | Judging, at embedding roadmap Stage 12 only |
| `ita_text_pair_codes` | Panel | Follows its pair's part and role |
| `ita_text_pair_members`, `naics_2022_paths` | Nothing | Not read |
| Outcome panel, validation split | Training's schedule, and reports | Drives each run's schedule and early stopping, reported beside ITA, decides nothing |
| Outcome panel, test split | Nothing | Stays sealed |
| Regressor panels | Nothing | Not read: they judge the economic artifact |

No ITA pair enters training (C3). Every ITA read is of Stage 4's table build, DATA_FP.

## 6. Gates and constraints

- **Gates, read at embedding-data 3382744.** Since 0e8312c, embedding-data added only the
  USAspending vendor-links rows to its spec (R1.1, R1.2, R7, R8.3). `specs/evaluation-protocol.md`,
  `specs/completed/ita-establishment-targets.md`, R8.4 (now spec `:578`) and V9 did not change. G2
  is open for selection on the validation part. G3, G4 (Section 1) and G5 are open for the ITA
  pairs. The plan re-reads them at its start and cites the SHA.
- **C1–C6 bind every read.** In particular C1: each table's SHA-256 is checked against
  `conf/data/ita_evidence.yaml` before the first read.
- **The user approves each real read and run at the time** (the Stage 5 entry's Objective).
- **`uv.lock`.** New backbones need no new package. If a plan step changes the lock, it says so. No
  economic read is planned, so the economic pairing does not bind (roadmap, Across chains).
- **The run-settings identity changes on purpose.** `CLAUDE.md` keeps the literal 21-key
  `run_settings` identity and the finished artifacts unchanged. The retention setting and the frozen
  switch extend that identity, so `remote/canonical.py`, `remote/sync.py`, exact resume and
  `tools sweep`'s `ArmSpec` see them. Each new setting defaults to today's behaviour. A checkpoint
  without the new keys, such as Stage 8's, still loads, exports and reads, and compares equal to a
  config that leaves the new settings at their defaults. Stage 8's records stay valid. §4.5's
  margin read depends on this. `CLAUDE.md` is updated to name the new count.
- **The selection log is append-only.** No step deletes or rewrites a line of it.
- **The known local failure.** The level-mode `CAPTURED_HERE` pin fails on this Mac (§ 19), so a
  local full-suite run shows one failure until that entry is done. The plans report it and do not
  fix it.
- **Parallel work.** Stage 7a may run at the same time. It works under `graph_model/`, `decision/`
  and `conf/`. Each plan checks open PRs and branches for the paths it touches before it starts.
- **Style and docs.** The repository's conventions apply. New modules get API pages in `docs/` and
  entries in `docs/.nav.yml`. `.claude/rules/text-to-code.md` and `CLAUDE.md` are updated where
  the reads, tools or traps change.

## 7. Handoff

- **Two plans (my call), as Stage 4 split into PR A and PR B.**
  - **Plan A: code on fixtures.** Retention (§4.4), the backbone pins and check (§4.3), the frozen
    switch (§4.1), the selection read and the per-seed choice (§4.6), the margin and decision
    paths (§4.5, §4.7) and the deferred entries (§4.9). No real read and no real run.
  - **Plan B: the real stage.** The margin read, the qualification and the campaign, the selection
    reads, the decision read and the finding. Each step waits for the user's approval at the time.
- **Embedding roadmap Stage 12** reads the decision record. If a candidate goes forward, its
  checkpoint, table and query path are what the sealed opening reads.
- **The next roadmap resume** adds a stage for the one-factor ablations (MoE, term weight, IC and
  channel-presence) on the backbone that this stage's decision names, or on MiniLM if the
  reference stands (U50). It also edits Stage 5's entry, whose Objective names the ablations.
- **Embedding roadmap Stage 9** stays HELD. The resume records that its backbone question was
  answered here and that its ablations moved to the new stage.

## 8. Left to the builder

- The settings' names: the frozen switch, the retention schedule and the backbone check.
- How the selection read reuses `read_reference` and `tools ita-read`, or adds a candidate command.
- The shape of the decision record and of the per-seed selection record.
- The qualification run's tolerance, stated before the run.
- How the reads are batched, and the drift check before each batch.

## 9. Verification — observable outcomes (roadmap Stage 5, Exit)

Plan A (fixtures):

- [ ] Training keeps the checkpoints of epochs 3, 7, …, 39, the last epoch and the outcome-best
      epoch, each once, and the pull and the preflight refuse a run with a missing or extra kept
      file (test).
- [ ] An early-stopped run keeps its grid epochs up to the stop, its last epoch and its
      outcome-best epoch (test).
- [ ] Exact resume refuses a changed retention setting or frozen switch (test).
- [ ] A Stage 8-shaped checkpoint without the new settings loads, exports and reads, and compares
      equal to a config at the new settings' defaults (test).
- [ ] Training refuses a backbone with no trained window, and one with an over-window text and no
      summaries pin, and accepts each of the three backbones with bundle 301cce28 at length 128
      (test).
- [ ] The summaries generator and `data summaries` write an artifact at a given window below the
      trained window, refuse one above it, and keep MiniLM's default artifact unchanged (test).
- [ ] The frozen control's backbone weights do not change over a step, and its projection does
      (test).
- [ ] A seed's selected checkpoint is the earliest kept epoch with the highest headline MRR (test).
- [ ] A d16 margin read with a checkpoint or table that is not the Stage 8 record's is refused
      before any log record (test).
- [ ] `check_ita_margin` refuses a record without reads of both reference arms, and one with a
      repeated seed (test).
- [ ] The decision path refuses a margin record whose data fingerprint differs from the results'
      (test).
- [ ] The decision chooses the forward candidate by §4.7, never the control, and lets the reference
      stand when none beats it (test).
- [ ] Each § 17 and § 20 entry, and the § 7 and § 9 entries, is done by its own condition or carries
      a written why.

Plan B (real, each step approved):

- [ ] The margin record exists, from reads of the reference arms only, and precedes every candidate
      read in the main checkout's log.
- [ ] The qualification run passed its stated tolerance before any other run started.
- [ ] No ITA pair entered training, and no ITA table went to an instance.
- [ ] Every selection read is an ITA validation read, logged after the margin record, of DATA_FP.
- [ ] The log holds no `ita_text` opening and no read of the outcome test split.
- [ ] The decision record exists and names the forward candidate, or says that the reference stands.
- [ ] The finding reports every arm's scores, the control, the pooling notes and the budget limit,
      and names the embedding-data commit, the tables' SHA-256 and the log's SHA-256 before and
      after.

Both plans: the suite, Ruff, the style gate and the strict docs build pass.

## Rollout note

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 5 — on plan completion, tick the stage
> and re-validate later stages against what shipped.
