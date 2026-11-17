# Text→code ITA panel — Design Spec

**Status:** APPROVED (2026-11-17, U41). Written in a brainstorming session. A
builder check raised three points, none of them the user's to rule. They are folded into §4.1
(the node-id level token), §4.4 and §4.5 (margin-first in code) and §4.5 (the reference's
checkpoints).

**Roadmap:** `specs/naics-artifact-redesign-roadmap.md`, Stage 2 (ROUTING: brainstorming). Source
spec: `specs/naics-artifact-redesign.md` (Q4) Reqs 25, 26 and 28, U24, U25, U27, U28 and P3–P5.
Evidence was read at origin/main 673bb21, and in embedding-data (`~/Projects/embedding-data`) at
ec6fb29. Paths are under `src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/`
or `docs/`, or name a top-level file.

**Next skill after approval:** `writing-plans` (plan 17) in a fresh Opus session, with this file
as the handoff. Execution then runs on Sonnet in its own fresh session.

**Design provenance.** A brainstorming session with the user on 2026-11-17. The inputs read were:

- Q4's spec, the roadmap's Stage 2, 4 and 5 entries, and the owner ruling of 2026-11-16 on
  text→code's reference (roadmap, above Gap analysis);
- `panels/outcome.py`, `panels/decoding.py`, `panels/selection_log.py`,
  `panels/lexical_encoder.py`, `text_model/arm_encoder.py`, `decision/resampling.py` and
  `decision/rule.py`;
- the findings outcome-panel-splits §5 (the lexical stub) and geometry-by-dimension §3
  (`euclidean-d16`);
- embedding-data's `specs/ita-establishment-targets.md` (its pair tables, called Stage 10 here)
  and `specs/completed/code-tables-ita-naics-mapping.md` (node ids), both at ec6fb29.

- **Locators.** (Req n) and (Pn) cite Q4's spec. (Un) cites a ruling below or in Q4's spec.
  (Stage 10) cites embedding-data's `ita-establishment-targets.md` at ec6fb29. "Stage N" alone is
  the roadmap's.
- **My calls.** Choices I made while writing, which the user has not ruled on, are marked
  (my call).

**User rulings (2026-11-17).** Rulings continue from U35, so each number is unique across the
specs. For a multiple-choice question, the ruling is the user's words as given.

| Id | Question | Ruling |
|---|---|---|
| U36 | Which population should the headline MRR stand for: owners (each pair counts once, P4) or phrasings (each distinct description counts once)? | "Owners as headline, phrasings reported beside it" |
| U37 | How is a headline pair with more than one reported code scored: any code counts, each code with weight 1/k, or out of the headline? | "Any reported code counts, report the conflict count" |
| U38 | How large is r, the relative margin? Four calls were played back with it: copied means the pair-level `_any` flags; a salted hash split of `description_key`, 50/50; the outcome test split stays sealed at Stage 12; δ = r × the reference's ITA validation headline MRR, fixed from reference reads only, with description groups as the resampling unit | "r = 0.05, your calls stand" |
| U39 | How is the panel built: a separate ITA panel, a shared base class with `OutcomePanel`, or one generalized scorer? | "A, separate ITA panel" |
| U40 | The existing rule makes δ a non-inferiority margin, which never binds on one panel. Should text→code beat the reference by δ, or use the existing rule? | "Beat by δ" |
| U41 | Do you approve the committed spec? | "Approve the spec" |

## 1. Intent

An owner describes an establishment in their own words and asks for its NAICS code (U24).
Text→code answers with a ranked shortlist of six-digit codes (Req 28). Until now, text→code was
judged on the outcome panel, whose queries are Census index entries. Q4 replaces that judge with
ITA employer-written descriptions: text written by real owners, for this exact purpose.

This stage builds that judge, and nothing else. It writes the split, the scoring, the reference
and the margin rule, so that Stages 4 and 5 can read real pairs under rules fixed before anyone
sees a score. It reads no real ITA pair (G4 does not block work on fixtures).

**Success.** On a fixture with Stage 10's pair schema, the panel splits, scores, logs and seals
exactly as this spec says, and the margin rule is code that Stage 5 runs without a new decision.

## 2. Goals, non-goals and anti-goals

**Goals.**

1. A deterministic split of the ITA pairs into a validation part and a sealed part, grouped by
   normalized description (Req 28, U27, U38).
2. Shortlist scoring of an encoder on a part, with the headline, the copied, the phrasing and the
   conflict reports (P3–P5, U36, U37).
3. Logged reads and one logged opening of the sealed part. A second opening needs a reason.
4. The reference choice and the beat-by-δ margin rule, as code (owner ruling 2026-11-16, U38,
   U40).
5. The role of every table and part this stage, Stage 4 and Stage 5 read (Req 26).

**Non-goals.**

- A CLI command that reads real pairs. Stage 4 adds it, from this API (my call: Stage 4's plan
  has the real paths and pins, and nothing in this stage runs on real data).
- Choosing among trained candidates, training on ITA text and the training monitor's selector.
  Those are Stage 5's (§7).
- Embedding-data's evaluation protocol for other tables.

**Anti-goals.** The design fails, even if it runs, when it:

- lets one normalized description appear in both parts;
- lets a rebuilt pair table count as a new sealed part, so a second opening passes without a
  reason;
- counts copied taxonomy text in the headline;
- scores a description against a code it does not report, or quietly drops a reported code;
- changes `OutcomePanel`, `score_decoding`, `compare_panel` or any logged record shape that
  `decide` checks.

## 3. Approaches

| Approach | Trade-off | Ruling |
|---|---|---|
| A. A separate ITA panel in `panels/ita.py`, with its own shortlist scorer that reuses `decoding.py`'s helpers, and the margin rule in `decision/` | Leaves `OutcomePanel` and `score_decoding` untouched. Costs about 60 lines that resemble the outcome panel's open, read and log logic | Chosen (U39) |
| B. A shared sealed-panel base class for both panels | Less repetition, but it rewrites the class that selects text checkpoints today. A changed log field or fingerprint makes `decide` refuse logged records | Rejected |
| C. One scorer that takes several true codes per query | One scorer, but it changes a function whose per-query output earlier results depend on | Rejected |

For the margin, the existing non-inferiority rule (`decision/rule.py` `compare_panel`) was
rejected: on one decision panel its δ never binds (U40).

## 4. Design

### 4.1 Inputs and code mapping

The panel reads two of Stage 10's tables, `ita_text_pairs.parquet` and
`ita_text_pair_codes.parquet`, and the bundle's codebook. It validates the columns it uses
against Stage 10's schema and refuses a table that lacks one.

- **Panel pairs.** Pairs with `flag_headline = true`. Stage 10's headline rule already excludes
  empty, placeholder and unknown-employer pairs. Other pairs are not read.
- **Reported codes.** For a panel pair, its pair-code rows with `flag_headline_basis = true`.
  Each such row's `naics_2022_node_id` has the form `naics:2022:national_industry:<code>`.
  The level is a name, not a digit count (verified at ec6fb29: `classification_tables.py`
  `NAICS_LEVELS`, and `leaf()` in `tests/test_ita_text_pairs.py`). The panel takes `<code>` and
  refuses, before any read:
  - a node id of another system, vintage or level than `naics`, `2022` and `national_industry`;
  - a code that is not one of the bundle's six-digit candidates;
  - a panel pair with no reported code.

  Refusal, not a drop, follows `score_decoding`, which refuses a query code that is not a
  candidate (my call). Stage 10 builds headline codes only from native six-digit leaves, so a
  refusal means that the inputs disagree.
- **Query text.** Each pair's `representative_string`, the most frequent delivered original. The
  normalized key is casefolded and is not what an owner typed (my call). A null string refuses.
- **Candidates.** The bundle's six-digit codes (1,012), the same set the outcome panel decodes to.
  Ancestors come from the codebook lineage (`utils/naics_hierarchy.code_lineage`), so the
  combined sectors 31–33, 44–45 and 48–49 count as one sector each (Req 25, U17).

### 4.2 The split

- **Unit.** `description_key`. Every pair with one key falls in one part (Req 26, Req 28).
- **Rule.** A part comes from the SHA-256 of a fixed salt and the key. The first 8 bytes of the
  hash, as an unsigned integer modulo 2, give the part: 0 is validation, 1 is sealed (U38; the
  byte detail is my call).
- **Config.** `conf/data/ita_panel.yaml` holds the salt, the rule version, the panel name and r.
  The salt is chosen once, when the file is first committed, and never changes (my call).
- **Stability.** The rule depends on the key alone. A rebuilt pair table puts an old key in its
  old part and a new key in a part of its own, with no stored assignment table.
- **Split fingerprint.** SHA-256 of the canonical JSON of the rule: salt, version, hash
  algorithm, modulus and part names. It identifies the split, not the rows (§4.4).

### 4.3 Scoring

The scorer ranks all candidates for each pair's query under the encoder's own distance. It uses
`decoding.py`'s distances and its rank rule: ties count against the truth.

- **Rank of a pair (U37).** The best rank over its reported codes. Each code's rank counts every
  candidate at a distance no greater than its own. A pair's other reported codes rank behind its
  best code, so they never count against it.
- **Per-pair metrics.** Reciprocal rank, hit@1, hit@5 and hit@10 (the existing `HIT_KS`), and,
  for the coarser levels (P5):
  - `lca_level`: the highest LCA level between the top-1 code and any reported code;
  - for each level L in 2–5, the first shortlist position whose code shares a reported code's
    level-L ancestor, with its reciprocal rank and hit@k (my call: this reads the shortlist at
    each level, which is what an owner who only knows their sector sees).
- **Headline (U36, P3, P4).** The mean over panel pairs that are not copied. A pair is copied
  when `flag_copied_title_any or flag_copied_index_any` (U38). Each pair counts once.
- **Copied report (P3).** The same metrics over copied pairs, apart from the headline.
- **Phrasing report (U36).** The mean over description groups of the mean over each group's
  non-copied pairs (my call). Every distinct description counts once. A group that holds pairs
  with different codes keeps them all.
- **Counts.** Per part: pairs, description groups, copied pairs, conflict pairs
  (`flag_headline_codes_conflict`, U37), and pairs with mojibake or company-fallback flags. These
  flags do not change the headline. Stage 10 leaves them to the protocol, and this spec keeps
  them as counts only (my call).
- **Result shape.** A per-pair frame (pair id, description key, reported codes, rank and
  metrics), a per-code frame, and a summary for each of headline, copied and phrasing.

### 4.4 The panel, its reads and the sealed opening

`ItaPanel` in `panels/ita.py` mirrors `OutcomePanel`'s behavior and shares no class with it
(U39). It writes to the same `SelectionLog` under its own panel name, `ita_text`.

- **Reads.** `score_logged(encoder, part, purpose, detail)` appends a `read` record before it
  encodes any query, as `OutcomePanel.score_logged` does. The record names the panel, the part,
  the split fingerprint, the number of pairs, and in its detail the encoder, the distance and the
  part's data fingerprint. The caller's detail names the arm (`arm`), its seed where it has one,
  its role (`reference` or `candidate`) and a fresh run id (`run`), as the economic reads do
  (my call). The margin-first check (§4.5) reads these fields.
- **Data fingerprint.** SHA-256 of what a read scores: each panel pair's id, query text, reported
  codes and copied flag, in pair order, and the candidates. It is logged with each read and is
  not a read itself.
- **Sealed part.** Reads of the sealed part fail until `open_sealed(purpose, reopen_reason=None)`
  is called on the panel object. That call checks `log.openings('ita_text', split_fingerprint)`.
  A prior opening without a non-empty `reopen_reason` is refused. Otherwise it appends `open`, or
  `reopen` with the reason.
- **Why the split fingerprint keys openings.** A rebuild in embedding-data changes the rows, so a
  row fingerprint would let a second opening pass as a first. The split fingerprint does not
  change on a rebuild, so the refusal holds.
- **Encoders.** Any object with `encode_queries` and `encode_codes`, as `QueryCodeEncoder`
  defines. `ArmEncoder` and `LexicalTrigramEncoder` both qualify. The distance is the encoder's
  own: cosine for the stub, as in outcome-panel-splits §5.

### 4.5 Reference and margin rule

- **Reference arms (owner ruling, 2026-11-16).** The lexical stub (outcome-panel-splits §5) and
  `euclidean-d16` (geometry-by-dimension §3), each through the query path the outcome panel uses.
  `euclidean-d16` keeps its recorded checkpoints and tables, one per seed, which the outcome panel
  selected. It is not selected again on ITA validation (my call): the reference is a recorded
  arm, and selecting it again would make it a candidate. Candidates selected on ITA validation
  may gain on that part, and the sealed read guards against it.
- **Choosing the reference.** The arm with the higher ITA validation headline MRR as a point
  estimate. `euclidean-d16`'s MRR is the mean over its seeds. A tie goes to the stub, the cheaper
  arm (my call). The choice is a selection read, so it happens in Stage 5's margin record, after
  G2 opens. Stage 4 reads both arms and chooses nothing.
- **The margin.** δ = 0.05 × the chosen reference's ITA validation headline MRR (U38). It is fixed
  from reference reads only, before any candidate read. The margin record holds r, both arms'
  MRRs, the chosen arm, δ, the split fingerprint, the data fingerprint and `fixed_at`.
- **Margin-first, in code (my call, after the economic pattern).** The margin record makes its
  own fresh reference reads under a new run id, as `fix_relative_margins` does. Stage 4's reads
  stay gap evidence and are not reused. A check, like `check_economic_margins_first`, refuses a
  margin record when the log holds any `ita_text` validation read with role `candidate`, or with
  no role, made before `fixed_at`. Training-monitor reads count as candidate reads, so Stage 5
  records the margin before any candidate trains against ITA validation.
- **The test (U40).** Δ = candidate MRR − reference MRR on the headline. A candidate beats the
  reference when the lower bound of the 95% interval on Δ exceeds +δ. The interval is
  `percentile_interval(replicates, superiority_level(1))`, two-sided, as the existing rule reads
  it. The headline is the one decision panel, so n = 1 and the level is 95%.
- **Intervals.** The paired two-stage bootstrap in `decision/resampling.py`: description groups
  are the units and headline pairs the items, with each arm's seeds nested within the resample.
  The stub counts as a one-seed arm. Replicates and the bootstrap seed come from
  `conf/data/decision.yaml`, as the existing decisions use them.
- **Which read decides (Req 28).** Stage 5 uses the validation part to choose one candidate to go
  forward. A candidate that does not beat the reference by δ on validation does not go forward,
  and the reference stands. At text→code's sealed opening (embedding roadmap Stage 12), the same
  test runs on the sealed part with the δ fixed on validation. If it fails there, the reference
  stands.
- **Report-only.** Phrasings, copied pairs, conflict counts and outcome-panel scores never decide
  and never add to n.
- **Code.** A module in `decision/` (the plan names it) holds the reference choice, the margin
  record and the beat-by-δ test as functions with fixture tests. It does not change
  `compare_panel` or the economic rule. Stage 2 runs it on fixtures only.

### 4.6 The outcome panel

For text→code, the outcome panel's validation split is reported beside the ITA scores and decides
nothing (Req 28). Its test split stays sealed at embedding roadmap Stage 12: a sealed split spent
on a report cannot judge anything later (U38). This is the ruling that Stage 2's `Produces` asks
for. The outcome panel still selects trained economic candidates' checkpoints (split Req 23).

## 5. One role per table or part (Req 26)

| Table or part | Read by | Role for text→code |
|---|---|---|
| Codebook text in the bundle (titles, descriptions, examples, exclusions, index entries, cross-reference phrases) | Training | Training (U25) |
| Codebook codes and lineage | Panel | The codebook: candidates and ancestors (P1), no role |
| `ita_text_pairs`, validation part | Stages 4 and 5 | Selection |
| `ita_text_pairs`, sealed part | Embedding roadmap Stage 12 | Judging, read once |
| `ita_text_pair_codes` | Panel | Follows its pair's part and role |
| `ita_text_pair_members`, `naics_2022_paths` | Nothing | Not read. Node ids carry the code (§4.1) |
| Outcome panel, validation split | Reports | Reported only, decides nothing (Req 28) |
| Outcome panel, test split | Nothing | Stays sealed (§4.6) |

No ITA pair enters training (U25). Stage 2 reads fixture tables only.

## 6. Constraints

- **Gates.** G2 holds selection on the validation part (Stage 5). G3, G4 and G5 hold every read
  of real pairs (Stages 4 and 5). This stage reads none.
- **Untouched seams.** `OutcomePanel`, `score_decoding`, `compare_panel`, the economic decision
  and every logged record shape that `decide` checks stay as they are.
- **Fixtures.** A fixture builds Stage 10's two tables with its exact column names and types. It
  holds copied pairs, conflict pairs, several employers per description, a node id of another
  level, and a code outside the candidates. Its outputs go under `tmp_path`.
- **`uv.lock`.** This stage needs no new dependency. If a plan step would change the lock, it
  stops and says how the economic reads still pair (roadmap, Across chains).
- **Style and docs.** The repository's conventions apply. The new modules get API pages in
  `docs/` and entries in `docs/.nav.yml`.

## 7. Handoff to later stages

- **Stage 4** adds the read command, pins the embedding-data commit and the two tables' SHA-256
  in its finding, and reads both reference arms on the validation part.
- **Stage 5** records the margin (§4.5), chooses the forward candidate on validation, and replaces
  the outcome-panel MRR as text→code's checkpoint selector with ITA validation reads (Req 26,
  U27). Today the training monitor selects text checkpoints on the outcome panel, which this spec
  makes report-only for text→code.
- **Embedding roadmap Stage 12** opens the sealed part once and applies §4.5's test. It does not
  open the outcome test split.

## 8. Left to the builder

- Module names in `decision/`, the shape of the result types, and the fixture builder.
- How the scorer shares `decoding.py`'s distance and rank helpers without changing
  `score_decoding`.
- The salt's value.

## 9. Verification — observable outcomes (roadmap Stage 2, Exit)

- [ ] On the fixture, each normalized description falls in one part only, and a rebuilt fixture
      with an added pair keeps every old key in its old part (test).
- [ ] The headline excludes copied pairs and counts each distinct pair once. The phrasing report
      counts each description once (test).
- [ ] A pair's rank is the best rank over its reported codes, and the coarser-level matches read
      the codebook lineage, combined sectors included (test).
- [ ] A node id of another level (for example `naics:2022:industry:517112`), or a code outside
      the candidates, is refused before any read (test).
- [ ] A margin record is refused when the log holds an earlier ITA validation read with role
      `candidate` or no role (test).
- [ ] A second opening of the sealed part is refused without a reason, also after the fixture
      tables are rebuilt. With a reason, it logs `reopen` (test).
- [ ] The reference choice, δ = 0.05 × MRR and the beat-by-δ test give the expected verdicts on
      fixture replicates, including a candidate that is superior to the reference but does not
      beat it by δ (test).
- [ ] `OutcomePanel`, `score_decoding` and `compare_panel` are unchanged, and their tests pass
      unchanged.
- [ ] §5 lists every table or part this stage, Stage 4 and Stage 5 read, with its one role.
- [ ] No real ITA pair was read.
- [ ] The suite, Ruff, the style gate and the strict docs build pass.

## Rollout note

Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 2
