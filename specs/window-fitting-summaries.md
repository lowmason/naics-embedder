# Window-fitting summaries — Design Spec

**Status:** APPROVED (2026-10-04)

**Roadmap:** `specs/naics-embedding-roadmap.md`, Stage 6b (ROUTING: brainstorming). Source spec
`specs/naics-embedding.md` at d9126ce. Evidence read at origin/main f5c8307. Paths are under
`src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/` or `docs/`.

**Next skill after approval:** `writing-plans` (plan 9) in a fresh Opus session; execution then
runs on the Sonnet default.

## 1. Purpose

Every channel text whose marked form exceeds the backbone's trained window is cut at the window
today (`text_model/fields.py`, `tokenize_field`, `truncation=True`). Replace that tail truncation
with frozen extractive summaries that fit the window, so that Stage 7 trains the reference
configuration, and fixes δ, on texts no tokenizer cuts. The summaries live in their own committed,
hash-pinned artifact, which the tokenization cache and the D9 text-only comparator both read, and
whose hash every link from checkpoint to decision record carries.

## 2. Scope

### 2.1 In scope

- Req 9, input windows, under the user's ruling of 2026-10-03 (S1): no channel text is truncated.
- Req 3, leakage: no summary can contain a held-out query (4.4).
- D9: the comparator reads the arm's text, now identified by the descriptions' and the
  summaries' hashes (4.8).
- Three parts of plan 8's deferred entries (section 10).

### 2.2 Out of scope

- **Queries.** Index entries and activity phrases go through `tokenize_field` unchanged.
- **The bundle.** Bundle 301cce28 and its descriptions (sha256 `fe8c54e3…`) stay byte-identical
  (S2). Plan 7's deferred manifest tokenizer revision keeps its trigger.
- **Stage 9:** summaries for other backbones or windows. The pin is keyed by backbone (4.6), so
  Stage 9 adds entries; it does not change this design.
- **Training and panel reads.** The Exit trains nothing and reads no split (section 7).
- **The rest of plan 8's deferred entries.**

## 3. Rulings

From the session brief, not re-triaged:

- **S1.** Over-long texts are summarized, not truncated (user ruling, 2026-10-03, Req 9).
- **S2.** Bundle 301cce28 and its descriptions stay unchanged. Summaries live in their own
  committed, hash-pinned artifact.
- **S3.** This spec settles which channels get summaries, and whether they are extractive or
  abstractive.
- **S4.** Auditing abstractive summaries against the validation and test queries is a
  non-selecting read of sealed text, which needs the user's explicit approval. S5 makes it moot:
  no step of this stage reads a validation or test query.

From the brainstorm, the user's answers on 2026-10-04:

- **S5. Method.** Extractive, for every channel. A summary is whole source units, in source
  order; no word is written that the source does not contain in that unit.
- **S6. Selection.** Backbone centrality (4.3).
- **S7. Design.** Sections 4.1–4.9 as presented: the units, the artifact, a code pin rather than a
  config key, one resolver for both readers, and the summaries' hash in the checkpoint contract,
  both provenances and the decision records.

## 4. Design

### 4.1 Which texts

A channel text is **over the window** when its marked form, `marker(field) + text`
(`text_model/fields.py`), tokenizes to more than the window under the backbone's tokenizer,
special tokens included, without truncation. Only over-window texts get summaries; every other
text is read verbatim.

Measured on 2026-10-04 on the descriptions `fe8c54e3…` under MiniLM's tokenizer (revision
1110a243) at the 128-token window:

| Channel | Present | Over the window | Over, unmarked | Levels 2–3 over |
|---|---:|---:|---:|---:|
| title | 2,125 | 0 | 0 | 0 |
| description | 2,111 | 162 | 153 | 75 (16 of 20 sectors, 59 of 96 subsectors) |
| examples | 1,075 | 106 | 105 | 0 |
| excluded | 1,117 | 485 | 464 | 3 |

The unmarked column is what the roadmap and Stage 6's Rollout note recorded, and what bundle
301cce28's manifest records under `input_window`. The cache tokenizes marked text, so the marked
column is the population that is truncated today: 753 rows. Each marker is two tokens under
MiniLM's tokenizer, so a summary's text budget is the window minus the two special tokens and the
two marker tokens: **124 tokens**. The budget is computed from the tokenizer and the marker, never
hard-coded.

### 4.2 Units

A summary is built from units, and every unit boundary is a boundary of the leakage segmenter
(`panels/leakage.py`: a description or exclusion text splits after `.` or `;` followed by
whitespace, `_SENTENCE_BREAK`; the examples channel splits at `'; '`, `EXAMPLES_SEPARATOR`). That
property carries the leakage argument (4.4).

- **title:** the whole title is one unit. A title over the window cannot be summarized, so the
  build raises; none is over under MiniLM.
- **examples:** the entries, split at `EXAMPLES_SEPARATOR`.
- **description and excluded:** the segmenter's pieces, merged into units at three levels:
  1. **sentence:** a unit closes at a piece that ends in `.`, or, for excluded, in `.` or `;`, so
     that each cross-reference is one unit;
  2. **clause:** a unit closes at a piece that ends in `.` or `;`;
  3. **piece:** each segmenter piece is a unit.

  At levels 1 and 2 a piece closes a unit only when the unit's parentheses balance (as many `(` as
  `)`) and the piece does not end in an abbreviation or a list numeral: `U.S.`, `i.e.`, `e.g.`,
  `etc.`, `No.`, `vs.`, a bare numeral such as `1.`, or a parenthesized one such as `(1);`. The
  probe's reference pattern for that test was
  `(?:\bU\.S|\bi\.e|\be\.g|\betc|\bNo|\bvs|^\d+|\s\d+|\(\d+\))[.;]$`. A text's units are its level-1
  units; a unit over the budget is re-split at level 2, and a level-2 unit over the budget at
  level 3. A level-3 piece over the budget raises. On 2026-10-04 no piece of any over-window text
  exceeded 124 tokens.

The segmenter's break pattern becomes a public name in `panels/leakage.py` (`SENTENCE_BREAK`), so
the unit builder and the leakage checker share one definition.

### 4.3 Selection (S6)

The frozen backbone (`sentence-transformers/all-MiniLM-L6-v2` at revision 1110a243, pretrained
weights with no adapter, on CPU in float32, loaded with `panels/text_only.py`'s `load_backbone`)
reads each unit unmarked and alone, and mean-pools its last hidden state over the attention mask.
Each unit vector is L2-normalized; its weight is its token count without special tokens. The
target is the weighted sum of all of the text's unit vectors.

The greedy selection:

1. Of the units whose marked summary fits the window on its own, keep the one whose weighted
   vector has the highest cosine with the target.
2. Repeatedly add the unit that most raises the cosine between the kept units' weighted sum and
   the target, among the units that keep the marked summary, joined in source order, within the
   window (each candidate tokenized as it would be read, special tokens included).
3. Stop when no remaining unit fits, or none raises the cosine strictly.

Exact float ties go to the earlier unit. The kept units are emitted in source order, joined by
`' '` (description, excluded) or `EXAMPLES_SEPARATOR` (examples). Every non-final source unit ends
in `.` or `;`, and the source's last unit can only be last in a summary, so the segmenter
re-splits each summary into exactly its kept pieces.

**Why centrality.** The cache's channel vector is a mean-pooled embedding, and truncation damages
exactly that vector. Centrality keeps the units whose pooled vector best approximates the whole
text's. On 2026-10-04 a probe compared each rule's summary, embedded as one marked text, with the
whole text's weighted unit-vector sum (mean cosine over the over-window texts):

| Channel | Truncation | Lead | Coverage | Centrality |
|---|---:|---:|---:|---:|
| description | 0.903 | 0.880 | 0.872 | 0.910 |
| examples | 0.776 | 0.770 | 0.777 | 0.777 |
| excluded | 0.885 | 0.870 | 0.894 | 0.902 |

The proxy favours centrality by construction, since centrality optimizes nearly the same target,
so it shows consistency, not superiority; Stage 7's panels are the real test. The gain over
truncation is in the tail (excluded p10 0.826 → 0.859, minimum 0.68 → 0.79). Centrality keeps a
description's first sentence in 115 of 162 texts without being told to.

**Determinism.** A rebuild on the build platform reproduces the selection. On another platform a
float near-tie could flip a choice, so the committed artifact is the authority, checked by its
invariants (fit, subset, keys; 4.7), never by rebuild equality.

### 4.4 Leakage without a sealed read

Each summary's segments, as `training_text_segments` cuts them (`text_segments` for description
and excluded, its normalized entries for examples), are a subset of its source text's segments
(4.2, 4.3). Bundle 301cce28's build checked every segment of the descriptions `fe8c54e3…` against
the validation and test queries, exactly and as near-duplicates, and recorded
`index_roles_no_leakage: true` in its manifest. The near-duplicate test compares a query with a
whole segment, and no segment changes, so no held-out query can match a summary. The resolver
re-checks the subset property on every read (4.7). No step of this stage reads a validation or
test query.

### 4.5 The artifact

`conf/data/window_summaries.csv`, committed, one row per summarized `(code, channel)`, with
`conf/data/window_summaries_provenance.json` beside it, as `index_roles.csv` and
`regressor_heldout_groups.csv` have theirs. UTF-8, written and read with Polars.

| Column | Type | Meaning |
|---|---|---|
| `code` | str | the code |
| `channel` | str | `description`, `examples` or `excluded` |
| `source_sha256` | str | sha256 of the source text's UTF-8 bytes |
| `window` | int | the target window (128) |
| `summary` | str | the kept units, joined (4.3) |
| `source_tokens` | int | the marked source's tokens, special tokens included |
| `summary_tokens` | int | the marked summary's tokens, special tokens included (≤ `window`) |
| `units_kept` | int | units in the summary |
| `units_total` | int | the text's units after any re-split |

Rows are sorted by `channel`, then `code`. `(code, channel)` is unique. Inherited descriptions
repeat a source text under several codes and get identical summaries, one row per code.

The provenance records:

- `descriptions`: the path and sha256 (`fe8c54e3…`);
- `backbone`, `revision`, `tokenizer` (the name) and the tokenizer's revision;
- `window` and the text budget per channel;
- `units` and `selection`: the rule names (`sentence-clause-piece-v1`, `centrality-v1`) and their
  settings (mean pooling, token weights, CPU, float32);
- per channel: texts present, over the window and summarized, and the mean share of source tokens
  kept;
- `artifact_sha256`, `library_versions` and `generated_at`.

### 4.6 The pin and the build command

**Pin.** A code constant in `panels/window_summaries.py`, keyed by backbone like `TRAINED_WINDOWS`:

```python
@dataclass(frozen=True)
class SummariesPin:
    path: str     # 'conf/data/window_summaries.csv', resolved like every './conf/...' path
    sha256: str
    window: int   # 128

WINDOW_SUMMARIES: Dict[str, SummariesPin] = {
    'sentence-transformers/all-MiniLM-L6-v2': SummariesPin(...),
}
```

`summaries_identity(backbone)` returns the pin's sha256, or None when no pin exists for the
backbone. It replaces `SUMMARIES` in `text_model/dataloader/tokenization_cache.py`, which is
deleted. Changing the summaries is a reviewed code change, as recording a new trained window is.

**Build.** `naics-embedder data summaries` (`cli/commands/data.py`) calls
`data/window_summaries.py`:

- options: `--descriptions` (default `./data/naics_descriptions.parquet`), `--backbone` (default
  `data_loader.tokenization.tokenizer_name`), `--output` (default the pin's path) and `--force`;
- reads the descriptions, loads the backbone from the local HF cache (`local_files_only`), takes
  the window from `TRAINED_WINDOWS`, builds the units (4.2) and selects (4.3);
- before writing, checks every invariant the resolver checks (4.7) and that every over-window text
  has a row;
- writes the CSV and its provenance, and prints the artifact's sha256 for the pin;
- refuses to overwrite an existing artifact without `--force`, as `data roles` does.

Like `data roles`, it runs once and its output is committed; neither `data preprocess` nor
`data all` runs it.

### 4.7 The resolver and its readers

`resolve_channel_texts(descriptions, tokenizer, backbone, max_length, *, pin=None)` in
`panels/window_summaries.py` returns the descriptions with every over-window channel text (4.1,
under `tokenizer` at `max_length`) replaced by its summary. `pin` defaults to
`WINDOW_SUMMARIES.get(backbone)`; tests pass their own. It is torch-free, logs how many texts per
channel it replaced, and is the only place summaries enter:

1. **The tokenization cache.** `_build_tokenization_cache` resolves the descriptions, then
   tokenizes them as now. `tokenize_field` keeps `truncation=True` as a backstop for queries; for
   channels it never triggers.
2. **The text-only builder.** `build_text_only_table` (`panels/text_only.py`) resolves with the
   same marked-fit rule, then embeds unmarked, as today. A summary that fits with its marker fits
   without it, so the comparator reads the arm's texts, markers aside (D9).

It raises ValueError, naming the code and channel where there is one, when:

- the artifact's sha256 is not the pin's;
- an over-window text has no row, or its row's `source_sha256` is not the text's, or its `window`
  is not `max_length`;
- a row names a text that is not over the window, or a code or channel the descriptions lack
  (the row set must equal the over-window set);
- texts are over the window and no pin exists for the backbone, or the pin's window is not
  `max_length` (a configured 64 raises rather than truncating);
- a summary's segments are not a subset of its source's (4.4);
- after substitution, any marked channel text, titles included, is still over the window.

The last check enforces "no channel text is truncated" on every cache build and every text-only
build. It also catches a tokenizer revision that counts differently from the one the artifact was
built with. A backbone whose texts all fit needs no pin.

**Cache identity.** The sidecar's `summaries` entry becomes `summaries_identity(tokenizer_name)`.
Every `channels-v3` cache built so far records null, so each rebuilds once. `CACHE_FORMAT` stays
`channels-v3`: the channel encoding does not change. When a sidecar does not match, the load
messages name the identity keys that differ, with both values, instead of only the fingerprints
(plan 8's polish item, `tokenization_cache.py:256-262`).

### 4.8 The identity chain

Each link records or checks the summaries' sha256 (null for a backbone with no pin):

1. **The checkpoint contract.** `CheckpointContract` (`supervision/checkpoints.py`) gains
   `summaries: Optional[str] = None`, absent from every contract saved before this stage, which
   therefore reads as null: those checkpoints trained on truncated text. `contract_for_bundle`
   and `validate_supervision_contract` take the run's or the read's value,
   `summaries_identity(tokenizer_name)` of its `code_token_config`. Consequently:
   - exact resume (`validate_checkpoint_contract`) refuses a checkpoint trained under other
     summaries;
   - export and reads (`load_arm_model` through `validate_supervision_contract`) refuse one, and
     the message names the summaries field;
   - weights-only migration compares only the encoder record, so a pre-6b checkpoint can still
     seed a run.
2. **The export provenance** (`text_model/export.py`) records `summaries` from
   `summaries_identity` and, new, `tokenizer`: `token_config.tokenizer_name` (plan 8's
   failure-handling item, its tokenizer-name part).
3. **`ArmEncoder.from_files`** (`text_model/arm_encoder.py`) also refuses a table whose
   provenance's `summaries` differs from the checkpoint contract's, or whose `tokenizer` differs
   from `token_config.tokenizer_name`, or that lacks either key. Today it checks the window only.
4. **The text-only provenance** (`panels/text_only.py`) records `summaries`.
5. **The decision records.**
   - `decision/store.py` reads `summaries_sha256` from a text-only provenance's `summaries` key
     beside the four fields it reads today. A provenance without the key, which means every
     text-only table built before this stage, is refused as lacking a field D9 reads.
   - `TextOnlyRef` and `ArmSpec` (`decision/records.py`) gain `summaries_sha256: Optional[str]`,
     required and nullable, so a record cannot omit it.
   - `check_text_only` (`decision/decide.py`) compares five fields: backbone, revision,
     descriptions sha256, summaries sha256 and window.

An arm's text identity is then (descriptions sha256, summaries sha256), and the bundle's
description fingerprint never moves (S2).

### 4.9 Files

**New:**

- `panels/window_summaries.py`: units (4.2), `SummariesPin`, `WINDOW_SUMMARIES`,
  `summaries_identity`, the artifact reader and `resolve_channel_texts` (4.7). Torch-free.
- `data/window_summaries.py`: the centrality selection (4.3) and the artifact and provenance
  writer behind `data summaries`. It imports torch and transformers.
- `conf/data/window_summaries.csv` and `conf/data/window_summaries_provenance.json`.
- `tests/unit/test_window_summaries.py` (and a build test module if the plan splits them).
- `docs/api/window_summaries.md` and its `docs/.nav.yml` entry.

**Edited:** `panels/leakage.py` (`SENTENCE_BREAK`), `panels/text_only.py`,
`text_model/dataloader/tokenization_cache.py`, `text_model/export.py`,
`text_model/arm_encoder.py`, `supervision/checkpoints.py`, the two `contract_for_bundle` callers
(`text_model/naics_model.py`, `cli/commands/training.py`), `decision/records.py`,
`decision/store.py`, `decision/decide.py`, `cli/commands/data.py`, and the tests and fixtures that
build contracts, provenances, `TextOnlyRef` or `ArmSpec`.

## 5. Error handling

Named refusals, all ValueError:

- **Build:** a title over the window; a level-3 piece over the budget; an invariant of 4.7 failing
  before the write; an existing artifact without `--force`; a backbone with no recorded window.
- **Resolver:** each case of 4.7.
- **Contract:** a summaries mismatch on exact resume, export or read.
- **Arm encoder:** a table provenance whose `summaries` or `tokenizer` differs, or is absent.
- **Decision records:** a text-only provenance without `summaries`; `check_text_only` on a
  summaries mismatch.

Guarantees rather than refusals:

- A cache sidecar whose `summaries` differs rebuilds the cache, as a format or marker change does.
- Weights-only migration ignores summaries (4.8).

## 6. Testing

Tests are written red to green. Unit tests download nothing: CI has neither `data/` nor the MiniLM
cache. Tokenizer-dependent unit tests use a stub or tiny tokenizer, and selection tests inject a
stub embedder.

- **Units:** every unit boundary is a segmenter boundary, on crafted texts with `U.S.`, `i.e.`,
  numbered lists, `;` inside parentheses and cross-references; the level-2 and level-3
  re-splits; the raise for a piece over the budget and for a title over the window.
- **Selection:** with a stub embedder, the greedy keeps the expected units; exact ties go to the
  earlier unit; both stop rules hold; output is in source order and fits with its marker.
- **Resolver:** substitution; every refusal of 4.7; texts that fit pass through unchanged; a
  backbone with no pin and no over-window text passes.
- **Cache identity:** changing only the summaries rebuilds the cache; changing only the markers
  rebuilds it (plan 8's missing test, `_cache_identity`); the mismatch message names the
  differing keys.
- **Text-only builder:** it reads resolved texts, and its provenance records `summaries`.
- **Contract:** a raw contract without `summaries` reads as null; exact resume and
  `validate_supervision_contract` refuse a mismatch; weights-only loading does not compare it.
- **Export and arm encoder:** the export provenance records `summaries` and `tokenizer`;
  `from_files` refuses a provenance with other or absent summaries or tokenizer.
- **Decision records:** the store reads `summaries_sha256` and refuses a provenance without it;
  `check_text_only` refuses a summaries mismatch; the decision fixtures carry the field.
- **The committed artifact:** its sha256 equals the pin's; `(code, channel)` is unique; every
  `summary_tokens` is at most `window`; every channel is one of the three.
- **Local-only, skipped without the cached backbone or `data/naics_descriptions.parquet`:** on the
  real descriptions, the resolver accepts the committed artifact (which re-tokenizes every row and
  re-checks fit and the subset property), and replaces 162, 106 and 485 texts.

## 7. Exit procedure

This runs locally. It trains nothing and reads no split, so the selection log gains no record.

1. Clone bundle 301cce28 and `data/naics_descriptions.parquet` (sha256 `fe8c54e3…`) from the main
   checkout into the worktree's `data/` with `cp -cR` and `cp -c`. Never symlink or rebuild them.
   Clone plan 8's Exit checkpoint and arm table from the main checkout's
   `checkpoints/plan8_exit/` the same way.
2. Point `supervision.manifest_path` at the clone with a `key=value` override on every command.
   Never commit it.
3. Run `HF_HUB_OFFLINE=1 uv run naics-embedder data summaries`. Commit the CSV and its
   provenance, and set the pin's sha256.
4. Build the tokenization cache through `tokenization_cache(code_token_config(cfg), …)` under the
   bundle's fingerprints, as `NAICSDataModule.prepare_data` does. The resolver logs 162, 106 and
   485 replaced texts, and the sidecar's `summaries` is the pin's sha256.
5. Run `HF_HUB_OFFLINE=1 uv run naics-embedder tools text-only-table`. Its provenance's `summaries`
   is the pin's sha256.
6. Run `tools export-table` on plan 8's Exit checkpoint. It refuses: the checkpoint was trained
   under null summaries.
7. Run `uv run pytest` (the local-only tests included), `./scripts/format_code.sh --check --all`
   and `uv run mkdocs build --strict`.

## 8. Documentation

- `docs/text_training.md`: over-window channel texts are summarized, not truncated (it still says
  they are tail-truncated until Stage 6b); the Cache Regeneration list names format, markers and
  summaries.
- `docs/usage.md`: the `data summaries` command.
- `CLAUDE.md`: the data commands (a `data summaries` line beside `data roles`) and the directory
  tree (`panels/window_summaries.py`, `data/window_summaries.py`, `conf/data/window_summaries.csv`).
- `docs/api/window_summaries.md` and `docs/.nav.yml`.

`uv run mkdocs build --strict` must pass locally, since PR CI never builds the docs.

## 9. Chosen approach and rejected alternatives

- **Method (S5).**
  - Chosen: extractive, every channel. Only official NAICS wording; leakage-safe by a structural
    property, so nothing reads sealed text; rebuildable from code.
  - Rejected: abstractive descriptions with extractive examples and excluded. It suits the 75
    top-level descriptions best, but needs a generator the repo lacks, generator provenance (model,
    prompt hash, date), the approved audit of all 162 against the validation and test queries, and
    a fix path that rewrites summaries.
  - Rejected: abstractive everywhere. An examples text lists a code's own index entries, whose
    siblings in the role table are that code's validation and test queries, so a paraphrase is
    likely to hit one.
  - Not considered: encoding a long text in several windows and pooling them. It is neither
    summarizing nor truncating, and S1 settled summaries.
- **Selection (S6).**
  - Chosen: backbone centrality (4.3).
  - Rejected: lexical coverage. Integer arithmetic rebuilds bit for bit anywhere, but its stopword
    list and weights need defending, and it reads below truncation on descriptions.
  - Rejected: lead (the first units that fit). It is truncation at sentence boundaries, and reads
    below truncation on every channel.
  - Rejected: centrality with the first description sentence forced in. It reads 0.886 against
    0.910, and centrality keeps that sentence in 115 of 162 texts anyway.
- **Units.**
  - Chosen: sentences and clauses whose boundaries are segmenter boundaries (4.2).
  - Rejected: the segmenter's pieces alone. They cut `U.S. Industry`, numbered lists and
    parenthesized lists into fragments.
- **Pin.**
  - Chosen: a code constant keyed by backbone. `TokenizationConfig` is built in three places
    (`NAICSDataModule`, `code_token_config`, `utils/validation.py`), and a constant is read in one;
    the pin stays out of `conf/config.yaml`, whose held local commits must replay cleanly.
  - Rejected: a `data_loader.tokenization` key with a pinned hash, which every construction site
    would have to pass.
- **Checkpoints.**
  - Chosen: the contract records the summaries; export, reads and exact resume refuse a mismatch.
  - Rejected: no record. A checkpoint trained on truncated text could be exported under summaries
    and paired with a comparator that read them, which breaks D9 unseen.
- **Format.**
  - Chosen: CSV beside a provenance JSON, as the repo's other committed tables are.
  - Rejected: parquet, which a review cannot diff.
- **Placement.**
  - Chosen: the frozen artifact's logic in `panels/`, its build in `data/`, as the role table and
    the held-out groups are split. `text_model` and `data` already import `panels`, and
    `panels/__init__.py` imports nothing, so nothing cycles.
  - Rejected: `utils/`, which imports nothing above it today, while the units need
    `panels/leakage.py`.

## 10. Rollout note

> Roadmap: specs/naics-embedding-roadmap.md, Stage 6b — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Switch at merge.** Every `channels-v3` cache rebuilds once. Every checkpoint, exported table and
text-only table made before this stage is refused by export, reads and `decide`; weights-only
loading still works. Plan 8's Exit numbers stay as the floor its finding records
(`specs/findings/shared-encoder-first-reading.md`). Stage 7 trains and builds its text-only table
on the summaries.

**Realized.** Plan completion records, in the roadmap's Stage 6b entry, the artifact's sha256, the
rows per channel and the mean share of source tokens kept.

**Deferred items,** handled through /deferred at plan completion:

- Plan 8's "export and outcome read handle a few failures untidily": its tokenizer-name part is
  discharged (4.8, links 2 and 3). The rest stays.
- Plan 8's "encoder, fusion and cache tests leave gaps": its markers-only sidecar test is
  discharged (section 6). The rest stays.
- Plan 8's "code and docs polish": the cache's load messages (4.7) and the `docs/text_training.md`
  note on Stage 6b (section 8) are discharged. The rest stays.
- Plan 7's manifest tokenizer revision keeps its trigger: no bundle is rebuilt.

**Model routing.** writing-plans for plan 9 runs in a fresh Opus session from this spec;
execution runs on the Sonnet default.
