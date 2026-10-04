# Window-fitting summaries — Design Spec

**Status:** APPROVED (2026-10-04); revised the same day after a five-lens review

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
whose hash every link from checkpoint to decision record carries and checks.

## 2. Scope

### 2.1 In scope

- Req 9, input windows, under the user's ruling of 2026-10-03 (S1): no channel text is truncated.
- Req 3, leakage: no segment of any summary matches a held-out query, exactly or as a
  near-duplicate, under Req 3's per-segment check (4.4).
- D9: the comparator reads the arm's text, now identified by the descriptions' and the
  summaries' hashes, and the seed sweep checks each seed's table against that identity (4.8).
- Three parts of plan 8's deferred entries (section 10).

### 2.2 Out of scope

- **Queries.** Index entries and activity phrases go through `tokenize_field` unchanged.
- **The bundle.** Bundle 301cce28 and its descriptions (sha256 `fe8c54e3…`) stay byte-identical
  (S2). Plan 7's deferred manifest tokenizer revision keeps its trigger.
- **Stage 9:** summaries for other backbones. The pin is keyed by backbone (4.6), so Stage 9 adds
  entries. Selection runs under each backbone's own frozen weights (4.3), so two candidates can
  read different summaries of one text; whether Stage 9 fixes a common selection backbone is its
  own brainstorm's call.
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

- **S5. Method.** Extractive, for every channel. A summary is whole source units, verbatim and in
  source order, each at most once.
- **S6. Selection.** Backbone centrality (4.3).
- **S7. Design.** Sections 4.1–4.9 as presented: the units, the artifact, a code pin rather than a
  config key, one resolver for both readers, and the summaries' hash in the checkpoint contract,
  both provenances and the decision records. The review's corrections (the model's contract
  input, the resolver's order, the test seam, the in-order check and the sweep's per-seed check)
  refine that design without changing it.

## 4. Design

### 4.1 Which texts

A channel text is **over the window** when its marked form, `marker(field) + text`
(`text_model/fields.py`), tokenizes to more than the window under the backbone's tokenizer,
special tokens included, without truncation. Only over-window texts get summaries; every other
text is read verbatim.

Measured on 2026-10-04 on the descriptions `fe8c54e3…` under MiniLM's tokenizer (revision
1110a243) at the 128-token window, and reproduced by the review:

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

**Exclusions (Req 8).** Req 8(a) and Verification "Exclusions" were discharged at generation in
Stage 5, on the bundle's exclusion text, which S2 keeps unchanged. A summary of an exclusion text
is a source-order subset of its cross-references, so it never duplicates one; like truncation
today, it leaves some out of the model's view of the channel. That view matters: 3,186 of the
bundle's 4,623 redirection rows, and 3 of its 9 lineal references, sit on the 485 over-window
exclusion texts. Each cross-reference is one unit, so the artifact's `units_kept` and
`units_total` record per text how many the model reads, and the Realized line reports the totals.
Every destination still reaches training through the bundle's `excluded_codes` and Req 8(b)'s
activity phrases, which this stage does not touch.

### 4.2 Units

A summary is built from units, and every unit boundary is a boundary of the leakage segmenter
(`panels/leakage.py`: a description or exclusion text splits after `.` or `;` followed by
whitespace; the examples channel splits at `'; '`, `EXAMPLES_SEPARATOR`). That property carries
the leakage argument (4.4). The segmenter's break pattern becomes the public name
`SENTENCE_BREAK` in `panels/leakage.py`, so the unit builder and the leakage checker share one
definition.

- **title:** the whole title is one unit. A title over the window cannot be summarized, so the
  build raises; none is over under MiniLM.
- **examples:** the entries, split at `EXAMPLES_SEPARATOR`. An entry over the budget raises; on
  2026-10-04 the longest was 46 tokens.
- **description and excluded:** the segmenter's pieces, merged into units at three levels:
  1. **sentence:** a unit closes at a piece that ends in `.`, or, for excluded, in `.` or `;`, so
     that each cross-reference is one unit;
  2. **clause:** a unit closes at a piece that ends in `.` or `;`;
  3. **piece:** each segmenter piece is a unit.

  At levels 1 and 2 a piece closes a unit only when the unit's parentheses balance (as many `(` as
  `)`) and the piece does not match this pattern, which is normative:

  ```python
  r'(?:\bU\.S|\bi\.e|\be\.g|\betc|\bNo|\bvs|(?:^|\s)\d{1,2}|\(\d{1,2}\))[.;]$'
  ```

  That is, it does not end in `U.S.`, `i.e.`, `e.g.`, `etc.`, `No.` or `vs.`, or in a one- or
  two-digit list numeral such as `1.` or `(1);`. A piece that ends in a code or a year, such as
  `… in Industry 111113.`, closes a unit. On the over-window texts this pattern gives the same
  units as the broader probe pattern did.

  A text's units are its level-1 units; a unit over the budget is re-split at level 2, and a
  level-2 unit over the budget at level 3. A level-3 piece over the budget raises. On 2026-10-04
  the longest piece of any over-window text was 94 tokens (description) and 79 (excluded).

`panels/window_summaries.py` imports `marker` from `text_model/fields.py`, a module with no
imports of its own, to compute the budget. It is the first `panels` module to import from
`text_model`, and it adds no cycle.

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

Exact float ties go to the earlier unit. The plateau stop in step 3 is intended: adding a unit
that moves the summary's vector away from the text's makes it a worse summary, so a summary can
end well under the window. The provenance reports how often that happens (4.5).

The kept units are emitted in source order, joined by `' '` (description, excluded) or
`EXAMPLES_SEPARATOR` (examples). The segmenter then re-splits each summary into exactly its kept
pieces:

- **description and excluded:** every non-final source unit ends in `.` or `;`, since
  `SENTENCE_BREAK` splits only after them, and the source's last unit can only be last in a
  summary;
- **examples:** no entry contains `EXAMPLES_SEPARATOR`, because the split consumed every
  occurrence, and `'; '` cannot straddle a join, because no proper prefix of it equals a proper
  suffix. Examples entries need not end in `.` or `;`, so no test may assert that they do.

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
invariants (4.7), never by rebuild equality.

### 4.4 Leakage without a sealed read

Each summary's segments, as `training_text_segments` cuts them (`text_segments` for description
and excluded, its normalized entries for examples), are a subset of its source text's segments
(4.2, 4.3). Bundle 301cce28's build checked every segment of the descriptions `fe8c54e3…` against
the validation and test queries, exactly and as near-duplicates (`verify_role_leakage`,
`panels/index_roles.py`), and recorded `index_roles_no_leakage: true` in its manifest. The
near-duplicate test compares a query with a whole segment, and no segment changes, so no held-out
query can match any of a summary's segments.

Word spans that cross a join between two kept units are not checked, just as spans across
adjacent source sentences, or across the joined rows of today's exclusion text, are not checked;
checking them would need a sealed read (S4). The resolver re-checks the subset property, and S5's
in-order property, on every read (4.7). No step of this stage reads a validation or test query.

### 4.5 The artifact

`conf/data/window_summaries.csv`, committed, one row per summarized `(code, channel)`, with
`conf/data/window_summaries_provenance.json` beside it, as `index_roles.csv` and
`regressor_heldout_groups.csv` have theirs. UTF-8, written with Polars and read with an explicit
`SUMMARIES_SCHEMA` (as `panels/index_roles.py` reads its table), so that `code` stays a string.

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

Rows are sorted by `channel`, then `code`. `(code, channel)` is unique. Identical source texts get
identical summaries, one row per code; inherited descriptions are usually reworded, so no test
may assume a parent and child share one.

The provenance records:

- `descriptions`: the path and sha256 (`fe8c54e3…`);
- `backbone`, `revision`, `tokenizer` (the name) and the tokenizer's revision, which is
  `load_backbone`'s snapshot revision, since the tokenizer and the model load from one snapshot;
- `window` and the text budget per channel;
- `units` and `selection`: the rule names (`sentence-clause-piece-v1`, `centrality-v1`) and their
  settings (the normative pattern, mean pooling, token weights, CPU, float32);
- per channel: texts present, over the window and summarized; the mean share of source tokens
  kept; the minimum and the 10th percentile of `summary_tokens`; and how many summaries ended at
  the plateau stop with a unit that still fit;
- `artifact_sha256`, `library_versions` and `generated_at`.

### 4.6 The pin and the build command

**Pin.** Code constants in `panels/window_summaries.py`, keyed by backbone like `TRAINED_WINDOWS`:

```python
WINDOW_SUMMARIES_PATH = 'conf/data/window_summaries.csv'  # resolved like every './conf/...' path

@dataclass(frozen=True)
class SummariesPin:
    path: str
    sha256: str
    window: int

WINDOW_SUMMARIES: Dict[str, SummariesPin] = {}  # the MiniLM entry lands with the artifact
```

`summaries_identity(backbone)` returns the pin's sha256, or None when no pin exists for the
backbone. Every identity site calls it at call time and never binds `WINDOW_SUMMARIES` or a pin at
import, so a test's patch reaches every site (section 6). It replaces `SUMMARIES` in
`text_model/dataloader/tokenization_cache.py`, which is deleted. Changing the summaries is a
reviewed code change, as recording a new trained window is.

**Bootstrapping.** `WINDOW_SUMMARIES` has no MiniLM entry until the Exit commits the artifact
(section 7, step 3); that commit adds the entry, with the artifact's sha256, and the tests that
need the real pin. Until then `summaries_identity` returns None in production, and a cache build
on the real descriptions raises (texts over the window, no pin), which is intended.

**Build.** `naics-embedder data summaries` (`cli/commands/data.py`) calls
`data/window_summaries.py`:

- options: `--descriptions` (default `./data/naics_descriptions.parquet`), `--backbone` (default
  the training config's `data_loader.tokenization.tokenizer_name`, read from `conf/config.yaml`
  with `load_config`), `--output` (default `WINDOW_SUMMARIES_PATH`) and `--force`;
- reads the descriptions, loads the backbone from the local HF cache (`local_files_only`), takes
  the window from `TRAINED_WINDOWS`, builds the units (4.2) and selects (4.3);
- writes the CSV to a temporary file beside `--output`, builds
  `SummariesPin(path=<temporary file>, sha256=<its sha256>, window=<the window>)`, and runs
  `resolve_channel_texts` on the descriptions with that pin, which checks every invariant of 4.7
  against the bytes about to be committed;
- only then moves the CSV into place, writes its provenance, and prints the sha256 for the pin;
- refuses to overwrite an existing artifact without `--force` by raising FileExistsError, which
  the CLI catches and reports with exit code 1, as `data roles` does. A second backbone passes its
  own `--output`; the refusal stops a collision.

Like `data roles`, it runs once and its output is committed; neither `data preprocess` nor
`data all` runs it.

### 4.7 The resolver and its readers

`resolve_channel_texts(descriptions, tokenizer, backbone, max_length, *, pin=DEFAULT)` in
`panels/window_summaries.py` returns the descriptions with every over-window channel text (4.1,
under `tokenizer` at `max_length`) replaced by its summary. `pin` defaults to a sentinel meaning
`WINDOW_SUMMARIES.get(backbone)`, looked up at call time; an explicit None means no pin. It is
torch-free, logs how many texts per channel it replaced, and is the only place summaries enter:

1. **The tokenization cache.** `_build_tokenization_cache` resolves the descriptions under
   `cfg.tokenizer_name` with the tokenizer it loads, then tokenizes them as now. `tokenize_field`
   keeps `truncation=True` as a backstop for queries; for channels it never triggers.
2. **The text-only builder.** `build_text_only_table` (`panels/text_only.py`) resolves under its
   `backbone` argument, with the tokenizer `load_backbone` returned and its `max_length`, by the
   same marked-fit rule; then it embeds unmarked, as today. A summary that fits with its marker
   fits without it, so the comparator reads the arm's texts, markers aside (D9).

The resolver works in this order, raising ValueError (naming the code and channel where there is
one) at the first failure:

1. Compute the over-window set. If it is empty, return the descriptions unchanged, with no pin
   lookup, no artifact read and no window check.
2. Otherwise require a pin, and require the pin's window to equal `max_length` (a configured 64
   raises rather than truncating).
3. Read the artifact (a missing file raises, naming its resolved absolute path) and require its
   sha256 to be the pin's.
4. Require the row set to equal the over-window set: every over-window text has a row, and no row
   names a text that is not over the window, or a code or channel the descriptions lack.
5. Per row, require `source_sha256` to be the text's sha256 and `window` to be `max_length`.
6. Per row, require S5: the summary's raw pieces (`SENTENCE_BREAK` pieces for description and
   excluded, `EXAMPLES_SEPARATOR` entries for examples) form an in-order subsequence of the
   source's raw pieces, each source piece used at most once (a two-pointer match), and its
   normalized segments are a subset of the source's (4.4). This needs no tokenizer and catches a
   reordered, repeated or edited unit.
7. After substitution, require every marked channel text, titles included, to fit the window.

Step 7 enforces "no channel text is truncated" on every cache build and every text-only build. It
also catches a tokenizer revision that counts differently from the one the artifact was built
with. Whole-unit structure is the build's construction (4.2), tested in unit tests; the resolver
does not rebuild units.

**Cache identity.** The sidecar's `summaries` entry becomes `summaries_identity(tokenizer_name)`.
Every `channels-v3` cache built so far records null, so each rebuilds once. `CACHE_FORMAT` stays
`channels-v3`: the channel encoding does not change. When a sidecar does not match, the load
messages (`load_verified_tokenization_cache`, `tokenization_cache.py:262-266`, and the no-locking
refusal in `tokenization_cache`) name the identity keys that differ, with both values, instead of
only the fingerprints (plan 8's polish item).

### 4.8 The identity chain

Each link records or checks the summaries' sha256 (null for a backbone with no pin):

1. **The checkpoint contract.** `CheckpointContract` (`supervision/checkpoints.py`) gains
   `summaries: Optional[str] = None`. It is absent from every contract saved before this stage,
   which therefore reads as null: those checkpoints trained on truncated text.
   - **Training.** `NAICSContrastiveModel` gains a constructor input `summaries: Optional[str] =
     None`, saved with its hyperparameters and passed to both `contract_for_bundle` and
     `containment_contract`, which each gain a keyword `summaries`. `build_model_from_config` and
     `runtime_contract_for` (`cli/commands/training.py`) pass
     `summaries_identity(cfg.data_loader.tokenization.tokenizer_name)`, the same key the cache
     uses. With the default None, a pre-6b checkpoint's saved hyperparameters rebuild a null
     contract on load; either way the named refusal below runs before any load.
   - **Reads.** `validate_supervision_contract(raw, manifest, supervision_mode='repaired', *,
     summaries)` and `load_arm_model(checkpoint_path, bundle, *, summaries, device='cpu')` take
     `summaries` keyword-only with no default, so a missed caller is a TypeError, not a silent
     None. `export_code_table` and `ArmEncoder.from_files` pass
     `summaries_identity(token_config.tokenizer_name)`; `tests/unit/test_export.py` calls
     `load_arm_model` directly and is updated.
   - **Consequences.** Exact resume (`validate_checkpoint_contract`) refuses a checkpoint trained
     under other summaries, legacy containment included. Export and reads refuse one, and the
     message names the summaries field. The HGCN feeder (`generate_embeddings_from_checkpoint`)
     refuses a pre-6b checkpoint through `validate_exact_resume` with `runtime_contract_for`'s
     contract; it needs no edit. Weights-only migration compares only the encoder record, so a
     pre-6b checkpoint can still seed a run.
2. **The export provenance** (`text_model/export.py`) records `summaries`, from
   `summaries_identity(token_config.tokenizer_name)`, and, new, `tokenizer`:
   `token_config.tokenizer_name` (plan 8's failure-handling item, its tokenizer-name part).
3. **`ArmEncoder.from_files`** (`text_model/arm_encoder.py`) refuses, before the model loads as
   its other provenance checks do, a table whose provenance lacks `summaries` or `tokenizer`, or
   whose `summaries` differs from `summaries_identity(token_config.tokenizer_name)`, or whose
   `tokenizer` differs from `token_config.tokenizer_name`. `load_arm_model`'s contract check
   against the same value then closes the chain to the checkpoint.
4. **The text-only provenance** (`panels/text_only.py`) records `summaries`, from
   `summaries_identity(backbone)`.
5. **The decision records.**
   - `decision/store.py` reads `summaries_sha256` from a text-only provenance's `summaries` key
     beside the four fields it reads today. A provenance without the key, which means every
     text-only table built before this stage, is refused as lacking a field D9 reads.
   - `TextOnlyRef` and `ArmSpec` (`decision/records.py`) gain `summaries_sha256: Optional[str]`,
     required and nullable, so a record cannot omit it.
   - `check_text_only` (`decision/decide.py`) compares five fields: backbone, revision,
     descriptions sha256, summaries sha256 and window.
   - `run_seed_sweep` (`decision/sweep.py`) reads each seed's export provenance
     (`provenance_path(artifacts.table)`) and requires its `backbone`, `revision`, descriptions
     sha256, `summaries` and `max_length` to equal the `ArmSpec`'s, refusing the seed otherwise.
     The check runs right after `runner.run`, before `store.put` and any panel read, as the
     dimension check does: a refusal at the first seed leaves the selection log empty, and one at
     a later seed leaves the earlier seeds' reads logged with no `ArmRecord`, as a dimension
     refusal does today. Until now `ArmSpec`'s D9 fields were declared and checked only against
     the text-only table; this ties them to what each seed read, for every D9 field.
   - Limits: `tools regressor-panel` pairs raw parquets and checks no provenance, and `decide`'s
     `check_arm` re-checks only the text-only provenance, since a `SeedRun` keeps no export
     provenance. A seed table's summaries are therefore checked by `run_seed_sweep` alone, and a
     record assembled outside it, as the synthetic decision fixtures are, is never checked.

An arm's text identity is then (descriptions sha256, summaries sha256), and the bundle's
description fingerprint never moves (S2).

### 4.9 Files

**New:**

- `panels/window_summaries.py`: the units (4.2), `WINDOW_SUMMARIES_PATH`, `SummariesPin`,
  `WINDOW_SUMMARIES`, `summaries_identity`, `SUMMARIES_SCHEMA`, the artifact reader and
  `resolve_channel_texts` (4.7). Torch-free.
- `data/window_summaries.py`: the centrality selection (4.3) and the artifact and provenance
  writer behind `data summaries`. It imports torch and transformers.
- `conf/data/window_summaries.csv` and `conf/data/window_summaries_provenance.json`.
- `tests/unit/test_window_summaries.py` (and a build test module if the plan splits them).
- `docs/api/window_summaries.md` and its `docs/.nav.yml` entry.

**Edited source:** `panels/leakage.py` (`SENTENCE_BREAK`), `panels/text_only.py`,
`text_model/dataloader/tokenization_cache.py`, `text_model/export.py`,
`text_model/arm_encoder.py`, `text_model/naics_model.py` (the constructor input and both contract
calls), `supervision/checkpoints.py`, `cli/commands/training.py` (`build_model_from_config`,
`runtime_contract_for`), `cli/commands/data.py`, `decision/records.py`, `decision/store.py`,
`decision/decide.py`, `decision/sweep.py`, and the `utils/input_window.py` module docstring.

**Edited tests:** `tests/conftest.py` (the seam, section 6); `tests/fixtures/shared_encoder.py`;
`tests/fixtures/decision.py`; `tests/unit/test_tokenization_cache.py` (the null-summaries sidecar
test, the test that patches the deleted `SUMMARIES`, and any cache test whose texts exceed its
window); `test_export.py`; `test_arm_encoder.py`; `test_cli_commands.py`; `test_cli_training.py`;
`test_naics_model.py`; `test_checkpoint_contract.py` (its direct `validate_supervision_contract`
calls); `test_decision_rule.py` (its direct `ArmSpec`); `test_decision_sweep.py` (its seed
tables gain export provenance); `test_decision_store.py`; and every other test that builds a
contract, a provenance, `TextOnlyRef` or `ArmSpec`.

## 5. Error handling

Named refusals, ValueError unless stated:

- **Build:** a title over the window; an examples entry or a level-3 piece over the budget; a
  backbone with no recorded window; any resolver check on the bytes about to be committed; an
  existing artifact without `--force` (FileExistsError, exit code 1 at the CLI).
- **Resolver:** each step of 4.7, in that order.
- **Contract:** a summaries mismatch on exact resume, export, read or the HGCN feeder; a caller
  that omits `summaries` (TypeError).
- **Arm encoder:** a table provenance whose `summaries` or `tokenizer` differs or is absent.
- **Decision records:** a text-only provenance without `summaries`; `check_text_only` on a
  summaries mismatch; `run_seed_sweep` on a seed whose export provenance differs from the
  `ArmSpec`.

Guarantees rather than refusals:

- A cache sidecar whose `summaries` differs rebuilds the cache, as a format or marker change does.
- Descriptions whose texts all fit pass the resolver with no pin and no artifact read.
- Weights-only migration ignores summaries (4.8).

## 6. Testing

Tests are written red to green. CI has no `data/` and no MiniLM weights, but it downloads MiniLM's
tokenizer, which the existing cache, export and arm-encoder fixtures load. New resolver and unit
tests use a stub or tiny tokenizer where they can, and selection tests inject a stub embedder.

**The seam.** An autouse fixture in `tests/conftest.py` mutates `WINDOW_SUMMARIES` in place
(`monkeypatch.setitem`, after removing any other entry with `monkeypatch.delitem`) so that it
holds one entry: the MiniLM backbone mapped to a dummy pin (a path that does not exist, a fixed
fake sha256, window 128). Tests reach the dict as `window_summaries.WINDOW_SUMMARIES`, never by
`from … import WINDOW_SUMMARIES`, so every patch acts on the one dict the identity sites read.
Fixture descriptions fit the window, so the resolver never reads the dummy (4.7, step 1), while
every identity site records the dummy's sha256. A site that records None instead fails from the
first task, not at the Exit, provided its test names the MiniLM backbone: the identity-site tests
(the sidecar, both provenances, the contracts) use MiniLM-named fixtures such as
`five_code_token_config` and `text_only_comparator_table`, never `tiny-bert` or
`tiny-backbone`, whose identity is None. A registered marker opts a test out of the fixture: the
committed-artifact test and the local-only tests below need the real pin. A test that needs "no
pin" deletes the entry with `monkeypatch.delitem` or uses an unpinned backbone name.

- **Units:** every unit boundary is a segmenter boundary, on crafted texts with `U.S.`, `i.e.`,
  numbered lists, `;` inside parentheses, cross-references, and `… in Industry 111113.` as a
  non-final piece (it closes a unit); the level-2 and level-3 re-splits; the raises for a level-3
  piece over the budget, an examples entry over the budget and a title over the window.
- **Selection:** with a stub embedder, the greedy keeps the expected units; exact ties go to the
  earlier unit; both stop rules hold, the plateau stop included; output is in source order and
  fits with its marker.
- **Resolver:** substitution; a pinned backbone with no over-window text passes without reading
  its artifact (the path does not exist); every refusal of 4.7, each at its step, the S5 check
  with a reordered, a repeated and an edited unit; texts that fit pass through unchanged.
- **Build:** the temporary-file validation runs before the move, so a failing invariant leaves no
  artifact; the overwrite refusal is a FileExistsError.
- **Cache substitution:** a cache built over descriptions with one over-window text, under a
  test-local pin (`monkeypatch.setitem` on `WINDOW_SUMMARIES`) pointing at a temporary CSV with
  the right sha256 and a window equal to the test's `max_length`, stores that text's
  `input_ids` as `tokenize_field` of its summary; with the pin deleted, the build raises. The
  dummy pin cannot serve here, since the resolver would try to read it. This is the test that
  fails if `_build_tokenization_cache` records the pin without applying it.
- **Cache identity:** changing only the pin rebuilds the cache (`monkeypatch.setitem` on
  `WINDOW_SUMMARIES`, replacing the test that patched `SUMMARIES`); changing only the markers
  rebuilds it (plan 8's missing test, `_cache_identity`); the sidecar records
  `summaries_identity(MINILM)`; the mismatch message names the differing keys.
- **Text-only builder:** it reads resolved texts, and its provenance records `summaries`.
- **Contract:** a raw contract without `summaries` reads as null; the model passes its
  `summaries` to both contract builders; exact resume refuses a mismatch, a containment
  checkpoint included; `validate_supervision_contract` refuses one; weights-only loading does not
  compare it; the HGCN feeder refuses a checkpoint trained under other summaries.
- **Export and arm encoder:** the export provenance records `summaries` and `tokenizer`;
  `from_files` refuses a provenance with other or absent summaries or tokenizer before the model
  loads.
- **Decision records:** the store reads `summaries_sha256` and refuses a provenance without it;
  `check_text_only` refuses a summaries mismatch; `run_seed_sweep` refuses a seed whose export
  provenance differs from the `ArmSpec` in any of its five fields, and a first-seed refusal
  leaves the selection log empty.
- **The committed artifact** (opted out of the seam): its sha256 equals the pin's; `(code,
  channel)` is unique; every `summary_tokens` is at most `window`; every channel is one of the
  three.
- **Local-only, opted out of the seam, skipped without the cached backbone or
  `data/naics_descriptions.parquet`:** on the real descriptions the resolver accepts the committed
  artifact, which re-tokenizes every row and re-checks fit, S5 and the subset property, and
  replaces 162, 106 and 485 texts.

## 7. Exit procedure

This runs locally. It trains nothing and reads no split, so the selection log gains no record.
`CLONE` below is the cloned bundle's manifest path.

1. Clone bundle 301cce28 and `data/naics_descriptions.parquet` (sha256 `fe8c54e3…`) from the main
   checkout into the worktree's `data/` with `cp -cR` and `cp -c`, and plan 8's
   `checkpoints/plan8_exit/` (`last.ckpt`, `arm_table.parquet`, `text_only.parquet` and their
   provenance files) into the worktree's `checkpoints/plan8_exit/` with `cp -c`. Never symlink or
   rebuild them.
2. Pass `supervision.manifest_path=CLONE` as a `key=value` override to every command that reads
   the bundle (steps 4 and 6). Never commit it.
3. Run `HF_HUB_OFFLINE=1 uv run naics-embedder data summaries`. Add the CSV and its provenance,
   the MiniLM entry in `WINDOW_SUMMARIES` with the printed sha256, and the committed-artifact and
   local-only tests; run those tests, the full suite and `./scripts/format_code.sh --check` on the
   touched files; then commit them together.
4. Under `HF_HUB_OFFLINE=1`, build the tokenization cache as `NAICSDataModule.prepare_data` does,
   with a short script: load
   `conf/config.yaml` with the `CLONE` override, validate the bundle, and call
   `tokenization_cache(code_token_config(cfg), description_fingerprint=…, codebook_fingerprint=…)`
   with the manifest's fingerprints. The resolver logs 162, 106 and 485 replaced texts, and the
   sidecar's `summaries` is the pin's sha256. This run is the leakage evidence: the subset and S5
   checks passed on all 753 rows, and the manifest records `index_roles_no_leakage: true` (4.4).
5. Run `HF_HUB_OFFLINE=1 uv run naics-embedder tools text-only-table --descriptions
   data/naics_descriptions.parquet --output checkpoints/plan9_exit/text_only.parquet`. Its
   provenance's `summaries` is the pin's sha256.
6. Show the chain's refusals and records with a short script and one command:
   - `ArtifactStore(<tmp root>).put_text_only` on step 5's table returns a `TextOnlyRef` whose
     `summaries_sha256` is the pin's sha256, and refuses plan 8's `text_only.parquet`, whose
     provenance has no `summaries` key;
   - `ArmEncoder.from_files` on plan 8's `last.ckpt` and `arm_table.parquet` refuses before the
     model loads: the table's provenance records null summaries and no tokenizer;
   - `HF_HUB_OFFLINE=1 uv run naics-embedder tools export-table --checkpoint
     checkpoints/plan8_exit/last.ckpt --output <tmp path> supervision.manifest_path=CLONE` refuses:
     the checkpoint was trained under null summaries.
7. Run `uv run pytest` (the local-only tests included), `./scripts/format_code.sh --check --all`
   and `uv run mkdocs build --strict`.

## 8. Documentation

- `docs/text_training.md`:
  - :71-73 says the sidecar's `summaries` entry is "null until Stage 6b"; it becomes the pin's
    sha256;
  - :334-338 says "Every tokenizing path truncates to the window"; over-window channel texts now
    read as their summaries, and no channel text is truncated;
  - the Cache Regeneration list (:414) names format, markers and summaries;
  - :437-439, the checkpoint contract's fields, adds `summaries`.
- Docstrings that say every tokenizing path truncates: the `utils/input_window.py` module
  docstring; `InputWindowRecord` (`supervision/schema.py:182-183`), whose counts stay the bundle's
  but no longer describe what truncation shortens; and `tokenization_cache.py:48` and `:66`. Each
  says instead that over-window channel texts read as their summaries, and that truncation is a
  backstop for queries.
- `docs/usage.md`: the `tools text-only-table` paragraph (:211-216), which says each channel is
  truncated to the window; the `tools export-table` provenance list (:273-275), which adds
  `tokenizer` and the refusal of a pre-6b checkpoint; and a new `data summaries` entry.
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
  - Chosen: the contract records the summaries, from a model input that training passes; export,
    reads and exact resume refuse a mismatch.
  - Rejected: no record. A checkpoint trained on truncated text could be exported under summaries
    and paired with a comparator that read them, which breaks D9 unseen.
  - Rejected: the model deriving the value from `base_model_name`. Nothing ties that name to
    `data_loader.tokenization.tokenizer_name`, the key the cache, export and reads use.
- **Seed check.**
  - Chosen: `run_seed_sweep` compares each seed's export provenance with the `ArmSpec`.
  - Rejected: comparing with the current pin, which a seed loaded from an earlier run need not
    match.
- **Format.**
  - Chosen: CSV beside a provenance JSON, as the repo's other committed tables are.
  - Rejected: parquet, which a review cannot diff.
- **Placement.**
  - Chosen: the frozen artifact's logic in `panels/`, its build in `data/`, as the role table and
    the held-out groups are split. `text_model` and `data` already import `panels`, and
    `panels/__init__.py` imports nothing, so nothing cycles.
  - Rejected: `utils/`, which does not import `panels`, while the units need `panels/leakage.py`.

## 10. Rollout note

> Roadmap: specs/naics-embedding-roadmap.md, Stage 6b — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Switch at merge.** Every `channels-v3` cache rebuilds once. A checkpoint made before this stage
is refused by exact resume, export, the outcome read and the HGCN feeder; an exported table by
`ArmEncoder.from_files` and `run_seed_sweep`; a text-only table by the store, and so by
`run_seed_sweep` and `decide`. Weights-only loading still works, and `tools regressor-panel`,
which checks no provenance, still pairs pre-6b tables (4.8, link 5). Plan 8's Exit numbers stay
as the floor its finding records (`specs/findings/shared-encoder-first-reading.md`). Stage 7
trains and builds its text-only table on the summaries.

**Realized.** Plan completion records, in the roadmap's Stage 6b entry, the artifact's sha256, the
rows per channel and the mean share of source tokens kept.

**Re-validation** at plan completion touches at least: Stage 9 (summaries are keyed by backbone,
the tokenizer included, not by window alone), Stage 10 (an arm table's provenance carries
`summaries` and `tokenizer`) and Stage 12 (`ArmEncoder.from_files` checks both).

**Deferred items,** handled through /deferred at plan completion:

- Plan 8's "export and outcome read handle a few failures untidily": its tokenizer-name part is
  discharged (4.8, links 2 and 3). The rest stays.
- Plan 8's "encoder, fusion and cache tests leave gaps": its markers-only sidecar test is
  discharged (section 6). The rest stays.
- Plan 8's "code and docs polish": the cache's load messages (4.7), its `docs/text_training.md`
  note on Stage 6b and the Cache Regeneration list (section 8) are discharged. The rest stays.
- Plan 7's manifest tokenizer revision keeps its trigger: no bundle is rebuilt.

**Model routing.** writing-plans for plan 9 runs in a fresh Opus session from this spec;
execution runs on the Sonnet default.
