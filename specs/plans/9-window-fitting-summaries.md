# Window-Fitting Summaries Implementation Plan

**Status: COMPLETE (2026-10-04)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 6b — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 6b. Replace the tail truncation of channel texts over the backbone's
trained window with frozen extractive summaries that fit it. The summaries live in a committed,
hash-pinned artifact, which the tokenization cache and the D9 text-only comparator read through
one resolver. Their sha256 runs from the checkpoint contract through both provenances to the
decision records, and every link checks it.

**Architecture:**

- **The pin and the test seam (Task 1).** `panels/window_summaries.py` maps each backbone to a
  `SummariesPin(path, sha256, window)` in `WINDOW_SUMMARIES`, which stays empty until the Exit.
  `summaries_identity(backbone)` returns the pin's sha256, or None. An autouse fixture in
  `tests/conftest.py` pins MiniLM to a dummy, so every identity site records a non-null value from
  the first task. The cache's sidecar records the identity. A stale sidecar's refusal names each
  key that differs.
- **Units (Task 2).** A summary is built from units whose boundaries are the leakage segmenter's.
  Description and exclusion texts give sentences, re-split into clauses and then segmenter pieces
  when over the budget. Examples texts give entries. The budget is the window less the marker and
  the special tokens: 124 under MiniLM.
- **The artifact and the resolver (Task 3).** `conf/data/window_summaries.csv` has one row per
  summarized (code, channel). `resolve_channel_texts` is the only place summaries enter. It runs
  seven checks in order (spec 4.7), then substitutes. It is torch-free.
- **The build (Task 4).** `naics-embedder data summaries` (`data/window_summaries.py`) selects by
  backbone centrality (spec 4.3). It checks the artifact under a temporary pin before moving it
  into place, and prints the pin to commit.
- **The readers (Task 5).** The tokenization cache and `build_text_only_table` resolve their
  descriptions first. The text-only provenance records `summaries`.
- **The identity chain (Tasks 6–8).**
  - The checkpoint contract records `summaries` from a model input that training passes. Export,
    reads and exact resume refuse a mismatch, and their callers must pass it.
  - The export provenance adds `tokenizer`. `ArmEncoder.from_files` refuses a table read under
    another tokenizer or other summaries, or exported before this stage.
  - The decision store, `TextOnlyRef`, `ArmSpec`, `check_text_only` and `run_seed_sweep`'s
    per-seed check compare the summaries' sha256 with the four fields D9 already compares.
- **Docs (Task 9).**
- **The Exit (Task 10).**
  - It builds the artifact on the real descriptions, pins it, and commits both.
  - It then builds the tokenization cache and a text-only table on them, and shows the chain's
    refusals of plan 8's artifacts.
  - It trains nothing and reads no split.

**Tech Stack:**

- Python 3.10 and 3.12 (CI runs both; `.python-version` pins 3.12).
- torch 2.9.1 and transformers 4.57.1, for the frozen MiniLM (revision 1110a243) that selects
  units. The resolver needs neither.
- polars 1.35.1 (the artifact), pydantic 2.12.4 (contracts and records), numpy (the selection).
- typer and rich (the CLI).
- pytest with xdist; ruff and yapf; mkdocs with mkdocstrings (strict build).

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/window-fitting-summaries.md`, APPROVED, last changed 2d0af69), verbatim

§3, Rulings:

> From the session brief, not re-triaged:
>
> - **S1.** Over-long texts are summarized, not truncated (user ruling, 2026-10-03, Req 9).
> - **S2.** Bundle 301cce28 and its descriptions stay unchanged. Summaries live in their own
>   committed, hash-pinned artifact.
> - **S3.** This spec settles which channels get summaries, and whether they are extractive or
>   abstractive.
> - **S4.** Auditing abstractive summaries against the validation and test queries is a
>   non-selecting read of sealed text, which needs the user's explicit approval. S5 makes it moot:
>   no step of this stage reads a validation or test query.
>
> From the brainstorm, the user's answers on 2026-10-04:
>
> - **S5. Method.** Extractive, for every channel. A summary is whole source units, verbatim and in
>   source order, each at most once.
> - **S6. Selection.** Backbone centrality (4.3).
> - **S7. Design.** Sections 4.1–4.9 as presented: the units, the artifact, a code pin rather than a
>   config key, one resolver for both readers, and the summaries' hash in the checkpoint contract,
>   both provenances and the decision records. The review's corrections (the model's contract
>   input, the resolver's order, the test seam, the in-order check and the sweep's per-seed check)
>   refine that design without changing it.

§5, Error handling:

> Named refusals, ValueError unless stated:
>
> - **Build:** a title over the window; an examples entry or a level-3 piece over the budget; a
>   backbone with no recorded window; any resolver check on the bytes about to be committed; an
>   existing artifact without `--force` (FileExistsError, exit code 1 at the CLI).
> - **Resolver:** each step of 4.7, in that order.
> - **Contract:** a summaries mismatch on exact resume, export, read or the HGCN feeder; a caller
>   that omits `summaries` (TypeError).
> - **Arm encoder:** a table provenance whose `summaries` or `tokenizer` differs or is absent.
> - **Decision records:** a text-only provenance without `summaries`; `check_text_only` on a
>   summaries mismatch; `run_seed_sweep` on a seed whose export provenance differs from the
>   `ArmSpec`.
>
> Guarantees rather than refusals:
>
> - A cache sidecar whose `summaries` differs rebuilds the cache, as a format or marker change does.
> - Descriptions whose texts all fit pass the resolver with no pin and no artifact read.
> - Weights-only migration ignores summaries (4.8).

§6, Testing:

> Tests are written red to green. CI has no `data/` and no MiniLM weights, but it downloads MiniLM's
> tokenizer, which the existing cache, export and arm-encoder fixtures load. New resolver and unit
> tests use a stub or tiny tokenizer where they can, and selection tests inject a stub embedder.
>
> **The seam.** An autouse fixture in `tests/conftest.py` mutates `WINDOW_SUMMARIES` in place
> (`monkeypatch.setitem`, after removing any other entry with `monkeypatch.delitem`) so that it
> holds one entry: the MiniLM backbone mapped to a dummy pin (a path that does not exist, a fixed
> fake sha256, window 128). Tests reach the dict as `window_summaries.WINDOW_SUMMARIES`, never by
> `from … import WINDOW_SUMMARIES`, so every patch acts on the one dict the identity sites read.
> Fixture descriptions fit the window, so the resolver never reads the dummy (4.7, step 1), while
> every identity site records the dummy's sha256. A site that records None instead fails from the
> first task, not at the Exit, provided its test names the MiniLM backbone: the identity-site tests
> (the sidecar, both provenances, the contracts) use MiniLM-named fixtures such as
> `five_code_token_config` and `text_only_comparator_table`, never `tiny-bert` or
> `tiny-backbone`, whose identity is None. A registered marker opts a test out of the fixture: the
> committed-artifact test and the local-only tests below need the real pin. A test that needs "no
> pin" deletes the entry with `monkeypatch.delitem` or uses an unpinned backbone name.
>
> - **Units:** every unit boundary is a segmenter boundary, on crafted texts with `U.S.`, `i.e.`,
>   numbered lists, `;` inside parentheses, cross-references, and `… in Industry 111113.` as a
>   non-final piece (it closes a unit); the level-2 and level-3 re-splits; the raises for a level-3
>   piece over the budget, an examples entry over the budget and a title over the window.
> - **Selection:** with a stub embedder, the greedy keeps the expected units; exact ties go to the
>   earlier unit; both stop rules hold, the plateau stop included; output is in source order and
>   fits with its marker.
> - **Resolver:** substitution; a pinned backbone with no over-window text passes without reading
>   its artifact (the path does not exist); every refusal of 4.7, each at its step, the S5 check
>   with a reordered, a repeated and an edited unit; texts that fit pass through unchanged.
> - **Build:** the temporary-file validation runs before the move, so a failing invariant leaves no
>   artifact; the overwrite refusal is a FileExistsError.
> - **Cache substitution:** a cache built over descriptions with one over-window text, under a
>   test-local pin (`monkeypatch.setitem` on `WINDOW_SUMMARIES`) pointing at a temporary CSV with
>   the right sha256 and a window equal to the test's `max_length`, stores that text's
>   `input_ids` as `tokenize_field` of its summary; with the pin deleted, the build raises. The
>   dummy pin cannot serve here, since the resolver would try to read it. This is the test that
>   fails if `_build_tokenization_cache` records the pin without applying it.
> - **Cache identity:** changing only the pin rebuilds the cache (`monkeypatch.setitem` on
>   `WINDOW_SUMMARIES`, replacing the test that patched `SUMMARIES`); changing only the markers
>   rebuilds it (plan 8's missing test, `_cache_identity`); the sidecar records
>   `summaries_identity(MINILM)`; the mismatch message names the differing keys.
> - **Text-only builder:** it reads resolved texts, and its provenance records `summaries`.
> - **Contract:** a raw contract without `summaries` reads as null; the model passes its
>   `summaries` to both contract builders; exact resume refuses a mismatch, a containment
>   checkpoint included; `validate_supervision_contract` refuses one; weights-only loading does not
>   compare it; the HGCN feeder refuses a checkpoint trained under other summaries.
> - **Export and arm encoder:** the export provenance records `summaries` and `tokenizer`;
>   `from_files` refuses a provenance with other or absent summaries or tokenizer before the model
>   loads.
> - **Decision records:** the store reads `summaries_sha256` and refuses a provenance without it;
>   `check_text_only` refuses a summaries mismatch; `run_seed_sweep` refuses a seed whose export
>   provenance differs from the `ArmSpec` in any of its five fields, and a first-seed refusal
>   leaves the selection log empty.
> - **The committed artifact** (opted out of the seam): its sha256 equals the pin's; `(code,
>   channel)` is unique; every `summary_tokens` is at most `window`; every channel is one of the
>   three.
> - **Local-only, opted out of the seam, skipped without the cached backbone or
>   `data/naics_descriptions.parquet`:** on the real descriptions the resolver accepts the committed
>   artifact, which re-tokenizes every row and re-checks fit, S5 and the subset property, and
>   replaces 162, 106 and 485 texts.

The tasks carry §4 (design), §7 (Exit procedure), §8 (documentation) and §10 (Rollout note). Read
the spec for their full text; a task that cites "4.7" means that section.

### The roadmap (`specs/naics-embedding-roadmap.md` at 2d0af69), verbatim

Stage 6b:

> Objective: Replace the tail truncation of channel texts beyond the backbone's trained
> window with frozen summaries that fit it, before Stage 7 trains the reference
> configuration on them.
> Spec: Req 9 (input windows), under the user's ruling of 2026-10-03 that over-long texts are
> summarized, not truncated; Req 3 (leakage); D9 (the text-only comparator reads the arm's
> text).
> Gap closed: none in the Gap analysis; the stage carries out the 2026-10-03 ruling on Req 9's
> input windows, and Stage 9 still records each candidate's window.
> Consumes: Bundle 301cce28's descriptions (sha256 `fe8c54e3…`), unchanged, since summaries
> written into them would change the description fingerprint the bundle records. The
> overflow at the 128-token window under MiniLM's tokenizer, special tokens included,
> measured on 2026-10-03 without the field marker
> (`specs/completed/shared-encoder-and-projection.md`, Rollout note): description 153 of
> 2,111 texts (sectors 16 of 20, median 246 tokens, maximum 1,131; subsectors 59 of 96),
> examples 105 of 1,075 (19 % of its tokens), excluded 464 of 1,117 (26 %). With the marker
> the cache adds, 162, 106 and 485 texts are truncated (resume, 2026-10-04). Stage 6's
> tokenization cache, whose sidecar carries a `summaries` entry that stays
> null until this stage (format `channels-v3`; a cache built under other summaries is
> rebuilt), and its field markers (`text_model/fields.py`), two tokens of each window under
> MiniLM's tokenizer (the field's name and `:`). Stage 2's
> leakage matcher (`panels/leakage.py`) and frozen role table: an abstractive summary can
> contain a held-out query, and a fix rewrites the summary, never the table.
> Produces: A frozen, committed summaries artifact with provenance, keyed by code, channel,
> source-text sha256 and target window, and pinned by hash; the tokenization cache and the
> text-only builder (`tools text-only-table`, D9) reading it; its hash in the text-only
> provenance, `TextOnlyRef`, `ArmSpec` and `decide`'s D9 check. The stage spec settles which
> channels it covers (the ruling names descriptions; examples carry the most leakage risk,
> and excluded carries Req 8's redirections) and whether summaries are extractive, which
> cannot add a match, or abstractive, which needs an audit against the validation and test
> queries: a non-selecting read of sealed text that needs the user's explicit approval.
> Exit: No covered channel's text is truncated: each text beyond the window reads as its
> summary, which fits with its field marker (test); the leakage check finds no held-out
> query in any summary; the artifact is committed under a pinned hash, and the cache, the
> text-only table's provenance and the decision records name it.
> ROUTING: brainstorming

The decision it names:

> - **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
>   representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
>   the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
>   dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
>   backbone the arm uses after. The comparator measures what taxonomy training adds over the same
>   encoder reading the same text.

The entry predates the spec in two places, and the spec governs both:

- It names the summaries' hash in `decide`'s D9 check. The spec also puts it in the checkpoint
  contract, the export provenance and `run_seed_sweep`'s per-seed check (4.8).
- "the leakage check finds no held-out query in any summary" is met structurally: every segment
  of a summary is a segment of its text, and bundle 301cce28's build checked every segment (4.4).
  No step reads a validation or test query.

### Deferred items this plan touches (`specs/deferred_items.md`), verbatim

Plan 8's three items, each discharged in part (spec §10). Plan completion records the parts; each
item stays open for the rest:

> - [ ] Review Minor: the export and the outcome read handle a few failures untidily.
>       src/naics_embedder/text_model/export.py writes the table (:246) before it hashes the
>       checkpoint and descriptions and writes the provenance (:273), so a failure between them
>       leaves a table without provenance, which `ArmEncoder.from_files` then refuses. It records
>       the checkpoint and descriptions paths as given, unresolved (:250, :258).
>       src/naics_embedder/cli/commands/tools.py puts exception text and paths into Rich markup
>       unescaped (:904, :972), and lets an `UnpicklingError` or `RuntimeError` from a corrupt
>       checkpoint through as a traceback (:903, :971). `tools outcome-panel` refuses a blank
>       `--purpose` only after the model loads. The provenance records no tokenizer name, so
>       `from_files` checks the token window but not the tokenizer. docs/usage.md's outcome-panel
>       paragraph does not say that a read refuses a table exported under another
>       `data_loader.streaming.max_length`. Deferred from plan 8's final review and its re-review:
>       none affects the Exit, whose export and reads shared one config. Size: plan. Done when:
>       each case is fixed or ruled no-action.

> - [ ] Review Minor: the encoder, fusion and cache tests leave gaps.
>       No test changes only the field markers in a tokenization cache's sidecar
>       (`_cache_identity`, src/naics_embedder/text_model/dataloader/tokenization_cache.py).
>       `AttentionFusion`'s autocast case, where the scores are narrower than the input
>       (src/naics_embedder/text_model/fusion.py:92-93), is untested, and
>       tests/unit/test_fusion.py:88 runs its finite-gradient test in eval mode, which drops the
>       train-mode (dropout) case. The tiny fixtures share one width: `WIDTH`, `TINY_HIDDEN` and
>       the default dimension are all 8 (tests/unit/test_encoder.py:37,
>       tests/fixtures/shared_encoder.py:31), which can hide a width and hidden-size mix-up.
>       tests/unit/test_naics_model.py:941 builds its other contract without an encoder record and
>       matches only 'bundle', so it passes for a reason other than the one it names. The HGCN
>       feeder's encoder-mismatch refusal (src/naics_embedder/cli/commands/training.py) has no
>       direct test, and no test runs two backward passes through one graph, the re-entry that
>       `_MpsStateReplay` (src/naics_embedder/text_model/shared_encoder.py) exists for. Deferred
>       from plan 8's reviews as coverage gaps, not defects. Size: plan. Done when: each gap has a
>       test, or a recorded ruling that it needs none.

> - [ ] Review Minor: code and docs polish left by plan 8's reviews.
>       Code: the tokenization cache's load messages still name only the fingerprints, though the
>       sidecar identity also covers format, markers and summaries
>       (src/naics_embedder/text_model/dataloader/tokenization_cache.py:256-262); its cache
>       annotations read `Dict[int, Dict[str, torch.Tensor]]`, but rows nest channel dicts (:254,
>       :282); `LoggingMixin`'s docstring (src/naics_embedder/text_model/mixins/logging.py:28-37)
>       omits its `fusion` dependency; three bare `'moe'` literals (mixins/curriculum.py:162,
>       naics_model.py:606, :676) could use a constant beside `FUSIONS`; and the HGCN feeder
>       (cli/commands/training.py) and the export (text_model/export.py) repeat a three-line
>       code-row flow. Docs: docs/text_training.md could say near :69-73 that channels stay
>       tail-truncated until Stage 6b (R15); its Cache Regeneration list (:414) omits format,
>       markers and summaries; its "Exact Resume versus Weights-Only Migration" heading (:435)
>       sits beside "nothing migrates it" (:92); "buffers" sits alone on :451; its "MoE gating"
>       compiled op (:602, and CLAUDE.md:862) applies under `moe` only; tests/README.md omits the
>       P8 pooler exemption; five statements of c = 1 (CLAUDE.md:41, README.md:212,
>       docs/overview.md:59 and :193, docs/text_training.md:83) could say "by default"; and
>       docs/overview.md:587 cites docs/sampling_architecture.md, deleted in a7517dd. Deferred
>       from plan 8's reviews: each is true or harmless as written. Size: quick-fix. Done when:
>       each is edited or ruled no-action.

Plan 7's item keeps its trigger, because this plan rebuilds no bundle:

> - [ ] Review Minor: the manifest omits the tokenizer revision behind its overflow counts.
>       `InputWindowRecord` (src/naics_embedder/supervision/schema.py:178) records the backbone
>       and window only. Bundle 301cce28's counts came from revision 1110a243, which
>       specs/findings/supervision-target-and-text.md records. Deferred from plan 7's final review
>       because the field changes bundle output. Size: quick-fix. Revisit if: a later stage
>       rebuilds the bundle or bumps its contract.

### Decisions already made (do not re-ask)

The spec's §3 is the authority, and the brief that ordered this plan restated it:

- Extractive summaries for every channel (S5), picked by backbone centrality (S6). Every unit
  boundary is a leakage-segmenter boundary.
- A code pin, `WINDOW_SUMMARIES`, rather than a config key.
- One lazy resolver, which the cache and the text-only builder both call.
- The summaries' sha256 runs through:
  - the checkpoint contract, from a model constructor input;
  - both provenances;
  - `TextOnlyRef` and `ArmSpec`;
  - `check_text_only`;
  - `run_seed_sweep`'s per-seed check, which the user approved explicitly.
- **Sealed text.** No step reads validation or test query text, because the leakage guarantee is
  structural (4.4). Any sealed-text audit needs the user's explicit approval first.
- **At completion.** The three parts of plan 8's items that §10 names are discharged. Plan 7's
  manifest tokenizer revision keeps its trigger.

### This plan's decisions

Each goes beyond a spec line, or reads one. Do not reopen them during execution.

- **P1. Order.** The seam lands first (Task 1), with `summaries_identity` already wired into the
  cache's sidecar and the export provenance. Every identity site then records the dummy's sha256
  from the first task, and a site that records None fails its test.
  - `WINDOW_SUMMARIES` has no MiniLM entry until Task 10.
  - The artifact, its provenance, the pin and the tests that need the real pin land in one
    commit, after the suite passes.
- **P2. The seam.** `tests/conftest.py` defines `DUMMY_SUMMARIES_PIN = SummariesPin(path=
  '/nonexistent/window_summaries.csv', sha256='5' * 64, window=128)`.
  - The autouse fixture `dummy_window_summaries` removes every entry with `monkeypatch.delitem`,
    then sets MiniLM's with `monkeypatch.setitem`.
  - The marker `real_window_summaries`, registered in `pytest_configure`, opts a test out.
  - Tests reach the dict as `window_summaries.WINDOW_SUMMARIES`, never through
    `from … import WINDOW_SUMMARIES`.
- **P3. A torch-free module.** `panels/window_summaries.py` imports the standard library, polars,
  `panels.leakage` and `text_model.fields`, and nothing else.
  - Its `token_counter(tokenizer, *, special_tokens=True)` is the torch-free twin of
    `utils.input_window.token_counter`, since importing `naics_embedder.utils` loads torch.
  - The counter returns `[]` for an empty batch, on which a fast tokenizer raises IndexError.
  - A subprocess test checks that importing the module loads no torch.
- **P4. The budget.** `summary_budget(count_marked, channel, window)` returns
  `window - count_marked([marker(channel)])[0]`: the window less the marker's tokens with `[CLS]`
  and `[SEP]`, so 124 for every channel under MiniLM. A unit's tokens are counted without special
  tokens against it.
- **P5. Units.** `text_units(channel, text, count, budget)` (4.2):
  - A title is one unit, and one over the budget raises.
  - Examples units are the entries split at `EXAMPLES_SEPARATOR`, with blank entries dropped.
  - Description and exclusion texts merge the segmenter's pieces at three levels. Level 1 closes
    a description unit on `.` and an exclusion unit on `.` or `;`. The parenthesis and
    `NO_BREAK_PATTERN` guards apply at levels 1 and 2 only.
  - Any other channel raises.
- **P6. The artifact's I/O.**
  - `write_window_summaries(rows, path)` casts the rows to `SUMMARIES_SCHEMA`, sorts them by
    channel and then code, writes UTF-8 CSV bytes, and returns their sha256.
  - `read_window_summaries` refuses a channel outside the three, and a repeated `(code, channel)`.
  - The resolver reads the bytes once, then hashes and parses those same bytes
    (`_parse_window_summaries`).
- **P7. The resolver.**
  - `pin` defaults to the enum sentinel `DEFAULT`, and None means no pin.
  - The over-window set covers all four channels, titles included, so a title over the window
    fails step 4.
  - Steps 5 and 6 are separate passes over every row, so a source mismatch on any row is reported
    before any extract failure. A test pins that order.
  - It logs `Window summaries for <backbone> replaced channel texts: <counts>`.
- **P8. The build.** `generate_window_summaries(descriptions_path, output_path, *, backbone,
  force=False, model=None, tokenizer=None, revision=None) -> SummariesPin`.
  - Its tests pass a stub tokenizer and model, and patch `backbone_embedder`.
  - The existing-artifact refusal comes first, then `trained_window(backbone)`.
  - Identical (channel, text) pairs are selected once, and each distinct unit is embedded once.
  - `source_tokens` and `summary_tokens` count marked text with special tokens.
  - The provenance's `p10_summary_tokens` is the lower 10th percentile.
- **P9. The command.** `data summaries`:
  - catches `OSError`, which includes FileExistsError and a backbone missing from the cache, and
    `ValueError`; it prints the message and exits 1;
  - echoes `Window summaries: <path>` and `Pin for <backbone>: SummariesPin(...)`, the repr to
    paste into `WINDOW_SUMMARIES` (4.6 says it "prints the sha256 for the pin");
  - defaults `--backbone` to `conf/config.yaml`'s `data_loader.tokenization.tokenizer_name`.
- **P10. The contract.**
  - `contract_for_bundle` and `containment_contract` take `summaries` keyword-only with no default,
    as `validate_supervision_contract` and `load_arm_model` do. 4.8 asks it of the reads only, but
    no builder can then record None silently.
  - `NAICSContrastiveModel` takes `summaries: Optional[str] = None`, just before
    `checkpoint_contract`.
- **P11. The arm encoder's order.** Before the model loads, `ArmEncoder.from_files` checks the
  provenance's entries, as today. It then:
  - requires `summaries` and `tokenizer` (else "exported before Stage 6b");
  - checks the checkpoint's and the table's hashes and the window, as today;
  - checks the tokenizer, then the summaries.

  `load_arm_model`'s contract check then refuses a pre-6b checkpoint behind a current table.
- **P12. The seed check.**
  - `decision/decide.py` gains `D9_FIELDS` and `check_seed_table(spec, seed, fields)`, and
    `check_text_only` compares the same five fields.
  - `decision/sweep.py`'s `_seed_table_fields(table)` reads the export provenance beside a seed's
    table through `provenance_fields`. The provenance must therefore describe that table (its
    sha256 and fingerprint), and a table without one is refused.
- **P13. Test helpers.**
  - `tests/fixtures/window_summaries.py` holds `words` (Task 2), then `WordTokenizer` and
    `pin_artifact` (Task 3). Tests import it; it is not a pytest plugin.
  - `tests/fixtures/shared_encoder.py` gains `truncated_checkpoint` (Task 6), which the export and
    the read tests share.
  - `tests/fixtures/decision.py` gains `SUMMARIES_SHA256 = 'e' * 64` and
    `write_export_provenance`.
  - The build's tests are in `tests/unit/test_window_summaries_build.py`, and the committed
    artifact's in `tests/unit/test_committed_window_summaries.py`.
- **P14. The local-only test** skips when any of these holds:
  - `data/naics_descriptions.parquet` is absent;
  - its sha256 is not the one the provenance records;
  - MiniLM's tokenizer is not in the local Hugging Face cache (`local_files_only=True`).
- **P15. Docs.**
  - Task 9 also edits `docs/api/input_window.md`, which §8's list omits. Its lead sentence says
    every tokenizing path truncates.
  - `_build_tokenization_cache`'s docstring changes with its code in Task 5.

### Recorded deviations

- **The pin's comment.** 4.6 shows `WINDOW_SUMMARIES` with an inline comment. The plan puts the
  comment on the line above, and gives `SummariesPin` an Attributes docstring.
- **§6's "Selection" output check.** "Output is in source order and fits with its marker" is
  checked through the rows (`summary_rows`), since `select_units` returns indices only.
- **§7's step numbers.** Task 10's steps follow §7. Its Step 2 adds a red run of the committed
  tests before the build.
- **§10's deferred items.** §10 handles them "through /deferred at plan completion". `/deferred`
  leaves ticking a plan's work to that plan's completion run and has no form for a part, so Plan
  completion's Step 5 records each of plan 8's three parts on its item, whose box stays open.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - To keep a vertical Polars chain, fence it with `# yapf: disable` / `# yapf: enable`.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. At the end,
  `./scripts/format_code.sh --check --all` must pass. The plan's code is already formatted, so the
  script should change nothing. If it changes a file, keep its layout.
- **Git.**
  - Never push, and never push to `main`.
  - Never push, cherry-pick or merge the held local commits "config" and "graph config". They are
    on local `main` only, and named by subject because every sync rewrites their SHAs.
  - Never run bare `git stash`.
  - Commit on this branch only, and end each message with the session's attribution trailer.
  - Add files by explicit path, never `git add -A` or `git add .`.
- **Data safety.**
  - The main checkout's `data/` and `checkpoints/` are reference only; never write there.
  - Bundle 301cce28, `data/naics_descriptions.parquet` (sha256 `fe8c54e3…`) and
    `checkpoints/plan8_exit/` are canonical. Task 10 clones them into this worktree with
    `cp -cR` and `cp -c`. Never symlink or rebuild them.
  - Tests write only under `tmp_path`.
- **Sealed text and splits.**
  - No step reads a validation or test query, and no step opens a split:
    - no `OutcomePanel.open_test`, `RegressorPanel.open_outer` or `--split test`;
    - no `tools outcome-panel` or `tools regressor-panel`.
  - The Exit trains nothing and reads no split, so the selection log gains no record.
  - A sealed-text audit needs the user's explicit approval first.
- **Configs.**
  - `conf/config.yaml` keeps `supervision.manifest_path: null`, and this plan adds no config key.
  - Task 10 passes `supervision.manifest_path=…` as a `key=value` override and never commits it:
    the pin test fails on a committed path.
- **Downloads.**
  - Never download Census or QCEW files.
  - The backbone and its tokenizer come from the local Hugging Face cache.
  - Task 10 runs every command under `HF_HUB_OFFLINE=1`.
- **Devices.** On MPS, cast with `.cpu().to(torch.float64)`, never `.to(device='cpu', dtype=…)`,
  which raises on MPS.
- **Deferred items.** Promote no open item of `specs/deferred_items.md` beyond the parts §10 names.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Docs.**
  - `uv run mkdocs build --strict` must pass locally, because PR CI never builds the docs.
  - Griffe is strict: every documented parameter gets its own `Args:` entry.
- **Tests.**
  - Rich wraps at 80 columns on CI, so CLI tests match on `result.output.replace('\n', '')`.
  - pyproject's `addopts` has `-v`, so a node-ID listing needs `--collect-only -q -q`.
- **Bash tool.** It runs zsh.
  - Brace variables (`${FILE}`): `$FILE:t` is a zsh modifier.
  - Quote `=`-leading words (`echo '====='`); an unquoted one aborts the command.
  - If the tool refuses a heredoc or a compound command, run one plain command per call and
    write files with the Write tool.

## Workspace

- **Worktree:** `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-9-window-summaries`,
  or the app worktree this session runs in. Run every command from its root.
- **Branch:** `claude/plan-9-window-summaries`, cut at this plan's commit
  (`docs(plans): add plan 9, window-fitting summaries`), which is the tip of
  `claude/stage-6b-window-summaries-08c971d1`. The handoff names its SHA.
  - An app worktree is cut from local `main`. Local main carries the held commits and lacks the
    spec, so first run `git checkout --no-track -B claude/plan-9-window-summaries <plan SHA>`.
  - Otherwise, from the main checkout, run `git worktree add -b claude/plan-9-window-summaries
    .claude/worktrees/plan-9-window-summaries <plan SHA>`.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on
  `claude/stage-6b-window-summaries-08c971d1`, and nothing is checked out there.
  - Local `main` is f5c8307 plus the held commits "config" and "graph config".
  - origin/main is f5c8307 (PR #122).
- **Real inputs, read only (Task 10):**
  - The bundle:
    `/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/`
    (`codebook_fingerprint` `4662b826…`, `description_fingerprint` `fe8c54e3…`).
  - The descriptions: `/Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`,
    sha256 `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.
  - Plan 8's Exit artifacts: `/Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/`
    (`last.ckpt`, `arm_table.parquet`, `arm_table_provenance.json`, `text_only.parquet`,
    `text_only_provenance.json`).
  - The backbone and tokenizer:
    `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` (revision
    1110a243).
- **Outputs (Task 10):**
  - Committed: `conf/data/window_summaries.csv` and `conf/data/window_summaries_provenance.json`.
  - Gitignored, in this worktree:
    - `data/` (the clones and the token cache);
    - `checkpoints/plan8_exit/` (clones) and `checkpoints/plan9_exit/` (the text-only table and
      its provenance);
    - `logs/`.
  - Before the worktree is removed (finishing-a-development-branch), copy
    `checkpoints/plan9_exit/` and the execution ledger into the main checkout. Removing a
    worktree deletes its ignored files.
- **Working directory.** The Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before every commit and before Task 10's commands. If it is not this
  worktree, `cd` back first, or prefix the command with `cd <worktree> &&`.

## File structure

| Path | Change | Responsibility |
|---|---|---|
| `src/naics_embedder/panels/window_summaries.py` | Create (Task 1), extend (Tasks 2, 3), pin (Task 10) | The pin and its identity; units; the artifact; the resolver |
| `src/naics_embedder/panels/leakage.py` | Modify (Task 2) | `SENTENCE_BREAK` made public |
| `src/naics_embedder/text_model/dataloader/tokenization_cache.py` | Modify (Tasks 1, 5, 9) | The sidecar's `summaries`; named mismatches; resolves before tokenizing |
| `src/naics_embedder/text_model/export.py` | Modify (Tasks 1, 6, 7) | Provenance `summaries` and `tokenizer`; `load_arm_model(…, summaries=…)` |
| `src/naics_embedder/data/window_summaries.py` | Create (Task 4) | Centrality selection, the rows, the build and its provenance |
| `src/naics_embedder/cli/commands/data.py` | Modify (Task 4) | `data summaries` |
| `src/naics_embedder/panels/text_only.py` | Modify (Task 5) | Resolves its descriptions; provenance `summaries` |
| `src/naics_embedder/supervision/checkpoints.py` | Modify (Task 6) | `CheckpointContract.summaries`; `summaries` keyword-only everywhere |
| `src/naics_embedder/text_model/naics_model.py` | Modify (Task 6) | The `summaries` input, recorded in either contract |
| `src/naics_embedder/cli/commands/training.py` | Modify (Task 6) | `build_model_from_config` and `runtime_contract_for` pass the identity |
| `src/naics_embedder/text_model/arm_encoder.py` | Modify (Tasks 6, 7) | Passes the identity; refuses pre-6b tables and other tokenizers or summaries |
| `src/naics_embedder/decision/records.py`, `store.py`, `decide.py`, `sweep.py` | Modify (Task 8) | `summaries_sha256`; five-field D9 checks; the per-seed check |
| `src/naics_embedder/supervision/schema.py`, `src/naics_embedder/utils/input_window.py` | Modify (Task 9) | Docstrings: summaries, not truncation |
| `conf/data/window_summaries.csv`, `conf/data/window_summaries_provenance.json` | Create (Task 10, by `data summaries`) | The artifact and its provenance |
| `tests/conftest.py` | Modify (Task 1) | The dummy-pin seam and its opt-out marker |
| `tests/fixtures/window_summaries.py` | Create (Task 2), extend (Task 3) | `words`; `WordTokenizer` and `pin_artifact` |
| `tests/unit/test_window_summaries.py` | Create (Task 1), extend (Tasks 2, 3) | The pin, the units, the artifact, the resolver |
| `tests/unit/test_tokenization_cache.py` | Modify (Tasks 1, 5) | Sidecar identity and messages; substitution |
| `tests/unit/test_export.py` | Modify (Tasks 1, 6, 7) | Provenance entries; the contract's refusals |
| `tests/unit/test_window_summaries_build.py` | Create (Task 4) | Selection, rows, the build |
| `tests/unit/test_cli_commands.py` | Modify (Task 4) | The command's wiring |
| `tests/unit/test_text_only.py` | Modify (Task 5) | The comparator reads summaries |
| `tests/fixtures/shared_encoder.py` | Modify (Task 6) | `shared_model` records summaries; `truncated_checkpoint` |
| `tests/unit/test_checkpoint_contract.py`, `test_naics_model.py`, `test_cli_training.py` | Modify (Task 6) | The contract's summaries |
| `tests/unit/test_arm_encoder.py` | Modify (Tasks 6, 7) | The read's refusals |
| `tests/fixtures/decision.py`, `tests/unit/test_decision.py`, `test_decision_rule.py`, `test_decision_store.py`, `test_decision_sweep.py` | Modify (Task 8) | The records and the sweep |
| `docs/text_training.md`, `docs/usage.md`, `docs/api/input_window.md`, `docs/.nav.yml`, `CLAUDE.md` | Modify (Task 9) | Summaries replace truncation; `data summaries` |
| `docs/api/window_summaries.md` | Create (Task 9) | The API page |
| `tests/unit/test_committed_window_summaries.py` | Create (Task 10) | The committed artifact; the local-only resolver check |

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would do any of these:
  - write to the main checkout's `data/` or `checkpoints/`;
  - commit `supervision.manifest_path`;
  - read a validation or test query, or open a split;
  - download a Census or QCEW file.
- `origin/main` gains a commit that touches a file in **File structure**, the roadmap or
  `specs/deferred_items.md`, or an open PR does.
- In Task 10, any of these:
  - `data summaries` raises;
  - its resolver logs other counts than `{'description': 162, 'examples': 106, 'excluded': 485}`;
  - the cache build raises, or its sidecar's `summaries` is not the pin's sha256;
  - a refusal in Step 7 does not happen.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/plan-9-window-summaries` and nothing else. If the line ends in
`...origin/main`, run `git branch --unset-upstream`.

Run: `git log --oneline origin/main..HEAD`
Expected: exactly five commits, newest first:
- this plan's commit (`docs(plans): add plan 9, window-fitting summaries`);
- `2d0af69 docs(specs): fold the second review round into the Stage 6b spec`;
- `4ff8b7b docs(specs): revise the Stage 6b spec after a five-lens review`;
- `567a187 docs(specs): Stage 6b window-fitting summaries design spec`;
- `b724f66 docs(roadmap): resume after Stage 6 and route Stage 6b`.

If "config" or "graph config" appears, stop.

Run: `git fetch origin`, then
`git log --oneline HEAD..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: no output.
- If anything landed, read it.
- If it touches a file in **File structure**, the roadmap or `specs/deferred_items.md`, stop and
  ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [x] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number.

Run: `uv run python -c "import peft, polars, pydantic, pytorch_lightning, torch, transformers; print(peft.__version__, polars.__version__, pydantic.__version__, pytorch_lightning.__version__, torch.__version__, transformers.__version__)"`
Expected: `0.17.1 1.35.1 2.12.4 2.5.5 2.9.1 4.57.1`. If they differ, stop and ask.

- [x] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `1845 passed, 1 skipped`, measured at 2d0af69 on 2026-10-04 (the skip needs CUDA).
- Each later full-suite run must pass with that one skip, and each task gives its count.
- The warnings count varies under xdist; ignore it.

| After | Passed | New tests |
|---|---:|---:|
| Baseline | 1845 | |
| Task 1 | 1851 | 6 (7 new, 1 renamed) |
| Task 2 | 1866 | 15 |
| Task 3 | 1889 | 23 |
| Task 4 | 1902 | 13 |
| Task 5 | 1906 | 4 |
| Task 6 | 1920 | 14 |
| Task 7 | 1925 | 5 |
| Task 8 | 1933 | 8 |
| Task 9 | 1933 | 0 |
| Task 10 | 1937 | 4 |

From Task 10 on, the local-only test runs, because Step 1 clones `data/naics_descriptions.parquet`.
Without that file, as on CI, the final count is 1936 passed, 2 skipped.

- [x] **Step 4: Check the real inputs, read-only**

Task 10 is the only reader of these files. Check them now so a missing input fails early.

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
Expected: `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Run: `ls /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/last.ckpt /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/arm_table.parquet /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/arm_table_provenance.json /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/text_only.parquet /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/text_only_provenance.json`
Expected: all six paths listed, no error.

- [x] **Step 5: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–9 each get a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**.
- Every code block in a task is exact, so an implementer copies it. Each "Replace" text occurs
  exactly once in its file when its edit is made, and the edits of one file are made in the order
  given.
- Task 9 (documentation) may go to the docs-writer agent. Its acceptance checks are the gate.
- Task 10, **Final verification** and **Plan completion** run inline in the controller session.
  Task 10 runs the build on the real descriptions and applies the stop-and-ask conditions.

### Task 1: The pin, its identity and the test seam

`panels/window_summaries.py` starts with the pin (4.6): `WINDOW_SUMMARIES`, empty until Task 10,
and `summaries_identity`. The seam in `tests/conftest.py` pins MiniLM to a dummy (P2), so the
cache's sidecar and the export provenance, which this task wires to the identity, record a
non-null value from now on (P1). The deleted `SUMMARIES` constant goes with them. The cache's two
load refusals name each sidecar key that differs (4.7, "Cache identity"; plan 8's polish item).

**Files:**
- Create: `src/naics_embedder/panels/window_summaries.py`
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py:14-19, 26-34, 213-219, 232-250, 259-268, 315-323`
- Modify: `src/naics_embedder/text_model/export.py:27-31, 33-39, 258-264`
- Modify: `tests/conftest.py:11-16, 42-47`
- Modify: `tests/unit/test_export.py:13-18, 187-193`
- Modify: `tests/unit/test_tokenization_cache.py:18-23, 26-31, 803-809, 815-821, 838-851`
- Create: `tests/unit/test_window_summaries.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - In `naics_embedder.panels.window_summaries`:
    - `WINDOW_SUMMARIES_PATH = 'conf/data/window_summaries.csv'`;
    - `SummariesPin(path: str, sha256: str, window: int)`, a frozen dataclass;
    - `WINDOW_SUMMARIES: Dict[str, SummariesPin]`, empty;
    - `summaries_identity(backbone: str) -> Optional[str]`.
  - In `tests/conftest.py`: `MINILM`, `DUMMY_SUMMARIES_PIN` (sha256 `'5' * 64`, window 128, a path
    that does not exist), the autouse fixture `dummy_window_summaries`, and the marker
    `real_window_summaries`.
  - In `tokenization_cache.py`: the sidecar's `summaries` is `summaries_identity(tokenizer_name)`.
    `_identity_mismatch(cfg, description_fingerprint, codebook_fingerprint) -> Optional[str]`
    names a stale sidecar's differing keys, as `{key: (recorded, expected)}`, with `'<absent>'`
    for a key one side lacks. `SUMMARIES` is gone.
  - The export provenance's `summaries` is `summaries_identity(token_config.tokenizer_name)`.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_window_summaries.py`:

```python
'''
Window-fitting summaries (roadmap Stage 6b): the pin and its identity (spec 4.6), the units
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import subprocess
import sys
from pathlib import Path

import pytest

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

# -------------------------------------------------------------------------------------------------
# The pin and its identity
# -------------------------------------------------------------------------------------------------

def test_the_identity_is_the_pins_sha256_and_none_without_a_pin(monkeypatch):
    pin = SummariesPin(path='summaries.csv', sha256='a' * 64, window=16)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, 'tiny-backbone', pin)

    assert summaries_identity('tiny-backbone') == 'a' * 64
    assert summaries_identity('unpinned/backbone') is None

def test_the_seam_pins_minilm_alone_to_a_pin_no_test_can_read():
    '''tests/conftest.py's autouse seam (spec section 6).'''

    assert list(window_summaries.WINDOW_SUMMARIES) == [MINILM]
    pin = window_summaries.WINDOW_SUMMARIES[MINILM]
    assert summaries_identity(MINILM) == pin.sha256
    assert pin.sha256 is not None
    assert pin.window == 128
    assert not Path(pin.path).exists()

@pytest.mark.real_window_summaries
def test_the_marker_leaves_the_committed_pins_alone():
    # Every committed pin names its artifact; the seam's dummy names a file that does not exist
    for pin in window_summaries.WINDOW_SUMMARIES.values():
        assert Path(pin.path).is_file()

def test_the_module_imports_no_torch():
    '''The resolver runs in the token cache and the text-only builder; it loads no model.'''

    imported = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import naics_embedder.panels.window_summaries; '
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert imported.stdout.strip() == 'False'
```

In `tests/conftest.py`, replace:

```python
import pytest
import torch

pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
```

with:

```python
import pytest
import torch

from naics_embedder.panels import window_summaries

pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
```

Replace:

```python
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Fixtures
# -------------------------------------------------------------------------------------------------
```

with:

```python
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

# -------------------------------------------------------------------------------------------------
# Window summaries: the dummy pin (Stage 6b spec, section 6)
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# A pin no test can read: its file does not exist. Fixture texts fit their windows, so the resolver
# never looks for it, while every identity site records its sha256.
DUMMY_SUMMARIES_PIN = window_summaries.SummariesPin(
    path='/nonexistent/window_summaries.csv', sha256='5' * 64, window=128
)

def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'real_window_summaries: reads the committed window-summaries pins; the dummy-pin seam '
        'stays off',
    )

@pytest.fixture(autouse=True)
def dummy_window_summaries(request, monkeypatch):
    '''
    ``WINDOW_SUMMARIES`` holds one entry, MiniLM's dummy pin, unless the test is marked
    ``real_window_summaries``.

    The dict is changed in place, and every identity site reads it at call time, so each site
    records the dummy's sha256. A test that needs no pin deletes the entry with
    ``monkeypatch.delitem`` or names an unpinned backbone.
    '''

    if request.node.get_closest_marker('real_window_summaries') is not None:
        return
    for backbone in list(window_summaries.WINDOW_SUMMARIES):
        monkeypatch.delitem(window_summaries.WINDOW_SUMMARIES, backbone)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, MINILM, DUMMY_SUMMARIES_PIN)

# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Fixtures
# -------------------------------------------------------------------------------------------------
```

In `tests/unit/test_tokenization_cache.py`, replace:

```python
import torch
from transformers import AutoTokenizer

from naics_embedder.text_model.dataloader.tokenization_cache import (
    _acquire_lock,
    _build_tokenization_cache,
```

with:

```python
import torch
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity
from naics_embedder.text_model.dataloader.tokenization_cache import (
    _acquire_lock,
    _build_tokenization_cache,
```

Replace:

```python
    _save_tokenization_cache,
    _write_cache_sidecar,
    get_tokens,
    tokenization_cache,
)
from naics_embedder.utils.config import TokenizationConfig
```

with:

```python
    _save_tokenization_cache,
    _write_cache_sidecar,
    get_tokens,
    load_verified_tokenization_cache,
    tokenization_cache,
)
from naics_embedder.utils.config import TokenizationConfig
```

Replace:

```python
    assert torch.equal(cache[0]['excluded']['input_ids'], _padded(tokenizer, ''))

@pytest.mark.unit
def test_the_sidecar_records_the_markers_and_null_summaries(tokenization_config, counted_builds):
    tokenization_cache(tokenization_config, **FINGERPRINTS)

    cache_path = Path(tokenization_config.output_path)
```

with:

```python
    assert torch.equal(cache[0]['excluded']['input_ids'], _padded(tokenizer, ''))

@pytest.mark.unit
def test_the_sidecar_records_the_markers_and_the_pins_summaries(
    tokenization_config, counted_builds
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)

    cache_path = Path(tokenization_config.output_path)
```

Replace:

```python
        'excluded': 'excluded: ',
        'examples': 'examples: ',
    }
    assert sidecar['summaries'] is None

@pytest.mark.unit
def test_a_cache_in_the_unmarked_v2_format_is_rebuilt(
```

with:

```python
        'excluded': 'excluded: ',
        'examples': 'examples: ',
    }
    # The seam's dummy pin for MiniLM (tests/conftest.py): a site that recorded None would fail
    assert sidecar['summaries'] == summaries_identity(tokenization_config.tokenizer_name)
    assert sidecar['summaries'] is not None

@pytest.mark.unit
def test_a_cache_in_the_unmarked_v2_format_is_rebuilt(
```

Replace:

```python
    assert len(counted_builds) == 1

@pytest.mark.unit
def test_a_cache_built_under_other_summaries_is_rebuilt(
    tokenization_config, counted_builds, monkeypatch
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)
    monkeypatch.setattr(
        'naics_embedder.text_model.dataloader.tokenization_cache.SUMMARIES', 'a' * 64
    )

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 2
```

with:

```python
    assert len(counted_builds) == 1

def _repin(monkeypatch, backbone: str) -> str:
    '''Pin other summaries for the backbone, as a new committed artifact would.'''

    monkeypatch.setitem(
        window_summaries.WINDOW_SUMMARIES,
        backbone,
        SummariesPin(path='other_summaries.csv', sha256='a' * 64, window=128),
    )
    return 'a' * 64

@pytest.mark.unit
def test_a_cache_built_under_other_summaries_is_rebuilt(
    tokenization_config, counted_builds, monkeypatch
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)
    _repin(monkeypatch, tokenization_config.tokenizer_name)

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 2

@pytest.mark.unit
def test_a_cache_built_under_other_markers_is_rebuilt(
    tokenization_config, counted_builds, monkeypatch
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)
    monkeypatch.setattr(
        'naics_embedder.text_model.dataloader.tokenization_cache.marker',
        lambda field: f'[{field}] ',
    )

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 2

@pytest.mark.unit
def test_a_stale_sidecar_is_refused_naming_each_key_that_differs(
    tokenization_config, counted_builds, monkeypatch
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)
    recorded = summaries_identity(tokenization_config.tokenizer_name)
    expected = _repin(monkeypatch, tokenization_config.tokenizer_name)

    with pytest.raises(RuntimeError, match='run prepare_data') as verified:
        load_verified_tokenization_cache(tokenization_config, **FINGERPRINTS)
    with pytest.raises(RuntimeError, match='Tokenization cache not found') as unlocked:
        tokenization_cache(tokenization_config, **FINGERPRINTS, use_locking=False)

    for refusal in (verified, unlocked):
        message = str(refusal.value)
        assert f"'summaries': ('{recorded}', '{expected}')" in message
        # Only the keys that differ, with both values
        assert 'description_fingerprint' not in message
```

In `tests/unit/test_export.py`, replace:

```python
from naics_embedder.cli.commands import training as training_cli
from naics_embedder.panels.regressor import coordinate_matrix, table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
```

with:

```python
from naics_embedder.cli.commands import training as training_cli
from naics_embedder.panels.regressor import coordinate_matrix, table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
```

Replace:

```python
        'path': str(five_code_descriptions_parquet),
        'sha256': sha256_file(five_code_descriptions_parquet),
    }
    assert provenance['summaries'] is None
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
```

with:

```python
        'path': str(five_code_descriptions_parquet),
        'sha256': sha256_file(five_code_descriptions_parquet),
    }
    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(MINILM)
    assert provenance['summaries'] is not None
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_window_summaries.py -q`
Expected: pytest stops at `ImportError while loading conftest '…/tests/conftest.py'`, with
`ImportError: cannot import name 'window_summaries' from 'naics_embedder.panels'`. Every test
stops there until the module exists.

- [x] **Step 3: Create the pin**

Create `src/naics_embedder/panels/window_summaries.py`:

```python
'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

``WINDOW_SUMMARIES`` pins, per backbone, the committed artifact of extractive summaries, and
``summaries_identity`` is the sha256 every identity site records: the token cache's sidecar, the
checkpoint contract, the export and text-only provenances, and the decision store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Optional

# -------------------------------------------------------------------------------------------------
# The pin
# -------------------------------------------------------------------------------------------------

WINDOW_SUMMARIES_PATH = 'conf/data/window_summaries.csv'

@dataclass(frozen=True)
class SummariesPin:
    '''
    A committed summaries artifact.

    Attributes:
        path: The artifact, relative to the repository root.
        sha256: The artifact's sha256, which every identity site records.
        window: The trained window the summaries fit.
    '''

    path: str
    sha256: str
    window: int

# Keyed by backbone. Plan 9's Exit adds MiniLM's entry together with the artifact it pins.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {}

def summaries_identity(backbone: str) -> Optional[str]:
    '''
    The sha256 of the backbone's pinned summaries, or None when it has no pin.

    Args:
        backbone: A Hugging Face model or tokenizer name.

    Returns:
        The pin's sha256, or None.
    '''

    pin = WINDOW_SUMMARIES.get(backbone)
    return None if pin is None else pin.sha256
```

- [x] **Step 4: Run the tests to see what the identity sites still record**

Run: `uv run pytest tests/unit/test_window_summaries.py tests/unit/test_tokenization_cache.py tests/unit/test_export.py -q`
Expected: `4 failed, 51 passed`:
- `test_the_sidecar_records_the_markers_and_the_pins_summaries` and
  `test_the_provenance_names_the_table_and_the_checkpoint` fail on
  `assert None == '5555…'`: the sidecar and the provenance still record `SUMMARIES`, which is None.
- `test_a_cache_built_under_other_summaries_is_rebuilt` fails on `assert 1 == 2`: a new pin does
  not rebuild the cache.
- `test_a_stale_sidecar_is_refused_naming_each_key_that_differs` fails with
  "DID NOT RAISE <class 'RuntimeError'>".
- `test_a_cache_built_under_other_markers_is_rebuilt` already passes: the sidecar has recorded the
  markers since plan 8, and the test is the one plan 8's review found missing.

- [x] **Step 5: Record the identity in the cache and the export**

In `src/naics_embedder/text_model/dataloader/tokenization_cache.py`, replace:

```python
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_embedder.text_model.fields import CHANNELS, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window
```

with:

```python
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.text_model.fields import CHANNELS, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window
```

Replace:

```python
# records another format in its sidecar, or none, so it is rebuilt.
CACHE_FORMAT = 'channels-v3'

# Stage 6b's summaries artifact, by hash: null until it lands. It is part of the cache's identity,
# so a cache built under other summaries is rebuilt.
SUMMARIES: Optional[str] = None

# Disable tokenizer parallelism to avoid fork issues with multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

with:

```python
# records another format in its sidecar, or none, so it is rebuilt.
CACHE_FORMAT = 'channels-v3'

# Disable tokenizer parallelism to avoid fork issues with multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

Replace:

```python
            channel: marker(channel)
            for channel in CHANNELS
        },
        'summaries': SUMMARIES,
    }

def _write_cache_sidecar(
```

with:

```python
            channel: marker(channel)
            for channel in CHANNELS
        },
        # The pinned summaries' sha256 (panels/window_summaries.py), so a cache built under other
        # summaries is rebuilt
        'summaries': summaries_identity(cfg.tokenizer_name),
    }

def _write_cache_sidecar(
```

Replace:

```python
    )
    temp_path.replace(sidecar)

def _sidecar_matches(
    cfg: TokenizationConfig,
    description_fingerprint: str,
    codebook_fingerprint: str,
) -> bool:
    sidecar = _sidecar_path(Path(cfg.output_path))
    if not sidecar.exists():
        return False
    try:
        recorded = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == _cache_identity(cfg, description_fingerprint, codebook_fingerprint)

def load_verified_tokenization_cache(
    cfg: TokenizationConfig,
```

with:

```python
    )
    temp_path.replace(sidecar)

def _identity_mismatch(
    cfg: TokenizationConfig,
    description_fingerprint: str,
    codebook_fingerprint: str,
) -> Optional[str]:
    '''
    Why the cache's sidecar does not record the expected identity, or None when it does.

    The sidecar matches only when it equals the expected identity exactly. The reason names each
    key that differs, with its recorded and expected values.
    '''

    sidecar = _sidecar_path(Path(cfg.output_path))
    if not sidecar.exists():
        return f'it has no fingerprint sidecar at {sidecar}'
    try:
        recorded = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return f'its fingerprint sidecar {sidecar} is unreadable'
    expected = _cache_identity(cfg, description_fingerprint, codebook_fingerprint)
    if recorded == expected:
        return None
    if not isinstance(recorded, dict):
        return f'its fingerprint sidecar {sidecar} is not a JSON object'
    differing = {
        key: (recorded.get(key, '<absent>'), expected.get(key, '<absent>'))
        for key in sorted(set(recorded) | set(expected))
        if key not in recorded or key not in expected or recorded[key] != expected[key]
    }
    return f'its sidecar differs (recorded, expected): {differing}'

def _sidecar_matches(
    cfg: TokenizationConfig,
    description_fingerprint: str,
    codebook_fingerprint: str,
) -> bool:
    return _identity_mismatch(cfg, description_fingerprint, codebook_fingerprint) is None

def load_verified_tokenization_cache(
    cfg: TokenizationConfig,
```

Replace:

```python
        RuntimeError: If the cache is missing or was built from other inputs.
    '''

    if not _sidecar_matches(cfg, description_fingerprint, codebook_fingerprint):
        raise RuntimeError(
            f'Tokenization cache at {cfg.output_path} is missing or does not match the expected '
            'descriptions/codebook fingerprints; run prepare_data() to rebuild it'
        )
    cache = _load_tokenization_cache(cfg.output_path)
    if cache is None:
```

with:

```python
        RuntimeError: If the cache is missing or was built from other inputs.
    '''

    mismatch = _identity_mismatch(cfg, description_fingerprint, codebook_fingerprint)
    if mismatch is not None:
        raise RuntimeError(
            f'Tokenization cache at {cfg.output_path} was not built from the expected inputs: '
            f'{mismatch}; run prepare_data() to rebuild it'
        )
    cache = _load_tokenization_cache(cfg.output_path)
    if cache is None:
```

Replace:

```python
    # If we're not using locking (e.g., cache should already exist), fail fast
    if not use_locking:
        raise RuntimeError(
            f'Tokenization cache not found at {cache_path} (or its fingerprint sidecar does not '
            'match) and locking disabled. '
            f'Cache should be built in prepare_data() before workers are spawned.'
        )

    lock_path = cache_path.with_suffix('.lock')
```

with:

```python
    # If we're not using locking (e.g., cache should already exist), fail fast
    if not use_locking:
        mismatch = _identity_mismatch(cfg, **identity)
        reason = f' ({mismatch})' if mismatch is not None else ''
        raise RuntimeError(
            f'Tokenization cache not found at {cache_path}{reason} and locking disabled. '
            'Cache should be built in prepare_data() before workers are spawned.'
        )

    lock_path = cache_path.with_suffix('.lock')
```

In `src/naics_embedder/text_model/export.py`, replace:

```python
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
```

with:

```python
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
```

Replace:

```python
    validate_supervision_contract,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import SUMMARIES, tokenization_cache
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig
```

with:

```python
    validate_supervision_contract,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig
```

Replace:

```python
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': SUMMARIES,
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
```

with:

```python
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': summaries_identity(token_config.tokenizer_name),
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_window_summaries.py tests/unit/test_tokenization_cache.py tests/unit/test_export.py -q`
Expected: `55 passed`.

Run: `git grep -n -w SUMMARIES -- src tests`
Expected: no output.

- [x] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/panels/window_summaries.py src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/text_model/export.py tests/conftest.py tests/unit/test_window_summaries.py tests/unit/test_tokenization_cache.py tests/unit/test_export.py`
Run: `uv run pytest -n auto -q`
Expected: `1851 passed, 1 skipped`.

```bash
git add src/naics_embedder/panels/window_summaries.py src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/text_model/export.py tests/conftest.py tests/unit/test_window_summaries.py tests/unit/test_tokenization_cache.py tests/unit/test_export.py
git commit -m "feat(window-summaries): pin summaries by backbone and record their identity"
```

### Task 2: Units

A summary keeps whole units, and every unit boundary is a boundary of the leakage segmenter
(4.2), whose break pattern becomes the public `SENTENCE_BREAK`. This task adds the token counter
(P3), the budget (P4) and `text_units` (P5). Its tests use a word-counting stub and MiniLM's
tokenizer, which CI downloads, as the cache tests already do.

**Files:**
- Modify: `src/naics_embedder/panels/leakage.py:33-39, 56-62`
- Modify: `src/naics_embedder/panels/window_summaries.py:10-17, 51-52`
- Create: `tests/fixtures/window_summaries.py`
- Modify: `tests/unit/test_window_summaries.py:8-21, 59-61`

**Interfaces:**
- Consumes: `marker(field)` (`text_model/fields.py`); `EXAMPLES_SEPARATOR` (`panels/leakage.py`).
- Produces:
  - In `naics_embedder.panels.leakage`: `SENTENCE_BREAK`, the compiled
    `r'(?<=[.;])\s+'`, renamed from `_SENTENCE_BREAK`.
  - In `naics_embedder.panels.window_summaries`:
    - `TokenCounter = Callable[[Sequence[str]], List[int]]`;
    - `token_counter(tokenizer, *, special_tokens: bool = True) -> TokenCounter`;
    - `SUMMARY_CHANNELS = ('description', 'examples', 'excluded')`;
    - `UNIT_RULE = 'sentence-clause-piece-v1'` and `NO_BREAK_PATTERN`, 4.2's normative pattern;
    - `summary_budget(count_marked: TokenCounter, channel: str, window: int) -> int`;
    - `text_units(channel: str, text: str, count: TokenCounter, budget: int) -> List[str]`, where
      `count` counts tokens without special tokens.
  - In `tests/fixtures/window_summaries.py`: `words(texts) -> List[int]`, one token per word.

- [x] **Step 1: Write the failing tests**

Create `tests/fixtures/window_summaries.py`:

```python
'''
Stubs for the window-fitting summaries tests (roadmap Stage 6b): a token counter that counts
words.
'''

from typing import List, Sequence

def words(texts: Sequence[str]) -> List[int]:
    '''A stub token counter: one token per whitespace-separated word.'''

    return [len(text.split()) for text in texts]
```

In `tests/unit/test_window_summaries.py`, replace:

```python
from pathlib import Path

import pytest

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

# -------------------------------------------------------------------------------------------------
# The pin and its identity
# -------------------------------------------------------------------------------------------------
```

with:

```python
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.leakage import SENTENCE_BREAK
from naics_embedder.panels.window_summaries import (
    SummariesPin,
    summaries_identity,
    summary_budget,
    text_units,
    token_counter,
)
from tests.fixtures.window_summaries import words

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

@pytest.fixture(scope='module')
def minilm_tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

# -------------------------------------------------------------------------------------------------
# The pin and its identity
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
    )

    assert imported.stdout.strip() == 'False'
```

with:

```python
    )

    assert imported.stdout.strip() == 'False'

# -------------------------------------------------------------------------------------------------
# Units
# -------------------------------------------------------------------------------------------------

def test_the_budget_is_the_window_less_the_marker_and_special_tokens(minilm_tokenizer):
    count_marked = token_counter(minilm_tokenizer)

    for channel in ('title', 'description', 'examples', 'excluded'):
        assert summary_budget(count_marked, channel, 128) == 124

def test_the_counter_counts_special_tokens_only_when_asked(minilm_tokenizer):
    # 'soybean' is three word pieces: soy, ##be, ##an
    assert token_counter(minilm_tokenizer)(['soybean farming', 'corn']) == [6, 3]
    assert token_counter(minilm_tokenizer, special_tokens=False)(['soybean farming']) == [4]
    # The tokenizer itself raises on an empty batch
    assert token_counter(minilm_tokenizer)([]) == []

@pytest.mark.parametrize(
    ('channel', 'text', 'expected'),
    [
        pytest.param(
            'description',
            'Farms in the U.S. grow corn. Growers, i.e. farmers, sell it. Fruit, e.g. apples, is '
            'excluded.',
            [
                'Farms in the U.S. grow corn.',
                'Growers, i.e. farmers, sell it.',
                'Fruit, e.g. apples, is excluded.',
            ],
            id='abbreviations',
        ),
        pytest.param(
            'description',
            'Mills grade wheat No. 2 and corn, etc. for feed. Bakers buy flour vs. meal.',
            ['Mills grade wheat No. 2 and corn, etc. for feed.', 'Bakers buy flour vs. meal.'],
            id='no-etc-vs',
        ),
        pytest.param(
            'description',
            'Farms do: 1. growing crops. 2. raising animals. Ranches are included.',
            ['Farms do: 1. growing crops.', '2. raising animals.', 'Ranches are included.'],
            id='numbered-list',
        ),
        pytest.param(
            'description',
            'Farms that grow onions are classified in Industry 111113. Others are not.',
            ['Farms that grow onions are classified in Industry 111113.', 'Others are not.'],
            id='a-code-closes-a-unit',
        ),
        pytest.param(
            'excluded',
            'Growing crops (1); raising animals (2); fishing--are classified in Industry 114111.',
            [
                'Growing crops (1); raising animals (2); fishing--are classified in Industry '
                '114111.'
            ],
            id='parenthesized-numerals',
        ),
        pytest.param(
            'excluded',
            'Growing soybeans--are classified in Industry 111110, Soybean Farming; Growing '
            'wheat--are classified in Industry 111140. Growing rice (except wild rice; see '
            '111199)--are classified in Industry 111160, Rice Farming.',
            [
                'Growing soybeans--are classified in Industry 111110, Soybean Farming;',
                'Growing wheat--are classified in Industry 111140.',
                'Growing rice (except wild rice; see 111199)--are classified in Industry 111160, '
                'Rice Farming.',
            ],
            id='cross-references',
        ),
    ],
)
def test_units_are_sentences_that_close_only_at_a_real_break(channel, text, expected):
    units = text_units(channel, text, words, budget=100)

    assert units == expected
    # Every unit boundary is a boundary of the leakage segmenter (spec 4.2)
    pieces = [piece for unit in units for piece in SENTENCE_BREAK.split(unit)]
    assert pieces == SENTENCE_BREAK.split(text)

def test_a_sentence_over_the_budget_is_re_split_at_its_clauses():
    text = 'Farms grow corn; farms grow wheat; farms grow rice. Ranches raise cattle.'

    assert text_units('description', text, words, budget=6) == [
        'Farms grow corn;',
        'farms grow wheat;',
        'farms grow rice.',
        'Ranches raise cattle.',
    ]

def test_a_clause_over_the_budget_is_re_split_at_its_pieces_without_the_guards():
    # The parentheses keep the sentence one clause; its pieces split inside them
    text = 'Farms grow (corn; wheat; rice) here. Ranches raise cattle.'

    assert text_units('description', text, words, budget=4) == [
        'Farms grow (corn;',
        'wheat;',
        'rice) here.',
        'Ranches raise cattle.',
    ]

def test_a_piece_over_the_budget_is_refused():
    with pytest.raises(ValueError, match='a description piece is over the 3-token budget'):
        text_units('description', 'Farms grow corn and wheat. Ranches raise cattle.', words, 3)

def test_examples_units_are_the_entries():
    assert text_units('examples', 'Corn farming; ; Wheat farming', words, 100) == [
        'Corn farming',
        'Wheat farming',
    ]

def test_an_examples_entry_over_the_budget_is_refused():
    with pytest.raises(ValueError, match='an examples entry is over the 1-token budget'):
        text_units('examples', 'Corn; Wheat farming', words, 1)

def test_a_title_is_one_unit_and_one_over_the_window_is_refused():
    assert text_units('title', 'Soybean Farming', words, 2) == ['Soybean Farming']
    with pytest.raises(ValueError, match='a title over the window cannot be summarized'):
        text_units('title', 'Soybean Farming', words, 1)

def test_a_channel_without_units_is_refused():
    with pytest.raises(ValueError, match="no units are defined for channel 'query'"):
        text_units('query', 'soybeans', words, 100)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_window_summaries.py -q`
Expected: `ERROR collecting tests/unit/test_window_summaries.py`, with
`ImportError: cannot import name 'SENTENCE_BREAK' from 'naics_embedder.panels.leakage'`.

- [x] **Step 3: Make the segmenter's break public**

In `src/naics_embedder/panels/leakage.py`, replace:

```python
TEXT_COLUMNS = ('title', 'description', 'examples', 'excluded')

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_SENTENCE_BREAK = re.compile(r'(?<=[.;])\s+')
_CHUNK_ROWS = 256

# -------------------------------------------------------------------------------------------------
```

with:

```python
TEXT_COLUMNS = ('title', 'description', 'examples', 'excluded')

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
# A description or exclusion text splits after '.' or ';' followed by whitespace. Window-fitting
# summaries (panels/window_summaries.py) build their units from the same pieces.
SENTENCE_BREAK = re.compile(r'(?<=[.;])\s+')
_CHUNK_ROWS = 256

# -------------------------------------------------------------------------------------------------
```

Replace:

```python
    if not text:
        return []
    pieces: List[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        pieces.append(sentence)
        if ACTIVITY_SEPARATOR in sentence:
            pieces.append(sentence.split(ACTIVITY_SEPARATOR, 1)[0])
```

with:

```python
    if not text:
        return []
    pieces: List[str] = []
    for sentence in SENTENCE_BREAK.split(text):
        pieces.append(sentence)
        if ACTIVITY_SEPARATOR in sentence:
            pieces.append(sentence.split(ACTIVITY_SEPARATOR, 1)[0])
```

Run: `git grep -n _SENTENCE_BREAK -- src tests`
Expected: no output.

- [x] **Step 4: Add the counter, the budget and the units**

In `src/naics_embedder/panels/window_summaries.py`, replace:

```python
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Optional

# -------------------------------------------------------------------------------------------------
# The pin
```

with:

```python
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from naics_embedder.panels.leakage import EXAMPLES_SEPARATOR, SENTENCE_BREAK
from naics_embedder.text_model.fields import marker

# -------------------------------------------------------------------------------------------------
# The pin
```

Replace:

```python
    pin = WINDOW_SUMMARIES.get(backbone)
    return None if pin is None else pin.sha256
```

with:

```python
    pin = WINDOW_SUMMARIES.get(backbone)
    return None if pin is None else pin.sha256

# -------------------------------------------------------------------------------------------------
# Token counts
# -------------------------------------------------------------------------------------------------

TokenCounter = Callable[[Sequence[str]], List[int]]

def token_counter(tokenizer: Any, *, special_tokens: bool = True) -> TokenCounter:
    '''
    Token counts under ``tokenizer``, without truncation.

    The torch-free twin of ``utils.input_window.token_counter``, which this module cannot import:
    ``naics_embedder.utils`` imports torch when the package loads.

    Args:
        tokenizer: A Hugging Face tokenizer.
        special_tokens: Count ``[CLS]`` and ``[SEP]``, as the model reads a text.

    Returns:
        A function from texts to their token counts.
    '''

    def count(texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        encoded = tokenizer(list(texts), add_special_tokens=special_tokens, truncation=False)
        return [len(ids) for ids in encoded['input_ids']]

    return count

# -------------------------------------------------------------------------------------------------
# Units
# -------------------------------------------------------------------------------------------------

SUMMARY_CHANNELS = ('description', 'examples', 'excluded')
UNIT_RULE = 'sentence-clause-piece-v1'
# At levels 1 and 2 a piece that ends in an abbreviation or a list numeral closes no unit (spec 4.2)
NO_BREAK_PATTERN = r'(?:\bU\.S|\bi\.e|\be\.g|\betc|\bNo|\bvs|(?:^|\s)\d{1,2}|\(\d{1,2}\))[.;]$'
_NO_BREAK = re.compile(NO_BREAK_PATTERN)

Span = Tuple[int, int]

def summary_budget(count_marked: TokenCounter, channel: str, window: int) -> int:
    '''
    The tokens a channel's text may take: the window less its marker and special tokens.

    Args:
        count_marked: Token counts, special tokens included.
        channel: A channel name.
        window: The trained window.

    Returns:
        The budget; 124 for every channel under MiniLM at 128.
    '''

    return window - count_marked([marker(channel)])[0]

def _pieces(text: str, start: int, end: int) -> List[Span]:
    '''The segmenter's pieces of ``text[start:end]``, as spans of ``text``; empty ones dropped.'''

    spans: List[Span] = []
    for match in SENTENCE_BREAK.finditer(text, start, end):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, end))
    return [(left, right) for left, right in spans if right > left]

def _closes_sentence(channel: str) -> Callable[[str], bool]:
    # An exclusion text's cross-references end in ';', so each is one unit
    endings = ('.', ';') if channel == 'excluded' else ('.', )
    return lambda piece: piece.endswith(endings)

def _closes_clause(piece: str) -> bool:
    return piece.endswith(('.', ';'))

def _merge(text: str, pieces: List[Span], closes: Callable[[str], bool]) -> List[Span]:
    '''
    Consecutive pieces merged into units: a piece closes its unit when ``closes`` accepts it, the
    unit's parentheses balance and the piece does not end in an abbreviation or list numeral.
    '''

    units: List[Span] = []
    first: Optional[int] = None
    for index, (start, end) in enumerate(pieces):
        first = start if first is None else first
        piece, unit = text[start:end], text[first:end]
        balanced = unit.count('(') == unit.count(')')
        if index == len(pieces) - 1 or (closes(piece) and balanced and not _NO_BREAK.search(piece)):
            units.append((first, end))
            first = None
    return units

def text_units(channel: str, text: str, count: TokenCounter, budget: int) -> List[str]:
    '''
    A channel text's units, in source order, each a verbatim span of ``text`` (spec 4.2).

    Every unit boundary is a boundary of the leakage segmenter. A title is one unit; the examples
    channel's units are its entries. A description or exclusion text's units are its sentences
    (level 1); a sentence over the budget is re-split at its clauses (level 2), and a clause over
    the budget at the segmenter's pieces (level 3).

    Args:
        channel: ``title``, ``description``, ``examples`` or ``excluded``.
        text: The channel text.
        count: Token counts without special tokens.
        budget: The channel's budget (``summary_budget``).

    Returns:
        The units.

    Raises:
        ValueError: If a title, an examples entry or a level-3 piece is over the budget.
    '''

    if channel == 'title':
        if count([text])[0] > budget:
            raise ValueError(f'a title over the window cannot be summarized: {text!r}')
        return [text]
    if channel == 'examples':
        entries = [entry for entry in text.split(EXAMPLES_SEPARATOR) if entry.strip()]
        over = [entry for entry, tokens in zip(entries, count(entries)) if tokens > budget]
        if over:
            raise ValueError(f'an examples entry is over the {budget}-token budget: {over[0]!r}')
        return entries
    if channel not in ('description', 'excluded'):
        raise ValueError(f'no units are defined for channel {channel!r}')

    def fits(span: Span) -> bool:
        return count([text[span[0]:span[1]]])[0] <= budget

    units: List[Span] = []
    for sentence in _merge(text, _pieces(text, 0, len(text)), _closes_sentence(channel)):
        if fits(sentence):
            units.append(sentence)
            continue
        for clause in _merge(text, _pieces(text, *sentence), _closes_clause):
            if fits(clause):
                units.append(clause)
                continue
            # Level 3: the segmenter's pieces, with no balance or no-break guard
            for piece in _pieces(text, *clause):
                if not fits(piece):
                    raise ValueError(
                        f'a {channel} piece is over the {budget}-token budget: '
                        f'{text[piece[0]:piece[1]]!r}'
                    )
                units.append(piece)
    return [text[start:end] for start, end in units]
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_window_summaries.py tests/unit/test_outcome_leakage.py -q`
Expected: `32 passed`. `test_the_module_imports_no_torch` still passes: the module now imports
`panels.leakage` and `text_model.fields`, and neither loads torch.

- [x] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/panels/leakage.py src/naics_embedder/panels/window_summaries.py tests/fixtures/window_summaries.py tests/unit/test_window_summaries.py`
Run: `uv run pytest -n auto -q`
Expected: `1866 passed, 1 skipped`.

```bash
git add src/naics_embedder/panels/leakage.py src/naics_embedder/panels/window_summaries.py tests/fixtures/window_summaries.py tests/unit/test_window_summaries.py
git commit -m "feat(window-summaries): cut channel texts into units at segmenter boundaries"
```

### Task 3: The artifact and the resolver

The artifact is a UTF-8 CSV read with an explicit schema (4.5, P6). `resolve_channel_texts` is the
only place summaries enter (4.7). It works in seven steps and raises ValueError at the first that
fails, naming the code and channel where there is one:

1. Find the over-window texts, and return the descriptions unchanged when there are none.
2. Require a pin, at the window texts are tokenized at.
3. Read the artifact, and require the pin's sha256.
4. Require exactly one row per over-window text.
5. Require each row's source sha256 and window.
6. Require each summary to be an extract of its text (S5).
7. After substitution, require every marked channel text to fit.

Its tests use a stub tokenizer that counts words, so no model is involved (P7).

**Files:**
- Modify: `src/naics_embedder/panels/window_summaries.py:1-21, 202-204`
- Modify: `tests/fixtures/window_summaries.py:1-11`
- Modify: `tests/unit/test_window_summaries.py:3-24, 197-199`

**Interfaces:**
- Consumes: Task 2's `TokenCounter`, `token_counter` and `SUMMARY_CHANNELS`; `CHANNELS` and
  `marked_text` (`text_model/fields.py`); `normalize_text` and `text_segments`
  (`panels/leakage.py`).
- Produces, in `naics_embedder.panels.window_summaries`:
  - `SUMMARIES_SCHEMA`, the nine columns of 4.5: `code`, `channel`, `source_sha256` and
    `summary` as `pl.Utf8`; `window`, `source_tokens`, `summary_tokens`, `units_kept` and
    `units_total` as `pl.Int64`;
  - `text_sha256(text: str) -> str`, the sha256 of the text's UTF-8 bytes;
  - `write_window_summaries(rows: pl.DataFrame, path: Path) -> str`, which returns the sha256 of
    the bytes written;
  - `read_window_summaries(path: Path) -> pl.DataFrame`;
  - `DEFAULT`, the sentinel for the backbone's own pin;
  - `over_window(descriptions: pl.DataFrame, count_marked: TokenCounter, window: int) ->
    List[Tuple[str, str]]`, the `(code, channel)` pairs sorted by channel, then code;
  - `resolve_channel_texts(descriptions: pl.DataFrame, tokenizer, backbone: str, max_length: int,
    *, pin=DEFAULT) -> pl.DataFrame`.
- Produces, in `tests/fixtures/window_summaries.py`:
  - `WordTokenizer`, a callable stub: one token per word, plus two special tokens when asked;
  - `pin_artifact(path: Path, rows: List[Dict[str, Any]], *, window: int) -> SummariesPin`.

- [x] **Step 1: Write the failing tests**

In `tests/fixtures/window_summaries.py`, replace:

```python
'''
Stubs for the window-fitting summaries tests (roadmap Stage 6b): a token counter that counts
words.
'''

from typing import List, Sequence

def words(texts: Sequence[str]) -> List[int]:
    '''A stub token counter: one token per whitespace-separated word.'''

    return [len(text.split()) for text in texts]
```

with:

```python
'''
Stubs for the window-fitting summaries tests (roadmap Stage 6b): a tokenizer that counts words,
and summaries rows written as a pinned artifact.
'''

from pathlib import Path
from typing import Any, Dict, List, Sequence

import polars as pl

from naics_embedder.panels.window_summaries import (
    SUMMARIES_SCHEMA,
    SummariesPin,
    write_window_summaries,
)

class WordTokenizer:
    '''A stub tokenizer: one token per whitespace-separated word, plus [CLS] and [SEP].'''

    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> Dict[str, List[List[int]]]:
        extra = 2 if add_special_tokens else 0
        return {'input_ids': [[0] * (len(text.split()) + extra) for text in texts]}

def words(texts: Sequence[str]) -> List[int]:
    '''A stub token counter: one token per whitespace-separated word.'''

    return [len(text.split()) for text in texts]

def pin_artifact(path: Path, rows: List[Dict[str, Any]], *, window: int) -> SummariesPin:
    '''Write summaries rows as an artifact at ``path``, and pin it.'''

    sha256 = write_window_summaries(pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), path)
    return SummariesPin(path=str(path), sha256=sha256, window=window)
```

In `tests/unit/test_window_summaries.py`, replace:

```python
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.leakage import SENTENCE_BREAK
from naics_embedder.panels.window_summaries import (
    SummariesPin,
    summaries_identity,
    summary_budget,
    text_units,
    token_counter,
)
from tests.fixtures.window_summaries import words

pytestmark = pytest.mark.unit
```

with:

```python
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import logging
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.leakage import SENTENCE_BREAK
from naics_embedder.panels.window_summaries import (
    SUMMARIES_SCHEMA,
    SummariesPin,
    read_window_summaries,
    resolve_channel_texts,
    summaries_identity,
    summary_budget,
    text_sha256,
    text_units,
    token_counter,
    write_window_summaries,
)
from tests.fixtures.window_summaries import WordTokenizer, pin_artifact, words

pytestmark = pytest.mark.unit
```

Replace:

```python
def test_a_channel_without_units_is_refused():
    with pytest.raises(ValueError, match="no units are defined for channel 'query'"):
        text_units('query', 'soybeans', words, 100)
```

with:

```python
def test_a_channel_without_units_is_refused():
    with pytest.raises(ValueError, match="no units are defined for channel 'query'"):
        text_units('query', 'soybeans', words, 100)

# -------------------------------------------------------------------------------------------------
# The artifact and the resolver
# -------------------------------------------------------------------------------------------------

WINDOW = 10
STUB = 'stub-backbone'
# 'description: ' and three three-word sentences: 1 + 9 words and [CLS] [SEP], 12 > 10 tokens
LONG = 'Farms grow corn. Farms grow wheat. Farms sell grain.'
SUMMARY = 'Farms grow corn. Farms sell grain.'
FITS = 'Farms grow oilseeds.'

def descriptions_frame(description=LONG, examples='Soybeans; Beans'):
    return pl.DataFrame(
        {
            'code': ['111110', '111120'],
            'title': ['Soybean Farming', 'Oilseed Farming'],
            'description': [description, FITS],
            'examples': [examples, None],
            'excluded': [None, '  '],
        }
    )

def summary_row(code='111110', channel='description', summary=SUMMARY, source=LONG, **overrides):
    row = {
        'code': code,
        'channel': channel,
        'source_sha256': text_sha256(source),
        'window': WINDOW,
        'summary': summary,
        'source_tokens': 12,
        'summary_tokens': 9,
        'units_kept': 2,
        'units_total': 3,
    }
    row.update(overrides)
    return row

def pin_rows(tmp_path, rows, window=WINDOW):
    return pin_artifact(tmp_path / 'window_summaries.csv', rows, window=window)

def resolve(descriptions, pin):
    return resolve_channel_texts(descriptions, WordTokenizer(), STUB, WINDOW, pin=pin)

def test_the_artifact_round_trips_sorted_by_channel_then_code(tmp_path):
    rows = [
        summary_row(code='222220', channel='excluded'),
        summary_row(code='111120'),
        summary_row(code='111110', channel='excluded'),
    ]
    path = tmp_path / 'window_summaries.csv'

    sha256 = write_window_summaries(pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), path)
    table = read_window_summaries(path)

    assert sha256 == text_sha256(path.read_text(encoding='utf-8'))
    assert table.schema == pl.Schema(SUMMARIES_SCHEMA)
    assert table.select('channel', 'code').rows() == [
        ('description', '111120'),
        ('excluded', '111110'),
        ('excluded', '222220'),
    ]
    # Rewriting what was read reproduces the bytes
    assert write_window_summaries(table, tmp_path / 'again.csv') == sha256

@pytest.mark.parametrize(
    ('rows', 'refusal'),
    [
        pytest.param(
            [summary_row(channel='title')],
            "a channel outside \\('description', 'examples', 'excluded'\\): title",
            id='channel',
        ),
        pytest.param(
            [summary_row(), summary_row(summary='Farms grow corn.')],
            "summarizes code 111110's description more than once",
            id='repeated',
        ),
    ],
)
def test_the_reader_refuses_a_malformed_artifact(tmp_path, rows, refusal):
    path = tmp_path / 'window_summaries.csv'
    pl.DataFrame(rows, schema=SUMMARIES_SCHEMA).write_csv(path)

    with pytest.raises(ValueError, match=refusal):
        read_window_summaries(path)

def test_an_over_window_text_is_replaced_by_its_summary(tmp_path, caplog):
    descriptions = descriptions_frame()

    with caplog.at_level(logging.INFO, logger='naics_embedder.panels.window_summaries'):
        resolved = resolve(descriptions, pin_rows(tmp_path, [summary_row()]))

    assert resolved.get_column('description').to_list() == [SUMMARY, FITS]
    assert resolved.drop('description').equals(descriptions.drop('description'))
    assert "{'description': 1, 'examples': 0, 'excluded': 0}" in caplog.text

def test_an_examples_text_is_replaced_by_its_entries(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    row = summary_row(channel='examples', summary='Soybeans; Corn', source=examples)

    descriptions = descriptions_frame(description='Farms grow corn.', examples=examples)

    resolved = resolve(descriptions, pin_rows(tmp_path, [row]))

    assert resolved.get_column('examples').to_list() == ['Soybeans; Corn', None]

def test_texts_that_fit_pass_through_without_a_pin_or_an_artifact():
    descriptions = descriptions_frame(description='Farms grow corn.')
    unreadable = SummariesPin(path='/nonexistent/window_summaries.csv', sha256='f' * 64, window=99)

    assert resolve(descriptions, None).equals(descriptions)
    assert resolve(descriptions, unreadable).equals(descriptions)
    # The seam's MiniLM pin names no file either
    assert resolve_channel_texts(descriptions, WordTokenizer(), MINILM, WINDOW).equals(descriptions)

def test_the_default_pin_is_the_backbones_entry_at_call_time(tmp_path, monkeypatch):
    pin = pin_rows(tmp_path, [summary_row()])
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, STUB, pin)

    resolved = resolve_channel_texts(descriptions_frame(), WordTokenizer(), STUB, WINDOW)

    assert resolved.get_column('description').to_list()[0] == SUMMARY

def test_an_over_window_text_without_a_pin_is_refused():
    with pytest.raises(ValueError, match="code 111110's description is over the 10-token window"):
        resolve(descriptions_frame(), None)

def test_a_pin_for_another_window_is_refused(tmp_path):
    pin = pin_rows(tmp_path, [summary_row()], window=12)

    with pytest.raises(ValueError, match='fit a 12-token window, but texts are tokenized at 10'):
        resolve(descriptions_frame(), pin)

def test_a_missing_artifact_is_refused_by_its_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin = SummariesPin(path='conf/window_summaries.csv', sha256='f' * 64, window=WINDOW)

    with pytest.raises(ValueError, match='are missing') as refusal:
        resolve(descriptions_frame(), pin)

    assert str((tmp_path / 'conf/window_summaries.csv').resolve()) in str(refusal.value)

def test_an_artifact_with_another_sha256_is_refused(tmp_path):
    pin = pin_rows(tmp_path, [summary_row()])
    Path(pin.path).write_text(Path(pin.path).read_text() + '\n')

    with pytest.raises(ValueError, match='but the pin names'):
        resolve(descriptions_frame(), pin)

@pytest.mark.parametrize(
    ('rows', 'refusal'),
    [
        pytest.param([], "code 111110's description is over the window", id='missing'),
        pytest.param(
            [summary_row(), summary_row(code='111120', source=FITS)],
            "code 111120's description fits the window",
            id='fits',
        ),
        pytest.param(
            [summary_row(), summary_row(code='999999')],
            "code 999999's description is not in the descriptions",
            id='unknown-code',
        ),
        pytest.param(
            [summary_row(), summary_row(channel='excluded')],
            "code 111110's excluded is not in the descriptions",
            id='absent-text',
        ),
    ],
)
def test_the_rows_must_be_exactly_the_over_window_texts(tmp_path, rows, refusal):
    with pytest.raises(ValueError, match=refusal):
        resolve(descriptions_frame(), pin_rows(tmp_path, rows))

@pytest.mark.parametrize(
    ('row', 'refusal'),
    [
        pytest.param(
            summary_row(source='Farms grow corn.'),
            "code 111110's description: the summary was built from another source text",
            id='source',
        ),
        pytest.param(
            summary_row(window=12),
            "code 111110's description: the summary fits a 12-token window, not 10",
            id='window',
        ),
    ],
)
def test_each_row_must_match_its_source_and_window(tmp_path, row, refusal):
    with pytest.raises(ValueError, match=refusal):
        resolve(descriptions_frame(), pin_rows(tmp_path, [row]))

def test_every_row_is_checked_against_its_source_before_any_is_checked_as_an_extract(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    rows = [
        # Reordered, so not an extract (step 6), and sorted first
        summary_row(summary='Farms sell grain. Farms grow corn.'),
        # Built from another text (step 5)
        summary_row(channel='examples', summary='Soybeans; Corn', source='Soybeans; Corn'),
    ]

    with pytest.raises(ValueError, match="code 111110's examples: the summary was built from"):
        resolve(descriptions_frame(examples=examples), pin_rows(tmp_path, rows))

@pytest.mark.parametrize(
    'summary',
    [
        pytest.param('Farms sell grain. Farms grow corn.', id='reordered'),
        pytest.param('Farms grow corn. Farms grow corn.', id='repeated'),
        pytest.param('Farms grow maize. Farms sell grain.', id='edited'),
    ],
)
def test_a_summary_that_is_not_an_extract_is_refused(tmp_path, summary):
    pin = pin_rows(tmp_path, [summary_row(summary=summary)])

    with pytest.raises(ValueError, match="code 111110's description: the summary is not an"):
        resolve(descriptions_frame(), pin)

def test_reordered_examples_entries_are_refused(tmp_path):
    examples = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
    row = summary_row(channel='examples', summary='Corn; Soybeans', source=examples)

    descriptions = descriptions_frame(description='Farms grow corn.', examples=examples)

    with pytest.raises(ValueError, match="code 111110's examples: the summary is not an extract"):
        resolve(descriptions, pin_rows(tmp_path, [row]))

def test_a_summary_over_the_window_is_refused(tmp_path):
    # The whole text is an extract of itself, and still over the window
    with pytest.raises(ValueError, match="code 111110's description does not fit the 10-token"):
        resolve(descriptions_frame(), pin_rows(tmp_path, [summary_row(summary=LONG)]))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_window_summaries.py -q`
Expected: `ERROR collecting tests/unit/test_window_summaries.py`, with
`ImportError: cannot import name 'SUMMARIES_SCHEMA' from 'naics_embedder.panels.window_summaries'`.

- [x] **Step 3: Add the artifact and the resolver**

In `src/naics_embedder/panels/window_summaries.py`, replace:

```python
'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

``WINDOW_SUMMARIES`` pins, per backbone, the committed artifact of extractive summaries, and
``summaries_identity`` is the sha256 every identity site records: the token cache's sidecar, the
checkpoint contract, the export and text-only provenances, and the decision store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from naics_embedder.panels.leakage import EXAMPLES_SEPARATOR, SENTENCE_BREAK
from naics_embedder.text_model.fields import marker

# -------------------------------------------------------------------------------------------------
# The pin
```

with:

```python
'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

A channel text whose marked form (``'<field>: <text>'``, special tokens included) is over the
backbone's trained window is read as its summary: whole units of the text, in source order, chosen
once by ``naics-embedder data summaries`` (``data/window_summaries.py``) and committed. Every unit
boundary is a boundary of the leakage segmenter (``panels/leakage.py``), so a summary's segments
are a subset of its text's, and leakage needs no sealed read (spec 4.4).

``WINDOW_SUMMARIES`` pins each backbone's artifact by sha256, and ``summaries_identity`` is the
sha256 every identity site records: the token cache's sidecar, the checkpoint contract, the export
and text-only provenances, and the decision store. ``resolve_channel_texts`` is the only place
summaries enter. The token cache and the text-only builder call it, and it checks the artifact's
invariants against the descriptions on every call. The module loads no model and imports no torch.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import polars as pl

from naics_embedder.panels.leakage import (
    EXAMPLES_SEPARATOR,
    SENTENCE_BREAK,
    normalize_text,
    text_segments,
)
from naics_embedder.text_model.fields import CHANNELS, marked_text, marker

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# The pin
```

Replace:

```python
                    )
                units.append(piece)
    return [text[start:end] for start, end in units]
```

with:

```python
                    )
                units.append(piece)
    return [text[start:end] for start, end in units]

# -------------------------------------------------------------------------------------------------
# The artifact
# -------------------------------------------------------------------------------------------------

SUMMARIES_SCHEMA = {
    'code': pl.Utf8,
    'channel': pl.Utf8,
    'source_sha256': pl.Utf8,
    'window': pl.Int64,
    'summary': pl.Utf8,
    'source_tokens': pl.Int64,
    'summary_tokens': pl.Int64,
    'units_kept': pl.Int64,
    'units_total': pl.Int64,
}

def text_sha256(text: str) -> str:
    '''The sha256 of a text's UTF-8 bytes.'''

    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def write_window_summaries(rows: pl.DataFrame, path: Path) -> str:
    '''
    Write summaries rows as the artifact: UTF-8 CSV, sorted by channel, then code.

    Args:
        rows: Rows with the columns of ``SUMMARIES_SCHEMA``.
        path: The CSV to write.

    Returns:
        The sha256 of the bytes written.
    '''

    # yapf: disable
    data = (
        rows
        .select([pl.col(name).cast(dtype) for name, dtype in SUMMARIES_SCHEMA.items()])
        .sort('channel', 'code')
        .write_csv()
        .encode('utf-8')
    )
    # yapf: enable
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def _parse_window_summaries(data: bytes, path: Path) -> pl.DataFrame:
    rows = pl.read_csv(io.BytesIO(data), schema=SUMMARIES_SCHEMA)
    channels = sorted(set(rows.get_column('channel').to_list()) - set(SUMMARY_CHANNELS))
    if channels:
        raise ValueError(f'{path} names a channel outside {SUMMARY_CHANNELS}: {channels[0]}')
    pairs = Counter(rows.select('code', 'channel').iter_rows())
    repeated = sorted(pair for pair, count in pairs.items() if count > 1)
    if repeated:
        code, channel = repeated[0]
        raise ValueError(f"{path} summarizes code {code}'s {channel} more than once")
    return rows

def read_window_summaries(path: Path) -> pl.DataFrame:
    '''
    Read a summaries artifact with ``SUMMARIES_SCHEMA``, so that ``code`` stays a string.

    Raises:
        ValueError: If a row names a channel other than the three, or a ``(code, channel)``
            repeats.
    '''

    return _parse_window_summaries(Path(path).read_bytes(), Path(path))

# -------------------------------------------------------------------------------------------------
# The resolver
# -------------------------------------------------------------------------------------------------

class _PinDefault(Enum):
    DEFAULT = 'default'

# The resolver's default pin: the backbone's entry in WINDOW_SUMMARIES, looked up at call time
DEFAULT = _PinDefault.DEFAULT

def over_window(
    descriptions: pl.DataFrame,
    count_marked: TokenCounter,
    window: int,
) -> List[Tuple[str, str]]:
    '''
    The present channel texts whose marked form is over the window, sorted by channel, then code.

    Args:
        descriptions: Codes and their four channel texts; a null or blank text is absent.
        count_marked: Token counts, special tokens included.
        window: The window.

    Returns:
        ``(code, channel)`` pairs.
    '''

    codes = descriptions.get_column('code').to_list()
    over: List[Tuple[str, str]] = []
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        present = [row for row, text in enumerate(texts) if text is not None and text.strip()]
        counts = count_marked([marked_text(channel, texts[row]) for row in present])
        over.extend(
            (codes[row], channel) for row, tokens in zip(present, counts) if tokens > window
        )
    return sorted(over, key=lambda pair: (pair[1], pair[0]))

def _raw_pieces(channel: str, text: str) -> List[str]:
    if channel == 'examples':
        return text.split(EXAMPLES_SEPARATOR)
    return SENTENCE_BREAK.split(text)

def _segments(channel: str, text: str) -> Set[str]:
    '''The text's segments, as ``training_text_segments`` cuts them.'''

    if channel == 'examples':
        return {
            segment
            for segment in map(normalize_text, text.split(EXAMPLES_SEPARATOR)) if segment
        }
    return set(text_segments(text))

def _is_extract(channel: str, summary: str, source: str) -> bool:
    '''
    S5: the summary's raw pieces are an in-order subsequence of the source's, each source piece
    used at most once, and its segments are a subset of the source's (spec 4.4).
    '''

    source_pieces = _raw_pieces(channel, source)
    position = 0
    for piece in _raw_pieces(channel, summary):
        while position < len(source_pieces) and source_pieces[position] != piece:
            position += 1
        if position == len(source_pieces):
            return False
        position += 1
    return _segments(channel, summary) <= _segments(channel, source)

def resolve_channel_texts(
    descriptions: pl.DataFrame,
    tokenizer: Any,
    backbone: str,
    max_length: int,
    *,
    pin: Union[SummariesPin, None, _PinDefault] = DEFAULT,
) -> pl.DataFrame:
    '''
    The descriptions, each over-window channel text replaced by its pinned summary (spec 4.7).

    Steps, raising at the first failure: (1) find the texts over the window, and return the
    descriptions unchanged when there are none; (2) require a pin for the window; (3) read the
    artifact and require the pin's sha256; (4) require one row per over-window text and no other;
    (5) require each row's source sha256 and window; (6) require each summary to be an extract of
    its source; (7) require every marked channel text to fit after substitution.

    Args:
        descriptions: Codes and their four channel texts.
        tokenizer: The backbone's tokenizer, which counts tokens as the backbone reads them.
        backbone: The backbone, whose pin is the default.
        max_length: The window texts are tokenized at.
        pin: The summaries to read: ``DEFAULT`` looks up ``WINDOW_SUMMARIES[backbone]`` at call
            time, and None means no pin.

    Returns:
        The descriptions with over-window texts replaced.

    Raises:
        ValueError: At the first step that fails, naming the code and channel where there is one.
    '''

    count_marked = token_counter(tokenizer)
    over = over_window(descriptions, count_marked, max_length)
    if not over:
        return descriptions
    if pin is DEFAULT:
        pin = WINDOW_SUMMARIES.get(backbone)
    if pin is None:
        code, channel = over[0]
        raise ValueError(
            f"code {code}'s {channel} is over the {max_length}-token window, and no window "
            f'summaries are pinned for {backbone} ({len(over)} texts are over)'
        )
    if pin.window != max_length:
        raise ValueError(
            f'the window summaries pinned for {backbone} fit a {pin.window}-token window, but '
            f'texts are tokenized at {max_length}'
        )

    path = Path(pin.path)
    if not path.is_file():
        raise ValueError(
            f'the window summaries pinned for {backbone} are missing: {path.resolve()}'
        )
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != pin.sha256:
        raise ValueError(f'{path} has sha256 {sha256}, but the pin names {pin.sha256}')
    rows = _parse_window_summaries(data, path)

    texts = {
        (code, channel): text
        for channel in CHANNELS
        for code, text in descriptions.select('code', channel).iter_rows()
        if text is not None and text.strip()
    }
    named = set(rows.select('code', 'channel').iter_rows())
    for code, channel in over:
        if (code, channel) not in named:
            raise ValueError(f"code {code}'s {channel} is over the window but has no summary")
    for code, channel in sorted(named - set(over)):
        if (code, channel) not in texts:
            raise ValueError(f"code {code}'s {channel} is not in the descriptions")
        raise ValueError(f"code {code}'s {channel} fits the window and needs no summary")

    summaries = {(row['code'], row['channel']): row for row in rows.iter_rows(named=True)}
    for (code, channel), row in summaries.items():
        if row['source_sha256'] != text_sha256(texts[(code, channel)]):
            raise ValueError(
                f"code {code}'s {channel}: the summary was built from another source text"
            )
        if row['window'] != max_length:
            raise ValueError(
                f"code {code}'s {channel}: the summary fits a {row['window']}-token window, not "
                f'{max_length}'
            )
    for (code, channel), row in summaries.items():
        if not _is_extract(channel, row['summary'], texts[(code, channel)]):
            raise ValueError(f"code {code}'s {channel}: the summary is not an extract of its text")

    resolved = descriptions.with_columns(
        [
            pl.Series(
                channel,
                [
                    summaries[(code, channel)]['summary'] if (code, channel) in summaries else text
                    for code, text in descriptions.select('code', channel).iter_rows()
                ],
                dtype=pl.Utf8,
            ) for channel in SUMMARY_CHANNELS
        ]
    )
    still_over = over_window(resolved, count_marked, max_length)
    if still_over:
        code, channel = still_over[0]
        raise ValueError(
            f"code {code}'s {channel} does not fit the {max_length}-token window after "
            'substitution'
        )
    replaced = {
        channel: sum(pair[1] == channel for pair in summaries)
        for channel in SUMMARY_CHANNELS
    }
    logger.info(f'Window summaries for {backbone} replaced channel texts: {replaced}')
    return resolved
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_window_summaries.py -q`
Expected: `42 passed`. `test_the_module_imports_no_torch` still passes: polars loads no torch.

- [x] **Step 5: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/panels/window_summaries.py tests/fixtures/window_summaries.py tests/unit/test_window_summaries.py`
Run: `uv run pytest -n auto -q`
Expected: `1889 passed, 1 skipped`.

```bash
git add src/naics_embedder/panels/window_summaries.py tests/fixtures/window_summaries.py tests/unit/test_window_summaries.py
git commit -m "feat(window-summaries): read the pinned artifact and resolve channel texts"
```

### Task 4: The build and `data summaries`

`data/window_summaries.py` selects each over-window text's units by backbone centrality (4.3) and
writes the artifact and its provenance (4.5, 4.6, P8). It writes the CSV to a temporary file and
runs `resolve_channel_texts` on it under a temporary pin, so a failing invariant leaves no
artifact. `naics-embedder data summaries` drives it and prints the pin to commit (P9). Like
`data roles`, it runs once, and its output is committed; neither `data preprocess` nor `data all`
runs it. Its tests use a stub tokenizer, a stub embedder and a stub backbone name, so they load no
model.

**Files:**
- Modify: `src/naics_embedder/cli/commands/data.py:16-21, 35-41, 58-63, 223-227`
- Create: `src/naics_embedder/data/window_summaries.py`
- Modify: `tests/unit/test_cli_commands.py:14-19, 182-187`
- Create: `tests/unit/test_window_summaries_build.py`

**Interfaces:**
- Consumes:
  - from Tasks 2–3: `NO_BREAK_PATTERN`, `SUMMARIES_SCHEMA`, `SUMMARY_CHANNELS`, `UNIT_RULE`,
    `SummariesPin`, `over_window`, `resolve_channel_texts`, `summary_budget`, `text_sha256`,
    `text_units`, `token_counter`, `write_window_summaries`, `WINDOW_SUMMARIES_PATH`;
  - `load_backbone(backbone) -> (model, tokenizer, revision)` and `provenance_path(path)`
    (`panels/text_only.py`); `trained_window(backbone)` (`utils/input_window.py`);
    `sha256_file` (`supervision/artifacts.py`).
- Produces, in `naics_embedder.data.window_summaries`:
  - `SELECTION_RULE = 'centrality-v1'`, and `UnitEmbedder = Callable[[Sequence[str]],
    np.ndarray]`;
  - `backbone_embedder(model, tokenizer, *, batch_size: int = 64) -> UnitEmbedder`;
  - `Selection(kept: List[int], plateau: bool)`, a frozen dataclass;
  - `select_units(vectors: np.ndarray, weights: np.ndarray, fits) -> Selection`, where `fits`
    maps candidate unit-index lists to booleans;
  - `joiner(channel: str) -> str`;
  - `summary_rows(descriptions, tokenizer, embed, *, window: int) -> Tuple[pl.DataFrame, Dict[str,
    int]]`, the rows and the plateau stops per channel;
  - `generate_window_summaries(descriptions_path, output_path, *, backbone: str, force: bool =
    False, model=None, tokenizer=None, revision: Optional[str] = None) -> SummariesPin`.
- Produces, in `naics_embedder.cli.commands.data`: the `summaries` command, with
  `--descriptions` (default `./data/naics_descriptions.parquet`), `--backbone`, `--output`
  (default `WINDOW_SUMMARIES_PATH`) and `--force`; and `TRAINING_CONFIG = 'conf/config.yaml'`.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_window_summaries_build.py`:

```python
'''
Building the window-fitting summaries (roadmap Stage 6b): the centrality selection (spec 4.3), the
artifact's rows (4.5) and the build (4.6), with a stub tokenizer and a stub embedder.
'''

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from naics_embedder.data import window_summaries as build
from naics_embedder.data.window_summaries import (
    Selection,
    generate_window_summaries,
    select_units,
    summary_rows,
)
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import (
    NO_BREAK_PATTERN,
    SummariesPin,
    read_window_summaries,
    resolve_channel_texts,
    text_sha256,
)
from naics_embedder.utils.input_window import TRAINED_WINDOWS
from tests.fixtures.window_summaries import WordTokenizer

pytestmark = pytest.mark.unit

WINDOW = 10
STUB = 'stub-backbone'
# 'description: ' and three three-word sentences: 12 tokens with [CLS] and [SEP], over 10
LONG = 'Farms grow corn. Farms grow wheat. Farms sell grain.'
# 'examples: ' and eight entries: 11 tokens
EIGHT = 'Soybeans; Beans; Corn; Wheat; Rice; Oats; Rye; Barley'
TEXT_SCHEMA = {name: pl.Utf8 for name in ('code', 'title', 'description', 'examples', 'excluded')}

# Unit vectors of the stub embedder. The description's target is (2, 1) up to scale: wheat is the
# most central unit (sell grain ties it and loses to the earlier unit), then corn raises the
# cosine most. The examples' target is (1, 1): soybeans, then beans, which reaches cosine 1.
VECTORS = {
    'Farms grow corn.': [0.0, 1.0],
    'Farms grow wheat.': [1.0, 0.0],
    'Farms sell grain.': [1.0, 0.0],
    'Soybeans': [1.0, 0.0],
    'Beans': [0.0, 1.0],
    'Corn': [1.0, 0.0],
    'Wheat': [0.0, 1.0],
    'Rice': [1.0, 0.0],
    'Oats': [0.0, 1.0],
    'Rye': [1.0, 0.0],
    'Barley': [0.0, 1.0],
}

def stub_embed(units):
    return np.array([VECTORS[unit] for unit in units])

def descriptions_frame(**columns):
    texts = {
        'code': ['111110', '111120'],
        'title': ['Soybean Farming', 'Oilseed Farming'],
        'description': [LONG, 'Farms grow oilseeds.'],
        'examples': [EIGHT, None],
        'excluded': [None, None],
    }
    texts.update(columns)
    return pl.DataFrame(texts, schema=TEXT_SCHEMA)

# -------------------------------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------------------------------

def fits_at_most(count):
    '''A stub fit: a candidate fits when it has at most ``count`` units.'''

    return lambda candidates: [len(candidate) <= count for candidate in candidates]

def test_the_greedy_adds_the_unit_that_most_raises_the_cosine_while_one_fits():
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    # Target (2, 1): unit 1 first, then unit 0; with three units allowed, unit 2 raises it to 1
    assert select_units(vectors, np.ones(3), fits_at_most(2)) == Selection([0, 1], plateau=False)
    assert select_units(vectors, np.ones(3), fits_at_most(3)) == Selection([0, 1, 2], False)

def test_an_exact_tie_goes_to_the_earlier_unit():
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    assert select_units(vectors, np.ones(3), fits_at_most(1)) == Selection([1], plateau=False)

def test_the_selection_stops_when_no_unit_raises_the_cosine():
    # Target (3, 0): unit 0 alone has cosine 1, and either other unit lowers it
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    weights = np.array([3.0, 1.0, 1.0])

    assert select_units(vectors, weights, fits_at_most(3)) == Selection([0], plateau=True)

def test_a_text_with_no_unit_that_fits_alone_is_refused():
    with pytest.raises(ValueError, match='no unit fits the window on its own'):
        select_units(np.eye(2), np.ones(2), fits_at_most(0))

# -------------------------------------------------------------------------------------------------
# Rows
# -------------------------------------------------------------------------------------------------

def test_a_summary_keeps_whole_units_in_source_order_and_fits_with_its_marker():
    rows, plateau = summary_rows(descriptions_frame(), WordTokenizer(), stub_embed, window=WINDOW)

    assert rows.to_dicts() == [
        {
            'code': '111110',
            'channel': 'description',
            'source_sha256': text_sha256(LONG),
            'window': WINDOW,
            # Picked wheat first, then corn; emitted in source order
            'summary': 'Farms grow corn. Farms grow wheat.',
            'source_tokens': 12,
            'summary_tokens': 9,
            'units_kept': 2,
            'units_total': 3,
        },
        {
            'code': '111110',
            'channel': 'examples',
            'source_sha256': text_sha256(EIGHT),
            'window': WINDOW,
            'summary': 'Soybeans; Beans',
            'source_tokens': 11,
            'summary_tokens': 5,
            'units_kept': 2,
            'units_total': 8,
        },
    ]
    assert plateau == {'description': 0, 'examples': 1, 'excluded': 0}

def test_identical_texts_get_one_summary_and_each_unit_is_embedded_once():
    embedded = []

    def counting_embed(units):
        embedded.append(list(units))
        return stub_embed(units)

    frame = descriptions_frame(description=[LONG, LONG], examples=[None, None])
    rows, _ = summary_rows(frame, WordTokenizer(), counting_embed, window=WINDOW)

    assert rows.select('code', 'summary').rows() == [
        ('111110', 'Farms grow corn. Farms grow wheat.'),
        ('111120', 'Farms grow corn. Farms grow wheat.'),
    ]
    assert embedded == [['Farms grow corn.', 'Farms grow wheat.', 'Farms sell grain.']]

def test_a_title_over_the_window_cannot_be_summarized():
    frame = descriptions_frame(title=['Farms that grow soybeans and other oilseed crops', 'Oil'])

    with pytest.raises(ValueError, match='a title over the window cannot be summarized'):
        summary_rows(frame, WordTokenizer(), stub_embed, window=WINDOW)

# -------------------------------------------------------------------------------------------------
# The build
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def stub_backbone(monkeypatch):
    '''A backbone with a 10-token window whose units the stub embedder reads.'''

    monkeypatch.setitem(TRAINED_WINDOWS, STUB, WINDOW)
    monkeypatch.setattr(build, 'backbone_embedder', lambda model, tokenizer: stub_embed)

@pytest.fixture
def descriptions_path(tmp_path):
    path = tmp_path / 'naics_descriptions.parquet'
    descriptions_frame().write_parquet(path)
    return path

def generate(tmp_path, descriptions_path, **options):
    return generate_window_summaries(
        descriptions_path,
        tmp_path / 'conf' / 'window_summaries.csv',
        backbone=STUB,
        model=object(),
        tokenizer=WordTokenizer(),
        revision='stub-revision',
        **options,
    )

def test_the_build_writes_a_checked_artifact_and_its_provenance(
    tmp_path, stub_backbone, descriptions_path
):
    pin = generate(tmp_path, descriptions_path)

    artifact = tmp_path / 'conf' / 'window_summaries.csv'
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert pin == SummariesPin(path=str(artifact), sha256=sha256, window=WINDOW)
    assert read_window_summaries(artifact).select('channel', 'summary').rows() == [
        ('description', 'Farms grow corn. Farms grow wheat.'),
        ('examples', 'Soybeans; Beans'),
    ]
    resolve_channel_texts(descriptions_frame(), WordTokenizer(), STUB, WINDOW, pin=pin)
    assert sorted(path.name for path in artifact.parent.iterdir()) == [
        'window_summaries.csv',
        'window_summaries_provenance.json',
    ]

    provenance = json.loads(provenance_path(artifact).read_text())
    assert provenance['artifact_sha256'] == sha256
    assert provenance['descriptions'] == {
        'path': str(descriptions_path),
        'sha256': hashlib.sha256(descriptions_path.read_bytes()).hexdigest(),
    }
    assert [provenance[key] for key in ('backbone', 'revision', 'tokenizer', 'window')] == [
        STUB,
        'stub-revision',
        STUB,
        WINDOW,
    ]
    assert provenance['budget'] == {channel: 7 for channel in TEXT_SCHEMA if channel != 'code'}
    assert provenance['units'] == {
        'rule': 'sentence-clause-piece-v1',
        'no_break_pattern': NO_BREAK_PATTERN
    }
    assert provenance['selection']['rule'] == 'centrality-v1'
    assert provenance['channels']['description'] == {
        'present': 2,
        'over_window': 1,
        'summarized': 1,
        'mean_kept_share': 0.75,
        'min_summary_tokens': 9,
        'p10_summary_tokens': 9,
        'plateau_stops': 0,
    }
    assert provenance['channels']['examples']['plateau_stops'] == 1
    assert provenance['channels']['title']['summarized'] == 0

def test_a_backbone_with_no_recorded_window_is_refused(tmp_path, descriptions_path):
    # No stub_backbone: the stub has no entry in TRAINED_WINDOWS
    with pytest.raises(ValueError, match="no trained input window is recorded for 'stub-backbone'"):
        generate(tmp_path, descriptions_path)
    assert not (tmp_path / 'conf').exists()

def test_an_existing_artifact_is_kept_without_force(tmp_path, stub_backbone, descriptions_path):
    first = generate(tmp_path, descriptions_path)

    with pytest.raises(FileExistsError, match='--force'):
        generate(tmp_path, descriptions_path)

    assert generate(tmp_path, descriptions_path, force=True) == first

def test_a_failing_check_leaves_no_artifact(
    tmp_path, stub_backbone, descriptions_path, monkeypatch
):

    def reordered(*args, **kwargs):
        rows, plateau = summary_rows(*args, **kwargs)
        swapped = pl.when(pl.col('channel') == 'description').then(
            pl.lit('Farms grow wheat. Farms grow corn.')
        ).otherwise(pl.col('summary'))
        return rows.with_columns(swapped.alias('summary')), plateau

    monkeypatch.setattr(build, 'summary_rows', reordered)

    with pytest.raises(ValueError, match="code 111110's description: the summary is not an"):
        generate(tmp_path, descriptions_path)

    assert list((tmp_path / 'conf').iterdir()) == []
```

In `tests/unit/test_cli_commands.py`, replace:

```python
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.utils.config import Config
from tests.fixtures.decision import spec, synthetic_arm
from tests.fixtures.regressor_panel import (
```

with:

```python
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.window_summaries import SummariesPin
from naics_embedder.utils.config import Config
from tests.fixtures.decision import spec, synthetic_arm
from tests.fixtures.regressor_panel import (
```

Replace:

```python
    assert result.exit_code == 1
    assert '--force' in result.output

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
    captured = {}
    monkeypatch.setattr(
```

with:

```python
    assert result.exit_code == 1
    assert '--force' in result.output

def test_data_summaries_builds_for_the_training_configs_backbone(monkeypatch, runner):
    calls = []

    def fake_generate(descriptions, output, *, backbone, force):
        calls.append((descriptions, output, backbone, force))
        return SummariesPin(path=str(output), sha256='a' * 64, window=128)

    monkeypatch.setattr(data_cli, 'generate_window_summaries', fake_generate)

    result = runner.invoke(data_cli.app, ['summaries'])

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            Path('./data/naics_descriptions.parquet'),
            Path('conf/data/window_summaries.csv'),
            'sentence-transformers/all-MiniLM-L6-v2',
            False,
        )
    ]
    assert f"sha256='{'a' * 64}'" in result.output.replace('\n', '')

def test_data_summaries_refuses_to_rebuild_without_force(monkeypatch, runner):

    def refuse(descriptions, output, *, backbone, force):
        raise FileExistsError('the summaries exist; pass --force only to rebuild them')

    monkeypatch.setattr(data_cli, 'generate_window_summaries', refuse)

    result = runner.invoke(data_cli.app, ['summaries', '--backbone', 'other/backbone'])

    assert result.exit_code == 1
    assert '--force' in result.output.replace('\n', '')

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
    captured = {}
    monkeypatch.setattr(
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_window_summaries_build.py -q`
Expected: `ERROR collecting tests/unit/test_window_summaries_build.py`, with
`ImportError: cannot import name 'window_summaries' from 'naics_embedder.data'`.

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `2 failed, 40 passed`. Both new tests fail with
`AttributeError: <module 'naics_embedder.cli.commands.data' …> has no attribute
'generate_window_summaries'`.

- [x] **Step 3: Write the build**

Create `src/naics_embedder/data/window_summaries.py`:

```python
'''
Build the window-fitting summaries (roadmap Stage 6b).

``naics-embedder data summaries`` runs this once per backbone. Every channel text whose marked
form is over the backbone's trained window is cut into units (``panels.window_summaries``), and
the summary keeps the units whose pooled vectors best approximate the whole text's (centrality):

1. Of the units whose marked summary fits the window on its own, keep the one whose weighted
   vector has the highest cosine with the target, the weighted sum of all the text's units.
2. Repeatedly add the unit that most raises that cosine, among the units that keep the marked
   summary, joined in source order, within the window.
3. Stop when no remaining unit fits, or none raises the cosine strictly.

The frozen backbone reads each unit unmarked and alone, on the CPU in float32; its last hidden
state is mean-pooled over the attention mask and L2-normalized, and a unit's weight is its token
count without special tokens. Exact ties go to the earlier unit. The kept units are emitted in
source order.

The artifact is written to a temporary file and checked by ``resolve_channel_texts`` under a
temporary pin before it is moved into place, so a failing invariant leaves no artifact. It is
committed and pinned in ``WINDOW_SUMMARIES``; neither ``data preprocess`` nor ``data all`` runs
this.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch

from naics_embedder.panels.leakage import EXAMPLES_SEPARATOR
from naics_embedder.panels.text_only import load_backbone, provenance_path
from naics_embedder.panels.window_summaries import (
    NO_BREAK_PATTERN,
    SUMMARIES_SCHEMA,
    SUMMARY_CHANNELS,
    UNIT_RULE,
    SummariesPin,
    over_window,
    resolve_channel_texts,
    summary_budget,
    text_sha256,
    text_units,
    token_counter,
    write_window_summaries,
)
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.fields import CHANNELS, marked_text
from naics_embedder.utils.input_window import trained_window

logger = logging.getLogger(__name__)

SELECTION_RULE = 'centrality-v1'

# Texts to unit vectors, one row per text
UnitEmbedder = Callable[[Sequence[str]], np.ndarray]

# -------------------------------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------------------------------

def backbone_embedder(model: Any, tokenizer: Any, *, batch_size: int = 64) -> UnitEmbedder:
    '''
    Unit vectors from a frozen backbone: each unit read unmarked and alone, its last hidden state
    mean-pooled over the attention mask.

    Args:
        model: A Hugging Face encoder returning ``last_hidden_state``.
        tokenizer: Its tokenizer.
        batch_size: Units per forward pass.

    Returns:
        A function from units to their vectors (float64).
    '''

    def embed(units: Sequence[str]) -> np.ndarray:
        parts = []
        for start in range(0, len(units), batch_size):
            tokens = tokenizer(
                list(units[start:start + batch_size]),
                padding=True,
                truncation=False,
                return_tensors='pt',
            )
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            parts.append(pooled.cpu().to(torch.float64).numpy())
        return np.concatenate(parts)

    return embed

@dataclass(frozen=True)
class Selection:
    '''
    The units a summary keeps.

    Attributes:
        kept: Indices of the kept units, in source order.
        plateau: True when the selection stopped because no unit that still fit raised the cosine.
    '''

    kept: List[int]
    plateau: bool

def _cosine(total: np.ndarray, target: np.ndarray) -> float:
    return float(total @ target / (np.linalg.norm(total) * np.linalg.norm(target)))

def select_units(
    vectors: np.ndarray,
    weights: np.ndarray,
    fits: Callable[[List[List[int]]], List[bool]],
) -> Selection:
    '''
    Greedy centrality selection over one text's units.

    Args:
        vectors: One row per unit, in source order.
        weights: Each unit's token count, without special tokens.
        fits: For each candidate (unit indices in source order), whether its marked summary fits
            the window.

    Returns:
        The kept units.

    Raises:
        ValueError: If no unit fits the window on its own.
    '''

    weighted = weights[:, None] * (vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    target = weighted.sum(axis=0)
    kept: List[int] = []
    best = -np.inf
    while True:
        remaining = [index for index in range(len(vectors)) if index not in kept]
        candidates = [sorted(kept + [index]) for index in remaining]
        fitting = [index for index, ok in zip(remaining, fits(candidates)) if ok]
        if not fitting:
            if not kept:
                raise ValueError('no unit fits the window on its own')
            return Selection(kept=sorted(kept), plateau=False)
        top, top_score = fitting[0], -np.inf
        for index in fitting:
            score = _cosine(weighted[kept + [index]].sum(axis=0), target)
            # Strictly greater: an exact tie goes to the earlier unit
            if score > top_score:
                top, top_score = index, score
        if kept and not top_score > best:
            return Selection(kept=sorted(kept), plateau=True)
        kept.append(top)
        best = top_score

def joiner(channel: str) -> str:
    '''What a summary's units are joined by: the examples separator, or one space.'''

    return EXAMPLES_SEPARATOR if channel == 'examples' else ' '

def summary_rows(
    descriptions: pl.DataFrame,
    tokenizer: Any,
    embed: UnitEmbedder,
    *,
    window: int,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    '''
    One summary row per over-window channel text.

    Identical source texts get identical summaries, one row per code. Each distinct unit is
    embedded once.

    Args:
        descriptions: Codes and their four channel texts.
        tokenizer: The backbone's tokenizer.
        embed: Unit vectors (``backbone_embedder``).
        window: The trained window.

    Returns:
        The rows, with the columns of ``SUMMARIES_SCHEMA``, and per channel the summaries that
        ended at the plateau stop.

    Raises:
        ValueError: If a title is over the window, or a unit is over its channel's budget.
    '''

    count_marked = token_counter(tokenizer)
    count_plain = token_counter(tokenizer, special_tokens=False)
    over = over_window(descriptions, count_marked, window)
    texts = {
        (code, channel): text
        for channel in CHANNELS
        for code, text in descriptions.select('code', channel).iter_rows()
    }
    units_of = {
        pair: text_units(
            pair[1], texts[pair], count_plain, summary_budget(count_marked, pair[1], window)
        )
        for pair in over
    }
    unique = list(dict.fromkeys(unit for units in units_of.values() for unit in units))
    logger.info(f'Embedding {len(unique):,} distinct units of {len(over):,} over-window texts')
    vectors = dict(zip(unique, embed(unique))) if unique else {}
    weights = dict(zip(unique, count_plain(unique)))

    rows: List[Dict[str, Any]] = []
    plateau = {channel: 0 for channel in SUMMARY_CHANNELS}
    chosen: Dict[Tuple[str, str], Tuple[str, int, bool]] = {}
    for code, channel in over:
        source, units = texts[(code, channel)], units_of[(code, channel)]
        join = joiner(channel)
        if (channel, source) not in chosen:

            def fits(candidates: List[List[int]]) -> List[bool]:
                summaries = [join.join(units[index] for index in kept) for kept in candidates]
                marked = [marked_text(channel, summary) for summary in summaries]
                return [tokens <= window for tokens in count_marked(marked)]

            selection = select_units(
                np.stack([vectors[unit] for unit in units]),
                np.array([weights[unit] for unit in units], dtype=np.float64),
                fits,
            )
            summary = join.join(units[index] for index in selection.kept)
            chosen[(channel, source)] = (summary, len(selection.kept), selection.plateau)
        summary, kept, stopped = chosen[(channel, source)]
        plateau[channel] += int(stopped)
        rows.append(
            {
                'code': code,
                'channel': channel,
                'source_sha256': text_sha256(source),
                'window': window,
                'summary': summary,
                'source_tokens': count_marked([marked_text(channel, source)])[0],
                'summary_tokens': count_marked([marked_text(channel, summary)])[0],
                'units_kept': kept,
                'units_total': len(units),
            }
        )
    return pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), plateau

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_window_summaries(
    descriptions_path: Path,
    output_path: Path,
    *,
    backbone: str,
    force: bool = False,
    model: Optional[Any] = None,
    tokenizer: Any = None,
    revision: Optional[str] = None,
) -> SummariesPin:
    '''
    Summarize every over-window channel text, check the artifact, and write it and its provenance.

    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``, from the local Hugging
    Face cache; tests pass small ones.

    Args:
        descriptions_path: The descriptions parquet.
        output_path: The artifact (CSV); its provenance is written beside it.
        backbone: The backbone, whose trained window the summaries fit.
        force: Overwrite an existing artifact.
        model: The backbone's model.
        tokenizer: The backbone's tokenizer.
        revision: The backbone's snapshot revision.

    Returns:
        The pin to commit in ``WINDOW_SUMMARIES``.

    Raises:
        FileExistsError: If the artifact exists and ``force`` is False.
        ValueError: If a text cannot be summarized, or the artifact fails a resolver check.
    '''

    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f'{output_path} exists: the summaries are built once and committed, and a new '
            'artifact needs a new pin. Pass --force only to rebuild it deliberately.'
        )
    window = trained_window(backbone)
    descriptions_path = Path(descriptions_path)
    descriptions = pl.read_parquet(descriptions_path)
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    rows, plateau = summary_rows(
        descriptions, tokenizer, backbone_embedder(model, tokenizer), window=window
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + '.tmp')
    try:
        sha256 = write_window_summaries(rows, temporary)
        # Every invariant of the resolver, against the bytes about to be committed
        resolve_channel_texts(
            descriptions,
            tokenizer,
            backbone,
            window,
            pin=SummariesPin(path=str(temporary), sha256=sha256, window=window),
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    provenance = _provenance(
        descriptions,
        descriptions_path,
        rows,
        plateau,
        tokenizer,
        backbone=backbone,
        revision=revision,
        window=window,
        sha256=sha256,
    )
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(f'Window summaries ({rows.height:,} texts, sha256 {sha256}): {output_path}')
    return SummariesPin(path=str(output_path), sha256=sha256, window=window)

def _provenance(
    descriptions: pl.DataFrame,
    descriptions_path: Path,
    rows: pl.DataFrame,
    plateau: Dict[str, int],
    tokenizer: Any,
    *,
    backbone: str,
    revision: Optional[str],
    window: int,
    sha256: str,
) -> Dict[str, Any]:
    count_marked = token_counter(tokenizer)
    over = over_window(descriptions, count_marked, window)
    channels: Dict[str, Dict[str, Any]] = {}
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        summarized = rows.filter(pl.col('channel') == channel)
        tokens = summarized.get_column('summary_tokens')
        share = summarized.get_column('summary_tokens') / summarized.get_column('source_tokens')
        channels[channel] = {
            'present': sum(text is not None and bool(text.strip()) for text in texts),
            'over_window': sum(pair[1] == channel for pair in over),
            'summarized': summarized.height,
            'mean_kept_share': share.mean() if summarized.height else None,
            'min_summary_tokens': tokens.min() if summarized.height else None,
            'p10_summary_tokens': (
                int(tokens.quantile(0.1, interpolation='lower')) if summarized.height else None
            ),
            'plateau_stops': plateau.get(channel, 0),
        }
    return {
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'backbone': backbone,
        'revision': revision,
        # The tokenizer and the model load from one snapshot
        'tokenizer': backbone,
        'tokenizer_revision': revision,
        'window': window,
        'budget': {
            channel: summary_budget(count_marked, channel, window)
            for channel in CHANNELS
        },
        'units': {
            'rule': UNIT_RULE,
            'no_break_pattern': NO_BREAK_PATTERN
        },
        'selection': {
            'rule': SELECTION_RULE,
            'pooling': 'attention-masked mean of the last hidden state, each unit unmarked',
            'weights': 'token count without special tokens',
            'device': 'cpu',
            'dtype': 'float32',
        },
        'channels': channels,
        'artifact_sha256': sha256,
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'polars', 'numpy')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
```

- [x] **Step 4: Run the build's tests to verify they pass**

Run: `uv run pytest tests/unit/test_window_summaries_build.py -q`
Expected: `11 passed`.

- [x] **Step 5: Add the command**

In `src/naics_embedder/cli/commands/data.py`, replace:

```python
        it.
    regressor-groups: Draw the regressor panel's held-out four-digit groups, once; they are
        committed and the panel reads them.
    preprocess: Download raw NAICS files and produce descriptions parquet.
    supervision: Build codebook, pair facts, compatibility distance/relation artifacts,
        training pairs, and curriculum thresholds as one versioned bundle.
```

with:

```python
        it.
    regressor-groups: Draw the regressor panel's held-out four-digit groups, once; they are
        committed and the panel reads them.
    summaries: Build the window-fitting summaries of over-long channel texts, once per
        backbone; they are committed and pinned in code.
    preprocess: Download raw NAICS files and produce descriptions parquet.
    supervision: Build codebook, pair facts, compatibility distance/relation artifacts,
        training pairs, and curriculum thresholds as one versioned bundle.
```

Replace:

```python
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
```

with:

```python
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.data.window_summaries import generate_window_summaries
from naics_embedder.panels.window_summaries import WINDOW_SUMMARIES_PATH
from naics_embedder.utils.config import (
    Config,
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
```

Replace:

```python
DOWNLOAD_CONFIG = 'data/download.yaml'
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'

SourceDirOption = Annotated[
    Optional[str],
```

with:

```python
DOWNLOAD_CONFIG = 'data/download.yaml'
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'
TRAINING_CONFIG = 'conf/config.yaml'

SourceDirOption = Annotated[
    Optional[str],
```

Replace:

```python
    typer.echo(f'Regressor held-out groups: {table_path}')

# -------------------------------------------------------------------------------------------------
# Build the Stage-3 supervision bundle
# -------------------------------------------------------------------------------------------------
```

with:

```python
    typer.echo(f'Regressor held-out groups: {table_path}')

# -------------------------------------------------------------------------------------------------
# Build the window-fitting summaries
# -------------------------------------------------------------------------------------------------

@app.command('summaries')
def summaries(
    descriptions: Annotated[
        str,
        typer.Option('--descriptions', help='The descriptions parquet whose texts are summarized'),
    ] = './data/naics_descriptions.parquet',
    backbone: Annotated[
        Optional[str],
        typer.Option('--backbone', help="The backbone (default: the training config's tokenizer)"),
    ] = None,
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the artifact (CSV)'),
    ] = WINDOW_SUMMARIES_PATH,
    force: Annotated[
        bool,
        typer.Option('--force', help='Rebuild an existing artifact, which then needs a new pin'),
    ] = False,
):
    '''
    Build the window-fitting summaries of over-long channel texts, once per backbone.

    Every channel text whose marked form is over the backbone's trained window is summarized by
    the whole units (sentences, clauses or examples entries) whose pooled vectors best approximate
    the text's (roadmap Stage 6b). The backbone is read from the local Hugging Face cache. The
    artifact is checked as every reader checks it before it is moved into place; commit it with
    the pin this prints, in ``WINDOW_SUMMARIES`` (``panels/window_summaries.py``).

    Output:
        ``conf/data/window_summaries.csv`` and ``conf/data/window_summaries_provenance.json``.

    Example:
        Summarize the descriptions bundle 301cce28 was built from::

            $ HF_HUB_OFFLINE=1 uv run naics-embedder data summaries
    '''

    configure_logging('data_summaries.log')

    console.rule('[bold green]Building Window-Fitting Summaries[/bold green]')

    if backbone is None:
        backbone = load_config(Config, TRAINING_CONFIG).data_loader.tokenization.tokenizer_name
    try:
        pin = generate_window_summaries(
            Path(descriptions), Path(output), backbone=backbone, force=force
        )
    # FileExistsError without --force, a missing file or cached backbone, or a failed check
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    typer.echo(f'Window summaries: {pin.path}')
    typer.echo(f'Pin for {backbone}: {pin!r}')

# -------------------------------------------------------------------------------------------------
# Build the Stage-3 supervision bundle
# -------------------------------------------------------------------------------------------------
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_window_summaries_build.py tests/unit/test_cli_commands.py -q`
Expected: `53 passed`.

Run: `uv run naics-embedder data summaries --help`
Expected: the usage, listing `--descriptions`, `--backbone`, `--output` and `--force`. This
reads no data.

- [x] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/data/window_summaries.py src/naics_embedder/cli/commands/data.py tests/unit/test_window_summaries_build.py tests/unit/test_cli_commands.py`
Run: `uv run pytest -n auto -q`
Expected: `1902 passed, 1 skipped`.

```bash
git add src/naics_embedder/data/window_summaries.py src/naics_embedder/cli/commands/data.py tests/unit/test_window_summaries_build.py tests/unit/test_cli_commands.py
git commit -m "feat(window-summaries): build the summaries by backbone centrality (data summaries)"
```

### Task 5: The cache and the text-only builder read summaries

Both readers now resolve their descriptions before tokenizing (4.7). The cache resolves under
`cfg.tokenizer_name`, with the tokenizer it loads. `build_text_only_table` resolves under its
`backbone`, with the tokenizer `load_backbone` returned, then embeds unmarked as today. Its
provenance records `summaries`. A summary that fits with its marker fits without it, so the
comparator reads the arm's texts, markers aside (D9).

The cache test reads one over-window text under a test-local pin (§6, "Cache substitution"). That
pin names a temporary CSV at the test's 32-token window, and the dummy pin cannot serve, because
the resolver would try to read it.

One existing test changes. `TestCacheInvalidation::test_cache_invalidation_max_length_change`
builds a 200-token description at windows 128 and 64. Once the cache resolves, that text would
need a summary at both windows, and the seam's pin names no file, so the build would raise. At 40
tokens the text fits both windows, and the shapes the test checks are unchanged.

**Files:**
- Modify: `src/naics_embedder/panels/text_only.py:31-35, 171-176, 185-190, 203-208`
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py:14-20, 60-76`
- Modify: `tests/unit/test_text_only.py:14-20, 23-28, 156-161`
- Modify: `tests/unit/test_tokenization_cache.py:19-25, 31-36, 534-540, 897-899`

**Interfaces:**
- Consumes: `resolve_channel_texts` and `summaries_identity` (Tasks 1, 3); `pin_artifact` and
  `text_sha256` in the tests.
- Produces:
  - `_build_tokenization_cache(descriptions_path, tokenizer_name, max_length)` caches each
    over-window text as its pinned summary, and raises the resolver's ValueError when there is
    no pin.
  - `build_text_only_table(...)` embeds resolved texts, and its provenance gains `summaries`
    (`summaries_identity(backbone)`).

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_tokenization_cache.py`, replace:

```python
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity
from naics_embedder.text_model.dataloader.tokenization_cache import (
    _acquire_lock,
    _build_tokenization_cache,
```

with:

```python
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity, text_sha256
from naics_embedder.text_model.dataloader.tokenization_cache import (
    _acquire_lock,
    _build_tokenization_cache,
```

Replace:

```python
    load_verified_tokenization_cache,
    tokenization_cache,
)
from naics_embedder.utils.config import TokenizationConfig

FINGERPRINTS = {'description_fingerprint': 'd' * 64, 'codebook_fingerprint': 'c' * 64}
```

with:

```python
    load_verified_tokenization_cache,
    tokenization_cache,
)
from naics_embedder.text_model.fields import tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from tests.fixtures.window_summaries import pin_artifact

FINGERPRINTS = {'description_fingerprint': 'd' * 64, 'codebook_fingerprint': 'c' * 64}
```

Replace:

```python
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food'],
            'description': ['A ' * 200],  # Long description
            'excluded': [''],
            'examples': [''],
        }
```

with:

```python
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food'],
            'description': ['A ' * 40],  # Long, and still inside both windows
            'excluded': [''],
            'examples': [''],
        }
```

Replace:

```python
        assert f"'summaries': ('{recorded}', '{expected}')" in message
        # Only the keys that differ, with both values
        assert 'description_fingerprint' not in message
```

with:

```python
        assert f"'summaries': ('{recorded}', '{expected}')" in message
        # Only the keys that differ, with both values
        assert 'description_fingerprint' not in message

# -------------------------------------------------------------------------------------------------
# Window-fitting summaries
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# 36 tokens marked, over a 32-token window; the summary is 12
OVER_WINDOW = (
    'Farms grow corn. Farms grow wheat. Farms sell grain. Farms buy seed. Farms hire labor. '
    'Farms rent land. Farms store crops. Farms ship feed.'
)
SUMMARY = 'Farms grow corn. Farms sell grain.'

@pytest.fixture
def over_window_descriptions(tmp_path):
    path = tmp_path / 'descriptions.parquet'
    pl.DataFrame(
        {
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food'],
            'description': [OVER_WINDOW],
            'excluded': [None],
            'examples': [None],
        },
        schema_overrides={
            'excluded': pl.Utf8,
            'examples': pl.Utf8
        },
    ).write_parquet(path)
    return path

@pytest.mark.unit
def test_an_over_window_text_is_cached_as_its_pinned_summary(
    tmp_path, monkeypatch, over_window_descriptions
):
    # The seam's MiniLM pin names no file; this test's pin names one, at the test's window
    row = {
        'code': '311111',
        'channel': 'description',
        'source_sha256': text_sha256(OVER_WINDOW),
        'window': 32,
        'summary': SUMMARY,
        'source_tokens': 36,
        'summary_tokens': 12,
        'units_kept': 2,
        'units_total': 8,
    }
    pin = pin_artifact(tmp_path / 'window_summaries.csv', [row], window=32)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, MINILM, pin)

    cache = _build_tokenization_cache(str(over_window_descriptions), MINILM, 32)

    expected = tokenize_field(AutoTokenizer.from_pretrained(MINILM), 'description', SUMMARY, 32)
    assert torch.equal(cache[0]['description']['input_ids'], expected['input_ids'])
    assert cache[0]['description']['present'] is True

@pytest.mark.unit
def test_an_over_window_text_without_a_pin_is_refused(monkeypatch, over_window_descriptions):
    monkeypatch.delitem(window_summaries.WINDOW_SUMMARIES, MINILM)

    with pytest.raises(ValueError, match="code 311111's description is over the 32-token window"):
        _build_tokenization_cache(str(over_window_descriptions), MINILM, 32)
```

In `tests/unit/test_text_only.py`, replace:

```python
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from naics_embedder.panels import text_only
from naics_embedder.panels.text_only import (
    CHANNELS,
    build_text_only_table,
```

with:

```python
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from naics_embedder.panels import text_only, window_summaries
from naics_embedder.panels.text_only import (
    CHANNELS,
    build_text_only_table,
```

Replace:

```python
    provenance_path,
    text_only_fingerprint,
)
from naics_embedder.utils.input_window import TRAINED_WINDOWS

pytestmark = pytest.mark.unit
```

with:

```python
    provenance_path,
    text_only_fingerprint,
)
from naics_embedder.panels.window_summaries import summaries_identity, text_sha256
from naics_embedder.utils.input_window import TRAINED_WINDOWS
from tests.fixtures.window_summaries import pin_artifact

pytestmark = pytest.mark.unit
```

Replace:

```python
    # The name a regressor read logs the table by, so a logged read matches this file
    assert provenance['matrix_fingerprint'] == text_only_fingerprint(table)

def test_a_max_length_beyond_the_trained_window_is_refused(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 8)
    output = tmp_path / 'text_only.parquet'
```

with:

```python
    # The name a regressor read logs the table by, so a logged read matches this file
    assert provenance['matrix_fingerprint'] == text_only_fingerprint(table)

def test_the_table_reads_an_over_window_text_as_its_summary(
    tmp_path, monkeypatch, model, tokenizer
):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 16)
    # 19 tokens marked, over the 16-token window; the summary is 10
    text = 'grows soybeans. grows canola. raises cattle. not here. soybean farming.'
    summary = 'grows soybeans. raises cattle.'
    row = {
        'code': '111110',
        'channel': 'description',
        'source_sha256': text_sha256(text),
        'window': 16,
        'summary': summary,
        'source_tokens': 19,
        'summary_tokens': 10,
        'units_kept': 2,
        'units_total': 5,
    }
    pin = pin_artifact(tmp_path / 'window_summaries.csv', [row], window=16)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, 'tiny-bert', pin)
    descriptions = tmp_path / 'naics_descriptions.parquet'
    _descriptions([('111110', 'soybean farming', text, None, None)]).write_parquet(descriptions)

    path = build_text_only_table(
        descriptions,
        tmp_path / 'text_only.parquet',
        backbone='tiny-bert',
        max_length=16,
        model=model,
        tokenizer=tokenizer,
    )

    expected = _encode(
        _descriptions([('111110', 'soybean farming', summary, None, None)]), model, tokenizer
    )
    np.testing.assert_array_equal(pl.read_parquet(path).drop('code').to_numpy(), expected)
    assert json.loads(provenance_path(path).read_text())['summaries'] == pin.sha256

def test_the_provenance_records_the_backbones_summaries(text_only_comparator_table):
    provenance = json.loads(provenance_path(text_only_comparator_table).read_text())

    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(provenance['backbone'])
    assert provenance['summaries'] is not None

def test_a_max_length_beyond_the_trained_window_is_refused(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 8)
    output = tmp_path / 'text_only.parquet'
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py -q`
Expected: `4 failed, 47 passed`:
- `test_an_over_window_text_is_cached_as_its_pinned_summary` fails on `torch.equal`: the cache
  holds the truncated text.
- `test_an_over_window_text_without_a_pin_is_refused` fails with "DID NOT RAISE".
- `test_the_table_reads_an_over_window_text_as_its_summary` fails on the array comparison: the
  table embedded the truncated text.
- `test_the_provenance_records_the_backbones_summaries` fails with `KeyError: 'summaries'`.

- [x] **Step 3: Resolve before tokenizing**

In `src/naics_embedder/text_model/dataloader/tokenization_cache.py`, replace:

```python
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.text_model.fields import CHANNELS, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window
```

with:

```python
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_embedder.panels.window_summaries import resolve_channel_texts, summaries_identity
from naics_embedder.text_model.fields import CHANNELS, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window
```

Replace:

```python
    '''
    Build tokenization cache from descriptions file.

    Every channel, titles included, is truncated and padded to ``max_length``, which may not
    exceed the backbone's trained window (None is the window).
    '''

    max_length = check_window(tokenizer_name, max_length)
    logger.info('Building tokenization cache...')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # DataFrame iterator
    df_iter = pl.read_parquet(descriptions_path).sort('index').iter_rows(named=True)

    # Tokenization cache
    cache, cnt = {}, {'title': 0, 'description': 0, 'excluded': 0, 'examples': 0}
```

with:

```python
    '''
    Build tokenization cache from descriptions file.

    Every channel text over the window is first replaced by its pinned window-fitting summary
    (``panels/window_summaries.py``), so no channel text is truncated. Every channel, titles
    included, is padded to ``max_length``, which may not exceed the backbone's trained window
    (None is the window).
    '''

    max_length = check_window(tokenizer_name, max_length)
    logger.info('Building tokenization cache...')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    descriptions = resolve_channel_texts(
        pl.read_parquet(descriptions_path), tokenizer, tokenizer_name, max_length
    )

    # DataFrame iterator
    df_iter = descriptions.sort('index').iter_rows(named=True)

    # Tokenization cache
    cache, cnt = {}, {'title': 0, 'description': 0, 'excluded': 0, 'examples': 0}
```

In `src/naics_embedder/panels/text_only.py`, replace:

```python
import torch
from sklearn.decomposition import PCA

from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.input_window import check_window
```

with:

```python
import torch
from sklearn.decomposition import PCA

from naics_embedder.panels.window_summaries import resolve_channel_texts, summaries_identity
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.input_window import check_window
```

Replace:

```python
    '''
    Embed every code's text with the frozen backbone and write the table and its provenance.

    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
```

with:

```python
    '''
    Embed every code's text with the frozen backbone and write the table and its provenance.

    A channel text whose marked form is over the window is read as its pinned window-fitting
    summary, as the arm reads it (``panels/window_summaries.py``); texts are embedded unmarked.
    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
```

Replace:

```python
    descriptions = pl.read_parquet(descriptions_path).sort('code')
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    vectors = encode_code_texts(
        descriptions, model, tokenizer, max_length=max_length, batch_size=batch_size
    )
```

with:

```python
    descriptions = pl.read_parquet(descriptions_path).sort('code')
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    descriptions = resolve_channel_texts(descriptions, tokenizer, backbone, max_length)
    vectors = encode_code_texts(
        descriptions, model, tokenizer, max_length=max_length, batch_size=batch_size
    )
```

Replace:

```python
        'channels': list(CHANNELS),
        'pooling': POOLING,
        'max_length': max_length,
        'codes': table.height,
        'hidden_size': vectors.shape[1],
        'table_sha256': sha256_file(output_path),
```

with:

```python
        'channels': list(CHANNELS),
        'pooling': POOLING,
        'max_length': max_length,
        'summaries': summaries_identity(backbone),
        'codes': table.height,
        'hidden_size': vectors.shape[1],
        'table_sha256': sha256_file(output_path),
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py -q`
Expected: `51 passed`.

- [x] **Step 5: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/panels/text_only.py tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py`
Run: `uv run pytest -n auto -q`
Expected: `1906 passed, 1 skipped`.

```bash
git add src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/panels/text_only.py tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py
git commit -m "feat(window-summaries): the token cache and the text-only builder read summaries"
```

### Task 6: The checkpoint contract records the summaries

A checkpoint records the summaries its token cache applied (4.8, link 1).
- `CheckpointContract` gains `summaries: Optional[str] = None`. A contract saved before this
  stage lacks it and reads as null: that checkpoint trained on truncated text.
- `NAICSContrastiveModel` takes `summaries` as a constructor input, saves it with its
  hyperparameters, and passes it to both contract builders. `build_model_from_config` and
  `runtime_contract_for` pass `summaries_identity(cfg.data_loader.tokenization.tokenizer_name)`,
  the key the cache resolves under.
- `contract_for_bundle`, `containment_contract`, `validate_supervision_contract` and
  `load_arm_model` take `summaries` keyword-only with no default (P10), so a caller that omits it
  is a TypeError, never a silent None. `export_code_table` and `ArmEncoder.from_files` pass
  `summaries_identity(token_config.tokenizer_name)`.

The consequences:
- Exact resume refuses a checkpoint trained under other summaries, legacy containment included.
- Export and reads refuse one, naming the field.
- The HGCN feeder refuses one through `validate_exact_resume`, and needs no edit.
- Weights-only migration compares only the encoder record, so a pre-6b checkpoint can still seed
  a run.

`truncated_checkpoint` (P13) is `shared_model` saved without the field, as a pre-6b checkpoint
is.

**Files:**
- Modify: `src/naics_embedder/cli/commands/training.py:23-28, 134-139, 167-173, 175-183`
- Modify: `src/naics_embedder/supervision/checkpoints.py:109-114, 124-131, 133-140, 148-153, 208-232`
- Modify: `src/naics_embedder/text_model/arm_encoder.py:29-34, 182-188`
- Modify: `src/naics_embedder/text_model/export.py:19-25, 140-156, 159-171, 214-220`
- Modify: `src/naics_embedder/text_model/naics_model.py:166-171, 217-222, 282-288, 292-298`
- Modify: `tests/fixtures/shared_encoder.py:22-27, 106-111, 116-120, 127-132`
- Modify: `tests/unit/test_arm_encoder.py:236-240`
- Modify: `tests/unit/test_checkpoint_contract.py:2-12, 20-31, 69-74, 108-121, 211-222, 232-246, 323-328`
- Modify: `tests/unit/test_cli_training.py:7-11, 16-24, 230-235, 239-247`
- Modify: `tests/unit/test_export.py:123-128, 154-159, 178-183, 220-225`
- Modify: `tests/unit/test_naics_model.py:914-924`

**Interfaces:**
- Consumes: `summaries_identity` (Task 1).
- Produces:
  - `CheckpointContract.summaries: Optional[str] = None`.
  - `contract_for_bundle(manifest, supervision_mode='repaired', *, encoder, summaries:
    Optional[str]) -> CheckpointContract`.
  - `containment_contract(*, encoder, summaries: Optional[str]) -> CheckpointContract`.
  - `validate_supervision_contract(raw, manifest, supervision_mode='repaired', *, summaries:
    Optional[str]) -> CheckpointContract`.
  - `NAICSContrastiveModel(..., summaries: Optional[str] = None, checkpoint_contract=None, ...)`.
  - `load_arm_model(checkpoint_path, bundle, *, summaries: Optional[str], device='cpu')`.
  - In `tests/fixtures/shared_encoder.py`: `shared_model` records `summaries_identity(MINILM)`,
    and the fixture `truncated_checkpoint(tmp_path, shared_model) -> Path` is new.

- [x] **Step 1: Write the failing tests**

In `tests/fixtures/shared_encoder.py`, replace:

```python
from transformers import AutoTokenizer, BertConfig, BertModel

from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import export_code_table
```

with:

```python
from transformers import AutoTokenizer, BertConfig, BertModel

from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import export_code_table
```

Replace:

```python
@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.'''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
```

with:

```python
@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.

    It records MiniLM's summaries, as training does: under the test seam, the dummy pin's sha256.
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
```

Replace:

```python
        dimension=ARM_DIMENSION,
        curvature=1.0,
        supervision_manifest_path=str(generated_bundle),
    )
    return model.eval()
```

with:

```python
        dimension=ARM_DIMENSION,
        curvature=1.0,
        supervision_manifest_path=str(generated_bundle),
        summaries=summaries_identity(MINILM),
    )
    return model.eval()
```

Replace:

```python
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
def exported_table(tmp_path, shared_checkpoint, validated_bundle, five_code_token_config) -> Path:
    '''``shared_checkpoint``'s code table, exported on the CPU, with its provenance beside it.'''
```

with:

```python
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
def truncated_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a checkpoint trained before Stage 6b: it records no summaries.'''

    checkpoint = lightning_checkpoint(shared_model)
    del checkpoint['stage3_supervision']['summaries']
    del checkpoint['hyper_parameters']['summaries']
    path = tmp_path / 'truncated.ckpt'
    torch.save(checkpoint, path)
    return path

@pytest.fixture
def exported_table(tmp_path, shared_checkpoint, validated_bundle, five_code_token_config) -> Path:
    '''``shared_checkpoint``'s code table, exported on the CPU, with its provenance beside it.'''
```

In `tests/unit/test_checkpoint_contract.py`, replace:

```python
import torch
from torch import nn

from naics_embedder.supervision.checkpoints import (
    D2_REFUSAL,
    LEGACY_ENCODER,
    CheckpointContract,
    EncoderArchitecture,
    contract_for_bundle,
    load_weights_only,
    saved_encoder,
```

with:

```python
import torch
from torch import nn

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.checkpoints import (
    D2_REFUSAL,
    LEGACY_ENCODER,
    CheckpointContract,
    EncoderArchitecture,
    containment_contract,
    contract_for_bundle,
    load_weights_only,
    saved_encoder,
```

Replace:

```python
SHARED = shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.fixture
def runtime_contract() -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=SHARED,
    )

@pytest.fixture
```

with:

```python
SHARED = shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.fixture
def summaries() -> str:
    '''MiniLM's summaries: under the test seam, the dummy pin's sha256.'''

    return summaries_identity(MINILM)

@pytest.fixture
def runtime_contract(summaries) -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=SHARED,
        summaries=summaries,
    )

@pytest.fixture
```

Replace:

```python
        {
            'mining_contract_version': 'other-mining'
        },
    ],
)
def test_legacy_or_mismatched_checkpoint_cannot_exact_resume(
```

with:

```python
        {
            'mining_contract_version': 'other-mining'
        },
        # Trained before Stage 6b, on truncated text, or under other summaries
        {
            'summaries': None
        },
        {
            'summaries': 'f' * 64
        },
    ],
)
def test_legacy_or_mismatched_checkpoint_cannot_exact_resume(
```

Replace:

```python
    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)

def test_contract_for_bundle_reads_manifest_identity(validated_bundle):
    contract = contract_for_bundle(validated_bundle.manifest, encoder=SHARED)

    assert contract.supervision_mode == 'repaired'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v2'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
    assert contract.encoder == SHARED

# -------------------------------------------------------------------------------------------------
# The encoder record (spec 4.4)
```

with:

```python
    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)

def test_contract_for_bundle_reads_manifest_identity(validated_bundle, summaries):
    contract = contract_for_bundle(validated_bundle.manifest, encoder=SHARED, summaries=summaries)

    assert contract.supervision_mode == 'repaired'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v2'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
    assert contract.encoder == SHARED
    assert contract.summaries == summaries

# -------------------------------------------------------------------------------------------------
# The summaries (Stage 6b spec, 4.8)
# -------------------------------------------------------------------------------------------------

def test_a_contract_saved_before_stage_6b_reads_as_null_summaries(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['summaries']

    assert CheckpointContract.model_validate(saved).summaries is None

def test_a_containment_checkpoint_under_other_summaries_cannot_exact_resume(tmp_path, summaries):
    runtime = containment_contract(encoder=SHARED, summaries=summaries)
    path = _save(
        tmp_path / 'containment.ckpt', containment_contract(encoder=SHARED, summaries=None)
    )

    with pytest.raises(ValueError, match='exact resume') as refusal:
        validate_exact_resume(path, runtime)

    assert f"'summaries': (None, '{summaries}')" in str(refusal.value)

def test_the_supervision_check_refuses_other_summaries_naming_the_field(
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=None)

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries)

def test_a_caller_that_omits_the_summaries_is_a_type_error(validated_bundle, summaries):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries).model_dump()

    with pytest.raises(TypeError):
        validate_supervision_contract(saved, manifest)
    with pytest.raises(TypeError):
        contract_for_bundle(manifest, encoder=SHARED)
    with pytest.raises(TypeError):
        containment_contract(encoder=SHARED)

# -------------------------------------------------------------------------------------------------
# The encoder record (spec 4.4)
```

Replace:

```python
# Export and reads compare the supervision fields only
# -------------------------------------------------------------------------------------------------

def test_the_supervision_check_takes_the_encoder_record_from_the_checkpoint(validated_bundle):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(fusion='attention', dimension=8, backbone=MINILM)
    saved = contract_for_bundle(manifest, encoder=other)

    assert validate_supervision_contract(saved.model_dump(), manifest) == saved

@pytest.mark.parametrize(
    'update',
```

with:

```python
# Export and reads compare the supervision fields only
# -------------------------------------------------------------------------------------------------

def test_the_supervision_check_takes_the_encoder_record_from_the_checkpoint(
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(fusion='attention', dimension=8, backbone=MINILM)
    saved = contract_for_bundle(manifest, encoder=other, summaries=summaries)

    assert validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries) == saved

@pytest.mark.parametrize(
    'update',
```

Replace:

```python
        },
    ],
)
def test_the_supervision_check_refuses_another_bundle(validated_bundle, update):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED).model_copy(update=update)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        validate_supervision_contract(saved.model_dump(), manifest)

def test_the_supervision_check_refuses_a_checkpoint_without_a_contract(validated_bundle):
    with pytest.raises(ValueError, match='no Stage-3 contract') as excinfo:
        validate_supervision_contract(None, validated_bundle.manifest)

    assert D2_REFUSAL in str(excinfo.value)
```

with:

```python
        },
    ],
)
def test_the_supervision_check_refuses_another_bundle(validated_bundle, summaries, update):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        validate_supervision_contract(
            saved.model_copy(update=update).model_dump(), manifest, summaries=summaries
        )

def test_the_supervision_check_refuses_a_checkpoint_without_a_contract(validated_bundle):
    with pytest.raises(ValueError, match='no Stage-3 contract') as excinfo:
        validate_supervision_contract(None, validated_bundle.manifest, summaries=None)

    assert D2_REFUSAL in str(excinfo.value)
```

Replace:

```python
    with pytest.raises(ValueError, match='no allowlisted encoder parameters'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_rejects_shape_mismatched_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
```

with:

```python
    with pytest.raises(ValueError, match='no allowlisted encoder parameters'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_loads_a_checkpoint_trained_under_other_summaries(
    tmp_path, runtime_contract, tiny_repaired_model
):
    # Weights-only compares the encoder record alone, so a pre-6b checkpoint can seed a run
    key = sorted(name for name in tiny_repaired_model.state_dict()
                 if name.startswith('encoder.'))[0]
    path = _save(
        tmp_path / 'truncated.ckpt',
        runtime_contract.model_copy(update={'summaries': None}),
        state_dict={key: torch.zeros_like(tiny_repaired_model.state_dict()[key])},
    )

    report = load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert report.loaded == (key, )

def test_weights_only_rejects_shape_mismatched_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
```

In `tests/unit/test_naics_model.py`, replace:

```python
            encoder=shared_encoder_architecture(
                fusion='masked_mean', dimension=8, backbone=model_config['base_model_name']
            ),
        )

        with pytest.raises(ValueError, match='does not match'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_on_save_checkpoint_writes_contract(self, naics_model):
        checkpoint = {}
        naics_model.on_save_checkpoint(checkpoint)
```

with:

```python
            encoder=shared_encoder_architecture(
                fusion='masked_mean', dimension=8, backbone=model_config['base_model_name']
            ),
            summaries=None,
        )

        with pytest.raises(ValueError, match='does not match'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_the_model_records_its_summaries_in_either_contract(self, model_config):
        repaired = NAICSContrastiveModel(**model_config, summaries='e' * 64)
        containment = NAICSContrastiveModel(
            **model_config, supervision_mode='legacy_containment', summaries='e' * 64
        )

        assert repaired.checkpoint_contract.summaries == 'e' * 64
        assert containment.checkpoint_contract.summaries == 'e' * 64
        # Saved with the hyperparameters, so load_from_checkpoint rebuilds the same contract
        assert repaired.hparams['summaries'] == 'e' * 64

    def test_on_save_checkpoint_writes_contract(self, naics_model):
        checkpoint = {}
        naics_model.on_save_checkpoint(checkpoint)
```

In `tests/unit/test_cli_training.py`, replace:

```python
from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    MigrationReport,
```

with:

```python
from naics_embedder.cli import app as cli_app
from naics_embedder.cli.commands import training
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    MigrationReport,
```

Replace:

```python
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationError, ValidationResult

# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone='sentence-transformers/all-MiniLM-L6-v2'
)

@pytest.fixture
```

with:

```python
from naics_embedder.utils.training import CheckpointInfo, HardwareInfo
from naics_embedder.utils.validation import ValidationError, ValidationResult

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM
)

@pytest.fixture
```

Replace:

```python
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=CONFIGURED_ENCODER,
    )

@pytest.mark.unit
```

with:

```python
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=CONFIGURED_ENCODER,
        # The seam's dummy pin for the configured tokenizer (tests/conftest.py)
        summaries=summaries_identity(MINILM),
    )

@pytest.mark.unit
```

Replace:

```python
    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone='sentence-transformers/all-MiniLM-L6-v2'
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)

@pytest.mark.unit
def test_exact_resume_contract_mismatch_fails_before_training(training_env, monkeypatch):
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)
```

with:

```python
    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)

@pytest.mark.unit
def test_the_model_and_its_contract_record_the_tokenizers_summaries(training_env):
    training.train(skip_validation=True)

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['summaries'] == summaries_identity(MINILM)
    assert model_kwargs['checkpoint_contract'].summaries == summaries_identity(MINILM)
    assert model_kwargs['summaries'] is not None

@pytest.mark.unit
def test_a_containment_run_records_the_summaries_too(training_env):
    training.train(skip_validation=True, overrides=['supervision.mode=legacy_containment'])

    contract = training_env.trainer.fit_calls[0]['model'].kwargs['checkpoint_contract']
    assert contract.supervision_mode == 'legacy_containment'
    assert contract.summaries == summaries_identity(MINILM)

@pytest.mark.unit
def test_exact_resume_contract_mismatch_fails_before_training(training_env, monkeypatch):
    training_env.checkpoint_info = CheckpointInfo(path='foo.ckpt', is_same_stage=True, exists=True)
```

In `tests/unit/test_export.py`, replace:

```python
    # Each row lies on the hyperboloid: -x0^2 + |x|^2 = -1
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)

# -------------------------------------------------------------------------------------------------
# The code-table export
# -------------------------------------------------------------------------------------------------
```

with:

```python
    # Each row lies on the hyperboloid: -x0^2 + |x|^2 = -1
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)

def test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text(
    monkeypatch, tmp_path, truncated_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(five_code_descriptions_parquet)
    output = tmp_path / 'encodings.parquet'

    with pytest.raises(ValueError, match="exact resume contract mismatch .*'summaries'"):
        training_cli.generate_embeddings_from_checkpoint(
            str(truncated_checkpoint), cfg, str(output)
        )
    assert not output.exists()

# -------------------------------------------------------------------------------------------------
# The code-table export
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
def test_the_table_holds_each_codes_capped_tangent(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    model, _ = load_arm_model(shared_checkpoint, validated_bundle)
    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    tangent = encode_token_rows(model, rows)['tangent']
```

with:

```python
def test_the_table_holds_each_codes_capped_tangent(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    model, _ = load_arm_model(
        shared_checkpoint, validated_bundle, summaries=summaries_identity(MINILM)
    )
    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    tangent = encode_token_rows(model, rows)['tangent']
```

Replace:

```python
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM
        ),
    )
    assert provenance['contract'] == expected.model_dump(mode='json')
    assert provenance['backbone'] == MINILM
```

with:

```python
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM
        ),
        summaries=summaries_identity(MINILM),
    )
    assert provenance['contract'] == expected.model_dump(mode='json')
    assert provenance['backbone'] == MINILM
```

Replace:

```python
    with pytest.raises(ValueError, match='supervision contract mismatch'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_a_four_copy_checkpoint_is_refused_with_d2(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
```

with:

```python
    with pytest.raises(ValueError, match='supervision contract mismatch'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_a_load_that_omits_the_summaries_is_a_type_error(shared_checkpoint, validated_bundle):
    with pytest.raises(TypeError):
        load_arm_model(shared_checkpoint, validated_bundle)

def test_a_checkpoint_trained_on_truncated_text_is_refused(
    tmp_path, truncated_checkpoint, validated_bundle, five_code_token_config
):
    output = tmp_path / 'table.parquet'

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        export_code_table(truncated_checkpoint, validated_bundle, five_code_token_config, output)
    assert not output.exists()

def test_a_four_copy_checkpoint_is_refused_with_d2(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
```

In `tests/unit/test_arm_encoder.py`, replace:

```python
    # The key the read's window comes from (code_token_config), so the remedy points at it
    assert 'data_loader.streaming.max_length' in str(refusal.value)

def test_queries_are_tokenized_at_the_tables_window(arm, exported_table):
    provenance = json.loads(provenance_path(exported_table).read_text())
```

with:

```python
    # The key the read's window comes from (code_token_config), so the remedy points at it
    assert 'data_loader.streaming.max_length' in str(refusal.value)

def test_a_checkpoint_trained_on_truncated_text_is_refused_on_read(
    truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
):
    # The provenance names the truncated checkpoint, so only its contract can refuse it
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance['checkpoint']['sha256'] = sha256_file(truncated_checkpoint)
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        ArmEncoder.from_files(
            truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_queries_are_tokenized_at_the_tables_window(arm, exported_table):
    provenance = json.loads(provenance_path(exported_table).read_text())
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_arm_encoder.py -q`
Expected: `14 failed, 94 passed, 62 errors`:
- Every fixture that builds a model with `summaries=` errors with `TypeError:
  NAICSContrastiveModel.__init__() got an unexpected keyword argument 'summaries'`.
- Contracts built with `summaries=` raise `ValidationError: 1 validation error for
  CheckpointContract`, since the model forbids extra fields.
- `contract_for_bundle`, `containment_contract` and `validate_supervision_contract` raise
  TypeError on the `summaries` keyword.

- [x] **Step 3: Record the summaries in the contract**

In `src/naics_embedder/supervision/checkpoints.py`, replace:

```python
    mining_contract_version: str = MINING_CONTRACT_VERSION
    # Absent from every contract saved before Stage 6, which therefore reads as four-copy
    encoder: EncoderArchitecture = LEGACY_ENCODER

@dataclass(frozen=True)
class MigrationReport:
```

with:

```python
    mining_contract_version: str = MINING_CONTRACT_VERSION
    # Absent from every contract saved before Stage 6, which therefore reads as four-copy
    encoder: EncoderArchitecture = LEGACY_ENCODER
    # The sha256 of the window-fitting summaries the model read (panels/window_summaries.py).
    # Absent from every contract saved before Stage 6b, which therefore reads as null: those
    # checkpoints trained on truncated text
    summaries: Optional[str] = None

@dataclass(frozen=True)
class MigrationReport:
```

Replace:

```python
    supervision_mode: str = 'repaired',
    *,
    encoder: EncoderArchitecture,
) -> CheckpointContract:
    '''The runtime contract for training against a validated bundle manifest.'''

    return CheckpointContract(
        supervision_mode=supervision_mode,
```

with:

```python
    supervision_mode: str = 'repaired',
    *,
    encoder: EncoderArchitecture,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    The runtime contract for training against a validated bundle manifest.

    ``summaries`` is the sha256 of the window-fitting summaries the model reads, or None for a
    backbone with no pin (``panels.window_summaries.summaries_identity``).
    '''

    return CheckpointContract(
        supervision_mode=supervision_mode,
```

Replace:

```python
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        encoder=encoder,
    )

def containment_contract(*, encoder: EncoderArchitecture) -> CheckpointContract:
    '''
    The tag every legacy-containment checkpoint carries.
```

with:

```python
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        encoder=encoder,
        summaries=summaries,
    )

def containment_contract(
    *,
    encoder: EncoderArchitecture,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    The tag every legacy-containment checkpoint carries.
```

Replace:

```python
        bundle_id=LEGACY_CONTAINMENT_BUNDLE_ID,
        codebook_fingerprint=UNVERSIONED_CODEBOOK_FINGERPRINT,
        encoder=encoder,
    )

def saved_encoder(raw: Optional[Dict[str, Any]]) -> EncoderArchitecture:
```

with:

```python
        bundle_id=LEGACY_CONTAINMENT_BUNDLE_ID,
        codebook_fingerprint=UNVERSIONED_CODEBOOK_FINGERPRINT,
        encoder=encoder,
        summaries=summaries,
    )

def saved_encoder(raw: Optional[Dict[str, Any]]) -> EncoderArchitecture:
```

Replace:

```python
    raw: Optional[Dict[str, Any]],
    manifest: Any,
    supervision_mode: str = 'repaired',
) -> CheckpointContract:
    '''
    Require a saved contract whose supervision fields match the configured bundle's.

    Export and reads take the encoder record from the checkpoint (spec 4.4), so it is not compared
    here. ``load_from_checkpoint`` rebuilds the checkpoint's own architecture from its saved
    hyperparameters, and refuses a four-copy one.

    Returns:
        The saved contract.

    Raises:
        ValueError: If the checkpoint has no contract, or a supervision field differs.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    configured = contract_for_bundle(manifest, supervision_mode, encoder=saved.encoder)
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
```

with:

```python
    raw: Optional[Dict[str, Any]],
    manifest: Any,
    supervision_mode: str = 'repaired',
    *,
    summaries: Optional[str],
) -> CheckpointContract:
    '''
    Require a saved contract whose supervision fields and summaries match the configured ones.

    Export and reads take the encoder record from the checkpoint (spec 4.4), so it is not compared
    here. ``load_from_checkpoint`` rebuilds the checkpoint's own architecture from its saved
    hyperparameters, and refuses a four-copy one.

    Args:
        raw: The checkpoint's saved contract, or None.
        manifest: The configured bundle's manifest.
        supervision_mode: The configured supervision mode.
        summaries: The sha256 of the summaries the read applies; keyword-only with no default,
            so a caller cannot omit it.

    Returns:
        The saved contract.

    Raises:
        ValueError: If the checkpoint has no contract, or a supervision field or the summaries
            differ.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    configured = contract_for_bundle(
        manifest, supervision_mode, encoder=saved.encoder, summaries=summaries
    )
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
```

- [x] **Step 4: Take the summaries as a model input**

In `src/naics_embedder/text_model/naics_model.py`, replace:

```python
        structural_preference_margin: Ordering margin for structural preference
        structural_preference_temperature: Softplus temperature for structural preference
        structural_preference_tie_tolerance: Structural distance tie tolerance
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
        supervision_bundle: Optional already-validated bundle for ``supervision_manifest_path``
            (not saved in hyperparameters)
```

with:

```python
        structural_preference_margin: Ordering margin for structural preference
        structural_preference_temperature: Softplus temperature for structural preference
        structural_preference_tie_tolerance: Structural distance tie tolerance
        summaries: The sha256 of the window-fitting summaries the token cache applied, or None
            for a backbone with no pin; recorded in the checkpoint contract
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
        supervision_bundle: Optional already-validated bundle for ``supervision_manifest_path``
            (not saved in hyperparameters)
```

Replace:

```python
        structural_preference_margin: float = 0.1,
        structural_preference_temperature: float = 1.0,
        structural_preference_tie_tolerance: float = 1e-6,
        checkpoint_contract: Optional[CheckpointContract] = None,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
    ):
```

with:

```python
        structural_preference_margin: float = 0.1,
        structural_preference_temperature: float = 1.0,
        structural_preference_tie_tolerance: float = 1e-6,
        summaries: Optional[str] = None,
        checkpoint_contract: Optional[CheckpointContract] = None,
        supervision_bundle: Optional[ValidatedSupervisionBundle] = None,
    ):
```

Replace:

```python
                    )
                bundle = supervision_bundle
            runtime_contract = contract_for_bundle(
                bundle.manifest, supervision_mode, encoder=encoder_record
            )
            self.relation_id_to_name = {
                relation_id: name
```

with:

```python
                    )
                bundle = supervision_bundle
            runtime_contract = contract_for_bundle(
                bundle.manifest, supervision_mode, encoder=encoder_record, summaries=summaries
            )
            self.relation_id_to_name = {
                relation_id: name
```

Replace:

```python
            self.selection_coordinator = NegativeSelectionCoordinator()
            self.naics_hierarchy = load_naics_hierarchy(str(bundle.artifact_path('relations')))
        else:
            runtime_contract = containment_contract(encoder=encoder_record)
            logger.warning(
                'LEGACY CONTAINMENT (%s): not contract-compliant Stage-3 training. Structural '
                'ranking and hierarchy losses, negative reordering, and pseudo-related handling '
```

with:

```python
            self.selection_coordinator = NegativeSelectionCoordinator()
            self.naics_hierarchy = load_naics_hierarchy(str(bundle.artifact_path('relations')))
        else:
            runtime_contract = containment_contract(encoder=encoder_record, summaries=summaries)
            logger.warning(
                'LEGACY CONTAINMENT (%s): not contract-compliant Stage-3 training. Structural '
                'ranking and hierarchy losses, negative reordering, and pseudo-related handling '
```

- [x] **Step 5: Pass the tokenizer's summaries from training**

In `src/naics_embedder/cli/commands/training.py`, replace:

```python
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
```

with:

```python
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
```

Replace:

```python
        structural_preference_margin=structural_preference.margin,
        structural_preference_temperature=structural_preference.temperature,
        structural_preference_tie_tolerance=structural_preference.tie_tolerance,
        checkpoint_contract=runtime_contract,
        **supervision_inputs,
    )
```

with:

```python
        structural_preference_margin=structural_preference.margin,
        structural_preference_temperature=structural_preference.temperature,
        structural_preference_tie_tolerance=structural_preference.tie_tolerance,
        # The key the token cache resolves under (spec 4.8)
        summaries=summaries_identity(cfg.data_loader.tokenization.tokenizer_name),
        checkpoint_contract=runtime_contract,
        **supervision_inputs,
    )
```

Replace:

```python
    cfg: Config, bundle: Optional[ValidatedSupervisionBundle]
) -> CheckpointContract:
    '''
    The checkpoint contract of the configured run, its encoder record included.

    The supervision gate returns no bundle only for explicit legacy containment. Training's exact
    resume and the HGCN feeder compare this whole contract with a checkpoint's; export and reads
```

with:

```python
    cfg: Config, bundle: Optional[ValidatedSupervisionBundle]
) -> CheckpointContract:
    '''
    The checkpoint contract of the configured run, its encoder record and summaries included.

    The supervision gate returns no bundle only for explicit legacy containment. Training's exact
    resume and the HGCN feeder compare this whole contract with a checkpoint's; export and reads
```

Replace:

```python
    '''

    encoder = encoder_architecture_for(cfg)
    if bundle is None:
        return containment_contract(encoder=encoder)
    return contract_for_bundle(bundle.manifest, cfg.supervision.mode, encoder=encoder)

def log_migration_report(report: MigrationReport) -> None:
    '''Report what a weights-only migration loaded, skipped, and left freshly initialized.'''
```

with:

```python
    '''

    encoder = encoder_architecture_for(cfg)
    summaries = summaries_identity(cfg.data_loader.tokenization.tokenizer_name)
    if bundle is None:
        return containment_contract(encoder=encoder, summaries=summaries)
    return contract_for_bundle(
        bundle.manifest, cfg.supervision.mode, encoder=encoder, summaries=summaries
    )

def log_migration_report(report: MigrationReport) -> None:
    '''Report what a weights-only migration loaded, skipped, and left freshly initialized.'''
```

- [x] **Step 6: Check the summaries on export and read**

In `src/naics_embedder/text_model/export.py`, replace:

```python
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import polars as pl
import torch
```

with:

```python
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import polars as pl
import torch
```

Replace:

```python
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    *,
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[NAICSContrastiveModel, CheckpointContract]:
    '''
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). Its supervision fields must match ``bundle``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        device: Where the model runs.

    Returns:
```

with:

```python
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    *,
    summaries: Optional[str],
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[NAICSContrastiveModel, CheckpointContract]:
    '''
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). Its supervision fields must match ``bundle``, and
    its summaries ``summaries``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        summaries: The sha256 of the summaries the read's token cache applies
            (``summaries_identity`` of its tokenizer); keyword-only with no default.
        device: Where the model runs.

    Returns:
```

Replace:

```python
    Raises:
        ValueError: If the curvature is not 1 (R8), the supervision contract is not the bundle's,
            or the checkpoint is of another encoder architecture (D2).
    '''

    # Lightning checkpoints carry pickled hyperparameters; they are trusted artifacts of this
    # project's own training runs
    raw = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    require_unit_curvature(raw.get('hyper_parameters', {}))
    contract = validate_supervision_contract(raw.get(CHECKPOINT_KEY), bundle.manifest)
    # on_load_checkpoint refuses another encoder architecture before the state dict loads (D2)
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
```

with:

```python
    Raises:
        ValueError: If the curvature is not 1 (R8), the supervision contract is not the bundle's,
            the checkpoint was trained under other summaries, or it is of another encoder
            architecture (D2).
    '''

    # Lightning checkpoints carry pickled hyperparameters; they are trusted artifacts of this
    # project's own training runs
    raw = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    require_unit_curvature(raw.get('hyper_parameters', {}))
    contract = validate_supervision_contract(
        raw.get(CHECKPOINT_KEY), bundle.manifest, summaries=summaries
    )
    # on_load_checkpoint refuses another encoder architecture before the state dict loads (D2)
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
```

Replace:

```python
            table, which is then not written.
    '''

    model, contract = load_arm_model(checkpoint_path, bundle, device=device)
    descriptions_path = Path(token_config.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path).sort('index')
    codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
```

with:

```python
            table, which is then not written.
    '''

    model, contract = load_arm_model(
        checkpoint_path,
        bundle,
        summaries=summaries_identity(token_config.tokenizer_name),
        device=device,
    )
    descriptions_path = Path(token_config.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path).sort('index')
    codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
```

In `src/naics_embedder/text_model/arm_encoder.py`, replace:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.export import encode_token_rows, load_arm_model
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.export import encode_token_rows, load_arm_model
```

Replace:

```python
                f'tokenizes queries at {token_config.max_length}: export the table and read under '
                'one data_loader.streaming.max_length'
            )
        model, _ = load_arm_model(checkpoint_path, bundle, device=device)
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
```

with:

```python
                f'tokenizes queries at {token_config.max_length}: export the table and read under '
                'one data_loader.streaming.max_length'
            )
        model, _ = load_arm_model(
            checkpoint_path,
            bundle,
            summaries=summaries_identity(token_config.tokenizer_name),
            device=device,
        )
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_arm_encoder.py -q`
Expected: `170 passed`.

- [x] **Step 8: Format, run the suite, commit**

> Deviation: the permission system denied the implementer's add and commit; the controller ran this step's exact add and commit (9d2623e) with the user's approval.

Run: `./scripts/format_code.sh src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py src/naics_embedder/text_model/export.py src/naics_embedder/text_model/arm_encoder.py tests/fixtures/shared_encoder.py tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_arm_encoder.py`
Run: `uv run pytest -n auto -q`
Expected: `1920 passed, 1 skipped`.

```bash
git add src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py src/naics_embedder/text_model/export.py src/naics_embedder/text_model/arm_encoder.py tests/fixtures/shared_encoder.py tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_arm_encoder.py
git commit -m "feat(checkpoints): record the summaries in the contract; export and reads check them"
```

### Task 7: The export records its tokenizer, and a read checks both

The export provenance gains `tokenizer`, `token_config.tokenizer_name` (4.8, link 2): the
tokenizer-name part of plan 8's "untidy failures" item. Before the model loads,
`ArmEncoder.from_files` refuses a table (link 3, P11):
- whose provenance lacks `summaries` or `tokenizer`, so the table was exported before this stage;
- exported with another tokenizer than `token_config.tokenizer_name`;
- exported under other summaries than `summaries_identity(token_config.tokenizer_name)`.

`load_arm_model`'s contract check against the same value then closes the chain to the
checkpoint (Task 6).

**Files:**
- Modify: `src/naics_embedder/text_model/arm_encoder.py:140-147, 156-163, 169-174, 183-194`
- Modify: `src/naics_embedder/text_model/export.py:272-277`
- Modify: `tests/unit/test_arm_encoder.py:236-241`
- Modify: `tests/unit/test_export.py:210-215`

**Interfaces:**
- Consumes: Task 6's `load_arm_model(..., summaries=...)`; the `no_model_load` fixture already in
  `tests/unit/test_arm_encoder.py`.
- Produces:
  - The export provenance's `tokenizer` entry.
  - Three new refusals in `ArmEncoder.from_files`, each a ValueError:
    - `'<table> was exported before Stage 6b: its provenance records no <key>; export the table
      again'`;
    - `'<table> was exported with the tokenizer <name>, but this read tokenizes queries with
      <name>'`;
    - `'<table> was exported under the summaries <sha256 or None>, but <tokenizer> reads under
      <sha256 or None>: export the table again'`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_arm_encoder.py`, replace:

```python
    # The key the read's window comes from (code_token_config), so the remedy points at it
    assert 'data_loader.streaming.max_length' in str(refusal.value)

def test_a_checkpoint_trained_on_truncated_text_is_refused_on_read(
    truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
):
```

with:

```python
    # The key the read's window comes from (code_token_config), so the remedy points at it
    assert 'data_loader.streaming.max_length' in str(refusal.value)

@pytest.mark.parametrize('missing', ['summaries', 'tokenizer'])
def test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads(
    no_model_load, missing, exported_table, shared_checkpoint, validated_bundle,
    five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    del provenance[missing]
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=f'before Stage 6b: its provenance records no {missing};'):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

@pytest.mark.parametrize(
    ('entry', 'value', 'refusal'),
    [
        ('tokenizer', 'other/tokenizer', 'exported with the tokenizer other/tokenizer'),
        ('summaries', None, 'exported under the summaries None'),
        ('summaries', 'f' * 64, f"exported under the summaries {'f' * 64}"),
    ],
)
def test_a_table_read_under_another_tokenizer_or_summaries_is_refused_before_any_model_loads(
    no_model_load, entry, value, refusal, exported_table, shared_checkpoint, validated_bundle,
    five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance[entry] = value
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=refusal):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_checkpoint_trained_on_truncated_text_is_refused_on_read(
    truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
):
```

In `tests/unit/test_export.py`, replace:

```python
    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(MINILM)
    assert provenance['summaries'] is not None
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
```

with:

```python
    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(MINILM)
    assert provenance['summaries'] is not None
    assert provenance['tokenizer'] == MINILM
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_export.py -q`
Expected: `6 failed, 35 passed`:
- `test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads[summaries]` and the
  three cases of
  `test_a_table_read_under_another_tokenizer_or_summaries_is_refused_before_any_model_loads` fail
  on `AssertionError: the model loaded before the provenance was refused`.
- `test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads[tokenizer]` and
  `test_the_provenance_names_the_table_and_the_checkpoint` fail with `KeyError: 'tokenizer'`.

- [x] **Step 3: Record the tokenizer, and refuse before the model loads**

In `src/naics_embedder/text_model/export.py`, replace:

```python
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': summaries_identity(token_config.tokenizer_name),
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
```

with:

```python
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': summaries_identity(token_config.tokenizer_name),
        # The tokenizer the codes were read with, which a read's queries must share
        'tokenizer': token_config.tokenizer_name,
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
```

In `src/naics_embedder/text_model/arm_encoder.py`, replace:

```python
        The arm of a checkpoint and the table exported from it.

        The table's provenance is checked before the model loads. It must name this checkpoint
        and this table file, and the window it records, which the table's codes were encoded at,
        must be the one the queries will be tokenized at: the arm has one preprocessing contract.

        Args:
            checkpoint_path: The arm's Lightning checkpoint.
```

with:

```python
        The arm of a checkpoint and the table exported from it.

        The table's provenance is checked before the model loads. It must name this checkpoint
        and this table file, and the window, tokenizer and summaries it records, which the table's
        codes were read under, must be the ones the queries will be read under: the arm has one
        preprocessing contract.

        Args:
            checkpoint_path: The arm's Lightning checkpoint.
```

Replace:

```python
        Raises:
            ValueError: If the table's provenance is no exported arm table's (it names no
                checkpoint, table hash or window), names another checkpoint, or the table file is
                not the one it names; if the table was exported at another window than
                ``token_config``'s; or as ``load_arm_model``: a curvature other than 1 (R8),
                another supervision contract, or another encoder architecture (D2).
            FileNotFoundError: If the checkpoint, the table or its provenance is missing.
        '''
```

with:

```python
        Raises:
            ValueError: If the table's provenance is no exported arm table's (it names no
                checkpoint, table hash or window), predates Stage 6b (it records no summaries or
                tokenizer), names another checkpoint, or the table file is not the one it names;
                if the table was exported at another window, with another tokenizer or under
                other summaries than ``token_config``'s; or as ``load_arm_model``: a curvature
                other than 1 (R8), another supervision contract or summaries, or another encoder
                architecture (D2).
            FileNotFoundError: If the checkpoint, the table or its provenance is missing.
        '''
```

Replace:

```python
        named_checkpoint = _provenance_entry(provenance, table_path, 'checkpoint', 'sha256')
        named_table = _provenance_entry(provenance, table_path, 'table_sha256')
        exported_window = _provenance_entry(provenance, table_path, 'max_length')
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if named_checkpoint != checkpoint_sha256:
            raise ValueError(
```

with:

```python
        named_checkpoint = _provenance_entry(provenance, table_path, 'checkpoint', 'sha256')
        named_table = _provenance_entry(provenance, table_path, 'table_sha256')
        exported_window = _provenance_entry(provenance, table_path, 'max_length')
        for key in ('summaries', 'tokenizer'):
            if key not in provenance:
                raise ValueError(
                    f'{table_path} was exported before Stage 6b: its provenance records no {key}; '
                    'export the table again'
                )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if named_checkpoint != checkpoint_sha256:
            raise ValueError(
```

Replace:

```python
                f'tokenizes queries at {token_config.max_length}: export the table and read under '
                'one data_loader.streaming.max_length'
            )
        model, _ = load_arm_model(
            checkpoint_path,
            bundle,
            summaries=summaries_identity(token_config.tokenizer_name),
            device=device,
        )
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
```

with:

```python
                f'tokenizes queries at {token_config.max_length}: export the table and read under '
                'one data_loader.streaming.max_length'
            )
        if provenance['tokenizer'] != token_config.tokenizer_name:
            raise ValueError(
                f"{table_path} was exported with the tokenizer {provenance['tokenizer']}, but this "
                f'read tokenizes queries with {token_config.tokenizer_name}'
            )
        summaries = summaries_identity(token_config.tokenizer_name)
        if provenance['summaries'] != summaries:
            raise ValueError(
                f"{table_path} was exported under the summaries {provenance['summaries']}, but "
                f'{token_config.tokenizer_name} reads under {summaries}: export the table again'
            )
        model, _ = load_arm_model(checkpoint_path, bundle, summaries=summaries, device=device)
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_export.py -q`
Expected: `41 passed`.

- [x] **Step 5: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/export.py src/naics_embedder/text_model/arm_encoder.py tests/unit/test_arm_encoder.py tests/unit/test_export.py`
Run: `uv run pytest -n auto -q`
Expected: `1925 passed, 1 skipped`.

```bash
git add src/naics_embedder/text_model/export.py src/naics_embedder/text_model/arm_encoder.py tests/unit/test_arm_encoder.py tests/unit/test_export.py
git commit -m "feat(export): record the tokenizer; a read refuses other tokenizers or summaries"
```

### Task 8: The decision records check the summaries

The decision records add the summaries' sha256 to the four fields D9 already compares (4.8, link
5; P12):
- The store reads `summaries_sha256` from a text-only provenance's `summaries` key. A provenance
  without it, which means every text-only table built before this stage, is refused as lacking a
  field D9 reads.
- `TextOnlyRef` and `ArmSpec` gain `summaries_sha256: Optional[str]`, required and nullable.
- `check_text_only` compares five fields: backbone, revision, descriptions sha256, summaries
  sha256 and window.

`run_seed_sweep` reads each seed's export provenance and requires the same five fields to equal
the `ArmSpec`'s, refusing the seed otherwise.
- The check runs right after `runner.run`, before `store.put` and any panel read, as the
  dimension check does.
- A refusal at the first seed leaves the selection log empty.
- The synthetic sweep's runner now writes the export provenance that `tools export-table`
  writes.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py:22-28, 80-104`
- Modify: `src/naics_embedder/decision/records.py:57-62, 70-75, 81-85`
- Modify: `src/naics_embedder/decision/store.py:35-41, 59-64`
- Modify: `src/naics_embedder/decision/sweep.py:16-37, 48-53, 67-72, 95-103, 110-115`
- Modify: `tests/fixtures/decision.py:32-37, 42-47, 73-78, 147-152, 162-167, 169-173`
- Modify: `tests/unit/test_decision.py:394-399`
- Modify: `tests/unit/test_decision_rule.py:29-33`
- Modify: `tests/unit/test_decision_store.py:14-20, 120-125, 144-148, 156-161`
- Modify: `tests/unit/test_decision_sweep.py:20-27, 52-63, 73-78, 215-220`

**Interfaces:**
- Consumes: `provenance_fields(provenance, table_sha256, matrix_fingerprint, name)`
  (`decision/store.py`); `table_fingerprint` (`panels/regressor.py`); `provenance_path`
  (`panels/text_only.py`); `sha256_file` (`supervision/artifacts.py`).
- Produces:
  - `TextOnlyRef.summaries_sha256` and `ArmSpec.summaries_sha256`, each `Optional[str]` with no
    default.
  - `provenance_fields(...)['summaries_sha256']`.
  - In `decision/decide.py`: `D9_FIELDS = ('backbone', 'revision', 'descriptions_sha256',
    'summaries_sha256', 'max_length')`, and `check_seed_table(spec: ArmSpec, seed: int, fields:
    Mapping[str, Any]) -> None`.
  - In `decision/sweep.py`: `_seed_table_fields(table: Path) -> Mapping[str, Any]`.
  - In `tests/fixtures/decision.py`: `SUMMARIES_SHA256 = 'e' * 64`; `spec()` records it;
    `write_text_only(..., summaries=SUMMARIES_SHA256)`; and `write_export_provenance(table_path,
    arm_spec, **entries) -> Path`.

- [x] **Step 1: Write the failing tests**

In `tests/fixtures/decision.py`, replace:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.regressor_panel import text_only_table as stub_text_only_table
```

with:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.regressor_panel import text_only_table as stub_text_only_table
```

Replace:

```python
BACKBONE = 'tiny-backbone'
REVISION = 'abc123'
DESCRIPTIONS_SHA256 = 'd' * 64
MAX_LENGTH = 16
PANEL_SET = PanelSet(
    outcome='outcome-roles',
```

with:

```python
BACKBONE = 'tiny-backbone'
REVISION = 'abc123'
DESCRIPTIONS_SHA256 = 'd' * 64
SUMMARIES_SHA256 = 'e' * 64
MAX_LENGTH = 16
PANEL_SET = PanelSet(
    outcome='outcome-roles',
```

Replace:

```python
        'backbone': BACKBONE,
        'backbone_revision': REVISION,
        'descriptions_sha256': DESCRIPTIONS_SHA256,
        'max_length': MAX_LENGTH,
        **overrides,
    }
```

with:

```python
        'backbone': BACKBONE,
        'backbone_revision': REVISION,
        'descriptions_sha256': DESCRIPTIONS_SHA256,
        'summaries_sha256': SUMMARIES_SHA256,
        'max_length': MAX_LENGTH,
        **overrides,
    }
```

Replace:

```python
    return pl.concat(parts).cast(SCORE_SCHEMA)

def write_text_only(
    directory: Path, revision: str = REVISION, codes: Sequence[str] = CODES
) -> Path:
    '''A text-only table with its provenance, as ``tools text-only-table`` writes them.'''
```

with:

```python
    return pl.concat(parts).cast(SCORE_SCHEMA)

def write_text_only(
    directory: Path,
    revision: str = REVISION,
    codes: Sequence[str] = CODES,
    summaries: Optional[str] = SUMMARIES_SHA256,
) -> Path:
    '''A text-only table with its provenance, as ``tools text-only-table`` writes them.'''
```

Replace:

```python
            'path': 'naics_descriptions.parquet',
            'sha256': DESCRIPTIONS_SHA256
        },
        'max_length': MAX_LENGTH,
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': text_only_fingerprint(table),
```

with:

```python
            'path': 'naics_descriptions.parquet',
            'sha256': DESCRIPTIONS_SHA256
        },
        'summaries': summaries,
        'max_length': MAX_LENGTH,
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': text_only_fingerprint(table),
```

Replace:

```python
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def _table(directory: Path, name: str, seed: int, dimension: int) -> Path:
    '''An arm's code table in the export form.'''
```

with:

```python
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def write_export_provenance(table_path: Path, arm_spec: ArmSpec, **entries) -> Path:
    '''
    The provenance ``tools export-table`` writes beside a seed's table, recording what
    ``arm_spec`` reads; ``entries`` replace its entries.
    '''

    provenance = {
        'backbone': arm_spec.backbone,
        'revision': arm_spec.backbone_revision,
        'descriptions': {
            'path': 'naics_descriptions.parquet',
            'sha256': arm_spec.descriptions_sha256
        },
        'summaries': arm_spec.summaries_sha256,
        'max_length': arm_spec.max_length,
        'table_sha256': sha256_file(table_path),
        'matrix_fingerprint': table_fingerprint(pl.read_parquet(table_path)),
        **entries,
    }
    path = provenance_path(table_path)
    path.write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def _table(directory: Path, name: str, seed: int, dimension: int) -> Path:
    '''An arm's code table in the export form.'''
```

In `tests/unit/test_decision_rule.py`, replace:

```python
        backbone='b',
        backbone_revision='r',
        descriptions_sha256='d',
        max_length=8,
    )
```

with:

```python
        backbone='b',
        backbone_revision='r',
        descriptions_sha256='d',
        summaries_sha256=None,
        max_length=8,
    )
```

In `tests/unit/test_decision_store.py`, replace:

```python
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import CODES, spec, synthetic_arm, write_text_only
from tests.fixtures.regressor_panel import coordinate_table

pytestmark = pytest.mark.unit
```

with:

```python
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import (
    CODES,
    SUMMARIES_SHA256,
    spec,
    synthetic_arm,
    write_text_only,
)
from tests.fixtures.regressor_panel import coordinate_table

pytestmark = pytest.mark.unit
```

Replace:

```python
    assert (reference.backbone, reference.revision, reference.max_length) == (
        provenance['backbone'], provenance['revision'], provenance['max_length']
    )

def test_a_text_only_table_needs_the_provenance_that_describes_it(store, tmp_path):
    path = write_text_only(tmp_path / 'text')
```

with:

```python
    assert (reference.backbone, reference.revision, reference.max_length) == (
        provenance['backbone'], provenance['revision'], provenance['max_length']
    )
    assert reference.summaries_sha256 == provenance['summaries'] == SUMMARIES_SHA256

def test_a_text_only_table_needs_the_provenance_that_describes_it(store, tmp_path):
    path = write_text_only(tmp_path / 'text')
```

Replace:

```python
    provenance.pop('max_length')
    provenance_path(path).write_text(json.dumps(provenance))

def _as_a_list(path, provenance):
    provenance_path(path).write_text(json.dumps([provenance]))
```

with:

```python
    provenance.pop('max_length')
    provenance_path(path).write_text(json.dumps(provenance))

def _without_summaries(path, provenance):
    # A text-only table built before Stage 6b
    provenance.pop('summaries')
    provenance_path(path).write_text(json.dumps(provenance))

def _as_a_list(path, provenance):
    provenance_path(path).write_text(json.dumps([provenance]))
```

Replace:

```python
    [
        (_shortened, 'describes another file'),
        (_without_max_length, "lacks the field 'max_length'"),
        (_as_a_list, 'is not a JSON object'),
        (_descriptions_as_text, 'descriptions field is not a JSON object'),
    ],
```

with:

```python
    [
        (_shortened, 'describes another file'),
        (_without_max_length, "lacks the field 'max_length'"),
        (_without_summaries, "lacks the field 'summaries'"),
        (_as_a_list, 'is not a JSON object'),
        (_descriptions_as_text, 'descriptions field is not a JSON object'),
    ],
```

In `tests/unit/test_decision.py`, replace:

```python
    with pytest.raises(ValueError, match='D9'):
        _decide([arm, reference], margins, store)

def test_the_text_only_check_reads_the_stored_provenance(store, tmp_path, reference, margins):
    stale = write_text_only(tmp_path / 'stale', revision='an-older-revision')
    # The record's copy claims the arm's revision; the stored provenance names the older one
```

with:

```python
    with pytest.raises(ValueError, match='D9'):
        _decide([arm, reference], margins, store)

def test_the_text_only_table_must_read_the_arms_summaries(store, tmp_path, reference, margins):
    # Built before the arm's summaries: on truncated text
    stale = write_text_only(tmp_path / 'stale', summaries=None)
    arm = synthetic_arm(store, tmp_path, spec('stale'), {}, text_only_table=stale)

    with pytest.raises(ValueError, match="'e{64}'.*D9"):
        _decide([arm, reference], margins, store)

def test_the_text_only_check_reads_the_stored_provenance(store, tmp_path, reference, margins):
    stale = write_text_only(tmp_path / 'stale', revision='an-older-revision')
    # The record's copy claims the arm's revision; the stored provenance names the older one
```

In `tests/unit/test_decision_sweep.py`, replace:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import DECISION_LEVEL, RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.schema import IndexRole
from tests.fixtures.decision import spec, write_text_only
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, SIX_DIGIT

pytestmark = pytest.mark.unit
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import DECISION_LEVEL, RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.schema import IndexRole
from tests.fixtures.decision import spec, write_export_provenance, write_text_only
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, SIX_DIGIT

pytestmark = pytest.mark.unit
```

Replace:

```python
        return self._axes(codes) + torch.from_numpy(noise)

class SyntheticRunner:
    '''One seed of the informed or the uninformed arm, written under ``directory``.'''

    def __init__(self, directory, informed, signal):
        self.directory = directory
        self.informed = informed
        self.signal = signal

    def run(self, arm_spec, seed):
        rng = np.random.default_rng([seed, int(self.informed), 7])
```

with:

```python
        return self._axes(codes) + torch.from_numpy(noise)

class SyntheticRunner:
    '''
    One seed of the informed or the uninformed arm, written under ``directory``.

    Each seed's table has the export provenance ``tools export-table`` writes, recording what the
    spec reads; ``provenance`` replaces entries of it.
    '''

    def __init__(self, directory, informed, signal, provenance=None):
        self.directory = directory
        self.informed = informed
        self.signal = signal
        self.provenance = provenance or {}

    def run(self, arm_spec, seed):
        rng = np.random.default_rng([seed, int(self.informed), 7])
```

Replace:

```python
        }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
        path = self.directory / f'{arm_spec.name}-{seed}.parquet'
        table.write_parquet(path)
        return SeedArtifacts(
            checkpoint=checkpoint,
            table=path,
```

with:

```python
        }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
        path = self.directory / f'{arm_spec.name}-{seed}.parquet'
        table.write_parquet(path)
        write_export_provenance(path, arm_spec, **self.provenance)
        return SeedArtifacts(
            checkpoint=checkpoint,
            table=path,
```

Replace:

```python
        )
    assert log.records() == []

def test_a_repeated_seed_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):
```

with:

```python
        )
    assert log.records() == []

@pytest.mark.parametrize(
    'entry',
    [
        {
            'backbone': 'another/backbone'
        },
        {
            'revision': 'another-revision'
        },
        {
            'descriptions': {
                'path': 'naics_descriptions.parquet',
                'sha256': 'f' * 64
            }
        },
        # Exported before Stage 6b, from truncated text
        {
            'summaries': None
        },
        {
            'max_length': 32
        },
    ],
)
def test_a_seed_exported_from_other_text_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log, entry
):
    runner = SyntheticRunner(tmp_path / 'runs' / 'misread', False, _signal(regressor_rows), entry)

    with pytest.raises(ValueError, match='misread seed 0: the table was exported from .*D9'):
        run_seed_sweep(
            spec('misread', dimension=3),
            SEEDS,
            runner,
            text_only_table=text_only,
            store=store,
            purpose=PURPOSE,
            **panels,
        )
    assert log.records() == []

def test_a_seed_table_without_its_export_provenance_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):

    class Unexported(SyntheticRunner):

        def run(self, arm_spec, seed):
            artifacts = super().run(arm_spec, seed)
            provenance_path(artifacts.table).unlink()
            return artifacts

    runner = Unexported(tmp_path / 'runs' / 'unexported', False, _signal(regressor_rows))

    with pytest.raises(ValueError, match='has no export provenance'):
        run_seed_sweep(
            spec('unexported', dimension=3),
            SEEDS,
            runner,
            text_only_table=text_only,
            store=store,
            purpose=PURPOSE,
            **panels,
        )
    assert log.records() == []

def test_a_repeated_seed_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision.py tests/unit/test_decision_rule.py tests/unit/test_decision_store.py tests/unit/test_decision_sweep.py -q`
Expected: `18 failed, 22 passed, 36 errors`. Most of them fail on `ValidationError: 1 validation
error for ArmSpec`, `summaries_sha256`, "Extra inputs are not permitted".

- [x] **Step 3: Record the summaries in the records and the store**

In `src/naics_embedder/decision/records.py`, replace:

```python
    backbone: str
    revision: Optional[str]
    descriptions_sha256: str
    max_length: int

# -------------------------------------------------------------------------------------------------
```

with:

```python
    backbone: str
    revision: Optional[str]
    descriptions_sha256: str
    # The window-fitting summaries the table read; null for a backbone with no pin
    summaries_sha256: Optional[str]
    max_length: int

# -------------------------------------------------------------------------------------------------
```

Replace:

```python
    Attributes:
        components: Stages or post-processing steps: Req 5's "fewer components".
        backbone, backbone_revision, descriptions_sha256, max_length: What the arm's encoder
            reads, which its text-only table must match (D9).
        settings: The configuration's own settings, recorded as given.
    '''
```

with:

```python
    Attributes:
        components: Stages or post-processing steps: Req 5's "fewer components".
        backbone, backbone_revision, descriptions_sha256, summaries_sha256, max_length: What
            the arm's encoder reads, which its text-only table and each seed's table must match
            (D9). ``summaries_sha256`` is null for a backbone with no pinned summaries.
        settings: The configuration's own settings, recorded as given.
    '''
```

Replace:

```python
    backbone: str
    backbone_revision: Optional[str]
    descriptions_sha256: str
    max_length: int = Field(ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)
```

with:

```python
    backbone: str
    backbone_revision: Optional[str]
    descriptions_sha256: str
    summaries_sha256: Optional[str]
    max_length: int = Field(ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)
```

In `src/naics_embedder/decision/store.py`, replace:

```python
    provenance: Mapping[str, Any], table_sha256: str, matrix_fingerprint: str, name: str
) -> Dict[str, Any]:
    '''
    The fields D9 checks, from a text-only table's provenance that describes the table.

    Args:
        provenance: The provenance's JSON content.
```

with:

```python
    provenance: Mapping[str, Any], table_sha256: str, matrix_fingerprint: str, name: str
) -> Dict[str, Any]:
    '''
    The fields D9 checks, from a table's provenance that describes the table.

    A text-only table's provenance (``tools text-only-table``) and an exported table's
    (``tools export-table``) both carry them.

    Args:
        provenance: The provenance's JSON content.
```

Replace:

```python
            'backbone': provenance['backbone'],
            'revision': provenance['revision'],
            'descriptions_sha256': provenance['descriptions']['sha256'],
            'max_length': provenance['max_length'],
        }
    except KeyError as exc:
```

with:

```python
            'backbone': provenance['backbone'],
            'revision': provenance['revision'],
            'descriptions_sha256': provenance['descriptions']['sha256'],
            'summaries_sha256': provenance['summaries'],
            'max_length': provenance['max_length'],
        }
    except KeyError as exc:
```

- [x] **Step 4: Check five fields, for the text-only table and each seed's table**

In `src/naics_embedder/decision/decide.py`, replace:

```python
# -------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
```

with:

```python
# -------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl
```

Replace:

```python
# Guards
# -------------------------------------------------------------------------------------------------

def check_text_only(spec: ArmSpec, text_only: TextOnlyRef) -> None:
    '''
    Require the text-only table to come from the arm's backbone reading the arm's text (D9).

    Raises:
        ValueError: If the provenance's backbone, revision, descriptions sha256 or window
            differs from the arm's.
    '''

    built = (
        text_only.backbone, text_only.revision, text_only.descriptions_sha256, text_only.max_length
    )
    reads = (spec.backbone, spec.backbone_revision, spec.descriptions_sha256, spec.max_length)
    if built != reads:
        raise ValueError(
            f'{spec.name}: the text-only table was built from {built}, the arm reads {reads} '
            "(D9: the arm's own backbone reading the arm's text)"
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
    Require an arm record to be complete, intact and read as the decision reads it.
```

with:

```python
# Guards
# -------------------------------------------------------------------------------------------------

# What an arm reads, as provenance_fields names it (decision/store.py)
D9_FIELDS = ('backbone', 'revision', 'descriptions_sha256', 'summaries_sha256', 'max_length')

def _arm_reads(spec: ArmSpec) -> Tuple[Any, ...]:
    return (
        spec.backbone,
        spec.backbone_revision,
        spec.descriptions_sha256,
        spec.summaries_sha256,
        spec.max_length,
    )

def check_text_only(spec: ArmSpec, text_only: TextOnlyRef) -> None:
    '''
    Require the text-only table to come from the arm's backbone reading the arm's text (D9).

    Raises:
        ValueError: If the provenance's backbone, revision, descriptions sha256, summaries sha256
            or window differs from the arm's.
    '''

    fields = text_only.model_dump()
    built = tuple(fields[name] for name in D9_FIELDS)
    reads = _arm_reads(spec)
    if built != reads:
        raise ValueError(
            f'{spec.name}: the text-only table was built from {built}, the arm reads {reads} '
            "(D9: the arm's own backbone reading the arm's text)"
        )

def check_seed_table(spec: ArmSpec, seed: int, fields: Mapping[str, Any]) -> None:
    '''
    Require a seed's table to have been exported from what the arm reads (D9).

    Args:
        spec: The arm.
        seed: The seed, which names the refusal.
        fields: The table's export provenance, as ``provenance_fields`` reads it.

    Raises:
        ValueError: If the backbone, revision, descriptions sha256, summaries sha256 or window
            differs from the arm's.
    '''

    exported = tuple(fields[name] for name in D9_FIELDS)
    reads = _arm_reads(spec)
    if exported != reads:
        raise ValueError(
            f'{spec.name} seed {seed}: the table was exported from {exported}, the arm reads '
            f'{reads} (D9)'
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
    Require an arm record to be complete, intact and read as the decision reads it.
```

In `src/naics_embedder/decision/sweep.py`, replace:

```python
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Union

import polars as pl

from naics_embedder.decision.decide import check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import DECISION_LEVEL, ArmTables, Regime, RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.schema import IndexRole

logger = logging.getLogger(__name__)
```

with:

```python
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Union

import polars as pl

from naics_embedder.decision.decide import check_seed_table, check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
from naics_embedder.decision.store import ArtifactStore, provenance_fields
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    ArmTables,
    Regime,
    RegressorPanel,
    table_fingerprint,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.supervision.schema import IndexRole

logger = logging.getLogger(__name__)
```

Replace:

```python
    Attributes:
        checkpoint: The encoder checkpoint file.
        table: The 2,125-code table in the export form (tangent coordinates if hyperbolic).
        encoder: Queries and codes embedded in one space, for the outcome panel.
        distance: The arm's decoding distance (``panels.decoding.DISTANCES``).
    '''
```

with:

```python
    Attributes:
        checkpoint: The encoder checkpoint file.
        table: The 2,125-code table in the export form (tangent coordinates if hyperbolic), with
            its export provenance beside it.
        encoder: Queries and codes embedded in one space, for the outcome panel.
        distance: The arm's decoding distance (``panels.decoding.DISTANCES``).
    '''
```

Replace:

```python
# Driver
# -------------------------------------------------------------------------------------------------

def _fit_settings(panel: RegressorPanel) -> Dict[str, Any]:
    settings = asdict(panel.settings)
    return {**settings, 'alphas': list(settings['alphas'])}
```

with:

```python
# Driver
# -------------------------------------------------------------------------------------------------

def _seed_table_fields(table: Path) -> Mapping[str, Any]:
    '''
    The D9 fields of a seed's table, from the export provenance beside it.

    Raises:
        ValueError: If the provenance is missing, or as ``provenance_fields``.
    '''

    provenance = provenance_path(table)
    if not provenance.is_file():
        raise ValueError(f'{table} has no export provenance at {provenance}')
    return provenance_fields(
        json.loads(provenance.read_text(encoding='utf-8')),
        sha256_file(table),
        table_fingerprint(pl.read_parquet(table)),
        str(provenance),
    )

def _fit_settings(panel: RegressorPanel) -> Dict[str, Any]:
    settings = asdict(panel.settings)
    return {**settings, 'alphas': list(settings['alphas'])}
```

Replace:

```python
    Run a configuration for each seed and read each seed once on each panel's validation split.

    Raises:
        ValueError: If a seed repeats, the text-only table was not built from the arm's
            backbone, revision, descriptions and window (D9), or a seed's table width is not
            the arm spec's dimension.
    '''

    if len(set(seeds)) != len(seeds):
```

with:

```python
    Run a configuration for each seed and read each seed once on each panel's validation split.

    Raises:
        ValueError: If a seed repeats; if the text-only table, or a seed's table by its export
            provenance, was not built from the arm's backbone, revision, descriptions, summaries
            and window (D9); or if a seed's table width is not the arm spec's dimension.
    '''

    if len(set(seeds)) != len(seeds):
```

Replace:

```python
    runs: List[SeedRun] = []
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
        table = store.put_table(artifacts.table)
```

with:

```python
    runs: List[SeedRun] = []
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        # What the seed read, before anything is stored or any panel is read
        check_seed_table(spec, seed, _seed_table_fields(Path(artifacts.table)))
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
        table = store.put_table(artifacts.table)
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision.py tests/unit/test_decision_rule.py tests/unit/test_decision_store.py tests/unit/test_decision_sweep.py tests/unit/test_cli_commands.py -q`
Expected: `118 passed`. `test_cli_commands.py` builds its decision records from the same
fixtures.

- [x] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/decision/records.py src/naics_embedder/decision/store.py src/naics_embedder/decision/decide.py src/naics_embedder/decision/sweep.py tests/fixtures/decision.py tests/unit/test_decision.py tests/unit/test_decision_rule.py tests/unit/test_decision_store.py tests/unit/test_decision_sweep.py`
Run: `uv run pytest -n auto -q`
Expected: `1933 passed, 1 skipped`.

```bash
git add src/naics_embedder/decision/records.py src/naics_embedder/decision/store.py src/naics_embedder/decision/decide.py src/naics_embedder/decision/sweep.py tests/fixtures/decision.py tests/unit/test_decision.py tests/unit/test_decision_rule.py tests/unit/test_decision_store.py tests/unit/test_decision_sweep.py
git commit -m "feat(decision): records and every seed's table carry the summaries' sha256 (D9)"
```

### Task 9: Documentation

§8's edits, plus `docs/api/input_window.md` (P15). The docstrings that said every tokenizing path
truncates now say that an over-window channel text reads as its summary, and that truncation
stays a backstop for queries. `InputWindowRecord`'s counts stay the bundle's record. The new API
page documents both modules, and the strict docs build checks every docstring the earlier tasks
wrote.

**Files:**
- Modify: `CLAUDE.md:67-72, 105-110, 175-180, 449-454`
- Modify: `docs/.nav.yml:17-22`
- Modify: `docs/api/input_window.md:1-6`
- Create: `docs/api/window_summaries.md`
- Modify: `docs/text_training.md:69-76, 334-340, 414-421, 437-442`
- Modify: `docs/usage.md:95-100, 211-218, 268-278`
- Modify: `src/naics_embedder/supervision/schema.py:179-186`
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py:42-48`
- Modify: `src/naics_embedder/utils/input_window.py:6-13`

**Interfaces:**
- Consumes: the modules of Tasks 1–8, whose docstrings mkdocstrings renders.
- Produces: no code.

- [x] **Step 1: Correct the docstrings**

> Deviation: by the user's ruling on the Task 9 review (7b52cce), the `InputWindowRecord` docstring in `schema.py` adds that the manifest's counts are measured on unmarked text, so they are lower than the readers'.

In `src/naics_embedder/text_model/dataloader/tokenization_cache.py`, replace:

```python
    max_length: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    '''
    Tokenize one channel text with its field marker, truncated and padded to ``max_length``.

    An absent channel (null or blank) is encoded as the empty string, ``[CLS] [SEP]``, with no
    marker and never as placeholder text, and its ``present`` flag is False so fusion can mask it
```

with:

```python
    max_length: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    '''
    Tokenize one channel text with its field marker, padded to ``max_length``.

    The text fits: ``_build_tokenization_cache`` has replaced an over-window text by its
    window-fitting summary, so truncation, which ``fields.tokenize_field`` keeps for queries, never
    shortens a channel text.

    An absent channel (null or blank) is encoded as the empty string, ``[CLS] [SEP]``, with no
    marker and never as placeholder text, and its ``present`` flag is False so fusion can mask it
```

In `src/naics_embedder/supervision/schema.py`, replace:

```python
    '''
    The backbone's trained input window, and each text channel's texts beyond it (Req 9).

    The window comes from the backbone's own documentation (``utils/input_window.py``), and
    every tokenizing path truncates to it, so these counts are the texts truncation shortens.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')
```

with:

```python
    '''
    The backbone's trained input window, and each text channel's texts beyond it (Req 9).

    The window comes from the backbone's own documentation (``utils/input_window.py``). The
    counts are the bundle's record of the texts beyond it, which readers since Stage 6b read as
    their window-fitting summaries (``panels/window_summaries.py``), never truncated.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')
```

In `src/naics_embedder/utils/input_window.py`, replace:

```python
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 says that in training "the sequence length was limited
to 128 tokens". Its 256 (``sentence_bert_config.json``'s ``max_seq_length``, the truncation it
applies at inference) and 512 (``config.json``'s ``max_position_embeddings``) are not the trained
window. Every tokenizing path truncates to the window and refuses a longer ``max_length``, and
the supervision bundle records each channel's share of texts beyond it.
'''

# -------------------------------------------------------------------------------------------------
```

with:

```python
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 says that in training "the sequence length was limited
to 128 tokens". Its 256 (``sentence_bert_config.json``'s ``max_seq_length``, the truncation it
applies at inference) and 512 (``config.json``'s ``max_position_embeddings``) are not the trained
window. Every tokenizing path refuses a ``max_length`` beyond the window. A channel text over it
is read as its window-fitting summary (``panels/window_summaries.py``), so no channel text is
truncated, and truncation stays a backstop for queries. The supervision bundle records each
channel's share of texts beyond the window.
'''

# -------------------------------------------------------------------------------------------------
```

- [x] **Step 2: The text-training guide**

> Deviation: by the user's ruling on the Task 9 review (7b52cce), the guide names description, examples and excluded texts rather than "channel text", says the resolver checks the artifact on every call that finds an over-window text, and adds that the manifest counts unmarked text (153, 105 and 464).

In `docs/text_training.md`, replace:

```markdown
  `'excluded: …'`, `'examples: …'` or `'query: …'` (`text_model/fields.py`). An absent text
  (null or blank) is the unmarked empty string with `present` False. The tokenization cache's
  format `channels-v3` stores the marked texts. Its sidecar records the markers and a `summaries`
  entry, null until Stage 6b. A cache built under another format, other markers or other
  summaries is rebuilt.
- **Present channels only.** Each field's present texts go through the backbone in calls of at most
  256 texts (`MAX_TEXTS_PER_CALL`), each trimmed to its own longest text, and absent texts never
  enter it. Each present text is mean-pooled over its tokens.
```

with:

```markdown
  `'excluded: …'`, `'examples: …'` or `'query: …'` (`text_model/fields.py`). An absent text
  (null or blank) is the unmarked empty string with `present` False. The tokenization cache's
  format `channels-v3` stores the marked texts. Its sidecar records the markers and a `summaries`
  entry, the sha256 of the pinned window-fitting summaries (see "Input window" below). A cache
  built under another format, other markers or other summaries is rebuilt, and a load that finds
  a stale sidecar names each entry that differs.
- **Present channels only.** Each field's present texts go through the backbone in calls of at most
  256 texts (`MAX_TEXTS_PER_CALL`), each trimmed to its own longest text, and absent texts never
  enter it. Each present text is mean-pooled over its tokens.
```

Replace:

```markdown
**Input window.** The manifest's `input_window` records the backbone's trained window: 128 tokens
for `sentence-transformers/all-MiniLM-L6-v2`, from its model card. Per text channel it records
the present texts, the texts beyond the window and their share. Every tokenizing path truncates
to the window (`utils/input_window.py`). An absent channel is null in the descriptions, and the
tokenization cache encodes it as the empty string, never as a placeholder.

### Three Independent Axes
```

with:

```markdown
**Input window.** The manifest's `input_window` records the backbone's trained window: 128 tokens
for `sentence-transformers/all-MiniLM-L6-v2`, from its model card. Per text channel it records
the present texts, the texts beyond the window and their share. A channel text whose marked form
is over the window is read as its window-fitting summary (roadmap Stage 6b): whole sentences,
clauses or examples entries of the text, chosen once by `naics-embedder data summaries` and
committed as `conf/data/window_summaries.csv`. `WINDOW_SUMMARIES` (`panels/window_summaries.py`)
pins the artifact by sha256, and the tokenization cache and the text-only comparator both read
their texts through `resolve_channel_texts`, which checks the artifact on every call. No channel
text is truncated; truncation stays a backstop for queries. Under MiniLM at 128 tokens, 162
descriptions, 106 examples texts and 485 exclusion texts are summarized. The checkpoint contract,
the export and text-only provenances and the decision records carry the summaries' sha256. An
absent channel is null in the descriptions, and the tokenization cache encodes it as the empty
string, never as a placeholder.

### Three Independent Axes
```

Replace:

```markdown
### Cache Regeneration

- **Tokenization cache** — reused only when its JSON sidecar (`<cache>.meta.json`) records the
  bundle's description and codebook fingerprints, tokenizer, and max length; otherwise it is
  rebuilt.
- **Streaming and multi-epoch caches** — stored in a versioned envelope keyed by contract, bundle
  ID, codebook fingerprint, and source-artifact fingerprints; caches from other bundles or legacy
  runs are rejected and regenerated.
```

with:

```markdown
### Cache Regeneration

- **Tokenization cache** — reused only when its JSON sidecar (`<cache>.meta.json`) records the
  bundle's description and codebook fingerprints, tokenizer, max length, cache format, field
  markers and window-fitting summaries; otherwise it is rebuilt.
- **Streaming and multi-epoch caches** — stored in a versioned envelope keyed by contract, bundle
  ID, codebook fingerprint, and source-artifact fingerprints; caches from other bundles or legacy
  runs are rejected and regenerated.
```

Replace:

```markdown
Every new checkpoint records its supervision contract under `stage3_supervision`: supervision
mode, contract version, bundle ID, codebook fingerprint, structural-preference-loss version,
mining-contract version, and the encoder architecture (layout, fusion, dimension and backbone).
Structural matrices are loaded from the validated bundle, not trusted from checkpoint state.

- **`--checkpoint-load-mode exact`** (default) restores optimizer, scheduler, epoch, global step,
```

with:

```markdown
Every new checkpoint records its supervision contract under `stage3_supervision`: supervision
mode, contract version, bundle ID, codebook fingerprint, structural-preference-loss version,
mining-contract version, the encoder architecture (layout, fusion, dimension and backbone), and
the sha256 of the window-fitting summaries its token cache applied (`summaries`). A checkpoint
trained before Stage 6b, on truncated text, records none and reads as null, so exact resume,
export, reads and the HGCN feeder refuse it; weights-only migration does not compare it.
Structural matrices are loaded from the validated bundle, not trusted from checkpoint state.

- **`--checkpoint-load-mode exact`** (default) restores optimizer, scheduler, epoch, global step,
```

- [x] **Step 3: The usage guide**

> Deviation: by the user's ruling on the Task 9 review (7b52cce), the guide names description, examples and excluded texts rather than "channel text", and says the decision store records the summaries' sha256 and `check_text_only` compares it.

In `docs/usage.md`, replace:

````markdown
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

### `data regressor-groups`

Draw the regressor panel's held-out four-digit groups, once: a fifth of each sector's groups
````

with:

````markdown
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

### `data summaries`

Build the window-fitting summaries of over-long channel texts, once per backbone (roadmap
Stage 6b). Every channel text whose marked form (`'description: …'`, special tokens included) is
over the backbone's trained window is summarized by whole units of the text: its sentences, or
clauses and segmenter pieces of an over-long sentence, or its examples entries. The frozen
backbone (from the local Hugging Face cache) keeps, greedily, the units whose pooled vectors best
approximate the whole text's, in source order, while the marked summary fits. Every unit boundary
is a boundary of the leakage segmenter, so a summary's segments are a subset of its text's and
the leakage checks need no sealed read. The artifact is checked as every reader checks it before
it is moved into place; commit it with the pin the command prints, in `WINDOW_SUMMARIES`
(`panels/window_summaries.py`). An existing artifact is replaced only with `--force`, and a new
artifact needs a new pin.

**Generates:** `conf/data/window_summaries.csv`, `conf/data/window_summaries_provenance.json`

```bash
HF_HUB_OFFLINE=1 uv run naics-embedder data summaries
```

**Options:**
- `--descriptions PATH` - The descriptions parquet (default: `./data/naics_descriptions.parquet`)
- `--backbone NAME` - The backbone (default: `data_loader.tokenization.tokenizer_name` in
  `conf/config.yaml`)
- `--output PATH` - The artifact (default: `conf/data/window_summaries.csv`)
- `--force` - Replace an existing artifact

### `data regressor-groups`

Draw the regressor panel's held-out four-digit groups, once: a fifth of each sector's groups
````

Replace:

```markdown
Embed every code's text with the arm's backbone, frozen: each of the four channels is mean-pooled
over its tokens, and a code's vector is the mean of its present channels (roadmap D9). The
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). Each channel is truncated to the backbone's trained input
window, and `text_only.max_length` may not exceed it (Req 9). The regressor panel reduces the
table to the arm's dimension by PCA.

**Generates:** the table and `<stem>_provenance.json` beside it
```

with:

```markdown
Embed every code's text with the arm's backbone, frozen: each of the four channels is mean-pooled
over its tokens, and a code's vector is the mean of its present channels (roadmap D9). The
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). A channel text over the backbone's trained input window is
read as its window-fitting summary, as the arm reads it, so no channel text is truncated, and
`text_only.max_length` may not exceed the window (Req 9). The provenance records the summaries'
sha256 (`summaries`), which the decision store checks against the arm's (D9). The regressor panel
reduces the table to the arm's dimension by PCA.

**Generates:** the table and `<stem>_provenance.json` beside it
```

Replace:

````markdown
The command resolves the bundle and the token cache as `train` does, from `--config` and
`key=value` overrides. The checkpoint's supervision contract must match the bundle. Its encoder
record is its own, so a d = 8 checkpoint exports under a d = 16 config. A checkpoint trained at a
curvature other than 1, or of the four-copy encoder (roadmap D2), is refused.

**Generates:** the table and `<stem>_provenance.json` beside it. The provenance records the
checkpoint's sha256 and contract, the backbone's revision, the window, the descriptions' sha256,
`summaries`, and the table's sha256 and `matrix_fingerprint`.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt \
````

with:

````markdown
The command resolves the bundle and the token cache as `train` does, from `--config` and
`key=value` overrides. The checkpoint's supervision contract must match the bundle. Its encoder
record is its own, so a d = 8 checkpoint exports under a d = 16 config. A checkpoint trained at a
curvature other than 1, or of the four-copy encoder (roadmap D2), is refused. So is a checkpoint
trained under other window-fitting summaries than the tokenizer's pin, such as one trained
before Stage 6b on truncated text: the refusal names `summaries`.

**Generates:** the table and `<stem>_provenance.json` beside it. The provenance records the
checkpoint's sha256 and contract, the backbone's revision, the tokenizer, the window, the
descriptions' sha256, `summaries`, and the table's sha256 and `matrix_fingerprint`. A read
(`tools outcome-panel`) refuses a table whose provenance records no `summaries` or `tokenizer`,
or other ones than its own.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt \
````

- [x] **Step 4: The API pages and the navigation**

Create `docs/api/window_summaries.md`:

```markdown
# Window-Fitting Summaries API

Extractive summaries of the channel texts over the backbone's trained window (Req 9, roadmap
Stage 6b): the units and the resolver every reader goes through, and the build behind
`naics-embedder data summaries`.

## Units, pin and resolver

::: naics_embedder.panels.window_summaries

## The build

::: naics_embedder.data.window_summaries
```

In `docs/api/input_window.md`, replace:

```markdown
# Input Window API

The backbone's trained input window (Req 9, roadmap Stage 5): every tokenizing path truncates to
it, and the supervision bundle records each channel's texts beyond it.

::: naics_embedder.utils.input_window
```

with:

```markdown
# Input Window API

The backbone's trained input window (Req 9, roadmap Stage 5): no tokenizing path reads beyond
it, a channel text over it is read as its window-fitting summary (roadmap Stage 6b), and the
supervision bundle records each channel's texts beyond it.

::: naics_embedder.utils.input_window
```

In `docs/.nav.yml`, replace:

```yaml
              - Graph Distance Measures: api/compute_distances.md
          - Create Contrastive Training Triplets: api/create_triplets.md
          - Redirection Table: api/redirections.md
      - Training:
          - Text Training:
              - Data Loading:
```

with:

```yaml
              - Graph Distance Measures: api/compute_distances.md
          - Create Contrastive Training Triplets: api/create_triplets.md
          - Redirection Table: api/redirections.md
          - Window-Fitting Summaries: api/window_summaries.md
      - Training:
          - Text Training:
              - Data Loading:
```

- [x] **Step 5: CLAUDE.md**

In `CLAUDE.md`, replace:

```markdown
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── redirections.py   # The redirection table (Req 8) and the exclusion channel
│   │   ├── index_role_table.py    # Draw the frozen index-entry role table (data roles)
│   │   ├── regressor_group_table.py  # Draw the regressor held-out groups (data regressor-groups)
│   │   ├── compute_relations.py   # Compute relationship measures
```

with:

```markdown
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── redirections.py   # The redirection table (Req 8) and the exclusion channel
│   │   ├── window_summaries.py    # Build the window-fitting summaries (data summaries)
│   │   ├── index_role_table.py    # Draw the frozen index-entry role table (data roles)
│   │   ├── regressor_group_table.py  # Draw the regressor held-out groups (data regressor-groups)
│   │   ├── compute_relations.py   # Compute relationship measures
```

Replace:

```markdown
│   │   ├── selection_log.py  # Append-only log of split reads and test-split openings
│   │   ├── outcome.py        # OutcomePanel: sealed validation and test query splits
│   │   ├── lexical_encoder.py  # Training-free trigram stub encoder
│   │   ├── qcew_rows.py      # QCEW national rows: cells, population, dated rows (D7)
│   │   ├── regressor_splits.py  # The regressor partition and its committed held-out draw
│   │   ├── ridge.py          # Ridge on standardized features along a penalty grid
```

with:

```markdown
│   │   ├── selection_log.py  # Append-only log of split reads and test-split openings
│   │   ├── outcome.py        # OutcomePanel: sealed validation and test query splits
│   │   ├── lexical_encoder.py  # Training-free trigram stub encoder
│   │   ├── window_summaries.py  # Window-fitting summaries: units, the pin, the resolver (Req 9)
│   │   ├── qcew_rows.py      # QCEW national rows: cells, population, dated rows (D7)
│   │   ├── regressor_splits.py  # The regressor partition and its committed held-out draw
│   │   ├── ridge.py          # Ridge on standardized features along a penalty grid
```

Replace:

```markdown
│   │   ├── index_roles.csv        # The frozen index-entry role table (committed)
│   │   ├── regressor_panel.yaml   # QCEW pins, held-out draw, ridge grid, folds, branch record
│   │   ├── regressor_heldout_groups.csv  # The regressor panel's held-out groups (committed)
│   │   ├── decision.yaml          # Decision rule: bootstrap replicates and seed, seed floor
│   │   ├── relations.yaml
│   │   ├── distances.yaml
```

with:

```markdown
│   │   ├── index_roles.csv        # The frozen index-entry role table (committed)
│   │   ├── regressor_panel.yaml   # QCEW pins, held-out draw, ridge grid, folds, branch record
│   │   ├── regressor_heldout_groups.csv  # The regressor panel's held-out groups (committed)
│   │   ├── window_summaries.csv   # Summaries of over-window channel texts (committed, pinned)
│   │   ├── decision.yaml          # Decision rule: bootstrap replicates and seed, seed floor
│   │   ├── relations.yaml
│   │   ├── distances.yaml
```

Replace:

```markdown
# (data relations / distances / triplets are deprecated and build nothing)
# (data roles drew conf/data/index_roles.csv once; it is committed, and preprocess applies it)
# (data regressor-groups drew conf/data/regressor_heldout_groups.csv once; it is committed)

# Training commands
uv run naics-embedder train            # Train model
```

with:

```markdown
# (data relations / distances / triplets are deprecated and build nothing)
# (data roles drew conf/data/index_roles.csv once; it is committed, and preprocess applies it)
# (data regressor-groups drew conf/data/regressor_heldout_groups.csv once; it is committed)
# (data summaries built conf/data/window_summaries.csv once; it is committed and pinned in code)

# Training commands
uv run naics-embedder train            # Train model
```

- [x] **Step 6: Check the docs**

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warning. `site/` is gitignored.

Run: `git grep -n -i 'null until' -- src tests docs CLAUDE.md README.md`
Expected: no output.

Run: `git grep -n -i -E 'truncates to (the|it)|is truncated to the' -- src docs CLAUDE.md README.md`
Expected: no output.

- [x] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/supervision/schema.py src/naics_embedder/utils/input_window.py`
Run: `uv run pytest -n auto -q`
Expected: `1933 passed, 1 skipped`.

```bash
git add src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/supervision/schema.py src/naics_embedder/utils/input_window.py docs/text_training.md docs/usage.md docs/api/window_summaries.md docs/api/input_window.md docs/.nav.yml CLAUDE.md
git commit -m "docs: over-window channel texts read as their summaries; data summaries"
```

### Task 10: The Exit (controller, inline)

This is spec §7, run locally. It builds the artifact on the real descriptions, commits it with its
pin, then shows the cache, a text-only table and the chain's refusals. It trains nothing and reads
no split, so the selection log gains no record. Each step names the §7 step it carries out.

Every command runs from the worktree root. `CLONE` below stands for the cloned bundle's manifest,
`data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`. It
goes into commands only as the `key=value` override `supervision.manifest_path=CLONE` (§7 step 2),
and is never committed. The commands below write it out in full.

The dry run of this task (on 2026-10-04, at this plan's code) gave the outputs quoted below. The
artifact's sha256 reproduced three times on this Mac. It is information only, because the
committed artifact is the authority, checked by its invariants (4.3). The counts are stop
conditions.

**Files:**
- Modify: `src/naics_embedder/panels/window_summaries.py:61-68`
- Create: `tests/unit/test_committed_window_summaries.py`
- Create (by `data summaries`): `conf/data/window_summaries.csv`,
  `conf/data/window_summaries_provenance.json`

**Interfaces:**
- Consumes: everything Tasks 1–9 built; the real inputs in **Workspace**.
- Produces:
  - The committed artifact, its provenance, and MiniLM's entry in `WINDOW_SUMMARIES`.
  - Gitignored evidence: the cache built on the summaries, and
    `checkpoints/plan9_exit/text_only.parquet` with its provenance.

- [x] **Step 1: Confirm the state, and clone the inputs (§7 steps 1–2)**

Run: `git status --short`
Expected: no output.

Run: `git log --oneline origin/main..HEAD`
Expected: the five commits of **Pre-flight** Step 1, then the commits of Tasks 1–9. No "config" or
"graph config".

Run: `ls logs/selection_log.jsonl conf/data/window_summaries.csv`
Expected: both "No such file or directory". If the log exists, stop and ask: it is append-only
(Req 4), so never delete or move one.

Run: `mkdir -p data/supervision/stage3-supervision-v2 checkpoints/plan8_exit checkpoints/plan9_exit logs`

Run: `cp -cR /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c data/supervision/stage3-supervision-v2/`

Run: `cp -c /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet data/naics_descriptions.parquet`

Run: `cp -c /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/last.ckpt /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/arm_table.parquet /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/arm_table_provenance.json /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/text_only.parquet /Users/lowell/Projects/naics-embedder/checkpoints/plan8_exit/text_only_provenance.json checkpoints/plan8_exit/`

`cp -c` makes APFS clones, so the copies cost no space and the originals stay untouched. Never
symlink, rebuild or edit them.

Run: `shasum -a 256 data/naics_descriptions.parquet`
Expected: `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.

Run: `uv run python -c "from naics_embedder.supervision.artifacts import load_validated_bundle; b = load_validated_bundle('data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json'); print(b.manifest.bundle_id, b.manifest.codebook_fingerprint[:8], b.manifest.description_fingerprint[:8])"`
Expected: `301cce28-539c-42ea-8781-496bbdcf511c 4662b826 fe8c54e3`. The load re-runs every
integrity and relational check.

- [x] **Step 2: Write the committed-artifact tests, and watch them fail**

Create `tests/unit/test_committed_window_summaries.py`:

```python
'''
The committed window-fitting summaries (roadmap Stage 6b): the artifact MiniLM's pin names, and,
where the real descriptions and the cached tokenizer are present, the resolver accepting it.

Every test here reads the committed pin, so the module opts out of the dummy-pin seam
(``tests/conftest.py``).
'''

import hashlib
import json
import logging
from pathlib import Path

import polars as pl
import pytest
from transformers import AutoTokenizer

from naics_embedder.panels import window_summaries
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import (
    SUMMARY_CHANNELS,
    WINDOW_SUMMARIES_PATH,
    read_window_summaries,
    resolve_channel_texts,
)

pytestmark = [pytest.mark.unit, pytest.mark.real_window_summaries]

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
DESCRIPTIONS = Path('data/naics_descriptions.parquet')

@pytest.fixture
def pin():
    return window_summaries.WINDOW_SUMMARIES[MINILM]

@pytest.fixture
def provenance(pin):
    return json.loads(provenance_path(Path(pin.path)).read_text())

def test_the_pin_names_the_committed_artifact(pin, provenance):
    assert pin.path == WINDOW_SUMMARIES_PATH
    assert hashlib.sha256(Path(pin.path).read_bytes()).hexdigest() == pin.sha256
    assert pin.window == 128
    assert provenance['artifact_sha256'] == pin.sha256
    assert (provenance['backbone'], provenance['window']) == (MINILM, pin.window)

def test_each_summarized_text_has_one_row_that_fits_the_window(pin):
    rows = read_window_summaries(Path(pin.path))

    assert rows.select('code', 'channel').is_duplicated().sum() == 0
    assert set(rows.get_column('channel').unique().to_list()) <= set(SUMMARY_CHANNELS)
    assert (rows.get_column('window') == pin.window).all()
    assert (rows.get_column('summary_tokens') <= pin.window).all()
    assert (rows.get_column('source_tokens') > pin.window).all()
    assert (rows.get_column('units_kept') <= rows.get_column('units_total')).all()

def test_the_provenance_counts_every_over_window_text_as_summarized(provenance):
    channels = provenance['channels']

    assert channels['title']['over_window'] == 0
    for channel in SUMMARY_CHANNELS:
        assert channels[channel]['summarized'] == channels[channel]['over_window'] > 0

def test_the_resolver_accepts_the_artifact_on_the_real_descriptions(pin, provenance, caplog):
    '''
    Local only: the resolver re-tokenizes every row and re-checks the source hashes, S5, the
    segment subset and the fit (spec 4.7).
    '''

    if not DESCRIPTIONS.is_file():
        pytest.skip(f'{DESCRIPTIONS} is not here')
    if hashlib.sha256(DESCRIPTIONS.read_bytes()).hexdigest() != provenance['descriptions']['sha256']:
        pytest.skip(f'{DESCRIPTIONS} is not the descriptions the summaries were built from')
    try:
        tokenizer = AutoTokenizer.from_pretrained(MINILM, local_files_only=True)
    except OSError:
        pytest.skip(f"{MINILM}'s tokenizer is not in the local Hugging Face cache")

    with caplog.at_level(logging.INFO, logger='naics_embedder.panels.window_summaries'):
        resolve_channel_texts(pl.read_parquet(DESCRIPTIONS), tokenizer, MINILM, pin.window)

    assert "{'description': 162, 'examples': 106, 'excluded': 485}" in caplog.text
```

Run: `uv run pytest tests/unit/test_committed_window_summaries.py -q`
Expected: `4 errors`, each at setup with `KeyError: 'sentence-transformers/all-MiniLM-L6-v2'`.
MiniLM has no pin yet.

- [x] **Step 3: Build the artifact (§7 step 3)**

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder data summaries`
Expected, in about 12 seconds:
- the rule `Building Window-Fitting Summaries` and `Loaded config from conf/config.yaml`;
- the tokenizer's warning `Token indices sequence length is longer than the specified maximum
  sequence length for this model (589 > 512)…`. It is harmless: the counter tokenizes whole texts
  without truncation, and no model reads them;
- `Embedding 6,257 distinct units of 753 over-window texts`;
- `Window summaries for sentence-transformers/all-MiniLM-L6-v2 replaced channel texts:
  {'description': 162, 'examples': 106, 'excluded': 485}`. This line is the build's own check,
  run on the temporary file under a temporary pin;
- `Window summaries: conf/data/window_summaries.csv`;
- `Pin for sentence-transformers/all-MiniLM-L6-v2: SummariesPin(path='conf/data/window_summaries.csv',
  sha256='dd425eb5ef9a7fa2be1b2e821c02f1b036f74f6ec503ff2fa70256ea7808a9a0', window=128)`.

If the counts differ, stop and ask. If the sha256 differs, go on with the printed one, and record
both values in a `> Deviation:` note.

Run: `ls conf/data/ | grep window_summaries`
Expected: `window_summaries.csv` and `window_summaries_provenance.json`, with no `.tmp` file.

Run: `shasum -a 256 conf/data/window_summaries.csv`
Expected: the printed sha256.

Run: `uv run python -c "import json; p = json.load(open('conf/data/window_summaries_provenance.json')); print(p['backbone'], p['revision'][:8], p['window'], p['budget'], p['descriptions']['sha256'][:8]); [print(c, v['present'], v['over_window'], v['summarized'], v['mean_kept_share'] and round(v['mean_kept_share'], 4), v['min_summary_tokens'], v['p10_summary_tokens'], v['plateau_stops']) for c, v in p['channels'].items()]"`
Expected:

```text
sentence-transformers/all-MiniLM-L6-v2 1110a243 128 {'description': 124, 'examples': 124, 'excluded': 124, 'title': 124} fe8c54e3
description 2111 162 162 0.6539 83 107 3
examples 1075 106 106 0.616 39 114 20
excluded 1117 485 485 0.6224 44 106 14
title 2125 0 0 None None None 0
```

The present, over and summarized counts are stop conditions. The kept shares, token floors and
plateau stops follow from the selection, so they are information, like the sha256.

Run: `wc -c conf/data/window_summaries.csv`
Expected: about 514 KB (514102 bytes in the dry run): one header and 753 rows, LF line endings,
one final newline.

- [x] **Step 4: Pin it; run the tests, the suite and the format check; commit (§7 step 3)**

Use the sha256 Step 3 printed.

In `src/naics_embedder/panels/window_summaries.py`, replace:

```python
    sha256: str
    window: int

# Keyed by backbone. Plan 9's Exit adds MiniLM's entry together with the artifact it pins.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {}

def summaries_identity(backbone: str) -> Optional[str]:
    '''
```

with:

```python
    sha256: str
    window: int

# Keyed by backbone, like TRAINED_WINDOWS (utils/input_window.py). Each entry is a reviewed
# change, committed with the artifact it pins; `naics-embedder data summaries` prints it.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {
    'sentence-transformers/all-MiniLM-L6-v2': SummariesPin(
        path=WINDOW_SUMMARIES_PATH,
        sha256='dd425eb5ef9a7fa2be1b2e821c02f1b036f74f6ec503ff2fa70256ea7808a9a0',
        window=128,
    ),
}

def summaries_identity(backbone: str) -> Optional[str]:
    '''
```

Run: `uv run pytest tests/unit/test_committed_window_summaries.py tests/unit/test_window_summaries.py -q`
Expected: `46 passed`. The local-only test runs, because Step 1 cloned the descriptions and the
tokenizer is in the local cache. It re-checks every row on the real descriptions.

Run: `uv run pytest -n auto -q`
Expected: `1937 passed, 1 skipped`.

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/window_summaries.py tests/unit/test_committed_window_summaries.py`
Expected: exit 0.

Run: `git status --short`
Expected, and nothing else:

```text
 M src/naics_embedder/panels/window_summaries.py
?? conf/data/window_summaries.csv
?? conf/data/window_summaries_provenance.json
?? tests/unit/test_committed_window_summaries.py
```

```bash
git add conf/data/window_summaries.csv conf/data/window_summaries_provenance.json src/naics_embedder/panels/window_summaries.py tests/unit/test_committed_window_summaries.py
git commit -m "feat(window-summaries): commit MiniLM's window-fitting summaries and pin them"
```

- [x] **Step 5: Build the tokenization cache (§7 step 4)**

This builds the cache as `NAICSDataModule.prepare_data` does. Write this script to
`/tmp/plan9_exit_cache.py` with the Write tool:

```python
'''
Plan 9 Exit, spec 7 step 4: build the tokenization cache as NAICSDataModule.prepare_data does.

Run from the worktree root: uv run python /tmp/plan9_exit_cache.py <CLONE manifest path>
'''

import json
import logging
import sys
from pathlib import Path

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import code_token_config
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import parse_config_overrides
from naics_embedder.utils.validation import require_valid_supervision_bundle

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

overrides, invalid = parse_config_overrides([f'supervision.manifest_path={sys.argv[1]}'])
assert not invalid, invalid
cfg = Config.from_yaml('conf/config.yaml').override(overrides)
bundle = require_valid_supervision_bundle(cfg)
token_config = code_token_config(cfg)
cache = tokenization_cache(
    token_config,
    description_fingerprint=bundle.manifest.description_fingerprint,
    codebook_fingerprint=bundle.manifest.codebook_fingerprint,
)

cache_path = Path(token_config.output_path)
sidecar = json.loads(cache_path.with_name(cache_path.name + '.meta.json').read_text())
pin = summaries_identity(token_config.tokenizer_name)
print(f'cache rows: {len(cache)}')
print(f'sidecar summaries: {sidecar["summaries"]}')
print(f'pin sha256:        {pin}')
print(f'sidecar matches the pin: {sidecar["summaries"] == pin}')
print(f'index_roles_no_leakage: {bundle.manifest.validation_results["index_roles_no_leakage"]}')
```

Run: `HF_HUB_OFFLINE=1 uv run python /tmp/plan9_exit_cache.py data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`
Expected, in about 10 seconds, after the config lines:

```text
naics_embedder.text_model.dataloader.tokenization_cache: Building tokenization cache (this may take a few minutes)...
naics_embedder.text_model.dataloader.tokenization_cache: Building tokenization cache...
Token indices sequence length is longer than the specified maximum sequence length for this model (589 > 512). Running this sequence through the model will result in indexing errors
naics_embedder.panels.window_summaries: Window summaries for sentence-transformers/all-MiniLM-L6-v2 replaced channel texts: {'description': 162, 'examples': 106, 'excluded': 485}
naics_embedder.text_model.dataloader.tokenization_cache: Cache built with:
naics_embedder.text_model.dataloader.tokenization_cache:    2,125 titles
naics_embedder.text_model.dataloader.tokenization_cache:    2,111 descriptions
naics_embedder.text_model.dataloader.tokenization_cache:    1,117 exclusions
naics_embedder.text_model.dataloader.tokenization_cache:    1,075 examples
naics_embedder.text_model.dataloader.tokenization_cache: Saved tokenization cache to: <worktree>/data/token_cache/token_cache.tmp
naics_embedder.text_model.dataloader.tokenization_cache: Tokenization cache built and saved successfully
cache rows: 2125
sidecar summaries: <the pin's sha256>
pin sha256:        <the pin's sha256>
sidecar matches the pin: True
index_roles_no_leakage: True
```

This run is the leakage evidence (4.4). The subset and S5 checks passed on all 753 rows, and the
bundle's manifest records `index_roles_no_leakage: true`. Copy the output into the ledger.

- [x] **Step 6: Build a text-only table on the summaries (§7 step 5)**

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet --output checkpoints/plan9_exit/text_only.parquet`
Expected, in about 11 seconds:
- `Loaded config from data/regressor_panel.yaml`;
- the same tokenizer warning, and the resolver's line with
  `{'description': 162, 'examples': 106, 'excluded': 485}`;
- `Text-only table (2,125 codes, width 384): checkpoints/plan9_exit/text_only.parquet`;
- `Provenance: checkpoints/plan9_exit/text_only_provenance.json`.

Run: `uv run python -c "import json; p = json.load(open('checkpoints/plan9_exit/text_only_provenance.json')); print(p['backbone'], p['revision'], p['max_length'], p['summaries'], p['codes'])"`
Expected: `sentence-transformers/all-MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 128
<the pin's sha256> 2125`.

- [x] **Step 7: Show the chain's records and refusals (§7 step 6)**

Write this script to `/tmp/plan9_exit_chain.py` with the Write tool:

```python
'''
Plan 9 Exit, spec 7 step 6: the identity chain's records and refusals.

Run from the worktree root: uv run python /tmp/plan9_exit_chain.py <CLONE manifest path>
'''

import json
import sys
import tempfile
from pathlib import Path

from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.text_model.arm_encoder import ArmEncoder
from naics_embedder.text_model.export import code_token_config
from naics_embedder.utils.config import Config
from naics_embedder.utils.training import parse_config_overrides
from naics_embedder.utils.validation import require_valid_supervision_bundle

PLAN8 = Path('checkpoints/plan8_exit')
PLAN9 = Path('checkpoints/plan9_exit')

overrides, invalid = parse_config_overrides([f'supervision.manifest_path={sys.argv[1]}'])
assert not invalid, invalid
cfg = Config.from_yaml('conf/config.yaml').override(overrides)
bundle = require_valid_supervision_bundle(cfg)
token_config = code_token_config(cfg)
pin = summaries_identity(token_config.tokenizer_name)
print(f'pin sha256: {pin}')

# The store reads the summaries from step 5's provenance, and refuses plan 8's, which has none
with tempfile.TemporaryDirectory() as root:
    store = ArtifactStore(Path(root) / 'store')
    reference = store.put_text_only(PLAN9 / 'text_only.parquet')
    print(f'plan 9 text-only table stored: summaries_sha256 {reference.summaries_sha256}')
    assert reference.summaries_sha256 == pin
    try:
        store.put_text_only(PLAN8 / 'text_only.parquet')
    except ValueError as exc:
        print(f'plan 8 text-only table refused: {exc}')
    else:
        raise AssertionError("plan 8's text-only table was stored")

# The arm encoder refuses plan 8's arm on its table's provenance, before the model loads
provenance = json.loads((PLAN8 / 'arm_table_provenance.json').read_text())
print(
    f"plan 8 arm table provenance: summaries {provenance['summaries']}, "
    f"tokenizer {provenance.get('tokenizer', '<absent>')}"
)
try:
    ArmEncoder.from_files(PLAN8 / 'last.ckpt', PLAN8 / 'arm_table.parquet', bundle, token_config)
except ValueError as exc:
    print(f'plan 8 arm refused: {exc}')
else:
    raise AssertionError("plan 8's arm was read")
```

Its comment says "step 5's provenance": that is §7's numbering, this task's Step 6.

Run: `HF_HUB_OFFLINE=1 uv run python /tmp/plan9_exit_chain.py data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`
Expected, after the config lines:

```text
pin sha256: <the pin's sha256>
plan 9 text-only table stored: summaries_sha256 <the pin's sha256>
plan 8 text-only table refused: checkpoints/plan8_exit/text_only_provenance.json lacks the field 'summaries'
plan 8 arm table provenance: summaries None, tokenizer <absent>
plan 8 arm refused: checkpoints/plan8_exit/arm_table.parquet was exported before Stage 6b: its provenance records no tokenizer; export the table again
```

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder tools export-table --checkpoint checkpoints/plan8_exit/last.ckpt --output checkpoints/plan9_exit/plan8_arm_table.parquet supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`
Expected: exit code 1, with `Export failed: supervision contract mismatch (saved, configured):
{'summaries': (None, '<the pin's sha256>')}`. Plan 8's checkpoint trained under null summaries.

Run: `ls checkpoints/plan9_exit/`
Expected: `text_only.parquet` and `text_only_provenance.json` only. The refused export wrote
nothing.

- [x] **Step 8: The full checks (§7 step 7)**

Run: `uv run pytest -n auto -q`
Expected: `1937 passed, 1 skipped`.

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warning.

Run: `ls logs/selection_log.jsonl`
Expected: "No such file or directory". No split was read.

Run: `rm /tmp/plan9_exit_cache.py /tmp/plan9_exit_chain.py`

Run: `git status --short`
Expected: no output.

## Final verification (controller, inline)

Run every check before the final review, and paste each output into the ledger.

- [x] **Step 1: The suite on both CI versions**

> Deviation: the final review's fix wave added 3 tests, so both versions end at 1940 passed, 1 skipped; CI should show 1939 passed, 2 skipped.

Run: `uv run pytest -n auto -q`
Expected:
- `1937 passed, 1 skipped`: the baseline's 1845 plus the 92 tests the plan added;
- the one skip needs CUDA, and the local-only test runs, because `data/` holds the clone. CI has
  no `data/`, so it shows `1936 passed, 2 skipped`.

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310 uv run pytest -n auto -q`
Expected:
- the same counts;
- a few hundred extra "encountered in matmul" RuntimeWarnings, which come from numpy 2.2 with
  Accelerate on 3.10 and are not a failure.

Run: `rm -rf /tmp/naics-py310`

- [x] **Step 2: Lint, format and docs**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warning.

- [x] **Step 3: Nothing of the truncation era is left**

Run: `git grep -n -w SUMMARIES -- src tests`
Expected: no output.

Run: `git grep -n _SENTENCE_BREAK -- src tests`
Expected: no output.

Run: `git grep -n -i 'null until' -- src tests docs CLAUDE.md README.md`
Expected: no output.

Run: `git grep -n -i -E 'truncates to (the|it)|is truncated to the' -- src docs CLAUDE.md README.md`
Expected: no output.

- [x] **Step 4: Spec §6's items, test by test**

Each item of spec §6 maps to tests the plan wrote (the table below). Step 1 ran them all. This
step checks that every one still exists under its name.

Write this script to `/tmp/plan9_check_map.py` with the Write tool:

```python
'''Plan 9: every test that spec §6's items map to is collected under its name.'''

import re
import subprocess

NAMES = [
    'test_the_seam_pins_minilm_alone_to_a_pin_no_test_can_read',
    'test_the_marker_leaves_the_committed_pins_alone',
    'test_the_identity_is_the_pins_sha256_and_none_without_a_pin',
    'test_the_module_imports_no_torch',
    'test_units_are_sentences_that_close_only_at_a_real_break',
    'test_a_sentence_over_the_budget_is_re_split_at_its_clauses',
    'test_a_clause_over_the_budget_is_re_split_at_its_pieces_without_the_guards',
    'test_a_piece_over_the_budget_is_refused',
    'test_an_examples_entry_over_the_budget_is_refused',
    'test_a_title_is_one_unit_and_one_over_the_window_is_refused',
    'test_a_title_over_the_window_cannot_be_summarized',
    'test_the_greedy_adds_the_unit_that_most_raises_the_cosine_while_one_fits',
    'test_an_exact_tie_goes_to_the_earlier_unit',
    'test_the_selection_stops_when_no_unit_raises_the_cosine',
    'test_a_summary_keeps_whole_units_in_source_order_and_fits_with_its_marker',
    'test_an_over_window_text_is_replaced_by_its_summary',
    'test_texts_that_fit_pass_through_without_a_pin_or_an_artifact',
    'test_an_over_window_text_without_a_pin_is_refused',
    'test_a_pin_for_another_window_is_refused',
    'test_a_missing_artifact_is_refused_by_its_absolute_path',
    'test_an_artifact_with_another_sha256_is_refused',
    'test_the_rows_must_be_exactly_the_over_window_texts',
    'test_each_row_must_match_its_source_and_window',
    'test_every_row_is_checked_against_its_source_before_any_is_checked_as_an_extract',
    'test_a_summary_that_is_not_an_extract_is_refused',
    'test_reordered_examples_entries_are_refused',
    'test_a_summary_over_the_window_is_refused',
    'test_a_failing_check_leaves_no_artifact',
    'test_an_existing_artifact_is_kept_without_force',
    'test_data_summaries_refuses_to_rebuild_without_force',
    'test_a_backbone_with_no_recorded_window_is_refused',
    'test_an_over_window_text_is_cached_as_its_pinned_summary',
    'test_a_cache_built_under_other_summaries_is_rebuilt',
    'test_a_cache_built_under_other_markers_is_rebuilt',
    'test_the_sidecar_records_the_markers_and_the_pins_summaries',
    'test_a_stale_sidecar_is_refused_naming_each_key_that_differs',
    'test_the_table_reads_an_over_window_text_as_its_summary',
    'test_the_provenance_records_the_backbones_summaries',
    'test_a_contract_saved_before_stage_6b_reads_as_null_summaries',
    'test_the_model_records_its_summaries_in_either_contract',
    'test_the_model_and_its_contract_record_the_tokenizers_summaries',
    'test_a_containment_run_records_the_summaries_too',
    'test_legacy_or_mismatched_checkpoint_cannot_exact_resume',
    'test_a_containment_checkpoint_under_other_summaries_cannot_exact_resume',
    'test_the_supervision_check_refuses_other_summaries_naming_the_field',
    'test_weights_only_loads_a_checkpoint_trained_under_other_summaries',
    'test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text',
    'test_a_caller_that_omits_the_summaries_is_a_type_error',
    'test_a_load_that_omits_the_summaries_is_a_type_error',
    'test_the_provenance_names_the_table_and_the_checkpoint',
    'test_a_checkpoint_trained_on_truncated_text_is_refused',
    'test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads',
    'test_a_table_read_under_another_tokenizer_or_summaries_is_refused_before_any_model_loads',
    'test_a_checkpoint_trained_on_truncated_text_is_refused_on_read',
    'test_a_text_only_table_is_stored_with_its_provenance',
    'test_nothing_is_stored_until_the_provenance_describes_the_table',
    'test_the_text_only_table_must_read_the_arms_summaries',
    'test_a_seed_exported_from_other_text_is_refused_before_any_read',
    'test_a_seed_table_without_its_export_provenance_is_refused_before_any_read',
    'test_the_pin_names_the_committed_artifact',
    'test_each_summarized_text_has_one_row_that_fits_the_window',
    'test_the_provenance_counts_every_over_window_text_as_summarized',
    'test_the_resolver_accepts_the_artifact_on_the_real_descriptions',
]

collected = subprocess.run(
    ['uv', 'run', 'pytest', '--collect-only', '-q', '-q', 'tests'],
    capture_output=True,
    text=True,
    check=True,
).stdout
# A name, then a parameter list or the end of its line: a prefix of a longer name does not count
print('missing:', [name for name in NAMES if not re.search(rf'::{name}(\[|$)', collected, re.M)])
```

Run: `uv run python /tmp/plan9_check_map.py`
Expected: `missing: []`. A missing name means a test was dropped or renamed during execution:
stop and report it as a deviation.

Run: `rm /tmp/plan9_check_map.py`

| Spec §6 | Tests |
|---|---|
| The seam | `test_window_summaries.py`: `test_the_seam_pins_minilm_alone_to_a_pin_no_test_can_read`, `test_the_marker_leaves_the_committed_pins_alone`, `test_the_identity_is_the_pins_sha256_and_none_without_a_pin` |
| Units: segmenter boundaries on crafted texts | `test_window_summaries.py::test_units_are_sentences_that_close_only_at_a_real_break` (six cases) |
| Units: the level-2 and level-3 re-splits | `test_a_sentence_over_the_budget_is_re_split_at_its_clauses`, `test_a_clause_over_the_budget_is_re_split_at_its_pieces_without_the_guards` |
| Units: the three raises | `test_a_piece_over_the_budget_is_refused`, `test_an_examples_entry_over_the_budget_is_refused`, `test_a_title_is_one_unit_and_one_over_the_window_is_refused`; `test_window_summaries_build.py::test_a_title_over_the_window_cannot_be_summarized` |
| Selection | `test_window_summaries_build.py`: `test_the_greedy_adds_the_unit_that_most_raises_the_cosine_while_one_fits`, `test_an_exact_tie_goes_to_the_earlier_unit`, `test_the_selection_stops_when_no_unit_raises_the_cosine`, `test_a_summary_keeps_whole_units_in_source_order_and_fits_with_its_marker` |
| Resolver: substitution; texts that fit pass, the artifact unread | `test_an_over_window_text_is_replaced_by_its_summary`, `test_texts_that_fit_pass_through_without_a_pin_or_an_artifact` |
| Resolver: every refusal, at its step | `test_window_summaries.py`: `test_an_over_window_text_without_a_pin_is_refused`, `test_a_pin_for_another_window_is_refused`, `test_a_missing_artifact_is_refused_by_its_absolute_path`, `test_an_artifact_with_another_sha256_is_refused`, `test_the_rows_must_be_exactly_the_over_window_texts`, `test_each_row_must_match_its_source_and_window`, `test_every_row_is_checked_against_its_source_before_any_is_checked_as_an_extract`, `test_a_summary_over_the_window_is_refused` |
| Resolver: S5 with a reordered, a repeated and an edited unit | `test_a_summary_that_is_not_an_extract_is_refused` (three cases), `test_reordered_examples_entries_are_refused` |
| Build: no artifact after a failing check; the overwrite refusal | `test_window_summaries_build.py`: `test_a_failing_check_leaves_no_artifact`, `test_an_existing_artifact_is_kept_without_force`, `test_a_backbone_with_no_recorded_window_is_refused`; `test_cli_commands.py::test_data_summaries_refuses_to_rebuild_without_force` |
| Cache substitution | `test_tokenization_cache.py`: `test_an_over_window_text_is_cached_as_its_pinned_summary`, `test_an_over_window_text_without_a_pin_is_refused` |
| Cache identity | `test_tokenization_cache.py`: `test_a_cache_built_under_other_summaries_is_rebuilt`, `test_a_cache_built_under_other_markers_is_rebuilt`, `test_the_sidecar_records_the_markers_and_the_pins_summaries`, `test_a_stale_sidecar_is_refused_naming_each_key_that_differs` |
| Text-only builder | `test_text_only.py`: `test_the_table_reads_an_over_window_text_as_its_summary`, `test_the_provenance_records_the_backbones_summaries` |
| Contract: absent reads as null; the model passes its summaries to both builders | `test_checkpoint_contract.py::test_a_contract_saved_before_stage_6b_reads_as_null_summaries`; `test_naics_model.py` `test_the_model_records_its_summaries_in_either_contract`; `test_cli_training.py`: `test_the_model_and_its_contract_record_the_tokenizers_summaries`, `test_a_containment_run_records_the_summaries_too` |
| Contract: exact resume, containment included; the supervision check; weights-only ignores it | `test_checkpoint_contract.py`: `test_legacy_or_mismatched_checkpoint_cannot_exact_resume` (its two summaries cases), `test_a_containment_checkpoint_under_other_summaries_cannot_exact_resume`, `test_the_supervision_check_refuses_other_summaries_naming_the_field`, `test_weights_only_loads_a_checkpoint_trained_under_other_summaries` |
| Contract: the HGCN feeder; a caller that omits it | `test_export.py`: `test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text`, `test_a_load_that_omits_the_summaries_is_a_type_error`; `test_checkpoint_contract.py::test_a_caller_that_omits_the_summaries_is_a_type_error` |
| Export and arm encoder | `test_export.py`: `test_the_provenance_names_the_table_and_the_checkpoint`, `test_a_checkpoint_trained_on_truncated_text_is_refused`; `test_arm_encoder.py`: `test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads`, `test_a_table_read_under_another_tokenizer_or_summaries_is_refused_before_any_model_loads`, `test_a_checkpoint_trained_on_truncated_text_is_refused_on_read` |
| Decision records | `test_decision_store.py`: `test_a_text_only_table_is_stored_with_its_provenance`, `test_nothing_is_stored_until_the_provenance_describes_the_table` (its `_without_summaries` case); `test_decision.py::test_the_text_only_table_must_read_the_arms_summaries`; `test_decision_sweep.py`: `test_a_seed_exported_from_other_text_is_refused_before_any_read` (five cases, each with an empty log), `test_a_seed_table_without_its_export_provenance_is_refused_before_any_read` |
| The committed artifact | `test_committed_window_summaries.py`: `test_the_pin_names_the_committed_artifact`, `test_each_summarized_text_has_one_row_that_fits_the_window`, `test_the_provenance_counts_every_over_window_text_as_summarized` |
| Local-only | `test_committed_window_summaries.py::test_the_resolver_accepts_the_artifact_on_the_real_descriptions` |
| P3: torch-free | `test_window_summaries.py::test_the_module_imports_no_torch` |

- [x] **Step 5: The branch carries only this plan**

Run: `git log --oneline origin/main..HEAD`
Expected:
- the five commits of **Pre-flight** Step 1, the commits of Tasks 1–10 and their review fixes;
- no "config" or "graph config".

Run: `git diff --stat origin/main...HEAD -- conf`
Expected: only `conf/data/window_summaries.csv` and `conf/data/window_summaries_provenance.json`.

Run: `git grep -n 'manifest_path:' -- conf/config.yaml`
Expected: `conf/config.yaml:12:  manifest_path: null  # …`. No path is committed.

- [x] **Step 6: The final review**

> Deviation: Codex (`-m gpt-6-astra`) reviewed beside the code-reviewer and found nothing; by the user's ruling one fix wave fixed 11 findings (3fc6507 tests, 817b5ec docstrings and dividers), Steps 1–5 were rerun, and the rest were deferred.

Dispatch the code-reviewer agent, which is pinned to Opus, on the whole branch, with:
- `git diff origin/main...HEAD -- src tests conf docs CLAUDE.md`;
- this plan;
- spec `specs/window-fitting-summaries.md`.

Then, for each finding:
- fix it, with a test where it is behavior, and rerun Steps 1–5; or
- triage it as deferred, which Plan completion's gate handles.

## Plan completion (controller, inline)

Run this after Final verification, once the final review's findings are resolved, and before
finishing-a-development-branch. It is writing-plans' Plan Completion Protocol, with this plan's
edits written out. Every "replace" text below occurs exactly once in its file, at 2d0af69 and after
the edits above it. Under a step that changed a name, adjust the text to what shipped.

- [x] **Step 1: Check for parallel sessions**

`specs/naics-embedding-roadmap.md` and `specs/deferred_items.md` are shared by every session.

Run: `git worktree list`
- Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.
- If another worktree belongs to a running session, or the user has mentioned one, hold the
  shared edits: Step 4 and Step 5.
- When holding, give the user the exact text of each held edit, and continue with the rest.

- [x] **Step 2: The resolve-before-defer gate**

Collect the leftovers:
- plan steps skipped or descoped during execution;
- final-review findings that were not fixed.

Partition them:
- **Needs the user's input.** Ask now, as one batched set of questions.
- **Unblocked by an answer.** Implement it now, then restart this protocol.
- **Everything else.** Defer it (Step 5).

Unanswered questions block Steps 3–7.

- [x] **Step 3: Mark up the plan**

In `specs/plans/9-window-fitting-summaries.md`:
- Tick every completed step (`- [x]`).
- Under a step that deviated, add a one-line `> Deviation: …` note.
- Under a skipped step, add `> Skipped: <why> → deferred`.
- After the title line, add the status header below.
  - When the gate deferred nothing, end it with `; nothing deferred` instead.
  - Under inline execution, write `executing-plans`.

```markdown
**Status: COMPLETE (YYYY-MM-DD)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md
```

- [x] **Step 4: The roadmap: tick Stage 6b and re-validate Stages 7–12**

First read the values the Realized line records, from the committed artifact, its provenance and
the pin.

Run: `uv run python -c "import json, polars as pl; from naics_embedder.panels.window_summaries import WINDOW_SUMMARIES; p = json.load(open('conf/data/window_summaries_provenance.json')); r = pl.read_csv('conf/data/window_summaries.csv', schema_overrides={'code': pl.Utf8}); print(WINDOW_SUMMARIES['sentence-transformers/all-MiniLM-L6-v2'].sha256[:8], r.height); [print(c, s['summarized'], round(s['mean_kept_share'], 3), *r.filter(pl.col('channel') == c).select(pl.col('units_kept').sum(), pl.col('units_total').sum()).row(0)) for c, s in sorted(p['channels'].items()) if s['summarized']]; print('titles over the window:', p['channels']['title']['over_window'])"`
Expected:

```text
dd425eb5 753
description 162 0.654 581 1235
examples 106 0.616 1179 2307
excluded 485 0.622 1718 3270
titles over the window: 0
```

Each line is the pin's sha256 prefix and the row count, then per channel the rows, the mean share
of source tokens kept, and the units kept and in total. The Realized text below carries the plan's
dry-run values, which these are. If any printed value differs, write the printed one, and add a
Deviation note under this step.

In `specs/naics-embedding-roadmap.md`, replace:

```markdown
- [ ] Stage 6b: Window-fitting summaries
```

with:

```markdown
- [x] Stage 6b: Window-fitting summaries
```

Replace:

```markdown
      text-only table's provenance and the decision records name it.
      ROUTING: brainstorming
```

with the text below, writing the completion date for `YYYY-MM-DD`:
- The Rollout note is the spec's §10, "Switch at merge".
- The Realized line holds what §10 asks for: the artifact's sha256, the rows per channel and the
  mean share of source tokens kept.

```markdown
      text-only table's provenance and the decision records name it.
      ROUTING: brainstorming
      Rollout note: the switch happens at merge. Every `channels-v3` cache rebuilds once. A
      checkpoint made before this stage records no summaries: exact resume, export, the outcome read
      and the HGCN feeder refuse it. `ArmEncoder.from_files` and `run_seed_sweep` refuse a table
      exported before it, and the artifact store refuses a text-only table built before it, so
      `run_seed_sweep` and `decide` do too. Weights-only loading still works, and
      `tools regressor-panel`, which checks no provenance, still pairs pre-6b tables. Plan 8's Exit
      numbers stay the floor its finding records; Stage 7 trains and builds its text-only table on
      the summaries.
      Realized: `conf/data/window_summaries.csv` (sha256 `dd425eb5…`), pinned for MiniLM at 128
      tokens by `WINDOW_SUMMARIES` (`panels/window_summaries.py`), holds 753 extractive summaries,
      picked by backbone centrality (`centrality-v1`) from whole sentences, clauses and examples
      entries: 162 descriptions, 106 examples texts and 485 exclusion texts. They keep a mean 0.654,
      0.616 and 0.622 of their source tokens, and 581 of 1,235, 1,179 of 2,307 and 1,718 of 3,270 of
      their units. No title is over the window. The token cache, the checkpoint contract, the export
      and text-only provenances and the decision records carry the sha256, and the export provenance
      also names the tokenizer. The Exit trained nothing and read no split.
      Stage 6b: COMPLETE (YYYY-MM-DD) — implemented by plan 9
      (specs/plans/completed/9-window-fitting-summaries.md). Next: resume the roadmap.
```

Then tell the later stages what shipped. §10 names Stages 9, 10 and 12; Stages 7 and 8 also read
what this plan changed.

**Stage 7.** It trains on the summaries, its text-only table must name them, and its seed sweep
now checks each seed's export provenance. Replace:

```markdown
      HGCN feeder still serves. Stage 6b's summaries, which the reference
      configuration and its text-only table read; Stage 5's D*, redirection table and unary
```

with:

```markdown
      HGCN feeder still serves. Stage 6b's summaries (`WINDOW_SUMMARIES` in
      `panels/window_summaries.py`), which the reference configuration and its text-only table read:
      a checkpoint's contract records their sha256, and exact resume, export and reads refuse one
      trained under other summaries, plan 8's Exit checkpoint included, so Stage 6's floor was read
      on truncated text; Stage 5's D*, redirection table and unary
```

Replace:

```markdown
      embeds bundle 18403d29's text, which Stage 5 replaces (D9). Stage 4's seed-sweep driver
      (`decision.sweep.run_seed_sweep`, whose `ArmRunner` returns each seed's `SeedArtifacts`:
      the checkpoint, the 2,125-code table in the export form, the `QueryCodeEncoder` and its
      distance), decision tooling (`tools margins`, `tools decide`) and δ procedure. The margins
```

with:

```markdown
      embeds bundle 18403d29's text, which Stage 5 replaces (D9), and the store refuses plan 8's
      Exit table (`checkpoints/plan8_exit/`), whose provenance predates the summaries; plan 9's Exit
      built one under them (`checkpoints/plan9_exit/text_only.parquet`). Stage 4's seed-sweep driver
      (`decision.sweep.run_seed_sweep`, whose `ArmRunner` returns each seed's `SeedArtifacts`: the
      checkpoint, the 2,125-code table in the export form with the provenance `tools export-table`
      writes beside it, the `QueryCodeEncoder` and its distance; before any panel read,
      `check_seed_table` refuses a seed whose table's backbone, revision, descriptions, summaries or
      window differ from the arm's `ArmSpec`, whose `summaries_sha256` is `summaries_identity` of
      the arm's backbone), decision tooling (`tools margins`, `tools decide`) and δ procedure. The
      margins
```

**Stage 8.** Each geometry arm's export must keep the provenance's shape, which the sweep and the
query path read. Replace:

```markdown
      Produces: Geometry as a configuration factor with per-arm distance, decoding and export
      (the radial term only in the hyperbolic arm); guards on the two tie-order keys Stage 4
```

with:

```markdown
      Produces: Geometry as a configuration factor with per-arm distance, decoding and export (the
      radial term only in the hyperbolic arm; every arm's export writes the provenance
      `tools export-table` writes, tokenizer and summaries included, which `run_seed_sweep` and
      `ArmEncoder.from_files` check); guards on the two tie-order keys Stage 4
```

**Stage 9.** The entry says the summaries are keyed by target window; they are pinned per backbone,
its tokenizer included (spec §10, 4.6). Replace:

```markdown
      dense layer, which mean pooling never reads; Stage 6b's summaries, keyed by
      target window, so a candidate with another window needs its own or none; Stage 4's
      tooling.
```

with:

```markdown
      dense layer, which mean pooling never reads; Stage 6b's summaries, pinned per backbone, not
      per window (`WINDOW_SUMMARIES` in `panels/window_summaries.py` holds one `SummariesPin`, with
      its window, per tokenizer name). A candidate whose channel texts all fit its window reads them
      as they are; one with a longer text is refused until
      `naics-embedder data summaries --backbone <name> --output <path>` writes its own artifact (the
      default path is MiniLM's) and its pin is committed. Each candidate's summaries are selected
      under its own frozen weights, so two candidates can read different summaries of one text;
      whether one selection backbone serves them all is this stage's brainstorm's call
      (`specs/completed/window-fitting-summaries.md`, section 2.2). Stage 4's tooling.
```

**Stage 10.** An arm table's provenance now carries `summaries` and `tokenizer`, and the sweep
checks it per seed. Replace:

```markdown
      2,125-code table from the selected arm. Every arm's per-seed table carries the export's
      provenance (`text_model/export.py`: its checkpoint's hash, its own and the window), which
      `ArmEncoder.from_files` requires before Stage 12 can embed sealed queries against it.
```

with:

```markdown
      2,125-code table from the selected arm. Every arm's per-seed table carries the export's
      provenance (`text_model/export.py`: its checkpoint's hash, its own, the window, the tokenizer
      and the summaries' sha256). Before any panel read, `run_seed_sweep` refuses a seed whose table
      lacks it or whose backbone, revision, descriptions, summaries or window differ from the arm's
      (`check_seed_table`, D9), and `ArmEncoder.from_files` requires it before Stage 12 can embed
      sealed queries against it, so an arm whose table no `tools export-table` call writes (C's
      smoothing, D's graph stage) writes the same provenance.
```

**Stage 11** consumes Stage 6b only through Stages 7–10's artifacts. If it drops the graph stage,
the HGCN feeder's summaries check leaves with the feeder. Read its entry anyway, and edit it only if
what shipped contradicts it.

**Stage 12.** `ArmEncoder.from_files` checks the tokenizer and the summaries as well as the window,
and the pin must still be the one the recorded arms were exported under. Replace:

```markdown
      text-only table with its provenance, without which the regressor panel reads no arm. Stage
      6's query path, `ArmEncoder.from_files` (`text_model/arm_encoder.py`), which embeds sealed
      queries from a checkpoint and its table: it takes the tokenizer and window from the arm's
      own config (`code_token_config`), refuses a table whose provenance names another window,
      and maps through the hyperbolic exp map only, so a Euclidean or spherical arm needs Stage
```

with:

```markdown
      text-only table with its provenance, without which the regressor panel reads no arm (it names
      the summaries' sha256, which `decide` compares with the arm's, D9). Stage 6's query path,
      `ArmEncoder.from_files` (`text_model/arm_encoder.py`), which embeds sealed queries from a
      checkpoint and its table: it takes the tokenizer and window from the arm's own config
      (`code_token_config`). Before any model loads, it refuses a table exported before Stage 6b or
      one whose provenance names another window, tokenizer or summaries than that config reads, so
      each recorded arm is read under the summaries pin it was exported under (`WINDOW_SUMMARIES`):
      re-pinning a backbone's artifact refuses its earlier tables and checkpoints. It maps through
      the hyperbolic exp map only, so a Euclidean or spherical arm needs Stage
```

Run: `python3 -c "lines = open('specs/naics-embedding-roadmap.md').read().split('\n'); print([i + 1 for i, line in enumerate(lines) if len(line) > 100 and not line.startswith('|')])"`
Expected: `[]`. The gap-analysis table's rows are exempt.

Run: `git grep -n -E '^- \[x\] Stage 6b|Stage 6b: COMPLETE' -- specs/naics-embedding-roadmap.md`
Expected: two lines, the ticked entry and the stamp.

- [x] **Step 5: Deferred items**

In `specs/deferred_items.md`:

1. **The ticking pass.** No earlier item is wholly done. Plan 8's three items that spec §10 names
   are done in part, so each keeps its open box and gains a line naming the part (Global
   Constraints, "Recorded deviations"). Replace:

```markdown
      none affects the Exit, whose export and reads shared one config. Size: plan. Done when:
      each case is fixed or ruled no-action.
```

with:

```markdown
      none affects the Exit, whose export and reads shared one config. Size: plan. Done when:
      each case is fixed or ruled no-action.
      → partly done in plan 9 (Task 7): the provenance records the tokenizer, and `from_files`
      refuses a table exported with another. The other cases stay open.
```

Replace:

```markdown
      from plan 8's reviews as coverage gaps, not defects. Size: plan. Done when: each gap has a
      test, or a recorded ruling that it needs none.
```

with:

```markdown
      from plan 8's reviews as coverage gaps, not defects. Size: plan. Done when: each gap has a
      test, or a recorded ruling that it needs none.
      → partly done in plan 9 (Task 1): `test_a_cache_built_under_other_markers_is_rebuilt`
      (tests/unit/test_tokenization_cache.py) changes only the markers. The other gaps stay open.
```

Replace:

```markdown
      from plan 8's reviews: each is true or harmless as written. Size: quick-fix. Done when:
      each is edited or ruled no-action.
```

with:

```markdown
      from plan 8's reviews: each is true or harmless as written. Size: quick-fix. Done when:
      each is edited or ruled no-action.
      → partly done in plan 9: the cache's load messages name each sidecar entry that differs
      (Task 1), and docs/text_training.md describes the summaries where the R15 note would have gone
      and lists format, markers and summaries under Cache Regeneration (Task 9). The rest stays
      open.
```

2. **Plan 7's manifest tokenizer revision** keeps its trigger: this plan rebuilt no bundle. Leave
   it unedited.

3. **This plan's own items.** Append the gate's deferred items under
   `## 9-window-fitting-summaries — YYYY-MM-DD`, newest last. Each item follows the schema in
   writing-plans' `references/deferred-backlog.md`:
   - self-contained: file paths, why it was deferred, what it would take;
   - a `Size:`;
   - a `Done when:` or a `Revisit if:`.

   Skip the section when nothing was deferred.

Run: `python3 -c "lines = open('specs/deferred_items.md').read().split('\n'); print([i + 1 for i, line in enumerate(lines) if len(line) > 100])"`
Expected: `[]`.

- [x] **Step 6: Backlog triage**

Run: `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
Expected: its summary line, with the open count, the closure rate and the aged tail. Put it in the
completion report. On 2026-10-04 it read 25 open, a 34 % closure rate and no aged tail; this plan
closes none, since its three are done only in part.

At 20 or more open items, or with any aged tail, present the read-only triage proposal: steps 1–4
of the Triage rubric in `references/deferred-backlog.md`. Say that `/deferred` acts on the user's
selection. Do not apply a disposition here.

- [x] **Step 7: Commit the completion markup**

Run: `git status --short`
Expected: the plan, the roadmap and `specs/deferred_items.md` modified, fewer if Step 1 held the
shared edits, and nothing else.

```bash
git add specs/plans/9-window-fitting-summaries.md specs/naics-embedding-roadmap.md specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 6b and re-validate Stages 7–12"
```

- [x] **Step 8: Retire the plan and the spec**

No other plan implements `specs/window-fitting-summaries.md`, so the spec retires with the plan.
Neither file has relative links to re-point, and the roadmap names the spec only by its retired
path (Step 4, Stage 9).

Run: `git mv specs/plans/9-window-fitting-summaries.md specs/plans/completed/9-window-fitting-summaries.md`

Run: `git mv specs/window-fitting-summaries.md specs/completed/window-fitting-summaries.md`

In `specs/completed/window-fitting-summaries.md`, replace:

```markdown
**Status:** APPROVED (2026-10-04); revised the same day after a five-lens review
```

with the text below. When the gate deferred nothing, end it with `; nothing deferred` after the
plan's path instead.

```markdown
**Status:** COMPLETE (YYYY-MM-DD) — implemented by
`specs/plans/completed/9-window-fitting-summaries.md`; deferred items in
`specs/deferred_items.md`
```

Run: `git grep -n 'specs/window-fitting-summaries' -- ':!specs/plans/completed' ':!specs/completed'`
Expected: no output.

```bash
git add specs/plans/completed/9-window-fitting-summaries.md specs/completed/window-fitting-summaries.md
git commit -m "chore(specs): retire plan 9"
```

- [x] **Step 9: Hand off**

Run: `git log --oneline origin/main..HEAD`
Expected:
- b724f66, 567a187, 4ff8b7b and 2d0af69 (the roadmap resume and the spec), then the plan's commit;
- this plan's ten task commits, any review-fix commits, and Steps 7 and 8's two commits;
- no "config" or "graph config".

Finishing removes this worktree, and with it every ignored file. Keep the Exit's outputs and the
execution ledger first. Skip this in the main checkout.

Run: `ls /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit /Users/lowell/Projects/naics-embedder/logs/plan9_worktree`
Expected: "No such file or directory" for both. If either exists, stop and ask.

Run: `cp -cR checkpoints/plan9_exit /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit`

Run: `cp -cR logs /Users/lowell/Projects/naics-embedder/logs/plan9_worktree`

Then hand off to finishing-a-development-branch. The user chooses how the branch lands. Never
push without asking.
