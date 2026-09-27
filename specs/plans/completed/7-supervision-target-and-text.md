# Supervision Target and Text Implementation Plan

**Status: COMPLETE (2026-09-26)** — executed via executing-plans; deferred items in specs/deferred_items.md (six, all non-blocking findings of the final review: a late tokenizer load; no build-time re-check of the text channels; no runtime tie between the bundle's backbone and training's, for Stage 9; activity phrases checked by placement, not value, before Stage 7; a narrower leakage check in `data roles --force`; no tokenizer revision in the manifest, until the next rebuild)

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 5 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 5: rebuild the supervision bundle around the tree metric D*, directed
redirections and Req 9's text-construction rules, under a new bundle contract version, and build
that bundle once from the real Census files.

**Architecture:** One torch-free D* function (`utils/naics_hierarchy.tree_distance_matrix`)
feeds the bundle's pair facts and Stage 4's diagnostics alike, and a validator ties the stored
values to it. `data preprocess` gains a redirection table (`data/redirections.py`): every
cross-reference row and every harvested "Excluded" paragraph, once, with its activity phrase,
named codes, lineal codes and a withheld flag. The exclusion channel is built from that table.
Descriptions gain a provenance column and a lone-child inheritance rule, and absent channels are
nulls. A new `utils/input_window.py` records the backbone's trained window (128 tokens), and every
tokenizing path truncates to it. The supervision bundle moves to contract `stage3-supervision-v2`:
- D* distances;
- a unary-pair flag;
- training pairs without exclusion negatives or unary positives;
- required `index_roles` and `redirections` members;
- required validation results;
- an input-window record.

Runtime selection loses its reserved exclusion slot.

**Tech Stack:** Python 3.10 and 3.12 (CI runs both); polars; numpy; scikit-learn
(`CountVectorizer` in the leakage matcher); torch; transformers (the backbone's tokenizer);
pydantic (config, manifest); typer and rich (the CLI); pytest with xdist; ruff and yapf.

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/naics-embedding.md` at d9126ce), verbatim

- Req 7: "The structural target $D^{\ast}$ is tree path length with a virtual root above the 20
  sectors (Claude C7; ChatGPT on S1-Q2):"
  - "$D^{\ast}_{ij} = h_i + h_j$ within a sector;"
  - "$D^{\ast}_{ij} = \lambda(i) + \lambda(j) - 2$ across sectors."

  After the list: "There is no half-step for lineal pairs (ChatGPT C2; Claude C6; E2) and no
  cross-sector constant (ChatGPT C5; Claude C7; Gemini C4). $D^{\ast}$ is a tree metric and so
  satisfies the triangle inequality."
- Req 8: "A cross-reference reroutes an activity rather than asserting that two codes are
  unrelated. 4,558 of the 4,601 cross-reference rows read "…are classified in Industry j"; for
  example, 111120 has "Growing soybeans--are classified in Industry 111110" (verified locally:
  cross-reference file)."
  - "(a) As de-duplicated exclusion-channel text, each cross-reference appearing once (Claude
    C10; ChatGPT C9)."
  - "(c) Never as code–code repulsion, never as a negative in the code–code term, and never as a
    graph edge by virtue of exclusion status (ChatGPT C8; Claude C8, C25; methodology
    Cross-component 3)."
  - "Lineal references stay text only and never act as negatives: eight codes name an ancestor
    and one names its child (methodology S1 procedure; Claude C9; ChatGPT on S1-Q3). The reserved
    exclusion slot is removed (methodology S2 procedure)."
- Req 9:
  - "**Masking.** Absent channels are masked out of fusion; there is no placeholder text (ChatGPT
    C9; Claude C15; Gemini C5). The examples channel is empty for every level-2–4 code, so a
    placeholder encodes level (methodology S1 limitation 6)."
  - "**Inheritance.** Every inherited description carries its provenance, and the 14 arbitrary
    inheritance choices become a deterministic, documented rule (ChatGPT C9; Claude on S1-Q5)."
  - "**Unary pairs.** The 522 unary pairs are five-digit industries whose only child is their
    six-digit code. They leave positive supervision and parent-retrieval scoring, and all 2,125
    codes stay in the deliverable (Claude C15, C22) (chosen)."
  - "**Input windows.** Inputs fit the backbone's trained input window, which is (open —
    resolved by verification, not argument). … Windows beyond it, such as today's 512
    (rejected) (Claude C16)."
- Req 3, "Leakage": "No test query may appear, exactly or as a near-duplicate, in any training
  text: titles, descriptions, remaining examples-channel entries, exclusion text, or training
  queries (including the cross-reference activity phrases of Req 8)."
- Verification "Backbone input window": "Record the chosen backbone's trained input window from
  its own documentation. After de-duplication, report the share of each channel's texts that
  exceed it; the channel policy leaves no input beyond it."
- Verification "Target": "$D^{\ast}$ satisfies the triangle inequality over all triples (checked
  through lowest common ancestors) and contains no 99. Cross-sector values equal
  $\lambda(i) + \lambda(j) - 2$."
- Verification "Exclusions": "No exclusion pair acts as a code–code negative. No graph edge
  exists because of exclusion status. Lineal references never act as negatives, and each
  cross-reference appears once in the exclusion text."
- Verification "Text": "No absent channel contributes to fusion. Unary pairs are absent from
  positive supervision and from parent retrieval. All 2,125 codes are in the deliverable."

### The roadmap (`specs/naics-embedding-roadmap.md` at 2db1797), verbatim

- Stage 5 Objective: "Rebuild the supervision bundle around the tree metric D*, directed
  redirections and Req 9's text-construction rules."
- Stage 5 Gap closed: "Req 7 (except the IC ablation); Req 8 (a, lineal, generation side of c);
  Req 9 (except the model-side mask and the channel-presence ablation)."
- Stage 5 Produces: "A new bundle contract version: pair facts carrying D* (no 99, no half-step,
  a virtual root above the sectors); a redirection table (activity phrase, referencing code,
  destination code, lineal flag) with each cross-reference once; exclusion text de-duplicated;
  descriptions with a provenance column and a documented deterministic inheritance rule; a
  unary-pair flag on the 522 five-digit codes; absent channels as nulls with the window policy
  applied; the 43-versus-68 non-redirection count reconciled; relation names kept only as
  edge-type and diagnostic labels (D5). The training-pairs member keeps generating, on D* and
  without exclusion negatives, until Stage 7 removes it. Consumers (losses, metrics, graph
  loader, curriculum thresholds) read the new values. The two unpushed config commits are
  superseded by the new manifest path."
- Stage 5 Exit: "A bundle validator asserts that D* satisfies the triangle inequality over all
  triples via lowest common ancestors, contains no 99, and gives λ(i) + λ(j) − 2 across
  sectors; each cross-reference appears once in the exclusion text; the build's held-out
  leakage check covers the redirection table's activity phrases, which Stage 7 trains on, and
  finds no match; the nine lineal references are flagged and no generated training row uses any
  exclusion as a negative; no placeholder string exists in any channel; every inherited
  description carries a provenance value and the 14 formerly arbitrary choices resolve by the
  documented rule; the 522 unary pairs are flagged and absent from generated positives; the
  current backbone's trained window is recorded with the share of each channel's texts that
  exceeded it, and no input exceeds it, the text-only builder's included."
- Stage 5 Consumes (excerpt): "Stage 4's diagnostics report (`metrics/diagnostics.py`), which
  computes D* from the codes' own lineage rather than reading the bundle, so the bundle's new D*
  must equal it on every pair."
- D5: "Decision: relation names survive only as arm D's edge types and as diagnostic labels; the
  margin axis leaves with the eligibility rules in Stage 7."
- D9: "Decision: the arm's own backbone, frozen, embedding each code's text, reduced by PCA to
  the arm's dimension."
- Deferred items: "M6 and M8 (legacy path fields; three commands each building a bundle) fall to
  Stage 5, which reversions the bundle contract. The degenerate graph-curriculum thresholds
  change with D* in Stage 5 and leave with Stage 11." Of plan 4's entries, "Stage 5 discharges
  two: its contract requires the three `index_roles_*` validation results, and its member pins
  the entry text under an artifact hash."

### Decisions already made (do not re-ask)

Rulings carried in from earlier sessions:

- Stage 5 makes the single contract bump (the `index_roles` member becomes required, plus D*)
  and the single bundle rebuild (user decision, 2026-09-24).
- D5: relation names survive only as arm D's edge types and diagnostic labels.
- D9: the comparator reads the arm's text, and Stage 5's window policy covers the text-only
  builder's `text_only.max_length` (user approval, Stage 3 resume).
- The training-pairs member keeps generating, on D* and without exclusion negatives, until
  Stage 7.
- The bundle's D* must equal `metrics/diagnostics.py`'s on every pair.
- The shipped `conf/config.yaml` keeps `supervision.manifest_path: null`
  (`tests/unit/test_config.py::test_base_config_parses_as_repaired_pre_generation`).
  Machine-local pointers to a bundle stay unpushed.

The user answered four questions at planning (2026-09-26):

1. **The switch happens at merge.** The contract bumps. Once this branch merges, main no longer
   loads bundle 18403d29, and checkpoints trained on it load weights-only. Plan completion hands
   the user the exact local edits: the held "config" and "graph config" commits, the Lambda
   upload and the canonical-bundle memory note. Bundle 18403d29 stays on disk.
2. **Withhold 5 rows from both uses.** Held-out queries leak into five cross-reference rows:
   - four rows' activity phrases are near-duplicates of frozen held-out queries (row 653,
     311830, "Manufacturing tortilla chips"; 849, 321219; 1702, 333994; 1920, 336110);
   - row 3279 (525920, "…trusts and bankruptcy estates…") names no code but contains a test
     query exactly.

   These rows stay in the redirection table, with a null activity phrase and `withheld` set, and
   their text leaves the exclusion channel. Every other row appears once, the 68 codeless rows
   included, and the build records the withheld rows.
3. **Lone child only.** A code without official description text inherits its only child's
   text, which covers the 522 five-digit and the 140 single-child four-digit codes, and records
   the source in a provenance column. The 14 multi-child four-digit codes (2111, 3231, 3241,
   4561, 4931, 5192, 7121, 9211, 9221, 9231, 9241, 9251, 9261, 9281) keep a null description,
   masked like any absent channel.
4. **The window is 128 tokens.** This is the backbone's trained length. Its model card at
   revision 1110a243 gives the training sequence length as 128 tokens, while 256 is only the
   inference truncation default and 512 the position limit. Every channel, titles included,
   and the text-only builder use it. Longer texts are truncated, and each channel's overflow
   share is recorded.

This plan's own decisions are stated here so that no reviewer needs to re-derive them:

- **Contract name.** `stage3-supervision-v2`, bundles under
  `./data/supervision/stage3-supervision-v2`. The family name stays because checkpoints,
  caches and configs already key on it; only the version moves.
- **One D* function.**
  - Where it lives: `utils/naics_hierarchy.py`, torch-free, beside `naics_parent_code`. It
    holds `code_lineage`, which moves there from `panels/decoding.py`, which re-imports it.
  - What it computes: `tree_distance_matrix(codes)` gives `depth_i + depth_j − 2·depth_LCA`.
    Sectors sit at depth 1 under the virtual root, and combined sectors count as one.
  - What it needs: code strings only, so ancestors need not be among the codes.
  - Who calls it: the bundle's distances and `metrics/diagnostics.py` both.
- **Cross-sector is read from the relation label.** No distance marks cross-sector any more.
  Cross-sector pairs carry relation ID 99 (`CROSS_SECTOR_RELATION_ID`, now in
  `supervision/schema.py`), and the validator requires that ID exactly where the two codes'
  lowest common ancestor is the virtual root. The generator and the runtime margins key on it.
  D5 keeps the label and its margin axis until Stage 7.
- **Margins for integer D*.** The distance axis reads D* throughout:
  - A negative's distance margin is D*(a, n) − D*(a, p).
  - The one special case left on that axis is the equal-distance tie with a farther relation,
    which keeps its margin of one third of a step (`EQUAL_DISTANCE_MARGIN`).
  - The half-step cases and the fixed cross-sector distance margin go (`LINEAL_DISTANCE_DELTA`,
    `LINEAL_ADJUSTED_DISTANCE_MARGIN`, `CROSS_SECTOR_DISTANCE_MARGIN`, `CROSS_SECTOR_DISTANCE`),
    because no half-step and no sentinel exist.
  - The relation axis keeps its cross-sector margin of 15 until Stage 7 (D5).

  Nothing is tuned. Req 1 and Req 6 forbid selecting on structural statistics.
- **Positives.** A positive is a canonical, within-sector pair that is not an exclusion and not a
  unary pair. "Within-sector" replaces the old "not the maximal distance", which only meant "not
  99".
- **Exclusions are never negatives.**
  - The generator drops every candidate that shares an exclusion with its anchor.
  - The runtime candidate pool no longer admits exclusions, and the coordinator's reserved slot
    goes, with its rotation and the model's `selection_seed`.
  - The curriculum mixin marks exclusions ineligible, and validation never scores one.
  - Exclusion pairs stay in the pair facts, so both paths can keep them out.
- **Unary pairs.**
  - Pair facts carry `unary_pair`, which is true on the 522 pairs of a five-digit code and its
    only child, computed by `unary_pairs(codes)`.
  - The generator and the positive sampler, text and graph alike, drop them as positives, the
    sampler in both directions.
  - Parent retrieval (`compute_hierarchy_retrieval_metrics`, which the text and graph stages
    share) stops scoring them, as Stage 4's diagnostics already do.
- **The redirection table** (`data/redirections.py`, written by `data preprocess` to
  `data/naics_redirections.parquet` and copied into the bundle):
  - It has one row per cross-reference row, 4,601 in file order with `reference_id` 0–4600,
    then one per harvested "Excluded" paragraph, 22 in code order.
  - Columns:
    - `reference_id`, `source` (`cross_reference` or `description`), `code` and `text`;
    - `activity`: the text before `--` or before " are/is classified|included", set on
      cross-reference rows that name a code;
    - `named_codes`: codebook codes other than the row's own, in order of first appearance.
      These are a cross-reference row's destinations;
    - `lineal_codes`: the named ancestors or descendants;
    - `withheld`.
  - A row is withheld when some held-out query leaks into one of its text's segments, as the
    channel check segments text, or into its whole activity phrase. The matcher is Stage 2's
    (`panels/leakage.py`) at 9/10.
  - The bundle checks the table (Task 10). Reference IDs run from zero in table order, sources
    are known, and every named code is in the codebook and not the row's own. `lineal_codes`
    must equal what lineage gives, and an activity phrase sits only on a kept cross-reference
    row that names a code. The exclusion channel must be the one the table builds, and the
    named pairs must be exactly the pair facts' directed exclusions.
- **The exclusion channel.** A code's `excluded` text is its non-withheld rows' text, joined by
  one space in `reference_id` order, each row once. `excluded_codes` lists the named codes of
  all its rows, withheld ones included, because those pairs are still exclusions and must never
  become negatives.
- **Descriptions.** `description_source` holds the code whose official text a description is,
  and is null exactly when `description` is. Inheritance keeps today's wording rewrites. A
  five-digit code turns "This industry" into "This NAICS industry". A four-digit code turns
  "This industry" and "This NAICS industry" into "This industry group". Absent channels are
  nulls, never empty strings. No channel contains the old placeholder `[EMPTY]`.
- **The window.**
  - Where it is recorded: `utils/input_window.py` records 128 for
    `sentence-transformers/all-MiniLM-L6-v2` and cites the model card.
  - What refuses a longer one: `TokenizationConfig`, `StreamingConfig` and `TextOnlyConfig`
    refuse a `max_length` above the backbone's window. `max_length: null` resolves to the
    window.
  - The tokenization cache:
    - tokenizes every channel at the window, titles included, since the fixed 24 goes;
    - encodes an absent channel as the empty string, whose tokens are `[CLS] [SEP]`, never as
      a placeholder;
    - records per channel whether it is present, for Stage 6's mask.
  - The text-only builder refuses a `max_length` above the window.
  - The bundle's manifest records the window and each channel's overflow share.
- **Bundle members.**
  - `index_roles` and `redirections` are required (Task 10).
  - The loader requires every validation result the build records, the three `index_roles_*`
    results included (Task 10).
  - The manifest records the input window (Task 11).
  - The loader reads the contract version before it parses the manifest, so an old bundle fails
    with the contract message (Task 11).
- **Versions.** Besides the bundle contract, three versions move:
  - `negative-selection-v2`: no slot is reserved, so a checkpoint trained under the quota cannot
    exact-resume (Task 8);
  - `pair-facts-v2`: the `unary_pair` column (Task 9);
  - `redirections-v1`: the new member (Task 10).
- **Test fixtures.** `tests/fixtures/supervision.py` gains a `build_bundle` factory, which always
  passes both required members and an input-window record, and a `hierarchy_manifest` that
  `generate_supervision_bundle` builds from a small hierarchy (Tasks 10 and 11). Tests that
  built their own bundle switch to these.
- **Removed with what they served.**
  - Task 6: `data/compute_distances.py`'s networkx helpers and their tests (34 tests become 6).
  - Task 8: the exclusion rotation and `selection_seed`.
  - Task 10: plan 4's test that a bundle may lack `index_roles`.
- **Deferred items.** M6 and M8 are fixed (Task 12):
  - A repaired config that sets a legacy streaming path to a non-default value fails
    validation.
  - The three deprecated data commands print their notice and exit with status 1 without
    building.

### Recorded deviations

- **"Each cross-reference appears once in the exclusion text" holds for all but five rows.** User
  decision 2 withholds the five leaking rows from the channel. The redirection table keeps them
  with `withheld` set, and the bundle records them. Plan completion writes this into the stage
  stamp.
- **The relation margin axis stays.** D5 retires it in Stage 7. Until then it keeps its
  cross-sector margin, keyed on the relation label rather than a distance sentinel.
- **Graph-curriculum thresholds now bind.** With D* their percentiles are 7, 9 and 10 instead of
  99, 99 and 99. HGCN's in-file curriculum (`conf/graph.yaml`) therefore limits phase-1
  negatives to D* ≤ 7 and phase-2 negatives to D* ≤ 9. The roadmap anticipated this ("change
  with D* in Stage 5"). The finding records it, and Stage 11 removes the thresholds.
- **The loss docstring changes only where it would be false.** `text_model/loss.py` still
  describes exclusions as always in the denominator, but after this plan no exclusion reaches
  selection, so the one sentence is corrected. The rest of 8(c)'s training side is Stage 7's.
- **Phase 1's sibling mask widens until Stage 7.** It masks every candidate at distance 2, and
  under D* a grandparent or grandchild is at 2 as well. Stage 7 deletes Phase 1 sampling.
- **The quota's counter stays.** `SelectionReason.EXCLUSION_QUOTA` and
  `train/integrity/quota_selections` remain, always zero, until Stage 7 deletes the sampling
  rules.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - To keep a vertical Polars chain, fence it with `# yapf: disable` / `# yapf: enable`.
- **Config.** Every config key is declared in a Pydantic model.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. At the end,
  `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push to `main`.
  - Never push, cherry-pick or merge the held local commits "config" and "graph config". They
    are named by subject because every sync rewrites their SHAs.
  - Never run bare `git stash`.
  - Commit on this branch only, and end each message with the session's attribution trailer.
- **Data safety.**
  - The main checkout's `data/` is reference only; never write there. Bundle 18403d29 and the
    descriptions file it pins (sha256 5107fb83…) predate plan 4's examples channel, and the
    Nov-2025 `data/naics_relations*`, `naics_distance*` and `naics_training_pairs/` are
    pre-repair: read structure from a bundle, never from `./data`.
  - Write only to this worktree's `data/` and to `/tmp`. Copy anything needed with `cp -cR`,
    never symlink `data/`.
  - Only Task 14 runs `data preprocess` and builds a bundle, both inside this worktree.
- **Sealed sets.** No `OutcomePanel.open_test`, `RegressorPanel.open_outer` or `--split test`
  on real data (Stage 12 only). No `logs/selection_log.jsonl` may appear in this worktree.
- **Configs.** `conf/config.yaml` keeps `supervision.manifest_path: null`, and `conf/graph.yaml`
  keeps `supervision_manifest_path: null` and is not edited. Tasks 4 and 6 change other keys of
  `conf/config.yaml`, and Task 13 two of its comments, as written.
- **Downloads.** Never download Census or QCEW files. The four Census files are read from
  `~/Downloads/Data` (`--source-dir`), and the backbone and its tokenizer from the local Hugging
  Face cache (`local_files_only=True`).
- **Deferred items.** Do not promote any open item of `specs/deferred_items.md` beyond the four
  this plan discharges (Plan completion Step 3).
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Bash tool.** It runs zsh. Quote `=`-leading words (`echo '====='`); an unquoted one aborts
  the command. If the tool refuses a heredoc or a compound command, run one plain command per call
  and write files with the Write tool.

## Workspace

- **Worktree:**
  `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text`.
  Run every command from its root.
- **Branch:** `claude/plan-7-supervision-target-and-text-c3a0baa0`, cut from origin/main
  `9011c27`. This plan is its first commit.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on local `main` `ca1b769`:
  origin/main 9011c27 plus the held commits "config" and "graph config". Do not check anything
  out there.
- **Real inputs, read only (Task 14):**
  - The four Census files in `~/Downloads/Data`: `2-6 digit_2022_Codes.xlsx`,
    `2022_NAICS_Descriptions.xlsx`, `2022_NAICS_Index_File.xlsx`,
    `2022_NAICS_Cross_References.xlsx`.
  - Backbone: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`.
- **Outputs (Task 14):** this worktree's `data/` (gitignored) and the scratch directory
  `/tmp/stage5-supervision-c3a0baa0/`. Final verification removes the scratch directory. The
  bundle stays in `data/`, its only copy until Plan completion Step 5 hands it over.
- **Working directory:** the Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before Task 14's commands and before every commit, and if it is not
  this worktree, `cd` back first.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/naics_embedder/utils/naics_hierarchy.py` | `code_lineage`, `tree_distance_matrix`, `unary_pairs` | 1 |
| `src/naics_embedder/panels/decoding.py`, `metrics/diagnostics.py` | Read lineage and D* from `naics_hierarchy` | 1 |
| `src/naics_embedder/data/download_data.py` | Lone-child inheritance, provenance, null channels (2); the exclusion channel built from the redirection table, and the preprocess wiring (3) | 2, 3 |
| `src/naics_embedder/data/redirections.py` | The redirection table and the exclusion channel | 3 |
| `src/naics_embedder/panels/leakage.py`, `panels/index_roles.py` | Per-text leakage flags; the leakage check reaches activity phrases | 3 |
| `conf/data/download.yaml` | The redirection table's path | 3 |
| `src/naics_embedder/utils/input_window.py` | The trained window, the check, the token counter, overflow shares | 4 |
| `src/naics_embedder/text_model/dataloader/tokenization_cache.py` | Every channel at the window, no placeholder, presence flags | 4 |
| `src/naics_embedder/panels/text_only.py` | The text-only builder refuses a longer window | 4 |
| `conf/data_loader/tokenization.yaml`, `conf/data/regressor_panel.yaml` | The window | 4 |
| `src/naics_embedder/data/compute_relations.py` | The cross-sector label's constants move to `supervision/schema.py` | 5 |
| `src/naics_embedder/data/compute_distances.py` | D* for every canonical pair; the networkx helpers go | 6 |
| `src/naics_embedder/supervision/mode.py` | The contract literal | 6 |
| `src/naics_embedder/data/create_triplets.py` | Cross-sector by label (5); integer-D* margins (6); no exclusion negatives (7); no unary positives (9) | 5, 6, 7, 9 |
| `src/naics_embedder/supervision/margins.py` | Cross-sector by label (5); integer-D* margins (6); its docstring (8) | 5, 6, 8 |
| `src/naics_embedder/supervision/schema.py` | Label constants (5); contract v2 and margin constants (6); mining v2 (8); pair facts v2 (9); redirections version (10); the input-window record (11) | 5, 6, 8, 9, 10, 11 |
| `src/naics_embedder/data/supervision_bundle.py`, `supervision/artifacts.py` | The D* validator (6); no exclusion negatives (7); the unary flag (9); required members and results (10); the window record and contract-first loading (11) | 6, 7, 9, 10, 11 |
| `src/naics_embedder/supervision/selection.py`, `text_model/dataloader/streaming_dataset.py`, `text_model/dataloader/difficulty_sampler.py`, `text_model/mixins/curriculum.py`, `text_model/mixins/logging.py`, `text_model/mixins/validation.py`, `text_model/naics_model.py`, `text_model/loss.py`, `cli/commands/training.py` | No reserved slot; no exclusion is a candidate, a selection or a validation score | 8 |
| `src/naics_embedder/data/positive_sampling.py`, `metrics/hierarchy_structure.py` | Unary pairs out of sampled positives and parent retrieval | 9 |
| `src/naics_embedder/panels/outcome.py` | Every bundle carries the index roles (docstring) | 10 |
| `src/naics_embedder/cli/commands/data.py` | The contract literal (6); deprecated commands build nothing (12) | 6, 12 |
| `src/naics_embedder/utils/config.py` | The redirections path (3); window validators (4); contract v2 (6); text (8); member paths (10); the backbone (11); legacy paths (12); a field description (13) | 3, 4, 6, 8, 10, 11, 12, 13 |
| `conf/config.yaml` | The window (4); contract v2 (6); two comments (13) | 4, 6, 13 |
| `conf/data/supervision.yaml` | Contract v2 (6); member paths (10); the backbone (11) | 6, 10, 11 |
| `docs/…`, `CLAUDE.md`, `README.md`, `src/naics_embedder/text_model/curriculum.py` | Documentation | 13 |
| `specs/findings/supervision-target-and-text.md` | The real-data finding | 14 |

Tests:

- New files: `tests/unit/test_redirections.py` (Task 3), `tests/unit/test_input_window.py`
  (Task 4).
- Edited files (by task):
  - Task 1: `test_naics_hierarchy.py`, `test_diagnostics.py`.
  - Task 2: `test_data_download.py`.
  - Task 3: `test_data_download.py`, `test_outcome_leakage.py`, `test_index_roles.py`,
    `test_config.py`.
  - Task 4: `test_tokenization_cache.py`, `test_text_only.py`, `test_config.py`,
    `test_cli_commands.py`.
  - Task 5: `test_data_triplets.py`, `test_structural_margins.py`.
  - Task 6: `tests/fixtures/supervision.py`, `test_data_distances.py`, `test_data_triplets.py`,
    `test_structural_margins.py`, `test_supervision_artifacts.py`,
    `test_supervision_schema.py`, `test_supervision_index.py`, `test_config.py`,
    `test_cli_commands.py`, `test_cli_training.py`, `test_checkpoint_contract.py`,
    `test_streaming_dataset.py`, `test_streaming_sampling.py`, `test_hard_negative_mining.py`,
    `test_hgcn_streaming_dataset.py`, `integration/test_stage3_training_step.py`.
  - Task 7: `test_data_triplets.py`, `test_supervision_artifacts.py`,
    `test_hgcn_streaming_dataset.py`.
  - Task 8: `test_negative_selection.py`, `test_hard_negative_mining.py`,
    `test_streaming_sampling.py`, `test_naics_model.py`, `test_checkpoint_contract.py`,
    `test_cli_training.py`, `test_config.py`, `integration/test_stage3_training_step.py`,
    `integration/test_distributed_supervision.py`.
  - Task 9: `tests/fixtures/supervision.py`, `test_data_triplets.py`,
    `test_supervision_artifacts.py`, `test_positive_sampling.py`, `test_hierarchy_metrics.py`.
  - Task 10: `tests/fixtures/supervision.py`, `test_supervision_artifacts.py`, `test_config.py`,
    `test_structural_margins.py`, `test_streaming_sampling.py`, `test_hard_negative_mining.py`,
    `test_utils_validation.py`, `test_datamodule.py`, `test_graph_preprocessing.py`,
    `test_outcome_panel.py`, `integration/test_stage3_training_step.py`,
    `integration/test_distributed_supervision.py`.
  - Task 11: `tests/fixtures/supervision.py`, `test_supervision_artifacts.py`,
    `test_supervision_schema.py`, `test_config.py`.
  - Task 12: `test_config.py`, `test_cli_commands.py`, `test_cli_training.py`.
  - Tasks 13 and 14 change no test.

## Expected real-data results

These were verified while writing this plan. Tasks 1–13's code ran on a copy of this worktree,
and Task 14's commands on the real inputs, under Python 3.12 with numpy 2.3.4, polars 1.35.1,
scipy 1.16.3, scikit-learn 1.9.1, torch 2.9.1 and transformers 4.57.1. Task 14 must reproduce
the counts exactly.

| Quantity | Value |
|---|---|
| Census files (sha256) | Codes `be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316`; Descriptions `6222c4d87dcf984970e0ff8a49862ed54b546b089d03956be3f285900cd3d66c`; Index `6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63`; Cross-References `3c50c3bfa9d76862aea471cc8831fd726f0b978619d833e48c54a749d9622144` |
| Role table | `conf/data/index_roles.csv` sha256 `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a` |
| Preprocess outputs (sha256) | `naics_descriptions.parquet` `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`; `naics_index_roles.parquet` `b5a1221b6a8a3413a9d8126b7e40b5c04cebbd2f900ae2dc62801dda58c993b3`; `naics_redirections.parquet` `69b455212583f7db5992e4d4cc65c871aa785118b1309a798f4d5d77564b9149` |
| Codebook | 2,125 codes: 20 sectors, 96 three-digit, 308 four-digit, 689 five-digit, 1,012 six-digit |
| Pair facts | 2,256,750; 1,984,647 cross-sector (relation 99), 272,103 within-sector |
| D* | integers 1–10: {1: 2,105; 2: 4,593; 3: 10,938; 4: 30,175; 5: 78,544; 6: 183,454; 7: 347,006; 8: 549,103; 9: 613,307; 10: 437,525}; 0 triangle violations over 9,595,703,125 ordered triples; equal to `metrics/diagnostics.py`'s on every pair |
| Curriculum thresholds | `phase1_max_distance` 7.0, `phase2_max_distance` 9.0, `phase3_max_distance` 10.0, `phase4_max_distance` 10.0 |
| Unary pairs | 522 |
| Descriptions | 1,449 official; 662 inherited (522 five-digit, 140 four-digit); 14 null |
| Redirection table | 4,623 rows: 4,601 cross-reference and 22 description; 4,533 cross-reference rows name a code, 68 name none; 4,529 activity phrases (3,326 distinct) |
| Cross-reference wording | reads "are classified in" and names a code 4,527; reads it and names none 31; other wording naming a code 6; other wording naming none 37 |
| Withheld rows | 5: `reference_id` 653 (311830), 849 (321219), 1702 (333994), 1920 (336110), 3279 (525920) |
| Lineal references | 9: 111191→1111, 111336→1113, 211120→2111, 211130→2111, 32111→321, 321114→321, 424410→42, 488490→48, 711→7113 |
| Exclusions | 3,954 pairs, 4,586 directed; every named code in the codebook |
| Exclusion channels | 1,117 codes |
| Index roles | 20,373 entries: examples 6,118, training 7,200, validation 4,042, test 3,013 |
| Held-out leakage, with activity phrases | validation 0 exact, 0 near-duplicate; test 0, 0 |
| Training pairs | 45,163,632 rows over 2,090 anchors (the same 35 anchorless codes as bundle 18403d29); 268,831 distinct positive pairs; no exclusion or lineal reference as a negative; no exclusion or unary pair as a positive |
| Window | 128 tokens; texts over it: title 0 of 2,125 (0.0000), description 153 of 2,111 (0.0725), examples 105 of 1,075 (0.0977), excluded 464 of 1,117 (0.4154) |

The descriptions row sums to 2,125: 1,449 official plus 662 inherited plus 14 null. The wording
row sums to 4,601: the spec's 4,558 rows that read "…are classified in" are 4,527 + 31, its 43
others are 6 + 37, and the 68 codeless rows are 31 + 37.

---

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A Task 14 count differs from **Expected real-data results**, or a Census file, the role table
  or the backbone's cached revision has another hash than the pre-flight's. Do not download a
  replacement.
- Task 14's `data preprocess` or bundle build fails a check, above all the leakage check.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would write to the main checkout's `data/`, set `supervision.manifest_path` in a
  committed config, open a sealed split, or download anything.
- `origin/main` gains a commit touching a file in **File structure**, or an open PR does.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/plan-7-supervision-target-and-text-c3a0baa0` and nothing else. If the line
ends in `...origin/main`, run `git branch --unset-upstream`.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 7 …`). If "config" or "graph config"
appears, stop.

Run: `git fetch origin`, then
`git log --oneline HEAD..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: no output. If anything landed, read it. If it touches a file in **File structure**,
the roadmap or `specs/deferred_items.md`, stop and ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [x] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number.

Run: `uv run python -c "import numpy, polars, scipy, sklearn, torch, transformers; print(numpy.__version__, polars.__version__, scipy.__version__, sklearn.__version__, torch.__version__, transformers.__version__)"`
Expected: `2.3.4 1.35.1 1.16.3 1.9.1 2.9.1 4.57.1`. If they differ, stop and ask.

- [x] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `1629 passed, 1 skipped` (the skip needs CUDA). Each later full-suite count is this
baseline plus the tests the plan has added by then, minus those it removed. The warnings count
varies under xdist; ignore it.

- [x] **Step 4: Check the real inputs, read-only**

Run: `shasum -a 256 ~/Downloads/Data/2-6\ digit_2022_Codes.xlsx ~/Downloads/Data/2022_NAICS_Descriptions.xlsx ~/Downloads/Data/2022_NAICS_Index_File.xlsx ~/Downloads/Data/2022_NAICS_Cross_References.xlsx conf/data/index_roles.csv`
Expected: the five hashes of **Expected real-data results**, in that order.

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

- [x] **Step 5: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–13: each gets a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**. Every code block in a task is exact: an implementer
  copies it, and each "Replace" text occurs exactly once in its file when its edit is made.
- Task 14, **Final verification** and **Plan completion**: run them inline in the controller
  session. Task 14 reads the real inputs and applies the stop-and-ask conditions.

### Task 1: One D* function and the unary pairs

Req 7's target and Req 9's unary pairs need one definition that the bundle, the samplers and
Stage 4's diagnostics all read, so the bundle's D* equals the diagnostics' on every pair by
construction. It lives in `utils/naics_hierarchy.py`, which imports no torch, so the bundle
builder can use it.

**Files:**
- Modify: `src/naics_embedder/utils/naics_hierarchy.py` (imports; after `naics_parent_code`)
- Modify: `src/naics_embedder/panels/decoding.py` (imports; `code_lineage` moves out)
- Modify: `src/naics_embedder/metrics/diagnostics.py` (docstring, imports, `Tree.unary_children`,
  `diagnostics_report`)
- Test: `tests/unit/test_naics_hierarchy.py`, `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces, in `naics_embedder.utils.naics_hierarchy`:
  - `code_lineage(code: str) -> Tuple[str, ...]`: sector first, the code last; cached.
  - `tree_distance_matrix(codes: Sequence[str]) -> np.ndarray`: int64 D*, shape
    `(len(codes), len(codes))`.
  - `unary_pairs(codes: Iterable[str]) -> List[Tuple[str, str]]`: `(five-digit parent, its only
    six-digit child)`, sorted.
- `naics_embedder.panels.decoding.code_lineage` stays importable (re-exported).

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_naics_hierarchy.py`, replace:

```python
from pathlib import Path

import polars as pl
import pytest

from naics_embedder.utils.naics_hierarchy import (
    HierarchyIntegrityError,
    NaicsHierarchy,
    load_naics_hierarchy,
)
```

with:

```python
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from naics_embedder.utils.naics_hierarchy import (
    HierarchyIntegrityError,
    NaicsHierarchy,
    code_lineage,
    load_naics_hierarchy,
    tree_distance_matrix,
    unary_pairs,
)
```

Append to the end of `tests/unit/test_naics_hierarchy.py`:

```python

# -------------------------------------------------------------------------------------------------
# D* and unary pairs (Req 7, Req 9)
# -------------------------------------------------------------------------------------------------

# Two sectors: 11, and the combined 31-33 keyed by 31. 11121 and 31111 each have one six-digit
# child; 11111 and 32111 have two.
TREE_CODES = (
    '11', '111', '1111', '11111', '111111', '111112', '1112', '11121', '111211', '31', '311',
    '3111', '31111', '311111', '321', '3211', '32111', '321111', '321112'
)

def _row(code):
    return TREE_CODES.index(code)

def test_code_lineage_runs_from_the_sector_and_joins_combined_sectors():
    assert code_lineage('321111') == ('31', '321', '3211', '32111', '321111')
    assert code_lineage('11') == ('11', )

def test_d_star_runs_through_the_lowest_common_ancestor():
    d = tree_distance_matrix(TREE_CODES)

    assert d[_row('111111'), _row('111112')] == 2  # siblings, through 11111
    assert d[_row('11111'), _row('111111')] == 1  # parent and child: no half-step
    assert d[_row('1111'), _row('111111')] == 2  # grandparent
    assert d[_row('111111'), _row('111211')] == 6  # through 111
    assert d[_row('311111'), _row('321111')] == 8  # 31-33 is one sector
    assert d.dtype == np.int64
    assert not np.diagonal(d).any()
    assert np.array_equal(d, d.T)

def test_across_sectors_d_star_is_both_levels_less_two():
    d = tree_distance_matrix(TREE_CODES)

    for i, code_a in enumerate(TREE_CODES):
        for j, code_b in enumerate(TREE_CODES):
            if code_a[:2] == '11' and code_b[:2] != '11':
                assert d[i, j] == len(code_a) + len(code_b) - 2
    assert d[_row('11'), _row('31')] == 2  # two sectors are siblings under the virtual root

def test_d_star_needs_only_the_code_strings():
    # No ancestor of these codes is present
    assert tree_distance_matrix(['111111', '111112', '222222']).tolist() == [
        [0, 2, 10],
        [2, 0, 10],
        [10, 10, 0],
    ]
    assert tree_distance_matrix([]).shape == (0, 0)

def test_d_star_satisfies_the_triangle_inequality():
    d = tree_distance_matrix(TREE_CODES)

    # d[i, k] <= d[i, j] + d[j, k] for every ordered triple
    assert not (d[:, None, :] > d[:, :, None] + d[None, :, :]).any()

def test_unary_pairs_are_five_digit_codes_with_one_six_digit_child():
    assert unary_pairs(TREE_CODES) == [('11121', '111211'), ('31111', '311111')]

def test_a_four_digit_code_with_one_child_forms_no_unary_pair():
    assert unary_pairs(['1111', '11111', '111111', '111112']) == []
```

In `tests/unit/test_diagnostics.py`, replace:

```python
from tests.fixtures.regressor_panel import CODEBOOK, coordinate_table
```

with:

```python
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix
from tests.fixtures.regressor_panel import CODEBOOK, coordinate_table
```

and append to the end of the file:

```python

def test_d_star_is_the_one_tree_distance_function(tree):
    # The supervision bundle's distances call the same function (Stage 5)
    assert np.array_equal(tree_distance_matrix(CODES), _target(tree))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_naics_hierarchy.py tests/unit/test_diagnostics.py -q`
Expected: collection fails with `ImportError: cannot import name 'code_lineage'` (and
`tree_distance_matrix`).

- [x] **Step 3: Implement the functions**

In `src/naics_embedder/utils/naics_hierarchy.py`, replace:

```python
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

SECTOR_CODE_LENGTH = 2
```

with:

```python
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

SECTOR_CODE_LENGTH = 2
UNARY_PARENT_LENGTH = 5
```

Then replace:

```python
    if len(code) == SECTOR_CODE_LENGTH + 1:
        sector = code[:SECTOR_CODE_LENGTH]
        return COMBINED_SECTOR_KEYS.get(sector, sector)
    return code[:-1]
```

with:

```python
    if len(code) == SECTOR_CODE_LENGTH + 1:
        sector = code[:SECTOR_CODE_LENGTH]
        return COMBINED_SECTOR_KEYS.get(sector, sector)
    return code[:-1]

@lru_cache(maxsize=None)
def code_lineage(code: str) -> Tuple[str, ...]:
    '''The code's ancestors from its sector down to the code itself.'''

    chain = [code]
    parent = naics_parent_code(code)
    while parent is not None:
        chain.append(parent)
        parent = naics_parent_code(parent)
    return tuple(reversed(chain))

def tree_distance_matrix(codes: Sequence[str]) -> np.ndarray:
    '''
    D* between every two codes (Req 7): the tree path length through a virtual root.

    A code's depth is the length of its lineage, 1 for a sector, and D* is
    ``depth_i + depth_j - 2 depth_LCA`` with the virtual root at depth 0. Pairs across sectors
    therefore get λ(i) + λ(j) − 2, where λ is the number of digits. Combined sectors (31-33,
    44-45, 48-49) count as one. Only the code strings are read, so ancestors need not be among
    ``codes``.

    Returns:
        ``(len(codes), len(codes))`` int64 matrix, zero on the diagonal.
    '''

    lineages = [code_lineage(code) for code in codes]
    if not lineages:
        return np.zeros((0, 0), dtype=np.int64)
    ids: Dict[str, int] = {}
    lineage = np.full((len(lineages), max(map(len, lineages))), -1, dtype=np.int64)
    for row, chain in enumerate(lineages):
        for depth, ancestor in enumerate(chain):
            lineage[row, depth] = ids.setdefault(ancestor, len(ids))
    depth = (lineage >= 0).sum(axis=1)
    shared = np.zeros((len(lineages), len(lineages)), dtype=np.int64)
    for column in lineage.T:
        shared += (column[:, None] == column[None, :]) & (column[:, None] >= 0)
    return depth[:, None] + depth[None, :] - 2 * shared

def unary_pairs(codes: Iterable[str]) -> List[Tuple[str, str]]:
    '''
    The unary pairs among ``codes`` (Req 9): a five-digit code and its only six-digit child.

    Returns:
        ``(parent, child)`` pairs sorted by parent.
    '''

    present = set(codes)
    children: Dict[str, List[str]] = defaultdict(list)
    for code in present:
        parent = naics_parent_code(code)
        if parent is not None and len(parent) == UNARY_PARENT_LENGTH and parent in present:
            children[parent].append(code)
    return sorted((parent, kids[0]) for parent, kids in children.items() if len(kids) == 1)
```

In `src/naics_embedder/panels/decoding.py`, replace:

```python
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F

from naics_embedder.utils.naics_hierarchy import naics_parent_code
```

with:

```python
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F

from naics_embedder.utils.naics_hierarchy import code_lineage
```

Then replace:

```python
@lru_cache(maxsize=None)
def code_lineage(code: str) -> Tuple[str, ...]:
    '''The code's ancestors from its sector down to the code itself.'''

    chain = [code]
    parent = naics_parent_code(code)
    while parent is not None:
        chain.append(parent)
        parent = naics_parent_code(parent)
    return tuple(reversed(chain))

def lca_level(code_a: str, code_b: str) -> int:
```

with:

```python
def lca_level(code_a: str, code_b: str) -> int:
```

(`code_lineage` is still imported into this module, so `from naics_embedder.panels.decoding
import code_lineage` keeps working.)

In `src/naics_embedder/metrics/diagnostics.py`, replace:

```python
sector has depth 1. It is computed here from the codes themselves (``panels.decoding``'s
lineage, combined sectors as one). Distances are the arm's own: Euclidean, cosine for a
```

with:

```python
sector has depth 1. It comes from the codes themselves, through
``utils.naics_hierarchy.tree_distance_matrix``, the function the supervision bundle's distances
use (combined sectors as one). Distances are the arm's own: Euclidean, cosine for a
```

Replace:

```python
from naics_embedder.panels.decoding import (
    code_lineage,
    cosine_distances,
    euclidean_distances,
    lorentz_distances,
)
from naics_embedder.panels.regressor import coordinate_matrix
```

with:

```python
from naics_embedder.panels.decoding import (
    cosine_distances,
    euclidean_distances,
    lorentz_distances,
)
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.utils.naics_hierarchy import code_lineage, tree_distance_matrix, unary_pairs
```

Replace:

```python
        parent = self.parent
        counts = np.bincount(parent[parent >= 0], minlength=len(self.codes))
        six_digit = np.array([len(code) == 6 for code in self.codes])
        return six_digit & (parent >= 0) & (counts[np.maximum(parent, 0)] == 1)
```

with:

```python
        children = {child for _, child in unary_pairs(self.codes)}
        return np.array([code in children for code in self.codes], dtype=bool)
```

Replace:

```python
    lca = tree.lca_depth()
    target = tree.depth[:, None] + tree.depth[None, :] - 2 * lca
```

with:

```python
    lca = tree.lca_depth()
    target = tree_distance_matrix(codes)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_naics_hierarchy.py tests/unit/test_diagnostics.py tests/unit/test_outcome_decoding.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1637 passed, 1 skipped` (8 new tests).

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/utils/naics_hierarchy.py src/naics_embedder/panels/decoding.py src/naics_embedder/metrics/diagnostics.py tests/unit/test_naics_hierarchy.py tests/unit/test_diagnostics.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/utils/naics_hierarchy.py \
  src/naics_embedder/panels/decoding.py \
  src/naics_embedder/metrics/diagnostics.py \
  tests/unit/test_naics_hierarchy.py \
  tests/unit/test_diagnostics.py
git commit -m "feat(hierarchy): one torch-free D* function and the unary pairs"
```

### Task 2: Descriptions: lone-child inheritance, provenance and null channels

Req 9 wants every inherited description to carry its provenance and the 14 arbitrary choices to
follow a documented rule. User decision 3 chose the rule: inherit only from an only child. Absent
channels are nulls, and a validator refuses blank and placeholder texts, so "no placeholder
string exists in any channel" is checked where the file is written.

**Files:**
- Modify: `src/naics_embedder/data/download_data.py` (imports; `_get_descriptions_2`;
  `build_descriptions`; a new `verify_text_channels`; `download_preprocess_data`)
- Test: `tests/unit/test_data_download.py`

**Interfaces:**
- Consumes: `naics_parent_code` (`utils/naics_hierarchy.py`).
- Produces, in `naics_embedder.data.download_data`:
  - `_get_descriptions_2(descriptions_3, descriptions_exclusions, descriptions_examples, codes:
    Set[str]) -> pl.DataFrame` with `code`, `description`, `description_source`, one row per
    code of `codes`.
  - `build_descriptions(...)` output gains `description_source` (Utf8) after `description`.
  - `TEXT_CHANNELS = ('title', 'description', 'examples', 'excluded')`,
    `PLACEHOLDER_TEXT = '[EMPTY]'`.
  - `verify_text_channels(descriptions: pl.DataFrame) -> Dict[str, int]`: present texts per
    channel; raises `ValueError` on a blank text, a placeholder, or provenance that does not
    match the description's presence.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_data_download.py`, replace:

```python
    descriptions_exclusions = pl.DataFrame({'code': ['111'], 'description_id': [2]})
    descriptions_examples = pl.DataFrame({'code': ['111'], 'description_id_min': [3]})

    cleaned = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )
```

with:

```python
    descriptions_exclusions = pl.DataFrame({'code': ['111'], 'description_id': [2]})
    descriptions_examples = pl.DataFrame({'code': ['111'], 'description_id_min': [3]})

    cleaned = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples, {'111'}
    )
```

Replace:

```python
    descriptions = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )

    assert descriptions.get_column('description').to_list() == [
        'This industry comprises establishments growing grain.'
    ]
```

with:

```python
    descriptions = download_data._get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples, {'111199'}
    )

    assert descriptions.get_column('description').to_list() == [
        'This industry comprises establishments growing grain.'
    ]
```

Then replace:

```python
# -------------------------------------------------------------------------------------------------
# Local sources and the pinned index file
# -------------------------------------------------------------------------------------------------
```

with:

```python
@pytest.mark.unit
def test_a_code_without_official_text_inherits_its_only_childs_description():
    blocks = [
        ('1111', ''),
        ('11111', ''),  # the Census pointer "See industry description for 111110." is removed
        ('111110', 'This industry comprises establishments growing soybeans.'),
        ('1112', ''),
        ('11121', 'This industry comprises establishments growing wheat.'),
        ('11122', 'This industry comprises establishments growing corn.'),
    ]
    codes = [code for code, _ in blocks]
    descriptions_3 = pl.DataFrame(
        {
            'code': codes,
            'description_id': pl.Series([1] * len(blocks), dtype=pl.UInt32),
            'description': [text for _, text in blocks],
        }
    )
    no_blocks = pl.DataFrame(schema={'code': pl.Utf8, 'description_id': pl.UInt32})
    no_examples = pl.DataFrame(schema={'code': pl.Utf8, 'description_id_min': pl.UInt32})

    descriptions = download_data._get_descriptions_2(
        descriptions_3, no_blocks, no_examples, set(codes)
    )

    assert descriptions.rows() == [
        ('1111', 'This industry group comprises establishments growing soybeans.', '111110'),
        ('11111', 'This NAICS industry comprises establishments growing soybeans.', '111110'),
        ('111110', 'This industry comprises establishments growing soybeans.', '111110'),
        # Two children and no official text: no pick at all (Req 9)
        ('1112', None, None),
        ('11121', 'This industry comprises establishments growing wheat.', '11121'),
        ('11122', 'This industry comprises establishments growing corn.', '11122'),
    ]

@pytest.mark.unit
def test_build_descriptions_records_provenance_and_leaves_absent_channels_null(naics_sources):
    entries = download_data.naics_index_entries(naics_sources)

    descriptions = download_data.build_descriptions(
        naics_sources, entries.filter(pl.col('entry_id') == 0)
    )

    assert descriptions.columns[4:6] == ['description', 'description_source']
    assert descriptions.get_column('description_source').to_list() == [code for code, _ in TITLES]
    assert descriptions.filter(pl.col('code') == '111120').get_column('examples').item() is None
    assert download_data.verify_text_channels(descriptions) == {
        'title': 9,
        'description': 9,
        'examples': 2,
        'excluded': 1,
    }

@pytest.mark.unit
@pytest.mark.parametrize(
    'column, value, message',
    [
        pytest.param('excluded', '', 'blank', id='empty'),
        pytest.param('examples', '  ', 'blank', id='whitespace'),
        pytest.param('examples', 'Soybean farming; [EMPTY]', 'placeholder', id='placeholder'),
        pytest.param('description_source', None, 'description_source', id='provenance'),
    ],
)
def test_verify_text_channels_refuses_blanks_placeholders_and_missing_provenance(
    column, value, message
):
    frame = pl.DataFrame(
        {
            'code': ['111110'],
            'title': ['Soybean Farming'],
            'description': ['This industry comprises establishments growing soybeans.'],
            'description_source': ['111110'],
            'examples': ['Soybean farming'],
            'excluded': [None],
        },
        schema_overrides={
            'excluded': pl.Utf8
        },
    ).with_columns(pl.lit(value, pl.Utf8).alias(column))

    with pytest.raises(ValueError, match=message):
        download_data.verify_text_channels(frame)

# -------------------------------------------------------------------------------------------------
# Local sources and the pinned index file
# -------------------------------------------------------------------------------------------------
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_download.py -q`
Expected: FAIL. The edited `_get_descriptions_2` calls raise `TypeError` (4 positional
arguments given), and the new tests fail on `description_source` or with `AttributeError`
(`verify_text_channels`).

- [x] **Step 3: Implement the rule, the column and the validator**

In `src/naics_embedder/data/download_data.py`, replace:

```python
import hashlib
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Sequence, Set, Tuple
```

with:

```python
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Set, Tuple
```

Replace:

```python
from naics_embedder.utils.config import DownloadConfig, load_config
from naics_embedder.utils.utilities import download_with_retry as _download_with_retry
```

with:

```python
from naics_embedder.utils.config import DownloadConfig, load_config
from naics_embedder.utils.naics_hierarchy import naics_parent_code
from naics_embedder.utils.utilities import download_with_retry as _download_with_retry
```

Replace the whole of `_get_descriptions_2`, from its `def _get_descriptions_2(` line through its
final `    return descriptions` line, with:

```python
def _inherited_wording(text: str, level: int) -> str:
    '''A child's description in its parent's wording, as the Census pointer texts read.'''

    if level == 5:
        return text.replace('This industry', 'This NAICS industry', 1)
    if level == 4:
        return text.replace('This industry', 'This industry group',
                            1).replace('This NAICS industry', 'This industry group', 1)
    return text

def _get_descriptions_2(
    descriptions_3: pl.DataFrame,
    descriptions_exclusions: pl.DataFrame,
    descriptions_examples: pl.DataFrame,
    codes: Set[str],
) -> pl.DataFrame:
    '''
    Each code's description, and the code whose official text it is (Req 9, "Inheritance").

    A code's official text is its description blocks less its exclusion paragraph and its
    illustrative examples. A code without official text inherits its only child's description,
    recursively, in its own level's wording: "This NAICS industry" for a five-digit code and
    "This industry group" for a four-digit one. This rule covers the 522 five-digit codes whose
    Census text points to their six-digit child and the 140 four-digit codes with one child. A
    code with several children and no official text keeps a null description: the 14 four-digit
    codes 2111, 3231, 3241, 4561, 4931, 5192, 7121, 9211, 9221, 9231, 9241, 9251, 9261 and 9281.

    Returns:
        One row per code of ``codes``, sorted: ``code``, ``description`` and
        ``description_source``, the code whose official text the description is (null exactly
        when the description is).
    '''

    # descriptions: exclude exclusion and example description blocks
    # yapf: disable
    descriptions_4 = (
        descriptions_3
        .join(
            descriptions_exclusions,
            how='anti',
            on=['code', 'description_id']
        )
        .join(
            descriptions_examples,
            how='left',
            on='code'
        )
        .with_columns(
            pl.col('description_id_min').fill_null(999)
        )
        .filter(
            pl.col('description_id').lt(pl.col('description_id_min'))
        )
        .group_by('code', maintain_order=True)
        .agg(
            pl.col('description')
        )
        .with_columns(
            description=pl.col('description').list.join(' ')
        )
    )
    # yapf: enable

    official = dict(
        descriptions_4.filter(pl.col('description').ne('')).select('code', 'description').rows()
    )
    children: Dict[str, List[str]] = defaultdict(list)
    for code in codes:
        parent = naics_parent_code(code)
        if parent in codes:
            children[parent].append(code)

    resolved: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    def resolve(code: str) -> Tuple[Optional[str], Optional[str]]:
        if code not in resolved:
            if code in official:
                resolved[code] = (official[code], code)
            elif len(children[code]) == 1:
                text, source = resolve(children[code][0])
                wording = None if text is None else _inherited_wording(text, len(code))
                resolved[code] = (wording, source)
            else:
                resolved[code] = (None, None)
        return resolved[code]

    descriptions = pl.DataFrame(
        [(code, *resolve(code)) for code in sorted(codes)],
        schema={
            'code': pl.Utf8,
            'description': pl.Utf8,
            'description_source': pl.Utf8
        },
        orient='row',
    )

    sources = descriptions.get_column('description_source')
    inherited = int((sources.is_not_null() & sources.ne(descriptions.get_column('code'))).sum())
    absent = int(sources.is_null().sum())
    logger.info('NAICS descriptions:')
    logger.info(f'  Official: {descriptions.height - inherited - absent: ,}')
    logger.info(f'  Inherited from an only child: {inherited: ,}')
    logger.info(f'  Absent (several children, no official text): {absent: ,}\n')

    return descriptions
```

In `build_descriptions`, replace:

```python
    descriptions = _get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples
    )
```

with:

```python
    descriptions = _get_descriptions_2(
        descriptions_3, descriptions_exclusions, descriptions_examples, codes
    )
```

and replace:

```python
            description=pl.col('description'),
            examples=pl.col('examples'),
```

with:

```python
            description=pl.col('description'),
            description_source=pl.col('description_source'),
            examples=pl.col('examples'),
```

Then replace:

```python
# -------------------------------------------------------------------------------------------------
# Guard the descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------
```

with:

```python
TEXT_CHANNELS = ('title', 'description', 'examples', 'excluded')
PLACEHOLDER_TEXT = '[EMPTY]'

def verify_text_channels(descriptions: pl.DataFrame) -> Dict[str, int]:
    '''
    Fail unless every channel holds real text or is null (Req 9, "Masking").

    No channel may be an empty or blank string, since an absent channel is null, and none may
    contain the placeholder the tokenizer once substituted. A description's
    ``description_source`` must be set exactly when the description is.

    Returns:
        The number of present texts per channel.
    '''

    for channel in TEXT_CHANNELS:
        column = pl.col(channel)
        blank = descriptions.filter(column.is_not_null() & column.str.strip_chars().eq('')).height
        if blank:
            raise ValueError(f'{blank:,} {channel} texts are blank; an absent channel is null')
        placeholder = descriptions.filter(column.str.contains(PLACEHOLDER_TEXT, literal=True))
        if placeholder.height:
            raise ValueError(
                f'{placeholder.height:,} {channel} texts contain the placeholder '
                f'{PLACEHOLDER_TEXT!r}'
            )
    if descriptions.filter(pl.col('title').is_null()).height:
        raise ValueError('every code needs a title')
    mismatched = descriptions.filter(
        pl.col('description').is_null() != pl.col('description_source').is_null()
    )
    if mismatched.height:
        raise ValueError(
            f'{mismatched.height:,} codes have a description_source without a description, or '
            'the reverse'
        )
    return {
        channel: int(descriptions.get_column(channel).is_not_null().sum())
        for channel in TEXT_CHANNELS
    }

# -------------------------------------------------------------------------------------------------
# Guard the descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------
```

In `download_preprocess_data`, replace:

```python
    verify_examples_channel(naics_final, role_rows)
    leakage = verify_role_leakage(naics_final, role_rows)
```

with:

```python
    verify_examples_channel(naics_final, role_rows)
    logger.info(f'Present texts per channel: {verify_text_channels(naics_final)}')
    leakage = verify_role_leakage(naics_final, role_rows)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_download.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1643 passed, 1 skipped` (6 new tests).

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/data/download_data.py tests/unit/test_data_download.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/data/download_data.py \
  tests/unit/test_data_download.py
git commit -m "feat(data): inherit only from an only child, record provenance, keep absent channels null"
```

### Task 3: The redirection table and the de-duplicated exclusion channel

Req 8(a) wants each cross-reference once in the exclusion text. Today every row repeats once per
code it names (332999's channel is 99,683 characters). The build's leakage check must also cover
the activity phrases Stage 7 trains on. `data preprocess` gains a redirection table, which
records every cross-reference row and every harvested "Excluded" paragraph once, and the
exclusion channel is built from it. User decision 2 withholds the rows that held-out queries
leak into.

**Files:**
- Create: `src/naics_embedder/data/redirections.py`
- Modify: `src/naics_embedder/panels/leakage.py` (a new `leaking_texts`)
- Modify: `src/naics_embedder/panels/index_roles.py` (`verify_role_leakage` gains `extra_texts`)
- Modify: `src/naics_embedder/data/download_data.py` (imports; `_get_exclusions` becomes
  `_exclusion_paragraphs`; a new `naics_redirections`; `build_descriptions`;
  `download_preprocess_data`)
- Modify: `src/naics_embedder/utils/config.py` (`DownloadConfig.redirections_parquet`)
- Modify: `conf/data/download.yaml`
- Test: `tests/unit/test_redirections.py` (new), `tests/unit/test_outcome_leakage.py`,
  `tests/unit/test_index_roles.py`, `tests/unit/test_data_download.py`,
  `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `code_lineage` (Task 1); `verify_text_channels` and the `description_source` column
  (Task 2).
- Produces:
  - In `naics_embedder.panels.leakage`: `leaking_texts(segments: Sequence[Sequence[str]],
    queries: Sequence[str], *, min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD) ->
    np.ndarray`, one flag per text.
  - In `naics_embedder.panels.index_roles`: `verify_role_leakage(descriptions, role_rows,
    min_jaccard=..., *, extra_texts: Sequence[str] = ())`.
  - In `naics_embedder.data.redirections`:
    - `REDIRECTIONS_SCHEMA`: `reference_id` Int64, `source` Utf8, `code` Utf8, `text` Utf8,
      `activity` Utf8 (nullable), `named_codes` List(Utf8), `lineal_codes` List(Utf8),
      `withheld` Boolean;
    - `CROSS_REFERENCE_SOURCE = 'cross_reference'`, `DESCRIPTION_SOURCE = 'description'`;
    - `activity_phrase(text) -> Optional[str]`, `named_codes(code, text, codes) -> List[str]`,
      `lineal_codes(code, named) -> List[str]`;
    - `build_redirections(references, paragraphs, codes, held_out_queries=(), *,
      min_jaccard=...) -> pl.DataFrame`;
    - `exclusion_channel(redirections) -> pl.DataFrame` with `code`, `excluded`,
      `excluded_codes`.
  - In `naics_embedder.data.download_data`:
    - `naics_redirections(sources, held_out_queries=()) -> pl.DataFrame`;
    - `build_descriptions(sources, examples_entries, redirections: Optional[pl.DataFrame] =
      None)`.
  - `DownloadConfig.redirections_parquet`, default `./data/naics_redirections.parquet`, which
    `data preprocess` writes.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_redirections.py`:

```python
'''
The redirection table and the exclusion channel (Req 8).

Expected values are worked out by hand from the rules in ``data/redirections.py``.
'''

import polars as pl
import pytest

from naics_embedder.data.redirections import (
    REDIRECTIONS_SCHEMA,
    activity_phrase,
    build_redirections,
    exclusion_channel,
    lineal_codes,
    named_codes,
)

pytestmark = pytest.mark.unit

CODES = {'11', '111', '1111', '11111', '111110', '11112', '111120'}
NO_PARAGRAPHS = pl.DataFrame(schema={'code': pl.Utf8, 'text': pl.Utf8})

SOYBEANS = (
    'Growing soybeans for green manure--are classified in Industry 111120, Oilseed (except '
    'Soybean) Farming.'
)
COMBINATIONS = (
    'Growing oilseed and grain combinations--are classified in Industry Group 1111, Oilseed and '
    'Grain Farming.'
)
MANAGEMENT = 'Farm management services are classified in the Agriculture sector.'
PARAGRAPH = 'Excluded from this industry group are soybean farms, classified in Industry 111110.'

# -------------------------------------------------------------------------------------------------
# One row's parts
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'text, activity',
    [
        pytest.param(
            'Growing soybeans--are classified in Industry 111110, Soybean Farming.',
            'Growing soybeans',
            id='dashes',
        ),
        pytest.param(
            'Establishments primarily engaged in growing hay are classified in Industry 111940.',
            'Establishments primarily engaged in growing hay',
            id='sentence',
        ),
        pytest.param(
            'Growing hay--are lclassified in Industry 111940, Hay Farming.',
            'Growing hay',
            id='misspelled',
        ),
        pytest.param(
            'Tax return preparation is included in Industry 541213.',
            'Tax return preparation',
            id='is-included',
        ),
        pytest.param('See Industry 111110 for soybean farming.', None, id='no-redirection'),
        pytest.param('--are classified in Industry 111110.', None, id='no-activity'),
    ],
)
def test_activity_phrase_is_the_text_before_the_redirection(text, activity):
    assert activity_phrase(text) == activity

def test_named_codes_are_other_codebook_codes_in_order_of_first_appearance():
    text = (
        'Growing hay--are classified in Industry 111120, Industry Group 1111, Industry 111120 '
        'again, Industry 999999 or Industry 111110.'
    )

    assert named_codes('111110', text, CODES) == ['111120', '1111']
    # A combined sector is named by its first code
    assert named_codes('111110', 'Retailing--are classified in Sector 44-45.', {'44'}) == ['44']

def test_lineal_codes_are_named_ancestors_and_descendants():
    assert lineal_codes('111120', ['1111', '111110', '11']) == ['1111', '11']
    # 711's "Excluded" paragraph names its child 7113
    assert lineal_codes('711', ['7113', '722']) == ['7113']

# -------------------------------------------------------------------------------------------------
# The table
# -------------------------------------------------------------------------------------------------

def test_the_table_lists_every_row_once_in_order():
    references = pl.DataFrame(
        {
            'code': ['111110', '111120', '111120'],
            'text': [SOYBEANS, COMBINATIONS, MANAGEMENT]
        }
    )
    paragraphs = pl.DataFrame({'code': ['1111'], 'text': [PARAGRAPH]})

    table = build_redirections(references, paragraphs, CODES)

    soybeans = 'Growing soybeans for green manure'
    combinations = 'Growing oilseed and grain combinations'
    assert table.schema == pl.Schema(REDIRECTIONS_SCHEMA)
    assert table.rows() == [
        (0, 'cross_reference', '111110', SOYBEANS, soybeans, ['111120'], [], False),
        (1, 'cross_reference', '111120', COMBINATIONS, combinations, ['1111'], ['1111'], False),
        # Names no code: its text stays, but it has no activity phrase
        (2, 'cross_reference', '111120', MANAGEMENT, None, [], [], False),
        (3, 'description', '1111', PARAGRAPH, None, ['111110'], ['111110'], False),
    ]

def test_a_row_a_held_out_query_leaks_into_is_withheld_and_loses_its_activity():
    references = pl.DataFrame(
        {
            'code': ['111110', '111120', '111120'],
            'text': [
                # The first query reorders this activity phrase, but no sentence of the text
                'Establishments primarily engaged in growing soybeans for green manure are '
                'classified in Industry 111120.',
                'Growing hay--are classified in Industry 111110.',
                # The second query occurs in this text as whole words
                MANAGEMENT,
            ],
        }
    )
    queries = [
        'Growing soybeans for green manure, establishments primarily engaged in',
        'Farm management services',
    ]

    table = build_redirections(references, NO_PARAGRAPHS, CODES, queries)

    assert table.get_column('withheld').to_list() == [True, False, True]
    assert table.get_column('activity').to_list() == [None, 'Growing hay', None]
    # A withheld row still names its destination, so the pair stays an exclusion
    assert table.get_column('named_codes').to_list() == [['111120'], ['111110'], []]

# -------------------------------------------------------------------------------------------------
# The exclusion channel
# -------------------------------------------------------------------------------------------------

def test_the_channel_joins_each_kept_text_once_and_keeps_withheld_destinations():
    soybeans = 'Growing soybeans for green manure--are classified in Industry 111120.'
    hay = 'Growing hay--are classified in Industry 111940 or Industry 111120.'
    peanuts = 'Excluded are peanut farms, classified in Industry 111992.'
    table = pl.DataFrame(
        [
            (0, 'cross_reference', '111110', soybeans, 'x', ['111120'], [], False),
            (1, 'cross_reference', '111110', hay, 'x', ['111940', '111120'], [], False),
            (2, 'cross_reference', '111120', 'Growing soybeans.', None, ['111110'], [], True),
            (3, 'description', '111110', peanuts, None, ['111992'], [], False),
            (4, 'cross_reference', '111940', MANAGEMENT, None, [], [], False),
        ],
        schema=REDIRECTIONS_SCHEMA,
        orient='row',
    )

    channel = exclusion_channel(table)

    assert channel.rows() == [
        ('111110', f'{soybeans} {hay} {peanuts}', ['111120', '111940', '111992']),
        # Every row withheld: no text, but the destination stays an exclusion
        ('111120', None, ['111110']),
        # Names no code: text only
        ('111940', MANAGEMENT, None),
    ]
```

In `tests/unit/test_outcome_leakage.py`, replace:

```python
from naics_embedder.panels.leakage import (
    find_leakage,
    find_leakage_within,
    normalize_text,
    text_segments,
    training_text_segments,
)
```

with:

```python
from naics_embedder.panels.leakage import (
    find_leakage,
    find_leakage_within,
    leaking_texts,
    normalize_text,
    text_segments,
    training_text_segments,
)
```

and append to the end of the file:

```python

def test_leaking_texts_flags_each_text_a_query_leaks_into():
    segments = [
        ['growing soybeans', 'soybean farming'],
        ['hay farming'],
        [],
        ['corn farming and sweet corn'],
    ]

    # 'soybeans growing' reorders the first text's segment; 'sweet corn' occurs in the last
    flags = leaking_texts(segments, ['Soybeans, growing', 'Sweet corn'])

    assert flags.tolist() == [True, False, False, True]
    assert leaking_texts([], ['Sweet corn']).tolist() == []
    assert leaking_texts([['hay farming']], []).tolist() == [False]
```

In `tests/unit/test_index_roles.py`, append to the end of the file:

```python

def test_role_leakage_covers_the_extra_texts(role_rows):
    descriptions = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed',
            '111120': 'Oilseed farming'
        }
    )

    # The validation query 'Edamame' occurs in an activity phrase
    with pytest.raises(ValueError, match='held-out queries match training text'):
        verify_role_leakage(descriptions, role_rows, extra_texts=['Growing edamame'])
```

In `tests/unit/test_config.py`, replace:

```python
        with pytest.raises(ValidationError):
            DownloadConfig(index_roles_parquet='./data/roles.csv')
```

with:

```python
        with pytest.raises(ValidationError):
            DownloadConfig(index_roles_parquet='./data/roles.csv')
        with pytest.raises(ValidationError):
            DownloadConfig(redirections_parquet='./data/redirections.csv')
```

In `tests/unit/test_data_download.py`, replace:

```python
from naics_embedder.data import download_data
from naics_embedder.utils.config import DownloadConfig
from tests.fixtures.naics_sources import TITLES
```

with:

```python
from naics_embedder.data import download_data
from naics_embedder.data.redirections import REDIRECTIONS_SCHEMA
from naics_embedder.utils.config import DownloadConfig
from tests.fixtures.naics_sources import EXCLUSIONS, TITLES
```

Replace the whole of `test_get_exclusions_combines_crossrefs_and_descriptions`, from its
`def test_get_exclusions_combines_crossrefs_and_descriptions():` line through its last line,
`    assert set(row['excluded_codes']) == {'222', '333'}`, with the following (the
`@pytest.mark.unit` line above it stays):

```python
def test_exclusion_paragraphs_are_final_blocks_naming_an_exclusion_and_a_code():
    descriptions_3 = pl.DataFrame(
        {
            'code': ['111', '111', '222', '222', '333'],
            'description_id': pl.Series([1, 2, 1, 2, 1], dtype=pl.UInt32),
            'description': [
                'Some text',
                'Excluded are farms, classified in Industry 333.',
                'Excluded are ranches, classified in Industry 111.',  # not the last block
                'More text',
                'Excluded are orchards.',  # names no code
            ],
        }
    )

    paragraphs = download_data._exclusion_paragraphs(descriptions_3)

    assert paragraphs.rows() == [('111', 2, 'Excluded are farms, classified in Industry 333.')]
```

Replace:

```python
    assert examples['111191'] is None
    assert 'Illustrative' not in descriptions.filter(pl.col('code') == '11119')['description'][0]
```

with:

```python
    assert examples['111191'] is None
    assert 'Illustrative' not in descriptions.filter(pl.col('code') == '11119')['description'][0]

@pytest.mark.unit
def test_build_descriptions_builds_the_exclusion_channel_from_the_redirections(naics_sources):
    entries = download_data.naics_index_entries(naics_sources).filter(pl.col('entry_id') == 0)
    kept = download_data.naics_redirections(naics_sources)
    # The query reorders the activity phrase of 111110's only cross-reference
    query = 'Soybeans for green manure, growing'
    withheld = download_data.naics_redirections(naics_sources, [query])

    open_channel = download_data.build_descriptions(naics_sources, entries, kept)
    closed_channel = download_data.build_descriptions(naics_sources, entries, withheld)

    columns = ['excluded', 'excluded_codes']
    assert kept.get_column('withheld').to_list() == [False]
    assert open_channel.filter(pl.col('code') == '111110').select(columns).row(0) == (
        EXCLUSIONS[0][1], ['111120']
    )
    assert withheld.get_column('withheld').to_list() == [True]
    assert closed_channel.filter(pl.col('code') == '111110').select(columns).row(0) == (
        None, ['111120']
    )
```

In the `preprocess_cfg` fixture, replace:

```python
        index_roles_parquet=str(tmp_path / 'data' / 'naics_index_roles.parquet'),
        index_roles_csv=str(roles_csv),
```

with:

```python
        index_roles_parquet=str(tmp_path / 'data' / 'naics_index_roles.parquet'),
        redirections_parquet=str(tmp_path / 'data' / 'naics_redirections.parquet'),
        index_roles_csv=str(roles_csv),
```

Replace:

```python
    assert roles.row(3) == (3, '111110', 'Soybean seed production', 'test')
```

with:

```python
    assert roles.row(3) == (3, '111110', 'Soybean seed production', 'test')
    redirections = pl.read_parquet(preprocess_cfg.redirections_parquet)
    assert redirections.schema == pl.Schema(REDIRECTIONS_SCHEMA)
    assert redirections.select('reference_id', 'code', 'activity', 'withheld').rows() == [
        (0, '111110', 'Growing soybeans for green manure', False)
    ]
```

Replace:

```python
    # Every check runs before either file is written
    assert not Path(preprocess_cfg.output_parquet).exists()
    assert not Path(preprocess_cfg.index_roles_parquet).exists()
```

with:

```python
    # Every check runs before any file is written
    assert not Path(preprocess_cfg.output_parquet).exists()
    assert not Path(preprocess_cfg.index_roles_parquet).exists()
    assert not Path(preprocess_cfg.redirections_parquet).exists()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_redirections.py tests/unit/test_outcome_leakage.py tests/unit/test_index_roles.py tests/unit/test_data_download.py tests/unit/test_config.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named
'naics_embedder.data.redirections'` and `ImportError: cannot import name 'leaking_texts'`.

- [x] **Step 3: Implement the per-text leakage flags**

Append to the end of `src/naics_embedder/panels/leakage.py`:

```python

def leaking_texts(
    segments: Sequence[Sequence[str]],
    queries: Sequence[str],
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> np.ndarray:
    '''
    Flag each text that some query leaks into, under ``find_leakage``'s two rules.

    Args:
        segments: Each text's normalized segments, as ``text_segments`` returns them.
        queries: The queries, normalized here.
        min_jaccard: The near-duplicate threshold.

    Returns:
        One boolean per text: some query occurs in, or near-duplicates, one of its segments.
    '''

    _check_threshold(min_jaccard)
    normalized = _normalized_queries(queries)
    flags = np.zeros(len(segments), dtype=bool)
    owners = np.array([row for row, parts in enumerate(segments) for _ in parts], dtype=np.int64)
    flat = [part for parts in segments for part in parts]
    if not flat or not normalized:
        return flags
    padded = [f' {query} ' for query in normalized]
    # yapf: disable
    exact = (
        pl.DataFrame({'segment': [f' {part} ' for part in flat]})
        .select(pl.col('segment').str.extract_many(padded, overlapping=True).list.len() > 0)
        .to_series()
        .to_numpy()
    )
    # yapf: enable
    near = _near_duplicate_flags(flat, normalized, min_jaccard, same_list=False)
    np.logical_or.at(flags, owners, exact | near)
    return flags
```

In `src/naics_embedder/panels/index_roles.py`, replace:

```python
def verify_role_leakage(
    descriptions: pl.DataFrame,
    role_rows: pl.DataFrame,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> Dict[str, Dict[str, int]]:
    '''
    Match every validation and test query against all training text; fail on any match.

    Training text is every code's title, description, examples channel and exclusion text in
    ``descriptions``, plus every training-role query.

    Returns:
        Exact and near-duplicate match counts per held-out split (all zero on success).
    '''

    training = role_rows.filter(pl.col('role') == IndexRole.TRAINING.value)
    corpus = training_text_segments(descriptions, extra_texts=training.get_column('text'))
```

with:

```python
def verify_role_leakage(
    descriptions: pl.DataFrame,
    role_rows: pl.DataFrame,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
    *,
    extra_texts: Sequence[str] = (),
) -> Dict[str, Dict[str, int]]:
    '''
    Match every validation and test query against all training text; fail on any match.

    Training text is every code's title, description, examples channel and exclusion text in
    ``descriptions``, every training-role query, and ``extra_texts``: the redirection table's
    activity phrases, which Stage 7 trains on as queries (Req 8).

    Returns:
        Exact and near-duplicate match counts per held-out split (all zero on success).
    '''

    training = role_rows.filter(pl.col('role') == IndexRole.TRAINING.value)
    corpus = training_text_segments(
        descriptions, extra_texts=[*training.get_column('text').to_list(), *extra_texts]
    )
```

- [x] **Step 4: Implement the redirection table**

Create `src/naics_embedder/data/redirections.py`:

```python
'''
The redirection table (Req 8): every Census cross-reference once, with where it sends an activity.

A cross-reference reroutes an activity rather than asserting that two codes are unrelated:
"Growing soybeans--are classified in Industry 111110" sends soybean growing to 111110. The table
holds one row per cross-reference row, in file order, then one per "Excluded" paragraph harvested
from a description, in code order. Its columns:

- ``reference_id``: the row's position in the table, so cross-reference rows keep their file
  positions.
- ``source``: ``cross_reference`` or ``description``.
- ``code`` and ``text``: the referencing code and the row's text.
- ``activity``: the text before ``--`` or before " are/is classified" (or "included"), on a
  cross-reference row that names a code. Stage 7 trains on these phrases as queries.
- ``named_codes``: the codebook codes the text names other than its own code, in order of first
  appearance. On a cross-reference row these are its destinations.
- ``lineal_codes``: the named codes that are the row's code's ancestors or descendants. Lineal
  references stay text only and never act as negatives.
- ``withheld``: a held-out query leaks into one of the text's segments or into its activity
  phrase (Req 3), so the text leaves the exclusion channel and the activity phrase is dropped.
  The row stays in the table, and its named codes still count as exclusions.

The exclusion channel is built from the table: each code's texts that are not withheld, once
each, joined in table order.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import re
from fractions import Fraction
from typing import List, Optional, Sequence, Set

import polars as pl

from naics_embedder.panels.leakage import (
    NEAR_DUPLICATE_MIN_JACCARD,
    leaking_texts,
    normalize_text,
    text_segments,
)
from naics_embedder.utils.naics_hierarchy import code_lineage

logger = logging.getLogger(__name__)

CROSS_REFERENCE_SOURCE = 'cross_reference'
DESCRIPTION_SOURCE = 'description'
REDIRECTIONS_SCHEMA = {
    'reference_id': pl.Int64,
    'source': pl.Utf8,
    'code': pl.Utf8,
    'text': pl.Utf8,
    'activity': pl.Utf8,
    'named_codes': pl.List(pl.Utf8),
    'lineal_codes': pl.List(pl.Utf8),
    'withheld': pl.Boolean,
}

# "Growing soybeans--are classified in ...", "Establishments ... are classified in ...", and the
# source's one misspelling, "are lclassified"
_REDIRECTION = re.compile(r'(?:--|\s+)(?=(?:are|is)\s+l?(?:classified|included)\b)')
_CODE_REFERENCE = re.compile(r' (\d{2,6})')

# -------------------------------------------------------------------------------------------------
# One row's parts
# -------------------------------------------------------------------------------------------------

def activity_phrase(text: str) -> Optional[str]:
    '''The activity a cross-reference redirects: its text before the redirection, or None.'''

    match = _REDIRECTION.search(text)
    if match is None:
        return None
    return text[:match.start()].strip() or None

def named_codes(code: str, text: str, codes: Set[str]) -> List[str]:
    '''The codebook codes ``text`` names other than ``code``, in order of first appearance.'''

    named: List[str] = []
    for number in _CODE_REFERENCE.findall(text):
        if number in codes and number != code and number not in named:
            named.append(number)
    return named

def lineal_codes(code: str, named: Sequence[str]) -> List[str]:
    '''The named codes that are ``code``'s ancestors or descendants.'''

    return [other for other in named if other in code_lineage(code) or code in code_lineage(other)]

def _activity_segments(activity: Optional[str]) -> List[str]:
    normalized = normalize_text(activity or '')
    return [normalized] if normalized else []

# -------------------------------------------------------------------------------------------------
# The table and the channel
# -------------------------------------------------------------------------------------------------

def build_redirections(
    references: pl.DataFrame,
    paragraphs: pl.DataFrame,
    codes: Set[str],
    held_out_queries: Sequence[str] = (),
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> pl.DataFrame:
    '''
    The redirection table: ``references`` rows in order, then ``paragraphs`` rows in order.

    Args:
        references: The cross-reference file's rows (``code``, ``text``), in file order.
        paragraphs: The "Excluded" paragraphs harvested from descriptions (``code``, ``text``),
            in code order.
        codes: The codebook's codes.
        held_out_queries: Validation and test queries; a row any of them leaks into is withheld.
        min_jaccard: The near-duplicate threshold of the leakage check.

    Returns:
        One row per input row, with the columns of ``REDIRECTIONS_SCHEMA``.
    '''

    rows = []
    for source, frame in ((CROSS_REFERENCE_SOURCE, references), (DESCRIPTION_SOURCE, paragraphs)):
        for code, text in frame.select('code', 'text').iter_rows():
            named = named_codes(code, text, codes)
            activity = activity_phrase(text) if source == CROSS_REFERENCE_SOURCE and named else None
            rows.append((len(rows), source, code, text, activity, named, lineal_codes(code, named)))

    in_text = leaking_texts(
        [text_segments(row[3]) for row in rows], held_out_queries, min_jaccard=min_jaccard
    )
    in_activity = leaking_texts(
        [_activity_segments(row[4]) for row in rows], held_out_queries, min_jaccard=min_jaccard
    )
    withheld = in_text | in_activity
    table = pl.DataFrame(
        [
            (*row[:4], None if flag else row[4], *row[5:], bool(flag))
            for row, flag in zip(rows, withheld)
        ],
        schema=REDIRECTIONS_SCHEMA,
        orient='row',
    )

    naming = table.filter(pl.col('named_codes').list.len() > 0).height
    activities = table.get_column('activity').drop_nulls().len()
    lineal = int(table.get_column('lineal_codes').list.len().sum())
    withheld_ids = table.filter('withheld').get_column('reference_id').to_list()
    logger.info('Redirection table:')
    logger.info(f'  Cross-reference rows: {references.height: ,}')
    logger.info(f'  Excluded paragraphs from descriptions: {paragraphs.height: ,}')
    logger.info(f'  Rows naming a code: {naming: ,}')
    logger.info(f'  Activity phrases: {activities: ,}')
    logger.info(f'  Lineal references: {lineal: ,}')
    logger.info(f'  Withheld rows: {withheld_ids}\n')
    return table

def exclusion_channel(redirections: pl.DataFrame) -> pl.DataFrame:
    '''
    Each code's exclusion channel from the redirection table (Req 8(a)).

    Returns:
        One row per code with a redirection row, sorted by code: ``excluded``, the texts of its
        rows that are not withheld, each once, joined by one space in table order (null when all
        are withheld), and ``excluded_codes``, the codes its rows name, withheld rows included,
        each once in order of first appearance (null when none).
    '''

    # yapf: disable
    return (
        redirections
        .sort('reference_id')
        .group_by('code', maintain_order=True)
        .agg(
            excluded=pl.col('text').filter(~pl.col('withheld')),
            excluded_codes=pl.col('named_codes').flatten().drop_nulls().unique(maintain_order=True),
        )
        .select(
            code=pl.col('code'),
            excluded=pl.when(pl.col('excluded').list.len() > 0).then(
                pl.col('excluded').list.join(' ')
            ),
            excluded_codes=pl.when(pl.col('excluded_codes').list.len() > 0).then(
                pl.col('excluded_codes')
            ),
        )
        .sort('code')
    )
    # yapf: enable
```

(`drop_nulls()` is needed because Polars explodes an empty list into a null. Without it, a code
whose rows name nothing would carry `[None]`.)

- [x] **Step 5: Build the channel from the table in `data preprocess`**

In `src/naics_embedder/data/download_data.py`, replace:

```python
import polars as pl
import yaml

from naics_embedder.panels.index_roles import (
```

with:

```python
import polars as pl
import yaml

from naics_embedder.data.redirections import build_redirections, exclusion_channel
from naics_embedder.panels.index_roles import (
```

Replace the whole of `_get_exclusions`, from its `def _get_exclusions(` line through its final
`    return exclusions, descriptions_exclusions` line, with:

```python
def _exclusion_paragraphs(descriptions_3: pl.DataFrame) -> pl.DataFrame:
    '''
    Each code's "Excluded" paragraph (``code``, ``description_id``, ``description``).

    The paragraph is a code's last description block, when that block mentions an exclusion and
    names a code. It leaves the description and joins the redirection table (Req 8).
    '''

    # yapf: disable
    return (
        descriptions_3
        .filter(
            pl.col('description_id').max().over('code').eq(pl.col('description_id')),
            pl.col('description').str.contains_any(['Excluded', 'excluded', 'Exclude', 'exclude']),
            pl.col('description').str.contains(r' \d{2,6}'),
        )
        .select(
            code=pl.col('code').str.strip_chars(),
            description_id=pl.col('description_id'),
            description=pl.col('description'),
        )
    )
    # yapf: enable
```

Replace the whole of `build_descriptions`, from its `def build_descriptions(` line through the
`    # yapf: enable` line that closes its return statement, with:

```python
def naics_redirections(
    sources: NaicsSources,
    held_out_queries: Sequence[str] = (),
) -> pl.DataFrame:
    '''
    The redirection table of the Census files (``redirections.build_redirections``).

    Rows are the cross-reference file's rows in file order, then the "Excluded" paragraphs
    harvested from descriptions in code order. A row that one of ``held_out_queries`` leaks into
    is withheld.
    '''

    _, codes = _get_titles(sources.titles)
    _, descriptions_3 = _get_descriptions_1(sources.descriptions)
    references = sources.exclusions.select('code', text=pl.col('excluded'))
    paragraphs = _exclusion_paragraphs(descriptions_3).select('code', text=pl.col('description'))
    return build_redirections(references, paragraphs.sort('code'), codes, held_out_queries)

def build_descriptions(
    sources: NaicsSources,
    examples_entries: pl.DataFrame,
    redirections: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    '''
    One row per code: title, description and its source, examples channel and exclusion channel.

    Args:
        sources: The four Census files.
        examples_entries: The index entries that form examples channels (``entry_id``,
            ``code``, ``text``); every other entry of a code with index entries is a query.
        redirections: The redirection table whose rows form the exclusion channel
            (``redirections.exclusion_channel``); built from ``sources`` with no row withheld
            when omitted.
    '''

    titles, codes = _get_titles(sources.titles)

    descriptions_2, descriptions_3 = _get_descriptions_1(sources.descriptions)

    if redirections is None:
        redirections = naics_redirections(sources)
    exclusions = exclusion_channel(redirections)

    index_codes = set(_get_index_entries(sources.index, codes).get_column('code').to_list())
    examples, descriptions_examples = _get_examples(
        index_codes, examples_entries, descriptions_2, descriptions_3
    )

    descriptions = _get_descriptions_2(
        descriptions_3,
        _exclusion_paragraphs(descriptions_3).select('code', 'description_id'),
        descriptions_examples,
        codes,
    )

    # yapf: disable
    return (
        titles.join(descriptions, how='inner', on='code')
        .join(exclusions, how='left', on='code')
        .join(examples, how='left', on='code')
        .select(
            index=pl.col('index'),
            level=pl.col('level'),
            code=pl.col('code'),
            title=pl.col('title'),
            description=pl.col('description'),
            description_source=pl.col('description_source'),
            examples=pl.col('examples'),
            excluded=pl.col('excluded'),
            excluded_codes=pl.col('excluded_codes'),
        )
        .sort('index')
    )
    # yapf: enable
```

In `download_preprocess_data`, replace:

```python
    Build the descriptions parquet and the index-roles parquet from the Census files.

    Every index entry takes its role from the frozen role table (``cfg.index_roles_csv``). A
    code's examples channel holds its examples-role entries only, and no validation or test query
    may match any training text (Req 3); both are checked before anything is written.
```

with:

```python
    Build the descriptions, index-roles and redirection parquets from the Census files.

    Every index entry takes its role from the frozen role table (``cfg.index_roles_csv``). A
    code's examples channel holds its examples-role entries only. A redirection row that a
    validation or test query leaks into is withheld from the exclusion channel, and no held-out
    query may match any training text or activity phrase (Req 3). Everything is checked before
    anything is written.
```

Replace:

```python
    role_rows = attach_role_text(read_role_table(roles_csv), naics_index_entries(sources))

    naics_final = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value)
    )
```

with:

```python
    role_rows = attach_role_text(read_role_table(roles_csv), naics_index_entries(sources))
    held_out = role_rows.filter(
        pl.col('role').is_in([IndexRole.VALIDATION.value, IndexRole.TEST.value])
    )
    redirections = naics_redirections(sources, held_out.get_column('text').to_list())

    naics_final = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value), redirections
    )
```

Replace:

```python
    leakage = verify_role_leakage(naics_final, role_rows)
    logger.info(f'Held-out queries matching training text: {leakage}\n')
```

with:

```python
    activities = redirections.get_column('activity').drop_nulls().to_list()
    leakage = verify_role_leakage(naics_final, role_rows, extra_texts=activities)
    logger.info(f'Held-out queries matching training text or activity phrases: {leakage}\n')
```

Replace:

```python
    _parquet_stats(
        parquet_df=role_rows,
        message='NAICS index entries and their roles written to',
        output_parquet=cfg.index_roles_parquet,
        logger=logger,
    )

    return naics_final
```

with:

```python
    _parquet_stats(
        parquet_df=role_rows,
        message='NAICS index entries and their roles written to',
        output_parquet=cfg.index_roles_parquet,
        logger=logger,
    )

    Path(cfg.redirections_parquet).parent.mkdir(parents=True, exist_ok=True)
    redirections.write_parquet(cfg.redirections_parquet)

    _parquet_stats(
        parquet_df=redirections,
        message='NAICS redirection table written to',
        output_parquet=cfg.redirections_parquet,
        logger=logger,
    )

    return naics_final
```

In `src/naics_embedder/utils/config.py`, replace:

```python
    index_roles_parquet: str = Field(
        default='./data/naics_index_roles.parquet',
        description='Output path for every index entry with its text and role',
    )
```

with:

```python
    index_roles_parquet: str = Field(
        default='./data/naics_index_roles.parquet',
        description='Output path for every index entry with its text and role',
    )
    redirections_parquet: str = Field(
        default='./data/naics_redirections.parquet',
        description='Output path for the redirection table: every cross-reference once (Req 8)',
    )
```

and replace:

```python
    @field_validator('output_parquet', 'index_roles_parquet')
    @classmethod
    def validate_output_parquet(cls, value: str) -> str:
```

with:

```python
    @field_validator('output_parquet', 'index_roles_parquet', 'redirections_parquet')
    @classmethod
    def validate_output_parquet(cls, value: str) -> str:
```

In `conf/data/download.yaml`, replace:

```yaml
output_parquet: ./data/naics_descriptions.parquet
index_roles_parquet: ./data/naics_index_roles.parquet
```

with:

```yaml
output_parquet: ./data/naics_descriptions.parquet
index_roles_parquet: ./data/naics_index_roles.parquet
redirections_parquet: ./data/naics_redirections.parquet
```

`data/index_role_table.py` needs no edit: it calls `build_descriptions` without a table, so it
checks eligibility against every row's text, none withheld.

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_redirections.py tests/unit/test_outcome_leakage.py tests/unit/test_index_roles.py tests/unit/test_data_download.py tests/unit/test_config.py tests/unit/test_index_role_table.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1657 passed, 1 skipped` (14 new tests; one test replaced).

- [x] **Step 7: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/data/redirections.py src/naics_embedder/panels/leakage.py src/naics_embedder/panels/index_roles.py src/naics_embedder/data/download_data.py src/naics_embedder/utils/config.py tests/unit/test_redirections.py tests/unit/test_outcome_leakage.py tests/unit/test_index_roles.py tests/unit/test_data_download.py tests/unit/test_config.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 8: Commit**

```bash
git add src/naics_embedder/data/redirections.py \
  src/naics_embedder/panels/leakage.py \
  src/naics_embedder/panels/index_roles.py \
  src/naics_embedder/data/download_data.py \
  src/naics_embedder/utils/config.py \
  conf/data/download.yaml \
  tests/unit/test_redirections.py \
  tests/unit/test_outcome_leakage.py \
  tests/unit/test_index_roles.py \
  tests/unit/test_data_download.py \
  tests/unit/test_config.py
git commit -m "feat(data): a redirection table, each cross-reference once in the exclusion channel"
```

### Task 4: The backbone's trained input window

Req 9 wants every input to fit the backbone's trained window, found from the backbone's own
documentation (Verification "Backbone input window"). User decision 4 set it at 128 tokens. Today
titles are fixed at 24 tokens, the other channels at 512, and absent channels become the
placeholder text `[EMPTY]`, which encodes a code's level. After this task every channel is
tokenized at the window, an absent channel is the empty string with a `present` flag, and every
config that tokenizes refuses a longer window.

**Files:**
- Create: `src/naics_embedder/utils/input_window.py`
- Modify: `src/naics_embedder/utils/config.py` (imports; `TextOnlyConfig`, `TokenizationConfig`,
  `StreamingConfig`)
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py` (imports,
  `_tokenize_text`, `_build_tokenization_cache`, `_cache_identity`)
- Modify: `src/naics_embedder/panels/text_only.py` (imports, `build_text_only_table`)
- Modify: `conf/config.yaml`, `conf/data_loader/tokenization.yaml`,
  `conf/data/regressor_panel.yaml`
- Test: `tests/unit/test_input_window.py` (new), `tests/unit/test_config.py`,
  `tests/unit/test_tokenization_cache.py`, `tests/unit/test_text_only.py`,
  `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, in `naics_embedder.utils.input_window`:
  - `TRAINED_WINDOWS: Dict[str, int]`, holding `{'sentence-transformers/all-MiniLM-L6-v2':
    128}`;
  - `trained_window(backbone: str) -> int`, which raises `ValueError` ("no trained input
    window") for a backbone without a record;
  - `check_window(backbone: str, max_length: Optional[int]) -> int`, which returns the window
    for None and raises `ValueError` ("exceeds the trained input window") above it;
  - `token_counter(tokenizer) -> Callable[[List[str]], List[int]]`, counting tokens with the
    special tokens and without truncation;
  - `overflow_shares(texts: Mapping[str, Sequence[Optional[str]]], count_tokens, window: int) ->
    Dict[str, Dict[str, Any]]`, giving per channel `{'present': int, 'over': int, 'share':
    float}`.
- Cache entries: each channel dict gains `'present': bool` beside `input_ids` and
  `attention_mask`. The cache sidecar records `'cache_format': 'channels-v2'`.
- Config defaults: `TokenizationConfig.max_length` None resolves to 128, and
  `StreamingConfig.max_length` and `TextOnlyConfig.max_length` default to 128.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_input_window.py`:

```python
'''
The backbone's trained input window (Req 9; Verification "Backbone input window").
'''

import pytest
from transformers import BertTokenizerFast

from naics_embedder.utils.input_window import (
    check_window,
    overflow_shares,
    token_counter,
    trained_window,
)

pytestmark = pytest.mark.unit

BACKBONE = 'sentence-transformers/all-MiniLM-L6-v2'

def test_the_backbones_trained_window_is_128_tokens():
    # Its model card at revision 1110a243: "The sequence length was limited to 128 tokens."
    assert trained_window(BACKBONE) == 128

def test_a_backbone_without_a_recorded_window_is_refused():
    with pytest.raises(ValueError, match='no trained input window'):
        trained_window('some/other-backbone')

def test_a_null_max_length_is_the_window_and_a_longer_one_is_refused():
    assert check_window(BACKBONE, None) == 128
    assert check_window(BACKBONE, 64) == 64
    assert check_window(BACKBONE, 128) == 128
    with pytest.raises(ValueError, match='exceeds the trained input window'):
        check_window(BACKBONE, 129)

def test_overflow_shares_count_present_texts_beyond_the_window():

    def words_and_two_special_tokens(texts):
        return [len(text.split()) + 2 for text in texts]

    shares = overflow_shares(
        {
            'title': ['a b', 'a b c d e'],
            'excluded': [None, '  ', 'a b c'],
            'examples': [None, None],
        },
        words_and_two_special_tokens,
        window=5,
    )

    assert list(shares) == ['title', 'excluded', 'examples']
    assert shares['title'] == {'present': 2, 'over': 1, 'share': 0.5}
    # 'a b c' is five tokens: at the window, not beyond it
    assert shares['excluded'] == {'present': 1, 'over': 0, 'share': 0.0}
    assert shares['examples'] == {'present': 0, 'over': 0, 'share': 0.0}

def test_token_counts_include_the_special_tokens(tmp_path):
    tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'soybean', 'farming']
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(tokens))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))

    assert token_counter(tokenizer)(['soybean farming', 'soybean']) == [4, 3]
```

In `tests/unit/test_config.py`, replace:

```python
    SamplingConfig,
    SansStaticConfig,
    StructuralPreferenceConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    load_config,
)
```

with:

```python
    SamplingConfig,
    SansStaticConfig,
    StreamingConfig,
    StructuralPreferenceConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    TextOnlyConfig,
    TokenizationConfig,
    load_config,
)
```

Replace:

```python
        assert text_only.max_length == arm['data_loader']['tokenization']['max_length']
```

with:

```python
        assert text_only.max_length == arm['data_loader']['tokenization']['max_length']
        assert text_only.max_length == arm['data_loader']['streaming']['max_length']
```

and append to the end of the file:

```python

# -------------------------------------------------------------------------------------------------
# The backbone's trained input window (Req 9)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_every_tokenizing_config_defaults_to_the_trained_window():
    assert TokenizationConfig().max_length == 128
    assert StreamingConfig().max_length == 128
    assert TextOnlyConfig().max_length == 128

@pytest.mark.unit
@pytest.mark.parametrize('config_class', [TokenizationConfig, StreamingConfig, TextOnlyConfig])
def test_a_max_length_beyond_the_trained_window_is_refused(config_class):
    with pytest.raises(ValidationError, match='trained input window'):
        config_class(max_length=512)
```

In `tests/unit/test_tokenization_cache.py`, replace:

```python
from pathlib import Path
from unittest.mock import patch
```

with:

```python
import json
from pathlib import Path
from unittest.mock import patch
```

Replace:

```python
        assert title_input_ids.shape == (24, )
        assert desc_input_ids.shape == (128, )
```

with:

```python
        # Every channel at the window, titles included
        assert title_input_ids.shape == (128, )
        assert desc_input_ids.shape == (128, )
        assert title_dict['present'] is True
        assert item['excluded']['present'] is False  # type: ignore
```

Replace:

```python
        assert len(cache) == 1
        # Empty text should be replaced with [EMPTY]
        assert cache[0]['code'] == '311111'
```

with:

```python
        assert len(cache) == 1
        assert cache[0]['code'] == '311111'
        # A blank channel is absent: [CLS] [SEP] only, never a placeholder text
        for channel in ('title', 'description', 'excluded', 'examples'):
            assert cache[0][channel]['present'] is False  # type: ignore
            assert int(cache[0][channel]['attention_mask'].sum()) == 2  # type: ignore
```

Replace:

```python
        desc_dict = cache[0]['description']  # type: ignore
        desc_input_ids = desc_dict['input_ids']  # type: ignore
        desc_attention_mask = desc_dict['attention_mask']  # type: ignore
        assert desc_input_ids.shape == (64, )
        assert desc_attention_mask.shape == (64, )
```

with:

```python
        desc_dict = cache[0]['description']  # type: ignore
        desc_input_ids = desc_dict['input_ids']  # type: ignore
        desc_attention_mask = desc_dict['attention_mask']  # type: ignore
        assert desc_input_ids.shape == (64, )
        assert desc_attention_mask.shape == (64, )

    def test_null_channels_are_absent(self, tmp_path):
        '''A null channel is encoded like a blank one and flagged absent (Req 9).'''
        path = tmp_path / 'null_descriptions.parquet'
        pl.DataFrame(
            {
                'index': [0],
                'code': ['311111'],
                'title': ['Dog Food Manufacturing'],
                'description': [None],
                'excluded': [None],
                'examples': [None],
            },
            schema_overrides={
                'description': pl.Utf8,
                'excluded': pl.Utf8,
                'examples': pl.Utf8
            },
        ).write_parquet(path)

        cache = _build_tokenization_cache(str(path), 'sentence-transformers/all-MiniLM-L6-v2', 128)

        assert cache[0]['title']['present'] is True  # type: ignore
        for channel in ('description', 'excluded', 'examples'):
            assert cache[0][channel]['present'] is False  # type: ignore
            assert int(cache[0][channel]['attention_mask'].sum()) == 2  # type: ignore

    def test_a_window_beyond_the_trained_one_is_refused(self, sample_descriptions_parquet):
        with pytest.raises(ValueError, match='trained input window'):
            _build_tokenization_cache(
                sample_descriptions_parquet, 'sentence-transformers/all-MiniLM-L6-v2', 256
            )
```

and append to the end of the file:

```python

def test_a_cache_in_the_placeholder_format_is_rebuilt(
    tokenization_config, sample_tokenization_cache, counted_builds
):
    cache_path = Path(tokenization_config.output_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample_tokenization_cache, cache_path)
    # The sidecar a cache built before Stage 5 wrote: the same inputs, no channel format
    earlier = {
        **FINGERPRINTS,
        'tokenizer_name': tokenization_config.tokenizer_name,
        'max_length': tokenization_config.max_length,
    }
    cache_path.with_name(cache_path.name + '.meta.json').write_text(json.dumps(earlier))

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 1
```

In `tests/unit/test_text_only.py`, replace:

```python
    text_only_fingerprint,
)

pytestmark = pytest.mark.unit
```

with:

```python
    text_only_fingerprint,
)
from naics_embedder.utils.input_window import TRAINED_WINDOWS

pytestmark = pytest.mark.unit
```

Replace:

```python
def test_the_table_and_its_provenance_are_written(tmp_path, model, tokenizer):
    descriptions = tmp_path / 'naics_descriptions.parquet'
```

with:

```python
def test_the_table_and_its_provenance_are_written(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 16)
    descriptions = tmp_path / 'naics_descriptions.parquet'
```

Replace:

```python
def test_the_backbone_is_read_from_the_local_cache_only(monkeypatch, model, tokenizer):
```

with:

```python
def test_a_max_length_beyond_the_trained_window_is_refused(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setitem(TRAINED_WINDOWS, 'tiny-bert', 8)
    output = tmp_path / 'text_only.parquet'

    # The window is checked before the descriptions are read
    with pytest.raises(ValueError, match='trained input window'):
        build_text_only_table(
            tmp_path / 'naics_descriptions.parquet',
            output,
            backbone='tiny-bert',
            max_length=16,
            model=model,
            tokenizer=tokenizer,
        )
    assert not output.exists()

def test_the_backbone_is_read_from_the_local_cache_only(monkeypatch, model, tokenizer):
```

In `tests/unit/test_cli_commands.py`, replace:

```python
    assert calls == [
        (Path('descriptions.parquet'), output, 'sentence-transformers/all-MiniLM-L6-v2', 512, 32)
    ]
    # Rich folds long paths at the terminal's width (80 columns on CI), wherever it falls
    assert 'text_only_provenance.json' in result.output.replace('\n', '')
```

with:

```python
    assert calls == [
        (Path('descriptions.parquet'), output, 'sentence-transformers/all-MiniLM-L6-v2', 128, 32)
    ]
    # Rich folds long paths at the terminal's width (80 columns on CI), wherever it falls
    assert 'text_only_provenance.json' in result.output.replace('\n', '')

@pytest.mark.unit
def test_text_only_table_refuses_a_backbone_without_a_recorded_window(runner, tmp_path):
    output = tmp_path / 'text_only.parquet'

    result = runner.invoke(
        tools_cli.app,
        [
            'text-only-table', '--descriptions', 'descriptions.parquet', '--output',
            str(output), '--backbone', 'some/other-backbone'
        ],
    )

    assert result.exit_code == 1
    # Rich may wrap the message at a space; compare with the whitespace collapsed
    assert 'no trained input window' in ' '.join(result.output.split())
    assert not output.exists()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_input_window.py tests/unit/test_config.py tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py tests/unit/test_cli_commands.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named
'naics_embedder.utils.input_window'`.

- [x] **Step 3: Record the window**

Create `src/naics_embedder/utils/input_window.py`:

```python
'''
The backbone's trained input window (Req 9, "Input windows"; Verification "Backbone input window").

Inputs fit the window the backbone was trained on, recorded here from the backbone's own
documentation. For sentence-transformers/all-MiniLM-L6-v2, the model card at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 says that in training "the sequence length was limited
to 128 tokens". Its 256 (``sentence_bert_config.json``'s ``max_seq_length``, the truncation it
applies at inference) and 512 (``config.json``'s ``max_position_embeddings``) are not the trained
window. Every tokenizing path truncates to the window and refuses a longer ``max_length``, and
the supervision bundle records each channel's share of texts beyond it.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

# Tokens per text, [CLS] and [SEP] included
TRAINED_WINDOWS: Dict[str, int] = {'sentence-transformers/all-MiniLM-L6-v2': 128}

# -------------------------------------------------------------------------------------------------
# The window
# -------------------------------------------------------------------------------------------------

def trained_window(backbone: str) -> int:
    '''
    The backbone's trained input window, in tokens.

    Raises:
        ValueError: If no window is recorded for the backbone.
    '''

    if backbone not in TRAINED_WINDOWS:
        raise ValueError(
            f'no trained input window is recorded for {backbone!r}; record it in '
            "utils/input_window.py from the backbone's documentation"
        )
    return TRAINED_WINDOWS[backbone]

def check_window(backbone: str, max_length: Optional[int]) -> int:
    '''
    The length to tokenize at: ``max_length``, or the trained window when it is None.

    Raises:
        ValueError: If ``max_length`` exceeds the backbone's trained window, or none is recorded.
    '''

    window = trained_window(backbone)
    if max_length is None:
        return window
    if max_length > window:
        raise ValueError(
            f'max_length {max_length} exceeds the trained input window of {backbone} '
            f'({window} tokens)'
        )
    return max_length

# -------------------------------------------------------------------------------------------------
# Texts beyond the window
# -------------------------------------------------------------------------------------------------

def token_counter(tokenizer: Any) -> Callable[[List[str]], List[int]]:
    '''Token counts under ``tokenizer``, special tokens included, without truncation.'''

    def count(texts: List[str]) -> List[int]:
        return [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]

    return count

def overflow_shares(
    texts: Mapping[str, Sequence[Optional[str]]],
    count_tokens: Callable[[List[str]], List[int]],
    window: int,
) -> Dict[str, Dict[str, Any]]:
    '''
    Each channel's share of present texts longer than ``window`` tokens.

    Args:
        texts: Each channel's texts; a null or blank text is absent and not counted.
        count_tokens: Token counts of a list of texts, special tokens included.
        window: The trained window.

    Returns:
        Per channel: ``present`` texts, ``over`` (longer than the window) and their ``share``.
    '''

    shares: Dict[str, Dict[str, Any]] = {}
    for channel, values in texts.items():
        present = [text for text in values if text is not None and text.strip()]
        over = sum(count > window for count in count_tokens(present)) if present else 0
        shares[channel] = {
            'present': len(present),
            'over': int(over),
            'share': over / len(present) if present else 0.0,
        }
    return shares
```

In `src/naics_embedder/utils/config.py`, replace:

```python
from naics_embedder.supervision.schema import CONTRACT_VERSION
```

with:

```python
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.utils.input_window import check_window
```

Replace:

```python
    max_length: int = Field(
        default=512,
        ge=1,
        description="Tokens kept per channel text: the arm's data_loader max_length",
    )
    batch_size: int = Field(default=32, ge=1, description='Texts per forward pass')
```

with:

```python
    max_length: int = Field(
        default=128,
        ge=1,
        description="Tokens kept per channel text: the arm's data_loader max_length",
    )
    batch_size: int = Field(default=32, ge=1, description='Texts per forward pass')

    @model_validator(mode='after')
    def fit_the_trained_window(self) -> 'TextOnlyConfig':
        '''Refuse a max_length beyond the backbone's trained window (Req 9).'''

        check_window(self.backbone, self.max_length)
        return self
```

Replace:

```python
    max_length: Optional[int] = Field(
        default=None, description='Maximum sequence length (None = use model default)'
    )
    output_path: str = Field(
        default='./data/token_cache/token_cache.pt', description='Path to save tokenization cache'
    )
```

with:

```python
    max_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Tokens kept per channel text; None is the backbone's trained window",
    )
    output_path: str = Field(
        default='./data/token_cache/token_cache.pt', description='Path to save tokenization cache'
    )

    @model_validator(mode='after')
    def fit_the_trained_window(self) -> 'TokenizationConfig':
        '''Resolve a null max_length to the trained window, and refuse a longer one (Req 9).'''

        self.max_length = check_window(self.tokenizer_name, self.max_length)
        return self
```

Replace:

```python
    max_length: int = Field(default=512, description='Maximum sequence length for tokenization')
```

with:

```python
    max_length: int = Field(
        default=128,
        ge=1,
        description="Tokens kept per channel text, at most the backbone's trained window",
    )
```

Replace:

```python
        if self.n_negatives_phase1 > self.n_candidates:
            raise ValueError(
                f'n_negatives_phase1 ({self.n_negatives_phase1}) must be <= '
                f'n_candidates ({self.n_candidates})'
            )
        return self
```

with:

```python
        if self.n_negatives_phase1 > self.n_candidates:
            raise ValueError(
                f'n_negatives_phase1 ({self.n_negatives_phase1}) must be <= '
                f'n_candidates ({self.n_candidates})'
            )
        return self

    @model_validator(mode='after')
    def fit_the_trained_window(self) -> 'StreamingConfig':
        '''Refuse a max_length beyond the backbone's trained window (Req 9).'''

        check_window(self.tokenizer_name, self.max_length)
        return self
```

In `conf/config.yaml`, replace:

```yaml
  tokenization:
    tokenizer_name: sentence-transformers/all-MiniLM-L6-v2
    max_length: 512
```

with:

```yaml
  tokenization:
    tokenizer_name: sentence-transformers/all-MiniLM-L6-v2
    max_length: 128  # The backbone's trained window (utils/input_window.py)
```

and replace:

```yaml
    tokenizer_name: sentence-transformers/all-MiniLM-L6-v2
    max_length: 512
    seed: 42
```

with:

```yaml
    tokenizer_name: sentence-transformers/all-MiniLM-L6-v2
    max_length: 128  # The backbone's trained window (utils/input_window.py)
    seed: 42
```

In `conf/data_loader/tokenization.yaml`, replace:

```yaml
max_length: null  # Use model default
```

with:

```yaml
max_length: 128  # The backbone's trained window (utils/input_window.py)
```

In `conf/data/regressor_panel.yaml`, replace:

```yaml
  max_length: 512
```

with:

```yaml
  max_length: 128
```

- [x] **Step 4: Tokenize every channel at the window, absent channels as the empty string**

In `src/naics_embedder/text_model/dataloader/tokenization_cache.py`, replace:

```python
from naics_embedder.utils.config import TokenizationConfig

logger = logging.getLogger(__name__)
```

with:

```python
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window

logger = logging.getLogger(__name__)

# How channels are encoded: every channel at the window, and an absent one as the empty string
# with ``present`` False. A cache in the earlier format (a placeholder text, titles at 24 tokens)
# records no format in its sidecar, so it is rebuilt.
CACHE_FORMAT = 'channels-v2'
```

Replace the whole of `_tokenize_text`, from its `def _tokenize_text(` line through its
`    return encoding, counter` line, with:

```python
def _tokenize_text(
    row: Dict[str, Any],
    field: str,
    counter: Dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    '''
    Tokenize one channel text, truncated and padded to ``max_length``.

    An absent channel (null or blank) is encoded as the empty string, ``[CLS] [SEP]``, never as
    placeholder text, and its ``present`` flag is False so fusion can mask it (Req 9).
    '''

    text = row.get(field) or ''
    present = bool(text.strip())
    if present:
        counter[field] += 1
    else:
        text = ''

    encoded = tokenizer(
        text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
    )

    encoding = {
        'input_ids': torch.squeeze(encoded['input_ids']),  # type: ignore
        'attention_mask': torch.squeeze(encoded['attention_mask']),  # type: ignore
        'present': present,
    }

    return encoding, counter
```

Replace:

```python
def _build_tokenization_cache(descriptions_path: str, tokenizer_name: str,
                              max_length: int) -> Dict[int, Dict[str, torch.Tensor]]:
    '''Build tokenization cache from descriptions file.'''

    logger.info('Building tokenization cache...')
```

with:

```python
def _build_tokenization_cache(
    descriptions_path: str, tokenizer_name: str, max_length: Optional[int]
) -> Dict[int, Dict[str, Any]]:
    '''
    Build tokenization cache from descriptions file.

    Every channel, titles included, is truncated and padded to ``max_length``, which may not
    exceed the backbone's trained window (None is the window).
    '''

    max_length = check_window(tokenizer_name, max_length)
    logger.info('Building tokenization cache...')
```

Replace:

```python
        title, cnt = _tokenize_text(row, 'title', cnt, tokenizer, 24)
```

with:

```python
        title, cnt = _tokenize_text(row, 'title', cnt, tokenizer, max_length)
```

Replace:

```python
        'tokenizer_name': cfg.tokenizer_name,
        'max_length': cfg.max_length,
    }
```

with:

```python
        'tokenizer_name': cfg.tokenizer_name,
        'max_length': cfg.max_length,
        'cache_format': CACHE_FORMAT,
    }
```

- [x] **Step 5: The text-only builder refuses a longer window**

In `src/naics_embedder/panels/text_only.py`, replace:

```python
from naics_embedder.supervision.artifacts import sha256_file
```

with:

```python
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.input_window import check_window
```

Replace:

```python
    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
        The table's path.
    '''

    descriptions_path = Path(descriptions_path)
```

with:

```python
    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
        The table's path.

    Raises:
        ValueError: If ``max_length`` exceeds the backbone's trained input window (Req 9).
    '''

    check_window(backbone, max_length)
    descriptions_path = Path(descriptions_path)
```

The `tools text-only-table` command needs no edit: it already reports a `ValueError` from the
builder and exits with code 1, as its new test checks for a `--backbone` without a window.

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_input_window.py tests/unit/test_config.py tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py tests/unit/test_cli_commands.py tests/unit/test_datamodule.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1671 passed, 1 skipped` (14 new tests).

- [x] **Step 7: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/utils/input_window.py src/naics_embedder/utils/config.py src/naics_embedder/text_model/dataloader/tokenization_cache.py src/naics_embedder/panels/text_only.py tests/unit/test_input_window.py tests/unit/test_config.py tests/unit/test_tokenization_cache.py tests/unit/test_text_only.py tests/unit/test_cli_commands.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 8: Commit**

```bash
git add src/naics_embedder/utils/input_window.py \
  src/naics_embedder/utils/config.py \
  src/naics_embedder/text_model/dataloader/tokenization_cache.py \
  src/naics_embedder/panels/text_only.py \
  conf/config.yaml \
  conf/data_loader/tokenization.yaml \
  conf/data/regressor_panel.yaml \
  tests/unit/test_input_window.py \
  tests/unit/test_config.py \
  tests/unit/test_tokenization_cache.py \
  tests/unit/test_text_only.py \
  tests/unit/test_cli_commands.py
git commit -m "feat(text): every channel at the backbone's trained 128-token window, no placeholder"
```

### Task 5: Cross-sector pairs are read from their relation label

Today a pair is cross-sector when its distance is the sentinel 99, and the generator's positives
exclude "the maximal distance". Under D* (Task 6) cross-sector pairs carry real distances, so every
reader must find them by their relation label, 99, instead. On today's data the two coincide
exactly: all 1,984,647 cross-sector pairs of bundle 18403d29 have both. This task is therefore a
pure refactor, and the existing tests pass unchanged. The new tests give cross-sector pairs a
D*-like distance, which only the label identifies.

**Files:**
- Modify: `src/naics_embedder/supervision/schema.py` (the structural margin contract)
- Modify: `src/naics_embedder/data/compute_relations.py` (imports; the two constants move out)
- Modify: `src/naics_embedder/data/create_triplets.py` (docstring, imports, `_positive_pairs`,
  `_structural_margins`, `_cap_cross_sector`, `_project`, `iter_training_pair_batches`,
  `build_training_pairs`)
- Modify: `src/naics_embedder/supervision/margins.py` (docstring, imports,
  `structural_margins`)
- Test: `tests/unit/test_data_triplets.py`, `tests/unit/test_structural_margins.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - In `naics_embedder.supervision.schema`: `CROSS_SECTOR_RELATION_ID = 99` and
    `CROSS_SECTOR_RELATION_NAME = 'cross_sector'`, which
    `naics_embedder.data.compute_relations` re-imports.
  - In `naics_embedder.data.create_triplets`: `_positive_pairs(anchor_view: pl.DataFrame) ->
    pl.DataFrame`, which no longer takes `max_distance`.
  - `supervision.margins.structural_margins` and `structurally_eligible` keep their signatures
    and read cross-sector from `negative_relation_id`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_data_triplets.py`, replace:

```python
from naics_embedder.data.create_triplets import (
    _structural_margins,
    _validate_training_pairs,
    build_training_pairs,
)
```

with:

```python
from naics_embedder.data.create_triplets import (
    _anchor_view,
    _positive_pairs,
    _structural_margins,
    _validate_training_pairs,
    build_training_pairs,
)
```

and replace:

```python
# -------------------------------------------------------------------------------------------------
# Deterministic cross-sector cap
# -------------------------------------------------------------------------------------------------
```

with:

```python
@pytest.fixture
def labelled_cross_sector_pair_facts() -> pl.DataFrame:
    # Codes 0 '111111' and 1 '111112' are siblings; 2 '22' is another sector and 3 '222222' its
    # six-digit descendant. The cross-sector pairs carry D* (6 or 10), not a sentinel, so only
    # their relation label, 99, marks them.
    return pl.DataFrame(
        {
            'code_i_id': [0, 0, 0, 1, 1, 2],
            'code_j_id': [1, 2, 3, 2, 3, 3],
            'code_i': ['111111', '111111', '111111', '111112', '111112', '22'],
            'code_j': ['111112', '22', '222222', '22', '222222', '222222'],
            'structural_distance': [2.0, 6.0, 10.0, 6.0, 10.0, 4.0],
            'structural_relation_id': [2, 99, 99, 99, 99, 6],
            'structural_relation_name': [
                'sibling',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'cross_sector',
                'great-great-grandchild',
            ],
            'code_i_excludes_code_j': [False] * 6,
            'code_j_excludes_code_i': [False] * 6,
            'is_explicit_exclusion': [False] * 6,
        },
        schema_overrides={
            'code_i_id': pl.Int32,
            'code_j_id': pl.Int32,
            'structural_distance': pl.Float32,
            'structural_relation_id': pl.Int16,
        },
    )

def test_cross_sector_pairs_are_read_from_their_relation_label(labelled_cross_sector_pair_facts):
    positives = _positive_pairs(_anchor_view(labelled_cross_sector_pair_facts))
    pairs = build_training_pairs(labelled_cross_sector_pair_facts)
    capped = build_training_pairs(labelled_cross_sector_pair_facts, cross_sector_cap=1)

    # A cross-sector pair is never a positive, whatever its distance
    assert positives.select('anchor_code_id', 'positive_code_id').rows() == [(0, 1), (2, 3)]
    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3)]
    assert pairs.get_column('relation_margin').to_list() == [15.0, 15.0]
    assert pairs.get_column('distance_margin').to_list() == [10.0, 10.0]
    assert pairs.get_column('unrelated').to_list() == [True, True]
    assert capped.height == 1

# -------------------------------------------------------------------------------------------------
# Deterministic cross-sector cap
# -------------------------------------------------------------------------------------------------
```

A test comment still calls positives "non-maximal". Replace:

```python
    # Positives are canonical, non-maximal, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is an
    # exclusion. A negative j needs rows positive -> j and anchor -> j.
```

with:

```python
    # Positives are canonical, within-sector, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is
    # an exclusion. A negative j needs rows positive -> j and anchor -> j.
```

In `tests/unit/test_structural_margins.py`, replace:

```python
        # Cross-sector negatives always receive the fixed legacy margins.
        ((99.0, 99), (0.5, 1), (15.0, 10.0)),
```

with:

```python
        # Cross-sector negatives always receive the fixed legacy margins.
        ((99.0, 99), (0.5, 1), (15.0, 10.0)),
        # The relation label marks a cross-sector negative, whatever its distance.
        ((10.0, 99), (2.0, 2), (15.0, 10.0)),
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py -q`
Expected: FAIL. `_positive_pairs()` is missing its `max_distance` argument, and the new
margins case gives `(97.0, 8.0)`.

- [x] **Step 3: Key every cross-sector test on the label**

In `src/naics_embedder/supervision/schema.py`, replace:

```python
CROSS_SECTOR_DISTANCE = 99.0
CROSS_SECTOR_RELATION_MARGIN = 15.0
```

with:

```python
CROSS_SECTOR_DISTANCE = 99.0
# The relation label cross-sector pairs carry: every reader finds them by it, not by a distance
CROSS_SECTOR_RELATION_ID = 99
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
CROSS_SECTOR_RELATION_MARGIN = 15.0
```

In `src/naics_embedder/data/compute_relations.py`, replace:

```python
import networkx as nx
import polars as pl

logger = logging.getLogger(__name__)
```

with:

```python
import networkx as nx
import polars as pl

from naics_embedder.supervision.schema import CROSS_SECTOR_RELATION_ID, CROSS_SECTOR_RELATION_NAME

logger = logging.getLogger(__name__)
```

and replace:

```python
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
CROSS_SECTOR_RELATION_ID = 99

def compute_structural_relations(
```

with:

```python
def compute_structural_relations(
```

In `src/naics_embedder/data/create_triplets.py`, replace:

```python
Positive/negative combinatorics reproduce the legacy generator: a positive is a canonical,
non-maximal, non-exclusion pair; a negative ``j`` for (anchor ``a``, positive ``p``) requires the
directed rows ``p -> j`` and ``a -> j``; cross-sector negatives are capped per (anchor, positive).
```

with:

```python
Positive/negative combinatorics reproduce the legacy generator: a positive is a canonical,
within-sector, non-exclusion pair; a negative ``j`` for (anchor ``a``, positive ``p``) requires the
directed rows ``p -> j`` and ``a -> j``; cross-sector negatives are capped per (anchor, positive).
A pair is cross-sector when it carries the ``cross_sector`` relation label.
```

Replace:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE,
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_MARGIN,
```

with:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
```

Replace:

```python
def _positive_pairs(anchor_view: pl.DataFrame, max_distance: float) -> pl.DataFrame:
    '''Canonical, non-maximal, non-exclusion pairs; reversed rows never become positives.'''

    return anchor_view.filter(
        ~pl.col('is_reversed'),
        pl.col('structural_distance').gt(0.0),
        pl.col('structural_distance').ne(max_distance),
        ~pl.col('is_explicit_exclusion'),
    ).select(
```

with:

```python
def _positive_pairs(anchor_view: pl.DataFrame) -> pl.DataFrame:
    '''Canonical, within-sector, non-exclusion pairs; reversed rows never become positives.'''

    return anchor_view.filter(
        ~pl.col('is_reversed'),
        pl.col('structural_distance').gt(0.0),
        pl.col('structural_relation_id').ne(CROSS_SECTOR_RELATION_ID),
        ~pl.col('is_explicit_exclusion'),
    ).select(
```

Replace:

```python
    cross_sector = pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE)
    return frame.with_columns(
```

with:

```python
    cross_sector = pl.col('negative_structural_relation_id').eq(CROSS_SECTOR_RELATION_ID)
    return frame.with_columns(
```

Replace:

```python
    capped = (
        pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE)
        & ~pl.col('negative_is_explicit_exclusion')
    )
```

with:

```python
    capped = (
        pl.col('negative_structural_relation_id').eq(CROSS_SECTOR_RELATION_ID)
        & ~pl.col('negative_is_explicit_exclusion')
    )
```

Replace:

```python
        pl.col('negative_structural_distance').eq(CROSS_SECTOR_DISTANCE).alias('unrelated'),
```

with:

```python
        pl.col('negative_structural_relation_id').eq(CROSS_SECTOR_RELATION_ID).alias('unrelated'),
```

Replace:

```python
    anchor_view = _anchor_view(pair_facts)
    max_distance = pair_facts.get_column('structural_distance').max()
    positives = _positive_pairs(anchor_view, max_distance)
```

with:

```python
    anchor_view = _anchor_view(pair_facts)
    positives = _positive_pairs(anchor_view)
```

Replace:

```python
        _positive_pairs(anchor_view, 0.0).head(0),
```

with:

```python
        _positive_pairs(anchor_view).head(0),
```

In `src/naics_embedder/supervision/margins.py`, replace:

```python
negative must be structurally farther from the anchor than the positive. Cross-sector
negatives receive fixed margins; equal distances and the -0.5 lineal adjustment receive fixed
```

with:

```python
negative must be structurally farther from the anchor than the positive. Cross-sector
negatives, those with the ``cross_sector`` relation label, receive fixed margins; equal
distances and the -0.5 lineal adjustment receive fixed
```

Replace:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE,
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_MARGIN,
```

with:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
```

Replace:

```python
    negative_distance = negative_distance.to(torch.float32)
    relation_delta = negative_relation_id.to(torch.float32) - positive_relation_id.to(torch.float32)
    distance_delta = negative_distance - positive_distance.to(torch.float32)
    negative_distance, relation_delta, distance_delta = torch.broadcast_tensors(
        negative_distance, relation_delta, distance_delta
    )
    cross_sector = negative_distance.eq(CROSS_SECTOR_DISTANCE)
```

with:

```python
    negative_relation_id = negative_relation_id.to(torch.float32)
    relation_delta = negative_relation_id - positive_relation_id.to(torch.float32)
    distance_delta = negative_distance.to(torch.float32) - positive_distance.to(torch.float32)
    negative_relation_id, relation_delta, distance_delta = torch.broadcast_tensors(
        negative_relation_id, relation_delta, distance_delta
    )
    cross_sector = negative_relation_id.eq(CROSS_SECTOR_RELATION_ID)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py tests/unit/test_data_relations.py tests/unit/test_negative_selection.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1673 passed, 1 skipped` (2 new tests).

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/schema.py src/naics_embedder/data/compute_relations.py src/naics_embedder/data/create_triplets.py src/naics_embedder/supervision/margins.py tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/supervision/schema.py \
  src/naics_embedder/data/compute_relations.py \
  src/naics_embedder/data/create_triplets.py \
  src/naics_embedder/supervision/margins.py \
  tests/unit/test_data_triplets.py \
  tests/unit/test_structural_margins.py
git commit -m "refactor(supervision): find cross-sector pairs by their relation label, not distance 99"
```

### Task 6: D* is the bundle's distance, under contract `stage3-supervision-v2`

The bundle's distances come from the networkx walk in `data/compute_distances.py`: a half-step
below every lineal pair and the constant 99 across sectors. Req 7 replaces both with D*, the tree
path length through a virtual root above the sectors, which Task 1's `tree_distance_matrix`
computes. This task makes that function the bundle's only distance, removes the margin special
cases that existed for the half-step and the constant, and adds a validator that fails closed on
any stored distance that is not D*. D* changes what `structural_distance` means, so the contract
moves to `stage3-supervision-v2` in the same task: no v1 loader can read a D* bundle, and no v2
loader a v1 bundle.

The validator (`validate_tree_distances`) runs inside `validate_structural_pairs`, so it runs
wherever pair facts are checked: twice at generation and again at every load. Its checks, in
order, are:

1. Every distance is an integer, since D* has no half-step.
2. No distance is 99.
3. Every pair across sectors has λ(i) + λ(j) − 2, where λ is the number of digits.
4. The `cross_sector` relation label marks exactly the pairs across sectors.
5. The stored distances satisfy the triangle inequality over every ordered triple.
6. Every distance equals `tree_distance_matrix` on its pair: the path length through the pair's
   lowest common ancestor.

Each check can fail on its own. Checks 3 and 5 run before the full comparison so that a table
breaking them is reported as such. The build records four new validation results.

The five-code test fixture keeps its hand-written relation labels, but its distances become the
D* of its codes: 2 between the three `1111xx` siblings and 10 across sectors.

**Files:**
- Modify: `src/naics_embedder/data/compute_distances.py` (overwritten: D* for every canonical
  pair; the networkx helpers go)
- Modify: `src/naics_embedder/supervision/artifacts.py` (imports, `validate_structural_pairs`,
  new `validate_tree_distances`)
- Modify: `src/naics_embedder/data/supervision_bundle.py` (`validate_pair_facts`' results)
- Modify: `src/naics_embedder/supervision/schema.py` (contract version, margin constants)
- Modify: `src/naics_embedder/supervision/margins.py`, `src/naics_embedder/data/create_triplets.py`
  (margins for integer D*)
- Modify (contract literal only): `src/naics_embedder/supervision/mode.py`,
  `src/naics_embedder/utils/config.py`, `src/naics_embedder/cli/commands/data.py`,
  `conf/config.yaml`, `conf/data/supervision.yaml`
- Test: `tests/fixtures/supervision.py`, `tests/unit/test_data_distances.py` (overwritten),
  `tests/unit/test_data_triplets.py`, `tests/unit/test_structural_margins.py`,
  `tests/unit/test_supervision_artifacts.py`, `tests/unit/test_supervision_index.py`,
  `tests/unit/test_hard_negative_mining.py`, `tests/unit/test_hgcn_streaming_dataset.py`,
  `tests/unit/test_streaming_sampling.py`, `tests/integration/test_stage3_training_step.py`
- Test (contract literal only): `tests/unit/test_checkpoint_contract.py`,
  `tests/unit/test_cli_commands.py`, `tests/unit/test_cli_training.py`,
  `tests/unit/test_config.py`, `tests/unit/test_streaming_dataset.py`,
  `tests/unit/test_supervision_schema.py`

**Interfaces:**
- Consumes:
  - From Task 1, in `naics_embedder.utils.naics_hierarchy`:
    `tree_distance_matrix(codes: Sequence[str]) -> np.ndarray` (int64, zero diagonal) and
    `code_lineage(code: str) -> Tuple[str, ...]` (sector first).
  - From Task 5, in `naics_embedder.supervision.schema`: `CROSS_SECTOR_RELATION_ID = 99`.
- Produces:
  - `naics_embedder.supervision.schema.CONTRACT_VERSION == 'stage3-supervision-v2'`; bundles
    default to `./data/supervision/stage3-supervision-v2`.
  - `naics_embedder.supervision.artifacts.validate_tree_distances(structural: pl.DataFrame) ->
    None`, which raises `ValueError`. `validate_structural_pairs` calls it last.
  - Four more keys in the manifest's `validation_results`: `distance_is_d_star`,
    `cross_sector_distance_formula`, `cross_sector_relation_label` and
    `distance_triangle_inequality`.
  - The schema's margin constants are now only `CROSS_SECTOR_RELATION_ID`,
    `CROSS_SECTOR_RELATION_NAME`, `CROSS_SECTOR_RELATION_MARGIN` (15.0) and
    `EQUAL_DISTANCE_MARGIN` (0.3333). `CROSS_SECTOR_DISTANCE`, `CROSS_SECTOR_DISTANCE_MARGIN`,
    `LINEAL_ADJUSTED_DISTANCE_MARGIN` and `LINEAL_DISTANCE_DELTA` are removed.
  - `compute_structural_distances(input_parquet, cfg)` keeps its signature and columns.

- [x] **Step 1: Write the failing tests**

Point every test at the new contract. `test_supervision_artifacts.py` is left out on purpose: it
names v2 as the *wrong* version, which the next edit handles.

Run: `sed -i '' 's/stage3-supervision-v1/stage3-supervision-v2/g' tests/fixtures/supervision.py tests/unit/test_checkpoint_contract.py tests/unit/test_cli_commands.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_streaming_dataset.py tests/unit/test_supervision_index.py tests/unit/test_supervision_schema.py`

Run: `grep -rn 'stage3-supervision-v1' tests`
Expected: no output. The next edit brings v1 back into one test, as the old contract that a v2
bundle must refuse.

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
    with pytest.raises(ValueError, match='expected supervision contract stage3-supervision-v2'):
        load_validated_bundle(generated_bundle, expected_contract='stage3-supervision-v2')
```

with:

```python
    with pytest.raises(ValueError, match='expected supervision contract stage3-supervision-v1'):
        load_validated_bundle(generated_bundle, expected_contract='stage3-supervision-v1')
```

In `tests/fixtures/supervision.py`, replace:

```python
        | {'structural_distance': [
            0.5,
            2.0,
            99.0,
            99.0,
            3.0,
            99.0,
            99.0,
            99.0,
            99.0,
            99.0,
        ]}
```

with:

```python
        | {'structural_distance': [
            2.0,
            2.0,
            10.0,
            10.0,
            2.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]}
```

and replace:

```python
            'structural_distance': [
                0.5,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ],
```

with:

```python
            'structural_distance': [
                2.0,
                2.0,
                10.0,
                10.0,
                2.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ],
```

Overwrite `tests/unit/test_data_distances.py` with:

```python
'''
Unit tests for the structural distances: D* (Req 7) for every canonical pair of codes.
'''

import polars as pl
import pytest

from naics_embedder.data.compute_distances import compute_structural_distances
from naics_embedder.utils.config import DistancesConfig
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix

# -------------------------------------------------------------------------------------------------
# Structural-only distances
# -------------------------------------------------------------------------------------------------

def _distance(frame: pl.DataFrame, code_i: str, code_j: str) -> float:
    return frame.filter(pl.col('code_i').eq(code_i)
                        & pl.col('code_j').eq(code_j)).item(0, 'structural_distance')

@pytest.mark.unit
class TestStructuralDistances:
    '''Exclusion processing never touches structural distance, and every distance is D*.'''

    @pytest.fixture
    def distances(self, hierarchy_descriptions_parquet):
        return compute_structural_distances(
            hierarchy_descriptions_parquet,
            DistancesConfig(input_parquet=hierarchy_descriptions_parquet),
        )

    def test_output_is_structural_only(self, distances):
        assert distances.columns == [
            'idx_i',
            'idx_j',
            'code_i',
            'code_j',
            'structural_distance',
        ]

    def test_excluded_pairs_keep_tree_distances(self, distances):
        # '311111' excludes '321111' (merged 31-33 sector) and '441111' excludes '311211'.
        assert _distance(distances, '311111', '321111') == 8.0
        assert _distance(distances, '311211', '441111') == 10.0

    def test_no_structural_distance_is_a_sentinel(self, distances):
        values = distances.get_column('structural_distance')
        assert values.min() > 0.0
        # Two six-digit codes in different sectors are the farthest pair: 6 + 6 - 2
        assert values.max() == 10.0
        assert values.round(0).equals(values)

    def test_values_follow_the_tree(self, distances):
        # No half-step for a lineal pair, and a virtual root above the sectors
        assert _distance(distances, '31', '321') == 1.0
        assert _distance(distances, '311', '321') == 2.0
        assert _distance(distances, '3112', '31111') == 3.0
        assert _distance(distances, '31', '44') == 2.0

    def test_every_value_comes_from_the_one_tree_distance_function(self, distances):
        # Stage 4's diagnostics read D* from the same function, so the two agree on every pair
        codes = sorted(set(distances.get_column('code_i')) | set(distances.get_column('code_j')))
        position = {code: row for row, code in enumerate(codes)}
        matrix = tree_distance_matrix(codes)
        expected = [
            float(matrix[position[code_i], position[code_j]])
            for code_i, code_j in distances.select('code_i', 'code_j').rows()
        ]
        assert distances.get_column('structural_distance').to_list() == expected

    def test_every_unordered_pair_appears_once_in_canonical_orientation(self, distances):
        n_codes = 17
        assert distances.height == n_codes * (n_codes - 1) // 2
        oriented = distances.with_columns(
            level_i=pl.col('code_i').str.len_chars(),
            level_j=pl.col('code_j').str.len_chars(),
        )
        non_canonical = oriented.filter(
            pl.col('level_i').gt(pl.col('level_j'))
            | (pl.col('level_i').eq(pl.col('level_j')) & pl.col('code_i').ge(pl.col('code_j')))
        )
        assert non_canonical.height == 0
        # Same-level pairs across a merged-sector prefix are no longer emitted twice.
        assert distances.filter(pl.col('code_i').eq('321') & pl.col('code_j').eq('311')).height == 0
```

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
    assert distance_matrix.row(0)[1] == 0.5
    assert distance_matrix.row(1)[0] == 0.5
```

with:

```python
    assert distance_matrix.row(0)[1] == 2.0
    assert distance_matrix.row(1)[0] == 2.0
```

Replace:

```python
    distances = pl.DataFrame(pairs | {'structural_distance': [0.5, 1.5, 0.5, 0.5, 2.0, 3.0]})
```

with:

```python
    distances = pl.DataFrame(pairs | {'structural_distance': [1.0, 2.0, 1.0, 1.0, 2.0, 3.0]})
```

Replace:

```python
    distances = pl.concat([distances, pl.DataFrame(reversed_key | {'structural_distance': [99.0]})])
    relations = pl.concat(
        [
            relations,
            pl.DataFrame(
                reversed_key
                | {
                    'structural_relation_id': [99],
                    'structural_relation_name': ['cross_sector'],
                }
            ),
        ]
    )
```

with:

```python
    distances = pl.concat([distances, pl.DataFrame(reversed_key | {'structural_distance': [2.0]})])
    relations = pl.concat(
        [
            relations,
            pl.DataFrame(
                reversed_key
                | {
                    'structural_relation_id': [2],
                    'structural_relation_name': ['sibling'],
                }
            ),
        ]
    )
```

Replace:

```python
    with pytest.raises(ValueError, match='codebook'):
        build_pair_facts(distances, relations, mislabeled, build_codebook(mislabeled))
```

with:

```python
    with pytest.raises(ValueError, match='codebook'):
        build_pair_facts(distances, relations, mislabeled, build_codebook(mislabeled))

# -------------------------------------------------------------------------------------------------
# D* (Req 7): every stored distance is checked
# -------------------------------------------------------------------------------------------------

def _set_pair(frame: pl.DataFrame, code_i: str, code_j: str, column: str, value) -> pl.DataFrame:
    chosen = pl.col('code_i').eq(code_i) & pl.col('code_j').eq(code_j)
    return frame.with_columns(
        pl.when(chosen).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )

@pytest.mark.parametrize(
    ('code_i', 'code_j', 'value', 'message'),
    [
        # D* has no half-step for a lineal pair
        ('311', '3111', 0.5, 'no half-step'),
        # 3112 and 31111 are 3 apart; at 4 they would be farther than through 311 (1 + 2)
        ('3112', '31111', 4.0, 'triangle inequality'),
        # A parent and its child are 1 apart; 2 keeps the triangle inequality but is not D*
        ('311', '3111', 2.0, 'differs from D\\*'),
    ],
)
def test_pair_facts_reject_a_distance_that_is_not_d_star(
    depth_first_frames, code_i, code_j, value, message
):
    descriptions, distances, relations = depth_first_frames
    distances = _set_pair(distances, code_i, code_j, 'structural_distance', value)

    with pytest.raises(ValueError, match=message):
        build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))

def test_pair_facts_reject_a_cross_sector_label_inside_a_sector(depth_first_frames):
    descriptions, distances, relations = depth_first_frames
    relations = _set_pair(relations, '3111', '3112', 'structural_relation_id', 99)

    with pytest.raises(ValueError, match='cross_sector relation label'):
        build_pair_facts(distances, relations, descriptions, build_codebook(descriptions))

@pytest.mark.parametrize(
    ('value', 'message'),
    [(99.0, 'retired cross-sector constant 99'), (9.0, 'cross-sector distances must equal')],
)
def test_pair_facts_reject_a_cross_sector_distance_off_the_formula(
    descriptions_fixture, structural_frames_fixture, value, message
):
    # '111111' and '222222' meet only at the virtual root: 6 + 6 - 2 = 10
    distances, relations = structural_frames_fixture
    distances = _set_pair(distances, '111111', '222222', 'structural_distance', value)

    with pytest.raises(ValueError, match=message):
        build_pair_facts(
            distances, relations, descriptions_fixture, build_codebook(descriptions_fixture)
        )
```

Replace:

```python
    assert all(manifest['validation_results'].values())
```

with:

```python
    assert all(manifest['validation_results'].values())
    for check in (
        'distance_is_d_star',
        'cross_sector_distance_formula',
        'cross_sector_relation_label',
        'distance_triangle_inequality',
    ):
        assert manifest['validation_results'][check] is True
```

Replace:

```python
    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*structural distance zero'):
        load_validated_bundle(generated_bundle)
```

with:

```python
    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*structural distance zero'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_legacy_cross_sector_distance(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(
            structural_distance=pl.when(pl.col('structural_relation_id').eq(99)).then(
                pl.lit(99.0, dtype=pl.Float32)
            ).otherwise(pl.col('structural_distance'))
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*retired cross-sector constant'):
        load_validated_bundle(generated_bundle)
```

In `tests/unit/test_supervision_index.py`, replace:

```python
    assert joined.structural_distance.tolist() == [[2.0], [99.0]]
```

with:

```python
    assert joined.structural_distance.tolist() == [[2.0], [10.0]]
```

and replace:

```python
    assert index.structural_distance[0, 1] == index.structural_distance[1, 0] == 0.5
```

with:

```python
    assert index.structural_distance[0, 1] == index.structural_distance[1, 0] == 2.0
```

In `tests/unit/test_data_triplets.py`, replace:

```python
            'structural_distance': [2.0, 99.0, 99.0, 99.0, 99.0, 2.0],
```

with:

```python
            'structural_distance': [2.0, 10.0, 10.0, 10.0, 10.0, 2.0],
```

Replace:

```python
    assert reverse['negative_structural_distance'] == 99.0
```

with:

```python
    assert reverse['negative_structural_distance'] == 10.0
```

Replace:

```python
    assert pairs.get_column('distance_margin').to_list() == [10.0, 10.0]
```

with:

```python
    # The distance margin is the D* difference: '22' is 6 from the anchor, '222222' 10, and the
    # positive 2
    assert pairs.get_column('distance_margin').to_list() == [4.0, 8.0]
```

Replace:

```python
                    'structural_distance': 2.0 if siblings else 99.0,
```

with:

```python
                    'structural_distance': 2.0 if siblings else 10.0,
```

Replace:

```python
def test_structural_margins_preserve_the_legacy_special_cases():
    frame = pl.DataFrame(
        {
            'positive_structural_relation_id': [7, 2, 1, 1, 7],
            'positive_structural_distance': [4.0, 2.0, 0.5, 0.5, 4.0],
            'negative_structural_relation_id': [8, 3, 99, 2, 2],
            'negative_structural_distance': [4.0, 1.5, 99.0, 2.0, 2.0],
        },
        schema_overrides={
            'positive_structural_relation_id': pl.Int16,
            'negative_structural_relation_id': pl.Int16,
            'positive_structural_distance': pl.Float32,
            'negative_structural_distance': pl.Float32,
        },
    )

    margins = _structural_margins(frame)

    # Rows: equal distance with a farther relation, a -0.5 lineal adjustment, a cross-sector
    # negative, an ordinary farther negative; the structurally closer negative is dropped.
    assert margins.get_column('relation_margin').to_list() == pytest.approx([1.0, 1.0, 15.0, 1.0])
    assert margins.get_column('distance_margin').to_list() == pytest.approx(
        [0.3333, 0.6667, 10.0, 1.5]
    )
    assert margins.get_column('margin').to_list() == pytest.approx(
        [
            1.0 / (1.0 * 0.3333 + 0.3333 * 0.6667),
            1.0 / (1.0 * 0.3333 + 0.6667 * 0.6667),
            1.0 / (15.0 * 0.3333 + 10.0 * 0.6667),
            1.0 / (1.0 * 0.3333 + 1.5 * 0.6667),
        ],
        rel=1e-5,
    )
```

with:

```python
def test_structural_margins_read_d_star_differences():
    frame = pl.DataFrame(
        {
            'positive_structural_relation_id': [7, 1, 1, 2, 7, 7],
            'positive_structural_distance': [4.0, 1.0, 1.0, 2.0, 4.0, 8.0],
            'negative_structural_relation_id': [8, 3, 99, 5, 2, 99],
            'negative_structural_distance': [4.0, 2.0, 6.0, 3.0, 2.0, 6.0],
        },
        schema_overrides={
            'positive_structural_relation_id': pl.Int16,
            'negative_structural_relation_id': pl.Int16,
            'positive_structural_distance': pl.Float32,
            'negative_structural_distance': pl.Float32,
        },
    )

    margins = _structural_margins(frame)

    # Rows: an equal D* with a farther relation (the one fixed distance margin), a lineal negative
    # one step past a lineal positive, a cross-sector negative, an ordinary farther negative. The
    # structurally closer negative is dropped, and so is a cross-sector negative that is closer in
    # D* than the positive.
    assert margins.get_column('relation_margin').to_list() == pytest.approx([1.0, 2.0, 15.0, 3.0])
    assert margins.get_column('distance_margin').to_list() == pytest.approx([0.3333, 1.0, 5.0, 1.0])
    assert margins.get_column('margin').to_list() == pytest.approx(
        [
            1.0 / (1.0 * 0.3333 + 0.3333 * 0.6667),
            1.0 / (2.0 * 0.3333 + 1.0 * 0.6667),
            1.0 / (15.0 * 0.3333 + 5.0 * 0.6667),
            1.0 / (3.0 * 0.3333 + 1.0 * 0.6667),
        ],
        rel=1e-5,
    )
```

In `tests/unit/test_structural_margins.py`, replace:

```python
        # Cross-sector negatives always receive the fixed legacy margins.
        ((99.0, 99), (0.5, 1), (15.0, 10.0)),
        # The relation label marks a cross-sector negative, whatever its distance.
        ((10.0, 99), (2.0, 2), (15.0, 10.0)),
        # Farther relation at an equal distance receives the fixed equal-distance margin.
        ((2.0, 3), (2.0, 2), (1.0, 0.3333)),
        # The -0.5 lineal adjustment receives its fixed margin when the relation is farther.
        ((1.5, 3), (2.0, 2), (1.0, 0.6667)),
        # Otherwise both margins are raw deltas.
        ((3.0, 7), (0.5, 1), (6.0, 2.5)),
    ],
)
def test_structural_margins_follow_the_generator_special_cases(negative, positive, expected):
```

with:

```python
        # A cross-sector negative, marked by its relation label, receives the fixed relation
        # margin; its distance margin is the D* difference.
        ((10.0, 99), (2.0, 2), (15.0, 8.0)),
        ((6.0, 99), (1.0, 1), (15.0, 5.0)),
        # Farther relation at an equal distance receives the fixed equal-distance margin.
        ((2.0, 3), (2.0, 2), (1.0, 0.3333)),
        # A lineal negative one step past a lineal positive: no half-step.
        ((2.0, 3), (1.0, 1), (2.0, 1.0)),
        # Otherwise both margins are raw deltas.
        ((3.0, 7), (1.0, 1), (6.0, 2.0)),
    ],
)
def test_structural_margins_follow_the_generator_rule(negative, positive, expected):
```

and replace:

```python
        ((0.5, 1), (1.5, 3)),  # the anchor's parent while the positive is its grandparent
        ((2.0, 2), (2.0, 2)),  # another sibling of a sibling positive: no relation margin
        ((1.5, 3), (3.0, 7)),  # structurally closer by both measures
```

with:

```python
        ((1.0, 1), (2.0, 3)),  # the anchor's parent while the positive is its grandparent
        ((2.0, 2), (2.0, 2)),  # another sibling of a sibling positive: no relation margin
        ((2.0, 3), (3.0, 7)),  # structurally closer by both measures
        ((6.0, 99), (8.0, 7)),  # a cross-sector negative closer in D* than the positive
```

In `tests/unit/test_hard_negative_mining.py`, replace:

```python
    assert joined.structural_distance.tolist() == [[2.0], [3.0]]
```

with:

```python
    # Both anchors are siblings of code 2 (D* 2), under different relation labels
    assert joined.structural_distance.tolist() == [[2.0], [2.0]]
    assert joined.structural_relation_id.tolist() == [[2], [3]]
```

Replace:

```python
        'positive_structural_distance': torch.tensor([0.5] * len(pools)),
```

with:

```python
        'positive_structural_distance': torch.tensor([2.0] * len(pools)),
```

Replace:

```python
    # Positive: child at distance 0.5. Code 2 is the anchor's sibling exclusion (relation 2,
    # distance 2.0 -> raw deltas); codes 3 and 4 are cross-sector (fixed 15 / 10 margins).
    expected_margins = {2: (1.0, 1.5), 3: (15.0, 10.0), 4: (15.0, 10.0)}
    for row in range(selected.code_id.shape[0]):
        for slot, code in enumerate(selected.code_id[row].tolist()):
            assert selected.relation_margin[row, slot].item() == expected_margins[code][0]
            assert selected.distance_margin[row, slot].item() == expected_margins[code][1]
```

with:

```python
    # Positive: code 1 at D* 2 (relation 1). Code 2 is the anchor's sibling exclusion at the same
    # D* with a farther relation (relation margin 1, the fixed equal-distance margin); codes 3 and
    # 4 are cross-sector (relation margin 15, distance margin 10 - 2).
    expected_margins = {2: (1.0, 0.3333), 3: (15.0, 8.0), 4: (15.0, 8.0)}
    for row in range(selected.code_id.shape[0]):
        for slot, code in enumerate(selected.code_id[row].tolist()):
            assert selected.relation_margin[row, slot].item() == expected_margins[code][0]
            assert selected.distance_margin[row, slot].item() == pytest.approx(
                expected_margins[code][1]
            )
```

In `tests/unit/test_hgcn_streaming_dataset.py`, replace:

```python
    assert negative == {
        'negative_idx': 2,
        'negative_code': '111113',
        'relation_margin': 1.0,
        'distance_margin': 1.5,
    }
```

with:

```python
    # '111113' is as far from '111111' as the positive '111112' (D* 2), under a farther relation,
    # so it takes the fixed equal-distance margin
    assert negative == {
        'negative_idx': 2,
        'negative_code': '111113',
        'relation_margin': 1.0,
        'distance_margin': pytest.approx(0.3333),
    }
```

In `tests/unit/test_streaming_sampling.py`, replace:

```python
def _index(size: int, anchor_code_id: int, exclusion_code_ids: tuple[int, ...]) -> SupervisionIndex:
    directed = torch.zeros((size, size), dtype=torch.bool)
    for code_id in exclusion_code_ids:
        directed[anchor_code_id, code_id] = True
    return SupervisionIndex(
        code_to_id={str(code_id): code_id
                    for code_id in range(size)},
        id_to_code=tuple(str(code_id) for code_id in range(size)),
        structural_distance=torch.full((size, size), 99.0),
        structural_relation_id=torch.full((size, size), 99, dtype=torch.int16),
        directed_exclusion=directed,
    )
```

with:

```python
def _index(
    size: int, anchor_code_id: int, positive_code_id: int, exclusion_code_ids: tuple[int, ...]
) -> SupervisionIndex:
    directed = torch.zeros((size, size), dtype=torch.bool)
    for code_id in exclusion_code_ids:
        directed[anchor_code_id, code_id] = True
    # Every pair is cross-sector (D* 10) except the anchor and its sibling positive (D* 2)
    distance = torch.full((size, size), 10.0)
    relation = torch.full((size, size), 99, dtype=torch.int16)
    for code_i, code_j in ((anchor_code_id, positive_code_id), (positive_code_id, anchor_code_id)):
        distance[code_i, code_j] = 2.0
        relation[code_i, code_j] = 2
    return SupervisionIndex(
        code_to_id={str(code_id): code_id
                    for code_id in range(size)},
        id_to_code=tuple(str(code_id) for code_id in range(size)),
        structural_distance=distance,
        structural_relation_id=relation,
        directed_exclusion=directed,
    )
```

and replace:

```python
        index = _index(size, anchor_code_id, exclusion_code_ids)
```

with:

```python
        index = _index(size, anchor_code_id, positive_code_id, exclusion_code_ids)
```

In `tests/integration/test_stage3_training_step.py`, replace:

```python
    assert spy.structural_distances == [99.0, 2.0, 99.0]
```

with:

```python
    assert spy.structural_distances == [10.0, 2.0, 10.0]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_distances.py tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py tests/unit/test_supervision_artifacts.py tests/unit/test_supervision_index.py tests/unit/test_streaming_sampling.py -q`
Expected: FAIL. Among others: `test_values_follow_the_tree` (0.5 where 1.0 is expected), the new
`test_pair_facts_reject_*` tests (nothing raises), the margin tests (the fixed cross-sector
margin 10 where the D* difference is expected), and every test on the `validated_bundle` fixture,
which now expects contract `stage3-supervision-v2`.

- [x] **Step 3: Compute D* for every canonical pair**

Overwrite `src/naics_embedder/data/compute_distances.py` with:

```python
# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging

import polars as pl

from naics_embedder.utils.config import DistancesConfig
from naics_embedder.utils.naics_hierarchy import tree_distance_matrix

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Structural distances
# -------------------------------------------------------------------------------------------------

def compute_structural_distances(input_parquet: str, cfg: DistancesConfig) -> pl.DataFrame:
    '''
    D* (Req 7) for every unordered pair of distinct codes.

    Rows follow the canonical pair orientation (shallower code first, code order on ties), so
    each unordered pair appears exactly once. Every value comes from
    :func:`~naics_embedder.utils.naics_hierarchy.tree_distance_matrix`, the tree path length
    through a virtual root above the sectors: no half-step for lineal pairs and no cross-sector
    constant. No exclusion processing happens here and nothing is written.

    Args:
        input_parquet: Descriptions parquet with ``index``, ``level``, and ``code`` columns.
        cfg: Distance configuration (logged for provenance).

    Returns:
        DataFrame with ``idx_i``, ``idx_j``, ``code_i``, ``code_j``, ``structural_distance``.
    '''

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    naics = pl.read_parquet(input_parquet).select('index', 'level', 'code').with_row_index('row')
    distances = tree_distance_matrix(naics.get_column('code').to_list())
    naics_i = naics.select(
        row_i=pl.col('row'), idx_i=pl.col('index'), lvl_i=pl.col('level'), code_i=pl.col('code')
    )
    naics_j = naics.select(
        row_j=pl.col('row'), idx_j=pl.col('index'), lvl_j=pl.col('level'), code_j=pl.col('code')
    )

    # Canonical orientation: shallower code first, numeric code order on ties.
    pairs = naics_i.join(naics_j, how='cross').filter(
        (pl.col('lvl_i') < pl.col('lvl_j'))
        | (
            (pl.col('lvl_i') == pl.col('lvl_j'))
            & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
        )
    )
    values = distances[pairs.get_column('row_i').to_numpy(), pairs.get_column('row_j').to_numpy()]
    logger.info(f'D* for {pairs.height:,} pairs of {naics.height:,} codes')

    return pairs.select(
        pl.col('idx_i'),
        pl.col('idx_j'),
        pl.col('code_i'),
        pl.col('code_j'),
        structural_distance=pl.Series(values, dtype=pl.Float32),
    ).sort('idx_i', 'idx_j')
```

- [x] **Step 4: Validate every stored distance**

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    IndexRole,
    SemanticTarget,
    SupervisionManifest,
)
```

with:

```python
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    CROSS_SECTOR_RELATION_ID,
    ArtifactFile,
    IndexRole,
    SemanticTarget,
    SupervisionManifest,
)
from naics_embedder.utils.naics_hierarchy import code_lineage, tree_distance_matrix
```

Replace:

```python
    '''Fail closed on identity, uniqueness, orientation, coverage, or sentinel violations.'''
```

with:

```python
    '''Fail closed on identity, uniqueness, orientation, coverage, sentinel, or D* violations.'''
```

Replace:

```python
    if structural.filter(
        pl.col('structural_relation_id').eq(0)
        | pl.col('structural_relation_name').eq('excluded')
    ).height:
        raise ValueError('structural relation fields contain an exclusion sentinel')
```

with:

```python
    if structural.filter(
        pl.col('structural_relation_id').eq(0)
        | pl.col('structural_relation_name').eq('excluded')
    ).height:
        raise ValueError('structural relation fields contain an exclusion sentinel')
    validate_tree_distances(structural)

def validate_tree_distances(structural: pl.DataFrame) -> None:
    '''
    Fail closed unless every structural distance is D* (Req 7).

    D* is an integer, never the retired cross-sector constant 99. A pair across sectors, whose
    lowest common ancestor is the virtual root, has λ(i) + λ(j) − 2, where λ is the number of
    digits, and carries the ``cross_sector`` relation label, which no other pair carries. The
    stored distances satisfy the triangle inequality over every ordered triple of codes, and each
    equals :func:`~naics_embedder.utils.naics_hierarchy.tree_distance_matrix` on its pair: the path
    length through the pair's lowest common ancestor.
    '''

    distance = structural.get_column('structural_distance').to_numpy().astype(np.float64)
    if (distance != np.round(distance)).any():
        raise ValueError('structural distances must be integers: D* has no half-step')
    if (distance == 99.0).any():
        raise ValueError('structural distances contain the retired cross-sector constant 99')

    codes = sorted(
        set(structural.get_column('code_i').to_list())
        | set(structural.get_column('code_j').to_list())
    )
    position = {code: row for row, code in enumerate(codes)}
    rows, columns = (
        structural.get_column(name).replace_strict(position, return_dtype=pl.Int64).to_numpy()
        for name in ('code_i', 'code_j')
    )

    sectors = np.array([code_lineage(code)[0] for code in codes])
    digits = np.array([len(code) for code in codes])
    across = sectors[rows] != sectors[columns]
    if not np.array_equal(distance[across], (digits[rows] + digits[columns] - 2)[across]):
        raise ValueError('cross-sector distances must equal λ(i) + λ(j) − 2')
    labelled = structural.get_column('structural_relation_id').to_numpy() == CROSS_SECTOR_RELATION_ID
    if not np.array_equal(labelled, across):
        raise ValueError(
            'the cross_sector relation label must mark exactly the pairs across sectors: '
            f'{int((labelled != across).sum()):,} pairs disagree'
        )

    matrix = np.zeros((len(codes), len(codes)), dtype=np.int16)
    matrix[rows, columns] = distance
    matrix[columns, rows] = distance
    for middle in range(len(codes)):
        if (matrix[:, middle, None] + matrix[None, middle, :] < matrix).any():
            raise ValueError(f'D* violates the triangle inequality through {codes[middle]}')

    expected = tree_distance_matrix(codes)[rows, columns]
    if not np.array_equal(distance, expected):
        wrong = np.flatnonzero(distance != expected)
        first = wrong[0]
        raise ValueError(
            f'structural distance differs from D* on {wrong.size:,} pairs, e.g. '
            f'{codes[rows[first]]}/{codes[columns[first]]}: {distance[first]:g} != '
            f'{expected[first]}'
        )
```

In `src/naics_embedder/data/supervision_bundle.py`, replace:

```python
        'nonzero_structural_distance': True,
        'no_structural_sentinel': True,
```

with:

```python
        'nonzero_structural_distance': True,
        'no_structural_sentinel': True,
        'distance_is_d_star': True,
        'cross_sector_distance_formula': True,
        'cross_sector_relation_label': True,
        'distance_triangle_inequality': True,
```

- [x] **Step 5: Margins for integer D***

In `src/naics_embedder/supervision/schema.py`, replace:

```python
# Shared by the training-pair generator (``data.create_triplets``) and the runtime eligibility rule
# (``supervision.margins``), so both apply the identical legacy special cases.
# -------------------------------------------------------------------------------------------------

CROSS_SECTOR_DISTANCE = 99.0
# The relation label cross-sector pairs carry: every reader finds them by it, not by a distance
CROSS_SECTOR_RELATION_ID = 99
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
CROSS_SECTOR_RELATION_MARGIN = 15.0
CROSS_SECTOR_DISTANCE_MARGIN = 10.0
EQUAL_DISTANCE_MARGIN = 0.3333
LINEAL_ADJUSTED_DISTANCE_MARGIN = 0.6667
LINEAL_DISTANCE_DELTA = -0.5
```

with:

```python
# Shared by the training-pair generator (``data.create_triplets``) and the runtime eligibility rule
# (``supervision.margins``), so both apply the identical margins. Distances are D* (Req 7):
# integers with no half-step and no cross-sector constant, so the distance axis has one special
# case left, the equal-distance tie.
# -------------------------------------------------------------------------------------------------

# The relation label cross-sector pairs carry: every reader finds them by it, not by a distance
CROSS_SECTOR_RELATION_ID = 99
CROSS_SECTOR_RELATION_NAME = 'cross_sector'
# The relation axis keeps its cross-sector margin until Stage 7 retires the axis (D5)
CROSS_SECTOR_RELATION_MARGIN = 15.0
EQUAL_DISTANCE_MARGIN = 0.3333
```

In `src/naics_embedder/supervision/margins.py`, replace:

```python
The torch mirror of the generator rule ``create_triplets._structural_margins``: an ordinary
negative must be structurally farther from the anchor than the positive. Cross-sector
negatives, those with the ``cross_sector`` relation label, receive fixed margins; equal
distances and the -0.5 lineal adjustment receive fixed
distance margins when the relation margin is positive. Candidates sourced at runtime (universe
backfill, the distributed global pool) pass through the same rule, so the repaired pipeline never
repels an *ordinary* candidate that the generated supervision would not treat as a negative.
```

with:

```python
The torch mirror of the generator rule ``create_triplets._structural_margins``: an ordinary
negative must be structurally farther from the anchor than the positive. The distance margin is
the difference in D*, except that an equal distance with a farther relation receives a fixed
margin. Cross-sector negatives, those with the ``cross_sector`` relation label, receive a fixed
relation margin. Candidates sourced at runtime (universe backfill, the distributed global pool)
pass through the same rule, so the repaired pipeline never repels an *ordinary* candidate that
the generated supervision would not treat as a negative.
```

Replace:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
    LINEAL_ADJUSTED_DISTANCE_MARGIN,
    LINEAL_DISTANCE_DELTA,
)
```

with:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
)
```

Replace:

```python
    ``[batch, 1]`` positives). Structural values are exact in float32 (half-step distances, small
    relation IDs), so the special-case equality tests match the generator exactly on every device.
```

with:

```python
    ``[batch, 1]`` positives). Structural values are exact in float32 (integer distances, small
    relation IDs), so the equality test matches the generator exactly on every device.
```

Replace:

```python
    relation_margin = relation_delta.masked_fill(cross_sector, CROSS_SECTOR_RELATION_MARGIN)
    # Fill in reverse precedence of the generator's when/then chain so earlier cases win.
    distance_margin = distance_delta.masked_fill(cross_sector, CROSS_SECTOR_DISTANCE_MARGIN)
    distance_margin = distance_margin.masked_fill(
        farther_relation & distance_delta.eq(LINEAL_DISTANCE_DELTA),
        LINEAL_ADJUSTED_DISTANCE_MARGIN,
    )
    distance_margin = distance_margin.masked_fill(
        farther_relation & distance_delta.eq(0.0),
        EQUAL_DISTANCE_MARGIN,
    )
    return relation_margin, distance_margin
```

with:

```python
    relation_margin = relation_delta.masked_fill(cross_sector, CROSS_SECTOR_RELATION_MARGIN)
    distance_margin = distance_delta.masked_fill(
        farther_relation & distance_delta.eq(0.0),
        EQUAL_DISTANCE_MARGIN,
    )
    return relation_margin, distance_margin
```

In `src/naics_embedder/data/create_triplets.py`, replace:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_DISTANCE_MARGIN,
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
    LINEAL_ADJUSTED_DISTANCE_MARGIN,
    LINEAL_DISTANCE_DELTA,
    SamplingRole,
```

with:

```python
from naics_embedder.supervision.schema import (
    CROSS_SECTOR_RELATION_ID,
    CROSS_SECTOR_RELATION_MARGIN,
    EQUAL_DISTANCE_MARGIN,
    SamplingRole,
```

Replace:

```python
# Legacy margin weights, preserved for graph-model compatibility. The structural margin special
# cases live in supervision.schema so the runtime eligibility rule shares them.
```

with:

```python
# Legacy margin weights, preserved for graph-model compatibility. The structural margin constants
# live in supervision.schema so the runtime eligibility rule shares them.
```

Replace:

```python
    Add relation/distance margins with the legacy special cases and keep ordered triplets.

    Cross-sector negatives receive fixed margins; equal distances and the -0.5 lineal adjustment
    receive fixed distance margins when the relation margin is positive. Triplets whose negative is
    not structurally farther than the positive are dropped.
```

with:

```python
    Add relation/distance margins and keep ordered triplets.

    The distance margin is the difference in D*, except that an equal distance receives a fixed
    margin when the relation margin is positive. Cross-sector negatives receive a fixed relation
    margin. Triplets whose negative is not structurally farther than the positive are dropped.
```

Replace:

```python
        distance_margin=pl.when(relation_delta.gt(0) & distance_delta.eq(0.0)).then(
            pl.lit(EQUAL_DISTANCE_MARGIN)
        ).when(relation_delta.gt(0) & distance_delta.eq(LINEAL_DISTANCE_DELTA)).then(
            pl.lit(LINEAL_ADJUSTED_DISTANCE_MARGIN)
        ).when(cross_sector).then(pl.lit(CROSS_SECTOR_DISTANCE_MARGIN)).otherwise(distance_delta),
```

with:

```python
        distance_margin=pl.when(relation_delta.gt(0) & distance_delta.eq(0.0)).then(
            pl.lit(EQUAL_DISTANCE_MARGIN)
        ).otherwise(distance_delta),
```

- [x] **Step 6: Move the contract to v2**

`supervision/artifacts.py` keeps its one mention of v1, a comment that Task 8 removes, so it is
not in this list.

Run: `sed -i '' 's/stage3-supervision-v1/stage3-supervision-v2/g' src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/mode.py src/naics_embedder/utils/config.py src/naics_embedder/cli/commands/data.py conf/config.yaml conf/data/supervision.yaml`

Run: `grep -rn 'stage3-supervision-v1' src conf`
Expected: one line, the comment in `src/naics_embedder/supervision/artifacts.py`.

- [x] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_distances.py tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py tests/unit/test_supervision_artifacts.py tests/unit/test_supervision_index.py tests/unit/test_streaming_sampling.py tests/unit/test_hard_negative_mining.py tests/unit/test_hgcn_streaming_dataset.py tests/integration/test_stage3_training_step.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1653 passed, 1 skipped`. The networkx helper tests go with the helpers (34 tests in
`test_data_distances.py` become 6), and the validator and margins add 8.

- [x] **Step 8: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/data/compute_distances.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/margins.py src/naics_embedder/data/create_triplets.py src/naics_embedder/supervision/mode.py src/naics_embedder/utils/config.py src/naics_embedder/cli/commands/data.py tests/fixtures/supervision.py tests/unit/test_data_distances.py tests/unit/test_data_triplets.py tests/unit/test_structural_margins.py tests/unit/test_supervision_artifacts.py tests/unit/test_supervision_index.py tests/unit/test_hard_negative_mining.py tests/unit/test_hgcn_streaming_dataset.py tests/unit/test_streaming_sampling.py tests/integration/test_stage3_training_step.py tests/unit/test_checkpoint_contract.py tests/unit/test_cli_commands.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_streaming_dataset.py tests/unit/test_supervision_schema.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 9: Commit**

```bash
git add src/naics_embedder/data/compute_distances.py \
  src/naics_embedder/supervision/artifacts.py \
  src/naics_embedder/data/supervision_bundle.py \
  src/naics_embedder/supervision/schema.py \
  src/naics_embedder/supervision/margins.py \
  src/naics_embedder/data/create_triplets.py \
  src/naics_embedder/supervision/mode.py \
  src/naics_embedder/utils/config.py \
  src/naics_embedder/cli/commands/data.py \
  conf/config.yaml \
  conf/data/supervision.yaml \
  tests/fixtures/supervision.py \
  tests/unit/test_data_distances.py \
  tests/unit/test_data_triplets.py \
  tests/unit/test_structural_margins.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_supervision_index.py \
  tests/unit/test_hard_negative_mining.py \
  tests/unit/test_hgcn_streaming_dataset.py \
  tests/unit/test_streaming_sampling.py \
  tests/integration/test_stage3_training_step.py \
  tests/unit/test_checkpoint_contract.py \
  tests/unit/test_cli_commands.py \
  tests/unit/test_cli_training.py \
  tests/unit/test_config.py \
  tests/unit/test_streaming_dataset.py \
  tests/unit/test_supervision_schema.py
git commit -m "feat(supervision): D* as the bundle distance under contract stage3-supervision-v2"
```

### Task 7: No generated training row uses an exclusion as a negative

Req 8(c): a cross-reference reroutes an activity, so it never makes its two codes a code–code
negative. Today the generator keeps every exclusion it finds among an anchor's candidates and
labels it `unrelated`. This task drops every candidate that is an explicit exclusion of its anchor,
in either direction, and the validators refuse such a row at build and at load. The pair facts
keep every exclusion pair, flagged, so Task 8's runtime can keep them out too. The manifest
records the check as `no_exclusion_negatives`.

On the five-code fixture two of the five training pairs go: anchor 0's through code 2, its
exclusion, and anchor 1's through code 3, which excludes it.

**Files:**
- Modify: `src/naics_embedder/data/create_triplets.py` (module docstring, `_negative_candidates`,
  `_cap_cross_sector`, `_validate_training_pairs`, `build_training_pairs`' docstring)
- Modify: `src/naics_embedder/supervision/artifacts.py` (`validate_training_pairs_members`'
  docstring, `_validate_training_chunk`)
- Modify: `src/naics_embedder/data/supervision_bundle.py` (validation results)
- Test: `tests/unit/test_data_triplets.py`, `tests/unit/test_supervision_artifacts.py`,
  `tests/unit/test_hgcn_streaming_dataset.py`

**Interfaces:**
- Consumes: Task 6's five-code fixture, whose distances are D*.
- Produces:
  - Generated training pairs never set `negative_is_explicit_exclusion`. The column stays in
    the schema, always false.
  - `create_triplets._validate_training_pairs` raises `ValueError('a training negative cannot be
    an explicit exclusion of its anchor')`. The load-time check raises a `ValueError` whose
    message ends in "training negatives are explicit exclusions of their anchors".
  - The manifest's `validation_results` gain `no_exclusion_negatives`.
  - `_cap_cross_sector` caps every cross-sector negative, since no exclusion is left to exempt.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_data_triplets.py`, replace:

```python
def test_training_pairs_keep_semantics_separate_from_structure(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    excluded = pairs.filter(pl.col('negative_is_explicit_exclusion')).row(0, named=True)
    ordinary = pairs.filter(~pl.col('negative_is_explicit_exclusion')).row(0, named=True)

    assert excluded['negative_semantic_target'] == 'unrelated'
    assert excluded['negative_semantic_source'] == 'explicit_exclusion'
    assert excluded['negative_structural_distance'] > 0.0
    assert excluded['negative_sampling_role'] == 'negative'
    assert ordinary['negative_semantic_target'] == 'unknown'
    assert ordinary['negative_semantic_source'] == 'unlabeled'
```

with:

```python
def test_no_training_negative_is_an_explicit_exclusion(pair_facts_fixture):
    # '111111' excludes '111113' and '222222' excludes '111112': neither pair is ever a negative
    pairs = build_training_pairs(pair_facts_fixture)

    assert not pairs.get_column('negative_is_explicit_exclusion').any()
    assert pairs.get_column('negative_semantic_target').unique().to_list() == ['unknown']
    assert pairs.get_column('negative_semantic_source').unique().to_list() == ['unlabeled']
    assert pairs.get_column('negative_sampling_role').unique().to_list() == ['negative']
```

Replace:

```python
    # Positives are canonical, within-sector, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is
    # an exclusion. A negative j needs rows positive -> j and anchor -> j.
    pairs = build_training_pairs(pair_facts_fixture)

    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (1, 2, 3), (1, 2, 4)]
```

with:

```python
    # Positives are canonical, within-sector, non-exclusion pairs: (0, 1) and (1, 2); (0, 2) is
    # an exclusion. A negative j needs rows positive -> j and anchor -> j, and is never an
    # exclusion of its anchor: 2 is anchor 0's, and 3 excludes anchor 1.
    pairs = build_training_pairs(pair_facts_fixture)

    assert _triples(pairs) == [(0, 1, 3), (0, 1, 4), (1, 2, 4)]
```

Replace:

```python
    # Anchor 2 ('222221') only reaches codes 0 and 1 through reversed same-level rows, exactly as
    # the legacy keep-filter admitted both orientations of cross-prefix pairs.
    pairs = build_training_pairs(cross_prefix_pair_facts)

    assert _triples(pairs) == [(0, 1, 2), (0, 1, 3), (2, 3, 0), (2, 3, 1)]
```

with:

```python
    # Anchor 2 ('222221') reaches code 1 only through a reversed same-level row, exactly as the
    # legacy keep-filter admitted both orientations of cross-prefix pairs. Code 0 excludes it, so
    # neither is ever the other's negative.
    pairs = build_training_pairs(cross_prefix_pair_facts)

    assert _triples(pairs) == [(0, 1, 3), (2, 3, 1)]
```

Replace:

```python
def test_reversed_rows_map_exclusion_directions_into_the_anchor_view(cross_prefix_pair_facts):
    pairs = build_training_pairs(cross_prefix_pair_facts)
    forward = pairs.filter(pl.col('anchor_code_id').eq(0)
                           & pl.col('negative_code_id').eq(2)).row(0, named=True)
    reverse = pairs.filter(pl.col('anchor_code_id').eq(2)
                           & pl.col('negative_code_id').eq(0)).row(0, named=True)

    assert (forward['anchor_excludes_negative'], forward['negative_excludes_anchor']) == (
        True,
        False,
    )
    assert (reverse['anchor_excludes_negative'], reverse['negative_excludes_anchor']) == (
        False,
        True,
    )
    assert reverse['negative_is_explicit_exclusion'] is True
    assert reverse['negative_semantic_target'] == 'unrelated'
    assert reverse['negative_structural_distance'] == 10.0
```

with:

```python
def test_reversed_rows_map_exclusion_directions_into_the_anchor_view(cross_prefix_pair_facts):
    # The directions still reach the anchor view, which is how the generator finds the exclusions
    # it must drop
    view = _anchor_view(cross_prefix_pair_facts)
    forward = view.filter(pl.col('anchor_code_id').eq(0)
                          & pl.col('candidate_code_id').eq(2)).row(0, named=True)
    reverse = view.filter(pl.col('anchor_code_id').eq(2)
                          & pl.col('candidate_code_id').eq(0)).row(0, named=True)

    assert (forward['anchor_excludes_candidate'], forward['candidate_excludes_anchor']) == (
        True,
        False,
    )
    assert (reverse['anchor_excludes_candidate'], reverse['candidate_excludes_anchor']) == (
        False,
        True,
    )
    assert reverse['is_explicit_exclusion'] is True
    assert reverse['structural_distance'] == 10.0
```

Replace:

```python
def test_cross_sector_cap_keeps_a_deterministic_subset_and_exempts_exclusions(
    wide_cross_sector_pair_facts,
):
    uncapped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=100)
    capped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)
    again = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)

    assert _triples(uncapped) == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 1, 6)]
    assert capped.equals(again)
    assert set(_triples(capped)) <= set(_triples(uncapped))
    assert capped.filter(pl.col('negative_is_explicit_exclusion')).height == 1
    assert capped.filter(~pl.col('negative_is_explicit_exclusion')).height == 2
```

with:

```python
def test_cross_sector_cap_keeps_a_deterministic_subset(wide_cross_sector_pair_facts):
    uncapped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=100)
    capped = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)
    again = build_training_pairs(wide_cross_sector_pair_facts, cross_sector_cap=2, cap_seed=11)

    # '444444' (code 4) is anchor 0's exclusion, so it is never a negative, capped or not
    assert _triples(uncapped) == [(0, 1, 2), (0, 1, 3), (0, 1, 5), (0, 1, 6)]
    assert capped.equals(again)
    assert set(_triples(capped)) <= set(_triples(uncapped))
    assert capped.height == 2
```

Replace:

```python
    bad = pairs.with_columns(negative_is_explicit_exclusion=pl.lit(False))
```

with:

```python
    bad = pairs.with_columns(anchor_excludes_negative=pl.lit(True))
```

Replace:

```python
    bad = pairs.with_columns(negative_semantic_target=pl.lit('unknown'))
```

with:

```python
    bad = pairs.with_columns(negative_semantic_target=pl.lit('unrelated'))
```

and append to the end of the file:

```python

def test_an_exclusion_negative_is_fatal(pair_facts_fixture):
    pairs = build_training_pairs(pair_facts_fixture)
    bad = pairs.with_columns(
        anchor_excludes_negative=pl.lit(True),
        negative_is_explicit_exclusion=pl.lit(True),
        negative_semantic_target=pl.lit('unrelated'),
    )

    with pytest.raises(ValueError, match='training negative cannot be an explicit exclusion'):
        _validate_training_pairs(bad)
```

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
    assert artifacts['training_pairs']['row_count'] == 5
    assert artifacts['training_pairs']['exclusion_count'] == 2
```

with:

```python
    # Two of the five triples ran through an exclusion pair, which is never a negative
    assert artifacts['training_pairs']['row_count'] == 3
    assert artifacts['training_pairs']['exclusion_count'] == 0
    assert manifest['validation_results']['no_exclusion_negatives'] is True
```

Replace:

```python
def test_loader_rejects_training_exclusions_that_disagree_with_pair_facts(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            anchor_excludes_negative=pl.lit(False),
            negative_excludes_anchor=pl.lit(False),
            negative_is_explicit_exclusion=pl.lit(False),
            negative_semantic_target=pl.lit('unknown'),
            negative_semantic_source=pl.lit('unlabeled'),
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)
```

with:

```python
def test_loader_rejects_training_exclusions_that_disagree_with_pair_facts(generated_bundle):
    # Anchor 0's negatives become code 2, its exclusion, with every exclusion flag still false
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            negative_code_id=pl.when(pl.col('anchor_code_id').eq(0)).then(
                pl.lit(2).cast(frame.schema['negative_code_id'])
            ).otherwise(pl.col('negative_code_id'))
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_exclusion_negative(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'training_pairs',
        lambda frame: frame.with_columns(
            anchor_excludes_negative=pl.lit(True),
            negative_is_explicit_exclusion=pl.lit(True),
            negative_semantic_target=pl.lit('unrelated'),
            negative_semantic_source=pl.lit('explicit_exclusion'),
        ),
    )

    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*explicit exclusions of'):
        load_validated_bundle(generated_bundle)
```

In `tests/unit/test_hgcn_streaming_dataset.py`, replace:

```python
    # '111113' is as far from '111111' as the positive '111112' (D* 2), under a farther relation,
    # so it takes the fixed equal-distance margin
    assert negative == {
        'negative_idx': 2,
        'negative_code': '111113',
        'relation_margin': 1.0,
        'distance_margin': pytest.approx(0.3333),
    }
```

with:

```python
    # '111113' is the anchor's exclusion and never a negative, so the first is '222222', across
    # sectors: the fixed relation margin, and a D* difference of 10 - 2
    assert negative == {
        'negative_idx': 3,
        'negative_code': '222222',
        'relation_margin': 15.0,
        'distance_margin': 8.0,
    }
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_hgcn_streaming_dataset.py -q`
Expected: FAIL. The generator still emits the two exclusion triples: `_triples` returns five,
`test_an_exclusion_negative_is_fatal` and the rehashed-exclusion loader test raise nothing, and
the manifest has no `no_exclusion_negatives`.

- [x] **Step 3: Drop exclusions from the generated negatives**

In `src/naics_embedder/data/create_triplets.py`, replace:

```python
directed rows ``p -> j`` and ``a -> j``; cross-sector negatives are capped per (anchor, positive).
A pair is cross-sector when it carries the ``cross_sector`` relation label.
```

with:

```python
directed rows ``p -> j`` and ``a -> j``; cross-sector negatives are capped per (anchor, positive).
A pair is cross-sector when it carries the ``cross_sector`` relation label. An explicit exclusion
of the anchor, in either direction, is never a negative (Req 8).
```

Replace:

```python
def _negative_candidates(anchor_view: pl.DataFrame) -> pl.DataFrame:
    return anchor_view.select(
```

with:

```python
def _negative_candidates(anchor_view: pl.DataFrame) -> pl.DataFrame:
    '''Every candidate of an anchor except its explicit exclusions, in either direction.'''

    return anchor_view.filter(~pl.col('is_explicit_exclusion')).select(
```

Replace:

```python
    Keep at most ``cap`` cross-sector, non-exclusion negatives per (anchor, positive).

    Rows are ranked by a stable hash of (seed, anchor, positive, negative), so the retained subset
    is reproducible across processes and platforms. Explicit exclusions are never capped.
    '''

    capped = (
        pl.col('negative_structural_relation_id').eq(CROSS_SECTOR_RELATION_ID)
        & ~pl.col('negative_is_explicit_exclusion')
    )
```

with:

```python
    Keep at most ``cap`` cross-sector negatives per (anchor, positive).

    Rows are ranked by a stable hash of (seed, anchor, positive, negative), so the retained subset
    is reproducible across processes and platforms.
    '''

    capped = pl.col('negative_structural_relation_id').eq(CROSS_SECTOR_RELATION_ID)
```

Replace:

```python
    if training_pairs.filter(pl.col('positive_is_explicit_exclusion')).height:
        raise ValueError('direct positive cannot be an explicit exclusion')
```

with:

```python
    if training_pairs.filter(pl.col('positive_is_explicit_exclusion')).height:
        raise ValueError('direct positive cannot be an explicit exclusion')
    if training_pairs.filter(pl.col('negative_is_explicit_exclusion')).height:
        raise ValueError('a training negative cannot be an explicit exclusion of its anchor')
```

Replace:

```python
        cross_sector_cap: Maximum cross-sector, non-exclusion negatives per (anchor, positive).
```

with:

```python
        cross_sector_cap: Maximum cross-sector negatives per (anchor, positive).
```

- [x] **Step 4: Refuse an exclusion negative at load and record the check**

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
    Every identity must be a known code ID, no direct positive may be an explicit exclusion,
    exclusion and semantic columns must be internally consistent, and every anchor/positive and
    anchor/negative view must match the pair facts (structure and both exclusion directions).
    Members are checked in bounded chunks of files, which is exact because every row check is
    row-local and every uniqueness check is a join against the pair facts.
```

with:

```python
    Every identity must be a known code ID, no direct positive or negative may be an explicit
    exclusion of its anchor, exclusion and semantic columns must be internally consistent, and
    every anchor/positive and anchor/negative view must match the pair facts (structure and both
    exclusion directions). Members are checked in bounded chunks of files, which is exact because
    every row check is row-local and every uniqueness check is a join against the pair facts.
```

Replace:

```python
        excluded_positives=pl.col('positive_is_explicit_exclusion').sum(),
```

with:

```python
        excluded_positives=pl.col('positive_is_explicit_exclusion').sum(),
        excluded_negatives=pl.col('negative_is_explicit_exclusion').sum(),
```

Replace:

```python
    if summary['excluded_positives']:
        raise ValueError('a direct positive is an explicit exclusion')
```

with:

```python
    if summary['excluded_positives']:
        raise ValueError('a direct positive is an explicit exclusion')
    if summary['excluded_negatives']:
        raise ValueError(
            f'{summary["excluded_negatives"]:,} training negatives are explicit exclusions of '
            'their anchors'
        )
```

In `src/naics_embedder/data/supervision_bundle.py`, replace:

```python
                'direct_positive_safety': True,
                'training_exclusion_derivation': True,
```

with:

```python
                'direct_positive_safety': True,
                'no_exclusion_negatives': True,
                'training_exclusion_derivation': True,
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_hgcn_streaming_dataset.py tests/unit/test_structural_margins.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1655 passed, 1 skipped` (2 new tests).

- [x] **Step 6: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/data/create_triplets.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_hgcn_streaming_dataset.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 7: Commit**

```bash
git add src/naics_embedder/data/create_triplets.py \
  src/naics_embedder/supervision/artifacts.py \
  src/naics_embedder/data/supervision_bundle.py \
  tests/unit/test_data_triplets.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_hgcn_streaming_dataset.py
git commit -m "feat(supervision): no generated training row uses an exclusion as a negative"
```

### Task 8: Runtime selection never picks an exclusion

Req 8(c), runtime half. Task 7 stops the generator emitting exclusion negatives, but the runtime
adds them back: `build_candidate_pool` puts every explicit exclusion of the anchor into the pool,
and `NegativeSelectionCoordinator` reserves one slot per anchor for one of them, rotating by epoch.
This task removes both. The pool holds no exclusion and treats a raw candidate that is one as
corruption; the coordinator never selects one; and the model's eligibility mask, in training and
in validation alike, drops the exclusions a distributed pool brings in from other ranks. With no
rotation left, `select` loses its `epoch` and `global_seed` arguments and the model its
`selection_seed` hyperparameter. The rule that chooses negatives changes, so
`MINING_CONTRACT_VERSION` moves to `negative-selection-v2` and a checkpoint trained under the
quota cannot exact-resume.

`SelectionReason.EXCLUSION_QUOTA` and the `train/integrity/quota_selections` counter stay, always
zero: Stage 7 deletes the quota machinery with the rest of the sampling rules.

**Files:**
- Modify: `src/naics_embedder/supervision/selection.py` (module docstring, the hash section's
  title, `NegativeSelectionCoordinator.select`)
- Modify: `src/naics_embedder/supervision/schema.py` (`MINING_CONTRACT_VERSION`)
- Modify: `src/naics_embedder/text_model/dataloader/streaming_dataset.py` (`build_candidate_pool`,
  the docstrings of `_compute_phase1_weights` and `sample_raw_candidates`)
- Modify: `src/naics_embedder/text_model/mixins/curriculum.py` (`_select_negative_batch`;
  `_selection_seed` goes)
- Modify: `src/naics_embedder/text_model/mixins/logging.py` (`_log_selection_health`)
- Modify: `src/naics_embedder/text_model/mixins/validation.py` (the repaired validation step's
  eligibility)
- Modify: `src/naics_embedder/text_model/naics_model.py` (the `selection_seed` parameter goes)
- Modify: `src/naics_embedder/cli/commands/training.py` (stops passing `selection_seed`)
- Modify (text only): `src/naics_embedder/supervision/margins.py`,
  `src/naics_embedder/utils/config.py`,
  `src/naics_embedder/text_model/dataloader/difficulty_sampler.py`,
  `src/naics_embedder/text_model/loss.py`
- Test: `tests/unit/test_negative_selection.py`, `tests/unit/test_streaming_sampling.py`,
  `tests/unit/test_hard_negative_mining.py`, `tests/unit/test_naics_model.py`,
  `tests/unit/test_cli_training.py`, `tests/unit/test_checkpoint_contract.py`,
  `tests/unit/test_config.py`, `tests/integration/test_stage3_training_step.py`,
  `tests/integration/test_distributed_supervision.py`

**Interfaces:**
- Consumes: Task 7's guarantee that a v2 bundle's training pairs hold no exclusion negative, and
  Task 6's D* distances in the five-code and hierarchy fixtures.
- Produces:
  - `NegativeSelectionCoordinator.select(candidates, *, anchor_code_ids, positive_code_ids, k,
    proposals) -> NegativeSelection`. `epoch` and `global_seed` are gone. It never selects a
    candidate whose `is_explicit_exclusion` is set, and when an anchor lacks `k` other codes it
    raises `ValueError` ending in "available {n} (unique non-exclusion codes)".
  - `build_candidate_pool(...)` keeps its signature. The pool holds no explicit exclusion of the
    anchor in either direction, every item's `negative_is_explicit_exclusion` is false, and a raw
    candidate that is an exclusion raises `ValueError('raw candidate code ID {id} is an explicit
    exclusion of anchor code ID {a}; an exclusion is never a negative')`.
  - `naics_embedder.supervision.schema.MINING_CONTRACT_VERSION == 'negative-selection-v2'`.
  - `NAICSContrastiveModel.__init__` no longer accepts `selection_seed`.
  - `stable_hash` stays in `supervision/selection.py`; the candidate-pool shuffle still uses it.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_negative_selection.py`, replace:

```python
def test_quota_selects_exactly_one_exclusion_and_rotates(candidate_batch_with_exclusions):
    coordinator = NegativeSelectionCoordinator()
    chosen = []
    for epoch in range(3):
        selection = coordinator.select(
            candidate_batch_with_exclusions,
            anchor_code_ids=torch.tensor([10]),
            positive_code_ids=torch.tensor([11]),
            k=3,
            epoch=epoch,
            global_seed=7,
            proposals=(),
        )
        selected = candidate_batch_with_exclusions.select(selection)
        assert selected.is_explicit_exclusion.sum().item() == 1
        chosen.append(selected.code_id[selected.is_explicit_exclusion].item())

    assert len(set(chosen)) == 3

def test_rotation_is_reproducible_for_same_seed_anchor_and_epoch(candidate_batch_with_exclusions):
    coordinator = NegativeSelectionCoordinator()
    args = {
        'candidates': candidate_batch_with_exclusions,
        'anchor_code_ids': torch.tensor([10]),
        'positive_code_ids': torch.tensor([11]),
        'k': 1,
        'epoch': 4,
        'global_seed': 123,
        'proposals': (),
    }

    first = coordinator.select(**args)
    second = coordinator.select(**args)

    assert torch.equal(first.source_indices, second.source_indices)
    assert first.reasons.item() == SelectionReason.EXCLUSION_QUOTA
```

with:

```python
def test_an_explicit_exclusion_is_never_selected(candidate_batch_with_exclusions):
    # Codes 20-22 are the anchor's exclusions (Req 8), so only 30-32 can fill the three slots
    selection = NegativeSelectionCoordinator().select(
        candidate_batch_with_exclusions,
        anchor_code_ids=torch.tensor([10]),
        positive_code_ids=torch.tensor([11]),
        k=3,
        proposals=(),
    )
    selected = candidate_batch_with_exclusions.select(selection)

    assert selected.code_id.tolist() == [[30, 31, 32]]
    assert not selected.is_explicit_exclusion.any()
    assert selection.reasons.tolist() == [[SelectionReason.BACKFILL] * 3]

def test_exclusions_add_no_selection_capacity(candidate_batch_with_exclusions):
    # Three non-exclusion codes cannot fill four slots, however many exclusions the pool holds
    with pytest.raises(ValueError, match='requested 4.*available 3 .unique non-exclusion codes'):
        _select(candidate_batch_with_exclusions, k=4)
```

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Quota protection and eligibility
# -------------------------------------------------------------------------------------------------

def _select(batch, *, k, proposals=(), anchor=10, positive=11, epoch=0):
    return NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([anchor]),
        positive_code_ids=torch.tensor([positive]),
        k=k,
        epoch=epoch,
        global_seed=7,
        proposals=proposals,
    )

def test_proposals_cannot_add_a_second_exclusion(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    greedy = CandidateProposal(
        source_indices=torch.tensor([[0, 1, 2, 3]]),
        scores=torch.tensor([[9.0, 8.0, 7.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )

    selected = batch.select(_select(batch, k=3, proposals=(greedy, )))

    assert selected.is_explicit_exclusion.sum().item() == 1
    assert selected.code_id.unique().numel() == 3
    assert selected.selection_reasons.tolist()[0][0] == SelectionReason.EXCLUSION_QUOTA
    assert 30 in selected.code_id.tolist()[0]

def test_one_slot_is_enough_for_the_reserved_exclusion(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    ordinary_only = CandidateProposal(
        source_indices=torch.tensor([[3, 4, 5]]),
        scores=torch.tensor([[3.0, 2.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )

    selected = batch.select(_select(batch, k=1, proposals=(ordinary_only, )))

    assert selected.is_explicit_exclusion.tolist() == [[True]]

def test_rotation_covers_every_exclusion_cyclically(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    chosen = [batch.select(_select(batch, k=1, epoch=epoch)).code_id.item() for epoch in range(6)]

    assert sorted(chosen[:3]) == [20, 21, 22]
    assert chosen[3:] == chosen[:3]
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Exclusions and eligibility
# -------------------------------------------------------------------------------------------------

def _select(batch, *, k, proposals=(), anchor=10, positive=11):
    return NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([anchor]),
        positive_code_ids=torch.tensor([positive]),
        k=k,
        proposals=proposals,
    )

def test_proposals_cannot_select_an_exclusion(candidate_batch_with_exclusions):
    batch = candidate_batch_with_exclusions
    greedy = CandidateProposal(
        source_indices=torch.tensor([[0, 1, 2, 3]]),
        scores=torch.tensor([[9.0, 8.0, 7.0, 1.0]]),
        reason=SelectionReason.GEOMETRIC,
    )

    selected = batch.select(_select(batch, k=3, proposals=(greedy, )))

    # The three exclusions outscore code 30 but are skipped; backfill supplies 31 and 32
    reasons = selected.selection_reasons.tolist()[0]
    assert selected.code_id.tolist() == [[30, 31, 32]]
    assert reasons == [SelectionReason.GEOMETRIC] + [SelectionReason.BACKFILL] * 2
```

Every other call in the file drops its rotation arguments:

Run: `sed -i '' -e '/^ *epoch=0,$/d' -e '/^ *global_seed=7,$/d' tests/unit/test_negative_selection.py`

Run: `grep -n 'epoch\|global_seed' tests/unit/test_negative_selection.py`
Expected: no output.

In `tests/unit/test_streaming_sampling.py`, replace:

```python
def test_candidate_pool_contains_every_exclusion_and_unique_ordinary_codes(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 12, 13, 14],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=2,
    )

    assert {20, 21, 22} <= {item['negative_code_id'] for item in pool}
    assert len({item['negative_code_id'] for item in pool}) == len(pool)
```

with:

```python
def test_candidate_pool_holds_unique_codes_and_no_exclusion(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 12, 13, 14],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=2,
    )
    codes = [item['negative_code_id'] for item in pool]

    assert len(set(codes)) == len(codes) == 4
    assert not set(codes) & {20, 21, 22}
    assert not any(item['negative_is_explicit_exclusion'] for item in pool)
```

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Candidate-pool capacity under the one-slot exclusion quota
# -------------------------------------------------------------------------------------------------

def _selection_capacity(pool: list[dict[str, Any]]) -> int:
    exclusions = sum(item['negative_is_explicit_exclusion'] for item in pool)
    ordinary = len(pool) - exclusions
    return ordinary + min(exclusions, 1)

def test_pool_supports_k_selections_when_several_exclusions_exist(pool_builder):
    # Three exclusions and K = 4: selection takes exactly one exclusion, so the pool needs at least
    # three ordinary codes; sizing by max(n, K, E) alone would leave only one.
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 13],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=0,
    )

    assert _selection_capacity(pool) >= 4
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Candidate-pool capacity: no exclusion fills a slot
# -------------------------------------------------------------------------------------------------

def test_exclusions_never_count_toward_pool_capacity(pool_builder):
    # Three exclusions and K = 4: none may fill a slot, so backfill supplies the two codes the raw
    # candidates lack.
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12, 13],
        exclusion_code_ids=(20, 21, 22),
        n_candidates=4,
        epoch=0,
    )
    codes = {item['negative_code_id'] for item in pool}

    assert len(codes) == 4
    assert {12, 13} <= codes
    assert not codes & {20, 21, 22}
```

Replace:

```python
    selection = NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([10]),
        positive_code_ids=torch.tensor([11]),
        k=4,
        epoch=1,
        global_seed=7,
        proposals=(),
    )

    assert batch.select(selection).is_explicit_exclusion.sum().item() == 1
```

with:

```python
    selection = NegativeSelectionCoordinator().select(
        batch,
        anchor_code_ids=torch.tensor([10]),
        positive_code_ids=torch.tensor([11]),
        k=4,
        proposals=(),
    )
    selected = batch.select(selection)

    assert selected.code_id.unique().numel() == 4
    assert not selected.is_explicit_exclusion.any()
```

Replace:

```python
    codes = {item['negative_code_id'] for item in pool}
    assert not codes & {10, 11}
    assert 20 in codes
```

with:

```python
    codes = {item['negative_code_id'] for item in pool}
    assert not codes & {10, 11, 20}
```

Replace:

```python
def test_pool_marks_exclusions_and_backfill_provenance(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12],
        exclusion_code_ids=(20, ),
        n_candidates=3,
        epoch=0,
    )
    by_code = {item['negative_code_id']: item for item in pool}

    assert by_code[20]['negative_is_explicit_exclusion'] is True
    assert by_code[12]['negative_is_explicit_exclusion'] is False
    assert by_code[12]['sampling_provenance_id'] == 2
```

with:

```python
def test_pool_marks_backfill_provenance_and_holds_no_exclusion(pool_builder):
    pool = pool_builder(
        anchor_code_id=10,
        positive_code_id=11,
        raw_candidate_code_ids=[12],
        exclusion_code_ids=(20, ),
        n_candidates=3,
        epoch=0,
    )
    by_code = {item['negative_code_id']: item for item in pool}

    assert 20 not in by_code
    assert not any(item['negative_is_explicit_exclusion'] for item in pool)
    assert by_code[12]['sampling_provenance_id'] == 2
```

Replace:

```python
def test_pool_is_reproducible_for_one_epoch(pool_builder):
```

with:

```python
def test_pool_rejects_a_raw_candidate_that_is_an_exclusion(pool_builder):
    # Generated training pairs hold no exclusion negative, so a raw candidate that is one is corrupt
    with pytest.raises(ValueError, match='raw candidate code ID 20 is an explicit exclusion'):
        pool_builder(
            anchor_code_id=10,
            positive_code_id=11,
            raw_candidate_code_ids=[12, 20],
            exclusion_code_ids=(20, ),
            n_candidates=3,
            epoch=0,
        )

def test_pool_is_reproducible_for_one_epoch(pool_builder):
```

Replace:

```python
# For anchor '311111' (4) with its grandparent '3111' (2; grandchild relation, distance 1.5) as the
# positive, only the parent '31111' (3; child relation, distance 0.5) is structurally closer. Every
# other non-forbidden code is farther: ancestors '31'/'311' (0, 1), the collateral codes 5-10, the
# exclusion '321111' (11), and the cross-sector '44' family (12-16).
# -------------------------------------------------------------------------------------------------

HIERARCHY_ELIGIBLE_ORDINARY = {0, 1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16}
```

with:

```python
# For anchor '311111' (4) with its grandparent '3111' (2; D* 2) as the positive, only the parent
# '31111' (3; D* 1) is structurally closer. Every other non-forbidden code is farther: ancestors
# '31'/'311' (0, 1), the collateral codes 5-10 and the cross-sector '44' family (12-16). So is the
# exclusion '321111' (11), which is nonetheless never a negative.
# -------------------------------------------------------------------------------------------------

HIERARCHY_ELIGIBLE = {0, 1, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16}
```

Replace:

```python
    codes = [candidate['negative_code_id'] for candidate in pool]
    ordinary = {
        candidate['negative_code_id']
        for candidate in pool if not candidate['negative_is_explicit_exclusion']
    }
    # The universe is exhausted, yet the parent is never backfilled.
    assert 3 not in codes
    assert ordinary == HIERARCHY_ELIGIBLE_ORDINARY
    assert codes[0] == 11
```

with:

```python
    codes = [candidate['negative_code_id'] for candidate in pool]
    # The universe is exhausted, yet neither the parent nor the exclusion is backfilled.
    assert sorted(codes) == sorted(HIERARCHY_ELIGIBLE)
```

Replace:

```python
def test_pool_capacity_counts_only_structurally_eligible_codes(hierarchy_index):
    with pytest.raises(ValueError, match='requires 15 .* structurally farther'):
        build_candidate_pool(
            anchor_code_id=4,
            positive_code_id=2,
            raw_candidates=[],
            supervision_index=hierarchy_index,
            n_candidates=15,
            final_k=15,
            epoch=0,
            seed=0,
        )
```

with:

```python
def test_pool_capacity_counts_only_eligible_non_exclusion_codes(hierarchy_index):
    # Thirteen codes are eligible, and the exclusion no longer adds a fourteenth slot
    with pytest.raises(ValueError, match='requires 14 .* only 13 exist'):
        build_candidate_pool(
            anchor_code_id=4,
            positive_code_id=2,
            raw_candidates=[],
            supervision_index=hierarchy_index,
            n_candidates=14,
            final_k=14,
            epoch=0,
            seed=0,
        )
```

In `tests/unit/test_hard_negative_mining.py`, replace:

```python
from dataclasses import replace
from types import SimpleNamespace
```

with:

```python
from dataclasses import replace
```

Replace:

```python
        self.current_epoch = 0
        self.hparams = SimpleNamespace(selection_seed=7)
```

with:

```python
        self.current_epoch = 0
```

Replace:

```python
    codes = selected.code_id.tolist()
    assert selected.is_explicit_exclusion.tolist()[0].count(True) == 1
    assert 2 in codes[0]
    assert all(code >= 0 for row in codes for code in row)
```

with:

```python
    codes = selected.code_id.tolist()
    # Code 2 fills a slot in neither row: it is anchor 0's exclusion, never a negative (Req 8)
    assert not selected.is_explicit_exclusion.any()
    assert [sorted(row) for row in codes] == [[3, 4], [3, 4]]
```

Replace:

```python
    # Positive: code 1 at D* 2 (relation 1). Code 2 is the anchor's sibling exclusion at the same
    # D* with a farther relation (relation margin 1, the fixed equal-distance margin); codes 3 and
    # 4 are cross-sector (relation margin 15, distance margin 10 - 2).
    expected_margins = {2: (1.0, 0.3333), 3: (15.0, 8.0), 4: (15.0, 8.0)}
    for row in range(selected.code_id.shape[0]):
        for slot, code in enumerate(selected.code_id[row].tolist()):
            assert selected.relation_margin[row, slot].item() == expected_margins[code][0]
            assert selected.distance_margin[row, slot].item() == pytest.approx(
                expected_margins[code][1]
            )
```

with:

```python
    # Positive: code 1 at D* 2 (relation 1). Codes 3 and 4 are cross-sector: relation margin 15,
    # distance margin 10 - 2.
    assert selected.relation_margin.tolist() == [[15.0, 15.0], [15.0, 15.0]]
    assert selected.distance_margin.tolist() == [[8.0, 8.0], [8.0, 8.0]]
```

Replace:

```python
    # The difficulty proposal covers the whole pool, yet mining decides once it is enabled.
    assert _reasons(mined) == [
        SelectionReason.EXCLUSION_QUOTA,
        SelectionReason.GEOMETRIC,
        SelectionReason.GEOMETRIC,
        SelectionReason.ROUTER,
    ]
    assert _reasons(unmined) == [SelectionReason.EXCLUSION_QUOTA] + [SelectionReason.DIFFICULTY] * 3
    # Geometric slots go to the codes nearest the anchor embedding (code 13's): on the hyperboloid
    # d(x, y) = |asinh(x) - asinh(y)| here, so 1.4 (code 14) is nearer 1.3 than 1.2 (code 12).
    assert set(mined.code_id[0, 1:3].tolist()) == {13, 14}
```

with:

```python
    # The difficulty proposal covers the whole pool, yet mining decides once it is enabled. The
    # exclusion takes no slot either way.
    assert _reasons(mined) == [SelectionReason.GEOMETRIC] * 2 + [SelectionReason.ROUTER] * 2
    assert _reasons(unmined) == [SelectionReason.DIFFICULTY] * 4
    assert HIERARCHY_EXCLUSION not in mined.code_id[0].tolist() + unmined.code_id[0].tolist()
    # Geometric slots go to the codes nearest the anchor embedding (code 13's): on the hyperboloid
    # d(x, y) = |asinh(x) - asinh(y)| here, so 1.4 (code 14) is nearer 1.3 than 1.2 (code 12).
    assert set(mined.code_id[0, :2].tolist()) == {13, 14}
```

Replace:

```python
        (0.0, [SelectionReason.GEOMETRIC] * 3),
        (1.0, [SelectionReason.ROUTER] * 3),
```

with:

```python
        (0.0, [SelectionReason.GEOMETRIC] * 4),
        (1.0, [SelectionReason.ROUTER] * 4),
```

Replace:

```python
    assert _reasons(selected) == [SelectionReason.EXCLUSION_QUOTA] + expected
```

with:

```python
    assert _reasons(selected) == expected
```

Replace:

```python
    assert _reasons(selected) == [SelectionReason.EXCLUSION_QUOTA] + [SelectionReason.GEOMETRIC] * 3
    assert selected.code_id[0].tolist() == [HIERARCHY_EXCLUSION, 13, 14, 12]
    # The duplicate code resolves to its smallest-UID occurrence (slot 1).
    assert selected.candidate_uid[0, 1].tolist() == [0, 0, 1]
```

with:

```python
    assert _reasons(selected) == [SelectionReason.GEOMETRIC] * 4
    assert selected.code_id[0].tolist() == [13, 14, 12, 15]
    # The duplicate code resolves to its smallest-UID occurrence (slot 1).
    assert selected.candidate_uid[0, 0].tolist() == [0, 0, 1]
```

Replace:

```python
        assert HIERARCHY_PARENT not in selected.code_id[0].tolist()
        assert HIERARCHY_EXCLUSION in selected.code_id[0].tolist()
        assert (entity_valid & ~candidates.valid_mask)[0].tolist() == [True] + [False] * 6
```

with:

```python
        assert HIERARCHY_PARENT not in selected.code_id[0].tolist()
        # The parent fails the structural rule and the exclusion fails Req 8
        assert (entity_valid & ~candidates.valid_mask)[0].tolist() == [True, True] + [False] * 5

def test_an_exclusion_is_never_selected_even_when_nearest(hierarchy_index):
    # The exclusion leads the difficulty proposal and sits at the anchor embedding, yet an
    # exclusion pair is never a code-code negative (Req 8), so no path may select it.
    pool = [HIERARCHY_EXCLUSION] + CROSS_SECTOR_CODES
    anchor_embedding = _code_embedding(HIERARCHY_EXCLUSION)

    for flags in ({}, {'enable_hard_negative_mining': True}):
        _, selected = _select_with_flags(
            hierarchy_index, flags, pool, HIERARCHY_GRANDPARENT, 4, anchor_embedding
        )

        assert HIERARCHY_EXCLUSION not in selected.code_id[0].tolist()
        assert not selected.is_explicit_exclusion.any()
```

In `tests/unit/test_naics_model.py`, replace:

```python
    pool: list,
    selection_k: int = 3,
) -> dict:
```

with:

```python
    pool: list,
    selection_k: int = 2,
) -> dict:
```

Replace:

```python
    '''
    Two repaired rows with uneven pools: row 0 (anchor 0) holds its exclusion (2) and one padding
    row; row 1 (anchor 2) repeats code 4 and has no exclusion in its pool.
    '''
```

with:

```python
    '''
    Two repaired rows with uneven pools: row 0 (anchor 0) holds its exclusion (2), which no path
    selects, and one padding row; row 1 (anchor 2) repeats code 4 and has no exclusion in its
    pool. Each row selects two negatives.
    '''
```

Replace:

```python
        '''Pseudo-related masking happens after selection and never clears an exclusion.'''
```

with:

```python
        '''Pseudo-related masking happens after selection, on the selected negatives only.'''
```

Replace:

```python
        assert flags == {2: False, 3: True, 4: False}
```

with:

```python
        assert flags == {3: True, 4: False}
```

Replace:

```python
        # Mining is off, so the difficulty proposal fills every ordinary slot: row 0 selects its
        # exclusion by quota plus two codes; row 1 (no exclusion in its pool) selects three.
        assert counters == {
            'train/integrity/anchors_with_exclusions': 1.0,
            'train/integrity/quota_selections': 1.0,
            'train/integrity/geometric_selections': 0.0,
            'train/integrity/router_selections': 0.0,
            'train/integrity/difficulty_selections': 5.0,
```

with:

```python
        # Mining is off, so the difficulty proposal fills every slot, two per row. Row 0's
        # exclusion sits in its pool but takes no slot, and the structural counter leaves it out.
        assert counters == {
            'train/integrity/anchors_with_exclusions': 1.0,
            'train/integrity/quota_selections': 0.0,
            'train/integrity/geometric_selections': 0.0,
            'train/integrity/router_selections': 0.0,
            'train/integrity/difficulty_selections': 4.0,
```

Replace:

```python
        '''An anchor without K selectable codes aborts the step with its code ID.'''

        batch = collate_fn(
            [_repaired_item(0, 1, 0.5, 1, [2, 3], selection_k=3)],
            supervision_mode='repaired',
        )
```

with:

```python
        '''An anchor without K selectable codes aborts the step; its exclusion adds no capacity.'''

        batch = collate_fn(
            [_repaired_item(0, 1, 0.5, 1, [2, 3, 4], selection_k=3)],
            supervision_mode='repaired',
        )
```

Replace:

```python
    def test_validation_step_embedding_storage(self, naics_model, repaired_training_batch):
```

with:

```python
    def test_validation_step_never_scores_an_exclusion(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Validation applies training's eligibility, so an exclusion in the pool is not scored.'''

        captured = {}
        forward = naics_model.loss_fn.forward

        def spy(*args, **kwargs):
            captured['valid_mask'] = kwargs['valid_mask'].tolist()
            return forward(*args, **kwargs)

        monkeypatch.setattr(naics_model.loss_fn, 'forward', spy)
        naics_model.eval()
        with torch.no_grad():
            naics_model.validation_step(repaired_training_batch, batch_idx=0)

        # Row 0's pool is [2, 3, 4] plus padding, and code 2 is anchor 0's exclusion
        assert captured['valid_mask'] == [[False, True, True, False], [True, True, True, True]]

    def test_validation_step_embedding_storage(self, naics_model, repaired_training_batch):
```

In `tests/unit/test_config.py`, replace:

```python
        match='phase1_exclusion_weight.*one-slot exclusion quota',
```

with:

```python
        match='phase1_exclusion_weight.*an explicit exclusion is never a negative',
```

In `tests/unit/test_cli_training.py`, replace:

```python
    assert model_kwargs['selection_seed'] == 42
```

with:

```python
    assert 'selection_seed' not in model_kwargs
```

Append to the end of `tests/unit/test_checkpoint_contract.py`:

```python

def test_a_checkpoint_trained_under_the_exclusion_quota_cannot_exact_resume(
    tmp_path, runtime_contract
):
    # negative-selection-v1 reserved a slot for an explicit exclusion; v2 never selects one
    path = tmp_path / 'quota.ckpt'
    saved = {**runtime_contract.model_dump(), 'mining_contract_version': 'negative-selection-v1'}
    torch.save({'stage3_supervision': saved}, path)

    assert runtime_contract.mining_contract_version == 'negative-selection-v2'
    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)
```

In `tests/integration/test_stage3_training_step.py`, replace:

```python
    monkeypatch.setattr(
        tiny_repaired_model.selection_coordinator,
        'select',
        forced_selection([2, 0, 1]),
    )

    loss = tiny_repaired_model.training_step(repaired_training_batch, batch_idx=0)
    loss.backward()

    assert spy.contrastive_uids == spy.structural_uids
    assert spy.code_ids == [4, 2, 3]
    assert spy.structural_distances == [10.0, 2.0, 10.0]
    assert spy.exclusion_flags == [False, True, False]
    # The stub computes gate probabilities in float32, so compare approximately.
    assert spy.router_first_column == pytest.approx([0.3, 0.1, 0.2])
    assert spy.false_negative_flags == [False, False, True]
```

with:

```python
    # Slots 2 and 1, reversed: codes 4 and 3. Slot 0 holds code 2, anchor 0's exclusion, which is
    # never selectable.
    monkeypatch.setattr(
        tiny_repaired_model.selection_coordinator,
        'select',
        forced_selection([2, 1]),
    )

    loss = tiny_repaired_model.training_step(repaired_training_batch, batch_idx=0)
    loss.backward()

    assert spy.contrastive_uids == spy.structural_uids
    assert spy.code_ids == [4, 3]
    assert spy.structural_distances == [10.0, 10.0]
    assert spy.exclusion_flags == [False, False]
    # The stub computes gate probabilities in float32, so compare approximately.
    assert spy.router_first_column == pytest.approx([0.3, 0.2])
    assert spy.false_negative_flags == [False, True]
```

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Real coordinator: mining decides once enabled, and never selects a structurally closer relative
# -------------------------------------------------------------------------------------------------
```

with:

```python
def test_a_selection_naming_an_exclusion_is_refused(
    tiny_repaired_model,
    repaired_training_batch,
    monkeypatch,
):
    # Slot 0 holds code 2, anchor 0's exclusion: it is never eligible (Req 8), so a selection that
    # names it cannot reach a loss
    monkeypatch.setattr(
        tiny_repaired_model.selection_coordinator,
        'select',
        forced_selection([0, 1]),
    )

    with pytest.raises(ValueError, match='invalid source candidate at row 0, slot 0'):
        tiny_repaired_model.training_step(repaired_training_batch, batch_idx=0)

# -------------------------------------------------------------------------------------------------
# Real coordinator: mining decides once enabled, and never selects a structurally closer relative
# or an exclusion
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
            [
                SelectionReason.EXCLUSION_QUOTA,
                SelectionReason.GEOMETRIC,
                SelectionReason.GEOMETRIC,
                SelectionReason.ROUTER,
            ],
        ),
        ({}, [SelectionReason.EXCLUSION_QUOTA] + [SelectionReason.DIFFICULTY] * 3),
```

with:

```python
            [SelectionReason.GEOMETRIC] * 2 + [SelectionReason.ROUTER] * 2,
        ),
        ({}, [SelectionReason.DIFFICULTY] * 4),
```

Replace:

```python
    # The parent leads the difficulty proposal and is the anchor's nearest code, yet it is
    # structurally closer than the grandparent positive, so no path may select it.
    assert 3 not in selected.code_id[0].tolist()
    assert 11 in selected.code_id[0].tolist()
```

with:

```python
    # The parent leads the difficulty proposal and is the anchor's nearest code, yet it is
    # structurally closer than the grandparent positive, so no path may select it. Nor may any
    # path select the exclusion (11).
    assert 3 not in selected.code_id[0].tolist()
    assert 11 not in selected.code_id[0].tolist()
```

In `tests/integration/test_distributed_supervision.py`, replace:

```python
from types import SimpleNamespace

import pytest
```

with:

```python
import pytest
```

Replace:

```python
        self.current_epoch = 0
        self.hparams = SimpleNamespace(selection_seed=0)
```

with:

```python
        self.current_epoch = 0
```

Replace:

```python
    # Rank 0: its exclusion by quota, then the two geometrically nearest eligible codes, both
    # remote. Its parent (3) is nearest of all but structurally closer than the grandparent
    # positive, so it is ineligible for this anchor and never selected.
    assert rank0[1] == [11, 1, 0]
    assert rank0[2] == [0, 1, 1]
    assert rank0[3] == [True, False, False]
    assert rank0[5] >= 1
    # Rank 1: its exclusion by quota, then the nearest eligible codes, both from rank 0.
    assert rank1[1] == [7, 13, 12]
    assert rank1[2] == [1, 0, 0]
    assert rank1[3] == [True, False, False]
```

with:

```python
    # Rank 0: the three geometrically nearest eligible codes, all remote. Its parent (3) is nearest
    # of all but structurally closer than the grandparent positive, and its exclusion (11) is never
    # a negative: both are ineligible for this anchor. Rank 1's exclusion (7) is an ordinary code
    # here, since exclusions belong to an anchor.
    assert rank0[1] == [1, 0, 7]
    assert rank0[2] == [1, 1, 1]
    assert rank0[3] == [False, False, False]
    assert rank0[5] == 2
    # Rank 1: the nearest eligible codes, all from rank 0 and rank 0's exclusion among them; its
    # own exclusion (7) is never selected.
    assert rank1[1] == [13, 12, 11]
    assert rank1[2] == [0, 0, 0]
    assert rank1[3] == [False, False, False]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_negative_selection.py tests/unit/test_streaming_sampling.py tests/unit/test_hard_negative_mining.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_checkpoint_contract.py tests/unit/test_config.py tests/integration/test_stage3_training_step.py tests/integration/test_distributed_supervision.py -q`
Expected: FAIL. `select` still requires `epoch` and `global_seed`, so every coordinator test
raises `TypeError`; the pool still carries the anchor's exclusions; the quota still fills a slot;
validation still scores the exclusion; the model still takes `selection_seed`; and the mining
contract is still `negative-selection-v1`.

- [x] **Step 3: Select without a reserved exclusion**

In `src/naics_embedder/supervision/selection.py`, replace:

```python
'''
Deterministic final negative selection with a protected, rotating one-slot exclusion quota.

For each anchor the coordinator (1) reserves exactly one explicit exclusion when any exists,
rotating through the anchor's exclusions by epoch; (2) merges strategy proposals in priority
order over non-exclusion candidates; (3) deduplicates by code ID, keeping the smallest occurrence
UID; (4) breaks ties by code ID, then UID; and (5) backfills deterministically. It returns one
``NegativeSelection`` of source indices; it never gathers candidate fields itself.
'''
```

with:

```python
'''
Deterministic final negative selection over non-exclusion candidates.

For each anchor the coordinator (1) merges strategy proposals in priority order; (2) deduplicates
by code ID, keeping the smallest occurrence UID; (3) breaks ties by code ID, then UID; and (4)
backfills deterministically. An explicit exclusion is never selected: an exclusion pair is not a
code-code negative (Req 8), so no slot is reserved for one. The coordinator returns one
``NegativeSelection`` of source indices; it never gathers candidate fields itself.
'''
```

Replace:

```python
# Stable rotation hash
```

with:

```python
# Stable hash
```

Replace:

```python
        positive_code_ids: torch.Tensor,
        k: int,
        epoch: int,
        global_seed: int,
        proposals: Sequence[CandidateProposal],
    ) -> NegativeSelection:
        '''
        Select ``k`` unique negative codes per anchor.

        Raises:
            ValueError: If ``k < 1``, shapes disagree, or an anchor lacks ``k`` selectable codes
                (one exclusion slot plus unique non-exclusion codes).
        '''
```

with:

```python
        positive_code_ids: torch.Tensor,
        k: int,
        proposals: Sequence[CandidateProposal],
    ) -> NegativeSelection:
        '''
        Select ``k`` unique negative codes per anchor, never an explicit exclusion.

        Raises:
            ValueError: If ``k < 1``, shapes disagree, or an anchor lacks ``k`` selectable codes
                (unique non-exclusion codes).
        '''
```

Replace:

```python
            for index in range(pool_size):
                if not valid_rows[row][index]:
                    continue
                code_id = int(codes[index])
```

with:

```python
            for index in range(pool_size):
                if not valid_rows[row][index] or explicit_rows[row][index]:
                    continue
                code_id = int(codes[index])
```

Replace:

```python
            exclusion_codes = sorted(
                code_id for code_id, index in available_by_code.items() if explicit_rows[row][index]
            )
            ordinary_codes: Set[int] = {
                code_id
                for code_id, index in available_by_code.items() if not explicit_rows[row][index]
            }
            chosen: List[int] = []
            chosen_scores: List[float] = []
            chosen_reasons: List[int] = []
            chosen_codes: Set[int] = set()

            if exclusion_codes:
                rotation = (stable_hash(global_seed, int(anchors[row])) + epoch) % len(
                    exclusion_codes
                )
                reserved_code = exclusion_codes[rotation]
                chosen.append(available_by_code[reserved_code])
                chosen_scores.append(math.inf)
                chosen_reasons.append(int(SelectionReason.EXCLUSION_QUOTA))
                chosen_codes.add(reserved_code)

```

with:

```python
            chosen: List[int] = []
            chosen_scores: List[float] = []
            chosen_reasons: List[int] = []
            chosen_codes: Set[int] = set()

```

Replace:

```python
                    if code_id not in ordinary_codes or code_id in chosen_codes:
```

with:

```python
                    if code_id not in available_by_code or code_id in chosen_codes:
```

Replace:

```python
                    (code_id, uids[index], index) for code_id, index in available_by_code.items()
                    if code_id in ordinary_codes and code_id not in chosen_codes
```

with:

```python
                    (code_id, uids[index], index) for code_id, index in available_by_code.items()
                    if code_id not in chosen_codes
```

Replace:

```python
            if len(chosen) != k:
                capacity = (1 if exclusion_codes else 0) + len(ordinary_codes)
                raise ValueError(
                    f'anchor code ID {int(anchors[row])} requested {k} unique negative codes; '
                    f'available {capacity} (one exclusion slot plus unique non-exclusion codes)'
                )
```

with:

```python
            if len(chosen) != k:
                raise ValueError(
                    f'anchor code ID {int(anchors[row])} requested {k} unique negative codes; '
                    f'available {len(available_by_code)} (unique non-exclusion codes)'
                )
```

In `src/naics_embedder/supervision/schema.py`, replace:

```python
MINING_CONTRACT_VERSION = 'negative-selection-v1'
```

with:

```python
# v2: no slot is reserved for an explicit exclusion, which is never a negative (Req 8)
MINING_CONTRACT_VERSION = 'negative-selection-v2'
```

- [x] **Step 4: Build pools without exclusions**

In `src/naics_embedder/text_model/dataloader/streaming_dataset.py`, replace:

```python
        exclusion_weight: Legacy-containment constant weight for excluded codes. ``None`` gives
            exclusions no special weight (repaired training reserves exactly one exclusion slot
            at selection time instead).
```

with:

```python
        exclusion_weight: Legacy-containment constant weight for excluded codes. ``None`` gives
            exclusions no special weight: repaired training never uses an exclusion as a negative.
```

Replace:

```python
    Build one canonical candidate pool for an (anchor, positive) pair.

    The pool contains every explicit exclusion of the anchor (either direction) and unique
    ordinary codes: the raw candidates in a stable, epoch-dependent shuffle, backfilled from the
    remaining non-exclusion universe when needed. Because final selection admits exactly one
    exclusion, the pool keeps at least ``final_k - 1`` ordinary codes when any exclusion exists (or
    ``final_k`` when none does), and at least ``n_candidates - exclusions``. The anchor and positive
    codes never appear. Ordinary codes are structurally farther than the positive, the rule every
    generated training negative satisfies; backfill draws only such codes.

    Raises:
        ValueError: If ``final_k < 1``, a raw ordinary candidate is not structurally farther than
            the positive, or the universe cannot supply ``final_k`` selectable codes.
```

with:

```python
    Build one canonical candidate pool for an (anchor, positive) pair.

    The pool holds unique codes: the raw candidates in a stable, epoch-dependent shuffle,
    backfilled from the remaining universe when needed, at least ``max(n_candidates, final_k)`` of
    them. The anchor and positive codes never appear, and neither does an explicit exclusion of
    the anchor in either direction: an exclusion pair is never a code-code negative (Req 8). Every
    code is structurally farther than the positive, the rule every generated training negative
    satisfies; backfill draws only such codes.

    Raises:
        ValueError: If ``final_k < 1``, a raw candidate is an explicit exclusion of the anchor or
            is not structurally farther than the positive, or the universe cannot supply
            ``final_k`` selectable codes.
```

Replace:

```python
        normalized['negative_code_id'] = int(item['negative_code_id'])
        normalized['negative_is_explicit_exclusion'] = normalized['negative_code_id'
                                                                  ] in exclusion_set
        return normalized
```

with:

```python
        normalized['negative_code_id'] = int(item['negative_code_id'])
        normalized['negative_is_explicit_exclusion'] = False
        return normalized
```

Replace:

```python
            'negative_is_explicit_exclusion': code_id in exclusion_set,
            'sampling_role_id': NEGATIVE_ROLE_ID,
```

with:

```python
            'negative_is_explicit_exclusion': False,
            'sampling_role_id': NEGATIVE_ROLE_ID,
```

Replace:

```python
        if code_id in forbidden:
            continue
        if code_id not in exclusion_set and not eligible_codes[code_id]:
            raise ValueError(
                f'raw candidate code ID {code_id} for anchor code ID {anchor_code_id} is not '
                f'structurally farther than positive code ID {positive_code_id}'
            )
        by_code.setdefault(code_id, normalized_candidate(item))
    for code_id in exclusion_ids:
        by_code.setdefault(code_id, backfill_candidate(code_id))

    exclusion_slots = min(len(exclusion_ids), 1)
    ordinary_target = max(n_candidates - len(exclusion_ids), final_k - exclusion_slots, 0)
    rng = np.random.default_rng((stable_hash(seed, anchor_code_id) + epoch) % (2**63))
    ordinary_ids = sorted(code_id for code_id in by_code if code_id not in exclusion_set)
    rng.shuffle(ordinary_ids)
    kept_ordinary = ordinary_ids[:ordinary_target]

    if len(kept_ordinary) < ordinary_target:
        universe = [
            code_id for code_id in range(len(supervision_index.id_to_code))
            if eligible_codes[code_id] and code_id not in forbidden and code_id not in exclusion_set
            and code_id not in by_code
        ]
        rng.shuffle(universe)
        for code_id in universe[:ordinary_target - len(kept_ordinary)]:
            by_code[code_id] = backfill_candidate(code_id)
            kept_ordinary.append(code_id)

    if len(kept_ordinary) + exclusion_slots < final_k:
        raise ValueError(
            f'anchor code ID {anchor_code_id} requires {final_k} selectable candidates; only '
            f'{len(kept_ordinary) + exclusion_slots} exist (one exclusion slot plus unique '
            'non-exclusion codes structurally farther than the positive)'
        )
    return [by_code[code_id] for code_id in (*exclusion_ids, *kept_ordinary)]
```

with:

```python
        if code_id in forbidden:
            continue
        if code_id in exclusion_set:
            raise ValueError(
                f'raw candidate code ID {code_id} is an explicit exclusion of anchor code ID '
                f'{anchor_code_id}; an exclusion is never a negative'
            )
        if not eligible_codes[code_id]:
            raise ValueError(
                f'raw candidate code ID {code_id} for anchor code ID {anchor_code_id} is not '
                f'structurally farther than positive code ID {positive_code_id}'
            )
        by_code.setdefault(code_id, normalized_candidate(item))

    target = max(n_candidates, final_k)
    rng = np.random.default_rng((stable_hash(seed, anchor_code_id) + epoch) % (2**63))
    kept = sorted(by_code)
    rng.shuffle(kept)
    kept = kept[:target]

    if len(kept) < target:
        universe = [
            code_id for code_id in range(len(supervision_index.id_to_code))
            if eligible_codes[code_id] and code_id not in forbidden and code_id not in exclusion_set
            and code_id not in by_code
        ]
        rng.shuffle(universe)
        for code_id in universe[:target - len(kept)]:
            by_code[code_id] = backfill_candidate(code_id)
            kept.append(code_id)

    if len(kept) < final_k:
        raise ValueError(
            f'anchor code ID {anchor_code_id} requires {final_k} selectable candidates; only '
            f'{len(kept)} exist (unique non-exclusion codes structurally farther than the '
            'positive)'
        )
    return [by_code[code_id] for code_id in kept]
```

Replace:

```python
    Exclusion representation belongs to the candidate pool (every exclusion) and the selection
    quota (exactly one), never to a sampling weight.
```

with:

```python
    An explicit exclusion is never a negative (Req 8): no sampling weight, candidate pool or
    selection admits one.
```

- [x] **Step 5: Drop exclusions from the model's eligibility**

In `src/naics_embedder/text_model/mixins/curriculum.py`, replace:

```python
    def _selection_seed(self) -> int:
        '''Global seed for deterministic exclusion rotation (the training seed).'''
        return int(getattr(self.hparams, 'selection_seed', 0))

    def _select_negative_batch(
```

with:

```python
    def _select_negative_batch(
```

Replace:

```python
        selection (exclusion quota, dedup, backfill) -> the single gather.
```

with:

```python
        selection (dedup, backfill) -> the single gather.
```

Replace:

```python
        # Anchor-relative margins follow the generator's rule: an ordinary candidate must be
        # structurally farther than the positive, exactly like every generated training negative.
        # Explicit exclusions are exempt. Runtime-sourced candidates (universe backfill, remote
        # ranks) that fail the rule are ineligible for this anchor, so no loss repels a relative
        # the generated supervision would never treat as a negative.
```

with:

```python
        # Anchor-relative margins follow the generator's rule: a candidate must be structurally
        # farther than the positive, exactly like every generated training negative, and never an
        # explicit exclusion of the anchor (Req 8). Runtime-sourced candidates (universe backfill,
        # remote ranks) that fail either rule are ineligible for this anchor, so no loss repels a
        # relative or an exclusion the generated supervision would never treat as a negative.
```

Replace:

```python
        eligible = entities.valid_mask & (pair.is_explicit_exclusion | structurally_farther)
```

with:

```python
        eligible = entities.valid_mask & ~pair.is_explicit_exclusion & structurally_farther
```

Replace:

```python
            k=selection_k,
            epoch=int(self.current_epoch),
            global_seed=self._selection_seed(),
            proposals=tuple(proposals),
```

with:

```python
            k=selection_k,
            proposals=tuple(proposals),
```

In `src/naics_embedder/text_model/mixins/logging.py`, replace:

```python
        ``entity_valid_mask`` marks real (non-padding) candidates; ``candidates.valid_mask``
        additionally applies anchor-relative structural eligibility. Counters carry no candidate
        identities in their names or values.
```

with:

```python
        ``entity_valid_mask`` marks real (non-padding) candidates; ``candidates.valid_mask``
        additionally applies anchor-relative eligibility: structurally farther than the positive
        and never an explicit exclusion. The structural counter leaves exclusions out, which
        ``anchors_with_exclusions`` reports. Counters carry no candidate identities in their names
        or values.
```

Replace:

```python
                entity_valid_mask & ~candidates.valid_mask
```

with:

```python
                entity_valid_mask & ~candidates.valid_mask & ~candidates.is_explicit_exclusion
```

In `src/naics_embedder/text_model/mixins/validation.py`, replace:

```python
            # The same eligibility as training selection: ordinary candidates must be
            # structurally farther than the positive; explicit exclusions are exempt.
```

with:

```python
            # The same eligibility as training selection: a candidate must be structurally
            # farther than the positive and never an explicit exclusion of the anchor (Req 8).
```

Replace:

```python
            valid_mask = batch['candidate_valid_mask'] & (explicit | structurally_farther)
```

with:

```python
            valid_mask = batch['candidate_valid_mask'] & ~explicit & structurally_farther
```

In `src/naics_embedder/text_model/naics_model.py`, replace:

```python
        selection_seed: Global seed for deterministic exclusion rotation
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
```

with:

```python
        checkpoint_contract: Optional runtime contract; must match the loaded bundle
```

Replace:

```python
        selection_seed: int = 0,
        checkpoint_contract: Optional[CheckpointContract] = None,
```

with:

```python
        checkpoint_contract: Optional[CheckpointContract] = None,
```

In `src/naics_embedder/cli/commands/training.py`, replace:

```python
        selection_seed=cfg.seed,
        checkpoint_contract=runtime_contract,
```

with:

```python
        checkpoint_contract=runtime_contract,
```

- [x] **Step 6: Reword what still describes the quota**

In `src/naics_embedder/supervision/margins.py`, replace:

```python
Explicit exclusions are exempt: their exclusion is authoritative regardless of structure (the
quota may select a structurally close exclusion, as the contract requires), so callers apply
eligibility to ordinary candidates only.
```

with:

```python
An explicit exclusion is never a negative (Req 8), whatever its structure: callers remove
exclusions alongside this rule.
```

In `src/naics_embedder/utils/config.py`, replace:

```python
            'Stage-3 training rejects it; the one-slot exclusion quota owns representation.'
```

with:

```python
            'Stage-3 training rejects it: an explicit exclusion is never a negative.'
```

Replace:

```python
                    'the one-slot exclusion quota owns representation'
```

with:

```python
                    'an explicit exclusion is never a negative'
```

In `src/naics_embedder/text_model/dataloader/difficulty_sampler.py`, replace:

```python
    Returns source positions (never gathered candidates) in proposal order. Explicit exclusions are
    never proposed: the one-slot exclusion quota owns their representation. Bucket shortfalls
    cascade to the next bucket and finally to any remaining non-exclusion candidate.
```

with:

```python
    Returns source positions (never gathered candidates) in proposal order. Explicit exclusions are
    never proposed, since an exclusion is never a negative (Req 8). Bucket shortfalls cascade to
    the next bucket and finally to any remaining non-exclusion candidate.
```

In `src/naics_embedder/text_model/loss.py`, replace:

```python
        Explicit exclusions always stay in the contrastive denominator; only eligible pseudo-related
        candidates (never exclusions) are removed; invalid padding never contributes to the loss or
        its gradients. Anchors without any eligible negative are skipped.
```

with:

```python
        Only eligible pseudo-related candidates leave the contrastive denominator, never a flagged
        explicit exclusion, though training and validation now admit none (Req 8). Invalid padding
        never contributes to the loss or its gradients. Anchors without any eligible negative are
        skipped.
```

Run: `grep -rn 'exclusion quota\|EXCLUSION_QUOTA\|selection_seed\|exempt' src/naics_embedder`
Expected: two lines, both left for Stage 7: `supervision/schema.py`'s `EXCLUSION_QUOTA = 1` and
`text_model/mixins/logging.py`'s `quota_selections` counter.

- [x] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_negative_selection.py tests/unit/test_streaming_sampling.py tests/unit/test_hard_negative_mining.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_checkpoint_contract.py tests/unit/test_config.py tests/integration/test_stage3_training_step.py tests/integration/test_distributed_supervision.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1658 passed, 1 skipped`. Five rotation and quota tests go, and eight tests are new.

- [x] **Step 8: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/selection.py src/naics_embedder/supervision/schema.py src/naics_embedder/text_model/dataloader/streaming_dataset.py src/naics_embedder/text_model/mixins/curriculum.py src/naics_embedder/text_model/mixins/logging.py src/naics_embedder/text_model/mixins/validation.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py src/naics_embedder/supervision/margins.py src/naics_embedder/utils/config.py src/naics_embedder/text_model/dataloader/difficulty_sampler.py src/naics_embedder/text_model/loss.py tests/unit/test_negative_selection.py tests/unit/test_streaming_sampling.py tests/unit/test_hard_negative_mining.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/unit/test_checkpoint_contract.py tests/unit/test_config.py tests/integration/test_stage3_training_step.py tests/integration/test_distributed_supervision.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 9: Commit**

```bash
git add src/naics_embedder/supervision/selection.py \
  src/naics_embedder/supervision/schema.py \
  src/naics_embedder/text_model/dataloader/streaming_dataset.py \
  src/naics_embedder/text_model/mixins/curriculum.py \
  src/naics_embedder/text_model/mixins/logging.py \
  src/naics_embedder/text_model/mixins/validation.py \
  src/naics_embedder/text_model/naics_model.py \
  src/naics_embedder/cli/commands/training.py \
  src/naics_embedder/supervision/margins.py \
  src/naics_embedder/utils/config.py \
  src/naics_embedder/text_model/dataloader/difficulty_sampler.py \
  src/naics_embedder/text_model/loss.py \
  tests/unit/test_negative_selection.py \
  tests/unit/test_streaming_sampling.py \
  tests/unit/test_hard_negative_mining.py \
  tests/unit/test_naics_model.py \
  tests/unit/test_cli_training.py \
  tests/unit/test_checkpoint_contract.py \
  tests/unit/test_config.py \
  tests/integration/test_stage3_training_step.py \
  tests/integration/test_distributed_supervision.py
git commit -m "feat(supervision): runtime selection never picks an explicit exclusion"
```

### Task 9: Unary pairs leave positives and parent retrieval

Req 9: "The 522 unary pairs are five-digit industries whose only child is their six-digit code.
They leave positive supervision and parent-retrieval scoring, and all 2,125 codes stay in the
deliverable." Today every stage treats such a pair as an ordinary parent and child. This task flags
them in the pair facts and drops them in four places:

- the generator, which never makes one a positive;
- the positive sampler, text and graph alike, which never offers one in either direction;
- the validators, which refuse a flag that disagrees with the codes and a training positive that
  is a unary pair, at build and at load;
- parent retrieval (`compute_hierarchy_retrieval_metrics`, shared by the text and graph stages),
  which stops scoring them, as Stage 4's `metrics.diagnostics.parent_retrieval` already does.

The pair facts gain a column, so their schema moves to `pair-facts-v2`. Every code stays in the
codebook; only these pairs lose their role as positives.

**Files:**
- Modify: `src/naics_embedder/supervision/schema.py` (`PAIR_FACTS_SCHEMA_VERSION`)
- Modify: `src/naics_embedder/supervision/artifacts.py` (`unary_pair_expression`,
  `validate_unary_pairs`, `_directed_pair_facts`, `validate_training_pairs_members`' docstring,
  `_validate_training_chunk`, the load-time pair-facts check)
- Modify: `src/naics_embedder/data/supervision_bundle.py` (`build_pair_facts`,
  `PAIR_FACT_SCHEMA`, `validate_pair_facts`, `_validate_no_unary_positives`,
  `_checked_training_batches`, validation results)
- Modify: `src/naics_embedder/data/create_triplets.py` (module docstring, `_SHARED_PAIR_COLUMNS`,
  `_positive_pairs`)
- Modify: `src/naics_embedder/data/positive_sampling.py` (`enumerate_positives`)
- Modify: `src/naics_embedder/metrics/hierarchy_structure.py`
  (`compute_hierarchy_retrieval_metrics`)
- Test: `tests/fixtures/supervision.py`, `tests/unit/test_data_triplets.py`,
  `tests/unit/test_supervision_artifacts.py`, `tests/unit/test_positive_sampling.py`,
  `tests/unit/test_hierarchy_metrics.py`

**Interfaces:**
- Consumes: from Task 1, `naics_embedder.utils.naics_hierarchy.unary_pairs(codes: Iterable[str])
  -> List[Tuple[str, str]]`, the `(parent, child)` unary pairs sorted by parent.
- Produces:
  - Pair facts carry `unary_pair` (Boolean), true exactly on a five-digit `code_i` and its only
    six-digit child `code_j`. `PAIR_FACTS_SCHEMA_VERSION == 'pair-facts-v2'`.
  - In `naics_embedder.supervision.artifacts`: `unary_pair_expression(codes: Iterable[str]) ->
    pl.Expr` over canonical `code_i`/`code_j` columns, and `validate_unary_pairs(pair_facts:
    pl.DataFrame, codebook: pl.DataFrame) -> None`, which raises `ValueError` naming
    `unary_pair`.
  - The manifest's `validation_results` gain `unary_pairs_flagged` and `no_unary_positives`.
  - Every frame passed to `create_triplets.build_training_pairs` or
    `generate_supervision_bundle_from_frames` needs the `unary_pair` column.
  - `enumerate_positives` never returns a unary pair in either direction.

- [x] **Step 1: Write the failing tests**

In `tests/fixtures/supervision.py`, replace:

```python
            'is_explicit_exclusion': [
                False, True, False, False, False, True, False, False, False, False
            ],
        }
    )
```

with:

```python
            'is_explicit_exclusion': [
                False, True, False, False, False, True, False, False, False, False
            ],
            # No five-digit code, so no unary pair
            'unary_pair': [False] * 10,
        }
    )
```

In `tests/unit/test_data_triplets.py`, replace:

```python
            'is_explicit_exclusion': [False, True, False, False, False, False],
```

with:

```python
            'is_explicit_exclusion': [False, True, False, False, False, False],
            'unary_pair': [False] * 6,
```

Replace:

```python
            'is_explicit_exclusion': [False] * 6,
```

with:

```python
            'is_explicit_exclusion': [False] * 6,
            'unary_pair': [False] * 6,
```

Replace:

```python
                    'is_explicit_exclusion': (i, j) == (0, 4),
```

with:

```python
                    'is_explicit_exclusion': (i, j) == (0, 4),
                    'unary_pair': False,
```

Replace:

```python
@pytest.fixture
def cross_prefix_pair_facts() -> pl.DataFrame:
```

with:

```python
def test_a_unary_pair_is_never_a_positive(pair_facts_fixture):
    # Flag (0, 1): its two triples go, and the third stays
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    assert _triples(build_training_pairs(flagged)) == [(1, 2, 4)]

@pytest.fixture
def cross_prefix_pair_facts() -> pl.DataFrame:
```

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
from naics_embedder.supervision.artifacts import load_validated_bundle, sha256_file
```

with:

```python
from naics_embedder.supervision.artifacts import (
    load_validated_bundle,
    sha256_file,
    validate_training_pairs_members,
)
```

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Fail-closed bundle loading
# -------------------------------------------------------------------------------------------------
```

with:

```python
def test_the_unary_pairs_are_flagged_and_never_generated_positives(
    tmp_path, hierarchy_descriptions_parquet
):
    manifest_path = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    bundle = load_validated_bundle(manifest_path)
    facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
    pairs = pl.read_parquet(list(bundle.member_paths('training_pairs')))

    unary = facts.filter(pl.col('unary_pair')).select('code_i', 'code_j').rows()
    positives = set(pairs.select('anchor_code', 'positive_code').unique().rows())
    # Each five-digit code in the hierarchy has one six-digit child
    assert unary == [
        ('31111', '311111'),
        ('31121', '311211'),
        ('32111', '321111'),
        ('44111', '441111'),
    ]
    assert not positives & set(unary)
    assert bundle.manifest.artifacts['pair_facts'].schema_version == 'pair-facts-v2'
    assert bundle.manifest.validation_results['unary_pairs_flagged'] is True
    assert bundle.manifest.validation_results['no_unary_positives'] is True

def test_pair_facts_refuse_a_unary_flag_the_codes_do_not_support(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    # '111111' and '111112' are six-digit siblings, not a five-digit code and its only child
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='unary_pair is wrong on 1 pairs, e.g. 111111/111112'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='bundle-a',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=flagged,
        )

def test_training_pairs_refuse_a_unary_positive(tmp_path, pair_facts_fixture):
    # Rows generated while (0, 1) was unflagged keep it as a positive; the flagged facts refuse them
    path = tmp_path / 'training_pairs.parquet'
    build_training_pairs(pair_facts_fixture).write_parquet(path)
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='a training positive is a unary pair'):
        validate_training_pairs_members([path], flagged, n_codes=5)

# -------------------------------------------------------------------------------------------------
# Fail-closed bundle loading
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
def test_loader_rejects_a_rehashed_exclusion_negative(generated_bundle):
```

with:

```python
def test_loader_rejects_a_rehashed_unary_flag(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'pair_facts',
        lambda frame: frame.with_columns(
            unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
        ),
    )

    with pytest.raises(ValueError, match='pair_facts.*bundle-a.*unary_pair is wrong'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_exclusion_negative(generated_bundle):
```

In `tests/unit/test_positive_sampling.py`, append to the end of the file:

```python

@pytest.mark.unit
def test_enumerate_positives_drops_the_unary_pairs(
    descriptions_parquet, relations_parquet, sample_descriptions_df
):
    '''A five-digit code and its only six-digit child are never each other's positive (Req 9).'''

    codes = sample_descriptions_df.get_column('code').to_list()
    code_to_idx = {code: position for position, code in enumerate(codes)}

    result = enumerate_positives(descriptions_parquet, relations_parquet, code_to_idx=code_to_idx)
    pairs = set(result.select('anchor_code', 'positive_code').iter_rows())

    for parent, child in [('32111', '321111'), ('33111', '331111'), ('44111', '441111')]:
        assert (parent, child) not in pairs
        assert (child, parent) not in pairs
    # '31111' has two six-digit children, so both stay its positives
    assert {('31111', '311111'), ('31111', '311112')} <= pairs
```

In `tests/unit/test_hierarchy_metrics.py`, append to the end of the file:

```python

def test_parent_retrieval_skips_the_unary_pairs():
    # '31111' has one six-digit child, '311111': a unary pair (Req 9). That child's nearest code is
    # its parent's sibling '31112', which would score a miss; the other two children find their
    # parent, so retrieval without the unary pair is perfect.
    hierarchy = NaicsHierarchy([('3111', '31111'), ('3111', '31112'), ('31111', '311111')])
    codes = ['3111', '31111', '31112', '311111']
    distance_matrix = torch.tensor(
        [
            [0.0, 0.1, 0.1, 0.5],
            [0.1, 0.0, 0.3, 0.4],
            [0.1, 0.3, 0.0, 0.2],
            [0.5, 0.4, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_hierarchy_retrieval_metrics(
        distance_matrix, codes, hierarchy, parent_top_k=1, child_top_k=0
    )

    assert metrics['parent_retrieval@1'] == pytest.approx(1.0)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_positive_sampling.py tests/unit/test_hierarchy_metrics.py -q`
Expected: FAIL. The generator and the bundle ignore `unary_pair`, so the flagged triples stay, no
wrong flag is refused, the hierarchy bundle has no `unary_pair` column, the sampler still offers
the three unary pairs, and parent retrieval scores 2/3.

- [x] **Step 3: Flag and check the unary pairs in the pair facts**

In `src/naics_embedder/supervision/schema.py`, replace:

```python
PAIR_FACTS_SCHEMA_VERSION = 'pair-facts-v1'
```

with:

```python
# v2: the unary_pair flag (Req 9)
PAIR_FACTS_SCHEMA_VERSION = 'pair-facts-v2'
```

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
from naics_embedder.utils.naics_hierarchy import code_lineage, tree_distance_matrix
```

with:

```python
from naics_embedder.utils.naics_hierarchy import code_lineage, tree_distance_matrix, unary_pairs
```

Replace:

```python
INDEX_ROLES_ARTIFACT = 'index_roles'
INDEX_ROLE_COLUMNS = ('entry_id', 'code', 'text', 'role')
```

with:

```python
def unary_pair_expression(codes: Iterable[str]) -> pl.Expr:
    '''
    Whether a canonical pair is a unary pair among ``codes`` (Req 9): a five-digit code and its
    only six-digit child. Canonical orientation puts the five-digit code in ``code_i``.
    '''

    children = [child for _, child in unary_pairs(codes)]
    return pl.col('code_j').is_in(children) & pl.col('code_i').eq(pl.col('code_j').str.slice(0, 5))

def validate_unary_pairs(pair_facts: pl.DataFrame, codebook: pl.DataFrame) -> None:
    '''
    ``unary_pair`` must mark exactly the unary pairs among the codebook's codes (Req 9).

    Raises:
        ValueError: If the column is missing, or wrong or null on any pair.
    '''

    if 'unary_pair' not in pair_facts.columns:
        raise ValueError('pair facts lack the unary_pair column')
    expected = unary_pair_expression(codebook.get_column('code').to_list())
    wrong = pair_facts.filter(pl.col('unary_pair').ne_missing(expected))
    if wrong.height:
        code_i, code_j = wrong.select('code_i', 'code_j').row(0)
        raise ValueError(
            f'unary_pair is wrong on {wrong.height:,} pairs, e.g. {code_i}/{code_j}: a unary pair '
            'is a five-digit code and its only six-digit child'
        )

INDEX_ROLES_ARTIFACT = 'index_roles'
INDEX_ROLE_COLUMNS = ('entry_id', 'code', 'text', 'role')
```

Replace:

```python
    def check_pair_facts() -> None:
        validate_structural_pairs(pair_facts.select(STRUCTURAL_PAIR_COLUMNS), codebook)
        validate_exclusion_derivation(pair_facts)
```

with:

```python
    def check_pair_facts() -> None:
        validate_structural_pairs(pair_facts.select(STRUCTURAL_PAIR_COLUMNS), codebook)
        validate_exclusion_derivation(pair_facts)
        validate_unary_pairs(pair_facts, codebook)
```

In `src/naics_embedder/data/supervision_bundle.py`, replace:

```python
    codebook_fingerprint,
    sha256_file,
    validate_exclusion_derivation,
    validate_index_role_table,
    validate_matrix,
    validate_structural_pairs,
    write_versioned_dataset_batches,
    write_versioned_parquet,
)
```

with:

```python
    codebook_fingerprint,
    sha256_file,
    unary_pair_expression,
    validate_exclusion_derivation,
    validate_index_role_table,
    validate_matrix,
    validate_structural_pairs,
    validate_unary_pairs,
    write_versioned_dataset_batches,
    write_versioned_parquet,
)
```

Replace:

```python
    Returns:
        One row per unordered pair of distinct codes in canonical orientation, with untouched
        structural columns plus ``code_i_excludes_code_j``, ``code_j_excludes_code_i``, and
        ``is_explicit_exclusion``.
    '''
```

with:

```python
    Returns:
        One row per unordered pair of distinct codes in canonical orientation, with untouched
        structural columns plus ``code_i_excludes_code_j``, ``code_j_excludes_code_i``,
        ``is_explicit_exclusion`` and ``unary_pair``.
    '''
```

Replace:

```python
    validate_structural_pairs(structural, codebook)
    return attach_exclusion_provenance(structural, descriptions, codebook)
```

with:

```python
    validate_structural_pairs(structural, codebook)
    return attach_exclusion_provenance(structural, descriptions, codebook).with_columns(
        unary_pair=unary_pair_expression(codebook.get_column('code').to_list())
    )
```

Replace:

```python
    'is_explicit_exclusion': pl.Boolean,
}
```

with:

```python
    'is_explicit_exclusion': pl.Boolean,
    'unary_pair': pl.Boolean,
}
```

Replace:

```python
    if not published.select(flags).equals(pair_facts.select(flags)):
        raise ValueError('pair facts exclusion flags disagree with the published exclusions')
```

with:

```python
    if not published.select(flags).equals(pair_facts.select(flags)):
        raise ValueError('pair facts exclusion flags disagree with the published exclusions')
    validate_unary_pairs(pair_facts, codebook)
```

Replace:

```python
        'exclusion_derivation': True,
        'exclusions_match_descriptions': True,
    }
```

with:

```python
        'exclusion_derivation': True,
        'exclusions_match_descriptions': True,
        'unary_pairs_flagged': True,
    }
```

Replace:

```python
        if keys.join(pair_keys, on=['low', 'high'], how='anti').height:
            raise ValueError(f'training pair {other} identities do not join to pair facts')
```

with:

```python
        if keys.join(pair_keys, on=['low', 'high'], how='anti').height:
            raise ValueError(f'training pair {other} identities do not join to pair facts')

def _validate_no_unary_positives(batch: pl.DataFrame, unary_keys: pl.DataFrame) -> None:
    '''No generated positive may be a unary pair (Req 9).'''

    keys = batch.select(
        low=pl.min_horizontal('anchor_code_id', 'positive_code_id').cast(pl.Int32),
        high=pl.max_horizontal('anchor_code_id', 'positive_code_id').cast(pl.Int32),
    )
    if keys.join(unary_keys, on=['low', 'high'], how='semi').height:
        raise ValueError('a generated training positive is a unary pair')
```

Replace:

```python
        high=pl.max_horizontal('code_i_id', 'code_j_id'),
    )
    for batch in iter_training_pair_batches(
        pair_facts, cross_sector_cap=cross_sector_cap, cap_seed=cap_seed
    ):
        _validate_training_identity(batch, pair_keys)
```

with:

```python
        high=pl.max_horizontal('code_i_id', 'code_j_id'),
    )
    unary_keys = pair_keys.filter(pair_facts.get_column('unary_pair'))
    for batch in iter_training_pair_batches(
        pair_facts, cross_sector_cap=cross_sector_cap, cap_seed=cap_seed
    ):
        _validate_training_identity(batch, pair_keys)
        _validate_no_unary_positives(batch, unary_keys)
```

Replace:

```python
                'no_exclusion_negatives': True,
```

with:

```python
                'no_exclusion_negatives': True,
                'no_unary_positives': True,
```

- [x] **Step 4: Drop unary positives from generation and loading**

In `src/naics_embedder/data/create_triplets.py`, replace:

```python
Positive/negative combinatorics reproduce the legacy generator: a positive is a canonical,
within-sector, non-exclusion pair; a negative ``j`` for (anchor ``a``, positive ``p``) requires the
```

with:

```python
Positive/negative combinatorics reproduce the legacy generator: a positive is a canonical,
within-sector pair that is neither an exclusion nor a unary pair (Req 9: a five-digit code and its
only six-digit child); a negative ``j`` for (anchor ``a``, positive ``p``) requires the
```

Replace:

```python
    'structural_relation_name',
    'is_explicit_exclusion',
)
```

with:

```python
    'structural_relation_name',
    'is_explicit_exclusion',
    'unary_pair',
)
```

Replace:

```python
    '''Canonical, within-sector, non-exclusion pairs; reversed rows never become positives.'''

    return anchor_view.filter(
        ~pl.col('is_reversed'),
        pl.col('structural_distance').gt(0.0),
        pl.col('structural_relation_id').ne(CROSS_SECTOR_RELATION_ID),
        ~pl.col('is_explicit_exclusion'),
    ).select(
```

with:

```python
    '''
    Canonical, within-sector pairs that are neither exclusions nor unary pairs; reversed rows never
    become positives.
    '''

    return anchor_view.filter(
        ~pl.col('is_reversed'),
        pl.col('structural_distance').gt(0.0),
        pl.col('structural_relation_id').ne(CROSS_SECTOR_RELATION_ID),
        ~pl.col('is_explicit_exclusion'),
        ~pl.col('unary_pair'),
    ).select(
```

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
    Every identity must be a known code ID, no direct positive or negative may be an explicit
    exclusion of its anchor, exclusion and semantic columns must be internally consistent, and
    every anchor/positive and anchor/negative view must match the pair facts (structure and both
    exclusion directions). Members are checked in bounded chunks of files, which is exact because
    every row check is row-local and every uniqueness check is a join against the pair facts.
```

with:

```python
    Every identity must be a known code ID, no direct positive or negative may be an explicit
    exclusion of its anchor, no positive may be a unary pair, exclusion and semantic columns must
    be internally consistent, and every anchor/positive and anchor/negative view must match the
    pair facts (structure and both exclusion directions). Members are checked in bounded chunks
    of files, which is exact because every row check is row-local and every uniqueness check is a
    join against the pair facts.
```

Replace:

```python
                fact_anchor_excludes=pl.col('code_i_excludes_code_j'),
                fact_other_excludes=pl.col('code_j_excludes_code_i'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
            ),
```

with:

```python
                fact_anchor_excludes=pl.col('code_i_excludes_code_j'),
                fact_other_excludes=pl.col('code_j_excludes_code_i'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
                fact_unary=pl.col('unary_pair'),
            ),
```

Replace:

```python
                fact_anchor_excludes=pl.col('code_j_excludes_code_i'),
                fact_other_excludes=pl.col('code_i_excludes_code_j'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
            ),
```

with:

```python
                fact_anchor_excludes=pl.col('code_j_excludes_code_i'),
                fact_other_excludes=pl.col('code_i_excludes_code_j'),
                fact_distance=pl.col('structural_distance').cast(pl.Float32),
                fact_relation=pl.col('structural_relation_id').cast(pl.Int16),
                fact_unary=pl.col('unary_pair'),
            ),
```

Replace:

```python
    if positives.filter(pl.col('fact_distance').is_null()).height:
        raise ValueError('positive identities do not join to pair facts')
```

with:

```python
    if positives.filter(pl.col('fact_distance').is_null()).height:
        raise ValueError('positive identities do not join to pair facts')
    if positives.filter(pl.col('fact_unary')).height:
        raise ValueError('a training positive is a unary pair')
```

- [x] **Step 5: Keep unary pairs out of sampled positives and parent retrieval**

In `src/naics_embedder/data/positive_sampling.py`, replace:

```python
from naics_embedder.utils.utilities import get_indices_codes
```

with:

```python
from naics_embedder.utils.naics_hierarchy import unary_pairs
from naics_embedder.utils.utilities import get_indices_codes
```

Replace:

```python
    '''Enumerate all possible positives for each anchor across three strata.

    Strata:
```

with:

```python
    '''Enumerate all possible positives for each anchor across three strata.

    A unary pair (Req 9), a five-digit code and its only six-digit child, is a positive in
    neither direction.

    Strata:
```

Replace:

```python
    if codebook_supplied:
        unknown = positives.filter(
```

with:

```python
    unary = unary_pairs(anchors.get_column('anchor').to_list())
    unary_keys = pl.DataFrame(
        unary + [(child, parent) for parent, child in unary],
        schema={
            'anchor_code': pl.Utf8,
            'positive_code': pl.Utf8
        },
        orient='row',
    )
    kept = positives.join(unary_keys, on=['anchor_code', 'positive_code'], how='anti')
    if kept.height < positives.height:
        logger.info(f'Dropped {positives.height - kept.height:,} unary positive pairs')
    positives = kept
    if codebook_supplied:
        unknown = positives.filter(
```

In `src/naics_embedder/metrics/hierarchy_structure.py`, replace:

```python
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy
```

with:

```python
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy, unary_pairs
```

Replace:

```python
    # Parent retrieval (child -> parent).
    parent_pairs = [
        (code_to_idx[parent], code_to_idx[child]) for parent, child in hierarchy.parent_child_pairs
        if parent in code_to_idx and child in code_to_idx
    ]
```

with:

```python
    # Parent retrieval (child -> parent) skips the unary pairs (Req 9), as
    # metrics.diagnostics.parent_retrieval does.
    unary = set(unary_pairs(code for pair in hierarchy.parent_child_pairs for code in pair))
    parent_pairs = [
        (code_to_idx[parent], code_to_idx[child]) for parent, child in hierarchy.parent_child_pairs
        if parent in code_to_idx and child in code_to_idx and (parent, child) not in unary
    ]
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_positive_sampling.py tests/unit/test_hierarchy_metrics.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1665 passed, 1 skipped` (7 new tests).

- [x] **Step 7: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/data/create_triplets.py src/naics_embedder/data/positive_sampling.py src/naics_embedder/metrics/hierarchy_structure.py tests/fixtures/supervision.py tests/unit/test_data_triplets.py tests/unit/test_supervision_artifacts.py tests/unit/test_positive_sampling.py tests/unit/test_hierarchy_metrics.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 8: Commit**

```bash
git add src/naics_embedder/supervision/schema.py \
  src/naics_embedder/supervision/artifacts.py \
  src/naics_embedder/data/supervision_bundle.py \
  src/naics_embedder/data/create_triplets.py \
  src/naics_embedder/data/positive_sampling.py \
  src/naics_embedder/metrics/hierarchy_structure.py \
  tests/fixtures/supervision.py \
  tests/unit/test_data_triplets.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_positive_sampling.py \
  tests/unit/test_hierarchy_metrics.py
git commit -m "feat(supervision): unary pairs leave positives and parent retrieval"
```

### Task 10: Every bundle carries its index roles and redirection table

Stage 2 shipped the index roles as an optional bundle member. This contract makes that member
required, together with Task 3's redirection table. The build checks the table three ways:

- the table is well formed;
- the descriptions' exclusion channel is the one the table builds, so each cross-reference
  appears in it once (Req 8(a));
- the table names exactly the pair facts' directed exclusions.

The build's leakage check also reads the table's activity phrases, which Stage 7 trains on as
queries (Req 3). The loader requires both members and re-runs the table checks. It also refuses
a manifest that lacks any validation result a build records (`REQUIRED_VALIDATION_RESULTS`),
the three `index_roles_*` results included. That discharges plan 4's two deferred items: the
contract requires those results, and the member pins the entry text under an artifact hash.

The table's schema constants move into `supervision/artifacts.py`, beside `INDEX_ROLE_COLUMNS`,
so the loader never imports `data/`; `data/redirections.py` imports them from there.

The fixtures change shape once, here. Every five-code bundle comes from one `build_bundle`
factory, and every hierarchy bundle from one `hierarchy_manifest` fixture, both with roles and
redirections. `generated_bundle_with_roles` goes, since every bundle now carries roles.

**Files:**
- Modify: `src/naics_embedder/supervision/artifacts.py` (required members and results, the
  redirection constants and validators, `_validate_relations`, `load_validated_bundle`)
- Modify: `src/naics_embedder/supervision/schema.py` (`REDIRECTIONS_SCHEMA_VERSION`)
- Modify: `src/naics_embedder/data/redirections.py` (imports its constants)
- Modify: `src/naics_embedder/data/supervision_bundle.py` (the members, their checks, both
  builders)
- Modify: `src/naics_embedder/utils/config.py` (`SupervisionBuildConfig`),
  `conf/data/supervision.yaml`
- Modify: `src/naics_embedder/panels/outcome.py` (`OutcomePanel.from_bundle`'s docstring)
- Test: `tests/fixtures/supervision.py`, `tests/unit/test_supervision_artifacts.py`,
  `tests/unit/test_outcome_panel.py`, `tests/unit/test_graph_preprocessing.py`,
  `tests/unit/test_streaming_sampling.py`, `tests/unit/test_hard_negative_mining.py`,
  `tests/unit/test_utils_validation.py`, `tests/unit/test_datamodule.py`,
  `tests/unit/test_structural_margins.py`, `tests/unit/test_config.py`,
  `tests/integration/test_stage3_training_step.py`,
  `tests/integration/test_distributed_supervision.py`

**Interfaces:**
- Consumes:
  - Task 3: `data/redirections.exclusion_channel(redirections)`, which returns `code`,
    `excluded` and `excluded_codes`; the descriptions' exclusion channel, built from it; and
    `verify_role_leakage(descriptions, role_rows, min_jaccard=..., *, extra_texts=())`.
  - Tasks 6, 7 and 9: the validation results a build records.
- Produces:
  - In `supervision/artifacts.py`:
    - `REDIRECTIONS_ARTIFACT = 'redirections'`, and `CROSS_REFERENCE_SOURCE`,
      `DESCRIPTION_SOURCE` and `REDIRECTIONS_SCHEMA`, moved from `data/redirections.py`, which
      re-imports them;
    - `REQUIRED_ARTIFACTS` gains `'index_roles'` and `'redirections'`;
    - `REQUIRED_VALIDATION_RESULTS`, the 27 checks a build records;
    - `validate_redirection_table(redirections, codes) -> None` and
      `validate_redirection_exclusions(redirections, pair_facts) -> None`, each raising
      `ValueError`.
  - `supervision.schema.REDIRECTIONS_SCHEMA_VERSION = 'redirections-v1'`.
  - `generate_supervision_bundle_from_frames` takes `index_roles: pl.DataFrame` and
    `redirections: pl.DataFrame`, both required keyword arguments.
  - `SupervisionBuildConfig.index_roles_parquet: str = './data/naics_index_roles.parquet'` and
    `redirections_parquet: str = './data/naics_redirections.parquet'`.
  - The manifest's `validation_results` gain `redirections_well_formed`,
    `redirections_match_exclusion_channel` and `redirections_match_pair_facts`.
  - Fixtures:
    - `build_bundle(**overrides) -> Path` builds the five-code bundle `bundle-a` under
      `tmp_path`, with any input overridden;
    - `redirections_fixture` and `hierarchy_redirections` hold the redirection tables;
    - `hierarchy_manifest` is the production-path hierarchy bundle, with an empty role table.
    The hierarchy descriptions gain `description`, `examples` and `excluded` columns.

- [x] **Step 1: Share the bundle inputs in the fixtures**

In `tests/fixtures/supervision.py`, replace:

```python
import polars as pl
import pytest
import torch

from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.candidates import NegativeCandidateBatch
```

with:

```python
from pathlib import Path

import polars as pl
import pytest
import torch

from naics_embedder.data.redirections import REDIRECTIONS_SCHEMA
from naics_embedder.data.supervision_bundle import (
    generate_supervision_bundle,
    generate_supervision_bundle_from_frames,
)
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.candidates import NegativeCandidateBatch
from naics_embedder.utils.config import SupervisionBuildConfig
```

Replace:

```python
@pytest.fixture
def generated_bundle(tmp_path, descriptions_fixture, pair_facts_fixture):
    return generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )

@pytest.fixture
def validated_bundle(generated_bundle):
    return load_validated_bundle(
        generated_bundle,
        expected_contract='stage3-supervision-v2',
    )

# -------------------------------------------------------------------------------------------------
# Index-entry roles: the optional bundle member
# -------------------------------------------------------------------------------------------------
```

with:

```python
# -------------------------------------------------------------------------------------------------
# The five-code bundle, with the index-roles and redirections members every bundle carries
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
@pytest.fixture
def text_descriptions_fixture(descriptions_fixture) -> pl.DataFrame:
    '''The five-code descriptions with text channels; examples hold examples-role entries only.'''

    examples = {'111111': 'Soybean farming', '111112': 'Canola farming', '222222': 'Coal mining'}
    return descriptions_fixture.with_columns(
        title=pl.concat_str(pl.lit('Industry '), pl.col('code')),
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.lit(None, pl.Utf8),
    )

@pytest.fixture
def generated_bundle_with_roles(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    return generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-roles',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=text_descriptions_fixture,
        pair_facts=pair_facts_fixture,
        index_roles=index_roles_fixture,
    )
```

with:

```python
# One cross-reference row per exclusion of the pair facts: '111111' sends peanut growing to
# '111113', and '222222' sends canola crushing to '111112'
REDIRECTION_ROWS = [
    (
        0,
        'cross_reference',
        '111111',
        'Growing peanuts--are classified in Industry 111113.',
        'Growing peanuts',
        ['111113'],
        [],
        False,
    ),
    (
        1,
        'cross_reference',
        '222222',
        'Canola crushing--are classified in Industry 111112.',
        'Canola crushing',
        ['111112'],
        [],
        False,
    ),
]

@pytest.fixture
def redirections_fixture() -> pl.DataFrame:
    return pl.DataFrame(REDIRECTION_ROWS, schema=REDIRECTIONS_SCHEMA, orient='row')

@pytest.fixture
def text_descriptions_fixture(descriptions_fixture) -> pl.DataFrame:
    '''
    The five-code descriptions with text channels.

    Examples hold examples-role entries only, and each exclusion channel is its code's one
    redirection row.
    '''

    examples = {'111111': 'Soybean farming', '111112': 'Canola farming', '222222': 'Coal mining'}
    excluded = {row[2]: row[3] for row in REDIRECTION_ROWS}
    return descriptions_fixture.with_columns(
        title=pl.concat_str(pl.lit('Industry '), pl.col('code')),
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.col('code').replace_strict(excluded, default=None),
    )

@pytest.fixture
def build_bundle(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture,
    redirections_fixture
):
    '''
    Build the five-code bundle with ``generate_supervision_bundle_from_frames``.

    The bundle is ``bundle-a`` under ``tmp_path``; keyword arguments override single inputs.
    '''

    def build(**overrides) -> Path:
        inputs = {
            'output_root': tmp_path,
            'bundle_id': 'bundle-a',
            'generator_revision': 'revision-a',
            'naics_vintage': 2022,
            'descriptions': text_descriptions_fixture,
            'pair_facts': pair_facts_fixture,
            'index_roles': index_roles_fixture,
            'redirections': redirections_fixture,
        }
        return generate_supervision_bundle_from_frames(**{**inputs, **overrides})

    return build

@pytest.fixture
def generated_bundle(build_bundle):
    return build_bundle()

@pytest.fixture
def validated_bundle(generated_bundle):
    return load_validated_bundle(
        generated_bundle,
        expected_contract='stage3-supervision-v2',
    )
```

Replace:

```python
@pytest.fixture
def hierarchy_descriptions() -> pl.DataFrame:
    excluded_codes = {
        '311111': ['321111'],
        '441111': ['311211'],
    }
    return pl.DataFrame(
        {
            'index': list(range(len(HIERARCHY_CODES))),
            'level': [len(code) for code in HIERARCHY_CODES],
            'code': list(HIERARCHY_CODES),
            'title': [f'Industry {code}' for code in HIERARCHY_CODES],
            'excluded_codes': [excluded_codes.get(code) for code in HIERARCHY_CODES],
        },
        schema_overrides={
            'index': pl.UInt32,
            'level': pl.UInt8,
            'excluded_codes': pl.List(pl.Utf8),
        },
    )
```

with:

```python
# '311111' sends sawmilling to '321111' in a cross-reference row, and an "Excluded" paragraph of
# '441111' names '311211': the hierarchy's two exclusions
HIERARCHY_REDIRECTION_ROWS = [
    (
        0,
        'cross_reference',
        '311111',
        'Sawmilling--are classified in Industry 321111.',
        'Sawmilling',
        ['321111'],
        [],
        False,
    ),
    (
        1,
        'description',
        '441111',
        'Flour milling is classified in Industry 311211.',
        None,
        ['311211'],
        [],
        False,
    ),
]

@pytest.fixture
def hierarchy_descriptions() -> pl.DataFrame:
    excluded_codes = {
        '311111': ['321111'],
        '441111': ['311211'],
    }
    excluded = {row[2]: row[3] for row in HIERARCHY_REDIRECTION_ROWS}
    return pl.DataFrame(
        {
            'index': list(range(len(HIERARCHY_CODES))),
            'level': [len(code) for code in HIERARCHY_CODES],
            'code': list(HIERARCHY_CODES),
            'title': [f'Industry {code}' for code in HIERARCHY_CODES],
            'description': ['This industry comprises establishments.'] * len(HIERARCHY_CODES),
            'examples': [None] * len(HIERARCHY_CODES),
            'excluded': [excluded.get(code) for code in HIERARCHY_CODES],
            'excluded_codes': [excluded_codes.get(code) for code in HIERARCHY_CODES],
        },
        schema_overrides={
            'index': pl.UInt32,
            'level': pl.UInt8,
            'examples': pl.Utf8,
            'excluded_codes': pl.List(pl.Utf8),
        },
    )
```

Append to the end of `tests/fixtures/supervision.py`:

```python

@pytest.fixture
def hierarchy_redirections() -> pl.DataFrame:
    return pl.DataFrame(HIERARCHY_REDIRECTION_ROWS, schema=REDIRECTIONS_SCHEMA, orient='row')

@pytest.fixture
def hierarchy_manifest(tmp_path, hierarchy_descriptions_parquet, hierarchy_redirections) -> Path:
    '''
    The manifest of the bundle that the production path builds from the hierarchy.

    The hierarchy has no index entries, so its role table is empty.
    '''

    roles_path = tmp_path / 'hierarchy_index_roles.parquet'
    redirections_path = tmp_path / 'hierarchy_redirections.parquet'
    pl.DataFrame(schema=INDEX_ROLE_SCHEMA).write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    return generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            index_roles_parquet=str(roles_path),
            redirections_parquet=str(redirections_path),
            output_root=str(tmp_path / 'bundles'),
        )
    )
```

Every test that built the hierarchy bundle itself now takes `hierarchy_manifest`.

In `tests/unit/test_streaming_sampling.py`, replace:

```python
@pytest.fixture
def hierarchy_index(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    return SupervisionIndex.from_bundle(load_validated_bundle(manifest))
```

with:

```python
@pytest.fixture
def hierarchy_index(hierarchy_manifest):
    from naics_embedder.supervision.artifacts import load_validated_bundle

    return SupervisionIndex.from_bundle(load_validated_bundle(hierarchy_manifest))
```

In `tests/unit/test_hard_negative_mining.py`, replace:

```python
@pytest.fixture
def hierarchy_index(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    return SupervisionIndex.from_bundle(load_validated_bundle(manifest))
```

with:

```python
@pytest.fixture
def hierarchy_index(hierarchy_manifest):
    from naics_embedder.supervision.artifacts import load_validated_bundle

    return SupervisionIndex.from_bundle(load_validated_bundle(hierarchy_manifest))
```

In `tests/unit/test_utils_validation.py`, replace:

```python
from naics_embedder.utils.config import Config, SupervisionBuildConfig, TokenizationConfig
```

with:

```python
from naics_embedder.utils.config import Config, TokenizationConfig
```

Replace:

```python
@pytest.fixture
def production_bundle(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle

    return generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
```

with:

```python
@pytest.fixture
def production_bundle(hierarchy_manifest):
    '''The hierarchy's bundle, built by the production path.'''

    return hierarchy_manifest
```

In `tests/unit/test_datamodule.py`, replace:

```python
@pytest.fixture
def hierarchy_bundle(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.supervision.index import SupervisionIndex
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    bundle = load_validated_bundle(manifest)
    return bundle, SupervisionIndex.from_bundle(bundle)
```

with:

```python
@pytest.fixture
def hierarchy_bundle(hierarchy_manifest):
    from naics_embedder.supervision.artifacts import load_validated_bundle
    from naics_embedder.supervision.index import SupervisionIndex

    bundle = load_validated_bundle(hierarchy_manifest)
    return bundle, SupervisionIndex.from_bundle(bundle)
```

In `tests/unit/test_structural_margins.py`, replace:

```python
from naics_embedder.data.create_triplets import _structural_margins
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.margins import structural_margins, structurally_eligible
from naics_embedder.utils.config import SupervisionBuildConfig
```

with:

```python
from naics_embedder.data.create_triplets import _structural_margins
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.supervision.index import SupervisionIndex
from naics_embedder.supervision.margins import structural_margins, structurally_eligible
```

Replace:

```python
def test_runtime_rule_equals_the_generator_rule_over_every_hierarchy_triple(
    tmp_path, hierarchy_descriptions_parquet
):
    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    index = SupervisionIndex.from_bundle(load_validated_bundle(manifest))
```

with:

```python
def test_runtime_rule_equals_the_generator_rule_over_every_hierarchy_triple(hierarchy_manifest):
    index = SupervisionIndex.from_bundle(load_validated_bundle(hierarchy_manifest))
```

In `tests/integration/test_stage3_training_step.py`, replace:

```python
@pytest.fixture
def hierarchy_model(monkeypatch, tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)
```

with:

```python
@pytest.fixture
def hierarchy_model(monkeypatch, hierarchy_manifest):
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)
```

Replace:

```python
        supervision_manifest_path=str(manifest),
```

with:

```python
        supervision_manifest_path=str(hierarchy_manifest),
```

In `tests/integration/test_distributed_supervision.py`, replace:

```python
def test_two_rank_selection_mines_the_global_pool_under_local_eligibility(
    tmp_path, hierarchy_descriptions_parquet
):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    manifest = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(_selection_worker, args=(2, init_file, str(manifest), queue), nprocs=2, join=True)
```

with:

```python
def test_two_rank_selection_mines_the_global_pool_under_local_eligibility(
    tmp_path, hierarchy_manifest
):
    init_file = tmp_path / 'gloo-init'
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(
        _selection_worker,
        args=(2, init_file, str(hierarchy_manifest), queue),
        nprocs=2,
        join=True,
    )
```

In `tests/unit/test_graph_preprocessing.py`, replace:

```python
import torch

from naics_embedder.data.supervision_bundle import generate_supervision_bundle_from_frames
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
```

with:

```python
import torch

from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
```

Replace:

```python
def test_graph_preprocessing_resolves_one_bundle_and_rejects_mixed_paths(
    tmp_path,
    generated_bundle,
    descriptions_fixture,
    pair_facts_fixture,
):
    first = resolve_graph_supervision_paths(generated_bundle)
    other_manifest = generate_supervision_bundle_from_frames(
        output_root=tmp_path / 'other',
        bundle_id='bundle-b',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
    other_bundle = load_validated_bundle(other_manifest)
```

with:

```python
def test_graph_preprocessing_resolves_one_bundle_and_rejects_mixed_paths(
    tmp_path, generated_bundle, build_bundle
):
    first = resolve_graph_supervision_paths(generated_bundle)
    other_manifest = build_bundle(output_root=tmp_path / 'other', bundle_id='bundle-b')
    other_bundle = load_validated_bundle(other_manifest)
```

The shared `hierarchy_manifest` replaces this file's own. Replace:

```python
@pytest.fixture
def hierarchy_manifest(tmp_path, hierarchy_descriptions_parquet):
    from naics_embedder.data.supervision_bundle import generate_supervision_bundle
    from naics_embedder.utils.config import SupervisionBuildConfig

    return generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )

def test_graph_config_takes_every_structural_input_from_its_bundle(generated_bundle):
```

with:

```python
def test_graph_config_takes_every_structural_input_from_its_bundle(generated_bundle):
```

In `tests/unit/test_outcome_panel.py`, the five-code bundle carries the roles now, and
`test_from_bundle_needs_the_member` goes: the loader refuses a bundle without the member
(`test_loader_requires_both_members`, Step 2), so `OutcomePanel.from_bundle` never sees one.
Replace:

```python
def test_from_bundle_reads_the_member_and_the_codebook(generated_bundle_with_roles, tmp_path):
    bundle = load_validated_bundle(generated_bundle_with_roles)

    panel = OutcomePanel.from_bundle(bundle, tmp_path / 'log.jsonl')

    assert panel.candidates == ('111111', '111112', '111113', '222222', '333333')
    assert panel.entryless_candidates == ('111113', '333333')
    assert panel.validation_queries('check')['entry_id'].to_list() == [1]

def test_from_bundle_needs_the_member(validated_bundle, tmp_path):
    with pytest.raises(ValueError, match='index_roles'):
        OutcomePanel.from_bundle(validated_bundle, tmp_path / 'log.jsonl')
```

with:

```python
def test_from_bundle_reads_the_member_and_the_codebook(generated_bundle, tmp_path):
    bundle = load_validated_bundle(generated_bundle)

    panel = OutcomePanel.from_bundle(bundle, tmp_path / 'log.jsonl')

    assert panel.candidates == ('111111', '111112', '111113', '222222', '333333')
    assert panel.entryless_candidates == ('111113', '333333')
    assert panel.validation_queries('check')['entry_id'].to_list() == [1]
```

- [x] **Step 2: Write the failing tests**

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
    generate_supervision_bundle,
    generate_supervision_bundle_from_frames,
    relation_matrix_from_pair_facts,
)
```

with:

```python
    generate_supervision_bundle,
    relation_matrix_from_pair_facts,
)
```

Replace:

```python
from naics_embedder.supervision.artifacts import (
    load_validated_bundle,
    sha256_file,
    validate_training_pairs_members,
)
```

with:

```python
from naics_embedder.panels.index_roles import verify_role_leakage
from naics_embedder.supervision.artifacts import (
    REDIRECTIONS_SCHEMA,
    REQUIRED_VALIDATION_RESULTS,
    load_validated_bundle,
    sha256_file,
    validate_redirection_table,
    validate_training_pairs_members,
)
```

Replace:

```python
def test_bundle_writes_manifest_last_with_matching_parquet_metadata(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifest_path = generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
```

with:

```python
def test_bundle_writes_manifest_last_with_matching_parquet_metadata(build_bundle):
    manifest_path = build_bundle()
```

Replace:

```python
def test_bundle_never_overwrites_an_existing_generation(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    kwargs = {
        'output_root': tmp_path,
        'bundle_id': 'bundle-a',
        'generator_revision': 'revision-a',
        'naics_vintage': 2022,
        'descriptions': descriptions_fixture,
        'pair_facts': pair_facts_fixture,
    }
    generate_supervision_bundle_from_frames(**kwargs)

    with pytest.raises(FileExistsError, match='bundle-a'):
        generate_supervision_bundle_from_frames(**kwargs)

def test_failed_validation_publishes_no_manifest(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    inconsistent = pair_facts_fixture.with_columns(is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='broken',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=inconsistent,
        )

    assert not (tmp_path / 'broken' / 'manifest.json').exists()

def test_two_generated_bundles_have_equal_logical_frames_but_distinct_ids(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifests = [
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id=bundle_id,
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=pair_facts_fixture,
        ) for bundle_id in ('bundle-a', 'bundle-b')
    ]
```

with:

```python
def test_bundle_never_overwrites_an_existing_generation(build_bundle):
    build_bundle()

    with pytest.raises(FileExistsError, match='bundle-a'):
        build_bundle()

def test_failed_validation_publishes_no_manifest(tmp_path, build_bundle, pair_facts_fixture):
    inconsistent = pair_facts_fixture.with_columns(is_explicit_exclusion=pl.lit(False))

    with pytest.raises(ValueError, match='exclusion derivation'):
        build_bundle(bundle_id='broken', pair_facts=inconsistent)

    assert not (tmp_path / 'broken' / 'manifest.json').exists()

def test_two_generated_bundles_have_equal_logical_frames_but_distinct_ids(build_bundle):
    manifests = [build_bundle(bundle_id=bundle_id) for bundle_id in ('bundle-a', 'bundle-b')]
```

Replace:

```python
def test_bundle_records_every_artifact_member_with_hash_and_contract_metadata(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    manifest_path = generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-a',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=descriptions_fixture,
        pair_facts=pair_facts_fixture,
    )
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest['artifacts']

    assert set(artifacts) == {
        'codebook',
        'pair_facts',
        'distances',
        'distance_matrix',
        'relations',
        'relation_matrix',
        'training_pairs',
        'difficulty_thresholds',
    }
```

with:

```python
def test_bundle_records_every_artifact_member_with_hash_and_contract_metadata(
    tmp_path, build_bundle
):
    manifest_path = build_bundle()
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest['artifacts']

    assert set(artifacts) == {
        'codebook',
        'pair_facts',
        'distances',
        'distance_matrix',
        'relations',
        'relation_matrix',
        'training_pairs',
        'difficulty_thresholds',
        'index_roles',
        'redirections',
    }
```

Replace:

```python
def test_failed_generation_leaves_no_staging_directory(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    with pytest.raises(ValueError):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='broken',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=pair_facts_fixture.with_columns(structural_distance=pl.lit(0.0)),
        )

    assert list(tmp_path.iterdir()) == []

def test_production_bundle_uses_a_uuid_and_the_descriptions_file_hash(
    tmp_path, hierarchy_descriptions_parquet
):
    cfg = SupervisionBuildConfig(
        descriptions_parquet=hierarchy_descriptions_parquet,
        output_root=str(tmp_path / 'bundles'),
    )

    manifest_path = generate_supervision_bundle(cfg)
    manifest = json.loads(manifest_path.read_text())

    assert str(uuid.UUID(manifest['bundle_id'])) == manifest['bundle_id']
    assert manifest_path.parent.name == manifest['bundle_id']
```

with:

```python
def test_failed_generation_leaves_no_staging_directory(tmp_path, build_bundle, pair_facts_fixture):
    with pytest.raises(ValueError):
        build_bundle(
            bundle_id='broken',
            pair_facts=pair_facts_fixture.with_columns(structural_distance=pl.lit(0.0)),
        )

    assert list(tmp_path.iterdir()) == []

def test_production_bundle_uses_a_uuid_and_the_descriptions_file_hash(
    hierarchy_manifest, hierarchy_descriptions_parquet
):
    manifest = json.loads(hierarchy_manifest.read_text())

    assert str(uuid.UUID(manifest['bundle_id'])) == manifest['bundle_id']
    assert hierarchy_manifest.parent.name == manifest['bundle_id']
```

Replace:

```python
def test_the_unary_pairs_are_flagged_and_never_generated_positives(
    tmp_path, hierarchy_descriptions_parquet
):
    manifest_path = generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            output_root=str(tmp_path / 'bundles'),
        )
    )
    bundle = load_validated_bundle(manifest_path)
```

with:

```python
def test_the_unary_pairs_are_flagged_and_never_generated_positives(hierarchy_manifest):
    bundle = load_validated_bundle(hierarchy_manifest)
```

Replace:

```python
def test_pair_facts_refuse_a_unary_flag_the_codes_do_not_support(
    tmp_path, descriptions_fixture, pair_facts_fixture
):
    # '111111' and '111112' are six-digit siblings, not a five-digit code and its only child
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='unary_pair is wrong on 1 pairs, e.g. 111111/111112'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='bundle-a',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=descriptions_fixture,
            pair_facts=flagged,
        )
```

with:

```python
def test_pair_facts_refuse_a_unary_flag_the_codes_do_not_support(build_bundle, pair_facts_fixture):
    # '111111' and '111112' are six-digit siblings, not a five-digit code and its only child
    flagged = pair_facts_fixture.with_columns(
        unary_pair=pl.col('code_i_id').eq(0) & pl.col('code_j_id').eq(1)
    )

    with pytest.raises(ValueError, match='unary_pair is wrong on 1 pairs, e.g. 111111/111112'):
        build_bundle(pair_facts=flagged)
```

The rest of the file is the members' section, rewritten whole.

Replace the whole of the file from its `# The optional index-roles member` line through its last
line, `    assert manifest['generation_parameters']['index_roles_parquet'] == str(roles_path.resolve())`,
with:

```python
# The index-roles and redirections members
# -------------------------------------------------------------------------------------------------

def test_bundle_carries_the_index_roles_after_checking_them(generated_bundle, index_roles_fixture):
    manifest = json.loads(generated_bundle.read_text())
    record = manifest['artifacts']['index_roles']

    assert record['path'] == 'naics_index_roles.parquet'
    assert record['schema_version'] == 'index-roles-v1'
    assert record['row_count'] == 6
    for check in (
        'index_roles_one_role_per_entry',
        'index_roles_examples_channel',
        'index_roles_no_leakage',
    ):
        assert manifest['validation_results'][check] is True
    bundle = load_validated_bundle(generated_bundle)
    assert pl.read_parquet(bundle.artifact_path('index_roles')).equals(index_roles_fixture)

def test_bundle_carries_the_redirection_table_after_checking_it(
    generated_bundle, redirections_fixture
):
    manifest = json.loads(generated_bundle.read_text())
    record = manifest['artifacts']['redirections']

    assert record['path'] == 'naics_redirections.parquet'
    assert record['schema_version'] == 'redirections-v1'
    assert record['row_count'] == 2
    for check in (
        'redirections_well_formed',
        'redirections_match_exclusion_channel',
        'redirections_match_pair_facts',
    ):
        assert manifest['validation_results'][check] is True
    bundle = load_validated_bundle(generated_bundle)
    assert pl.read_parquet(bundle.artifact_path('redirections')).equals(redirections_fixture)

def test_bundle_refuses_an_examples_channel_holding_queries(
    tmp_path, build_bundle, text_descriptions_fixture
):
    stale = text_descriptions_fixture.with_columns(
        examples=pl.when(pl.col('code') == '111111').then(
            pl.lit('Soybean farming; Edamame farming')
        ).otherwise('examples')
    )

    with pytest.raises(ValueError, match='examples channel other than'):
        build_bundle(bundle_id='stale', descriptions=stale)
    assert list(tmp_path.iterdir()) == []

def test_bundle_refuses_a_held_out_query_matching_training_text(build_bundle, index_roles_fixture):
    # Entry 1 (validation) becomes another code's title
    leaky = index_roles_fixture.with_columns(
        text=pl.when(pl.col('entry_id') == 1).then(pl.lit('Industry 222222')).otherwise('text')
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        build_bundle(bundle_id='leaky', index_roles=leaky)

def test_the_leakage_check_reads_the_activity_phrases(monkeypatch, build_bundle):
    # Stage 7 trains on the activity phrases as queries, so no held-out query may match one
    seen = []

    def spy(descriptions, role_rows, *args, extra_texts=(), **kwargs):
        seen.append(list(extra_texts))
        return verify_role_leakage(
            descriptions, role_rows, *args, extra_texts=extra_texts, **kwargs
        )

    monkeypatch.setattr('naics_embedder.data.supervision_bundle.verify_role_leakage', spy)
    build_bundle()

    assert seen == [['Growing peanuts', 'Canola crushing']]

def test_bundle_refuses_an_exclusion_channel_the_table_does_not_build(
    tmp_path, build_bundle, text_descriptions_fixture
):
    # Code 111111's channel repeats its one cross-reference, which must appear once
    doubled = text_descriptions_fixture.with_columns(
        excluded=pl.when(pl.col('code') == '111111').then(
            pl.concat_str('excluded', pl.lit(' '), 'excluded')
        ).otherwise('excluded')
    )

    with pytest.raises(ValueError, match='exclusion channel other than the redirection table'):
        build_bundle(bundle_id='doubled', descriptions=doubled)
    assert list(tmp_path.iterdir()) == []

# A well-formed table over the codes '11111', '111111', '111112' and '222222': a cross-reference,
# a cross-reference naming its code's parent, and a withheld "Excluded" paragraph
WELL_FORMED_REDIRECTIONS = [
    (
        0,
        'cross_reference',
        '111111',
        'Growing peanuts--are classified in Industry 111112.',
        'Growing peanuts',
        ['111112'],
        [],
        False,
    ),
    (
        1,
        'cross_reference',
        '111112',
        'Mixed farming--are classified in Industry 11111.',
        'Mixed farming',
        ['11111'],
        ['11111'],
        False,
    ),
    (
        2,
        'description',
        '222222',
        'Farm supplies are classified in Industry 111111.',
        None,
        ['111111'],
        [],
        True,
    ),
]
REDIRECTION_CODES = ['11111', '111111', '111112', '222222']

def _redirections(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=REDIRECTIONS_SCHEMA, orient='row')

def test_redirection_table_accepts_a_well_formed_table():
    validate_redirection_table(_redirections(WELL_FORMED_REDIRECTIONS), REDIRECTION_CODES)

@pytest.mark.parametrize(
    ('row', 'column', 'value', 'message'),
    [
        (0, 'reference_id', 5, 'reference IDs must run from zero'),
        (0, 'source', 'index', 'unknown redirection sources'),
        (0, 'named_codes', ['999999'], 'names a code outside the codebook'),
        (0, 'named_codes', ['111111'], 'names its own code'),
        (1, 'lineal_codes', [], 'lineal_codes must be'),
        (2, 'activity', 'Farm supplies', 'activity phrase'),
    ],
)
def test_redirection_table_refuses_a_malformed_row(row, column, value, message):
    rows = [list(values) for values in WELL_FORMED_REDIRECTIONS]
    rows[row][list(REDIRECTIONS_SCHEMA).index(column)] = value

    with pytest.raises(ValueError, match=message):
        validate_redirection_table(_redirections(rows), REDIRECTION_CODES)

def test_redirection_table_refuses_other_columns():
    table = _redirections(WELL_FORMED_REDIRECTIONS).drop('withheld')

    with pytest.raises(ValueError, match='redirection columns must be'):
        validate_redirection_table(table, REDIRECTION_CODES)

def test_loader_rejects_an_index_entry_with_two_roles(generated_bundle):
    _rewrite_member(
        generated_bundle,
        'index_roles',
        lambda frame: frame.with_columns(entry_id=pl.lit(0, pl.Int64)),
    )

    with pytest.raises(ValueError, match='index_roles .*more than one role'):
        load_validated_bundle(generated_bundle)

def test_loader_rejects_a_rehashed_redirection_naming_another_code(generated_bundle):
    # Row 1 now sends canola crushing to '333333', which '222222' does not exclude
    _rewrite_member(
        generated_bundle,
        'redirections',
        lambda frame: frame.with_columns(
            named_codes=pl.Series([['111113'], ['333333']], dtype=pl.List(pl.Utf8))
        ),
    )

    with pytest.raises(
        ValueError, match='redirections .*bundle-a.*1 named pairs are not exclusions'
    ):
        load_validated_bundle(generated_bundle)

@pytest.mark.parametrize('member', ['index_roles', 'redirections'])
def test_loader_requires_both_members(generated_bundle, member):
    manifest = json.loads(generated_bundle.read_text())
    del manifest['artifacts'][member]
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(ValueError, match=rf"lacks required artifacts: \['{member}'\]"):
        load_validated_bundle(generated_bundle)

def test_a_build_records_exactly_the_required_validation_results(generated_bundle):
    recorded = json.loads(generated_bundle.read_text())['validation_results']

    assert set(recorded) == set(REQUIRED_VALIDATION_RESULTS)
    # The member checks are among them, Req 3's leakage check included
    assert {
        'index_roles_one_role_per_entry',
        'index_roles_examples_channel',
        'index_roles_no_leakage',
        'redirections_well_formed',
        'redirections_match_exclusion_channel',
        'redirections_match_pair_facts',
    } <= set(REQUIRED_VALIDATION_RESULTS)

def test_loader_rejects_a_manifest_missing_a_required_validation_result(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())
    del manifest['validation_results']['index_roles_no_leakage']
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(
        ValueError, match=r"lacks required validation results: \['index_roles_no_leakage'\]"
    ):
        load_validated_bundle(generated_bundle)

def test_production_bundle_takes_its_members_from_its_config(
    tmp_path, hierarchy_descriptions, hierarchy_redirections
):
    roles = pl.DataFrame(
        [
            (0, '311111', 'Dog food manufacturing', 'examples'),
            (1, '311111', 'Cat food manufacturing', 'validation'),
            (2, '441111', 'New car dealers', 'examples'),
        ],
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )
    examples = {'311111': 'Dog food manufacturing', '441111': 'New car dealers'}
    descriptions = hierarchy_descriptions.with_columns(
        examples=pl.col('code').replace_strict(examples, default=None)
    )
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    roles_path = tmp_path / 'naics_index_roles.parquet'
    redirections_path = tmp_path / 'naics_redirections.parquet'
    descriptions.write_parquet(descriptions_path)
    roles.write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    cfg = SupervisionBuildConfig(
        descriptions_parquet=str(descriptions_path),
        index_roles_parquet=str(roles_path),
        redirections_parquet=str(redirections_path),
        output_root=str(tmp_path / 'bundles'),
    )

    manifest = json.loads(generate_supervision_bundle(cfg).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
    assert manifest['artifacts']['redirections']['row_count'] == 2
    parameters = manifest['generation_parameters']
    assert parameters['index_roles_parquet'] == str(roles_path.resolve())
    assert parameters['redirections_parquet'] == str(redirections_path.resolve())
```

In `tests/unit/test_config.py`, replace:

```python
        # The shipped build carries the index roles; the default (for fixtures) does not
        assert cfg == SupervisionBuildConfig(index_roles_parquet='./data/naics_index_roles.parquet')
```

with:

```python
        assert cfg == SupervisionBuildConfig()
        assert cfg.index_roles_parquet == './data/naics_index_roles.parquet'
        assert cfg.redirections_parquet == './data/naics_redirections.parquet'
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py -q`
Expected: `Interrupted: 1 error during collection`, with `ImportError: cannot import name
'REDIRECTIONS_SCHEMA' from 'naics_embedder.supervision.artifacts'`.

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_structural_margins.py -q`
Expected: `1 failed, 77 passed, 1 error`.
- `TestSupervisionBuildConfig::test_yaml_matches_defaults` fails its first assertion.
- The hierarchy test errors at setup in `hierarchy_manifest`: `redirections_parquet`, "Extra
  inputs are not permitted".

- [x] **Step 4: Move the redirection constants and add the member validators**

In `src/naics_embedder/supervision/schema.py`, replace:

```python
INDEX_ROLES_SCHEMA_VERSION = 'index-roles-v1'
```

with:

```python
INDEX_ROLES_SCHEMA_VERSION = 'index-roles-v1'
REDIRECTIONS_SCHEMA_VERSION = 'redirections-v1'
```

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
    'training_pairs',
    'difficulty_thresholds',
)

STRUCTURAL_PAIR_COLUMNS = (
```

with:

```python
    'training_pairs',
    'difficulty_thresholds',
    'index_roles',
    'redirections',
)

# Every check a bundle build records. The loader refuses a manifest that lacks one, so a bundle
# built without a check never loads.
REQUIRED_VALIDATION_RESULTS = (
    'codebook_contiguous_unique',
    'codebook_identity',
    'pair_keys_unique',
    'canonical_orientation',
    'pair_coverage',
    'nonzero_structural_distance',
    'no_structural_sentinel',
    'distance_is_d_star',
    'cross_sector_distance_formula',
    'cross_sector_relation_label',
    'distance_triangle_inequality',
    'exclusion_derivation',
    'exclusions_match_descriptions',
    'unary_pairs_flagged',
    'matrix_reconciliation',
    'redirections_well_formed',
    'redirections_match_exclusion_channel',
    'redirections_match_pair_facts',
    'index_roles_one_role_per_entry',
    'index_roles_examples_channel',
    'index_roles_no_leakage',
    'direct_positive_safety',
    'no_exclusion_negatives',
    'no_unary_positives',
    'training_exclusion_derivation',
    'training_identity_joins',
    'artifact_hashes_recorded',
)

STRUCTURAL_PAIR_COLUMNS = (
```

Replace:

```python
    if short.height:
        raise ValueError(
            f'{short.height:,} codes have fewer than {min_examples_per_code} examples-role entries'
        )

def validate_matrix(
```

with:

```python
    if short.height:
        raise ValueError(
            f'{short.height:,} codes have fewer than {min_examples_per_code} examples-role entries'
        )

REDIRECTIONS_ARTIFACT = 'redirections'
CROSS_REFERENCE_SOURCE = 'cross_reference'
DESCRIPTION_SOURCE = 'description'
REDIRECTIONS_SCHEMA = {
    'reference_id': pl.Int64,
    'source': pl.Utf8,
    'code': pl.Utf8,
    'text': pl.Utf8,
    'activity': pl.Utf8,
    'named_codes': pl.List(pl.Utf8),
    'lineal_codes': pl.List(pl.Utf8),
    'withheld': pl.Boolean,
}

def validate_redirection_table(redirections: pl.DataFrame, codes: Collection[str]) -> None:
    '''
    Fail closed unless the redirection table is well formed (Req 8).

    Its columns are ``REDIRECTIONS_SCHEMA``'s, in order, and ``reference_id`` runs from zero in
    table order. Every row comes from a known source and names codebook codes other than its
    own, and ``lineal_codes`` holds exactly the named codes that are the row's code's ancestors
    or descendants. Only a cross-reference row that names a code and is not withheld may carry
    an activity phrase.
    '''

    if redirections.schema != pl.Schema(REDIRECTIONS_SCHEMA):
        raise ValueError(f'redirection columns must be {list(REDIRECTIONS_SCHEMA)}, as typed there')
    required = [name for name in REDIRECTIONS_SCHEMA if name != 'activity']
    if redirections.select(pl.any_horizontal(pl.col(required).is_null()).any()).item():
        raise ValueError('a redirection row lacks a required value')
    if redirections.get_column('reference_id').to_list() != list(range(redirections.height)):
        raise ValueError('reference IDs must run from zero in table order')
    sources = set(redirections.get_column('source').to_list())
    unknown = sorted(sources - {CROSS_REFERENCE_SOURCE, DESCRIPTION_SOURCE})
    if unknown:
        raise ValueError(f'unknown redirection sources: {unknown}')
    known = set(codes)
    for row in redirections.iter_rows(named=True):
        code, named = row['code'], row['named_codes']
        where = f'redirection {row["reference_id"]} ({code})'
        if code not in known or not set(named) <= known:
            raise ValueError(f'{where} names a code outside the codebook')
        if code in named:
            raise ValueError(f'{where} names its own code')
        lineal = [
            other for other in named if other in code_lineage(code) or code in code_lineage(other)
        ]
        if row['lineal_codes'] != lineal:
            raise ValueError(
                f'{where}: lineal_codes must be the named ancestors and descendants, {lineal}'
            )
        redirects = row['source'] == CROSS_REFERENCE_SOURCE and bool(named)
        if row['activity'] is not None and (row['withheld'] or not redirects):
            raise ValueError(
                f'{where} carries an activity phrase, which only a cross-reference row that '
                'names a code and is not withheld may carry'
            )

def validate_redirection_exclusions(redirections: pl.DataFrame, pair_facts: pl.DataFrame) -> None:
    '''
    Fail closed unless the redirection table names exactly the pair facts' explicit exclusions.

    Every (code, named code) pair of the table, withheld rows included, must be a directed
    exclusion of the pair facts, and every directed exclusion must be named by some row.
    '''

    # yapf: disable
    named = (
        redirections
        .select('code', other=pl.col('named_codes'))
        .explode('other')
        .drop_nulls()
        .unique()
    )
    # yapf: enable
    excluded = pl.concat(
        [
            pair_facts.filter('code_i_excludes_code_j').select(code='code_i', other='code_j'),
            pair_facts.filter('code_j_excludes_code_i').select(code='code_j', other='code_i'),
        ]
    ).unique()
    unmatched = named.join(excluded, on=['code', 'other'], how='anti').height
    unnamed = excluded.join(named, on=['code', 'other'], how='anti').height
    if unmatched or unnamed:
        raise ValueError(
            f'the redirection table and the pair facts disagree: {unmatched:,} named pairs are '
            f'not exclusions, and {unnamed:,} exclusions are named by no row'
        )

def validate_matrix(
```

Replace:

```python
    # Optional under stage3-supervision-v1: bundles built before the outcome panel lack it
    if INDEX_ROLES_ARTIFACT in manifest.artifacts:
        roles = read(INDEX_ROLES_ARTIFACT)
        six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        _in_context(
            INDEX_ROLES_ARTIFACT,
            bundle_id,
            lambda: validate_index_role_table(roles, six_digit_codes.to_list()),
        )
```

with:

```python
    roles = read(INDEX_ROLES_ARTIFACT)
    six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
    _in_context(
        INDEX_ROLES_ARTIFACT,
        bundle_id,
        lambda: validate_index_role_table(roles, six_digit_codes.to_list()),
    )

    redirections = read(REDIRECTIONS_ARTIFACT)

    def check_redirections() -> None:
        validate_redirection_table(redirections, codebook.get_column('code').to_list())
        validate_redirection_exclusions(redirections, pair_facts)

    _in_context(REDIRECTIONS_ARTIFACT, bundle_id, check_redirections)
```

Replace:

```python
    Checks the contract version, every member's existence, hash, row count, and Parquet contract
    metadata, the recorded validation results, and then re-runs the relational checks: codebook
    order and fingerprint, pair-fact identity/orientation/coverage/sentinels/exclusion derivation,
    long-form and matrix reconciliation, training-pair identity, exclusion, and structure, and,
    when the bundle carries one, the index-entry role table (one known role per entry, six-digit
    codes only, the examples-channel floor).
```

with:

```python
    Checks the contract version, every member's existence, hash, row count, and Parquet contract
    metadata, and that every required validation result is recorded as passed. It then re-runs
    the relational checks: codebook order and fingerprint; pair-fact identity, orientation,
    coverage, sentinels and exclusion derivation; long-form and matrix reconciliation;
    training-pair identity, exclusion, and structure; the index-entry role table (one known role
    per entry, six-digit codes only, the examples-channel floor); and the redirection table (well
    formed, naming exactly the pair facts' exclusions).
```

Replace:

```python
    root = path.parent
    _validate_members(root, manifest)
    if not all(manifest.validation_results.values()):
```

with:

```python
    root = path.parent
    _validate_members(root, manifest)
    missing = sorted(set(REQUIRED_VALIDATION_RESULTS) - set(manifest.validation_results))
    if missing:
        raise ValueError(
            f'bundle {manifest.bundle_id} lacks required validation results: {missing}'
        )
    if not all(manifest.validation_results.values()):
```

In `src/naics_embedder/data/redirections.py`, replace:

```python
from naics_embedder.utils.naics_hierarchy import code_lineage

logger = logging.getLogger(__name__)

CROSS_REFERENCE_SOURCE = 'cross_reference'
DESCRIPTION_SOURCE = 'description'
REDIRECTIONS_SCHEMA = {
    'reference_id': pl.Int64,
    'source': pl.Utf8,
    'code': pl.Utf8,
    'text': pl.Utf8,
    'activity': pl.Utf8,
    'named_codes': pl.List(pl.Utf8),
    'lineal_codes': pl.List(pl.Utf8),
    'withheld': pl.Boolean,
}
```

with:

```python
from naics_embedder.supervision.artifacts import (
    CROSS_REFERENCE_SOURCE,
    DESCRIPTION_SOURCE,
    REDIRECTIONS_SCHEMA,
)
from naics_embedder.utils.naics_hierarchy import code_lineage

logger = logging.getLogger(__name__)
```

- [x] **Step 5: Build and require both members**

In `src/naics_embedder/data/supervision_bundle.py`, replace:

```python
from typing import Any, Dict, Iterator, Mapping, Optional
```

with:

```python
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
```

Replace:

```python
from naics_embedder.panels.index_roles import verify_examples_channel, verify_role_leakage
from naics_embedder.supervision.artifacts import (
    INDEX_ROLE_COLUMNS,
    INDEX_ROLES_ARTIFACT,
    STRUCTURAL_PAIR_COLUMNS,
    codebook_fingerprint,
    sha256_file,
    unary_pair_expression,
    validate_exclusion_derivation,
    validate_index_role_table,
    validate_matrix,
    validate_structural_pairs,
```

with:

```python
from naics_embedder.data.redirections import exclusion_channel
from naics_embedder.panels.index_roles import verify_examples_channel, verify_role_leakage
from naics_embedder.supervision.artifacts import (
    INDEX_ROLE_COLUMNS,
    INDEX_ROLES_ARTIFACT,
    REDIRECTIONS_ARTIFACT,
    STRUCTURAL_PAIR_COLUMNS,
    codebook_fingerprint,
    sha256_file,
    unary_pair_expression,
    validate_exclusion_derivation,
    validate_index_role_table,
    validate_matrix,
    validate_redirection_exclusions,
    validate_redirection_table,
    validate_structural_pairs,
```

Replace:

```python
    PAIR_FACTS_SCHEMA_VERSION,
    RELATION_MATRIX_SCHEMA_VERSION,
```

with:

```python
    PAIR_FACTS_SCHEMA_VERSION,
    REDIRECTIONS_SCHEMA_VERSION,
    RELATION_MATRIX_SCHEMA_VERSION,
```

Replace:

```python
    INDEX_ROLES_ARTIFACT: 'naics_index_roles.parquet',
}
```

with:

```python
    INDEX_ROLES_ARTIFACT: 'naics_index_roles.parquet',
    REDIRECTIONS_ARTIFACT: 'naics_redirections.parquet',
}
```

Replace:

```python
    INDEX_ROLES_ARTIFACT: INDEX_ROLES_SCHEMA_VERSION,
}
```

with:

```python
    INDEX_ROLES_ARTIFACT: INDEX_ROLES_SCHEMA_VERSION,
    REDIRECTIONS_ARTIFACT: REDIRECTIONS_SCHEMA_VERSION,
}
```

Replace:

```python
def _validate_index_roles(
    index_roles: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> Dict[str, bool]:
    '''Check the optional index-roles member against the bundle's own descriptions and codes.'''

    six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
    validate_index_role_table(index_roles, six_digit_codes.to_list())
    verify_examples_channel(descriptions, index_roles)
    verify_role_leakage(descriptions, index_roles)
    return {
        'index_roles_one_role_per_entry': True,
        'index_roles_examples_channel': True,
        'index_roles_no_leakage': True,
    }
```

with:

```python
def _validate_index_roles(
    index_roles: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
    activities: Sequence[str],
) -> Dict[str, bool]:
    '''
    Check the index-roles member against the bundle's descriptions, codes and activity phrases.

    No held-out query may match training text, which includes the redirection table's activity
    phrases, because Stage 7 trains on them as queries (Req 3).
    '''

    six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
    validate_index_role_table(index_roles, six_digit_codes.to_list())
    verify_examples_channel(descriptions, index_roles)
    verify_role_leakage(descriptions, index_roles, extra_texts=activities)
    return {
        'index_roles_one_role_per_entry': True,
        'index_roles_examples_channel': True,
        'index_roles_no_leakage': True,
    }

def _validate_redirections(
    redirections: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
    pair_facts: pl.DataFrame,
) -> Dict[str, bool]:
    '''
    Check the redirections member against the codebook, the descriptions and the pair facts.

    The descriptions' exclusion channel must be the one the table builds, so each cross-reference
    appears in it once (Req 8(a)), and the table must name exactly the pair facts' exclusions.
    '''

    validate_redirection_table(redirections, codebook.get_column('code').to_list())
    built = descriptions.select('code').join(exclusion_channel(redirections), on='code', how='left')
    carried = descriptions.select('code', 'excluded', 'excluded_codes')
    if not built.sort('code').equals(carried.sort('code')):
        raise ValueError(
            'descriptions carry an exclusion channel other than the redirection table builds'
        )
    validate_redirection_exclusions(redirections, pair_facts)
    return {
        'redirections_well_formed': True,
        'redirections_match_exclusion_channel': True,
        'redirections_match_pair_facts': True,
    }
```

Replace:

```python
    cross_sector_cap: int,
    cap_seed: int,
    index_roles: Optional[pl.DataFrame] = None,
) -> Dict[str, ArtifactRecord]:
```

with:

```python
    cross_sector_cap: int,
    cap_seed: int,
    index_roles: pl.DataFrame,
    redirections: pl.DataFrame,
) -> Dict[str, ArtifactRecord]:
```

Replace:

```python
        'relation_matrix': _record('relation_matrix', parquet('relation_matrix', relation_matrix)),
    }
    if index_roles is not None:
        records[INDEX_ROLES_ARTIFACT] = _record(
            INDEX_ROLES_ARTIFACT,
            parquet(INDEX_ROLES_ARTIFACT,
                    index_roles.select(INDEX_ROLE_COLUMNS).sort('entry_id')),
        )
```

with:

```python
        'relation_matrix': _record('relation_matrix', parquet('relation_matrix', relation_matrix)),
        INDEX_ROLES_ARTIFACT: _record(
            INDEX_ROLES_ARTIFACT,
            parquet(INDEX_ROLES_ARTIFACT,
                    index_roles.select(INDEX_ROLE_COLUMNS).sort('entry_id')),
        ),
        REDIRECTIONS_ARTIFACT: _record(
            REDIRECTIONS_ARTIFACT, parquet(REDIRECTIONS_ARTIFACT, redirections)
        ),
    }
```

Replace:

```python
    descriptions: pl.DataFrame,
    pair_facts: pl.DataFrame,
    description_fingerprint: Optional[str] = None,
    structural_relation_ids: Optional[Mapping[str, int]] = None,
    generation_parameters: Optional[Mapping[str, Any]] = None,
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
    index_roles: Optional[pl.DataFrame] = None,
) -> Path:
    '''
    Validate canonical frames and publish them as one immutable supervision bundle.

    ``index_roles`` (every index entry with its text and role) becomes the optional
    ``index_roles`` member after three checks against ``descriptions``: one known role per entry,
    examples channels built from examples-role entries only, and no held-out query matching any
    training text.
```

with:

```python
    descriptions: pl.DataFrame,
    pair_facts: pl.DataFrame,
    index_roles: pl.DataFrame,
    redirections: pl.DataFrame,
    description_fingerprint: Optional[str] = None,
    structural_relation_ids: Optional[Mapping[str, int]] = None,
    generation_parameters: Optional[Mapping[str, Any]] = None,
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
) -> Path:
    '''
    Validate canonical frames and publish them as one immutable supervision bundle.

    ``index_roles`` (every index entry with its text and role) and ``redirections`` (the
    redirection table) become required members after checks against ``descriptions``. Each entry
    holds one known role, examples channels hold examples-role entries only, and no held-out
    query matches any training text or activity phrase. The table is well formed, the
    descriptions' exclusion channel is the one it builds, and it names exactly the pair facts'
    exclusions.
```

Replace:

```python
    validation_results['matrix_reconciliation'] = True
    if index_roles is not None:
        validation_results.update(_validate_index_roles(index_roles, descriptions, codebook))
```

with:

```python
    validation_results['matrix_reconciliation'] = True
    validation_results.update(
        _validate_redirections(redirections, descriptions, codebook, pair_facts)
    )
    activities = redirections.get_column('activity').drop_nulls().to_list()
    validation_results.update(
        _validate_index_roles(index_roles, descriptions, codebook, activities)
    )
```

Replace:

```python
            cap_seed=cap_seed,
            index_roles=index_roles,
        )
```

with:

```python
            cap_seed=cap_seed,
            index_roles=index_roles,
            redirections=redirections,
        )
```

Replace:

```python
def generate_supervision_bundle(cfg: SupervisionBuildConfig) -> Path:
    '''
    Build and publish a new supervision bundle from the configured descriptions.

    The bundle ID is a fresh UUID4, and the description fingerprint is the SHA-256 of the exact
    descriptions file, so training can later verify it runs against the same input.

    Returns:
        Path to the published ``manifest.json``.
    '''

    descriptions_path = Path(cfg.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path)
    parameters = {
        'descriptions_parquet': str(descriptions_path.resolve()),
        'output_root': str(Path(cfg.output_root).resolve()),
    }
    index_roles = None
    if cfg.index_roles_parquet is not None:
        index_roles_path = Path(cfg.index_roles_parquet)
        index_roles = pl.read_parquet(index_roles_path)
        parameters['index_roles_parquet'] = str(index_roles_path.resolve())
    codebook = build_codebook(descriptions)
```

with:

```python
def generate_supervision_bundle(cfg: SupervisionBuildConfig) -> Path:
    '''
    Build and publish a new supervision bundle from the files ``data preprocess`` writes.

    It reads the configured descriptions, index roles and redirection table. The bundle ID is a
    fresh UUID4, and the description fingerprint is the SHA-256 of the exact descriptions file,
    so training can later verify it runs against the same input.

    Returns:
        Path to the published ``manifest.json``.
    '''

    descriptions_path = Path(cfg.descriptions_parquet)
    index_roles_path = Path(cfg.index_roles_parquet)
    redirections_path = Path(cfg.redirections_parquet)
    descriptions = pl.read_parquet(descriptions_path)
    parameters = {
        'descriptions_parquet': str(descriptions_path.resolve()),
        'index_roles_parquet': str(index_roles_path.resolve()),
        'redirections_parquet': str(redirections_path.resolve()),
        'output_root': str(Path(cfg.output_root).resolve()),
    }
    codebook = build_codebook(descriptions)
```

Replace:

```python
        descriptions=descriptions,
        pair_facts=pair_facts,
        description_fingerprint=sha256_file(descriptions_path),
        structural_relation_ids=cfg.relation_id,
        generation_parameters=parameters,
        index_roles=index_roles,
    )
```

with:

```python
        descriptions=descriptions,
        pair_facts=pair_facts,
        index_roles=pl.read_parquet(index_roles_path),
        redirections=pl.read_parquet(redirections_path),
        description_fingerprint=sha256_file(descriptions_path),
        structural_relation_ids=cfg.relation_id,
        generation_parameters=parameters,
    )
```

In `src/naics_embedder/utils/config.py`, replace:

```python
    index_roles_parquet: Optional[str] = Field(
        default=None,
        description=(
            'Index entries with their roles, from `data preprocess`; when set, the bundle carries '
            'them as its optional index_roles member'
        ),
    )
```

with:

```python
    index_roles_parquet: str = Field(
        default='./data/naics_index_roles.parquet',
        description='The index_roles member: index entries and roles from `data preprocess`',
    )
    redirections_parquet: str = Field(
        default='./data/naics_redirections.parquet',
        description='The redirections member: the redirection table from `data preprocess`',
    )
```

In `src/naics_embedder/panels/outcome.py`, replace:

```python
        The bundle checked the roles against its descriptions when it was built.

        Raises:
            ValueError: If the bundle has no ``index_roles`` member.
        '''
```

with:

```python
        Every loadable bundle carries the member, whose roles its build checked against its
        descriptions.
        '''
```

In `conf/data/supervision.yaml`, replace:

```yaml
index_roles_parquet: ./data/naics_index_roles.parquet
```

with:

```yaml
index_roles_parquet: ./data/naics_index_roles.parquet
redirections_parquet: ./data/naics_redirections.parquet
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_config.py tests/unit/test_structural_margins.py tests/unit/test_redirections.py tests/unit/test_outcome_panel.py tests/unit/test_graph_preprocessing.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1680 passed, 1 skipped` (16 new tests, one removed).

- [x] **Step 7: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/artifacts.py src/naics_embedder/supervision/schema.py src/naics_embedder/data/redirections.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/utils/config.py src/naics_embedder/panels/outcome.py tests/fixtures/supervision.py tests/unit/test_supervision_artifacts.py tests/unit/test_outcome_panel.py tests/unit/test_graph_preprocessing.py tests/unit/test_streaming_sampling.py tests/unit/test_hard_negative_mining.py tests/unit/test_utils_validation.py tests/unit/test_datamodule.py tests/unit/test_structural_margins.py tests/unit/test_config.py tests/integration/test_stage3_training_step.py tests/integration/test_distributed_supervision.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 8: Commit**

```bash
git add src/naics_embedder/supervision/artifacts.py \
  src/naics_embedder/supervision/schema.py \
  src/naics_embedder/data/redirections.py \
  src/naics_embedder/data/supervision_bundle.py \
  src/naics_embedder/utils/config.py \
  src/naics_embedder/panels/outcome.py \
  conf/data/supervision.yaml \
  tests/fixtures/supervision.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_outcome_panel.py \
  tests/unit/test_graph_preprocessing.py \
  tests/unit/test_streaming_sampling.py \
  tests/unit/test_hard_negative_mining.py \
  tests/unit/test_utils_validation.py \
  tests/unit/test_datamodule.py \
  tests/unit/test_structural_margins.py \
  tests/unit/test_config.py \
  tests/integration/test_stage3_training_step.py \
  tests/integration/test_distributed_supervision.py
git commit -m "feat(supervision): every bundle carries its index roles and redirection table"
```

### Task 11: The manifest records the backbone's input window

Verification "Backbone input window" asks for two records: the backbone's trained window, from
its own documentation, and the share of each channel's texts that exceed it, counted after
de-duplication. Task 4 recorded the window in `utils/input_window.py` and made every tokenizing
path truncate to it. This task writes the window into every bundle's manifest as `input_window`,
which holds:

- the backbone and its window;
- per text channel, the present texts, the texts over the window, and their share.

The build counts tokens with the backbone's own tokenizer, read from the local cache only. Tests
pass a word-count stub instead, so no tokenizer loads in the suite. One test replaces
`AutoTokenizer.from_pretrained` to pin the call the build makes. `SupervisionBuildConfig.backbone`
names the backbone, and a test holds it equal to `model.base_model_name` in `conf/config.yaml`.

A v2 manifest carries a field that a v1 manifest lacks, so the loader now reads the contract
version from the raw JSON before it parses the manifest. An old bundle then fails with the
contract message, not a missing-field error.

**Files:**
- Modify: `src/naics_embedder/supervision/schema.py` (`ChannelOverflow`, `InputWindowRecord`,
  `SupervisionManifest.input_window`)
- Modify: `src/naics_embedder/supervision/artifacts.py` (`load_validated_bundle` reads the
  contract first)
- Modify: `src/naics_embedder/data/supervision_bundle.py` (`input_window_record`, the record's
  check, both builders)
- Modify: `src/naics_embedder/utils/config.py` (`SupervisionBuildConfig.backbone`),
  `conf/data/supervision.yaml`
- Test: `tests/fixtures/supervision.py`, `tests/unit/test_supervision_artifacts.py`,
  `tests/unit/test_supervision_schema.py`, `tests/unit/test_config.py`

**Interfaces:**
- Consumes:
  - Task 4's `utils/input_window.py`: `trained_window(backbone) -> int`,
    `token_counter(tokenizer)`, and `overflow_shares(texts, count_tokens, window)`, which returns
    `{channel: {'present', 'over', 'share'}}`.
  - Task 2's `data/download_data.TEXT_CHANNELS`, which is
    `('title', 'description', 'examples', 'excluded')`.
  - Task 10's `build_bundle`, `hierarchy_redirections` and `hierarchy_manifest` fixtures.
- Produces:
  - `supervision.schema.ChannelOverflow(present: int, over: int, share: float)`, which refuses
    `over > present`, and `InputWindowRecord(backbone: str, window: int, channels: Dict[str,
    ChannelOverflow])`.
  - `SupervisionManifest.input_window: InputWindowRecord`, a required field.
  - `supervision_bundle.input_window_record(descriptions, backbone, count_tokens) ->
    InputWindowRecord`.
  - `generate_supervision_bundle(cfg, *, count_tokens=None)`. When `count_tokens` is None, it
    counts with `AutoTokenizer.from_pretrained(cfg.backbone, local_files_only=True)`.
  - `generate_supervision_bundle_from_frames` gains `input_window: InputWindowRecord`, a required
    argument. It refuses a record whose window is not the backbone's trained window, whose
    channels are not the four text channels, or whose present counts differ from the
    descriptions'.
  - `SupervisionBuildConfig.backbone: str = 'sentence-transformers/all-MiniLM-L6-v2'`, which
    refuses a backbone with no recorded window.
  - Fixtures:
    - `count_words`, a stub counter: one token per word, plus two for `[CLS]` and `[SEP]`;
    - `hierarchy_build_config`, the hierarchy's build config, with its inputs written;
    - `hierarchy_manifest`, now built from `hierarchy_build_config` with `count_words`;
    - `FIVE_CODE_INPUT_WINDOW`, the five-code bundle's record, as a dict.

- [x] **Step 1: Write the failing tests**

In `tests/fixtures/supervision.py`, replace:

```python
from pathlib import Path

import polars as pl
```

with:

```python
from pathlib import Path
from typing import List

import polars as pl
```

Replace:

```python
from naics_embedder.supervision.candidates import NegativeCandidateBatch
from naics_embedder.utils.config import SupervisionBuildConfig
```

with:

```python
from naics_embedder.supervision.candidates import NegativeCandidateBatch
from naics_embedder.supervision.schema import InputWindowRecord
from naics_embedder.utils.config import SupervisionBuildConfig
```

Replace:

```python
@pytest.fixture
def redirections_fixture() -> pl.DataFrame:
```

with:

```python
# The five codes' texts are short, so none exceeds the window
FIVE_CODE_INPUT_WINDOW = {
    'backbone': 'sentence-transformers/all-MiniLM-L6-v2',
    'window': 128,
    'channels': {
        'title': {
            'present': 5,
            'over': 0,
            'share': 0.0
        },
        'description': {
            'present': 5,
            'over': 0,
            'share': 0.0
        },
        'examples': {
            'present': 3,
            'over': 0,
            'share': 0.0
        },
        'excluded': {
            'present': 2,
            'over': 0,
            'share': 0.0
        },
    },
}

@pytest.fixture
def redirections_fixture() -> pl.DataFrame:
```

Replace:

```python
            'index_roles': index_roles_fixture,
            'redirections': redirections_fixture,
        }
        return generate_supervision_bundle_from_frames(**{**inputs, **overrides})
```

with:

```python
            'index_roles': index_roles_fixture,
            'redirections': redirections_fixture,
            'input_window': InputWindowRecord.model_validate(FIVE_CODE_INPUT_WINDOW),
        }
        return generate_supervision_bundle_from_frames(**{**inputs, **overrides})
```

Replace:

```python
@pytest.fixture
def hierarchy_manifest(tmp_path, hierarchy_descriptions_parquet, hierarchy_redirections) -> Path:
    '''
    The manifest of the bundle that the production path builds from the hierarchy.

    The hierarchy has no index entries, so its role table is empty.
    '''

    roles_path = tmp_path / 'hierarchy_index_roles.parquet'
    redirections_path = tmp_path / 'hierarchy_redirections.parquet'
    pl.DataFrame(schema=INDEX_ROLE_SCHEMA).write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    return generate_supervision_bundle(
        SupervisionBuildConfig(
            descriptions_parquet=hierarchy_descriptions_parquet,
            index_roles_parquet=str(roles_path),
            redirections_parquet=str(redirections_path),
            output_root=str(tmp_path / 'bundles'),
        )
    )
```

with:

```python
def _count_words(texts: List[str]) -> List[int]:
    return [len(text.split()) + 2 for text in texts]

@pytest.fixture
def count_words():
    '''Token counts for tests: one token per word, plus two for [CLS] and [SEP].'''

    return _count_words

@pytest.fixture
def hierarchy_build_config(
    tmp_path, hierarchy_descriptions_parquet, hierarchy_redirections
) -> SupervisionBuildConfig:
    '''
    The build configuration of the hierarchy's bundle, its inputs written under ``tmp_path``.

    The hierarchy has no index entries, so its role table is empty.
    '''

    roles_path = tmp_path / 'hierarchy_index_roles.parquet'
    redirections_path = tmp_path / 'hierarchy_redirections.parquet'
    pl.DataFrame(schema=INDEX_ROLE_SCHEMA).write_parquet(roles_path)
    hierarchy_redirections.write_parquet(redirections_path)
    return SupervisionBuildConfig(
        descriptions_parquet=hierarchy_descriptions_parquet,
        index_roles_parquet=str(roles_path),
        redirections_parquet=str(redirections_path),
        output_root=str(tmp_path / 'bundles'),
    )

@pytest.fixture
def hierarchy_manifest(hierarchy_build_config) -> Path:
    '''The manifest of the bundle that the production path builds from the hierarchy.'''

    return generate_supervision_bundle(hierarchy_build_config, count_tokens=_count_words)
```

In `tests/unit/test_supervision_artifacts.py`, replace:

```python
    generate_supervision_bundle,
    relation_matrix_from_pair_facts,
)
```

with:

```python
    generate_supervision_bundle,
    input_window_record,
    relation_matrix_from_pair_facts,
)
```

Replace:

```python
from naics_embedder.supervision.schema import CONTRACT_VERSION
from naics_embedder.utils.config import SupervisionBuildConfig
```

with:

```python
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ChannelOverflow,
    InputWindowRecord,
)
from naics_embedder.utils.config import SupervisionBuildConfig
from tests.fixtures.supervision import FIVE_CODE_INPUT_WINDOW
```

Replace:

```python
def test_production_bundle_takes_its_members_from_its_config(
    tmp_path, hierarchy_descriptions, hierarchy_redirections
):
```

with:

```python
def test_production_bundle_takes_its_members_from_its_config(
    tmp_path, hierarchy_descriptions, hierarchy_redirections, count_words
):
```

Replace:

```python
    manifest = json.loads(generate_supervision_bundle(cfg).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
```

with:

```python
    manifest = json.loads(generate_supervision_bundle(cfg, count_tokens=count_words).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
```

Append to the end of `tests/unit/test_supervision_artifacts.py`:

```python

# -------------------------------------------------------------------------------------------------
# The input-window record
# -------------------------------------------------------------------------------------------------

def _five_code_record(examples: int = 3, **changes) -> InputWindowRecord:
    '''The five-code record, counting ``examples`` present examples texts, with fields changed.'''

    channels = dict(FIVE_CODE_INPUT_WINDOW['channels'])
    channels['examples'] = {'present': examples, 'over': 0, 'share': 0.0}
    fields = {**FIVE_CODE_INPUT_WINDOW, 'channels': channels, **changes}
    return InputWindowRecord.model_validate(fields)

def test_the_input_window_record_counts_each_channels_texts_beyond_the_window(
    text_descriptions_fixture, count_words
):
    # Under the word count, 127 words make 129 tokens, one beyond the window; 126 words fit it
    texts = {'111111': ' '.join(['farming'] * 127), '111112': ' '.join(['farming'] * 126)}
    descriptions = text_descriptions_fixture.with_columns(
        description=pl.col('code').replace_strict(texts, default=pl.col('description'))
    )

    record = input_window_record(
        descriptions, 'sentence-transformers/all-MiniLM-L6-v2', count_words
    )

    assert record.window == 128
    assert record.channels['description'] == ChannelOverflow(present=5, over=1, share=0.2)
    assert record.channels['examples'] == ChannelOverflow(present=3, over=0, share=0.0)
    assert record.channels['excluded'] == ChannelOverflow(present=2, over=0, share=0.0)

def test_a_bundle_records_its_input_window(generated_bundle):
    manifest = json.loads(generated_bundle.read_text())

    assert manifest['input_window'] == FIVE_CODE_INPUT_WINDOW
    assert load_validated_bundle(generated_bundle).manifest.input_window == _five_code_record()

@pytest.mark.parametrize(
    ('record', 'message'),
    [
        (_five_code_record(window=256), 'all-MiniLM-L6-v2 is 128 tokens, not 256'),
        (_five_code_record(channels={}), 'must cover the channels'),
        (_five_code_record(examples=4), 'counts 4 examples texts, but the descriptions hold 3'),
    ],
)
def test_bundle_refuses_an_input_window_record_that_does_not_fit(
    tmp_path, build_bundle, record, message
):
    with pytest.raises(ValueError, match=message):
        build_bundle(input_window=record)
    assert list(tmp_path.iterdir()) == []

def test_production_bundle_counts_tokens_with_the_backbones_cached_tokenizer(
    monkeypatch, hierarchy_build_config
):
    import transformers

    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        # Every text is 130 tokens, beyond the window
        return lambda texts, truncation: {'input_ids': [[0] * 130 for _ in texts]}

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', from_pretrained)

    manifest = json.loads(generate_supervision_bundle(hierarchy_build_config).read_text())

    assert calls == [('sentence-transformers/all-MiniLM-L6-v2', {'local_files_only': True})]
    assert manifest['input_window'] == {
        'backbone': 'sentence-transformers/all-MiniLM-L6-v2',
        'window': 128,
        'channels': {
            'title': {
                'present': 17,
                'over': 17,
                'share': 1.0
            },
            'description': {
                'present': 17,
                'over': 17,
                'share': 1.0
            },
            'examples': {
                'present': 0,
                'over': 0,
                'share': 0.0
            },
            'excluded': {
                'present': 2,
                'over': 2,
                'share': 1.0
            },
        },
    }

def test_loader_names_the_contract_of_a_manifest_it_cannot_parse(generated_bundle):
    # A v1 manifest predates the input-window record, so it does not parse as a v2 one
    manifest = json.loads(generated_bundle.read_text())
    manifest['contract_version'] = 'stage3-supervision-v1'
    del manifest['input_window']
    generated_bundle.write_text(json.dumps(manifest, indent=2))

    with pytest.raises(
        ValueError,
        match='expected supervision contract stage3-supervision-v2, found stage3-supervision-v1',
    ):
        load_validated_bundle(generated_bundle)
```

In `tests/unit/test_supervision_schema.py`, replace:

```python
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    ArtifactRecord,
    SemanticSource,
    SemanticTarget,
    SupervisionManifest,
)
```

with:

```python
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    ArtifactFile,
    ArtifactRecord,
    ChannelOverflow,
    InputWindowRecord,
    SemanticSource,
    SemanticTarget,
    SupervisionManifest,
)
```

Replace:

```python
        validation_results={'codebook_unique': True},
    )
```

with:

```python
        validation_results={'codebook_unique': True},
        input_window=InputWindowRecord(
            backbone='sentence-transformers/all-MiniLM-L6-v2',
            window=128,
            channels={'title': ChannelOverflow(present=3, over=0, share=0.0)},
        ),
    )
```

Append to the end of `tests/unit/test_supervision_schema.py`:

```python

def test_manifest_requires_the_input_window_record():
    manifest = _manifest().model_dump()
    del manifest['input_window']

    with pytest.raises(ValidationError, match='input_window'):
        SupervisionManifest.model_validate(manifest)

def test_channel_overflow_refuses_more_texts_beyond_the_window_than_present():
    with pytest.raises(ValidationError, match='more texts exceed the window than are present'):
        ChannelOverflow(present=1, over=2, share=1.0)
```

In `tests/unit/test_config.py`, replace:

```python
    def test_rejects_other_contract_versions(self):
        with pytest.raises(ValidationError):
            SupervisionBuildConfig(contract_version='legacy')
```

with:

```python
    def test_rejects_other_contract_versions(self):
        with pytest.raises(ValidationError):
            SupervisionBuildConfig(contract_version='legacy')

    def test_backbone_is_the_training_backbone(self, valid_config_dict):
        # The manifest records the window of the backbone that training reads
        assert SupervisionBuildConfig().backbone == valid_config_dict['model']['base_model_name']

    def test_rejects_a_backbone_without_a_recorded_window(self):
        with pytest.raises(ValidationError, match='no trained input window is recorded'):
            SupervisionBuildConfig(backbone='bert-base-uncased')
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_supervision_schema.py -q`
Expected: the run stops before collecting anything, because the shared fixtures cannot import
the record: `ImportError: Error importing plugin "tests.fixtures.supervision": cannot import
name 'InputWindowRecord' from 'naics_embedder.supervision.schema'`.

- [x] **Step 3: Add the record to the manifest**

In `src/naics_embedder/supervision/schema.py`, replace:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
```

with:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
```

Replace:

```python
class SupervisionManifest(BaseModel):
    '''Top-level, write-last description of one immutable supervision bundle.'''
```

with:

```python
class ChannelOverflow(BaseModel):
    '''One text channel's present texts and those longer than the input window.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    present: int = Field(ge=0)
    over: int = Field(ge=0)
    share: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_counts(self) -> 'ChannelOverflow':
        if self.over > self.present:
            raise ValueError('more texts exceed the window than are present')
        return self

class InputWindowRecord(BaseModel):
    '''
    The backbone's trained input window, and each text channel's texts beyond it (Req 9).

    The window comes from the backbone's own documentation (``utils/input_window.py``), and
    every tokenizing path truncates to it, so these counts are the texts truncation shortens.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')

    backbone: str = Field(min_length=1)
    window: int = Field(gt=0)
    channels: Dict[str, ChannelOverflow]

class SupervisionManifest(BaseModel):
    '''Top-level, write-last description of one immutable supervision bundle.'''
```

Replace:

```python
    artifacts: Dict[str, ArtifactRecord]
    validation_results: Mapping[str, bool]
```

with:

```python
    artifacts: Dict[str, ArtifactRecord]
    validation_results: Mapping[str, bool]
    input_window: InputWindowRecord
```

In `src/naics_embedder/supervision/artifacts.py`, replace:

```python
import hashlib
from dataclasses import dataclass
```

with:

```python
import hashlib
import json
from dataclasses import dataclass
```

Replace:

```python
    Checks the contract version, every member's existence, hash, row count, and Parquet contract
    metadata, and that every required validation result is recorded as passed. It then re-runs
```

with:

```python
    Reads the contract version from the raw manifest before parsing it, so an older contract's
    bundle fails with the contract message rather than a parse error. Then checks every member's
    existence, hash, row count, and Parquet contract metadata, and that every required
    validation result is recorded as passed. It then re-runs
```

Replace:

```python
    manifest = SupervisionManifest.model_validate_json(path.read_text())
    if manifest.contract_version != expected_contract:
        raise ValueError(
            f'expected supervision contract {expected_contract}, '
            f'found {manifest.contract_version} in {path}'
        )
```

with:

```python
    raw = json.loads(path.read_text())
    found = raw.get('contract_version') if isinstance(raw, dict) else None
    if found != expected_contract:
        raise ValueError(
            f'expected supervision contract {expected_contract}, found {found} in {path}'
        )
    manifest = SupervisionManifest.model_validate(raw)
```

- [x] **Step 4: Count the texts beyond the window at build time**

In `src/naics_embedder/data/supervision_bundle.py`, replace:

```python
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
```

with:

```python
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence
```

Replace:

```python
from naics_embedder.data.redirections import exclusion_channel
```

with:

```python
from naics_embedder.data.download_data import TEXT_CHANNELS
from naics_embedder.data.redirections import exclusion_channel
```

Replace:

```python
    ArtifactFile,
    ArtifactRecord,
    SupervisionManifest,
)
from naics_embedder.utils.config import DistancesConfig, SupervisionBuildConfig
```

with:

```python
    ArtifactFile,
    ArtifactRecord,
    InputWindowRecord,
    SupervisionManifest,
)
from naics_embedder.utils.config import DistancesConfig, SupervisionBuildConfig
from naics_embedder.utils.input_window import overflow_shares, token_counter, trained_window
```

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Bundle layout
# -------------------------------------------------------------------------------------------------
```

with:

```python
# -------------------------------------------------------------------------------------------------
# The input window
# -------------------------------------------------------------------------------------------------

def input_window_record(
    descriptions: pl.DataFrame,
    backbone: str,
    count_tokens: Callable[[List[str]], List[int]],
) -> InputWindowRecord:
    '''
    The backbone's trained window and each text channel's texts beyond it (Req 9).

    Each code's text counts once per channel. The exclusion channel is already de-duplicated,
    with each cross-reference in it once.

    Args:
        descriptions: Descriptions with the four text channels.
        backbone: The backbone whose trained window applies (``utils/input_window.py``).
        count_tokens: Token counts of a list of texts under the backbone's tokenizer, special
            tokens included.
    '''

    window = trained_window(backbone)
    texts = {channel: descriptions.get_column(channel).to_list() for channel in TEXT_CHANNELS}
    return InputWindowRecord(
        backbone=backbone,
        window=window,
        channels=overflow_shares(texts, count_tokens, window),
    )

def _validate_input_window(record: InputWindowRecord, descriptions: pl.DataFrame) -> None:
    '''The record must hold the backbone's trained window and count these descriptions' texts.'''

    window = trained_window(record.backbone)
    if record.window != window:
        raise ValueError(
            f'the trained input window of {record.backbone} is {window} tokens, not '
            f'{record.window}'
        )
    if sorted(record.channels) != sorted(TEXT_CHANNELS):
        raise ValueError(f'the input-window record must cover the channels {list(TEXT_CHANNELS)}')
    for channel in TEXT_CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        present = sum(1 for text in texts if text is not None and text.strip())
        if record.channels[channel].present != present:
            raise ValueError(
                f'the input-window record counts {record.channels[channel].present:,} {channel} '
                f'texts, but the descriptions hold {present:,}'
            )

# -------------------------------------------------------------------------------------------------
# Bundle layout
# -------------------------------------------------------------------------------------------------
```

Replace:

```python
    index_roles: pl.DataFrame,
    redirections: pl.DataFrame,
    description_fingerprint: Optional[str] = None,
```

with:

```python
    index_roles: pl.DataFrame,
    redirections: pl.DataFrame,
    input_window: InputWindowRecord,
    description_fingerprint: Optional[str] = None,
```

Replace:

```python
    descriptions' exclusion channel is the one it builds, and it names exactly the pair facts'
    exclusions.
```

with:

```python
    descriptions' exclusion channel is the one it builds, and it names exactly the pair facts'
    exclusions. ``input_window`` must hold the backbone's trained window and count the texts of
    ``descriptions``.
```

Replace:

```python
    codebook = build_codebook(descriptions)
    pair_facts = _normalize_pair_facts(pair_facts)
```

with:

```python
    codebook = build_codebook(descriptions)
    _validate_input_window(input_window, descriptions)
    pair_facts = _normalize_pair_facts(pair_facts)
```

Replace:

```python
            artifacts=artifacts,
            validation_results=validation_results,
        )
```

with:

```python
            artifacts=artifacts,
            validation_results=validation_results,
            input_window=input_window,
        )
```

Replace:

```python
def generate_supervision_bundle(cfg: SupervisionBuildConfig) -> Path:
    '''
    Build and publish a new supervision bundle from the files ``data preprocess`` writes.

    It reads the configured descriptions, index roles and redirection table. The bundle ID is a
    fresh UUID4, and the description fingerprint is the SHA-256 of the exact descriptions file,
    so training can later verify it runs against the same input.

    Returns:
        Path to the published ``manifest.json``.
    '''
```

with:

```python
def generate_supervision_bundle(
    cfg: SupervisionBuildConfig,
    *,
    count_tokens: Optional[Callable[[List[str]], List[int]]] = None,
) -> Path:
    '''
    Build and publish a new supervision bundle from the files ``data preprocess`` writes.

    It reads the configured descriptions, index roles and redirection table. The bundle ID is a
    fresh UUID4, and the description fingerprint is the SHA-256 of the exact descriptions file,
    so training can later verify it runs against the same input.

    Args:
        cfg: The build configuration.
        count_tokens: Token counts of a list of texts, for the input-window record. By default
            the backbone's own tokenizer counts them, read from the local cache only.

    Returns:
        Path to the published ``manifest.json``.
    '''
```

Replace:

```python
    pair_facts = build_pair_facts(distances, relations, descriptions, codebook)
    return generate_supervision_bundle_from_frames(
```

with:

```python
    pair_facts = build_pair_facts(distances, relations, descriptions, codebook)
    if count_tokens is None:
        # Imported here: only a real build loads the tokenizer
        from transformers import AutoTokenizer

        count_tokens = token_counter(
            AutoTokenizer.from_pretrained(cfg.backbone, local_files_only=True)
        )
    input_window = input_window_record(descriptions, cfg.backbone, count_tokens)
    logger.info(f'Input window: {input_window.window} tokens ({input_window.backbone})')
    for channel, overflow in input_window.channels.items():
        logger.info(
            f'  {channel}: {overflow.over:,} of {overflow.present:,} texts beyond it '
            f'({overflow.share:.4f})'
        )
    return generate_supervision_bundle_from_frames(
```

Replace:

```python
        index_roles=pl.read_parquet(index_roles_path),
        redirections=pl.read_parquet(redirections_path),
        description_fingerprint=sha256_file(descriptions_path),
```

with:

```python
        index_roles=pl.read_parquet(index_roles_path),
        redirections=pl.read_parquet(redirections_path),
        input_window=input_window,
        description_fingerprint=sha256_file(descriptions_path),
```

In `src/naics_embedder/utils/config.py`, replace:

```python
from naics_embedder.utils.input_window import check_window
```

with:

```python
from naics_embedder.utils.input_window import check_window, trained_window
```

Replace:

```python
    redirections_parquet: str = Field(
        default='./data/naics_redirections.parquet',
        description='The redirections member: the redirection table from `data preprocess`',
    )
    output_root: str = './data/supervision/stage3-supervision-v2'
```

with:

```python
    redirections_parquet: str = Field(
        default='./data/naics_redirections.parquet',
        description='The redirections member: the redirection table from `data preprocess`',
    )
    backbone: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description="The arm's backbone (model.base_model_name); the manifest records its window",
    )
    output_root: str = './data/supervision/stage3-supervision-v2'
```

Replace:

```python
class OutcomePanelConfig(BaseModel):
```

with:

```python
    @field_validator('backbone')
    @classmethod
    def has_a_recorded_window(cls, value: str) -> str:
        '''Refuse a backbone whose trained input window is not recorded (Req 9).'''

        trained_window(value)
        return value

class OutcomePanelConfig(BaseModel):
```

In `conf/data/supervision.yaml`, replace:

```yaml
redirections_parquet: ./data/naics_redirections.parquet
```

with:

```yaml
redirections_parquet: ./data/naics_redirections.parquet
backbone: sentence-transformers/all-MiniLM-L6-v2
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_supervision_schema.py tests/unit/test_config.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1691 passed, 1 skipped` (11 new tests).

- [x] **Step 6: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/utils/config.py tests/fixtures/supervision.py tests/unit/test_supervision_artifacts.py tests/unit/test_supervision_schema.py tests/unit/test_config.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 7: Commit**

```bash
git add src/naics_embedder/supervision/schema.py \
  src/naics_embedder/supervision/artifacts.py \
  src/naics_embedder/data/supervision_bundle.py \
  src/naics_embedder/utils/config.py \
  conf/data/supervision.yaml \
  tests/fixtures/supervision.py \
  tests/unit/test_supervision_artifacts.py \
  tests/unit/test_supervision_schema.py \
  tests/unit/test_config.py
git commit -m "feat(supervision): the manifest records the backbone's input window"
```

### Task 12: Legacy paths and commands build nothing (M6, M8)

Two deferred items from plan 1 fall to this stage, because it reversions the bundle contract
(roadmap, "Deferred items"):

- **M6.** In repaired mode, a legacy `data_loader.streaming` path set to a non-default value is
  ignored rather than rejected. Such a config now fails validation, and the message says where
  repaired training reads structure. Legacy containment keeps its paths.
- **M8.** `data relations`, `data distances` and `data triplets` each build a complete bundle, so
  the old three-step sequence leaves three bundles. Each now prints its notice and exits with
  status 1, building nothing, so a script running the old sequence stops.

The `data preprocess`, `data supervision` and `data all` docstrings also gain the inputs and
outputs that Tasks 3, 10 and 11 added. The build needs three files from `data preprocess`, the
descriptions, index roles and redirection table, plus the backbone's cached tokenizer.

**Files:**
- Modify: `src/naics_embedder/utils/config.py` (`LEGACY_STREAMING_PATHS`,
  `Config.validate_supervision_contract`)
- Modify: `src/naics_embedder/cli/commands/data.py` (docstrings, `_migrate_stage` and the three
  deprecated commands)
- Test: `tests/unit/test_config.py`, `tests/unit/test_cli_commands.py`,
  `tests/unit/test_cli_training.py`

**Interfaces:**
- Consumes: Task 10's required inputs (`index_roles_parquet`, `redirections_parquet`) and Task
  11's `backbone`.
- Produces:
  - `utils.config.LEGACY_STREAMING_PATHS = ('distances_parquet', 'distance_matrix_parquet',
    'relations_parquet', 'triplets_parquet')`. In repaired mode, `Config` raises a `ValueError`
    naming `data_loader.streaming.<name>` when one of them differs from its default.
  - `naics-embedder data relations|distances|triplets` print a notice naming
    `naics-embedder data supervision` and exit with status 1 without building.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_config.py`, replace:

```python
def test_overrides_cannot_reintroduce_legacy_keys_in_repaired_mode():
    with pytest.raises(ValidationError, match='rank_order_weight'):
        Config().override({'loss.rank_order_weight': 0.35})

def test_legacy_containment_is_the_only_mode_accepting_legacy_keys(valid_config_dict):
    valid_config_dict['supervision'] = {'mode': 'legacy_containment'}
    valid_config_dict['loss']['rank_order_weight'] = 0.35
    valid_config_dict['data_loader']['streaming']['phase1_exclusion_weight'] = 100.0

    cfg = Config.model_validate(valid_config_dict)

    assert cfg.supervision.mode == 'legacy_containment'
    assert cfg.loss.rank_order_weight == 0.35
```

with:

```python
def test_overrides_cannot_reintroduce_legacy_keys_in_repaired_mode():
    with pytest.raises(ValidationError, match='rank_order_weight'):
        Config().override({'loss.rank_order_weight': 0.35})

@pytest.mark.parametrize(
    'name',
    ['distances_parquet', 'distance_matrix_parquet', 'relations_parquet', 'triplets_parquet'],
)
def test_repaired_config_rejects_a_legacy_streaming_path(valid_config_dict, name):
    # Repaired training reads structure and training pairs from the bundle only
    valid_config_dict['data_loader']['streaming'][name] = './data/elsewhere'

    with pytest.raises(
        ValidationError, match=f'data_loader.streaming.{name} is a legacy path.*manifest_path'
    ):
        Config.model_validate(valid_config_dict)

def test_overrides_cannot_point_repaired_training_at_a_legacy_path():
    with pytest.raises(ValidationError, match='data_loader.streaming.relations_parquet'):
        Config().override({'data_loader.streaming.relations_parquet': './data/other.parquet'})

def test_legacy_containment_is_the_only_mode_accepting_legacy_keys(valid_config_dict):
    valid_config_dict['supervision'] = {'mode': 'legacy_containment'}
    valid_config_dict['loss']['rank_order_weight'] = 0.35
    valid_config_dict['data_loader']['streaming']['phase1_exclusion_weight'] = 100.0
    valid_config_dict['data_loader']['streaming']['relations_parquet'] = './data/other.parquet'

    cfg = Config.model_validate(valid_config_dict)

    assert cfg.supervision.mode == 'legacy_containment'
    assert cfg.loss.rank_order_weight == 0.35
    assert cfg.data_loader.streaming.relations_parquet == './data/other.parquet'
```

In `tests/unit/test_cli_commands.py`, replace:

```python
@pytest.mark.parametrize('command', ['relations', 'distances', 'triplets'])
def test_legacy_stage_commands_build_the_complete_bundle(monkeypatch, runner, tmp_path, command):
    manifest = tmp_path / 'bundle-id' / 'manifest.json'
    calls = []

    def fake_generate(cfg):
        calls.append(cfg)
        return manifest

    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', fake_generate)

    result = runner.invoke(data_cli.app, [command])

    assert result.exit_code == 0
    assert 'data supervision' in result.output
    assert len(calls) == 1
    assert str(manifest) in result.output
```

with:

```python
@pytest.mark.parametrize('command', ['relations', 'distances', 'triplets'])
def test_legacy_stage_commands_build_nothing(monkeypatch, runner, command):
    calls = []
    monkeypatch.setattr(data_cli, 'generate_supervision_bundle', calls.append)

    result = runner.invoke(data_cli.app, [command])

    # A script running the old three-step sequence stops at its first step
    assert result.exit_code == 1
    assert 'naics-embedder data supervision' in result.output.replace('\n', ' ')
    assert calls == []
```

In `tests/unit/test_cli_training.py`, the fixture's config sets `triplets_parquet`, a legacy path
that repaired training never reads and that would now fail validation. Replace:

```python
        desc_path = tmp_path / 'descriptions.parquet'
        desc_path.write_text('data')
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir(exist_ok=True)
        cfg.data_loader.streaming.descriptions_parquet = str(desc_path)
        cfg.data_loader.streaming.triplets_parquet = str(triplets_dir)
        cfg.supervision.manifest_path = str(tmp_path / 'bundle' / 'manifest.json')
```

with:

```python
        desc_path = tmp_path / 'descriptions.parquet'
        desc_path.write_text('data')
        cfg.data_loader.streaming.descriptions_parquet = str(desc_path)
        cfg.supervision.manifest_path = str(tmp_path / 'bundle' / 'manifest.json')
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_cli_commands.py -q`
Expected: `8 failed, 102 passed`. The four `test_repaired_config_rejects_a_legacy_streaming_path` cases and
`test_overrides_cannot_point_repaired_training_at_a_legacy_path` fail with `DID NOT RAISE`. The
three `test_legacy_stage_commands_build_nothing` cases fail on `assert 0 == 1`.

- [x] **Step 3: Reject a legacy path in repaired mode**

In `src/naics_embedder/utils/config.py`, replace:

```python
class Config(BaseModel):
    '''Main configuration for NAICS training.'''
```

with:

```python
# Streaming paths that only legacy containment reads. Repaired training reads structural facts
# and training pairs from the supervision bundle.
LEGACY_STREAMING_PATHS = (
    'distances_parquet',
    'distance_matrix_parquet',
    'relations_parquet',
    'triplets_parquet',
)

class Config(BaseModel):
    '''Main configuration for NAICS training.'''
```

Replace:

```python
            if self.data_loader.streaming.phase1_exclusion_weight is not None:
                raise ValueError(
                    'data_loader.streaming.phase1_exclusion_weight is invalid in repaired mode; '
                    'an explicit exclusion is never a negative'
                )
        return self
```

with:

```python
            if self.data_loader.streaming.phase1_exclusion_weight is not None:
                raise ValueError(
                    'data_loader.streaming.phase1_exclusion_weight is invalid in repaired mode; '
                    'an explicit exclusion is never a negative'
                )
            for name in LEGACY_STREAMING_PATHS:
                if getattr(self.data_loader.streaming, name) != (
                    StreamingConfig.model_fields[name].default
                ):
                    raise ValueError(
                        f'data_loader.streaming.{name} is a legacy path, which repaired training '
                        'never reads: structural facts and training pairs come from the bundle '
                        'at supervision.manifest_path. Remove the key, or set supervision.mode '
                        'to legacy_containment'
                    )
        return self
```

- [x] **Step 4: Make the deprecated commands build nothing**

In `src/naics_embedder/cli/commands/data.py`, replace:

```python
    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.
        ``data/naics_index_roles.parquet`` - Every index entry with its role.
```

with:

```python
    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.
        ``data/naics_index_roles.parquet`` - Every index entry with its role.
        ``data/naics_redirections.parquet`` - Every cross-reference and harvested "Excluded"
        paragraph, with the codes it names.
```

Replace:

```python
    Computes structural distances and relations in canonical pair orientation, attaches both
    directional exclusion flags, and derives the codebook, pair facts, compatibility
    distance/relation artifacts and matrices, training pairs, and curriculum difficulty
    thresholds from those facts. Every artifact carries the bundle ID and schema version; the
    manifest is written only after all artifacts validate, and an existing bundle is never
    overwritten.

    Requires:
        ``data/naics_descriptions.parquet`` - From the preprocess stage.
```

with:

```python
    Computes structural distances (D*) and relations in canonical pair orientation, attaches
    both directional exclusion flags, and derives the codebook, pair facts, compatibility
    distance/relation artifacts and matrices, training pairs, and curriculum difficulty
    thresholds from those facts. The index roles and the redirection table become members, and
    the manifest records the backbone's input window. Every artifact carries the bundle ID and
    schema version; the manifest is written only after all artifacts validate, and an existing
    bundle is never overwritten.

    Requires:
        ``data/naics_descriptions.parquet``, ``data/naics_index_roles.parquet`` and
        ``data/naics_redirections.parquet`` - From the preprocess stage.
        The backbone's tokenizer in the local Hugging Face cache - To count each channel's texts
        beyond the backbone's trained input window.
```

Replace:

```python
def _migrate_stage(stage: str) -> None:
    typer.echo(
        f'`data {stage}` no longer publishes a standalone artifact; relations, distances, and '
        'triplets are generated together by `naics-embedder data supervision`. Building the '
        'complete supervision bundle now.'
    )
    supervision()

@app.command('relations')
def relations():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('relations')

@app.command('distances')
def distances():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('distances')

@app.command('triplets')
def triplets():
    '''Deprecated: builds the complete supervision bundle (see ``data supervision``).'''

    _migrate_stage('triplets')
```

with:

```python
def _migrate_stage(stage: str) -> None:
    typer.echo(
        f'`data {stage}` builds nothing: relations, distances and training pairs are members of '
        'one supervision bundle, which `naics-embedder data supervision` builds.'
    )
    raise typer.Exit(code=1)

@app.command('relations')
def relations():
    '''Deprecated: builds nothing; ``data supervision`` builds the complete bundle.'''

    _migrate_stage('relations')

@app.command('distances')
def distances():
    '''Deprecated: builds nothing; ``data supervision`` builds the complete bundle.'''

    _migrate_stage('distances')

@app.command('triplets')
def triplets():
    '''Deprecated: builds nothing; ``data supervision`` builds the complete bundle.'''

    _migrate_stage('triplets')
```

Replace:

```python
    Output:
        ``data/naics_descriptions.parquet`` and a new supervision bundle whose manifest path is
        printed.
```

with:

```python
    Output:
        ``data/naics_descriptions.parquet``, ``data/naics_index_roles.parquet``,
        ``data/naics_redirections.parquet`` and a new supervision bundle whose manifest path is
        printed.
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_cli_commands.py tests/unit/test_utils_validation.py tests/unit/test_cli_training.py -q`
Expected: all pass.

Run: `uv run pytest -n auto -q`
Expected: `1696 passed, 1 skipped` (5 new tests; the deprecated-command test is rewritten).

- [x] **Step 6: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/utils/config.py src/naics_embedder/cli/commands/data.py tests/unit/test_config.py tests/unit/test_cli_commands.py tests/unit/test_cli_training.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 7: Commit**

```bash
git add src/naics_embedder/utils/config.py \
  src/naics_embedder/cli/commands/data.py \
  tests/unit/test_config.py \
  tests/unit/test_cli_commands.py \
  tests/unit/test_cli_training.py
git commit -m "fix(supervision): reject legacy streaming paths; deprecated data commands build nothing"
```

### Task 13: Documentation

The docs still describe the v1 contract: the raw tree distance, a reserved exclusion slot with
its rotation, validation that scores exclusions, deprecated commands that build a bundle, and an
optional `index_roles` member. This task rewrites every statement that Tasks 1–12 made false.
It documents D*, the redirection table, unary pairs and the input window, and adds API pages for
the three modules this plan created or grew: `data/redirections.py`, `utils/input_window.py` and
`utils/naics_hierarchy.py`.

PR CI never runs the strict docs build, so this task runs it.

**Files:**
- Create: `docs/api/redirections.md`, `docs/api/input_window.md`, `docs/api/naics_hierarchy.md`
- Modify: `docs/.nav.yml`, `docs/text_training.md`, `docs/usage.md`, `docs/api/config.md`,
  `docs/overview.md`, `docs/hgcn_training.md`, `README.md`, `CLAUDE.md`, `conf/config.yaml`
  (two comments), `src/naics_embedder/text_model/curriculum.py` (one docstring),
  `src/naics_embedder/utils/config.py` (one field description)

**Interfaces:**
- Consumes: everything Tasks 1–12 produced, as described in **Global Constraints**.
- Produces: documentation, one docstring and one field description. No behavior changes, so the
  full suite count stays at 1696.

- [x] **Step 1: Add the API pages and their navigation**

Create `docs/api/redirections.md` with:

```markdown
# Redirections API

The redirection table (Req 8, roadmap Stage 5): every Census cross-reference and harvested
"Excluded" paragraph once, with the codes it names. Each code's exclusion channel is built from
it.

::: naics_embedder.data.redirections
```

Create `docs/api/input_window.md` with:

```markdown
# Input Window API

The backbone's trained input window (Req 9, roadmap Stage 5): every tokenizing path truncates to
it, and the supervision bundle records each channel's texts beyond it.

::: naics_embedder.utils.input_window
```

Create `docs/api/naics_hierarchy.md` with:

```markdown
# NAICS Hierarchy API

Code lineage, the tree metric D* (Req 7) and unary pairs (Req 9). The supervision bundle and
Stage 4's diagnostics read D* from the same function.

::: naics_embedder.utils.naics_hierarchy
```

In `docs/.nav.yml`, replace:

```yaml
          - Create Contrastive Training Triplets: api/create_triplets.md
```

with:

```yaml
          - Create Contrastive Training Triplets: api/create_triplets.md
          - Redirection Table: api/redirections.md
```

Replace:

```yaml
          - Configuration: api/config.md
          - General Utilities: api/utilities.md
```

with:

```yaml
          - Configuration: api/config.md
          - General Utilities: api/utilities.md
          - Input Window: api/input_window.md
          - NAICS Hierarchy: api/naics_hierarchy.md
```

- [x] **Step 2: Update the training guide**

In `docs/text_training.md`, replace:

```markdown
Stage-3 training runs under the **repaired supervision contract** (`stage3-supervision-v1`): one
```

with:

```markdown
Stage-3 training runs under the **repaired supervision contract** (`stage3-supervision-v2`): one
```

Replace:

```markdown
    - [Exactly-One Exclusion Rotation](#exactly-one-exclusion-rotation)
```

with:

```markdown
    - [Negative Selection](#negative-selection)
```

Replace:

```markdown
1. **Generate** — `data supervision` builds a complete bundle in a staging directory, validates
   every artifact, writes the manifest last, and atomically publishes
   `data/supervision/stage3-supervision-v1/<bundle-id>/`. It prints
   `Supervision manifest: <path>`; nothing partial is ever visible under that path. The legacy
   stage commands (`data relations`, `data distances`, `data triplets`) print a migration notice
   and build the same complete bundle.
```

with:

```markdown
1. **Generate** — `data supervision` builds a complete bundle in a staging directory, validates
   every artifact, writes the manifest last, and atomically publishes
   `data/supervision/stage3-supervision-v2/<bundle-id>/`. It reads the descriptions, index roles
   and redirection table that `data preprocess` writes, and the backbone's tokenizer from the
   local Hugging Face cache. It prints `Supervision manifest: <path>`; nothing partial is ever
   visible under that path. The legacy stage commands (`data relations`, `data distances`,
   `data triplets`) print a migration notice and exit with status 1 without building anything.
```

Replace:

```markdown
3. **Validate** — before any DataModule, checkpoint, or model work, `train` re-validates the whole
   bundle (contract version, member hashes and row counts, Parquet contract metadata, codebook
   order and fingerprint, pair-fact coverage and orientation, matrix reconciliation, training-pair
   joins) and checks that `data_loader.streaming.descriptions_parquet` is the file the bundle was
   generated from. `--skip-validation` does not skip this gate.
```

with:

```markdown
3. **Validate** — before any DataModule, checkpoint, or model work, `train` re-validates the whole
   bundle and checks that `data_loader.streaming.descriptions_parquet` is the file the bundle was
   generated from. The bundle checks cover the contract version, member hashes and row counts,
   Parquet contract metadata, and every required validation result. They also cover codebook
   order and fingerprint, pair-fact coverage and orientation, D* and the unary-pair flags,
   matrix reconciliation, training-pair joins, the index roles and the redirection table.
   `--skip-validation` does not skip this gate.
```

Replace:

```markdown
A bundle is immutable and versioned. Its manifest records the contract and per-artifact schema
versions, NAICS vintage, codebook order and fingerprint, input fingerprints, generator revision,
generation parameters, the structural relation-ID mapping, and every member's path, SHA-256,
row count, and exclusion count. Every Parquet member also carries the contract version, bundle ID,
and schema version in its metadata, so artifacts from different bundles can never be mixed.
```

with:

```markdown
A bundle is immutable and versioned. Its manifest records:

- the contract and per-artifact schema versions, and the NAICS vintage;
- the codebook order and fingerprint, the input fingerprints, the generator revision and the
  generation parameters;
- the structural relation-ID mapping;
- every member's path, SHA-256, row count, and exclusion count;
- every validation result the build ran;
- the input window.

Every Parquet member also carries the contract version, bundle ID, and schema version in its
metadata, so artifacts from different bundles can never be mixed.
```

Replace:

```markdown
| `naics_pair_facts.parquet` | One row per unordered code pair: structural distance and relation, both exclusion directions, and their OR |
```

with:

```markdown
| `naics_pair_facts.parquet` | One row per unordered code pair: D* and the structural relation, both exclusion directions and their OR, and the unary-pair flag |
```

Replace:

```markdown
| `naics_training_pairs/` | Training pairs with identities, semantic fields, exclusion provenance, and raw structure |
| `curriculum_difficulty_thresholds.json` | Curriculum thresholds derived from the same bundle |
```

with:

```markdown
| `naics_training_pairs/` | Training pairs with identities, semantic fields, exclusion provenance, and raw structure; no negative is an explicit exclusion and no positive is a unary pair |
| `curriculum_difficulty_thresholds.json` | Curriculum thresholds derived from the same bundle |
| `naics_index_roles.parquet` | Every Census index entry with its one role: examples-channel text, or a training, validation or test query |
| `naics_redirections.parquet` | Every cross-reference and harvested "Excluded" paragraph once: its code and text, activity phrase, named codes, lineal codes and withheld flag |
```

Replace:

```markdown
`expected supervision contract stage3-supervision-v1, found <version> in <path>`,
`a direct positive is an explicit exclusion`, or
`structural relation fields contain an exclusion sentinel`. Regenerate the bundle rather than
editing members.
```

with:

```markdown
`expected supervision contract stage3-supervision-v2, found <version> in <path>`,
`a direct positive is an explicit exclusion`, or
`structural relation fields contain an exclusion sentinel`. Regenerate the bundle rather than
editing members.

**D\*.** The pair facts carry D*, the tree path length through a virtual root above the 20
sectors (Req 7): `depth_i + depth_j - 2 * depth_LCA`. Sectors sit at depth 1, and the combined
sectors 31-33, 44-45 and 48-49 count as one. There is no half-step and no cross-sector constant:
across sectors D* is λ(i) + λ(j) − 2, where λ is the number of digits. The build checks every
pair against `utils/naics_hierarchy.tree_distance_matrix`, the function Stage 4's diagnostics
read, and checks the triangle inequality over all triples. Cross-sector pairs are found by their
relation label (`cross_sector`, relation ID 99), never by a distance.

**Redirections.** A cross-reference reroutes an activity; it does not assert that two codes are
unrelated (Req 8). The redirection table holds every cross-reference row and every "Excluded"
paragraph harvested from a description, once each. A code's exclusion channel is its rows' text,
each row once, in table order. A held-out query that leaks into a row withholds it: the row
stays in the table and its named codes stay exclusions, but its text leaves the channel and its
activity phrase is dropped. The build's leakage check also reads the activity phrases, which
Stage 7 trains on as queries. A lineal reference, a code naming its own ancestor or descendant,
stays text only.

**Unary pairs.** A five-digit code whose only child is its six-digit code forms a unary pair
(Req 9). The pair facts flag the 522 unary pairs. They are never generated or sampled
positives, and parent retrieval never scores them.

**Input window.** The manifest's `input_window` records the backbone's trained window: 128 tokens
for `sentence-transformers/all-MiniLM-L6-v2`, from its model card. Per text channel it records
the present texts, the texts beyond the window and their share. Every tokenizing path truncates
to the window (`utils/input_window.py`). An absent channel is null in the descriptions, and the
tokenization cache encodes it as the empty string, never as a placeholder.
```

Replace:

```markdown
- **Structure** — the raw NAICS tree distance and relation (`cross_sector` = 99 across sectors).
  Exclusion processing never alters these values.
```

with:

```markdown
- **Structure** — D* and the NAICS relation (`cross_sector`, relation ID 99, across sectors).
  Exclusion processing never alters these values.
```

Replace the whole of the section from its `### Exactly-One Exclusion Rotation` line through its
last line, `or attract an explicit exclusion.`, with:

```markdown
### Negative Selection

An explicit exclusion is never a negative (Req 8(c)). The generator drops every candidate that is
an explicit exclusion of its anchor, the canonical pool never admits one, and final selection
refuses one. No slot is reserved for exclusions. The `K` slots come from strategy proposals, with
duplicates removed by code (keeping the smallest UID), ties broken by code ID then UID, and a
deterministic backfill. Proposals are consulted in order:

1. **Phase 2+ miners.** With hard-negative mining on, the geometric miner proposes its share of the
   `K` slots, `K - int(K * router_mix_ratio)`; with router-guided mining also on, the router fills
   the rest. `router_mix_ratio` comes from `curriculum.anneal` (default 0.5). Miners score one
   occurrence per code (the smallest candidate UID, which the coordinator keeps) and never the
   anchor or positive code, so on multiple GPUs, where a code repeats across rows and ranks of the
   global pool, the miners still fill their slots with distinct codes.
2. **The difficulty proposal** from the data layer, which is the only proposal in Phase 1 and the
   fallback afterwards.
3. **Deterministic backfill** from the remaining eligible codes.

A candidate is **eligible** only if it is not an explicit exclusion of the anchor and is
structurally farther from the anchor than the positive. That is the rule every generated
training negative satisfies, including its cross-sector and equal-distance special cases.
Candidates sourced at runtime, such as universe backfill and the multi-GPU global pool, therefore
never repel a relative that the generated supervision would not treat as a negative. The
repaired configuration rejects the legacy `phase1_exclusion_weight`. No exclusion is ever
selected, so none reaches the contrastive denominator.
```

Replace:

```markdown
`train/integrity/anchors_with_exclusions`, the per-reason selections (`quota_selections`,
`geometric_selections`, `router_selections`, `difficulty_selections`, `deterministic_backfills`),
`invalid_candidates_ignored` (padding), `structurally_ineligible_candidates`, and
`duplicate_candidates_removed`.

Validation scores every eligible candidate of each validation pool, including every exclusion,
```

with:

```markdown
`train/integrity/anchors_with_exclusions`, the per-reason selections (`geometric_selections`,
`router_selections`, `difficulty_selections`, `deterministic_backfills`, and `quota_selections`,
which stays at zero because no slot is reserved), `invalid_candidates_ignored` (padding),
`structurally_ineligible_candidates`, and `duplicate_candidates_removed`.

Validation scores every eligible candidate of each validation pool, never an explicit exclusion,
```

Replace:

```markdown
- Build one canonical candidate pool per (anchor, positive) from the bundle's training pairs: all
  of the anchor's explicit exclusions plus unique ordinary codes, never the anchor or positive.
- Phase 1 sampling:
  - Inverse tree-distance weighting (`P(n) ∝ 1 / d_tree(a, n)^α`).
  - Sibling masking (`d_tree <= 2` set to zero).
  - Difficulty proposals over the pool (explicit exclusions are represented by the selection quota,
    not by sampling weight).
```

with:

```markdown
- Build one canonical candidate pool per (anchor, positive) from the bundle's training pairs:
  unique codes structurally farther from the anchor than the positive, never an explicit
  exclusion of the anchor, the anchor or the positive.
- Phase 1 sampling:
  - Inverse tree-distance weighting over D* (`P(n) ∝ 1 / d_tree(a, n)^α`).
  - Sibling masking (`d_tree == 2` set to zero, which under D* also masks a grandparent or
    grandchild).
  - Difficulty proposals over the pool.
```

- [x] **Step 3: Update the other pages, CLAUDE.md and two config comments**

In `docs/usage.md`, replace:

```markdown
entries are outcome-panel queries. Preprocessing fails if any validation or test query matches
training text.

**Requires:** `conf/data/index_roles.csv`  
**Generates:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`
```

with:

```markdown
entries are outcome-panel queries. It also writes the redirection table: every cross-reference
row and harvested "Excluded" paragraph once, with the codes it names. Each code's exclusion
channel is built from that table, and a row that a held-out query leaks into is withheld from
the channel. A code without official description text inherits its only child's, and
`description_source` records whose text it is. An absent channel is null. Preprocessing fails if
any validation or test query matches training text or an activity phrase.

**Requires:** `conf/data/index_roles.csv`  
**Generates:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`,
`data/naics_redirections.parquet`
```

Replace:

```markdown
Build one immutable, validated Stage-3 supervision bundle: the codebook, pair facts (structural
distance and relation plus directional explicit exclusions), legacy-compatible distance and
relation views and matrices, training pairs, and curriculum difficulty thresholds. The manifest is
written last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet` (carried as the
bundle's optional `index_roles` member)  
**Generates:** `data/supervision/stage3-supervision-v1/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)
```

with:

```markdown
Build one immutable, validated Stage-3 supervision bundle. It holds the codebook, the pair facts
(D*, the structural relation, directional explicit exclusions and the unary-pair flag),
legacy-compatible distance and relation views and matrices, and training pairs. It also holds
the curriculum difficulty thresholds, the index roles and the redirection table. The manifest
records the backbone's input window and each channel's texts beyond it. The manifest is written
last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`,
`data/naics_redirections.parquet` (the last two become required bundle members), and the
backbone's tokenizer in the local Hugging Face cache  
**Generates:** `data/supervision/stage3-supervision-v2/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)
```

Replace:

```markdown
Deprecated stage commands. Publishing one partial authority would let artifacts from different
generations mix, so each prints a migration notice and builds the complete supervision bundle
(same as `data supervision`).
```

with:

```markdown
Deprecated stage commands. Publishing one partial authority would let artifacts from different
generations mix, so each prints a migration notice and exits with status 1 without building
anything; `data supervision` builds the complete bundle.
```

Replace:

```markdown
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). The regressor panel reduces the table to the arm's dimension
by PCA.
```

with:

```markdown
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). Each channel is truncated to the backbone's trained input
window, and `text_only.max_length` may not exceed it (Req 9). The regressor panel reduces the
table to the arm's dimension by PCA.
```

In `docs/api/config.md`, replace:

```markdown
  contract_version: stage3-supervision-v1
```

with:

```markdown
  contract_version: stage3-supervision-v2
```

Replace:

```markdown
- Repaired configurations reject `loss.rank_order_weight` (a legacy LambdaRank setting; configure
  `loss.structural_preference` instead) and `data_loader.streaming.phase1_exclusion_weight` (the
  one-slot exclusion quota owns exclusion representation). The old key is never reinterpreted as
  the new loss because the objectives differ.
- Both legacy keys are accepted only with the explicit `supervision.mode: legacy_containment`.
```

with:

```markdown
- Repaired configurations reject `loss.rank_order_weight` (a legacy LambdaRank setting; configure
  `loss.structural_preference` instead) and `data_loader.streaming.phase1_exclusion_weight` (an
  explicit exclusion is never a negative). The old key is never reinterpreted as the new loss
  because the objectives differ.
- Repaired configurations also reject a legacy streaming path (`distances_parquet`,
  `distance_matrix_parquet`, `relations_parquet`, `triplets_parquet`) set to anything but its
  default: repaired training reads structural facts and training pairs from the bundle.
- The legacy keys and paths are accepted only with the explicit
  `supervision.mode: legacy_containment`.
```

In `docs/overview.md`, replace:

```markdown
- **Data Layer (Streaming Dataset):** Builds candidate pools, applies Phase 1 inverse tree-distance weighting, masks siblings, and prioritizes explicit exclusions. Negatives carry `explicit_exclusion` flags for downstream logging.
```

with:

```markdown
- **Data Layer (Streaming Dataset):** Builds candidate pools that never admit an explicit exclusion of the anchor (Req 8(c)), applies Phase 1 inverse tree-distance weighting over D*, and masks siblings.
```

In `docs/hgcn_training.md`, replace:

```markdown
  --codebook data/supervision/stage3-supervision-v1/<bundle-id>/naics_codebook.parquet
```

with:

```markdown
  --codebook data/supervision/stage3-supervision-v2/<bundle-id>/naics_codebook.parquet
```

In `README.md`, replace:

```markdown
  --codebook data/supervision/stage3-supervision-v1/<bundle-id>/naics_codebook.parquet
```

with:

```markdown
  --codebook data/supervision/stage3-supervision-v2/<bundle-id>/naics_codebook.parquet
```

Replace:

```markdown
`data supervision` prints `Supervision manifest: <path>`. The former `data relations`,
`data distances`, and `data triplets` commands now print a migration notice and build the same
bundle.
```

with:

```markdown
`data supervision` prints `Supervision manifest: <path>`. The former `data relations`,
`data distances`, and `data triplets` commands now print a migration notice and exit with status
1 without building anything.
```

In `CLAUDE.md`, replace:

```text
│   │   ├── download_data.py  # Download and preprocess NAICS data
```

with:

```text
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── redirections.py   # The redirection table (Req 8) and the exclusion channel
```

Replace:

```text
│       ├── hyperbolic.py     # LorentzManifold, CurvatureManager, ManifoldAdapter
```

with:

```text
│       ├── hyperbolic.py     # LorentzManifold, CurvatureManager, ManifoldAdapter
│       ├── input_window.py   # The backbone's trained input window (Req 9)
│       ├── naics_hierarchy.py  # Code lineage, the tree metric D* (Req 7), unary pairs
```

Replace:

```text
# (data relations / distances / triplets are deprecated and build the same bundle)
```

with:

```text
# (data relations / distances / triplets are deprecated and build nothing)
```

Replace:

```text
- Leverages tree distance and exclusion relationships
```

with:

```text
- Leverages the tree metric D*; an explicit exclusion is never a negative (Req 8)
```

In `src/naics_embedder/text_model/curriculum.py`, replace:

```python
    - Data layer (streaming_dataset.py): performs Phase 1 tree-distance weighting,
      sibling masking, and explicit exclusion mining before batches reach the model.
```

with:

```python
    - Data layer (streaming_dataset.py): performs Phase 1 tree-distance weighting and
      sibling masking before batches reach the model. Its pools never hold an explicit
      exclusion (Req 8).
```

In `src/naics_embedder/utils/config.py`, replace:

```python
            '(inverse weighting, sibling masking, exclusion mining)'
```

with:

```python
            '(inverse weighting and sibling masking)'
```

In `conf/config.yaml`, replace:

```yaml
    use_phase1_sampling: true  # Use inverse weighting, sibling masking, exclusion mining
```

with:

```yaml
    use_phase1_sampling: true  # Use inverse weighting and sibling masking
```

Replace:

```yaml
    # Exclusions are represented by the one-slot exclusion quota at selection time
```

with:

```yaml
    # An explicit exclusion is never a negative: pools and selection leave it out (Req 8(c))
```

- [x] **Step 4: Build the docs and check for stale statements**

Run: `uv run mkdocs build --strict -d /tmp/stage5-docs-site`
Expected: exit 0 with no `WARNING` lines.

Run: `rm -rf /tmp/stage5-docs-site`

Run: `git grep -n -e 'stage3-supervision-v1' -e 'Exactly-One' -e 'exclusion quota' -e 'build the same' -e 'including every exclusion' -e 'exclusion mining' -e 'exclusion relationships' -- docs README.md CLAUDE.md conf src`
Expected: no output.

Run: `./scripts/format_code.sh --check src/naics_embedder/text_model/curriculum.py src/naics_embedder/utils/config.py`
Expected: `Clean: no lint issues and no formatting changes.`

Run: `uv run pytest -n auto -q`
Expected: `1696 passed, 1 skipped`.

- [x] **Step 5: Commit**

```bash
git add docs/api/redirections.md \
  docs/api/input_window.md \
  docs/api/naics_hierarchy.md \
  docs/.nav.yml \
  docs/text_training.md \
  docs/usage.md \
  docs/api/config.md \
  docs/overview.md \
  docs/hgcn_training.md \
  README.md \
  CLAUDE.md \
  conf/config.yaml \
  src/naics_embedder/text_model/curriculum.py \
  src/naics_embedder/utils/config.py
git commit -m "docs: document the stage3-supervision-v2 bundle, D*, redirections and the window"
```

### Task 14: The real bundle and the finding (controller, inline)

This task builds the bundle once from the four Census files, inside this worktree, and recounts
it independently of the build. It writes `specs/findings/supervision-target-and-text.md` and
commits only that file: `data/` is gitignored. No panel is read and no sealed split is opened. If
a number differs from **Expected real-data results**, or a command fails a check, stop and ask.

**Files:**
- Create: `specs/findings/supervision-target-and-text.md`
- Scratch: `/tmp/stage5-supervision-c3a0baa0/check_bundle.py`
- Outputs, gitignored: `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`,
  `data/naics_redirections.parquet`, `data/supervision/stage3-supervision-v2/<bundle-id>/`

**Interfaces:**
- Consumes: Tasks 1–13; `load_validated_bundle` (`supervision/artifacts.py`),
  `tree_distance_matrix` as `metrics/diagnostics.py` imports it, and `unary_pairs`
  (`utils/naics_hierarchy.py`).
- Produces: the real bundle, whose ID Plan completion quotes, and the finding.

- [x] **Step 1: Check the workspace**

Run: `pwd`
Expected: `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text`.
If it prints anything else, `cd` there first.

Run: `ls data`
Expected: no `naics_*.parquet` file and no `supervision` directory. An empty `streaming_cache`
directory that tests leave is fine, and so is `ls` reporting that `data` does not exist:
`data preprocess` creates it.

Run: `mkdir -p /tmp/stage5-supervision-c3a0baa0`

- [x] **Step 2: Preprocess the Census files**

Run: `COLUMNS=200 uv run naics-embedder data preprocess --source-dir ~/Downloads/Data`
Expected: about 25 s, ending `Preprocessing complete.` Among the lines it prints:

```text
Redirection table:
  Cross-reference rows:  4,601
  Excluded paragraphs from descriptions:  22
  Rows naming a code:  4,555
  Activity phrases:  4,529
  Lineal references:  9
  Withheld rows: [653, 849, 1702, 1920, 3279]
```

```text
NAICS descriptions:
  Official:  1,449
  Inherited from an only child:  662
  Absent (several children, no official text):  14

Present texts per channel: {'title': 2125, 'description': 2111, 'examples': 1075, 'excluded': 1117}
Held-out queries matching training text or activity phrases: {'validation': {'exact': 0, 'near_duplicate': 0}, 'test': {'exact': 0, 'near_duplicate': 0}}
```

It writes 2,125 codes, 20,373 index entries and 4,623 redirection rows. The command reads the
local files by name and downloads nothing, although its log says "Downloaded NAICS files
successfully".

Run: `shasum -a 256 data/naics_descriptions.parquet data/naics_index_roles.parquet data/naics_redirections.parquet`
Expected:

```text
fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36  data/naics_descriptions.parquet
b5a1221b6a8a3413a9d8126b7e40b5c04cebbd2f900ae2dc62801dda58c993b3  data/naics_index_roles.parquet
69b455212583f7db5992e4d4cc65c871aa785118b1309a798f4d5d77564b9149  data/naics_redirections.parquet
```

Two independent runs produced these bytes while this plan was written: under the pinned
versions the three files are deterministic.

- [x] **Step 3: Build the bundle**

Run: `COLUMNS=200 HF_HUB_OFFLINE=1 uv run naics-embedder data supervision`
Expected: about 50 s, at a peak of about 7 GB of memory. It prints
`D* for 2,256,750 pairs of 2,125 codes` and one line per sector, then:

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (587 > 512). Running this sequence through the model will result in indexing errors
Input window: 128 tokens (sentence-transformers/all-MiniLM-L6-v2)
  title: 0 of 2,125 texts beyond it (0.0000)
  description: 153 of 2,111 texts beyond it (0.0725)
  examples: 105 of 1,075 texts beyond it (0.0977)
  excluded: 464 of 1,117 texts beyond it (0.4154)
Computing difficulty thresholds...
  • Distance range: [1.00, 10.00]
  • Distance thresholds: P1<=7.00, P2<=9.00, P3<=10.00
  • Margin range: [0.1035, 1.8001]
  • Margin thresholds: P1>=0.1579, P2>=0.1304, P3>=0.1200
```

The warning is expected. The window record counts each text's tokens without truncation, and
the tokenizer warns once when a text exceeds the backbone's 512 positions. Nothing runs the
model on it: every tokenizing path truncates to 128.

The output ends with:

```text
Published supervision bundle <bundle-id>: 2,256,750 pair facts (3,954 explicit exclusions), 45,163,632 training pairs
Supervision manifest: data/supervision/stage3-supervision-v2/<bundle-id>/manifest.json
```

`<bundle-id>` is a fresh UUID. Note it: the finding and Plan completion quote it.

- [x] **Step 4: Recount the bundle**

Create `/tmp/stage5-supervision-c3a0baa0/check_bundle.py` with the Write tool:

```python
'''Plan 7, Task 14: recount the real bundle and descriptions, independently of the build.'''

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from naics_embedder.metrics.diagnostics import tree_distance_matrix
from naics_embedder.supervision.artifacts import load_validated_bundle
from naics_embedder.utils.naics_hierarchy import unary_pairs

manifests = sorted(Path('data/supervision/stage3-supervision-v2').glob('*/manifest.json'))
if len(manifests) != 1:
    raise SystemExit(f'expected one v2 bundle, found {len(manifests)}')
bundle = load_validated_bundle(manifests[0])
manifest = bundle.manifest
results = manifest.validation_results
print('bundle:', manifest.contract_version, len(results), 'results, all passed:', all(results.values()))

codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
codes = codebook.get_column('code').to_list()
ids = {code: i for i, code in enumerate(codes)}
print('codebook:', len(codes), dict(sorted(Counter(len(code) for code in codes).items())))

facts = pl.read_parquet(bundle.artifact_path('pair_facts'))
cross = facts.filter(pl.col('structural_relation_name') == 'cross_sector')
relation_ids = cross.get_column('structural_relation_id').unique().to_list()
print('pair facts:', facts.height, 'cross-sector', cross.height, relation_ids, 'within', facts.height - cross.height)
dstar = facts.get_column('structural_distance').to_numpy()
print('D*:', dict(sorted(Counter(int(d) for d in dstar).items())))
print('D* integer, no 99:', bool(np.all(dstar == np.round(dstar))), bool(np.all(dstar != 99)))
lam = cross.get_column('code_i').str.len_chars() + cross.get_column('code_j').str.len_chars() - 2
print('cross-sector D* = λ(i) + λ(j) − 2:', bool((lam.cast(pl.Float32) == cross.get_column('structural_distance')).all()))
rows = facts.get_column('code_i_id').to_numpy()
cols = facts.get_column('code_j_id').to_numpy()
print('D* equals metrics.diagnostics on every pair:', bool(np.array_equal(tree_distance_matrix(codes)[rows, cols], dstar)))
names = dict(sorted(Counter(facts.get_column('structural_relation_name').to_list()).items()))
print('relation names:', names)

flagged = {frozenset(pair) for pair in facts.filter('unary_pair').select('code_i', 'code_j').rows()}
print('unary pairs:', len(flagged), 'equal unary_pairs(codes):', flagged == {frozenset(pair) for pair in unary_pairs(codes)})
directed = facts.get_column('code_i_excludes_code_j').sum() + facts.get_column('code_j_excludes_code_i').sum()
print('exclusions:', facts.get_column('is_explicit_exclusion').sum(), 'pairs,', directed, 'directed')

pairs = pl.scan_parquet([str(path) for path in bundle.member_paths('training_pairs')])
counts = pairs.select(
    pl.len().alias('rows'),
    pl.col('anchor_code_id').n_unique().alias('anchors'),
    pl.col('negative_is_explicit_exclusion').sum().alias('exclusion_negatives'),
    pl.col('positive_is_explicit_exclusion').sum().alias('exclusion_positives'),
).collect().row(0, named=True)
print('training pairs:', counts)
excluded = facts.filter('is_explicit_exclusion').select(pl.col('code_i_id').alias('a'), pl.col('code_j_id').alias('b'))
excluded = pl.concat([excluded, excluded.select(pl.col('b').alias('a'), pl.col('a').alias('b'))])
negatives = pairs.select(pl.col('anchor_code_id').alias('a'), pl.col('negative_code_id').alias('b')).unique().collect()
positives = pairs.select(pl.col('anchor_code_id').alias('a'), pl.col('positive_code_id').alias('b')).unique().collect()
unary = pl.DataFrame([(ids[p], ids[c]) for p, c in unary_pairs(codes)] + [(ids[c], ids[p]) for p, c in unary_pairs(codes)], schema={'a': pl.Int32, 'b': pl.Int32}, orient='row')
print('negatives that are exclusions:', negatives.join(excluded, on=['a', 'b']).height)
print('positives that are unary pairs:', positives.join(unary, on=['a', 'b']).height, 'of', positives.height, 'distinct positives')

redirections = pl.read_parquet(bundle.artifact_path('redirections'))
xref = redirections.filter(pl.col('source') == 'cross_reference').with_columns(
    reads=pl.col('text').str.contains('are classified in'),
    names=pl.col('named_codes').list.len() > 0,
)
print('redirections:', redirections.height, dict(sorted(Counter(redirections.get_column('source').to_list()).items())))
print('cross-reference rows (reads "are classified in", names a code):', sorted(xref.group_by('reads', 'names').len().rows()))
activities = redirections.get_column('activity').drop_nulls()
print('activity phrases:', activities.len(), 'distinct', activities.n_unique())
print('withheld:', redirections.filter('withheld').select('reference_id', 'code').rows())
lineal = redirections.filter(pl.col('lineal_codes').list.len() > 0).select('code', 'lineal_codes').explode('lineal_codes')
print('lineal:', sorted(lineal.unique().rows()))
lineal_ids = pl.DataFrame([(ids[a], ids[b]) for a, b in lineal.rows()], schema={'a': pl.Int32, 'b': pl.Int32}, orient='row')
lineal_ids = pl.concat([lineal_ids, lineal_ids.select(pl.col('b').alias('a'), pl.col('a').alias('b'))])
print('lineal pairs used as negatives:', negatives.join(lineal_ids, on=['a', 'b']).height)
named = set(redirections.get_column('named_codes').explode().drop_nulls().to_list())
print('named codes outside the codebook:', len(named - set(codes)))

roles = pl.read_parquet(bundle.artifact_path('index_roles'))
print('index roles:', roles.height, dict(sorted(Counter(roles.get_column('role').to_list()).items())))

descriptions = pl.read_parquet('data/naics_descriptions.parquet')
channels = ['title', 'description', 'examples', 'excluded']
print('present:', {channel: descriptions.get_column(channel).drop_nulls().len() for channel in channels})
blank = sum(descriptions.filter(pl.col(c).str.strip_chars().eq('') | pl.col(c).str.contains('[EMPTY]', literal=True)).height for c in channels)
print('blank or placeholder texts:', blank)
source = descriptions.select('code', 'description_source')
official = source.filter(pl.col('description_source') == pl.col('code')).height
inherited = source.filter(pl.col('description_source').is_not_null() & (pl.col('description_source') != pl.col('code')))
levels = dict(sorted(Counter(len(code) for code in inherited.get_column('code')).items()))
descends = inherited.filter(pl.col('description_source').str.starts_with(pl.col('code'))).height
print('descriptions:', official, 'official,', inherited.height, 'inherited', levels, 'from a descendant', descends)
print('null descriptions:', source.filter(pl.col('description_source').is_null()).get_column('code').to_list())

print('window:', manifest.input_window.model_dump())
thresholds = json.loads(bundle.artifact_path('difficulty_thresholds').read_text())
print('thresholds:', {key: thresholds[key] for key in sorted(thresholds) if key.endswith('max_distance')})
```

The script re-derives what the build's validators already asserted, from the bundle's own
members and the descriptions, through `metrics/diagnostics.py`'s import of D*: an independent
count, not a re-run of the build's code path.

Run: `uv run python /tmp/stage5-supervision-c3a0baa0/check_bundle.py`
Expected: about 7 s, at a peak of about 5 GB, printing exactly:

```text
bundle: stage3-supervision-v2 27 results, all passed: True
codebook: 2125 {2: 20, 3: 96, 4: 308, 5: 689, 6: 1012}
pair facts: 2256750 cross-sector 1984647 [99] within 272103
D*: {1: 2105, 2: 4593, 3: 10938, 4: 30175, 5: 78544, 6: 183454, 7: 347006, 8: 549103, 9: 613307, 10: 437525}
D* integer, no 99: True True
cross-sector D* = λ(i) + λ(j) − 2: True
D* equals metrics.diagnostics on every pair: True
relation names: {'child': 2105, 'cousin': 9583, 'cousin_1_times_removed': 29503, 'cousin_2_times_removed': 34434, 'cross_sector': 1984647, 'grand-grand-nephew/niece': 9338, 'grand-nephew/niece': 9561, 'grandchild': 2009, 'great-grandchild': 1701, 'great-great-grandchild': 1012, 'nephew/niece': 7413, 'second_cousin': 27807, 'second_cousin_1_times_removed': 70835, 'sibling': 2394, 'third_cousin': 64408}
unary pairs: 522 equal unary_pairs(codes): True
exclusions: 3954 pairs, 4586 directed
training pairs: {'rows': 45163632, 'anchors': 2090, 'exclusion_negatives': 0, 'exclusion_positives': 0}
negatives that are exclusions: 0
positives that are unary pairs: 0 of 268831 distinct positives
redirections: 4623 {'cross_reference': 4601, 'description': 22}
cross-reference rows (reads "are classified in", names a code): [(False, False, 37), (False, True, 6), (True, False, 31), (True, True, 4527)]
activity phrases: 4529 distinct 3326
withheld: [(653, '311830'), (849, '321219'), (1702, '333994'), (1920, '336110'), (3279, '525920')]
lineal: [('111191', '1111'), ('111336', '1113'), ('211120', '2111'), ('211130', '2111'), ('32111', '321'), ('321114', '321'), ('424410', '42'), ('488490', '48'), ('711', '7113')]
lineal pairs used as negatives: 0
named codes outside the codebook: 0
index roles: 20373 {'examples': 6118, 'test': 3013, 'training': 7200, 'validation': 4042}
present: {'title': 2125, 'description': 2111, 'examples': 1075, 'excluded': 1117}
blank or placeholder texts: 0
descriptions: 1449 official, 662 inherited {4: 140, 5: 522} from a descendant 662
null descriptions: ['2111', '3231', '3241', '4561', '4931', '5192', '7121', '9211', '9221', '9231', '9241', '9251', '9261', '9281']
window: {'backbone': 'sentence-transformers/all-MiniLM-L6-v2', 'window': 128, 'channels': {'title': {'present': 2125, 'over': 0, 'share': 0.0}, 'description': {'present': 2111, 'over': 153, 'share': 0.07247749881572714}, 'examples': {'present': 1075, 'over': 105, 'share': 0.09767441860465116}, 'excluded': {'present': 1117, 'over': 464, 'share': 0.41539838854073413}}}
thresholds: {'phase1_max_distance': 7.0, 'phase2_max_distance': 9.0, 'phase3_max_distance': 10.0, 'phase4_max_distance': 10.0}
```

- [x] **Step 5: Compare with Expected real-data results**

Compare every number of Steps 2–4 with **Expected real-data results** and with the blocks above.
They must match exactly: the computation is deterministic under the pinned versions. If any
differs, stop and ask. Keep the bundle, and keep the scratch directory until Final verification
Step 8.

- [x] **Step 6: Write the finding**

> Deviation: the bundle ID lengthened two of the finding's lines past 100 columns, the opening list's bundle bullet and section 6's "The bundle." bullet; both were reflowed.

Create `specs/findings/supervision-target-and-text.md` with the content below. Replace every
`<bundle-id>` with the ID Step 3 printed, and `YYYY-MM-DD` with today's date.

````markdown
# Supervision target and text: finding

**Status: FINAL (YYYY-MM-DD).** Roadmap Stage 5 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 7
(`specs/plans/completed/7-supervision-target-and-text.md`):

- the preprocessed descriptions, index-entry roles and redirection table
- bundle `<bundle-id>` under contract `stage3-supervision-v2`, the first bundle carrying the
  `index_roles` and `redirections` members
- the backbone's trained input window and each channel's overflow share

No panel was read and no sealed split was opened. Section 6 lists what later stages read.

## Sources

The four Census 2022 NAICS files were read from `~/Downloads/Data` (`data preprocess
--source-dir`), and nothing was downloaded. The backbone's tokenizer came from the local Hugging
Face cache at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, under `HF_HUB_OFFLINE=1`. The
run used Python 3.12 with numpy 2.3.4, polars 1.35.1, scipy 1.16.3, scikit-learn 1.9.1, torch
2.9.1 and transformers 4.57.1.

| File | sha256 |
|---|---|
| `2-6 digit_2022_Codes.xlsx` | `be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316` |
| `2022_NAICS_Descriptions.xlsx` | `6222c4d87dcf984970e0ff8a49862ed54b546b089d03956be3f285900cd3d66c` |
| `2022_NAICS_Index_File.xlsx` | `6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63` |
| `2022_NAICS_Cross_References.xlsx` | `3c50c3bfa9d76862aea471cc8831fd726f0b978619d833e48c54a749d9622144` |
| `conf/data/index_roles.csv` | `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a` |
| `data/naics_descriptions.parquet` (output) | `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36` |
| `data/naics_index_roles.parquet` (output) | `b5a1221b6a8a3413a9d8126b7e40b5c04cebbd2f900ae2dc62801dda58c993b3` |
| `data/naics_redirections.parquet` (output) | `69b455212583f7db5992e4d4cc65c871aa785118b1309a798f4d5d77564b9149` |

The three outputs are byte-identical across runs. The bundle's members are not, because each
carries its bundle ID, so the bundle is identified by its ID, not by a hash.

## 1. The target, D*

The bundle's 2,256,750 pair facts cover every unordered pair of the 2,125 codes (20 sectors, 96
three-digit, 308 four-digit, 689 five-digit and 1,012 six-digit). Their distance is D*, the path
length through a virtual root above the sectors, with the combined sectors 31-33, 44-45 and 48-49
counted as one:

| D* | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pairs | 2,105 | 4,593 | 10,938 | 30,175 | 78,544 | 183,454 | 347,006 | 549,103 | 613,307 | 437,525 |

- **No half-step and no 99.** Every value is an integer from 1 to 10.
- **Across sectors.** The 1,984,647 cross-sector pairs have D* = λ(i) + λ(j) − 2 and relation ID
  99, `cross_sector`, which no within-sector pair carries. The other 272,103 pairs are within a
  sector.
- **A tree metric.** The build checked the triangle inequality over all 9,595,703,125 ordered
  triples (2,125³), and that every distance equals the path length through the pair's lowest
  common ancestor. The recount found every pair equal to the D* that `metrics/diagnostics.py`
  computes from the codes' own lineage.
- **Relation names.** Fifteen labels remain, from `child` (2,105 pairs) to `cross_sector`, as
  edge types and diagnostic labels only (D5). The relation margin axis keeps its cross-sector
  margin until Stage 7.
- **Curriculum thresholds.** The 20th, 60th and 90th distance percentiles are 7, 9 and 10, where
  every threshold was 99 before. HGCN's curriculum therefore limits phase-1 negatives to D* ≤ 7
  and phase-2 negatives to D* ≤ 9 until Stage 11 removes the thresholds.

## 2. Redirections and exclusions

The redirection table has 4,623 rows: the 4,601 cross-reference rows in file order (reference
IDs 0–4600), then 22 "Excluded" paragraphs harvested from descriptions, in code order. Every
row appears once. A code's exclusion channel is its rows' text in table order, and 1,117 codes
have one.

**The 43 and the 68.** The spec counts 4,558 rows that read "…are classified in" and so 43 that
do not. The old pipeline dropped 68 rows without a code reference. Both counts hold, and they
measure different things:

| Cross-reference rows | Name a codebook code | Name none | Total |
|---|---|---|---|
| Read "are classified in" | 4,527 | 31 | 4,558 |
| Other wording | 6 | 37 | 43 |
| Total | 4,533 | 68 | 4,601 |

- The 31 rows name their destination only in words, for example "are classified in the
  Manufacturing sector according to the products made".
- The 6 name a code with other wording, such as "are classified elsewhere in Subsector 321".
  One reads "are lclassified in Industry 423730", a typo in the Census file.
- The 68 codeless rows carry no digits at all. They stay in the table with no named code and no
  activity phrase. All but row 3279, which is withheld, stay in their code's exclusion channel.

**Activity phrases.** 4,529 rows carry one, 3,326 of them distinct: the text before `--`, or
before " are/is classified" or "included", on a cross-reference row that names a code. Stage 7
trains on them as queries, so the build's held-out leakage check reads them too. It found no
exact and no near-duplicate match for any validation or test query.

**Withheld rows.** Held-out queries leak into five rows (user decision 2): `reference_id` 653
(311830), 849 (321219), 1702 (333994) and 1920 (336110), whose activity phrases are
near-duplicates of held-out queries, and 3279 (525920), which contains a test query exactly.
They stay in the table with `withheld` set and no activity phrase, and their text leaves the
exclusion channel. So each cross-reference appears once in the exclusion text except these
five, which appear nowhere in it.

**Lineal references.** Nine rows name an ancestor or descendant of their own code: 111191→1111,
111336→1113, 211120→2111, 211130→2111, 32111→321, 321114→321, 424410→42, 488490→48 and
711→7113. They stay text only.

**Exclusions.** The named codes give 4,586 directed exclusions over 3,954 unordered pairs, and
every named code is in the codebook. No exclusion is a generated negative: of the 45,163,632
training pairs, none has an explicit exclusion or a lineal reference as its negative, and none
has an exclusion as its positive. Runtime selection admits no exclusion either (Task 8's tests).

## 3. Text

- **Descriptions.** 1,449 codes have official text. 662 inherit their only child's: the 522
  five-digit codes of the unary pairs and 140 single-child four-digit codes. The 14 four-digit
  codes with several children and no official text keep a null description: 2111, 3231, 3241,
  4561, 4931, 5192, 7121, 9211, 9221, 9231, 9241, 9251, 9261 and 9281. `description_source`
  names the code whose text each description is, and every inherited source descends from its
  code.
- **Channels.** Present texts: title 2,125, description 2,111, examples 1,075, excluded 1,117.
  An absent channel is null. No channel holds a blank string or the old placeholder `[EMPTY]`.
- **Index roles.** The 20,373 six-digit index entries hold one role each: 6,118 examples-channel
  entries, 7,200 training, 4,042 validation and 3,013 test queries. The bundle carries them under
  its artifact hash, which pins the entry text plan 4 left unpinned.
- **Unary pairs.** The pair facts flag 522, equal to `unary_pairs` over the codebook. None is a
  generated positive: the training pairs hold 268,831 distinct (anchor, positive) pairs and no
  unary pair among them. Parent retrieval scores none (Task 9's tests).

## 4. The input window

`sentence-transformers/all-MiniLM-L6-v2`'s model card at revision 1110a243 gives its training
sequence length as 128 tokens. 256 is only the inference truncation default and 512 the position
limit. Every channel and the text-only builder truncate to 128, and the manifest records the
texts beyond it:

| Channel | Present | Beyond 128 tokens | Share |
|---|---|---|---|
| title | 2,125 | 0 | 0.0000 |
| description | 2,111 | 153 | 0.0725 |
| examples | 1,075 | 105 | 0.0977 |
| excluded | 1,117 | 464 | 0.4154 |

Counting runs the tokenizer without truncation, and it warns once that a text is longer than 512
tokens. Nothing runs the model on such a text.

## 5. Training pairs

The training-pairs member keeps generating until Stage 7, on D* and without exclusion negatives:
45,163,632 rows over 2,090 anchors. The 35 codes that anchor no generated pair are the same 35 as
in bundle 18403d29, so this plan did not change which codes anchor.

## 6. What later stages read

- **The bundle.** `data/supervision/stage3-supervision-v2/<bundle-id>/manifest.json`, built in
  this worktree from its own `data preprocess`. Its `description_fingerprint` names
  `naics_descriptions.parquet` sha256 `fe8c54e3…`. After this plan merges, main loads no v1
  bundle, and a checkpoint trained on bundle 18403d29 loads weights-only (user decision 1).
  The bundle is copied, never rebuilt: to the main checkout's `data/` and to every Lambda
  instance.
- **Stage 6** reads null channels, the tokenization cache's per-channel `present` flag (an
  absent channel is encoded as the empty string) and the 128-token window. It rebuilds the
  text-only table from these descriptions.
- **Stage 7** reads D*, the redirection table's activity phrases with their referencing `code`
  (withheld rows carry none) and the unary flags. Two things this plan left for it:
  - the `quota_selections` counter and `SelectionReason.EXCLUSION_QUOTA`, which stay at zero
    now that no slot is reserved;
  - Phase 1's sibling mask, which masks every candidate at D* 2, grandparents and grandchildren
    included.
- **Stage 9** records each candidate backbone's window in `TRAINED_WINDOWS`
  (`utils/input_window.py`) from its own documentation. No config accepts a backbone without
  one.
- **Stages 10 and 11.** Arm D runs HGCN with thresholds that now bind (section 1) until Stage 11
  removes them.

## Reproduction

In a worktree at this plan's final commit, with the four Census files in `~/Downloads/Data` and
the backbone in the local Hugging Face cache:

```bash
uv run naics-embedder data preprocess --source-dir ~/Downloads/Data
HF_HUB_OFFLINE=1 uv run naics-embedder data supervision
```

The three preprocess outputs reproduce the hashes above. The bundle gets a new ID, and plan 7's
Task 14 script recounts it.
````

Run: `grep -n '<bundle-id>\|YYYY-MM-DD' specs/findings/supervision-target-and-text.md`
Expected: no output.

- [x] **Step 7: Commit**

```bash
git add specs/findings/supervision-target-and-text.md
git commit -m "docs(findings): record Stage 5's real-data bundle"
```

## Final verification (controller, inline)

- [x] **Step 1: Full suite on Python 3.12**

> Deviation: 1697 passed, 1 skipped: the final review's fix (025de2a) added `test_missing_legacy_artifacts_name_no_command_that_cannot_build_them`.

Run: `uv run pytest -n auto -q`
Expected: `1696 passed, 1 skipped`.

- [x] **Step 2: Full suite on Python 3.10, CI's other leg**

> Deviation: 1697 passed, 1 skipped, as in Step 1. The only warnings 3.10 adds are the spurious matmul RuntimeWarnings, at 15 locations in `panels/ridge.py` and `decision/resampling.py`.

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310-c3a0baa0 uv run pytest -n auto -q`
Expected: `1696 passed, 1 skipped`, the same as Step 1. The warnings count runs far higher than
on 3.12: numpy 2.2.6, which the lock pins for Python 3.10, raises spurious "encountered in
matmul" RuntimeWarnings with Accelerate on this Mac. CI's Linux wheels are unaffected.

Run: `rm -rf /tmp/naics-py310-c3a0baa0`

- [x] **Step 3: The CI lint job**

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 4: The docs build**

Run: `uv run mkdocs build --strict -q -d /tmp/stage5-docs-c3a0baa0`, then
`rm -rf /tmp/stage5-docs-c3a0baa0`
Expected: no output and exit 0. PR CI never runs the docs workflow, so this build stands in for
it. After the merge, check the first docs run on `main`.

- [x] **Step 5: The branch carries only this plan's commits**

> Deviation: 87 paths: the review fix (025de2a) adds `docs/quickstart.md` and `src/naics_embedder/utils/validation.py`. The grep also prints `conf/graph.yaml:9`, a comment unchanged from origin/main.

Run: `git log --oneline origin/main..HEAD`
Expected, read bottom up, because `git log` prints the newest commit first:

- the plan's commit, at the bottom;
- then each task's commit in task order, Task 1's through Task 14's. Task 14 commits only the
  finding.
- Review fixes may add commits between them.

Neither "config" nor "graph config" appears.

Run: `git diff --name-only origin/main...HEAD`
Expected: exactly these 85 paths. The three dots diff from the merge base, so a commit that lands
on `origin/main` meanwhile does not show up. `AGENTS.md` is a symlink to `CLAUDE.md`, so it does
not appear.

```text
CLAUDE.md
README.md
conf/config.yaml
conf/data/download.yaml
conf/data/regressor_panel.yaml
conf/data/supervision.yaml
conf/data_loader/tokenization.yaml
docs/.nav.yml
docs/api/config.md
docs/api/input_window.md
docs/api/naics_hierarchy.md
docs/api/redirections.md
docs/hgcn_training.md
docs/overview.md
docs/text_training.md
docs/usage.md
specs/findings/supervision-target-and-text.md
specs/plans/7-supervision-target-and-text.md
src/naics_embedder/cli/commands/data.py
src/naics_embedder/cli/commands/training.py
src/naics_embedder/data/compute_distances.py
src/naics_embedder/data/compute_relations.py
src/naics_embedder/data/create_triplets.py
src/naics_embedder/data/download_data.py
src/naics_embedder/data/positive_sampling.py
src/naics_embedder/data/redirections.py
src/naics_embedder/data/supervision_bundle.py
src/naics_embedder/metrics/diagnostics.py
src/naics_embedder/metrics/hierarchy_structure.py
src/naics_embedder/panels/decoding.py
src/naics_embedder/panels/index_roles.py
src/naics_embedder/panels/leakage.py
src/naics_embedder/panels/outcome.py
src/naics_embedder/panels/text_only.py
src/naics_embedder/supervision/artifacts.py
src/naics_embedder/supervision/margins.py
src/naics_embedder/supervision/mode.py
src/naics_embedder/supervision/schema.py
src/naics_embedder/supervision/selection.py
src/naics_embedder/text_model/curriculum.py
src/naics_embedder/text_model/dataloader/difficulty_sampler.py
src/naics_embedder/text_model/dataloader/streaming_dataset.py
src/naics_embedder/text_model/dataloader/tokenization_cache.py
src/naics_embedder/text_model/loss.py
src/naics_embedder/text_model/mixins/curriculum.py
src/naics_embedder/text_model/mixins/logging.py
src/naics_embedder/text_model/mixins/validation.py
src/naics_embedder/text_model/naics_model.py
src/naics_embedder/utils/config.py
src/naics_embedder/utils/input_window.py
src/naics_embedder/utils/naics_hierarchy.py
tests/fixtures/supervision.py
tests/integration/test_distributed_supervision.py
tests/integration/test_stage3_training_step.py
tests/unit/test_checkpoint_contract.py
tests/unit/test_cli_commands.py
tests/unit/test_cli_training.py
tests/unit/test_config.py
tests/unit/test_data_distances.py
tests/unit/test_data_download.py
tests/unit/test_data_triplets.py
tests/unit/test_datamodule.py
tests/unit/test_diagnostics.py
tests/unit/test_graph_preprocessing.py
tests/unit/test_hard_negative_mining.py
tests/unit/test_hgcn_streaming_dataset.py
tests/unit/test_hierarchy_metrics.py
tests/unit/test_index_roles.py
tests/unit/test_input_window.py
tests/unit/test_naics_hierarchy.py
tests/unit/test_naics_model.py
tests/unit/test_negative_selection.py
tests/unit/test_outcome_leakage.py
tests/unit/test_outcome_panel.py
tests/unit/test_positive_sampling.py
tests/unit/test_redirections.py
tests/unit/test_streaming_dataset.py
tests/unit/test_streaming_sampling.py
tests/unit/test_structural_margins.py
tests/unit/test_supervision_artifacts.py
tests/unit/test_supervision_index.py
tests/unit/test_supervision_schema.py
tests/unit/test_text_only.py
tests/unit/test_tokenization_cache.py
tests/unit/test_utils_validation.py
```

Run: `git grep -n "manifest_path" -- conf/config.yaml conf/graph.yaml`
Expected:

```text
conf/config.yaml:12:  manifest_path: null  # data supervision prints the exact immutable path to set before training
conf/graph.yaml:15:supervision_manifest_path: null
```

- [x] **Step 6: The roadmap's Stage 5 Exit, outcome by outcome**

Check each row against the evidence that backs it. Test names are given as `file::test`, and a
`::test` alone continues the file before it.

| Exit outcome | Evidence |
|---|---|
| A bundle validator asserts that D* satisfies the triangle inequality over all triples via lowest common ancestors, contains no 99, and gives λ(i) + λ(j) − 2 across sectors | `test_supervision_artifacts.py::test_pair_facts_reject_a_distance_that_is_not_d_star`, `::test_pair_facts_reject_a_cross_sector_distance_off_the_formula` and `::test_loader_rejects_a_rehashed_legacy_cross_sector_distance`; `test_naics_hierarchy.py::test_d_star_satisfies_the_triangle_inequality`, `::test_d_star_runs_through_the_lowest_common_ancestor` and `::test_across_sectors_d_star_is_both_levels_less_two`; `test_diagnostics.py::test_d_star_is_the_one_tree_distance_function`; `test_data_distances.py::test_every_value_comes_from_the_one_tree_distance_function` and `::test_no_structural_distance_is_a_sentinel`; Task 14 Step 4's D* lines |
| Each cross-reference appears once in the exclusion text | `test_redirections.py::test_the_table_lists_every_row_once_in_order` and `::test_the_channel_joins_each_kept_text_once_and_keeps_withheld_destinations`; `test_data_download.py::test_build_descriptions_builds_the_exclusion_channel_from_the_redirections`; `test_supervision_artifacts.py::test_bundle_refuses_an_exclusion_channel_the_table_does_not_build`; Task 14 Step 4's redirection lines. The five withheld rows are the recorded deviation |
| The build's held-out leakage check covers the redirection table's activity phrases and finds no match | `test_supervision_artifacts.py::test_the_leakage_check_reads_the_activity_phrases`; `test_index_roles.py::test_role_leakage_covers_the_extra_texts`; `test_redirections.py::test_a_row_a_held_out_query_leaks_into_is_withheld_and_loses_its_activity`; Task 14 Step 2's leakage line |
| The nine lineal references are flagged, and no generated training row uses any exclusion as a negative | `test_redirections.py::test_lineal_codes_are_named_ancestors_and_descendants`; `test_data_triplets.py::test_no_training_negative_is_an_explicit_exclusion` and `::test_an_exclusion_negative_is_fatal`; `test_supervision_artifacts.py::test_loader_rejects_a_rehashed_exclusion_negative`; at runtime, `test_negative_selection.py::test_an_explicit_exclusion_is_never_selected` and `::test_proposals_cannot_select_an_exclusion`, `test_streaming_sampling.py::test_candidate_pool_holds_unique_codes_and_no_exclusion`, `test_hard_negative_mining.py::test_an_exclusion_is_never_selected_even_when_nearest`, `test_naics_model.py::test_validation_step_never_scores_an_exclusion`, `test_stage3_training_step.py::test_a_selection_naming_an_exclusion_is_refused`; Task 14 Step 4's lineal and negative lines |
| No placeholder string exists in any channel | `test_data_download.py::test_verify_text_channels_refuses_blanks_placeholders_and_missing_provenance` and `::test_build_descriptions_records_provenance_and_leaves_absent_channels_null`; `test_tokenization_cache.py::test_null_channels_are_absent`; Task 14 Step 4's blank-text line |
| Every inherited description carries a provenance value, and the 14 formerly arbitrary choices resolve by the documented rule | `test_data_download.py::test_a_code_without_official_text_inherits_its_only_childs_description` and `::test_build_descriptions_records_provenance_and_leaves_absent_channels_null`; Task 14 Step 4's description lines |
| The 522 unary pairs are flagged and absent from generated positives | `test_naics_hierarchy.py::test_unary_pairs_are_five_digit_codes_with_one_six_digit_child`; `test_supervision_artifacts.py::test_the_unary_pairs_are_flagged_and_never_generated_positives` and `::test_training_pairs_refuse_a_unary_positive`; `test_data_triplets.py::test_a_unary_pair_is_never_a_positive`; `test_positive_sampling.py::test_enumerate_positives_drops_the_unary_pairs`; `test_hierarchy_metrics.py::test_parent_retrieval_skips_the_unary_pairs`; Task 14 Step 4's unary lines |
| The current backbone's trained window is recorded with each channel's overflow share, and no input exceeds it, the text-only builder's included | `test_input_window.py::test_the_backbones_trained_window_is_128_tokens` and `::test_a_null_max_length_is_the_window_and_a_longer_one_is_refused`; `test_supervision_artifacts.py::test_a_bundle_records_its_input_window` and `::test_the_input_window_record_counts_each_channels_texts_beyond_the_window`; `test_config.py::test_every_tokenizing_config_defaults_to_the_trained_window` and `::test_a_max_length_beyond_the_trained_window_is_refused`; `test_tokenization_cache.py::test_a_window_beyond_the_trained_one_is_refused`; `test_text_only.py::test_a_max_length_beyond_the_trained_window_is_refused`; `test_cli_commands.py::test_text_only_table_refuses_a_backbone_without_a_recorded_window`; Task 14 Step 4's window line |
| (Produces) The new contract version, its required members and results, and M6 and M8 | `test_supervision_artifacts.py::test_loader_names_the_contract_of_a_manifest_it_cannot_parse`, `::test_loader_requires_both_members`, `::test_loader_rejects_a_manifest_missing_a_required_validation_result` and `::test_a_build_records_exactly_the_required_validation_results`; `test_checkpoint_contract.py::test_a_checkpoint_trained_under_the_exclusion_quota_cannot_exact_resume`; `test_config.py::test_repaired_config_rejects_a_legacy_streaming_path`; `test_cli_commands.py::test_legacy_stage_commands_build_nothing` |

Run: `uv run pytest -n auto -q tests/unit/test_naics_hierarchy.py tests/unit/test_diagnostics.py tests/unit/test_data_distances.py tests/unit/test_supervision_artifacts.py tests/unit/test_redirections.py tests/unit/test_data_download.py tests/unit/test_index_roles.py tests/unit/test_data_triplets.py tests/unit/test_tokenization_cache.py tests/unit/test_positive_sampling.py tests/unit/test_hierarchy_metrics.py tests/unit/test_input_window.py tests/unit/test_config.py tests/unit/test_text_only.py tests/unit/test_cli_commands.py tests/unit/test_hard_negative_mining.py tests/unit/test_negative_selection.py tests/unit/test_streaming_sampling.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py tests/unit/test_checkpoint_contract.py`
Expected: `566 passed`.

- [x] **Step 7: No panel was read and no selection log was written**

Run: `uv run python -c "import glob; print(sorted(glob.glob('logs/*.jsonl') + glob.glob('/tmp/stage5-supervision-c3a0baa0/*.jsonl')))"`
Expected: `[]`. Tests write their selection logs under pytest's temporary directories, and Task 14
reads no panel. If a `.jsonl` file appears, stop and ask.

- [x] **Step 8: Remove the scratch directory**

Run: `rm -rf /tmp/stage5-supervision-c3a0baa0`

The bundle and the three preprocess outputs stay in this worktree's `data/`. They are the only
copies until Plan completion Step 5 hands them over.

## Plan completion

Run the Plan Completion Protocol of writing-plans after the final review. The completion commits
are the branch's last commits. Before editing `specs/naics-embedding-roadmap.md` or
`specs/deferred_items.md`, check whether another Claude session is active in this repository. If
one is, hold both edits and hand your human partner the exact text below.

In every step below, replace `<bundle-id>` with the ID Task 14 Step 3 printed, and `YYYY-MM-DD`
with the completion date.

- [x] **Step 1: Tick the roadmap stage and add the rollout note and the stamp**

> Deviation: the bundle ID lengthened the rollout note's third line past 100 columns; the note was reflowed through its withheld-rows sentence.

In `specs/naics-embedding-roadmap.md`, replace:

```markdown
- [ ] Stage 5: Supervision target and text
```

with:

```markdown
- [x] Stage 5: Supervision target and text
```

Make a second edit. Replace the Stage 5 entry's last lines:

```markdown
      exceeded it, and no input exceeds it, the text-only builder's included.
      ROUTING: writing-plans

- [ ] Stage 6: Shared encoder and low-dimensional projection
```

with:

```markdown
      exceeded it, and no input exceeds it, the text-only builder's included.
      ROUTING: writing-plans
      Rollout note: the switch happens at merge. Main then loads only `stage3-supervision-v2`
      bundles, and a checkpoint trained on bundle 18403d29 loads weights-only. Bundle
      `<bundle-id>`, built from a fresh `data preprocess` (descriptions sha256 `fe8c54e3…`), is
      copied to the main checkout and to every Lambda instance, never rebuilt. Held-out queries
      leak into five cross-references, which are withheld from the exclusion text (user decision
      2), so "each cross-reference appears once" holds for the other 4,596. Until Stage 7, the
      relation margin axis keeps its cross-sector margin, keyed on the `cross_sector` label; the
      `quota_selections` counter stays at zero; and Phase 1's sibling mask masks every candidate
      at D* 2, grandparents and grandchildren included. HGCN's curriculum thresholds, 7, 9 and 10
      under D*, bind until Stage 11 removes them.
      Realized: D* takes integers 1–10 over 2,256,750 pairs, equals `metrics/diagnostics.py`'s on
      every pair and passes the triangle inequality over all 2,125³ ordered triples; 522 unary
      pairs; 4,623 redirection rows (4,601 cross-references and 22 harvested paragraphs; 4,529
      activity phrases; 9 lineal references), with the spec's 43 and the old 68 reconciled; no
      held-out leakage, activity phrases included; 45,163,632 training pairs, none with an
      exclusion or lineal reference as its negative or a unary pair as its positive; a 128-token
      window, beyond which lie 0.0000 of titles, 0.0725 of descriptions, 0.0977 of examples and
      0.4154 of exclusion texts (`specs/findings/supervision-target-and-text.md`).
      Stage 5: COMPLETE (YYYY-MM-DD) — implemented by plan 7
      (specs/plans/completed/7-supervision-target-and-text.md). Next: resume the roadmap.

- [ ] Stage 6: Shared encoder and low-dimensional projection
```

- [x] **Step 2: Re-validate the later stages against what shipped**

> Deviation: the bundle ID lengthened one line of `specs/lambda-remote-workflow.md` past 100 columns; it was reflowed.

Four later entries consume what Stage 5 shipped in ways their text does not yet say. Each edit's
Replace text occurs exactly once in the roadmap.

In the Stage 6 entry, replace:

```markdown
      Consumes: Stage 5's bundle (null channels, window policy). The current objective and
```

with:

```markdown
      Consumes: Stage 5's bundle (null channels, window policy) and its tokenization cache,
      which encodes an absent channel as the empty string with a per-channel `present` flag for
      the mask and tokenizes every channel at the 128-token window `utils/input_window.py`
      records. The current objective and
```

In the Stage 7 entry, make two edits. First, replace:

```markdown
      Consumes: Stage 6's encoder and query path; Stage 5's D*, redirection table and unary
      flags; Stage 2's query splits, scorer and selection log (the log's path is
```

with:

```markdown
      Consumes: Stage 6's encoder and query path; Stage 5's D*, redirection table and unary
      flags (the table's `activity` phrases with their referencing `code`; its five withheld rows
      carry none); Stage 2's query splits, scorer and selection log (the log's path is
```

Second, replace:

```markdown
      inverse-distance draws, the exclusion quota, the relation margin axis (D5), and legacy
```

with:

```markdown
      inverse-distance draws and Phase 1's sibling mask (at D* 2 it also masks grandparents),
      what remains of the exclusion quota (Stage 5 removed its slot; the `quota_selections`
      counter and `SelectionReason.EXCLUSION_QUOTA` remain), the relation margin axis (D5), and
      legacy
```

In the Stage 9 entry, replace:

```markdown
      trained window recorded per candidate with each channel's overflow share; if IC is
```

with:

```markdown
      trained window recorded per candidate with each channel's overflow share (a window enters
      `TRAINED_WINDOWS` in `utils/input_window.py` from the candidate's own documentation, since
      no config accepts a backbone without one, and `input_window_record` computes the shares);
      if IC is
```

In the Stage 10 entry, replace:

```markdown
      6's export command; Stage 5's bundle as the graph stage's structural input.
```

with:

```markdown
      6's export command; Stage 5's bundle as the graph stage's structural input. Under D* its
      curriculum thresholds bind (phase-1 negatives at D* ≤ 7, phase-2 at D* ≤ 9), and arm D runs
      with them, since only Stage 11 removes them.
```

Stages 8, 11 and 12 need no edit:

- Stage 8 consumes Stage 7's reference configuration, not Stage 5's outputs.
- Stage 11 already removes the curriculum filters if the graph stage is kept, and the whole
  stage if it is dropped, so the thresholds leave either way.
- Stage 12's bundle codebook keeps the pinned 2,125 codes, in the same order.

Re-point the Req 8 gap row. First re-read the line:

Run: `grep -n "df_rel = pl.read_parquet(relations_path).select(" src/naics_embedder/graph_model/hgcn.py`
Expected: `1157:    df_rel = pl.read_parquet(relations_path).select(`. The citation covers that
line through the `)` that closes the select, five lines below. If the grep prints another line
number N, cite N through N + 5 instead.

In `specs/naics-embedding-roadmap.md`, replace:

```markdown
`graph_model/hgcn.py:1173-1178`
```

with:

```markdown
`graph_model/hgcn.py:1157-1162`
```

One more file names bundle 18403d29 as current. In `specs/lambda-remote-workflow.md`, replace:

```markdown
  `data/supervision/stage3-supervision-v1/<bundle_id>/manifest.json`. The current bundle,
  `18403d29-3b23-444e-9e81-371d0ca8b7ea`, was built on 2026-09-23 and is pinned to the
  regenerated parquet. Rebuild it only when the parquet changes; a new bundle means older
  checkpoints can only load with `--checkpoint-load-mode weights_only`.
```

with:

```markdown
  `data/supervision/stage3-supervision-v2/<bundle_id>/manifest.json`. The current bundle,
  `<bundle-id>`, was built on YYYY-MM-DD by roadmap Stage 5 (plan 7) and is pinned to that
  stage's descriptions parquet (sha256 `fe8c54e3…`). It replaced bundle
  `18403d29-3b23-444e-9e81-371d0ca8b7ea`, whose contract main no longer loads. Rebuild it only
  when the parquet changes; a new bundle means older checkpoints can only load with
  `--checkpoint-load-mode weights_only`.
```

Keep `<bundle_id>`, with an underscore, as written: it is the path's placeholder. Replace only
`<bundle-id>`, with a hyphen.

Commit these edits with the plan markup in Step 3's commit.

- [x] **Step 3: Mark up this plan and resolve the gate**

> Deviation: the gate asked two batched questions. The user kept the threshold item open for Stage 11, deferred the final review's non-blocking findings 4–9 as six entries and dropped finding 3 (the loader's triangle loop: 1.4 s per load, and exact equality with D* implies it); finding 10 found no overlap. The bundle ID lengthened the entry-text item's note past 100 columns; it was reflowed. The backlog stands at 19 open, under the triage threshold.

Follow the protocol:

- Run the resolve-before-defer gate. Include in its batched questions that the open
  graph-curriculum threshold item's "Revisit if" has fired: under D* the thresholds gate
  training. The roadmap assigns its removal to Stage 11.
- Tick every completed step and add `> Deviation:` notes.
- Add the status header.
- Tick the four items this plan discharged in `specs/deferred_items.md`, and annotate the
  threshold item, as below.
- Append this plan's deferred items, if any.
- Run `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
  and surface its summary line.

M6: replace

```markdown
- [ ] Review M6: in repaired mode, explicitly set legacy
```

with

```markdown
- [x] Review M6: in repaired mode, explicitly set legacy
```

and replace

```markdown
      those paths to a non-default value fails validation with a migration message.
```

with

```markdown
      those paths to a non-default value fails validation with a migration message.
      → done in plan 7 (Task 12: `validate_supervision_contract` rejects a legacy streaming path
      set to anything but its default, with a migration message, also after `Config.override`).
```

M8: replace

```markdown
- [ ] Review M8: `data relations`, `data distances`, and `data triplets` each build a complete
```

with

```markdown
- [x] Review M8: `data relations`, `data distances`, and `data triplets` each build a complete
```

and replace

```markdown
      the notice and exit without building).
```

with

```markdown
      the notice and exit without building).
      → done in plan 7 (Task 12: each prints the migration notice and exits with status 1
      without building).
```

Plan 4's validation-results item: replace

```markdown
- [ ] Review Minor: when a bundle carries the `index_roles` member, `load_validated_bundle`
```

with

```markdown
- [x] Review Minor: when a bundle carries the `index_roles` member, `load_validated_bundle`
```

and replace

```markdown
      `index_roles` member lacks those three validation results.
```

with

```markdown
      `index_roles` member lacks those three validation results.
      → done in plan 7 (Task 10: every bundle carries the member, and the loader requires all 27
      validation results a build records, the three `index_roles_*` among them).
```

Plan 4's entry-text item: replace

```markdown
- [ ] Review Minor: nothing pins the index-entry text until a bundle carries `index_roles`.
```

with

```markdown
- [x] Review Minor: nothing pins the index-entry text until a bundle carries `index_roles`.
```

and replace

```markdown
      before Stage 5 builds the first bundle with the `index_roles` member.
```

with

```markdown
      before Stage 5 builds the first bundle with the `index_roles` member.
      → done in plan 7 (Task 14's bundle `<bundle-id>` carries the member, entry text included,
      under its artifact hash).
```

The threshold item stays open. Replace

```markdown
      Revisit if: HGCN curriculum phases are tuned or thresholds gate training.
```

with

```markdown
      Revisit if: HGCN curriculum phases are tuned or thresholds gate training.
      Note, plan 7: under D* the thresholds are 7, 9 and 10, so they now gate HGCN's phases 1
      and 2. Still open: roadmap Stage 11 removes them.
```

Then commit:

```bash
git add specs/naics-embedding-roadmap.md \
  specs/lambda-remote-workflow.md \
  specs/plans/7-supervision-target-and-text.md \
  specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 5 and re-validate Stages 6, 7, 9 and 10"
```

- [x] **Step 4: Retire the plan**

```bash
git mv specs/plans/7-supervision-target-and-text.md specs/plans/completed/7-supervision-target-and-text.md
git commit -m "chore(specs): retire plan 7"
```

This plan has no relative links to re-point, and no spec file retires with it: Stage 5 has no
stage spec.

- [x] **Step 5: Hand over the switch**

User decision 1 makes the switch at merge, and the edits below are your human partner's, on this
machine, after the merge. Do not run them yourself. Give your human partner this step's text,
with `<bundle-id>` and `YYYY-MM-DD` filled in.

This worktree's `data/` holds the only copy of the new bundle and of the three preprocess
outputs. Neither remove the worktree nor archive its session before the first edit below is
done.

1. **Copy the canonical inputs into the main checkout**, once the merge has landed:

   ```bash
   mkdir -p /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2
   ```

   ```bash
   cp -cR /Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text/data/supervision/stage3-supervision-v2/<bundle-id> /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/
   ```

   ```bash
   cp -c /Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text/data/naics_descriptions.parquet /Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text/data/naics_index_roles.parquet /Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-7-supervision-target-and-text/data/naics_redirections.parquet /Users/lowell/Projects/naics-embedder/data/
   ```

   The descriptions file replaces the one bundle 18403d29 pins (sha256 `5107fb83…`). Bundle
   18403d29 stays on disk, although main no longer loads it.

2. **The held commits.** Replaying "config" onto the merged main conflicts: its
   `manifest_path` line sits next to the `contract_version` line this plan changed. Resolve
   `conf/config.yaml`'s block to:

   ```yaml
   supervision:
     mode: repaired  # Options: repaired, legacy_containment (explicit, tagged, never exact-resumes)
     contract_version: stage3-supervision-v2
     manifest_path: data/supervision/stage3-supervision-v2/<bundle-id>/manifest.json
   ```

   and set "graph config"'s line in `conf/graph.yaml` to:

   ```yaml
   supervision_manifest_path: data/supervision/stage3-supervision-v2/<bundle-id>/manifest.json
   ```

   Both commits stay local: never push, cherry-pick or merge them. Check the result from the
   main checkout:

   ```bash
   uv run python -c "from naics_embedder.supervision.artifacts import load_validated_bundle; print(load_validated_bundle('data/supervision/stage3-supervision-v2/<bundle-id>/manifest.json').manifest.bundle_id)"
   ```

   It prints the bundle ID.
3. **Lambda.** On every instance, upload `data/naics_descriptions.parquet` and
   `data/supervision/stage3-supervision-v2/<bundle-id>/` to the same repo-relative paths, and
   check them there with the command above. Never rebuild the bundle on an instance. A run begun
   on bundle 18403d29 cannot exact-resume under the new contract: it continues with
   `--checkpoint-load-mode weights_only --ckpt-path CHECKPOINT`.
4. **The canonical-bundle memory note.** With your human partner's go-ahead, rewrite the
   project memory `canonical-supervision-bundle.md` to say: bundle `<bundle-id>`
   (`stage3-supervision-v2`), built YYYY-MM-DD by plan 7 from the Stage 5 worktree's
   `data preprocess` (descriptions sha256 `fe8c54e3…`) and copied to the main checkout; point
   `supervision.manifest_path` at it; upload it, never rebuild it, on every Lambda instance;
   bundle 18403d29 (v1) stays on disk, but main no longer loads it; the "config" and "graph
   config" commits that set the path stay local and are never pushed.

- [x] **Step 6: Integrate**

Hand over to finishing-a-development-branch. Before opening any PR, check two things:

- `git log --oneline origin/main..HEAD` shows only this branch's commits.
- `git diff --name-only origin/main...HEAD -- conf/graph.yaml` prints nothing, and
  `git grep -n "manifest_path: null" -- conf/config.yaml` prints one line.

Never push to `main`. A PR merges only after two things:

- CI passes: `lint`, `test (3.10)` and `test (3.12)`;
- Codex's review has given its 👍, or its inline findings are fixed.

After the merge, check the first docs run on `main`: it is the first build of the three new API
pages and their `docs/.nav.yml` entries in CI. Then Step 5's edits apply.
