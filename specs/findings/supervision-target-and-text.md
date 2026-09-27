# Supervision target and text: finding

**Status: FINAL (2026-09-26).** Roadmap Stage 5 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 7
(`specs/plans/completed/7-supervision-target-and-text.md`):

- the preprocessed descriptions, index-entry roles and redirection table
- bundle `301cce28-539c-42ea-8781-496bbdcf511c` under contract `stage3-supervision-v2`, the first
  bundle carrying the `index_roles` and `redirections` members
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

- **The bundle.**
  `data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`,
  built in this worktree from its own `data preprocess`. Its `description_fingerprint` names
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
