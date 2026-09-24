# Outcome panel and sealed splits: finding

**Status: FINAL (2026-09-24).** Roadmap Stage 2 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 4
(`specs/plans/completed/4-outcome-panel-sealed-splits.md`):

- the frozen index-entry role table
- the leakage check with its removal count (Verification "Leakage")
- the examples channel rebuilt from examples-role entries
- a training-free stub's scores on the validation split

Section 6 lists what later stages read.

## Sources

The four Census NAICS 2022 files were read from local copies in `~/Downloads/Data/` through
`--source-dir`, with nothing downloaded. The run used Python 3.12 with numpy 2.3.4, polars 1.35.1
and scikit-learn 1.9.1.

| File | sha256 | Used for |
|---|---|---|
| `2022_NAICS_Index_File.xlsx` (sheet `2022NAICS`) | `6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63` | Index entries; pinned as `index_sha256` in `conf/data/download.yaml` |
| `2-6 digit_2022_Codes.xlsx` | `be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316` | Codes and titles |
| `2022_NAICS_Descriptions.xlsx` | `6222c4d87dcf984970e0ff8a49862ed54b546b089d03956be3f285900cd3d66c` | Descriptions; fallback examples |
| `2022_NAICS_Cross_References.xlsx` | `3c50c3bfa9d76862aea471cc8831fd726f0b978619d833e48c54a749d9622144` | Exclusion text |

## 1. Index entries

- **Rows.** The index sheet has 20,398 rows:
  - 20,373 entries for 1,010 six-digit codes
  - 25 cross-reference ("see") rows, coded `******`, which name no code
- **Entry IDs.** `entry_id` is an entry's 0-based row position in the sheet. The entries are
  rows 0 to 20,372, and the "see" rows are the last 25.
- **Entry-less codes.** 112130 and 541120 have no entries. They are decoding candidates and never
  queries. The candidates are all 1,012 six-digit codes.
- **Entries per code.** The median is 13, the quartiles 7 and 24, and the maximum 320 (315250).
  22 codes have one entry, and 100 have at most three.
- **Cleaning.** 41 entries carried surrounding whitespace, stripped on reading. 4 contain
  non-ASCII characters, which normalization reads as word breaks.

## 2. Leakage: the rule and the removal count

**Normalization.** Texts are lowercased, keeping ASCII letters and digits, and every other run
of characters becomes one space.

**Matches.** A query leaks into a training segment in either of two ways:

- **Exact:** it occurs in the segment as whole words, equality included.
- **Near-duplicate:** its character-trigram Jaccard similarity with the segment is at least
  **9/10**, the stated similarity. The trigrams are scikit-learn's `char_wb`, and the comparison
  is in integers.

**Training text.** Every code's title and description sentences count. So do the exclusion
sentences, together with each cross-reference's activity phrase (Req 8), and the fallback
examples of the 65 codes without index entries. Every other index entry counts too.

**Eligibility.** An entry may become a validation or test query only if it matches none of that
text. Checking against every other entry, not only those that became training text, makes the
held-out splits leak-free whatever the draw.

| Entries that… | Count |
|---|---:|
| occur in static training text | 768 |
| near-duplicate static training text | 813 |
| occur in another index entry | 941 |
| near-duplicate another index entry | 2,685 |
| **are withheld for an exact match** | **1,539** |
| **are withheld as near-duplicates only (the removal count)** | **2,516** |
| are withheld in all | 4,055 |
| are eligible for validation or test | 16,318 |

The first four rows overlap. The two bold rows partition the withheld entries.

**Removal.** Verification "Leakage" asks that near-duplicates above the stated similarity be
removed from the test split and counted. Here they are removed before the draw, from both
held-out splits, because validation MRR selects (D6):

- 2,516 entries are withheld as near-duplicates only.
- 1,539 more are withheld for an exact match.

They become examples-channel text or training queries.

**Realized check.** After the draw, the 4,042 validation queries and the 3,013 test queries were
matched against all realized training text, including the rebuilt examples channels and the
7,200 training queries. Both splits had 0 exact matches and 0 near-duplicates (provenance
`held_out_leakage`).

**Limitations.** Matching compares whole queries with whole segments, so it misses two kinds of
query:

- a query whose words appear reordered inside a longer segment;
- a query that spans a segment break: a sentence end (a break also follows abbreviations such as
  "U.S.") or the `; ` joining two examples-channel entries.

A review audit measured the second kind after the draw. It matched every validation and test
query, normalized, as whole words against unsplit texts: each whole title, description and
exclusion text, and each code's joined examples channel. Every hit spans a sentence end or two
adjacent examples-channel entries.

| Unsplit text | Validation queries found | Test queries found |
|---|---:|---:|
| Titles | 0 | 0 |
| Descriptions | 2 | 0 |
| Exclusion texts | 0 | 0 |
| Joined examples channels | 2 | 1 |

Req 3 lists its training text as titles, descriptions, examples-channel entries, exclusion text and
training queries. Whole titles, descriptions and exclusion texts hold no test query, and the
realized check covers individual examples-channel entries and training queries, so no test query
matches Req 3's training text exactly and Verification "Leakage" holds as written. The one test
query in a joined channel spans two entries, neither of which contains it.

## 3. Roles (D4)

**Quotas.** Per code, roles get largest-remainder quotas of examples 3/10, training 7/20,
validation 1/5 and test 3/20.

- Remainder ties are broken by draws seeded with (20260924, code).
- Every code with entries keeps at least one examples-channel entry.
- Held-out quotas are capped at the code's eligible entries, and the excess goes to training.

| Role | Entries | Share | Codes |
|---|---:|---:|---:|
| examples | 6,118 | 30.0 % | 1,010 |
| training | 7,200 | 35.3 % | 988 |
| validation | 4,042 | 19.8 % | 939 |
| test | 3,013 | 14.8 % | 893 |

**Coverage.**

- The 22 single-entry codes keep their entry as examples-channel text, so they have no queries.
- 71 codes have no validation query and 117 no test query. Their quotas round to zero, or
  they have too few eligible entries.

**The frozen table.**

- File: `conf/data/index_roles.csv`, with columns `entry_id`, `code` and `role`; 433,147 bytes.
- sha256: `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a`.
- Provenance: `conf/data/index_roles_provenance.json`.
- `data preprocess` applies the table and never redraws it. `data roles --force` would redraw it
  and unseal both splits.

## 4. The rebuilt examples channel

The rebuilt `data/naics_descriptions.parquet` was compared with the file bundle 18403d29 pins
(sha256 `5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d`):

- **Unchanged.** Both have the same 2,125 codes and columns, and every column but `examples` is
  identical.
- **Changed.** `examples` differs for 988 codes, all with index entries. Each now holds only its
  examples-role entries, all of them among its old items. The other 22 codes with index entries
  are the single-entry codes, so their channel is unchanged.
- **Coverage.** 1,075 codes have an examples channel before and after: 1,010 from index entries
  and 65 from fallback examples.
- **No bundle.** No bundle was built. Bundle 18403d29 and the main checkout's `data/` are
  untouched, and the rebuilt files lived only in the plan's worktree. Stage 5 builds the first
  bundle that carries the `index_roles` member.

## 5. The lexical stub on the validation split

`naics-embedder tools outcome-baseline` embeds with hashed character trigrams (4,096 buckets):

- each query's text
- each code's title, description and rebuilt examples channel

It decodes under cosine distance over all 1,012 candidates.

| Metric | Value |
|---|---:|
| Queries / codes / candidates | 4,042 / 939 / 1,012 |
| Top-1 accuracy | 0.5163 |
| MRR | 0.6165 |
| Hit@1 / Hit@5 / Hit@10 | 0.5163 / 0.7333 / 0.7971 |
| Mean lowest-common-ancestor level | 4.3355 |

**The read.** The selection log's only record is this read:

- event `read`, panel `outcome`, split `validation`
- 4,042 queries
- fingerprint `05099381…`, the table's hash
- detail `{"distance": "cosine", "encoder": "LexicalTrigramEncoder"}`

The log was a scratch file, and this record is its copy.

**Sealing.** The test split was not opened: no `open` record exists for it. The logged
open-then-read path runs on fixture data in `tests/unit/test_outcome_panel.py`. `OutcomePanel`
alone enforces the seal: the committed table, `data/naics_index_roles.parquet` and a bundle's
`index_roles` member all carry every entry's role, so later stages read queries only through the
panel. Drawing the table, rebuilding the descriptions and the review's leakage audit (section 2)
read every entry directly; none of those reads selects anything.

**What the numbers mean.** These are a floor, not memorization. The code texts hold
examples-role entries and never queries, and no held-out query matches any training segment
(section 2).

## 6. What later stages read

- **Stage 3** logs its reads to the same kind of file. `SelectionLog` takes any panel name.
- **Stage 4** resamples by code from `DecodingResult.per_query`, one row per query with its
  code.
- **Stage 5** does two things:
  - makes `index_roles` a required member in its contract version
  - builds the first bundle carrying it, from `data/naics_index_roles.parquet` through
    `SupervisionBuildConfig.index_roles_parquet`
- **Stage 6** implements `QueryCodeEncoder` and takes its first live reading with
  `OutcomePanel.score(encoder, 'validation', purpose)`.
- **Stage 7** does three things:
  - trains the query→code term on `OutcomePanel.training_queries()`
  - selects on validation MRR (D6), logged to `OutcomePanelConfig.selection_log`
  - opens the test split once, with `OutcomePanel.open_test`, for the final configuration

## Reproduction

The draw is deterministic, but it depends on numpy's generator streams and the leakage matcher's
libraries. The table is therefore frozen and committed rather than redrawn. To check it
reproduces:

1. Run `uv run naics-embedder data roles --force --source-dir ~/Downloads/Data` in a throwaway
   worktree at this commit.
2. Compare `shasum -a 256 conf/data/index_roles.csv` with the hash above.
3. Never commit a redraw.

The descriptions and the baseline reproduce with these two commands:

```bash
uv run naics-embedder data preprocess --source-dir ~/Downloads/Data
uv run naics-embedder tools outcome-baseline --log /tmp/selection_log.jsonl
```
