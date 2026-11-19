# Text→code gap: finding

**Status: FINAL (2026-11-19).** Redesign roadmap Stage 4
(`specs/naics-artifact-redesign-roadmap.md`), implemented by plan 20
(`specs/plans/completed/20-text-to-code-reference-read.md`).

Both reference arms find owners' own words harder than held-out Census phrases. For the lexical
stub, C − H is 0.1843 [0.1126, 0.2533]. For `euclidean-d16`, the mean of five seeds, it is 0.1427
[0.0812, 0.2020]. Over the 689 codes that both panels hold, the gap is 0.1511 [0.1179, 0.1848] for
the stub and 0.1631 [0.1363, 0.1909] for d16. Both lower bounds exceed r × C, so D2's answer is
yes: this finding recommends a later ITA training part.

Every score here is a development measurement on the ITA validation part, not evaluation (C3).
ITA is the federal Injury Tracking Application, which receives establishments' OSHA Form 300A
filings. The sealed part was not opened. No arm was chosen, and no margin was fixed.

## 1. Universe (C4)

The pairs come from embedding-data's run `stage10_ita_establishment_20261116_r1`, pinned by the
SHA-256 of its `outputs.json` (Section 2). ITA is a program-selected filing population. It
represents no wider set of establishments. Its codes are self-reported by the filer and are not
validated. A pair's headline codes are native NAICS 2022 codes from filing years 2023 to 2025. A
pair's text is the most frequent string over all of its records, and those records can include
filing years 2016 to 2022. The copied flags catch exact normalized identity with a Census title
or index entry only. A near copy counts as the owner's own words.

C is the MRR on the outcome panel's held-out Census index entries (validation). H is the MRR on
the ITA validation part's headline pairs, the pairs that are not copied, each counted once. K is
the MRR on its copied pairs.

## 2. Sources

| Input | Identity |
|---|---|
| embedding-data commit (EDATA_SHA) | 6be42f2373112a4f963f05036642eae7bb5f1f13 |
| `specs/evaluation-protocol.md`, last changed at | 63c06664bdcce3867e2ec76c2607683de1c4320d |
| Run | `stage10_ita_establishment_20261116_r1` (not its replay) |
| `outputs.json` | e19b0d58f61f106c3994aa6fe51f4fb71244188145dd90f8cccaa8dc8a8b09df |
| `ita_text_pairs.parquet`, 644,265 rows | 60aad7642961e70b987e9d3222d660147861458e12ac52d71d2739d2d21e1e1c |
| `ita_text_pair_codes.parquet`, 771,971 rows | c8084e4ad297e3f78391e035ebe51119b07982639ae4daf47f1d47caeb1f8916 |
| `conf/data/ita_evidence.yaml` | 3d513930ae23a222989c2bc26ff9fdc917dbadef32f64bba20b01032bb16e725 |
| `conf/data/ita_panel.yaml` | d97da34469f91ac6f861346a25c485174967c4618ecb909c0a2bb11a6a1f2433 |
| ITA split fingerprint | 8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c |
| ITA validation data fingerprint (DATA_FP) | 8980e3586ec0a18d4623c3a4b3e81527f8c59e34a1f0b4b73d07b046ed86b1a3 |
| Outcome panel fingerprint | 05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a |
| Outcome validation data fingerprint | ca91f2e0e361515070fee53a909bbf51ecae6019995b85a8062745185982e81f |
| Bundle 301cce28 `manifest.json` | 68545c669c4fdbc0de9958df6270ec7426994159da0bc119a0f1fd49177ef40f |
| The stub's `data/naics_descriptions.parquet` | fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36 |
| d16 seed 1 `epoch=039.ckpt` | 24171c64d390dc59cb8b298b1ee35b3c26cc4b5dd27e46bbf50742efba7efd50 |
| d16 seed 1 `arm_table_epoch=039.parquet` | b67f487ed4e5e337cfb1202a0230aaa62dd36d0ad2649b2d77f8bca3cb035fab |
| d16 seed 2 `epoch=039.ckpt` | 0f592e2ec514e2aad488d7b6a0668dda3a1868d40ef55a04a811ee626a36eb40 |
| d16 seed 2 `arm_table_epoch=039.parquet` | 3a88c77e08fe8eaa32959f61a25864e8d4fee57b44b3a795c9fb1ccdcc42345d |
| d16 seed 3 `epoch=039.ckpt` | 09692c6f82cadfe8d17dc8b849cd2479013bb1f74fbfa4d94a06578ac416203a |
| d16 seed 3 `arm_table_epoch=039.parquet` | 3054d864d9cf42562fe9781d8b513dfc81267096399daf4b399c8ae1f9682f48 |
| d16 seed 4 `epoch=038.ckpt` | 0d0317e723bc2cd4711abe088f671e97deadae94f5402ed7f10954ffceef1b43 |
| d16 seed 4 `arm_table_epoch=038.parquet` | 87b9200c3ef7ba9c697bcd319f37c26d75e063bfac4fe20023fc17ef0d88c6fd |
| d16 seed 5 `epoch=039.ckpt` | 68c07a914b65aa745a9e883062b371a57ac3a6d945993ba638abaab15782dc90 |
| d16 seed 5 `arm_table_epoch=039.parquet` | 107c85100fa5ba88bdf0bf7a37d5fe000c2f9a3aba0d57d2c162ffe25e12f75c |
| Stage 8 record `records/stage8/euclidean-d16.json` | 3d5a80d5a3ac5466e10b019773fa09a21e2ad0592af32d7b22273fea3e1bab2c |
| OUT `lexical-stub.json` | e200e50fd900a0075a4555f269d778de09d3dfe7e0a7cfe522216b1c0a2f2ee1 |
| OUT `euclidean-d16-s1.json` | 3c6c7d18c03908d7a5290af9e9f18dacc2668669b7f68fb0bc88264d11db45f2 |
| OUT `euclidean-d16-s2.json` | b8ceaabe247eef8cc1085647220c443f9515898d0d2176ee2f9ea51733528621 |
| OUT `euclidean-d16-s3.json` | 2a546f34560022423ac23ca432714ce8c9d57f847ebfb61310f9de2f1c589706 |
| OUT `euclidean-d16-s4.json` | 017ddb88d40e77cb6a47b2d89ab930433059795c3b11ff85ac4fd7d67c4f5eaa |
| OUT `euclidean-d16-s5.json` | 00e483277fd0eef60644f2a5f07ee8d8292db28446130f67a03ee2afc2700523 |
| OUT `gap.json` (GAP_SHA) | f126926250d84178636a6bc087b2f520fbea11de971feec62c22bbd71a256997 |
| `uv.lock` | 6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948 |

OUT is `/Users/lowell/naics-artifacts/records/text_to_code/stage4/`, outside the repository.
The d16 checkpoints and tables are in the main checkout's `checkpoints/stage8-euclidean-d16-sN/`,
and their SHA-256 are the Stage 8 record's. The bundle is
`stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c`.

Every read ran at READ_SHA 9475b43913dc69031d8979d3d2394c5c0318dac8, the merge of PR A (#176),
from the main checkout's root, on `main` with a clean tree. A drift check before each read
confirmed both. The environment: macOS 27.2.1 on arm64, Python 3.12.12, torch 2.9.1, numpy 2.3.4
and polars 1.35.1, with MPS available and no BLAS thread variable set.

## 3. The reads

The stub's read, with its outcome read beside it:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools ita-read --arm lexical-stub --outcome \
  --pairs RUN_DIR/ita_text_pairs.parquet --pair-codes RUN_DIR/ita_text_pair_codes.parquet \
  --descriptions /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet \
  --run RUN --purpose 'text-to-code reference gap read: redesign roadmap Stage 4, plan 20 (Req 28)' \
  --store /Users/lowell/naics-artifacts \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --output OUT/lexical-stub.json
```

A d16 seed's read drops `--outcome` and `--descriptions`, and adds `--arm euclidean-d16 --seed N
--checkpoint checkpoints/stage8-euclidean-d16-sN/epoch=0EE.ckpt --table
checkpoints/stage8-euclidean-d16-sN/arm_table_epoch=0EE.parquet --output
OUT/euclidean-d16-sN.json`, with EE 38 for seed 4 and 39 for the others. RUN_DIR is the run's
directory in embedding-data's `data/derived/osha/establishment-targets/`.

- RUN: `ita-gap/d1dd4024708744a9aa33ecf8c346ad36`, one run id for all six reads.
- PURPOSE: `text-to-code reference gap read: redesign roadmap Stage 4, plan 20 (Req 28)`.
- Order and wall times: the stub 33 s, then d16 seeds 1 to 5 at 170, 147, 138, 142 and 144 s.
  Each record passed its checks before the next read started.
- Six dry runs before the reads, under the run `ita-gap/dry-run`, logged nothing and gave one
  pair count and one DATA_FP.

The validation part's counts, the same in all six reads:

| Count | Value |
|---|---|
| Pairs | 125,616 |
| Description groups | 9,657 |
| Copied pairs | 117,108 |
| Headline pairs (not copied) | 8,508 |
| Description groups of headline pairs (phrasing) | 3,768 |
| Conflict pairs (`flag_headline_codes_conflict`) | 227 |
| Pairs with a mojibake flag | 0 |
| Pairs with a company-fallback flag | 16,506 |

The flags do not change the headline. They are counts only (text-to-code spec §4.3).

## 4. Scores

A d16 value is the mean of its five seeds. The stub decodes by cosine distance and d16 by
euclidean distance. The phrasing report gives each distinct description of headline pairs one
weight. A level-L MRR counts a candidate as a match when its level-L ancestor is a reported
code's level-L ancestor, from the sector (level 2) to the code itself (level 6).

| Arm | Report | Pairs or groups | MRR | Top-1 | Hit@5 | Hit@10 | Mean LCA level | MRR L2 | MRR L3 | MRR L4 | MRR L5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lexical-stub` | headline | 8,508 | 0.4322 | 0.3606 | 0.4999 | 0.5549 | 3.2920 | 0.6957 | 0.5754 | 0.4915 | 0.4564 |
| `euclidean-d16` | headline | 8,508 | 0.4498 | 0.3377 | 0.5751 | 0.6660 | 3.5429 | 0.7958 | 0.6618 | 0.5336 | 0.4813 |
| `lexical-stub` | copied | 117,108 | 0.7448 | 0.6552 | 0.8490 | 0.8911 | 4.7833 | 0.9258 | 0.8610 | 0.8053 | 0.7622 |
| `euclidean-d16` | copied | 117,108 | 0.7223 | 0.5887 | 0.8950 | 0.9388 | 4.8635 | 0.9558 | 0.9207 | 0.8417 | 0.7644 |
| `lexical-stub` | phrasing | 3,768 | 0.4041 | 0.3240 | 0.4900 | 0.5448 | 3.2066 | 0.7059 | 0.5602 | 0.4762 | 0.4350 |
| `euclidean-d16` | phrasing | 3,768 | 0.3862 | 0.2813 | 0.5037 | 0.5837 | 3.2516 | 0.7324 | 0.5883 | 0.4900 | 0.4258 |

d16's MRR by seed:

| Seed | Headline | Copied | Phrasing |
|---|---|---|---|
| 1 | 0.4446 | 0.6987 | 0.3733 |
| 2 | 0.4422 | 0.6997 | 0.3812 |
| 3 | 0.4496 | 0.7526 | 0.3994 |
| 4 | 0.4578 | 0.7160 | 0.3800 |
| 5 | 0.4549 | 0.7446 | 0.3971 |

The stub's outcome read: MRR 0.6165, top-1 0.5163, hit@5 0.7333, hit@10 0.7971 and mean LCA level
4.3355, on 4,042 validation queries (`n_codes` 939, `n_candidates` 1,012). The outcome-panel-splits
finding (§5, 2026-09-24) gave the stub 0.6165 too, but that read came before plan 7 rebuilt the
descriptions. Plan 20 reports this value beside the older one and does not check it (P4). d16's C
comes from the Stage 8 record's stored outcome scores, not from a new read.

## 5. The gap (D1)

Each interval is a 95 % percentile interval from 10,000 bootstrap replicates, with seed 20260924.
Each stratum draws its own units, and one draw of seeds per arm serves every stratum (P5). The
code-matched values give each code one weight (P6).

| Arm | Estimate | Point | 95 % interval |
|---|---|---|---|
| `lexical-stub` | C, Census phrases | 0.6165 | [0.5905, 0.6416] |
| `lexical-stub` | K, copied pairs | 0.7448 | [0.6984, 0.7895] |
| `lexical-stub` | H, headline pairs | 0.4322 | [0.3675, 0.4990] |
| `lexical-stub` | C over the matched codes | 0.5681 | [0.5440, 0.5921] |
| `lexical-stub` | H over the matched codes | 0.4169 | [0.3930, 0.4416] |
| `lexical-stub` | C − H | 0.1843 | [0.1126, 0.2533] |
| `lexical-stub` | C − K | -0.1283 | [-0.1806, -0.0763] |
| `lexical-stub` | K − H | 0.3127 | [0.2305, 0.3923] |
| `lexical-stub` | C − H over the matched codes | 0.1511 | [0.1179, 0.1848] |
| `euclidean-d16` | C, Census phrases | 0.5925 | [0.5667, 0.6177] |
| `euclidean-d16` | K, copied pairs | 0.7223 | [0.6762, 0.7667] |
| `euclidean-d16` | H, headline pairs | 0.4498 | [0.3963, 0.5064] |
| `euclidean-d16` | C over the matched codes | 0.5704 | [0.5496, 0.5920] |
| `euclidean-d16` | H over the matched codes | 0.4073 | [0.3851, 0.4298] |
| `euclidean-d16` | C − H | 0.1427 | [0.0812, 0.2020] |
| `euclidean-d16` | C − K | -0.1299 | [-0.1802, -0.0791] |
| `euclidean-d16` | K − H | 0.2725 | [0.2005, 0.3435] |
| `euclidean-d16` | C − H over the matched codes | 0.1631 | [0.1363, 0.1909] |

Both arms match 689 codes: each has at least one headline validation pair and at least one
outcome validation query. 41 codes have headline pairs but no outcome validation query. They are
left out of the matched gap and counted.

What each gap isolates:

- **C − H** sets held-out Census phrases against owners' own words. It is positive for both arms.
- **C − K** sets the same phrases against owners who copied a Census phrase. It is negative for
  both arms: a copied phrase is easier than a held-out one.
- **K − H** is the wording effect within ITA. Training overlap can inflate it, because a copied
  phrase can be an entry that the arm trained on.
- **C − H over the matched codes** removes the difference in code mix between the two panels.

## 6. A training part (Req 28, Later)

D2, from plan 20: "The finding recommends a later ITA training part when, for both reference
arms, the lower bound of the 95 % interval on the code-matched headline gap exceeds 0.05 × that
arm's Census-phrase MRR C."

| Arm | Lower bound of matched C − H | Threshold (r × C) | Large |
|---|---|---|---|
| `lexical-stub` | 0.1179 | 0.0308 | yes |
| `euclidean-d16` | 0.1363 | 0.0296 | yes |

Both arms are large, so this finding recommends a later ITA training part. The recommendation
starts nothing. A training part needs a later spec, and no embedding-data table trains an artifact
until the owner amends R8.4 (G1). This finding changes nothing in Stage 5.

## 7. The selection log

| | Lines | SHA-256 |
|---|---|---|
| Before the reads | 277 | 85013962623b5751a1d01dd199a2d2de876817e47f8f2b92713ef33cab18cad0 |
| After the reads | 284 | 5b98c867ed96e7c05d3d3bca46146b45ada2245d351cfeae58ed6efc4980f6f0 |

The first 277 lines kept their SHA-256, and no other session added a record. RUN's seven records
are lines 278 to 284: the stub's `ita_text` validation read, its `outcome` validation read, then
d16 seeds 1 to 5's `ita_text` validation reads. All seven have role `reference` and PURPOSE. The
log holds no `ita_text` opening and no `ita-margin/` run. Nothing was selected, and no margin
record exists.

## 8. What Stage 5 reads

Stage 5 reads the six read records and `gap.json` in OUT (Section 2), and this finding. Stage 4's
reads are gap evidence. They are reference reads under `REFERENCE_ARMS`, so they leave the
margin-first check open. The margin record makes its own fresh reads (text-to-code spec §4.5).
