# Regressor panel and sealed outer sets: finding

**Status: FINAL (2026-09-24).** Roadmap Stage 3 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 5 (`specs/plans/completed/5-regressor-panel.md`):

- the panel's rows, on the population and grain Stage 1's finding verified
- the held-out four-digit groups, drawn once and committed, and the partition they fix
- the text-only comparator's table (D9)
- a stub arm's validation reads, which run the panel end to end

No outer set was opened: every read below is a validation read. Section 6 lists what later stages
read.

## Sources

The QCEW national slices were read from local copies in `~/Downloads/Data/QCEW/` under the
sha256 values of Stage 1's finding (Sources), and nothing was downloaded. The codebook and the
descriptions were read by absolute path from the main checkout, where bundle 18403d29 pins them.
The run used Python 3.12 with numpy 2.3.4, polars 1.35.1, scikit-learn 1.9.1, torch 2.9.1 and
transformers 4.57.1.

| File | sha256 | Used for |
|---|---|---|
| `2022_US000_annual.csv` | `c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18` | Feature year 2022 |
| `2023_US000_annual.csv` | `fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9` | Feature year 2023; outcome year 2023 |
| `2024_US000_annual.csv` | `48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46` | Feature year 2024 (the seen outer set); outcome year 2024 |
| `2025_US000_annual.csv` | `0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811` | Outcome year 2025 |
| Bundle 18403d29's `naics_codebook.parquet` | `5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b` | The 2,125 codes (codes fingerprint `9b646af1…`) |
| `data/naics_descriptions.parquet` | `5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d` | The text-only table's text: the bundle's `description_fingerprint` |

## 1. Rows

A cell is national (`US000`), private (`own_code` 5), all sizes and annual. It is usable when it
is disclosed with positive employment, establishments and wages. A code joins its level's
population when all four window years are usable, and each population code gives three rows,
feature years 2022–2024, each with the outcome dated one year later (D7).

| Level | Codebook codes | Population | Rows | Left out (no private national cell) |
|---|---|---|---|---|
| 2 | 20 | 19 | 57 | 92 |
| 3 | 96 | 88 | 264 | 921 to 928 |
| 4 | 308 | 300 | 900 | the eight NAICS 92 groups |
| 5 | 689 | 658 | 1,974 | 29 NAICS 92 codes, 11213 and 54112 |
| 6 | 1,012 | 980 | 2,940 | section 4's 32 codes of Stage 1's finding |

Every code left out lacks a private national cell; no code with one was lost to suppression or a
non-positive value in the window. At six digits the population is the branch record's 980, and
every load of the panel checks it: the record in `conf/data/regressor_panel.yaml` matches Stage 1's
decision block field by field (`tests/unit/test_regressor_branch_record.py`). BLS publishes 19
six-digit NAICS 238 codes (238110 to 238990) only as residential and nonresidential codes; each is
its five-digit parent's only child, so its cell is read from the parent's row.

## 2. The held-out draw

`data regressor-groups` held out 60 of the population's 300 four-digit groups: a fifth of each
sector's, by largest remainder, with seed 20260924. The floors give 51. The other 9 went to the
seven sectors with remainder 0.8 (11, 42, 44, 48, 54, 71 and 81) and to two of the three with
remainder 0.6 (22 and 53, by the seeded tie-break; 62 stayed at 3).

| Sector | Groups | Held out | Sector | Groups | Held out |
|---|---|---|---|---|---|
| 11 | 19 | 4 | 53 | 8 | 2 |
| 21 | 5 | 1 | 54 | 9 | 2 |
| 22 | 3 | 1 | 55 | 1 | 0 |
| 23 | 10 | 2 | 56 | 11 | 2 |
| 31 | 86 | 17 | 61 | 7 | 1 |
| 42 | 19 | 4 | 62 | 18 | 3 |
| 44 | 24 | 5 | 71 | 9 | 2 |
| 48 | 29 | 6 | 72 | 6 | 1 |
| 51 | 11 | 2 | 81 | 14 | 3 |
| 52 | 11 | 2 | Total | 300 | 60 |

The table, `conf/data/regressor_heldout_groups.csv` (306 bytes), has sha256
`deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4`. That hash is both regressor
panels' fingerprint, and `tests/unit/test_committed_regressor_groups.py` pins it.

## 3. The partition

The held-out outer set (H) holds every row of a code whose employment includes a held-out group.
The seen outer set (S) holds the other codes' 2024 rows, whose outcome is 2025. The remainder (R)
holds the other codes' 2022 and 2023 rows, and validation reads only R (user decision: disjoint).
At levels 2 and 3 a code joins H when any held-out group rolls into it (user decision: strict
seal).

| Level | R rows (codes) | S rows | H rows (codes) | R groups | Seen regime | Held-out regime |
|---|---|---|---|---|---|---|
| 2 | 2 (1) | 1 | 54 (18) | 1 | undefined: 1 group, fewer than 10 | undefined: no four-digit parent |
| 3 | 92 (46) | 46 | 126 (42) | 46 | defined | undefined: no four-digit parent |
| 4 | 480 (240) | 240 | 180 (60) | 240 | defined | defined |
| 5 | 1,046 (523) | 523 | 405 (135) | 240 | defined | defined |
| 6 | 1,554 (777) | 777 | 609 (203) | 240 | defined | defined |

Sector 55, with one group, is the only sector with no held-out group, so it is the level-2
remainder. A group is the four-digit parent at levels 4–6 and the code itself at levels 2 and 3.
D8's two regressor panels are the level-6 cells; levels 3–5 are the multi-level variant.

## 4. Fitting and the text-only table

- **Ridge.** Features are standardized on the fit rows, over 17 penalties from 0.001 to 100,000.
  The whole path comes from one singular value decomposition per fit.
- **Seen regime.** It fits R's 2022 rows and scores R's 2023 rows in 5 grouped folds, each fold's
  penalty chosen on the other folds, over 5 repeats. The test split, after an opening, tunes on
  that forward split and fits all of R.
- **Held-out regime.** It runs 5 repeats of 5 grouped folds over R, each with 5 grouped inner
  folds for the penalty. The test split tunes by 5 grouped folds over R.
- **Pairing.** Fold seeds depend on the regime, level, repeat and group ids alone, so every arm
  meets the same folds.
- **Text-only table (D9).** `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, frozen and on the CPU, embedded 2,125 codes at
  width 384 in about 33 seconds, with `max_length` 512. Each of the four channels is mean-pooled,
  and a code's vector is the mean of its present channels. The table's sha256 on this run was
  `6386f912…`. It is not pinned, because floats can differ across machines. The table stays
  outside the repo; its provenance records the backbone, revision, descriptions hash and library
  versions.

## 5. A stub arm on the validation split

The stub arm is the text-only table itself, reduced by PCA to 16 dimensions, so its `embedding`
comparator coincides with `text_only` by construction. The run checks the panel end to end on the
real rows; it measures no arm. One `tools regressor-panel` call scored levels 2–6 in both regimes
in about half a minute. It logged 7 validation reads: seen at levels 3–6 and held-out at 4–6,
each with `n_queries` equal to its level's remainder rows. It wrote 179,170 predictions.

Each cell is RMSE / R² / median penalty over the five repeats.

| `regressor_seen` | Level 3 | Level 4 | Level 5 | Level 6 |
|---|---|---|---|---|
| covariates | 0.4755 / 0.9354 / 0.001 | 0.4214 / 0.9286 / 0.001 | 0.4142 / 0.9263 / 0.001 | 0.3897 / 0.9330 / 0.001 |
| embedding | 1.2817 / 0.5308 / 0.001 | 1.2853 / 0.3358 / 0.001 | 1.2383 / 0.3413 / 0.001 | 1.2637 / 0.2959 / 0.001 |
| covariates+embedding | 0.2634 / 0.9802 / 0.001 | 0.2812 / 0.9682 / 0.001 | 0.2777 / 0.9669 / 0.001 | 0.2730 / 0.9671 / 0.001 |
| one_hot | 0.0322 / 0.9997 / 0.001 | 0.0623 / 0.9984 / 0.001 | 0.0636 / 0.9983 / 0.001 | 0.0601 / 0.9984 / 0.001 |
| covariates+one_hot | 0.0471 / 0.9994 / 0.001 | 0.0614 / 0.9985 / 0.001 | 0.0633 / 0.9983 / 0.001 | 0.0605 / 0.9984 / 0.001 |
| ancestors | 1.3971 / 0.4425 / 0.001 | 0.9004 / 0.6741 / 0.001 | 0.7805 / 0.7383 / 0.001 | 0.6059 / 0.8381 / 0.001 |
| covariates+ancestors | 0.2519 / 0.9819 / 0.001 | 0.1829 / 0.9866 / 0.32 | 0.1714 / 0.9874 / 0.32 | 0.1303 / 0.9925 / 1 |
| text_only | 1.2817 / 0.5308 / 0.001 | 1.2853 / 0.3358 / 0.001 | 1.2383 / 0.3413 / 0.001 | 1.2637 / 0.2959 / 0.001 |
| covariates+text_only | 0.2634 / 0.9802 / 0.001 | 0.2812 / 0.9682 / 0.001 | 0.2777 / 0.9669 / 0.001 | 0.2730 / 0.9671 / 0.001 |

| `regressor_heldout` | Level 4 | Level 5 | Level 6 |
|---|---|---|---|
| covariates | 0.4229 / 0.9277 / 0.32 | 0.4155 / 0.9255 / 0.32 | 0.3903 / 0.9325 / 0.32 |
| embedding | 1.4010 / 0.2064 / 100 | 1.3200 / 0.2480 / 100 | 1.3206 / 0.2274 / 100 |
| covariates+embedding | 0.3041 / 0.9626 / 1 | 0.2917 / 0.9633 / 0.001 | 0.2871 / 0.9635 / 0.32 |
| ancestors | 1.3702 / 0.2409 / 320 | 1.4033 / 0.1501 / 100 | 1.3602 / 0.1804 / 0.001 |
| covariates+ancestors | 0.2946 / 0.9649 / 1 | 0.3100 / 0.9585 / 0.001 | 0.3163 / 0.9557 / 1 |
| text_only | 1.4010 / 0.2064 / 100 | 1.3200 / 0.2480 / 100 | 1.3206 / 0.2274 / 100 |
| covariates+text_only | 0.3041 / 0.9626 / 1 | 0.2917 / 0.9633 / 0.001 | 0.2871 / 0.9635 / 0.32 |

What the run shows about the panel, not about any arm:

- **Covariates.** Alone they reach R² 0.93 at every level: size carries most of next-year log
  employment. The margin an arm can add lies above that.
- **One-hot in the seen regime.** It reaches R² 0.998 (RMSE 0.06 at six digits). With every
  code's earlier row in the fit set, code identity amounts to last year's level. This is where
  Req 2 expects one-hot to be a real competitor.
- **Seen-regime penalties.** Almost all are the grid's smallest, 0.001. On standardized features
  with hundreds of rows per fit, that is effectively least squares. The held-out regime picks
  larger ones: 0.32 for the covariates and 100 for the 16-dimensional stub.
- **Ancestor indicators.** They help in the seen regime (R² 0.84 alone at six digits) far more
  than in the held-out regime (0.18). A held-out group's own level-4 and level-5 indicators never
  appear in its fit set.

## 6. What later stages read

- **The command.** `naics-embedder tools regressor-panel --coordinates TABLE --text-only TABLE
  --codebook CODEBOOK` writes the per-row predictions with `--output`; in code,
  `RegressorPanel.validation(regime, level, arm, purpose)` returns them. The columns are
  `panel`, `split`, `level`, `comparator`, `repeat`, `fold` (−1 on the test split), `code`,
  `group`, `feature_year`, `outcome_year`, `alpha`, `outcome` and `prediction`.
- **Stage 4** computes each panel's statistic from these rows (Open questions). D8's two
  regressor panels are `regressor_seen` and `regressor_heldout` at level 6. It resamples by
  `group` within a panel and level: the four-digit parent at levels 4–6 (Req 5's unit), the code
  itself at levels 2–3. Every arm's validation rows share folds, so paired differences align row
  for row. An undefined cell (`RegressorPanel.cell_status`) has no rows.
- **Stages 7–11** read the validation split only, with `--log` pointing at the log their decision
  records keep. For these panels `n_queries` counts rows.
- **The text-only table** follows the arm's backbone and text (D9), so it is rebuilt whenever
  either changes: Stage 5 replaces the text this run embedded, and Stage 9 tries other
  backbones. The command is
  `tools text-only-table --descriptions ARM_DESCRIPTIONS --backbone ARM_BACKBONE --output TABLE`.
  The panel reduces the table to each arm's dimension, so Stage 8's dimensions need no rebuild.
- **Stage 6's export** writes tangent coordinates at the origin for a hyperbolic arm. The panel
  refuses Lorentz points.
- **Stage 12** opens each regime's outer set once, with `tools regressor-panel --split test
  --open-purpose …` or `RegressorPanel.open_outer`, under fingerprint `deddfd4c…`. The log's
  panel names are `regressor_seen` and `regressor_heldout`. A second opening needs
  `--reopen-reason` and is logged as `reopen`.

## Reproduction

The draw is deterministic, but it depends on numpy's generator streams, so the table is frozen
and committed rather than redrawn. To check that it reproduces:

1. Run `uv run naics-embedder data regressor-groups --force --codebook CODEBOOK` in a throwaway
   worktree at this commit.
2. Compare `shasum -a 256 conf/data/regressor_heldout_groups.csv` with the hash above.
3. Never commit a redraw.

The text-only table and the stub run reproduce with these commands, where `stub_arm.py` is the
script in plan 5's Task 9:

```bash
uv run naics-embedder tools text-only-table --descriptions DESCRIPTIONS --output /tmp/text_only.parquet
uv run python stub_arm.py /tmp/text_only.parquet /tmp/stub_arm.parquet
uv run naics-embedder tools regressor-panel --coordinates /tmp/stub_arm.parquet --text-only /tmp/text_only.parquet --codebook CODEBOOK --level 2 --level 3 --level 4 --level 5 --level 6 --log /tmp/selection_log.jsonl
```
