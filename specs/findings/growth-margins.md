# Growth margins: finding

**Status: FINAL (2026-10-17).** Split roadmap Stage 3
(`specs/naics-representation-split-roadmap.md`), implemented by plan 15
(`specs/plans/completed/15-growth-margin-read.md`).

The reference `covariates+ancestors` was read alone, on the Mac, on both growth panels' level-6
validation splits, before any economic candidate exists (Req 24). Each panel's δ is r = 0.05
times the reference's validation MSE there (U13): `growth_seen` δ = 0.00015933956325255588 and
`growth_heldout` δ = 0.0001619577593991326. The selection log held no growth read before this
one, and it gained exactly the read's two validation reads. No outer set was opened. Every
economic candidate is decided against this record.

## Sources

| Input | SHA-256 |
|---|---|
| `~/Downloads/Data/QCEW/2022_US000_annual.csv` | c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18 |
| `~/Downloads/Data/QCEW/2023_US000_annual.csv` | fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9 |
| `~/Downloads/Data/QCEW/2024_US000_annual.csv` | 48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46 |
| `~/Downloads/Data/QCEW/2025_US000_annual.csv` | 0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811 |
| Bundle 301cce28 `manifest.json` | 68545c669c4fdbc0de9958df6270ec7426994159da0bc119a0f1fd49177ef40f |
| Bundle 301cce28 `naics_codebook.parquet` | 0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d |
| The codebook's 2,125 codes (`codebook_codes_sha256`) | 9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a |
| The held-out draw (`heldout_groups_sha256`) | deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4 |
| `uv.lock` | 6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948 |

The bundle is `stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c`. Its codes hash to
the panel config's pin (plan 15, Task 2), though the comment beside the pin names bundle
18403d29. The read ran at `origin/main` 30aeff652f7fa8a90715da831e9948c085c95e5d, from the main
checkout's root, on `main` with a clean tree. A candidate rebuilds its lock in a worktree at
30aeff652f7fa8a90715da831e9948c085c95e5d with `uv sync --locked`.

## 1. The read

```bash
cd /Users/lowell/Projects/naics-embedder && pwd && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools growth-margins \
  --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet \
  --name growth-margins \
  --store /Users/lowell/naics-artifacts \
  --output /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --purpose 'economic margin read: split roadmap Stage 3, plan 15 (Req 24)' \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --lock /Users/lowell/Projects/naics-embedder/uv.lock \
  > /tmp/naics_plan15/read.txt 2>&1; echo "exit=$?"
```

| Environment | Value |
|---|---|
| Platform | macOS-26.6.2-arm64-arm-64bit |
| Python | 3.12.12 |
| numpy | 2.3.4 |
| polars | 1.35.1 |
| BLAS | accelerate (found True) |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `POLARS_MAX_THREADS` | all unset (Task 1 Step 3) |

The record names the platform, Python and lock. The rest is recorded here because the record does
not, and a candidate read that fails to pair should check it first (plan 14's Handoff: BLAS
threading is the first suspect). Pairing compares the reference's predictions bit for bit; the
environments only name where each read ran (`check_reference_pairing`).

The command printed:

```text
Growth margins 'growth-margins', fixed 2026-10-18T00:43:32.478131+00:00

  • growth_seen: δ 0.00015934 = 0.05 × MSE 0.00318679 of covariates+ancestors
  • growth_heldout: δ 0.000161958 = 0.05 × MSE 0.00323916 of covariates+ancestors

Read on macOS-26.6.2-arm64-arm-64bit, Python 3.12.12, uv.lock 6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948: every candidate read must run there too
Reads logged to /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl (fingerprint deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4)

Margin record: /Users/lowell/naics-artifacts/records/economic/growth_margins.json
```

## 2. The margins

| Panel | Statistic | Reference MSE | δ |
|---|---|---|---|
| `growth_seen` | `covariates+ancestors` | 0.0031867912650511174 | 0.00015933956325255588 |
| `growth_heldout` | `covariates+ancestors` | 0.0032391551879826517 | 0.0001619577593991326 |

r = 0.05 is `relative_margin` in `conf/data/decision.yaml`. The margins were fixed at
2026-10-18T00:43:32.478131Z (UTC; the evening of 2026-10-17 on the Mac, which is the date this
finding carries), by the read `margins/17a7e118f926405ea895bd0f908a549c` (Task 4 Step 2). The
level-6 data fingerprint is 3684e65e87bcb163fa5b37e93e6c8e202ffabb5fbfefdea7a374ae97c167c941.

| Record | SHA-256 |
|---|---|
| `~/naics-artifacts/records/economic/growth_margins.json` | 4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0 |

The read's predictions are
`objects/5a/5a61e8d88922c2cb5822bf2ffe827cf972123d59039240923025aa4478117824/predictions.parquet`
(sha256 5a61e8d88922c2cb5822bf2ffe827cf972123d59039240923025aa4478117824) and its scores
`objects/d8/d8decf7b1a3fd7e36fd12d6f8569f465c220f8a84de3d1dfe22b38ab64d05e5c/scores.parquet`
(sha256 d8decf7b1a3fd7e36fd12d6f8569f465c220f8a84de3d1dfe22b38ab64d05e5c), in the store
`~/naics-artifacts`. The record passes `check_relative_margins` and
`check_economic_margins_first`, as `tools decide` will run them. Its stored scores give each
reference MSE within `STATISTIC_TOLERANCE` (1e-12), as a rescore of its predictions does (plan
15, Task 4). The record, verbatim:

```json
{
  "kind": "relative_margins",
  "name": "growth-margins",
  "r": 0.05,
  "reference": "covariates+ancestors",
  "read": {
    "run_id": "margins/17a7e118f926405ea895bd0f908a549c",
    "panels": {
      "regressor": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4",
      "regressor_data": "3684e65e87bcb163fa5b37e93e6c8e202ffabb5fbfefdea7a374ae97c167c941",
      "fit_settings": {
        "alphas": [
          0.001,
          0.0032,
          0.01,
          0.032,
          0.1,
          0.32,
          1.0,
          3.2,
          10.0,
          32.0,
          100.0,
          320.0,
          1000.0,
          3200.0,
          10000.0,
          32000.0,
          100000.0
        ],
        "folds": 5,
        "repeats": 5,
        "inner_folds": 5,
        "fold_seed": 20260924,
        "min_groups": 10
      }
    },
    "predictions": {
      "path": "objects/5a/5a61e8d88922c2cb5822bf2ffe827cf972123d59039240923025aa4478117824/predictions.parquet",
      "sha256": "5a61e8d88922c2cb5822bf2ffe827cf972123d59039240923025aa4478117824",
      "bytes": 258283
    },
    "scores": {
      "path": "objects/d8/d8decf7b1a3fd7e36fd12d6f8569f465c220f8a84de3d1dfe22b38ab64d05e5c/scores.parquet",
      "sha256": "d8decf7b1a3fd7e36fd12d6f8569f465c220f8a84de3d1dfe22b38ab64d05e5c",
      "bytes": 82312
    },
    "log_records": [
      {
        "detail": {
          "comparators": [
            "covariates",
            "one_hot",
            "covariates+one_hot",
            "ancestors",
            "covariates+ancestors"
          ],
          "level": 6,
          "run": "margins/17a7e118f926405ea895bd0f908a549c"
        },
        "event": "read",
        "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4",
        "n_queries": 1554,
        "panel": "growth_seen",
        "purpose": "economic margin read: split roadmap Stage 3, plan 15 (Req 24)",
        "split": "validation",
        "time": "2026-10-18T00:43:09.849910+00:00"
      },
      {
        "detail": {
          "comparators": [
            "covariates",
            "ancestors",
            "covariates+ancestors"
          ],
          "level": 6,
          "run": "margins/17a7e118f926405ea895bd0f908a549c"
        },
        "event": "read",
        "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4",
        "n_queries": 1554,
        "panel": "growth_heldout",
        "purpose": "economic margin read: split roadmap Stage 3, plan 15 (Req 24)",
        "split": "validation",
        "time": "2026-10-18T00:43:17.260357+00:00"
      }
    ],
    "environment": {
      "platform": "macOS-26.6.2-arm64-arm-64bit",
      "python": "3.12.12",
      "lock_sha256": "6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948"
    }
  },
  "margins": [
    {
      "panel": "growth_seen",
      "statistic": "covariates+ancestors",
      "reference_mse": 0.0031867912650511174,
      "margin": 0.00015933956325255588
    },
    {
      "panel": "growth_heldout",
      "statistic": "covariates+ancestors",
      "reference_mse": 0.0032391551879826517,
      "margin": 0.0001619577593991326
    }
  ],
  "fixed_at": "2026-10-18T00:43:32.478131Z"
}
```

## 3. The selection log

Before the read, `logs/selection_log.jsonl` in the main checkout had 157 records, SHA-256
aa223df2340617dc635ede104905f40a8c5b1adebc1154eda469183133b6a8e8. Every record had the fields
the read's log lookup reads.

| Records | Event | Split | Panel |
|---|---|---|---|
| 55 | read | validation | `outcome` |
| 51 | read | validation | `regressor_heldout` |
| 51 | read | validation | `regressor_seen` |

- **Records other than validation reads:** none.
- **Growth reads:** none. The log held no `growth_seen` or `growth_heldout` read before the margin
  read, so the margins are first (Req 24).
- **Existing openings** under the held-out draw `deddfd4c…`, checked with
  `SelectionLog.openings(seal_names(regime), fingerprint)` (Req 22): none for either regime.

After the read the log has 159 records, SHA-256
51aef24099f0303c218271fc9ab110c3f1a8e7a56198d97eab3c84a0731a2a41. Its first 157 lines still hash
to aa223df2340617dc635ede104905f40a8c5b1adebc1154eda469183133b6a8e8. The read appended exactly
these two validation reads, the record's own `log_records`. The seen regime logged `covariates`,
`one_hot`, `covariates+one_hot`, `ancestors` and `covariates+ancestors`, and the held-out regime
`covariates`, `ancestors` and `covariates+ancestors`. Neither logged `embedding` or `text_only`,
and no outer set was opened.

```json
{"detail": {"comparators": ["covariates", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors"], "level": 6, "run": "margins/17a7e118f926405ea895bd0f908a549c"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "growth_seen", "purpose": "economic margin read: split roadmap Stage 3, plan 15 (Req 24)", "split": "validation", "time": "2026-10-18T00:43:09.849910+00:00"}
{"detail": {"comparators": ["covariates", "ancestors", "covariates+ancestors"], "level": 6, "run": "margins/17a7e118f926405ea895bd0f908a549c"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "growth_heldout", "purpose": "economic margin read: split roadmap Stage 3, plan 15 (Req 24)", "split": "validation", "time": "2026-10-18T00:43:17.260357+00:00"}
```

## 4. What the candidates' spec reads

- **The record:** `~/naics-artifacts/records/economic/growth_margins.json`, SHA-256
  4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0, named `growth-margins`, with
  δ = 0.00015933956325255588 on `growth_seen` and 0.0001619577593991326 on `growth_heldout`.
  `tools decide --margins` takes it.
- **Fixed once.** Never run `tools growth-margins` again. The command refuses an `--output` that
  already exists (`_require_new_record`), so the identical command fails while the record stays
  in place: never move or delete it. A run to another `--output` is not refused; it would log a
  second margin read, and a candidate read before a record's `fixed_at` is refused under it.
- **Where candidates read.** `decide` pairs a candidate only where the reference's predictions are
  bit-identical to this read's. Read candidates on the Mac, at the `uv.lock`
  (6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948) that
  30aeff652f7fa8a90715da831e9948c085c95e5d carries, with Python 3.12.12, numpy 2.3.4 on
  Accelerate and no BLAS thread variable set. Bit identity was verified on fixtures only. If a
  real candidate read fails to pair, suspect BLAS threading first. The check fails closed.
- **Open items.** The deferred items under `## 14-economic-decision-tooling` in
  `specs/deferred_items.md` go to the candidates' spec. Plan 15 checked this record's stored
  scores once; the stored-scores item stays open.

## Reproduction

Do not re-run the read: the margins are fixed once. The commands are recorded so that their
inputs can be checked.

```bash
cd /Users/lowell/Projects/naics-embedder && uv sync --locked
cd /Users/lowell/Projects/naics-embedder && uv sync --locked --check
cd /Users/lowell/Projects/naics-embedder && pwd && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools growth-margins \
  --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet \
  --name growth-margins \
  --store /Users/lowell/naics-artifacts \
  --output /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --purpose 'economic margin read: split roadmap Stage 3, plan 15 (Req 24)' \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --lock /Users/lowell/Projects/naics-embedder/uv.lock \
  > /tmp/naics_plan15/read.txt 2>&1; echo "exit=$?"
shasum -a 256 /Users/lowell/naics-artifacts/records/economic/growth_margins.json
```
