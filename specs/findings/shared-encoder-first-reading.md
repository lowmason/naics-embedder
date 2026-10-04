# Shared encoder first reading: finding

**Status: FINAL (2026-10-03).** Roadmap Stage 6 (`specs/naics-embedding-roadmap.md`).

This finding records the Exit run of plan 8
(`specs/plans/completed/8-shared-encoder-and-projection.md`), where Plan completion moves the
plan:

- one local epoch of the default arm (d = 16, `masked_mean`)
- its export
- the first validation reads of both panels

No sealed split was opened, and nothing was selected. The checkpoint is `last.ckpt`, and every read
below is a validation read. Section 6 lists what later stages read.

## Sources

**Inputs:**

- **Bundle.** `301cce28-539c-42ea-8781-496bbdcf511c` (`stage3-supervision-v2`), cloned from the
  main checkout with `cp -cR`. Loading it re-ran every integrity and relational check. It printed
  codebook fingerprint `4662b826…` and description fingerprint `fe8c54e3…`.
- **Descriptions.** `data/naics_descriptions.parquet`, cloned from the main checkout with `cp -c`.
- **QCEW slices.** The national annual slices for 2022–2025, `2022_US000_annual.csv` through
  `2025_US000_annual.csv`. They were read from `~/Downloads/Data/QCEW` under the sha256 pins in
  `conf/data/regressor_panel.yaml`, and nothing was downloaded.
- **Backbone.** `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, read from the local Hugging Face cache under
  `HF_HUB_OFFLINE=1`.

**Environment:**

- MPS, where the trainer picks `32-true`.
- Python 3.12.12.
- The export's `library_versions`: peft 0.17.1, polars 1.35.1, torch 2.9.1, transformers 4.57.1.

**Commits:**

- Training ran at `fef1a38`.
- The export and the reads ran at `7799e2e`, two commits later. Those commits do two things:
  - The arm encoder now refuses a provenance that lacks the checkpoint hash, the table hash or the
    token window, and one that names another window.
  - They edit docs and tests.

  Their only change to the training path is a docstring, and they change no score.

| File | sha256 |
|---|---|
| `data/naics_descriptions.parquet` | `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36` |
| `checkpoints/sadc_default/last.ckpt` | `8efd4ba79caa2d717556c26579794e44dd7a651260dcee87f51f0f6330fdbf07` |
| `data/plan8/arm_table.parquet` | `5248495bfe17b93e2782bb8103530fc6cc66a1e4e4bd96dfa51984f8ce213a80` |
| `data/plan8/text_only.parquet` | `6bde204e01a197a580ee4db5f146658bcd573c8f2c7d9c1b30db7af061c4d384` |

**Matrix fingerprints:**

- The arm's, from its provenance:
  `b5e322cedb57de48dcd351d6e3d16116155d3b7981573d2ad7a42ae0d7e997c8`.
- The text-only table's, from the regressor records' `detail.text_only`:
  `f6677234d7366babd628fad6b1b0a287c1f21010b1e797b4406792752402f12d`.

## 1. The run

The command is written below with the manifest path as `MANIFEST`. It adds `data_loader.n_epochs=1`
to the plan's Step 4 (see the path to this run, below):

```bash
HF_HUB_OFFLINE=1 nohup uv run naics-embedder train training.trainer.max_epochs=1 data_loader.n_epochs=1 supervision.manifest_path=MANIFEST < /tmp/plan8-train-answer.txt > logs/plan8_train.out 2>&1 &
```

**Timing.**

- The run started at 22:10:39 EDT on 2026-10-03 (`logs/plan8_train.start`).
- The training summary is stamped `2026-10-03T22:40:25.575560`.
- A watcher found the process gone at 22:40:40 EDT, after 29.9 minutes of wall-clock time.
- The memory footprint grew from 11 GiB to a peak of 23.0 GiB across the epoch. The MPS limit is
  47.74 GiB.

**Progress and loss.**

- The training dataset held 3,181 rows. The run trained on them in 199 batches of 16, in 28:29, then
  ran 199 validation batches.
- Training averaged 0.12 batches per second, against 0.22 for the probe at `fef1a38` below. The
  run shared the machine with the final verification's test suites and the Codex review.
- The checkpoint records epoch 0 and global step 100, with `accumulate_grad_batches: 2`.
- The last logged `val/contrastive_loss` is −1.042767. It is in-sample and selects nothing.

**Token cache.** `data/token_cache/token_cache.pt.meta.json` records:

- format `channels-v3`;
- `summaries` null;
- `max_length` 128;
- field markers for title, description, excluded and examples.

**The path to this run.** The user ruled on each change below, and each was reviewed before the run.

- **First attempt.**
  - It was launched at 19:53:59 EDT at `f4f0697`, Task 12's tip. Task 13, which changed only docs,
    ran beside it.
  - It stopped at step 0 of 19,883 with an MPS out-of-memory error: 46.40 GiB allocated, against
    47.74 GiB allowed, in the candidate pool's single backbone call. Its log is
    `logs/plan8_train_oom.out`.
  - One Lightning epoch read all 100 of the datamodule's pre-sampled epochs: 318,124 training rows.
  - R9 measured 5,958 rows in 373 steps at about 9.5 s each. That was not reproduced.
- **Change A (`386c817`).**
  - The backbone now makes one call per field, in chunks of at most 256 texts, each trimmed to its
    own longest text. This amends P4.
  - With dropout off, outputs agree up to float noise.
- **Change B (`d920884`).** `data_loader.n_epochs` sets the datamodule's pre-sampled epoch count.
  The default stays 100. This run set 1.
- **Probe at `d920884`.**
  - It ran out of memory again, at 47.48 GiB.
  - `from_pretrained` returns the backbone in eval mode, and Lightning 2.5.5's `fit` never calls
    `.train()`. So gradient checkpointing never engaged, and BERT's dropout was off.
  - Main's four-copy encoder is built the same way.
- **Change C (`b8d0d7f`).** The backbone is in train mode from construction, so checkpointing
  engages and BERT's dropout (0.1) is on in training.
- **Change D (`fef1a38`).**
  - torch 2.9.1's checkpoint does not replay the MPS random state, so with dropout on the recompute
    drew new masks.
  - On a two-layer probe, the gradients' relative difference from those without checkpointing was
    1.78. A second seed differed by 1.85.
  - Checkpointing now runs non-reentrant, with a context that replays the MPS state. On the real
    MiniLM, the worst relative gradient difference fell from 1.594 to 0.000.
- **Probe at `fef1a38`.** Its first steps peaked at 10.0 GiB and ran at about 4.5 s per batch.

## 2. The export

**Shape.** 2,125 rows. The columns are `code`, `index`, `level` and `e0 … e15` (float64).

| Level | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Codes | 20 | 96 | 308 | 689 | 1,012 |

**Tangent norms.**

- Minimum 0.7688, median 1.7054, maximum 2.0000.
- 0.2038 of the codes sit at the cap of 2. Stage 7 removes the cap (Req 13), so this share is what
  it changes.

**The provenance's contract**, verbatim:

```json
{
  "bundle_id": "301cce28-539c-42ea-8781-496bbdcf511c",
  "codebook_fingerprint": "4662b826d0166eab27b20890ce7463edc60f34082297772bdcebb3d04c6d3f04",
  "contract_version": "stage3-supervision-v2",
  "encoder": {
    "backbone": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 16,
    "fusion": "masked_mean",
    "layout": "shared"
  },
  "mining_contract_version": "negative-selection-v2",
  "structural_preference_loss_version": "structural-preference-v1",
  "supervision_mode": "repaired"
}
```

## 3. The regressor panel

Each cell is RMSE / R² / median penalty over the five repeats, from Step 9's printed lines. Each
comparator has 3,885 rows in the seen regime and 7,770 in the held-out regime.

| `regressor_seen` | Level 6 |
|---|---|
| covariates | 0.3897 / 0.9330 / 0.001 |
| embedding | 1.2964 / 0.2591 / 0.001 |
| covariates+embedding | 0.3009 / 0.9601 / 0.32 |
| one_hot | 0.0601 / 0.9984 / 0.001 |
| covariates+one_hot | 0.0605 / 0.9984 / 0.001 |
| ancestors | 0.6059 / 0.8381 / 0.001 |
| covariates+ancestors | 0.1303 / 0.9925 / 1 |
| text_only | 1.2623 / 0.2975 / 0.001 |
| covariates+text_only | 0.2738 / 0.9669 / 0.001 |

| `regressor_heldout` | Level 6 |
|---|---|
| covariates | 0.3903 / 0.9325 / 0.32 |
| embedding | 1.3430 / 0.2009 / 320 |
| covariates+embedding | 0.3177 / 0.9553 / 10 |
| ancestors | 1.3602 / 0.1804 / 0.001 |
| covariates+ancestors | 0.3163 / 0.9557 / 1 |
| text_only | 1.3228 / 0.2248 / 100 |
| covariates+text_only | 0.2890 / 0.9630 / 1 |

The `embedding` comparator is this arm's table. `text_only` is the frozen backbone reduced by PCA to
16 dimensions (D9). The numbers are a floor for Stage 7's arms, not a target.

## 4. The outcome panel

The arm decodes under Lorentz distance over all 1,012 candidates. The lexical stub's numbers are a
reference column. They come from `specs/findings/outcome-panel-splits.md` section 5, which uses
cosine distance. No decision reads either column: Stage 7 compares arms under Req 5.

| Metric | This arm | Lexical stub |
|---|---:|---:|
| Queries / codes / candidates | 4,042 / 939 / 1,012 | 4,042 / 939 / 1,012 |
| Top-1 accuracy | 0.1249 | 0.5163 |
| MRR | 0.2301 | 0.6165 |
| Hit@1 / Hit@5 / Hit@10 | 0.1249 / 0.3350 / 0.4488 | 0.5163 / 0.7333 / 0.7971 |
| Mean lowest-common-ancestor level | 2.5242 | 4.3355 |

## 5. The selection log

```json
{"detail": {"arm": "b5e322cedb57de48dcd351d6e3d16116155d3b7981573d2ad7a42ae0d7e997c8", "comparators": ["covariates", "embedding", "covariates+embedding", "one_hot", "covariates+one_hot", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "text_only": "f6677234d7366babd628fad6b1b0a287c1f21010b1e797b4406792752402f12d"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_seen", "purpose": "plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)", "split": "validation", "time": "2026-10-04T03:08:12.602300+00:00"}
{"detail": {"arm": "b5e322cedb57de48dcd351d6e3d16116155d3b7981573d2ad7a42ae0d7e997c8", "comparators": ["covariates", "embedding", "covariates+embedding", "ancestors", "covariates+ancestors", "text_only", "covariates+text_only"], "dimension": 16, "level": 6, "text_only": "f6677234d7366babd628fad6b1b0a287c1f21010b1e797b4406792752402f12d"}, "event": "read", "fingerprint": "deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4", "n_queries": 1554, "panel": "regressor_heldout", "purpose": "plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)", "split": "validation", "time": "2026-10-04T03:08:20.016911+00:00"}
{"detail": {"checkpoint": "8efd4ba79caa2d717556c26579794e44dd7a651260dcee87f51f0f6330fdbf07", "distance": "lorentz", "encoder": "ArmEncoder", "table": "b5e322cedb57de48dcd351d6e3d16116155d3b7981573d2ad7a42ae0d7e997c8"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)", "split": "validation", "time": "2026-10-04T03:09:06.985459+00:00"}
```

The log was the worktree's gitignored `logs/selection_log.jsonl`, and these lines are its copy.

## 6. What later stages read

**Stage 6b.** The token cache's `summaries` entry is null, and its format is `channels-v3`.

**Stage 7:**

- `ArmEncoder.from_files` and `read_outcome_validation` (`text_model/arm_encoder.py`).
- `tools export-table` and `tools outcome-panel`.
- The cap of 2, which 0.2038 of the codes sit at (section 2).
- Selection moves to the validation query split (D6).
- The backbone trains with dropout on and gradient checkpointing engaged from the first step
  (Change C, section 1).

**Stages 7–11.** An arm's regressor read takes the exported table as `--coordinates`. Section 5
shows that the three logged names agree:

- the provenance's `matrix_fingerprint`
- the regressor records' `detail.arm`
- the outcome record's `detail.table`

## Reproduction

These are plan 8's Task 14 Steps 2–10, run from the worktree root, with the manifest path written as
`MANIFEST`. Steps 5 and 7 also run check scripts, which the plan gives in full.

```bash
mkdir -p data/supervision/stage3-supervision-v2 data/plan8 logs
cp -cR /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c data/supervision/stage3-supervision-v2/
cp -c /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet data/naics_descriptions.parquet
shasum -a 256 data/naics_descriptions.parquet
uv run python -c "from naics_embedder.supervision.artifacts import load_validated_bundle; b = load_validated_bundle('MANIFEST'); print(b.manifest.bundle_id, b.manifest.codebook_fingerprint[:8], b.manifest.description_fingerprint[:8])"
printf 'n\n' > /tmp/plan8-train-answer.txt
HF_HUB_OFFLINE=1 nohup uv run naics-embedder train training.trainer.max_epochs=1 data_loader.n_epochs=1 supervision.manifest_path=MANIFEST < /tmp/plan8-train-answer.txt > logs/plan8_train.out 2>&1 &
# Once the run has ended:
tail -n 40 logs/plan8_train.out
ls checkpoints/sadc_default
HF_HUB_OFFLINE=1 uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt --output data/plan8/arm_table.parquet supervision.manifest_path=MANIFEST
shasum -a 256 data/plan8/arm_table.parquet checkpoints/sadc_default/last.ckpt
HF_HUB_OFFLINE=1 uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet --output data/plan8/text_only.parquet
uv run naics-embedder tools regressor-panel --coordinates data/plan8/arm_table.parquet --text-only data/plan8/text_only.parquet --codebook data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet --purpose 'plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)' --output data/plan8/regressor_validation.parquet
HF_HUB_OFFLINE=1 uv run naics-embedder tools outcome-panel --checkpoint checkpoints/sadc_default/last.ckpt --table data/plan8/arm_table.parquet --purpose 'plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)' --output data/plan8/outcome_validation.json supervision.manifest_path=MANIFEST
```

Training is not bitwise reproducible on MPS, so a rerun's hashes differ. The panel reads reproduce
from the table only if that table is kept, and it is not committed.
