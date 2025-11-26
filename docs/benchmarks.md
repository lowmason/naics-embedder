# Sampling Benchmarks

Static negative sampling strategies were requested in [Issue #43](https://github.com/lowmason/naics-embedder/issues/43) to provide an ablation against the dynamic Structure-Aware Dynamic Curriculum (SADC). This page captures the knobs and expected metrics for reproducing those comparisons.

## 1. Config Switches

| Strategy | Config Override | Notes |
| --- | --- | --- |
| **SADC (default)** | _none_ | Uses inverse tree-distance weighting + curriculum flags. |
| **SANS (static)** | `sampling.strategy=sans_static` | Enables fixed near/far buckets. Tune `sampling.sans_static.*` for bucket ratios. |

Suggested defaults:

```yaml
sampling:
  strategy: sans_static
  sans_static:
    near_distance_threshold: 4.0
    near_bucket_weight: 0.65
    far_bucket_weight: 0.35
```

## 2. Metrics to Track

The dataloader now emits `batch['sampling_metadata']` so the model logs:

- `train/sans_static/sample_near_pct` – share of sampled negatives within the near bucket.
- `train/sans_static/candidate_near_pct` – share of candidate pool tagged near.
- `train/sans_static/effective_near_weight` – probability mass assigned to the near bucket after normalization.

Log these side-by-side with validation metrics to understand whether static sampling under- or over-selects close neighbors.

## 3. Example Commands

```bash
# Dynamic curriculum (baseline)
uv run naics-embedder train --config conf/config.yaml

# Static near/far sampling with custom weights
uv run naics-embedder train --config conf/config.yaml \
  sampling.strategy=sans_static \
  sampling.sans_static.near_bucket_weight=0.7 \
  sampling.sans_static.far_bucket_weight=0.3
```

Record validation MAP, hierarchy metrics, and the three SANS traces per run. This keeps the benchmarking script lightweight while still honoring the request from Issue #43 for a static baseline.

