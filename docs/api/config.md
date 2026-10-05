# Configuration API

## Reference Text Configuration

```yaml
experiment_name: reference
seed: 42
supervision:
  contract_version: stage3-supervision-v2
  manifest_path: null
model:
  base_model_name: sentence-transformers/all-MiniLM-L6-v2
  fusion: masked_mean
  dimension: 16
  radius_bound: 8.0
loss:
  code_code_weight: 1.0
  radial_weight: 1.0
  target_temperature: 1.0
  radial_step: 1.0
  logit_scale_init: 1.0
  logit_scale_range: [0.01, 100.0]
data_loader:
  queries_per_step: 128
training:
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_epochs: 1
  lr_plateau_factor: 0.5
  lr_plateau_patience: 2
  early_stopping_patience: 5
  trainer:
    max_epochs: 40
    accelerator: auto
    devices: 1
    precision: bf16-mixed
    gradient_clip_val: 1.0
    accumulate_grad_batches: 1
```

The complete config also supplies LoRA/MoE settings and tokenization/description paths.
`manifest_path: null` is a valid pre-generation state; `train` then refuses at the mandatory
bundle gate and names the command to generate a bundle. The objective is `req11-v1`, and
retired loss, curvature, mining, curriculum and streaming-path overrides are rejected.

CUDA uses the configured `bf16-mixed` backbone precision. Fusion, projection, head, geometry and
losses remain float32; CPU and MPS use `32-true`. Text training uses one device. The learned
scales have no weight decay. `training.trainer.val_check_interval` remains a parsed field but is
unread; the outcome monitor runs at each epoch end without a validation loader.

`train --checkpoint-load-mode exact` is the sole load mode. Exact resume requires the same
contract, seed, experiment directory and 21 run settings, including effective precision and
budget. Old objective checkpoints cannot migrate weights. See
[exact resume](../text_training.md#exact-resume).

## Reference

::: naics_embedder.utils.config
