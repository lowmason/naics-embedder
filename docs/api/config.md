# Config API

## Stage-3 supervision and structural preference

Repaired Stage-3 training reads one immutable supervision bundle and configures the structural
preference loss that replaces LambdaRank:

```yaml
supervision:
  mode: repaired
  contract_version: stage3-supervision-v1
  manifest_path: null  # data supervision prints the exact immutable path to set before training

loss:
  temperature: 0.07
  curvature: 1.0
  base_margin: 0.5
  hierarchy_weight: 0.45
  structural_preference:
    weight: 0.35
    margin: 0.1
    temperature: 1.0
    tie_tolerance: 0.000001
  radius_reg_weight: 0.10
  level_radius_weight: 0.15
```

- `supervision.manifest_path: null` is a valid pre-generation state: the configuration parses, and
  `train` stops at the mandatory supervision gate with the command that generates a bundle.
- `loss.structural_preference.temperature` must be positive; `margin` and `tie_tolerance` must be
  nonnegative.
- Repaired configurations reject `loss.rank_order_weight` (a legacy LambdaRank setting; configure
  `loss.structural_preference` instead) and `data_loader.streaming.phase1_exclusion_weight` (the
  one-slot exclusion quota owns exclusion representation). The old key is never reinterpreted as
  the new loss because the objectives differ.
- Both legacy keys are accepted only with the explicit `supervision.mode: legacy_containment`.
- `train --checkpoint-load-mode [exact|weights_only]` selects exact resume (identical supervision
  contract) or an explicit weights-only migration; see the
  [Training Guide](../text_training.md#exact-resume-versus-weights-only-migration).

## Reference

::: naics_embedder.utils.config
