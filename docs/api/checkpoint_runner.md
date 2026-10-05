# Checkpoint Runner API

All-seed preflight compares saved constructor hyperparameters in both last and selected
checkpoints against the current arm config: LoRA rank/alpha/dropout and, under active MoE,
expert count/top-k/hidden dimension/load-balancing coefficient. Missing required controls fail
closed; the arm retains its 21 `settings` keys.

All-seed preflight validates complete monitor epochs, settings and the unambiguous earliest best
checkpoint before exports or decision reads. The returned artifacts carry the monitor records.

::: naics_embedder.text_model.checkpoint_runner
