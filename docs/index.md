# NAICS Hyperbolic Embedding System

The system learns a shared space for North American Industry Classification System (NAICS)
codes and activity queries. One LoRA-adapted transformer reads marked text fields, masked fusion
combines each code's present channels, and one linear map produces a direction and a live radius
on the unit-curvature Lorentz hyperboloid. An optional HGCN stage refines the code graph.

## Text Training

The reference configuration uses dimension 16, masked-mean fusion and three terms: task-query
cross-entropy, code-to-code soft-target cross-entropy, and radial error against taxonomy level.
Every epoch visits two streams, each once: all code anchors and all eligible training queries.
A detached code cache supplies candidates; the current step's code anchors remain live.

The outcome validation panel's MRR selects the earliest best epoch. Each read is recorded in
`monitor_reads.jsonl`, and `epoch_summary.jsonl` records MRR, losses, logit scales and radius
health. CUDA uses `bf16-mixed` for the backbone; geometry and losses remain float32. CPU and
MPS use `32-true`, and the text Trainer uses one device.

## Evaluation and Decisions

A seed sweep reads the outcome panel and both regressor regimes from each run's kept checkpoint.
The arm record carries every epoch's monitor reads. Margins are fixed before candidate reads,
and decisions use the three panels under Req 5. Structural metrics describe an arm through
`tools diagnostics`; they do not select the text checkpoint or schedule its learning rate.

## Guides

- [Quickstart](quickstart.md): prepare the bundle and start one run.
- [Text training](text_training.md): the objective, cache, monitor, exact resume and campaign.
- [System overview](overview.md): geometry, architecture and HGCN diagnostics.
- [CLI usage](usage.md): commands, options and artifacts.
- [HGCN training](hgcn_training.md): the separate graph refinement stage.
