# Relatedness Panel API

The relatedness panel judges relatedness on ITA co-ownership (roadmap Stage 6). ITA co-ownership
judges relatedness and never trains it (U42). The panel splits the co-ownership tables by anchor
industry. Its validation part selects, and its sealed part opens once, at roadmap Stage 7b (U43,
U45).

## One role per table or part

| Table or part | Read by | Role for relatedness |
|---|---|---|
| Codebook codes and lineage | Panel, model | The codebook (P1), no role: candidates, D* and tree edges |
| The tree's parent–child edges | Stage 7 | Training |
| The text-only table (D9), unreduced | Stage 7, panel | Training input (node features), and the two text reference arms. It is codebook text, not an embedding-data table |
| `ita_area_mix.parquet`, 2023 | Stage 7 | Training, after G1 |
| `ita_co_ownership.parquet` and `co_ownership_marginals.parquet`, validation part | Stages 6b and 7 | Selection: the reference, the margin, checkpoints and the forward candidate, after G6 |
| The same tables, sealed part | The opening stage (§4.8) | Judging, read once |
| `co_ownership_support.parquet`, the cross-year link table | Nothing | Not read |
| ITA text–code pairs | Nothing | Off-limits: text→code's (protocol C3) |
| TRI parent links | Nothing yet | Judging, only through an amendment before the sealed opening, with its own margin record first (§4.1) |
| ACS, CPS, BEA and USAspending families | Nothing | Unassigned (§4.1) |

No co-ownership row enters training (Req 26). Stage 6 reads fixture tables only.

## Split

::: naics_embedder.panels.relatedness_split

## Panel and selection log

::: naics_embedder.panels.relatedness

## Relatedness scores

::: naics_embedder.panels.relatedness_scoring

## Reference arms

::: naics_embedder.panels.relatedness_arms
