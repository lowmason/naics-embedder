# ITA Panel API

The ITA panel judges text→code on ITA employer-written descriptions (roadmap Stage 2). Its
validation part selects, and its sealed part is read once. The outcome panel's validation split is
reported beside the ITA scores and decides nothing. Its test split stays sealed.

## One role per table or part

| Table or part | Read by | Role for text→code |
|---|---|---|
| Codebook text in the bundle (titles, descriptions, examples, exclusions, index entries, cross-reference phrases) | Training | Training (U25) |
| Codebook codes and lineage | Panel | The codebook: candidates and ancestors (P1), no role |
| `ita_text_pairs`, validation part | Stages 4 and 5 | Selection |
| `ita_text_pairs`, sealed part | Embedding roadmap Stage 12 | Judging, read once |
| `ita_text_pair_codes` | Panel | Follows its pair's part and role |
| `ita_text_pair_members`, `naics_2022_paths` | Nothing | Not read. Node ids carry the code (§4.1) |
| Outcome panel, validation split | Reports | Reported only, decides nothing (Req 28) |
| Outcome panel, test split | Nothing | Stays sealed (§4.6) |

No ITA pair enters training (U25). Stage 2 reads fixture tables only.

## Split

::: naics_embedder.panels.ita_split

## Panel and selection log

::: naics_embedder.panels.ita

## Shortlist scores

::: naics_embedder.panels.ita_scoring
