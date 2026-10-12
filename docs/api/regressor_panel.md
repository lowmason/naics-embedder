# Regressor Panel API

Arms' coordinates as regressors for QCEW employment, in two sealed regimes (roadmap Stage 3).
The `outcome` setting picks next-year log employment (`level`, the default) or its log change from
year t (`growth`, Req 21), on the same rows, partition and folds. One opening of a regime's outer
set counts for both outcomes (Req 22).

## QCEW rows

::: naics_embedder.panels.qcew_rows

## Partition and held-out draw

::: naics_embedder.panels.regressor_splits

::: naics_embedder.data.regressor_group_table

## Fitting

::: naics_embedder.panels.ridge

## Text-only comparator

::: naics_embedder.panels.text_only

## Panel

::: naics_embedder.panels.regressor
