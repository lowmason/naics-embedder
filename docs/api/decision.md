# Decision Rule API

Req 5's decision procedure over the three panels of roadmap D8, and the records that carry it
(roadmap Stage 4); and the economic artifact's decision over the two growth panels, against the
reference `covariates+ancestors` (Req 24); and text→code's margin on the ITA panel.

## Decision statistics

::: naics_embedder.decision.scores

## Paired resampling

::: naics_embedder.decision.resampling

## The rule

::: naics_embedder.decision.rule

## Records and artifacts

::: naics_embedder.decision.records

::: naics_embedder.decision.store

## Margins and decisions

::: naics_embedder.decision.decide

## Seed sweeps

::: naics_embedder.decision.sweep

## The economic decision

::: naics_embedder.decision.economic

## The candidate list

The frozen list of the economic candidates (`conf/data/economic_candidates.yaml`). The decision
requires exactly the candidates it names.

::: naics_embedder.decision.candidates

## Text and tree only

The rule that a candidate's provenance comes from the codebook's text and its tree, and from no
panel, QCEW file or selection log.

::: naics_embedder.decision.provenance

## Trained candidates

The import of a trained candidate from its Stage 7/8 arm record and run directories.

::: naics_embedder.decision.trained

## Text→code margin

The text→code decision on the ITA panel's headline. It fixes the margin δ from the reference arms'
own reads, before any candidate read, and compares a candidate with the reference.

::: naics_embedder.decision.text_to_code
