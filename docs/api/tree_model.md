# Tree Model API

The economic artifact's three deterministic candidates. Each is built from the codebook's text and
its tree, and nothing trains.

- The text-only builder reduces the shared text-only table to d dimensions by PCA.
- The tree builder applies classical scaling to the tree metric D*. D* is the path length between
  two codes in the NAICS tree. Classical scaling gives coordinates from the leading eigenvectors
  of a distance matrix.
- The smoothed builder replaces each text-only vector by its weighted mean over the code's
  lineage, then reduces it by PCA. The lineage is the code's chain of ancestors, down from its
  sector.

## Classical scaling

::: naics_embedder.tree_model.scaling

## Lineage smoothing

::: naics_embedder.tree_model.smoothing

## The builders

::: naics_embedder.tree_model.builders
