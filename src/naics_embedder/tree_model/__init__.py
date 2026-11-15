'''
Tree model module for NAICS embeddings: the deterministic economic candidates.

The economic artifact's three deterministic candidates (spec ``economic-candidates``, 4.2) are
the text-only table reduced by PCA, the classical scaling of the tree metric D* and the text-only
vectors smoothed over each code's lineage. None of them trains, and none fits ``text_model/`` or
``graph_model/``. ADR 0001 gives a method that fits neither a new model layer beside them, so the
three live here:

  - ``scaling``: classical scaling of a distance matrix, its sign rule and its cut refusals
  - ``smoothing``: the lineage weights and the lineage mean
  - ``builders``: the three builders, each writing a table in the export form and its provenance

The package reads the codebook's text and tree only. It imports nothing from ``decision/``.
'''
