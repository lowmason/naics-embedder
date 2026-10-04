# Input Window API

The backbone's trained input window (Req 9, roadmap Stage 5): no tokenizing path reads beyond
it, a channel text over it is read as its window-fitting summary (roadmap Stage 6b), and the
supervision bundle records each channel's texts beyond it.

::: naics_embedder.utils.input_window
