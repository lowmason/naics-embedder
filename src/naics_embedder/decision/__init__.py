'''
Req 5's decision procedure over D8's three panels (roadmap Stage 4; D8, D10, D11).

- ``scores``: each panel's per-unit scores and its decision statistic (D10).
- ``resampling``: the paired two-stage bootstrap, units shared by every arm, seeds nested.
- ``rule``: the intervals, non-inferiority, superiority, the non-dominated set, the tie order.
- ``records``: the arm, margin and decision records.
- ``store``: the content-addressed store the records' artifact references point into.
- ``decide``: margins from a reference arm, and a decision over arms, with every guard.
- ``sweep``: the seed-sweep driver that runs a configuration for N seeds.
'''
