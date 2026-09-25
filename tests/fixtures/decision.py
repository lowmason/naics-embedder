'''
Synthetic arms with known effects on D8's three panels (roadmap Stage 4 Exit).

Every arm is scored on one set of items per panel, each with a base value. Seed k of an arm
scores item i at base_i + effect + offset_k on the outcome panel (reciprocal rank: higher is
better) and at base_i − effect + offset_k on each regressor regime (squared error: lower is
better), so a positive effect favours the arm on every panel. Effects are given in units of
``SIGMA``. The five offsets are fixed, ``SIGMA`` × (−2, −1, 0, 1, 2), so every arm's across-seed
standard deviation is ``SIGMA`` × √2.5 on every panel, and δ = multiple × that. Paired Δ cancels
the base values exactly: only the effects and the seed draws remain.

The records carry real artifact references (a store under the test's tmp path) and log records
shaped as the panels write them.
'''

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import (
    DECISION_STATISTIC,
    HELDOUT_PANEL,
    PANELS,
    SCORE_SCHEMA,
    SEEN_PANEL,
    panel_statistic,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.regressor_panel import text_only_table as stub_text_only_table

SIGMA = 0.01
SEED_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)
SD = SIGMA * float(np.sqrt(2.5))
BACKBONE = 'tiny-backbone'
REVISION = 'abc123'
DESCRIPTIONS_SHA256 = 'd' * 64
MAX_LENGTH = 16
PANEL_SET = PanelSet(
    outcome='outcome-roles',
    regressor='heldout-draw',
    fit_settings={
        'alphas': [0.1, 1.0],
        'folds': 2,
        'repeats': 2,
        'inner_folds': 2,
        'fold_seed': 20260924,
        'min_groups': 4,
    },
)
GROUPS = tuple(str(1100 + index) for index in range(30))
CODES = tuple(f'{group}{suffix}' for group in GROUPS for suffix in ('11', '12'))
# The sparse comparators never read the arm: their error is the base plus a constant
SPARSE_EXTRA = {SEEN_PANEL: 0.02, HELDOUT_PANEL: 0.03}

def spec(name: str, **overrides) -> ArmSpec:
    '''An arm spec reading the synthetic backbone and text.'''

    fields = {
        'name': name,
        'components': 1,
        'dimension': 16,
        'geometry': 'hyperbolic',
        'backbone': BACKBONE,
        'backbone_revision': REVISION,
        'descriptions_sha256': DESCRIPTIONS_SHA256,
        'max_length': MAX_LENGTH,
        **overrides,
    }
    return ArmSpec(**fields)

def _items(panel: str) -> pl.DataFrame:
    '''The panel's items (unit, item, feature_year) with their base values.'''

    rng = np.random.default_rng(len(panel))
    if panel == OUTCOME_PANEL:
        units = [code for code in CODES for _ in range(2)]
        items = [str(index) for index in range(len(units))]
        years: List[Optional[int]] = [None] * len(units)
        base = rng.uniform(0.2, 0.7, size=len(units))
    else:
        feature_years = (2023, ) if panel == SEEN_PANEL else (2022, 2023)
        rows = [(code, year) for code in CODES for year in feature_years]
        units = [code[:4] for code, _ in rows]
        items = [f'{code}/{year}' for code, year in rows]
        years = [year for _, year in rows]
        base = rng.uniform(0.05, 0.5, size=len(units))
    return pl.DataFrame(
        {
            'unit': units,
            'item': items,
            'feature_year': years,
            'base': base
        },
        schema={
            'unit': pl.Utf8,
            'item': pl.Utf8,
            'feature_year': pl.Int32,
            'base': pl.Float64
        },
    )

def synthetic_scores(effects: Mapping[str, float], offset: float) -> pl.DataFrame:
    '''One seed's scores (``SCORE_COLUMNS``), effects in units of ``SIGMA``.'''

    parts = []
    for panel in PANELS:
        items = _items(panel)
        base = pl.col('base')
        effect = effects.get(panel, 0.0) * SIGMA
        if panel == OUTCOME_PANEL:
            statistics = {
                'mrr': base + effect + offset,
                'top1': (base > 0.5).cast(pl.Float64),
                'hit_at_1': (base > 0.5).cast(pl.Float64),
                'hit_at_5': pl.lit(1.0),
                'hit_at_10': pl.lit(1.0),
                'lca_level': pl.lit(5.0),
            }
        else:
            sparse = 'covariates+one_hot' if panel == SEEN_PANEL else 'covariates+ancestors'
            statistics = {
                'covariates': base + 0.05,
                'covariates+embedding': base - effect + offset,
                sparse: base + SPARSE_EXTRA[panel],
            }
        for statistic, value in statistics.items():
            parts.append(
                items.select(
                    panel=pl.lit(panel),
                    statistic=pl.lit(statistic),
                    unit='unit',
                    item='item',
                    feature_year='feature_year',
                    value=value,
                )
            )
    return pl.concat(parts).cast(SCORE_SCHEMA)

def write_text_only(
    directory: Path, revision: str = REVISION, codes: Sequence[str] = CODES
) -> Path:
    '''A text-only table with its provenance, as ``tools text-only-table`` writes them.'''

    directory.mkdir(parents=True, exist_ok=True)
    table = stub_text_only_table(codes)
    path = directory / 'text_only.parquet'
    table.write_parquet(path)
    provenance = {
        'backbone': BACKBONE,
        'revision': revision,
        'descriptions': {
            'path': 'naics_descriptions.parquet',
            'sha256': DESCRIPTIONS_SHA256
        },
        'max_length': MAX_LENGTH,
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': text_only_fingerprint(table),
    }
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def _table(directory: Path, name: str, seed: int, dimension: int) -> Path:
    '''An arm's code table in the export form.'''

    values = np.random.default_rng([seed, len(name)]).normal(size=(len(CODES), dimension))
    schema = {f'e{index}': pl.Float64 for index in range(dimension)}
    table = pl.DataFrame({
        'code': list(CODES)
    }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
    path = directory / f'{name}-{seed}.parquet'
    table.write_parquet(path)
    return path

def _read(panel: str, run_id: str, table: str, text_only: str, time: str) -> Dict:
    '''A log record shaped as the panel writes it.'''

    detail = {'run': run_id, 'arm_name': run_id.split('/')[0], 'seed': int(run_id.split('-')[-1])}
    if panel == OUTCOME_PANEL:
        detail.update(encoder='SyntheticEncoder', distance='cosine', table=table)
        fingerprint = PANEL_SET.outcome
    else:
        detail.update(level=6, comparators=[], arm=table, text_only=text_only, dimension=16)
        fingerprint = PANEL_SET.regressor
    return {
        'time': time,
        'event': 'read',
        'panel': panel,
        'split': 'validation',
        'purpose': 'synthetic arm',
        'fingerprint': fingerprint,
        'n_queries': 0,
        'detail': detail,
    }

def synthetic_arm(
    store: ArtifactStore,
    directory: Path,
    arm_spec: ArmSpec,
    effects: Mapping[str, float],
    *,
    offsets: Sequence[float] = SEED_OFFSETS,
    text_only_table: Optional[Path] = None,
) -> ArmRecord:
    '''An arm record whose seeds score ``synthetic_scores``, read now.'''

    directory = Path(directory) / arm_spec.name
    directory.mkdir(parents=True, exist_ok=True)
    text_only = store.put_text_only(text_only_table or write_text_only(directory / 'text'))
    runs = []
    for seed, offset in enumerate(offsets):
        run_id = f'{arm_spec.name}/seed-{seed}'
        checkpoint = directory / f'checkpoint-{seed}.ckpt'
        checkpoint.write_bytes(f'{arm_spec.name} {seed}'.encode())
        table = store.put_table(_table(directory, arm_spec.name, seed, arm_spec.dimension))
        scores = synthetic_scores(effects, offset * SIGMA)
        time = datetime.now(timezone.utc).isoformat()
        runs.append(
            SeedRun(
                seed=seed,
                run_id=run_id,
                checkpoint=store.put(checkpoint),
                table=table,
                scores=store.put_frame(scores, 'scores.parquet'),
                decoding=store.put_frame(pl.DataFrame({'query_id': [seed]}), 'decoding.parquet'),
                predictions=store.put_frame(
                    pl.DataFrame({'code': [CODES[0]]}), 'predictions.parquet'
                ),
                statistics={
                    panel: panel_statistic(scores, panel, DECISION_STATISTIC[panel])
                    for panel in PANELS
                },
                log_records=[
                    _read(
                        panel, run_id, table.matrix_fingerprint, text_only.table.matrix_fingerprint,
                        time
                    ) for panel in PANELS
                ],
            )
        )
    return ArmRecord(
        spec=arm_spec,
        text_only=text_only,
        store=str(store.root),
        panels=PANEL_SET,
        runs=runs,
        created_at=datetime.now(timezone.utc),
    )
