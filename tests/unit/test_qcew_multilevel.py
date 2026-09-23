'''
Unit tests for the multi-level QCEW benchmark summary and comparison table.

The QCEW loader and the per-level ridge regressions are replaced by table-driven fakes, so these
tests check what run_qcew_multilevel_benchmark does with per-level results: which levels it
skips, how it names result keys, and what its summary aggregates.

The benchmark visits levels in ascending order, and the largest level here (6) is skipped. When
its loop ends, `level` is 6 and `level_results` still holds level 4's results. A summary
comprehension that binds one name but reads one of these leftovers raises KeyError or silently
collapses the summary onto a single level. Ruff's F rules don't flag that, so the exact summary
assertions below have to.
'''

import copy
from pathlib import Path
from statistics import mean, pstdev

import polars as pl
import pytest

from naics_embedder.metrics import qcew
from naics_embedder.metrics.qcew import (
    QCEWMultilevelConfig,
    print_multilevel_comparison,
    run_qcew_multilevel_benchmark,
)

# -------------------------------------------------------------------------------------------------
# Fake per-level stages
# -------------------------------------------------------------------------------------------------

# Level 1 has no NAICS_LEVEL_NAMES entry; levels 3 and 6 are skipped.
LEVELS = (1, 2, 3, 4, 6)

# The fake loader returns an empty QCEW slice for these levels, so they never reach the runner.
EMPTY_SLICE_LEVELS = {6}

# What the fake runner returns for each level: None skips level 3, and one-hot beats embedding on
# level 4. Every summarized metric differs across levels, so a summary that repeats one level's
# values fails.
LEVEL_RESULTS: dict[int, dict[str, dict[str, float]] | None] = {
    1: {
        'embedding': {
            'r2': 0.82,
            'rmse': 0.40
        },
        'one_hot': {
            'r2': 0.61,
            'rmse': 0.55
        },
        'hybrid': {
            'r2': 0.85,
            'rmse': 0.38
        },
        'metadata': {
            'n_samples': 40.0
        },
    },
    2: {
        'embedding': {
            'r2': 0.70,
            'rmse': 0.50
        },
        'one_hot': {
            'r2': 0.52,
            'rmse': 0.65
        },
        'hybrid': {
            'r2': 0.75,
            'rmse': 0.46
        },
        'metadata': {
            'n_samples': 20.0
        },
    },
    3: None,
    4: {
        'embedding': {
            'r2': 0.45,
            'rmse': 0.80
        },
        'one_hot': {
            'r2': 0.58,
            'rmse': 0.70
        },
        'hybrid': {
            'r2': 0.60,
            'rmse': 0.66
        },
        'metadata': {
            'n_samples': 300.0
        },
    },
}

@pytest.fixture
def fake_level_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    '''Replace the QCEW loader and the per-level regressions with the tables above.'''

    def fake_load_qcew_slice(config, code_length=None):
        if code_length in EMPTY_SLICE_LEVELS:
            return pl.DataFrame()
        # Any non-empty frame will do: the fake runner never reads it.
        return pl.DataFrame({'industry_code': ['11']})

    def fake_run_single_level_benchmark(qcew_df, embed_df, embed_cols, config, level):
        # Copy so the benchmark can't alias or mutate the shared table.
        return copy.deepcopy(LEVEL_RESULTS[level])

    monkeypatch.setattr(qcew, '_load_qcew_slice', fake_load_qcew_slice)
    monkeypatch.setattr(qcew, '_run_single_level_benchmark', fake_run_single_level_benchmark)

def _make_config(tmp_path: Path, levels: tuple[int, ...] = LEVELS) -> QCEWMultilevelConfig:
    '''Write the two input files the benchmark checks for and return a config over `levels`.'''

    embedding_parquet = tmp_path / 'embeddings.parquet'
    embeddings = pl.DataFrame({'code': ['11'], 'hgcn_e0': [1.0], 'hgcn_e1': [0.0]})
    embeddings.write_parquet(embedding_parquet)

    # The fake loader never reads the CSV; the benchmark only checks that it exists.
    qcew_csv = tmp_path / 'qcew.csv'
    qcew_csv.touch()

    return QCEWMultilevelConfig(
        qcew_csv_path=qcew_csv, embedding_parquet=embedding_parquet, levels=levels
    )

# -------------------------------------------------------------------------------------------------
# run_qcew_multilevel_benchmark
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('fake_level_stages')
def test_results_keep_only_evaluated_levels(tmp_path: Path):
    results = run_qcew_multilevel_benchmark(_make_config(tmp_path))

    # Level 1's key falls back to 'unknown'. Level 3 (runner returned None) and level 6 (empty
    # slice) are left out.
    assert list(results) == [
        'level_1_unknown',
        'level_2_sector',
        'level_4_industry_group',
        'summary',
    ]
    assert results['level_1_unknown'] == LEVEL_RESULTS[1]
    assert results['level_2_sector'] == LEVEL_RESULTS[2]
    assert results['level_4_industry_group'] == LEVEL_RESULTS[4]

@pytest.mark.unit
@pytest.mark.usefixtures('fake_level_stages')
def test_summary_aggregates_only_evaluated_levels(tmp_path: Path):
    summary = run_qcew_multilevel_benchmark(_make_config(tmp_path))['summary']

    # Copied by hand from LEVEL_RESULTS for evaluated levels 1, 2 and 4, in that order.
    embedding_r2 = [0.82, 0.70, 0.45]
    one_hot_r2 = [0.61, 0.52, 0.58]
    hybrid_r2 = [0.85, 0.75, 0.60]
    embedding_rmse = [0.40, 0.50, 0.80]
    one_hot_rmse = [0.55, 0.65, 0.70]

    # pytest.approx rejects nested dicts, so it wraps each float instead. std_r2 is the
    # population standard deviation (numpy's default, ddof=0).
    assert summary == {
        'levels_evaluated': {
            '1': 'unknown',
            '2': 'sector',
            '4': 'industry_group'
        },
        'embedding': {
            'avg_r2': pytest.approx(mean(embedding_r2)),
            'std_r2': pytest.approx(pstdev(embedding_r2)),
            'avg_rmse': pytest.approx(mean(embedding_rmse)),
        },
        'one_hot': {
            'avg_r2': pytest.approx(mean(one_hot_r2)),
            'std_r2': pytest.approx(pstdev(one_hot_r2)),
            'avg_rmse': pytest.approx(mean(one_hot_rmse)),
        },
        'hybrid': {
            'avg_r2': pytest.approx(mean(hybrid_r2)),
            'std_r2': pytest.approx(pstdev(hybrid_r2)),
        },
        'comparison': {
            'embedding_vs_onehot_r2_diff': pytest.approx(mean(embedding_r2) - mean(one_hot_r2)),
            # Embedding wins levels 1 and 2; one-hot wins level 4.
            'embedding_wins': 2,
            'onehot_wins': 1,
            'per_level_r2_diff': {
                'level_1': pytest.approx(0.21),
                'level_2': pytest.approx(0.18),
                'level_4': pytest.approx(-0.13),
            },
        },
    }

@pytest.mark.unit
@pytest.mark.usefixtures('fake_level_stages')
def test_every_level_skipped_returns_empty_results(tmp_path: Path):
    # Level 3's runner returns None and level 6's slice is empty, so there is nothing to summarize.
    assert run_qcew_multilevel_benchmark(_make_config(tmp_path, levels=(3, 6))) == {}

# -------------------------------------------------------------------------------------------------
# print_multilevel_comparison
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('fake_level_stages')
def test_comparison_table_is_aligned_with_signed_diffs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    results = run_qcew_multilevel_benchmark(_make_config(tmp_path))
    capsys.readouterr()  # Discard anything printed while the benchmark ran.

    print_multilevel_comparison(results)

    # Numbers are right-aligned under their headers, and the Δ column always carries a sign.
    assert capsys.readouterr().out.splitlines() == [
        '',
        '=' * 80,
        'QCEW MULTI-LEVEL BENCHMARK RESULTS',
        '=' * 80,
        '',
        'Level                     Embedding R²   One-Hot R²    Hybrid R²   Δ (Emb-OH)        N',
        '-' * 80,
        '1 Unknown                        0.820        0.610        0.850       +0.210       40',
        '2 Sector                         0.700        0.520        0.750       +0.180       20',
        '4 Industry Group                 0.450        0.580        0.600       -0.130      300',
        '-' * 80,
        'AVERAGE                          0.657        0.570        0.733       +0.087',
        '',
        '-' * 40,
        'SUMMARY',
        '-' * 40,
        'Embedding wins: 2 levels',
        'One-hot wins:   1 levels',
        'Avg R² improvement: +0.087',
        '=' * 80,
        '',
    ]

@pytest.mark.unit
def test_comparison_without_summary_prints_only_the_frame(capsys: pytest.CaptureFixture[str]):
    # {} is what the benchmark returns when every level is skipped.
    print_multilevel_comparison({})

    assert capsys.readouterr().out.splitlines() == [
        '',
        '=' * 80,
        'QCEW MULTI-LEVEL BENCHMARK RESULTS',
        '=' * 80,
        '',
        'Level                     Embedding R²   One-Hot R²    Hybrid R²   Δ (Emb-OH)        N',
        '-' * 80,
        '=' * 80,
        '',
    ]
