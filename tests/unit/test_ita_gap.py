'''
The gap between Census phrases and owner text (plan 20, Task 3; D1, D2, P5, P6).

``arm_gap`` is checked on hand-made frames whose means are known, against ``arm_replicates`` for
the headline, and on codes that one panel lacks or a pair reports twice. ``gap_seeds`` refuses
reads that do not pair, and ``gap_record`` is reproducible on ``gap_fixture``'s six reads.
'''

import math

import numpy as np
import polars as pl
import pytest

from naics_embedder.decision import ita_gap
from naics_embedder.decision.ita_gap import (
    COPIED_STREAM,
    GAP_LEVEL,
    MATCHED_STREAM,
    GapSeed,
    ItaGapRecord,
    arm_gap,
    gap_record,
    gap_seeds,
)
from naics_embedder.decision.ita_reads import ItaReferenceRead
from naics_embedder.decision.records import ArmRecord, read_record
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.text_to_code import arm_replicates, headline_values
from naics_embedder.panels.ita_scoring import ItaResult
from tests.fixtures.ita import gap_fixture

pytestmark = pytest.mark.unit

REPLICATES = 1000
SEED = 1

# -------------------------------------------------------------------------------------------------
# Hand-made frames
# -------------------------------------------------------------------------------------------------

def _per_pair(rows):
    '''A per-pair frame: (pair_id, description_key, copied, reciprocal_rank, codes) per row.'''

    return pl.DataFrame(
        {
            'pair_id': [row[0] for row in rows],
            'description_key': [row[1] for row in rows],
            'copied': [row[2] for row in rows],
            'reciprocal_rank': [float(row[3]) for row in rows],
            'codes': [list(row[4]) for row in rows],
        },
        schema={
            'pair_id': pl.Utf8,
            'description_key': pl.Utf8,
            'copied': pl.Boolean,
            'reciprocal_rank': pl.Float64,
            'codes': pl.List(pl.Utf8),
        },
    )

def _census(rows):
    '''Census rows: (code, query id, reciprocal rank) per row.'''

    return pl.DataFrame(
        {
            'unit': [row[0] for row in rows],
            'item': [row[1] for row in rows],
            'value': [float(row[2]) for row in rows],
        },
        schema={
            'unit': pl.Utf8,
            'item': pl.Utf8,
            'value': pl.Float64
        },
    )

# H: 0.5 and 0.0 over two keys; K: 1.0 and 0.5; C: 1.0 and 0.0
BASE_PAIRS = [
    ('a', 'k1', False, 0.5, ['111110']),
    ('b', 'k2', False, 0.0, ['311111']),
    ('c', 'k3', True, 1.0, ['513210']),
    ('d', 'k4', True, 0.5, ['513210']),
]
BASE_CENSUS = [('111110', 'q1', 1.0), ('311111', 'q2', 0.0)]

def _gap(seeds, arm='lexical-stub', r=0.05, replicates=REPLICATES, bootstrap_seed=SEED):
    return arm_gap(arm, seeds, r=r, replicates=replicates, bootstrap_seed=bootstrap_seed)

# -------------------------------------------------------------------------------------------------
# arm_gap
# -------------------------------------------------------------------------------------------------

def test_points_are_the_means_and_their_differences():
    gap = _gap([GapSeed(None, _per_pair(BASE_PAIRS), _census(BASE_CENSUS))])

    assert (gap.arm, gap.n_seeds) == ('lexical-stub', 1)
    assert gap.headline.point == pytest.approx(0.25)
    assert gap.copied.point == pytest.approx(0.75)
    assert gap.census.point == pytest.approx(0.5)
    assert gap.headline_gap.point == pytest.approx(0.25)
    assert gap.copied_gap.point == pytest.approx(-0.25)
    assert gap.wording_gap.point == pytest.approx(0.5)
    assert gap.threshold == pytest.approx(0.025)
    # The matched codes are 111110 (H 0.5, C 1.0) and 311111 (H 0.0, C 0.0)
    assert (gap.n_matched_codes, gap.n_unmatched_codes) == (2, 0)
    assert gap.census_matched.point == pytest.approx(0.5)
    assert gap.headline_matched.point == pytest.approx(0.25)
    assert gap.matched_headline_gap.point == pytest.approx(0.25)

def test_each_interval_is_ordered_and_a_resampled_one_has_width():
    gap = _gap([GapSeed(None, _per_pair(BASE_PAIRS), _census(BASE_CENSUS))])

    for estimate in (
        gap.headline, gap.copied, gap.census, gap.headline_gap, gap.copied_gap, gap.wording_gap,
        gap.census_matched, gap.headline_matched, gap.matched_headline_gap
    ):
        lower, upper = estimate.interval
        assert lower <= upper
    # Two units each, so a resample can draw one unit twice: the intervals are not points
    assert gap.headline.interval[0] < gap.headline.interval[1]
    assert GAP_LEVEL == 0.95

def _three_seeds():
    '''Three seeds' frames: the same pairs and queries, other values.'''

    seeds = []
    for shift in (0.0, 0.1, 0.2):
        pairs = [
            ('a', 'k1', False, 0.5 + shift, ['111110']),
            ('b', 'k1', False, 0.2 + shift, ['111110']),
            ('c', 'k2', False, 0.1 + shift, ['311111']),
            ('d', 'k3', False, 0.6 - shift, ['311111']),
            ('e', 'k4', False, 0.3, ['211120']),
            ('f', 'k5', True, 0.9 - shift, ['513210']),
            ('g', 'k6', True, 0.4 + shift, ['513210']),
        ]
        census = [
            ('111110', 'q1', 0.7 - shift),
            ('111110', 'q2', 0.4),
            ('311111', 'q3', 0.2 + shift),
            ('211120', 'q4', 0.9),
            ('513210', 'q5', 0.5 + shift),
        ]
        seeds.append((_per_pair(pairs), _census(census)))
    return seeds

def test_the_headline_estimate_is_arm_replicates():
    frames = _three_seeds()
    seeds = [GapSeed(index + 1, pairs, census) for index, (pairs, census) in enumerate(frames)]
    results = [
        ItaResult(
            per_pair=pairs, per_code=pl.DataFrame(), headline={}, copied={}, phrasing={}, counts={}
        ) for pairs, _ in frames
    ]
    items = PanelItems.from_values(headline_values(results[0]))

    point, replicates = arm_replicates(
        items, 'euclidean-d16', results, replicates=2000, bootstrap_seed=7
    )
    gap = _gap(seeds, arm='euclidean-d16', replicates=2000, bootstrap_seed=7)

    assert gap.headline.point == point
    assert gap.headline.interval == percentile_interval(replicates, GAP_LEVEL)
    assert gap.n_seeds == 3

def _expected(frames, stream, arm, replicates=REPLICATES, bootstrap_seed=SEED):
    '''One stratum by the plan's algorithm (P5), from the resampling primitives alone.'''

    items = PanelItems.from_values(frames[0])
    sums = items.sums(np.stack([items.values(frame) for frame in frames]))
    units = unit_draws(stream, len(items.units), replicates, bootstrap_seed)
    seeds = seed_draws('ita_text', arm, len(frames), replicates, bootstrap_seed)
    return (
        point_statistic(sums, items.sizes),
        replicate_statistics(sums, items.sizes, units, seeds),
    )

def _pairs(frame, copied):
    return frame.filter(pl.col('copied') == copied
                        ).select(unit='description_key', item='pair_id', value='reciprocal_rank')

def _matched(frame, census, codes):
    '''Each matched code's mean census value and mean headline value, one row per code.'''

    headline = frame.filter(~pl.col('copied')).explode('codes')
    rows_c, rows_h = [], []
    for code in codes:
        values_c = census.filter(pl.col('unit') == code).get_column('value').to_list()
        values_h = headline.filter(pl.col('codes') == code).get_column('reciprocal_rank').to_list()
        rows_c.append((code, code, math.fsum(values_c) / len(values_c)))
        rows_h.append((code, code, math.fsum(values_h) / len(values_h)))
    return _census(rows_c), _census(rows_h)

def test_each_stratum_draws_on_its_own_stream_with_shared_seed_draws():
    frames = _three_seeds()
    seeds = [GapSeed(index + 1, pairs, census) for index, (pairs, census) in enumerate(frames)]

    gap = _gap(seeds, arm='euclidean-d16')

    assert (COPIED_STREAM, MATCHED_STREAM) == ('ita_text/copied', 'ita_text/matched')
    strata = {
        'headline': _expected(
            [_pairs(pairs, False) for pairs, _ in frames], 'ita_text', 'euclidean-d16'
        ),
        'copied': _expected(
            [_pairs(pairs, True) for pairs, _ in frames], COPIED_STREAM, 'euclidean-d16'
        ),
        'census': _expected([census for _, census in frames], 'outcome', 'euclidean-d16'),
    }
    codes = ['111110', '211120', '311111']
    matched = [_matched(pairs, census, codes) for pairs, census in frames]
    strata['census_matched'] = _expected(
        [pair[0] for pair in matched], MATCHED_STREAM, 'euclidean-d16'
    )
    strata['headline_matched'] = _expected(
        [pair[1] for pair in matched], MATCHED_STREAM, 'euclidean-d16'
    )
    for name, (point, replicates) in strata.items():
        estimate = getattr(gap, name)
        assert estimate.point == pytest.approx(point, abs=1e-15), name
        assert estimate.interval == pytest.approx(percentile_interval(replicates, GAP_LEVEL)), name
    differences = {
        'headline_gap': ('census', 'headline'),
        'copied_gap': ('census', 'copied'),
        'wording_gap': ('copied', 'headline'),
        'matched_headline_gap': ('census_matched', 'headline_matched'),
    }
    for name, (first, second) in differences.items():
        estimate = getattr(gap, name)
        assert estimate.point == pytest.approx(strata[first][0] - strata[second][0], abs=1e-15)
        assert estimate.interval == pytest.approx(
            percentile_interval(strata[first][1] - strata[second][1], GAP_LEVEL)
        ), name
    assert (gap.n_matched_codes, gap.n_unmatched_codes) == (3, 0)

def test_the_matched_gap_skips_codes_one_panel_lacks():
    pairs = BASE_PAIRS + [('e', 'k5', False, 0.75, ['236220'])]
    census = BASE_CENSUS + [('111120', 'q3', 0.25)]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))])

    # 236220 has an ITA pair and no census row; 111120 a census row and no ITA pair
    assert (gap.n_matched_codes, gap.n_unmatched_codes) == (2, 1)
    assert gap.census_matched.point == pytest.approx(0.5)
    assert gap.headline_matched.point == pytest.approx(0.25)
    # The unmatched codes stay in the unmatched means
    assert gap.headline.point == pytest.approx((0.5 + 0.0 + 0.75) / 3)
    assert gap.census.point == pytest.approx((1.0 + 0.0 + 0.25) / 3)

def test_the_matched_gap_counts_only_headline_codes():
    # 513210 is reported by copied pairs alone, and has a census row: it is not matched
    census = BASE_CENSUS + [('513210', 'q3', 0.25)]

    gap = _gap([GapSeed(None, _per_pair(BASE_PAIRS), _census(census))])

    assert (gap.n_matched_codes, gap.n_unmatched_codes) == (2, 0)

def test_a_pair_with_two_codes_counts_under_each_in_the_matched_gap():
    pairs = [
        ('a', 'k1', False, 0.5, ['111110', '311111']),
        ('b', 'k2', False, 1.0, ['311111']),
        ('c', 'k3', True, 1.0, ['513210']),
    ]
    census = [('111110', 'q1', 0.0), ('311111', 'q2', 1.0), ('311111', 'q3', 1.0)]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))])

    # H: 111110 → 0.5, 311111 → (0.5 + 1.0) / 2; C: 111110 → 0.0, 311111 → 1.0
    assert gap.n_matched_codes == 2
    assert gap.headline_matched.point == pytest.approx((0.5 + 0.75) / 2)
    assert gap.census_matched.point == pytest.approx(0.5)
    assert gap.matched_headline_gap.point == pytest.approx(0.5 - 0.625)

def test_a_code_mean_is_correctly_rounded():
    # Ten values of 0.1 sum to 1.0 under fsum and to 0.9999999999999999 under +
    pairs = [(f'p{index}', f'k{index}', False, 0.1, ['111110']) for index in range(10)]
    pairs += [('c', 'kc', True, 1.0, ['513210'])]
    census = [('111110', f'q{index}', 0.1) for index in range(10)]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))])

    assert gap.n_matched_codes == 1
    assert gap.headline_matched.point == 0.1
    assert gap.census_matched.point == 0.1

@pytest.mark.parametrize(
    'census_value, pair_value, large', [(1.0, 0.0, True), (0.5, 0.5, False), (0.5, 0.49, False)]
)
def test_large_follows_the_registered_rule(census_value, pair_value, large):
    pairs = [
        (f'p{index}', f'k{index}', False, pair_value, ['111110' if index % 2 else '311111'])
        for index in range(8)
    ]
    pairs += [('c', 'kc', True, 1.0, ['513210'])]
    census = [
        ('111110' if index % 2 else '311111', f'q{index}', census_value) for index in range(8)
    ]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))])

    assert gap.threshold == pytest.approx(0.05 * census_value)
    assert gap.large is large
    assert gap.large is (gap.matched_headline_gap.interval[0] > gap.threshold)

def test_a_part_with_no_copied_pair_or_no_matched_code_is_refused():
    no_copied = [row for row in BASE_PAIRS if not row[2]]
    with pytest.raises(ValueError, match='copied'):
        _gap([GapSeed(None, _per_pair(no_copied), _census(BASE_CENSUS))])
    with pytest.raises(ValueError, match='code'):
        _gap([GapSeed(None, _per_pair(BASE_PAIRS), _census([('111120', 'q1', 1.0)]))])

# -------------------------------------------------------------------------------------------------
# gap_seeds
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def fixture_reads(tmp_path):
    store = ArtifactStore(tmp_path / 'store')
    paths, stage8_path = gap_fixture(store, tmp_path / 'fixture')
    reads = [read_record(path, ItaReferenceRead) for path in paths]
    return store, paths, stage8_path, reads, read_record(stage8_path, ArmRecord)

def test_gap_seeds_take_census_rows_from_the_stored_scores(fixture_reads):
    store, _, _, reads, stage8 = fixture_reads

    seeds = gap_seeds(reads, store, stage8)

    assert list(seeds) == ['lexical-stub', 'euclidean-d16']
    [stub] = seeds['lexical-stub']
    stub_read = next(read for read in reads if read.arm == 'lexical-stub')
    stored = store.read_frame(stub_read.outcome.scores)
    expected = stored.filter((pl.col('panel') == 'outcome')
                             & (pl.col('statistic') == 'mrr')).select('unit', 'item', 'value')
    assert stub.seed is None
    assert stub.census.sort('item').equals(expected.sort('item'))
    assert stub.ita.equals(store.read_frame(stub_read.per_pair))
    assert [seed.seed for seed in seeds['euclidean-d16']] == [1, 2, 3, 4, 5]
    for seed, run in zip(seeds['euclidean-d16'], sorted(stage8.runs, key=lambda run: run.seed)):
        scores = store.read_frame(run.scores)
        census = scores.filter((pl.col('panel') == 'outcome')
                               & (pl.col('statistic') == 'mrr')).select('unit', 'item', 'value')
        assert seed.census.sort('item').equals(census.sort('item'))

def _mutate(reads, stage8, case):
    '''The fixture's reads and record, changed so that they no longer pair.'''

    stub = next(read for read in reads if read.arm == 'lexical-stub')
    d16 = [read for read in reads if read.arm == 'euclidean-d16']
    first, second = d16[0], d16[1]
    changed = {
        'two runs': {
            second.run: second.model_copy(update={'run': 'ita-gap/other'})
        },
        'two evidences': {
            second.run: second.model_copy(
                update={'evidence': second.evidence.model_copy(update={'run': 'other_r1'})}
            )
        },
        'two split fingerprints': {
            second.run: second.model_copy(update={'split_fingerprint': 'f' * 64})
        },
        'two data fingerprints': {
            second.run: second.model_copy(update={'data_fingerprint': 'f' * 64})
        },
        'stub without outcome': {
            stub.run: stub.model_copy(update={'outcome': None})
        },
        'd16 with outcome': {
            second.run: second.model_copy(update={'outcome': stub.outcome})
        },
        "checkpoint not the record's": {
            second.run: second.model_copy(update={'checkpoint_sha256': 'f' * 64})
        },
        "table not the record's": {
            second.run: second.model_copy(update={'table_fingerprint': 'f' * 64})
        },
        "seed not the record's": {
            second.run: second.model_copy(update={'seed': 6})
        },
        'repeated seed': {
            second.run: second.model_copy(update={'seed': first.seed})
        },
        'other outcome split': {
            stub.run: stub.model_copy(
                update={'outcome': stub.outcome.model_copy(update={'data_fingerprint': 'f' * 64})}
            )
        },
        'other outcome roles': {
            stub.run: stub.model_copy(
                update={'outcome': stub.outcome.model_copy(update={'fingerprint': 'f' * 64})}
            )
        },
    }
    if case == 'arm missing':
        return [read for read in reads if read.arm != 'lexical-stub'], stage8
    if case == 'two stub reads':
        return reads + [stub], stage8
    if case == 'a seed missing':
        return [read for read in reads if read is not second], stage8
    target = stub if case in (
        'stub without outcome', 'other outcome split', 'other outcome roles'
    ) else second
    replacement = changed[case][target.run]
    return [replacement if read is target else read for read in reads], stage8

REFUSALS = {
    'two runs': 'values of run',
    'two evidences': 'values of evidence',
    'two split fingerprints': 'values of split_fingerprint',
    'two data fingerprints': 'values of data_fingerprint',
    'stub without outcome': 'no outcome read',
    'd16 with outcome': 'has an outcome read',
    "checkpoint not the record's": 'read checkpoint',
    "table not the record's": 'read table',
    "seed not the record's": "not the Stage 8 record's",
    'repeated seed': 'repeats a seed',
    'other outcome split': 'outcome split',
    'other outcome roles': 'outcome split',
    'arm missing': 'hold arms',
    'two stub reads': 'reads, not one',
    'a seed missing': "not the Stage 8 record's",
}

@pytest.mark.parametrize('case', list(REFUSALS))
def test_gap_seeds_refuse_reads_that_do_not_pair(fixture_reads, case):
    store, _, _, reads, stage8 = fixture_reads
    reads, stage8 = _mutate(reads, stage8, case)

    with pytest.raises(ValueError, match=REFUSALS[case]):
        gap_seeds(reads, store, stage8)

def test_a_tampered_stored_score_is_refused(fixture_reads):
    store, _, _, reads, stage8 = fixture_reads
    run = stage8.runs[0]
    stored = store.resolve(run.scores)
    stored.chmod(0o644)
    stored.write_bytes(b'not the scores')

    with pytest.raises(ValueError):
        gap_seeds(reads, store, stage8)

# -------------------------------------------------------------------------------------------------
# gap_record
# -------------------------------------------------------------------------------------------------

def test_the_record_is_reproducible(fixture_reads):
    store, paths, stage8_path, _, _ = fixture_reads
    settings = {'r': 0.05, 'replicates': 500, 'bootstrap_seed': 20260924}

    first = gap_record(paths, store, stage8_path, **settings)
    second = gap_record(paths, store, stage8_path, **settings)

    assert first.model_dump(exclude={'created_at'}) == second.model_dump(exclude={'created_at'})
    assert [arm.arm for arm in first.arms] == ['lexical-stub', 'euclidean-d16']
    assert first.arms[1].n_seeds == 5
    assert first.created_at.utcoffset() is not None

def test_the_record_names_its_inputs(fixture_reads):
    store, paths, stage8_path, reads, stage8 = fixture_reads

    record = gap_record(paths, store, stage8_path, r=0.05, replicates=200, bootstrap_seed=3)

    assert record.kind == 'ita_gap'
    assert record.run == reads[0].run
    assert record.evidence == reads[0].evidence
    assert record.split_fingerprint == reads[0].split_fingerprint
    assert record.data_fingerprint == reads[0].data_fingerprint
    assert (record.outcome_fingerprint, record.outcome_data_fingerprint) == (
        stage8.panels.outcome, stage8.panels.outcome_data
    )
    assert record.stage8_sha256 == ita_gap.sha256_file(stage8_path)
    assert record.read_sha256 == {path.name: ita_gap.sha256_file(path) for path in paths}
    assert (record.replicates, record.bootstrap_seed, record.level, record.r) == (
        200, 3, GAP_LEVEL, 0.05
    )
    assert ItaGapRecord.model_validate_json(record.model_dump_json()) == record

@pytest.mark.parametrize(
    'stub_large, d16_large, recommend',
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_the_record_recommends_a_training_part_only_when_both_arms_are_large(
    monkeypatch, fixture_reads, stub_large, d16_large, recommend
):
    store, paths, stage8_path, _, _ = fixture_reads
    real = ita_gap.arm_gap
    large = {'lexical-stub': stub_large, 'euclidean-d16': d16_large}

    def forced(arm, seeds, **settings):
        return real(arm, seeds, **settings).model_copy(update={'large': large[arm]})

    monkeypatch.setattr(ita_gap, 'arm_gap', forced)

    record = gap_record(paths, store, stage8_path, r=0.05, replicates=100, bootstrap_seed=1)

    assert record.recommend_training_part is recommend

def test_two_reads_of_one_file_name_are_refused(fixture_reads, tmp_path):
    store, paths, stage8_path, _, _ = fixture_reads
    copy = tmp_path / 'elsewhere' / paths[1].name
    copy.parent.mkdir()
    copy.write_bytes(paths[1].read_bytes())

    with pytest.raises(ValueError, match='file name'):
        gap_record([*paths, copy], store, stage8_path, r=0.05, replicates=100, bootstrap_seed=1)

def test_gap_seeds_order_d16_by_seed_whatever_the_reads_order(fixture_reads):
    store, _, _, reads, stage8 = fixture_reads

    seeds = gap_seeds(list(reversed(reads)), store, stage8)

    assert [seed.seed for seed in seeds['euclidean-d16']] == [1, 2, 3, 4, 5]
    by_seed = {read.seed: read for read in reads if read.arm == 'euclidean-d16'}
    for seed in seeds['euclidean-d16']:
        assert seed.ita.equals(store.read_frame(by_seed[seed.seed].per_pair))

def test_large_reads_the_lower_bound_not_the_upper():
    # Two codes: a gap of 1 on one and 0 on the other, so a resample's matched gap is 0, 0.5 or 1
    pairs = [
        ('a', 'k1', False, 0.0, ['111110']),
        ('b', 'k2', False, 0.0, ['311111']),
        ('c', 'kc', True, 1.0, ['513210']),
    ]
    census = [('111110', 'q1', 1.0), ('311111', 'q2', 0.0)]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))])

    assert gap.matched_headline_gap.interval == (0.0, 1.0)
    assert gap.threshold == pytest.approx(0.025)
    assert gap.large is False

def test_a_gap_equal_to_the_threshold_is_not_large():
    # Exact binary values: C = 1, H = 0.9375, r = 0.0625, so the gap equals r × C exactly
    pairs = [('a', 'k1', False, 0.9375, ['111110']), ('c', 'kc', True, 1.0, ['513210'])]
    census = [('111110', 'q1', 1.0)]

    gap = _gap([GapSeed(None, _per_pair(pairs), _census(census))], r=0.0625)

    assert gap.matched_headline_gap.interval[0] == gap.threshold == 0.0625
    assert gap.large is False

def test_the_records_refuse_unknown_keys_and_assignment(fixture_reads):
    store, paths, stage8_path, _, _ = fixture_reads
    record = gap_record(paths, store, stage8_path, r=0.05, replicates=100, bootstrap_seed=1)

    with pytest.raises(ValueError):
        ItaGapRecord.model_validate({**record.model_dump(), 'chosen': 'lexical-stub'})
    with pytest.raises(ValueError):
        ita_gap.ArmGap.model_validate({**record.arms[0].model_dump(), 'reference': True})
    with pytest.raises(ValueError):
        record.recommend_training_part = True
    with pytest.raises(ValueError):
        record.arms[0].large = True

def test_a_seed_without_a_matched_code_is_refused():
    first = GapSeed(1, _per_pair(BASE_PAIRS), _census(BASE_CENSUS))
    # The second seed's census lacks 311111, which the first seed matches
    second = GapSeed(2, _per_pair(BASE_PAIRS), _census([('111110', 'q1', 1.0)]))

    with pytest.raises(ValueError, match='holds no value'):
        _gap([first, second], arm='euclidean-d16')
