'''
The seed-sweep driver on the fixture panels (roadmap Stage 4 Exit): every read logged under its
run, every artifact stored, and a decision over two swept arms.

A synthetic runner stands in for training until Stage 6. The informed arm's encoder puts each
query near its code's axis and its table carries each code's employment level and trend; the
uninformed arm's encoder puts each query near a random code's axis and its table is noise.
'''

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import DecisionRecord, read_record, write_record
from naics_embedder.decision.scores import PANELS
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import SeedArtifacts, run_seed_sweep
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.decision import spec, write_text_only
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, SIX_DIGIT

pytestmark = pytest.mark.unit

SEEDS = (0, 1, 2, 3, 4)
PURPOSE = 'fixture seed sweep'

class SyntheticEncoder:
    '''Codes on their own axes; each query near its code's axis, or a random code's.'''

    def __init__(self, informed: bool, seed: int):
        self.axis = {code: index for index, code in enumerate(SIX_DIGIT)}
        self.informed = informed
        self.rng = np.random.default_rng([seed, int(informed)])

    def _axes(self, codes):
        indices = torch.tensor([self.axis[code] for code in codes])
        return torch.nn.functional.one_hot(indices, len(self.axis)).to(torch.float64)

    def encode_codes(self, codes):
        return self._axes(codes)

    def encode_queries(self, texts):
        codes = [text.split()[0] for text in texts]
        if not self.informed:
            codes = list(self.rng.choice(SIX_DIGIT, size=len(codes)))
        noise = self.rng.normal(0.0, 0.3, size=(len(codes), len(self.axis)))
        return self._axes(codes) + torch.from_numpy(noise)

class SyntheticRunner:
    '''One seed of the informed or the uninformed arm, written under ``directory``.'''

    def __init__(self, directory, informed, signal):
        self.directory = directory
        self.informed = informed
        self.signal = signal

    def run(self, arm_spec, seed):
        rng = np.random.default_rng([seed, int(self.informed), 7])
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.directory / f'{arm_spec.name}-{seed}.ckpt'
        checkpoint.write_bytes(f'{arm_spec.name} {seed}'.encode())
        values = rng.normal(size=(len(CODEBOOK), 3))
        if self.informed:
            values[:, :2] = self.signal + rng.normal(0.0, 0.01, size=(len(CODEBOOK), 2))
        schema = {f'e{index}': pl.Float64 for index in range(3)}
        table = pl.DataFrame({
            'code': list(CODEBOOK)
        }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
        path = self.directory / f'{arm_spec.name}-{seed}.parquet'
        table.write_parquet(path)
        return SeedArtifacts(
            checkpoint=checkpoint,
            table=path,
            encoder=SyntheticEncoder(self.informed, seed),
            distance='cosine',
        )

def _signal(regressor_rows):
    '''Each codebook code's mean outcome and its trend over the feature years.'''

    rows = pl.concat([frame for frame in regressor_rows.values()])
    # yapf: disable
    by_code = (
        rows
        .sort('code', 'feature_year')
        .group_by('code', maintain_order=True)
        .agg(
            level=pl.col('outcome').mean(),
            trend=(pl.col('outcome').last() - pl.col('outcome').first()) / 2,
        )
    )
    # yapf: enable
    table = pl.DataFrame({'code': list(CODEBOOK)}).join(by_code, on='code', how='left')
    return table.select('level', 'trend').fill_null(0.0).to_numpy()

def _role_rows():
    rows, entry = [], 0
    for code in SIX_DIGIT:
        for role in ('examples', 'training', 'validation', 'validation', 'test'):
            rows.append((entry, code, f'{code} entry {entry}', role))
            entry += 1
    return pl.DataFrame(
        rows,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'logs' / 'selection_log.jsonl')

@pytest.fixture
def panels(regressor_rows, log):
    return {
        'outcome_panel': OutcomePanel(_role_rows(), SIX_DIGIT, log),
        'regressor_panel': RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS),
    }

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

def _sweep(name, informed, tmp_path, regressor_rows, panels, store, text_only, **overrides):
    runner = SyntheticRunner(tmp_path / 'runs' / name, informed, _signal(regressor_rows))
    return run_seed_sweep(
        spec(name, dimension=3, **overrides),
        SEEDS,
        runner,
        text_only_table=text_only,
        store=store,
        purpose=PURPOSE,
        **panels,
    )

def test_every_seed_is_read_once_per_panel_and_its_records_are_the_logs(
    tmp_path, regressor_rows, panels, store, text_only, log
):
    arm = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)

    assert [run.seed for run in arm.runs] == list(SEEDS)
    records = log.records()
    assert len(records) == 3 * len(SEEDS)
    for run in arm.runs:
        assert run.log_records == [r for r in records if r['detail']['run'] == run.run_id]
        assert sorted(r['panel'] for r in run.log_records) == sorted(PANELS)
        outcome = next(r for r in run.log_records if r['panel'] == 'outcome')
        assert outcome['detail']['table'] == run.table.matrix_fingerprint
        for record in run.log_records:
            assert (record['event'], record['split']) == ('read', 'validation')
            assert record['detail']['arm_name'] == 'uninformed'
            if record['panel'] != 'outcome':
                assert record['detail']['arm'] == run.table.matrix_fingerprint
                assert record['detail']['text_only'] == arm.text_only.table.matrix_fingerprint
        assert set(run.statistics) == set(PANELS)
    assert arm.panels.outcome == panels['outcome_panel'].fingerprint
    assert arm.panels.regressor == panels['regressor_panel'].fingerprint

def test_the_artifacts_outlive_the_runners_files(
    tmp_path, regressor_rows, panels, store, text_only
):
    arm = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)
    for path in (tmp_path / 'runs' / 'uninformed').iterdir():
        path.unlink()

    for run in arm.runs:
        assert store.resolve(run.checkpoint).read_bytes().startswith(b'uninformed')
        assert pl.read_parquet(store.resolve(run.table)).height == len(CODEBOOK)
        predictions = store.read_frame(run.predictions)
        assert set(predictions.get_column('panel').unique().to_list()) == {
            'regressor_seen', 'regressor_heldout'
        }
        assert store.read_frame(run.decoding).height == 2 * len(SIX_DIGIT)

def test_a_text_only_table_from_another_backbone_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):
    with pytest.raises(ValueError, match='D9'):
        _sweep(
            'uninformed',
            False,
            tmp_path,
            regressor_rows,
            panels,
            store,
            text_only,
            backbone='another/backbone',
        )
    assert log.records() == []

def test_a_decision_over_swept_arms_adopts_the_informed_one(
    tmp_path, regressor_rows, panels, store, text_only
):
    reference = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)
    margins = fix_margins(reference, 1.0, 'fixture reference', store, min_seeds=5)
    informed = _sweep('informed', True, tmp_path, regressor_rows, panels, store, text_only)

    record = decide(
        'fixture decision',
        'does the informed arm beat the uninformed one?',
        [informed, reference],
        margins,
        store,
        replicates=2000,
        bootstrap_seed=20260924,
        min_seeds=5,
    )
    path = write_record(record, tmp_path / 'decision.json')

    assert read_record(path, DecisionRecord) == record
    adopted = next(item for item in record.comparisons if item.a == 'informed')
    assert adopted.adopted
    assert all(panel.non_inferior for panel in adopted.panels)
    assert record.chosen == 'informed'
    report = next(item for item in record.reports if item.arm == 'informed')
    assert report.statistics['outcome'] > 0.5
    assert report.gain['regressor_heldout'].point > 0
