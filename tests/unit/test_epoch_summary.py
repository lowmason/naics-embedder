'''The durable float64 epoch summary and its model hooks (spec 4.4, P20).'''

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from naics_embedder.text_model.monitor import MonitorRead
from tests.fixtures.epoch_summary import sample_health, summary_rows

pytestmark = pytest.mark.unit

def _module():
    return importlib.import_module('naics_embedder.text_model.epoch_summary')

def test_a_fresh_summary_starts_absent_and_appends_exact_health_values(tmp_path):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    summary = module.EpochSummary(path)
    summary.start(resumed_epoch=None)
    assert not path.exists()
    health = sample_health()
    health['loss/task'] = 1 / 7
    summary.append(epoch=0, mrr=0.123456789012345, health=health)
    assert module.read_epoch_summary(path) == [{'epoch': 0, 'mrr': 0.123456789012345, **health}]
    assert float(torch.tensor(health['loss/task'], dtype=torch.float32)) != health['loss/task']

@pytest.mark.parametrize('resumed_epoch', [0, 1, 2])
def test_resume_keeps_exact_lines_through_the_restored_epoch(tmp_path, resumed_epoch):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    lines = [json.dumps(row, indent=None, sort_keys=False) for row in summary_rows()]
    path.write_text('\n'.join(lines) + '\n')
    summary = module.EpochSummary(path)
    summary.start(resumed_epoch=resumed_epoch)
    assert path.read_text().splitlines() == lines[:resumed_epoch + 1]
    summary.append(epoch=resumed_epoch + 1, mrr=0.75, health=sample_health())
    assert [row['epoch']
            for row in module.read_epoch_summary(path)] == list(range(resumed_epoch + 2))
    assert not path.with_name(path.name + '.tmp').exists()

def test_a_fresh_fit_refuses_an_existing_summary_without_mutating_it(tmp_path):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    path.write_text(json.dumps(summary_rows()[0]) + '\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        module.EpochSummary(path).start(resumed_epoch=None)
    assert path.read_bytes() == before

def test_resume_requires_the_prior_summary_file(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        _module().EpochSummary(tmp_path / 'absent.jsonl').start(resumed_epoch=1)

@pytest.mark.parametrize('epoch', [-1, True, 0.5])
def test_summary_refuses_invalid_epochs(tmp_path, epoch):
    with pytest.raises(ValueError, match='epoch'):
        _module().EpochSummary(tmp_path / 'summary.jsonl'
                               ).append(epoch=epoch, mrr=0.5, health=sample_health())

@pytest.mark.parametrize('epoch', [0, 1])
def test_append_refuses_repeated_or_earlier_epochs_without_mutating(tmp_path, epoch):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    path.write_text(''.join(json.dumps(row) + '\n' for row in summary_rows()[:2]))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='repeat|reorder'):
        module.EpochSummary(path).append(epoch=epoch, mrr=0.75, health=sample_health())
    assert path.read_bytes() == before

@pytest.mark.parametrize(
    'field,value', [
        ('mrr', float('nan')), ('mrr', True), ('mrr', 1.1), ('loss/task', float('inf')),
        ('loss/task', '1.0')
    ]
)
def test_malformed_rows_are_refused_before_the_file_changes(tmp_path, field, value):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    row = summary_rows()[0]
    row[field] = value
    path.write_text(json.dumps(row) + '\n')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        module.read_epoch_summary(path)
    with pytest.raises(ValueError):
        module.EpochSummary(path).start(resumed_epoch=0)
    assert path.read_bytes() == before
    fresh_path = tmp_path / 'fresh.jsonl'
    health = sample_health()
    mrr = value if field == 'mrr' else 0.5
    if field != 'mrr':
        health[field] = value
    with pytest.raises(ValueError):
        module.EpochSummary(fresh_path).append(epoch=0, mrr=mrr, health=health)
    assert not fresh_path.exists()

def test_summary_records_an_unmonitored_epoch_without_inventing_an_mrr(tmp_path):
    module = _module()
    path = tmp_path / module.EPOCH_SUMMARY
    module.EpochSummary(path).append(epoch=0, mrr=None, health=sample_health())
    assert module.read_epoch_summary(path)[0]['mrr'] is None

def test_model_appends_the_float64_health_dict_and_the_monitors_mrr(
    tmp_path, reference_arm_model, reference_arm_code_rows, monkeypatch
):
    model = reference_arm_model
    path = tmp_path / 'checkpoints' / 'epoch_summary.jsonl'
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(code_rows=reference_arm_code_rows),
        current_epoch=0,
        checkpoint_callback=SimpleNamespace(dirpath=str(path.parent)),
        callback_metrics={'loss/task': torch.tensor(1 / 7)},
        logger=None
    )
    model.trainer = trainer
    model.log = Mock()
    monitor = SimpleNamespace(
        start=Mock(), append=Mock(), read=Mock(return_value=MonitorRead(0, 0.123456789012345, {}))
    )
    model.monitor = monitor
    model.on_train_start()
    values = sample_health()
    values['loss/task'] = 1 / 7
    monkeypatch.setattr(model, '_log_health', lambda: values)
    monkeypatch.setattr(model, '_step_plateau', Mock())
    model.on_train_epoch_end()
    assert path.exists()
    row = json.loads(path.read_text())
    assert row == {'epoch': 0, 'mrr': 0.123456789012345, **values}
    assert row['loss/task'] != trainer.callback_metrics['loss/task'].item()
    assert monitor.start.call_args.kwargs == {'resumed_epoch': None}
    assert monitor.append.call_count == 1

def test_model_resume_starts_the_summary_at_fit_start_not_checkpoint_load(
    tmp_path, reference_arm_model, reference_arm_code_rows
):
    model = reference_arm_model
    path = tmp_path / 'epoch_summary.jsonl'
    rows = summary_rows()
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    before = path.read_bytes()
    checkpoint = {
        'stage3_supervision': model.checkpoint_contract.model_dump(),
        'training_run': 'resume-run',
        'epoch': 1
    }
    model.on_load_checkpoint(checkpoint)
    assert path.read_bytes() == before
    model.trainer = SimpleNamespace(
        datamodule=SimpleNamespace(code_rows=reference_arm_code_rows),
        current_epoch=2,
        checkpoint_callback=SimpleNamespace(dirpath=str(tmp_path)),
        logger=None
    )
    model.log = Mock()
    model.on_train_start()
    assert json.loads(path.read_text().splitlines()[-1])['epoch'] == 1
    assert model.training_run == 'resume-run'
