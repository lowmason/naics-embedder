'''Radius and term diagnostics on fixture tables and tiny backbones (spec 4.2 and 6).'''

import importlib
import json
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from typer.testing import CliRunner

from naics_embedder.cli.commands import tools
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.export import export_code_table
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
    build_reference_model,
    lightning_checkpoint,
    reference_step_dataset,
)

pytestmark = pytest.mark.unit

CODES = ('31', '44', '311', '441', '3111', '4411', '31111', '44111', '311111', '441111')

def _module():
    return importlib.import_module('naics_embedder.text_model.radius_report')

def _table(radii=None):
    radii = np.linspace(0.8, 7.8, len(CODES)) if radii is None else np.asarray(radii)
    directions = np.random.default_rng(7).normal(size=(len(CODES), 16))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    matrix = radii[:, None] * directions
    return pl.DataFrame(
        {
            'code': CODES,
            'index': range(len(CODES)),
            'level': [len(code) for code in CODES],
            **{
                f'e{i}': matrix[:, i]
                for i in range(matrix.shape[1])
            }
        }
    )

def test_report_quantifies_every_radius_criterion():
    radii = np.linspace(0.8, 7.8, len(CODES))
    report = _module().radius_report(_table(radii), anchor_radius_gradient=np.array([0.2, -0.3]))
    assert report.passed
    assert report.failures == ()
    assert report.anchor_radius_gradient == (0.2, -0.3)
    assert report.anchor_gradient_nonzero is True
    assert report.level_sd == pytest.approx(
        {level: (radii[1] - radii[0]) / 2
         for level in range(2, 7)}
    )
    assert report.sector_radii == pytest.approx(dict(zip(CODES[:2], radii[:2])))
    assert report.sector_min_gap == pytest.approx(radii[1] - radii[0])
    assert report.max_radius == pytest.approx(7.8)
    assert report.max_radius_code == CODES[-1]
    assert report.manifold_error <= report.manifold_tolerance
    assert report.manifold_tolerance == pytest.approx(1e-9 * np.cosh(7.8)**2)
    assert report.pairs == len(CODES)**2
    assert report.zero_distance_pairs == len(CODES)
    assert report.max_relative_error <= 1e-3
    json.dumps(asdict(report), allow_nan=False)

@pytest.mark.parametrize(
    'gradient', [np.array([1.0, 0.0]),
                 np.array([np.nan]),
                 np.array([np.inf]),
                 np.array([])]
)
def test_every_anchor_must_have_a_finite_nonzero_radius_gradient(gradient):
    report = _module().radius_report(_table(), anchor_radius_gradient=gradient)
    assert not report.passed
    assert report.anchor_gradient_nonzero is False
    assert any('anchor' in reason for reason in report.failures)
    json.dumps(asdict(report), allow_nan=False)

def test_a_table_only_report_marks_the_gradient_unmeasured():
    report = _module().radius_report(_table())
    assert report.anchor_gradient_nonzero is None
    assert report.anchor_radius_gradient is None
    assert not report.passed
    assert any('not supplied' in reason for reason in report.failures)

def test_a_capped_table_fails_the_level_spread_and_sector_gap():
    report = _module().radius_report(
        _table(np.full(len(CODES), 2.0)), anchor_radius_gradient=np.ones(3)
    )
    assert not report.passed
    assert all(sd == pytest.approx(0) for sd in report.level_sd.values())
    assert report.sector_min_gap == pytest.approx(0)
    assert any('SD' in reason for reason in report.failures)
    assert any('distinct' in reason for reason in report.failures)

def test_the_level_sd_boundary_is_strict():
    radii = np.linspace(0.8, 7.8, len(CODES))
    radii[-2:] = (0.0, 0.002)
    report = _module().radius_report(_table(radii), anchor_radius_gradient=np.ones(2))
    assert report.level_sd[6] <= 1e-3
    assert any('level 6' in reason for reason in report.failures)

def test_a_nonpositive_sector_radius_fails():
    radii = np.linspace(0.8, 7.8, len(CODES))
    radii[0] = 0
    report = _module().radius_report(_table(radii), anchor_radius_gradient=np.ones(2))
    assert any('positive' in reason for reason in report.failures)

def test_no_sector_pair_leaves_the_gap_unmeasured():
    report = _module().radius_report(
        _table().filter(pl.col('level') > 2), anchor_radius_gradient=np.ones(2)
    )
    assert report.sector_min_gap is None
    assert any('two sectors' in reason for reason in report.failures)

def test_the_manifold_check_uses_the_largest_radius(monkeypatch):
    module = _module()
    original = module.exp_map_origin

    def distorted(tangent):
        points = original(tangent)
        points[-1, 0] *= 1.001
        return points

    monkeypatch.setattr(module, 'exp_map_origin', distorted)
    report = module.radius_report(_table(), anchor_radius_gradient=np.ones(2))
    assert report.manifold_error > report.manifold_tolerance
    assert any('manifold' in reason for reason in report.failures)

def test_all_pairs_are_checked_in_bounded_row_chunks(monkeypatch):
    module = _module()
    count = 70
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(count, 16))
    table = pl.DataFrame(
        {
            'code': [str(100000 + i) for i in range(count)],
            'level': [6] * count,
            **{
                f'e{i}': matrix[:, i]
                for i in range(16)
            }
        }
    )
    calls = []
    original = module.polar_distance

    def observed(radius_a, direction_a, radius_b, direction_b):
        calls.append((len(radius_a), len(radius_b), radius_a.dtype))
        return original(radius_a, direction_a, radius_b, direction_b)

    monkeypatch.setattr(module, 'polar_distance', observed)
    report = module.radius_report(table, anchor_radius_gradient=np.ones(2))
    assert report.pairs == count**2
    assert calls == [
        (32, count, torch.float32), (32, count, torch.float32), (6, count, torch.float32)
    ]

def test_an_error_in_the_final_chunk_fails_the_strict_all_pairs_check(monkeypatch):
    module = _module()
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(70, 16))
    table = pl.DataFrame(
        {
            'code': [str(100000 + i) for i in range(70)],
            'level': [6] * 70,
            **{
                f'e{i}': matrix[:, i]
                for i in range(16)
            }
        }
    )
    original = module.polar_distance

    def inaccurate(radius_a, direction_a, radius_b, direction_b):
        distances = original(radius_a, direction_a, radius_b, direction_b)
        if len(radius_a) == 6:
            distances[-1, 0] *= 1.002
        return distances

    monkeypatch.setattr(module, 'polar_distance', inaccurate)
    report = module.radius_report(table, anchor_radius_gradient=np.ones(2))
    assert report.max_relative_error > 1e-3
    assert any('relative' in reason for reason in report.failures)

def test_mathematical_zero_pairs_record_read_cancellation_separately(monkeypatch):
    table = _table()
    report = _module().radius_report(table, anchor_radius_gradient=np.ones(2))
    assert report.zero_distance_pairs == len(CODES)
    assert report.zero_training_max_error == 0.0
    assert report.zero_read_max_error > 0.0
    assert report.passed
    # Different codes at one point are also mathematical zeros, not just the diagonal.
    matrix = table.select([f'e{i}' for i in range(16)]).to_numpy().copy()
    matrix[-1] = matrix[-2]
    duplicate = table.with_columns(**{f'e{i}': matrix[:, i] for i in range(16)})
    report = _module().radius_report(duplicate, anchor_radius_gradient=np.ones(2))
    assert report.zero_distance_pairs == len(CODES) + 2
    assert report.zero_training_max_error == 0.0

    module = _module()
    original = module.polar_distance

    def nonzero_at_coincidence(*args):
        distances = original(*args)
        distances[0, 0] = 1e-4
        return distances

    monkeypatch.setattr(module, 'polar_distance', nonzero_at_coincidence)
    report = module.radius_report(table, anchor_radius_gradient=np.ones(2))
    assert report.zero_training_max_error == pytest.approx(1e-4)
    assert not report.passed
    assert any('mathematically coincident' in reason for reason in report.failures)

def test_a_noncoincident_pair_read_as_zero_fails_without_a_denominator_floor():
    table = _table()
    matrix = table.select([f'e{i}' for i in range(16)]).to_numpy().copy()
    matrix[-2:] = 0
    matrix[-2, 0] = 2.0
    matrix[-1, 0] = 2.0 + 1e-10
    near = table.with_columns(**{f'e{i}': matrix[:, i] for i in range(16)})
    report = _module().radius_report(near, anchor_radius_gradient=np.ones(2))
    assert report.nonzero_pairs_read_as_zero >= 2
    assert not report.passed
    assert any('noncoincident' in reason for reason in report.failures)

@pytest.mark.parametrize('fusion', ['masked_mean', 'attention', 'moe'])
def test_terms_scales_and_every_anchor_radius_have_nonzero_gradient(
    fusion, tiny_backbone, reference_manifest, reference_bundle, reference_arm_code_rows,
    reference_arm_steps, monkeypatch
):
    model = build_reference_model(
        reference_manifest, reference_bundle, fusion=fusion, moe_hidden_dim=16
    )
    model.eval()
    model.refresh_code_cache(reference_arm_code_rows)
    model.encoder.projection.train()
    flags = [module.training for module in model.modules()]
    before = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
            before[name] = parameter.grad.clone()
    log = model.log
    utilization = model._log_expert_utilization
    with warnings.catch_warnings(record=True) as caught:
        result = _module().term_gradients(model, reference_arm_steps[0])
    assert not caught
    expected = {'task', 'code_code', 'radial', 'logit_scale_task', 'logit_scale_code'}
    if fusion == 'moe':
        expected.add('load_balancing')
    assert all(np.isfinite(result[name]) and result[name] > 0 for name in expected)
    anchor_keys = {key for key in result if key.startswith('anchor_radius/')}
    assert len(anchor_keys) == len(reference_arm_steps[0]['codes']['ids'])
    assert all(np.isfinite(result[key]) and result[key] != 0 for key in anchor_keys)
    with monkeypatch.context() as quiet:
        quiet.setattr(model, '_log_expert_utilization', lambda *args, **kwargs: None)
        losses = model.compute_losses(reference_arm_steps[0])
        expected_radius = torch.autograd.grad(losses.total, losses.anchor_radius)[0]
    assert (expected_radius < 0).any(), expected_radius
    measured_radius = torch.tensor(
        [result[f'anchor_radius/{row}'] for row in range(len(expected_radius))],
        dtype=expected_radius.dtype
    )
    assert torch.equal(measured_radius, expected_radius.detach().cpu())
    assert set(result) == expected | anchor_keys
    assert flags == [module.training for module in model.modules()]
    assert model.log == log
    assert model._log_expert_utilization == utilization
    for name, parameter in model.named_parameters():
        if name in before:
            assert torch.equal(parameter.grad, before[name])

def test_disabled_terms_are_reported_as_inert(
    reference_arm_model, reference_arm_code_rows, reference_arm_steps
):
    model = reference_arm_model
    model.hparams.code_code_weight = 0.0
    model.hparams.radial_weight = 0.0
    model.refresh_code_cache(reference_arm_code_rows)
    result = _module().term_gradients(model, reference_arm_steps[0])
    assert result['code_code'] == 0
    assert result['radial'] == 0
    assert result['logit_scale_code'] == 0

@pytest.fixture
def cli_arm(
    tmp_path, monkeypatch, request, tiny_backbone, reference_manifest, reference_bundle,
    reference_arm_token_config, reference_arm_code_rows
):
    model = build_reference_model(
        reference_manifest,
        reference_bundle,
        fusion=getattr(request, 'param', 'masked_mean'),
        moe_hidden_dim=16
    ).eval()
    with torch.no_grad():
        model.encoder.projection.weight.mul_(3)
    model.hparams.seed = 7
    model.hparams.run_settings = {'queries_per_step': 4}
    checkpoint = tmp_path / 'selected.ckpt'
    torch.save(lightning_checkpoint(model), checkpoint)
    table = export_code_table(
        checkpoint, reference_bundle, reference_arm_token_config, tmp_path / 'table.parquet'
    )
    cfg = Config().override(
        {
            'supervision.manifest_path': str(reference_bundle.manifest_path),
            'data_loader.streaming.descriptions_parquet': reference_arm_token_config
            .descriptions_parquet
        }
    )
    config = tmp_path / 'run.yaml'
    cfg.to_yaml(config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tools, 'pick_device', lambda *args: 'cpu')
    monkeypatch.setattr(tools, 'code_token_config', lambda cfg: reference_arm_token_config)
    return SimpleNamespace(
        checkpoint=checkpoint,
        table=table,
        config=config,
        bundle=reference_bundle,
        rows=reference_arm_code_rows,
        model=model
    )

def _cli_args(arm, output):
    return [
        'radius-report', '--checkpoint',
        str(arm.checkpoint), '--table',
        str(arm.table), '--config',
        str(arm.config), '--output',
        str(output), 'data_loader.queries_per_step=128', 'seed=0'
    ]

@pytest.mark.parametrize('cli_arm', ['masked_mean', 'moe'], indirect=True)
def test_cli_runs_the_saved_seeds_first_batch_without_reading_a_panel(
    cli_arm, tmp_path, minilm_tokenizer, monkeypatch
):
    monkeypatch.setattr(
        tools.OutcomePanel, 'score', lambda *args, **kwargs: pytest.fail('panel read')
    )
    output = tmp_path / 'report.json'
    result = CliRunner().invoke(tools.app, _cli_args(cli_arm, output))
    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads(output.read_text())
    assert report['passed'] is True
    assert report['seed'] == 7
    assert report['epoch'] == report['step'] == 0
    dataset = reference_step_dataset(cli_arm.bundle, cli_arm.rows, minilm_tokenizer, seed=7)
    dataset.set_epoch(0)
    assert report['anchor_ids'] == dataset[0]['codes']['ids'].tolist()
    assert len(report['radius']['anchor_radius_gradient']) == len(report['anchor_ids'])
    expected = {'task', 'code_code', 'radial', 'logit_scale_task', 'logit_scale_code'}
    if cli_arm.model.fusion == 'moe':
        expected.add('load_balancing')
    assert set(report['term_gradients']) == expected
    assert not list(tmp_path.rglob('*selection_log*'))

    original = tools.term_gradients

    def inert_radial(*args):
        gradients = original(*args)
        gradients['radial'] = 0.0
        return gradients

    monkeypatch.setattr(tools, 'term_gradients', inert_radial)
    failed_output = tmp_path / 'inert.json'
    failed = CliRunner().invoke(tools.app, _cli_args(cli_arm, failed_output))
    assert failed.exit_code == 1, failed.output
    failed_report = json.loads(failed_output.read_text())
    assert failed_report['passed'] is False
    assert failed_report['inert_terms'] == ['radial']
    assert failed_report['radius']['failures'] == []
    assert not list(tmp_path.rglob('*selection_log*'))

def test_cli_writes_a_failed_capped_report_and_exits_one(cli_arm, tmp_path):
    table = pl.read_parquet(cli_arm.table)
    matrix = table.select([f'e{i}' for i in range(16)]).to_numpy()
    matrix *= 2 / np.linalg.norm(matrix, axis=1, keepdims=True)
    table.with_columns(**{f'e{i}': matrix[:, i] for i in range(16)}).write_parquet(cli_arm.table)
    path = provenance_path(cli_arm.table)
    provenance = json.loads(path.read_text())
    provenance['table_sha256'] = sha256_file(cli_arm.table)
    path.write_text(json.dumps(provenance))
    output = tmp_path / 'failed.json'
    result = CliRunner().invoke(tools.app, _cli_args(cli_arm, output))
    assert result.exit_code == 1, result.output
    report = json.loads(output.read_text())
    assert report['passed'] is False
    assert any('SD' in reason for reason in report['radius']['failures'])

def test_cli_refuses_a_table_from_another_checkpoint(cli_arm, tmp_path):
    checkpoint = tmp_path / 'other.ckpt'
    checkpoint.write_bytes(b'other checkpoint')
    args = _cli_args(cli_arm, tmp_path / 'report.json')
    args[2] = str(checkpoint)
    result = CliRunner().invoke(tools.app, args)
    assert result.exit_code == 1
    assert 'another checkpoint' in result.output
    assert not (tmp_path / 'report.json').exists()
