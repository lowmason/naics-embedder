'''
Unit tests for loss functions: Req 11's three terms and their learned logit scales.

Req 11's three terms follow spec 4.1 and §6 "Task term" and "Code–code term". The task term sums
probability over each query's targets T within its candidates C. The code–code term is the cross
entropy to softmax(−D* / τ_t) over J_a, every code but the anchor and its unary partner. The radial
term pulls each code to ρ(λ − 1). The logit scales are learned, start at their init and stay in
their range.
'''

import math
from typing import Any, Dict

import pytest
import torch

from naics_embedder.text_model import loss as terms

# -------------------------------------------------------------------------------------------------
# Req 11's terms: steps and references
# -------------------------------------------------------------------------------------------------

def _distances(rows: int, codes: int, seed: int) -> torch.Tensor:
    '''Seeded float64 distances in [0, 4).'''

    generator = torch.Generator().manual_seed(seed)
    return 4 * torch.rand(rows, codes, generator=generator, dtype=torch.float64)

def _scale(value: float) -> torch.Tensor:
    '''A float64 0-d scale, so a float64 reference meets no float32 rounding of the scale.'''

    return torch.tensor(value, dtype=torch.float64)

def _task_step() -> Dict[str, torch.Tensor]:
    '''
    Three queries over six codes. The second and third have two targets each, as a phrase that
    names two codes at its level does. Every query has a candidate that is not a target, so every
    candidate takes gradient.
    '''

    candidates = torch.tensor(
        [
            [True, True, True, False, False, True],
            [False, True, True, True, True, False],
            [True, False, True, True, False, True],
        ]
    )
    targets = torch.tensor(
        [
            [False, True, False, False, False, False],
            [False, False, True, False, True, False],
            [True, False, False, True, False, False],
        ]
    )
    return {
        'distances': _distances(3, 6, seed=11),
        'scale': _scale(1.5),
        'candidates': candidates,
        'targets': targets,
    }

def _reference_task_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    candidates: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    '''Spec 4.1(i) by hand: each query's softmax over its own C, summed over its T.'''

    per_query = []
    for row in range(distances.shape[0]):
        columns = candidates[row].nonzero().squeeze(1)
        probabilities = torch.softmax(-scale * distances[row, columns], dim=0)
        per_query.append(-torch.log(probabilities[targets[row, columns]].sum()))
    return torch.stack(per_query).mean()

def _code_code_step() -> Dict[str, Any]:
    '''
    Three anchors, codes 0, 2 and 4, among six. Codes 0 and 1 are a unary pair, as are 4 and 5, and
    code 2 has no partner, so J_a drops {0, 1}, {2} and {4, 5}. D*(a, a) and d(a, a) are 0, and each
    partner is the nearest code on both sides, so unmasked they would take most of the probability.
    '''

    rows = torch.arange(3)
    anchors = torch.tensor([0, 2, 4])
    partners = torch.tensor([1, -1, 5])
    paired = partners >= 0
    keep = torch.ones(3, 6, dtype=torch.bool)
    keep[rows, anchors] = False
    keep[rows[paired], partners[paired]] = False
    generator = torch.Generator().manual_seed(12)
    structural = 1 + 5 * torch.rand(3, 6, generator=generator, dtype=torch.float64)
    structural[rows, anchors] = 0.0
    structural[rows[paired], partners[paired]] = 0.5
    distances = _distances(3, 6, seed=13)
    distances[rows, anchors] = 0.0
    distances[rows[paired], partners[paired]] = 0.1
    return {
        'distances': distances,
        'scale': _scale(1.5),
        'structural': structural,
        'keep': keep,
        'target_temperature': 1.0,
    }

def _reference_code_code_loss(
    distances: torch.Tensor,
    scale: torch.Tensor,
    structural: torch.Tensor,
    keep: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    '''Spec 4.1(ii) by hand: over each anchor's own J_a, CE(p_a, softmax of −s_c · d).'''

    per_anchor = []
    for row in range(distances.shape[0]):
        kept = keep[row]
        target = torch.softmax(-structural[row, kept] / target_temperature, dim=0)
        log_model = torch.log_softmax(-scale * distances[row, kept], dim=0)
        per_anchor.append(-(target * log_model).sum())
    return torch.stack(per_anchor).mean()

def _mean_target_entropy(structural: torch.Tensor, keep: torch.Tensor, temperature: float) -> float:
    '''The mean over anchors of the entropy of p_a = softmax over J_a of −D* / τ_t.'''

    entropies = []
    for row in range(structural.shape[0]):
        target = torch.softmax(-structural[row, keep[row]] / temperature, dim=0)
        entropies.append(-(target * target.log()).sum())
    return torch.stack(entropies).mean().item()

# -------------------------------------------------------------------------------------------------
# Req 11's task term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTaskLoss:
    '''Spec 4.1(i): the mean over queries of −log Σ_{t ∈ T} softmax_C(−s_q · d)_t.'''

    def test_two_targets_and_a_negative_at_one_distance_cost_minus_log_two_thirds(self):
        '''Probability is summed over T: the two targets hold 2/3 of it together.'''

        loss = terms.task_loss(
            distances=torch.ones(1, 3, dtype=torch.float64),
            scale=_scale(1.0),
            candidates=torch.tensor([[True, True, True]]),
            targets=torch.tensor([[True, True, False]]),
        )

        # The mean of each target's −log p would be log 3, as would a single target
        assert loss.item() == pytest.approx(-math.log(2 / 3), rel=1e-12)

    @pytest.mark.parametrize('scale', [0.5, 1.0, 3.0])
    def test_the_loss_is_minus_the_log_of_the_probability_summed_over_the_targets(self, scale):
        '''The value and its gradient in the distances and the scale are the by-hand reference's.'''

        step = _task_step()
        distances = step['distances'].clone().requires_grad_(True)
        scales = _scale(scale).requires_grad_(True)
        inputs = {**step, 'distances': distances, 'scale': scales}

        loss = terms.task_loss(**inputs)
        reference = _reference_task_loss(**inputs)

        torch.testing.assert_close(loss, reference, rtol=1e-12, atol=0.0)
        # A normalizer over T that took no gradient would keep the value and push every target away
        gradients = torch.autograd.grad(loss, (distances, scales))
        expected = torch.autograd.grad(reference, (distances, scales))
        for got, want in zip(gradients, expected):
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)

    def test_the_softmax_runs_over_exactly_the_candidates(self):
        '''
        C is exactly the candidates mask. A code outside it carries no probability and takes no
        gradient, however near it is, and every code inside it takes gradient.
        '''

        step = _task_step()
        outside = ~step['candidates']
        distances = step['distances'].clone().requires_grad_(True)
        loss = terms.task_loss(**{**step, 'distances': distances})
        loss.backward()

        for value in (0.0, 1e3):
            moved = step['distances'].clone()
            moved[outside] = value
            assert torch.equal(terms.task_loss(**{**step, 'distances': moved}), loss.detach())
        assert torch.count_nonzero(distances.grad[outside]) == 0
        assert torch.count_nonzero(distances.grad[step['candidates']]) == step['candidates'].sum()

    def test_the_task_loss_passes_gradient_to_the_distances_and_the_scale(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        step = _task_step()
        distances = step['distances'].float().requires_grad_(True)

        loss = terms.task_loss(distances, scale(), step['candidates'], step['targets'])
        loss.backward()

        assert torch.isfinite(distances.grad).all()
        assert torch.count_nonzero(distances.grad) > 0
        assert torch.isfinite(scale.log_scale.grad)
        assert scale.log_scale.grad.item() != 0.0

    def test_a_target_outside_the_candidates_is_refused(self):
        '''T lies in C: a target outside it would sum probability that no softmax gave it.'''

        step = _task_step()
        # Code 3 is not one of the first query's candidates
        step['targets'][0, 3] = True

        with pytest.raises(ValueError, match='query 0 has a target outside its candidates'):
            terms.task_loss(**step)

    def test_a_query_without_a_target_is_refused(self):
        step = _task_step()
        step['targets'][1] = False

        with pytest.raises(ValueError, match='query 1 has no target'):
            terms.task_loss(**step)

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            pytest.param(
                {
                    'distances': torch.zeros(0, 6, dtype=torch.float64),
                    'candidates': torch.zeros(0, 6, dtype=torch.bool),
                    'targets': torch.zeros(0, 6, dtype=torch.bool),
                },
                'task_loss needs a non-empty',
                id='an-empty-step',
            ),
            pytest.param(
                {'scale': torch.ones(1, dtype=torch.float64)},
                'task_loss needs a 0-d logit scale',
                id='a-scale-that-is-not-0-d',
            ),
            pytest.param(
                {'candidates': torch.ones(3, 5, dtype=torch.bool)},
                'candidates must be a bool mask shaped like the distances',
                id='misaligned-candidates',
            ),
            pytest.param(
                {'targets': torch.ones(3, 6)},
                'targets must be a bool mask shaped like the distances',
                id='targets-that-are-not-bool',
            ),
        ],
    )
    def test_a_malformed_step_is_refused(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            terms.task_loss(**{**_task_step(), **overrides})

# -------------------------------------------------------------------------------------------------
# Req 11's code–code term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCodeCodeLoss:
    '''
    Spec 4.1(ii): the mean over anchors of CE(p_a, softmax over J_a of −s_c · d(a, j)), with the
    target p_a = softmax over J_a of −D*_{aj} / τ_t.
    '''

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
    def test_the_loss_is_the_cross_entropy_to_softmax_of_minus_d_star_over_tau(self, temperature):
        step = {**_code_code_step(), 'target_temperature': temperature}

        loss = terms.code_code_loss(**step)

        torch.testing.assert_close(loss, _reference_code_code_loss(**step), rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
    def test_the_loss_is_stationary_where_the_model_reproduces_the_target(self, temperature):
        '''
        With d = D* / (τ_t · s) on J_a, the model's softmax is the target, so the gradient vanishes
        and the loss is the target's entropy. Any other target would leave a gradient here.
        '''

        step = _code_code_step()
        scale = 1.5
        distances = (step['structural'] / (temperature * scale)).requires_grad_(True)

        loss = terms.code_code_loss(
            distances, _scale(scale), step['structural'], step['keep'], temperature
        )
        loss.backward()

        entropy = _mean_target_entropy(step['structural'], step['keep'], temperature)
        assert distances.grad.abs().max().item() < 1e-12
        assert loss.item() == pytest.approx(entropy, rel=1e-12)

    def test_the_anchor_and_its_unary_partner_carry_no_probability_on_either_side(self):
        '''
        J_a drops the anchor and its unary partner (Req 9). Their D* never reaches the target and
        their d never reaches the model's softmax, though unmasked they would be the nearest codes
        on both sides. Every other code takes gradient.
        '''

        step = _code_code_step()
        masked, kept = ~step['keep'], step['keep']
        distances = step['distances'].clone().requires_grad_(True)
        loss = terms.code_code_loss(**{**step, 'distances': distances})
        loss.backward()

        for value in (0.0, 0.25, 50.0):
            target_side = {**step, 'structural': step['structural'].clone()}
            target_side['structural'][masked] = value
            model_side = {**step, 'distances': step['distances'].clone()}
            model_side['distances'][masked] = value
            assert torch.equal(terms.code_code_loss(**target_side), loss.detach())
            assert torch.equal(terms.code_code_loss(**model_side), loss.detach())
        assert torch.count_nonzero(distances.grad[masked]) == 0
        assert torch.count_nonzero(distances.grad[kept]) == kept.sum()

    def test_the_code_code_loss_passes_gradient_to_the_distances_and_the_scale(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        step = _code_code_step()
        distances = step['distances'].float().requires_grad_(True)

        loss = terms.code_code_loss(
            distances, scale(), step['structural'].float(), step['keep'], 1.0
        )
        loss.backward()

        assert torch.isfinite(distances.grad).all()
        assert torch.count_nonzero(distances.grad) > 0
        assert torch.isfinite(scale.log_scale.grad)
        assert scale.log_scale.grad.item() != 0.0

    def test_an_anchor_that_keeps_no_code_is_refused(self):
        step = _code_code_step()
        step['keep'][2] = False

        with pytest.raises(ValueError, match='anchor 2 keeps no code'):
            terms.code_code_loss(**step)

    @pytest.mark.parametrize('temperature', [0.0, -1.0, math.inf, math.nan])
    def test_a_target_temperature_that_is_not_positive_and_finite_is_refused(self, temperature):
        step = {**_code_code_step(), 'target_temperature': temperature}

        with pytest.raises(ValueError, match='target_temperature must be a positive finite number'):
            terms.code_code_loss(**step)

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            pytest.param(
                {
                    'distances': torch.zeros(0, 6, dtype=torch.float64),
                    'structural': torch.zeros(0, 6, dtype=torch.float64),
                    'keep': torch.zeros(0, 6, dtype=torch.bool),
                },
                'code_code_loss needs a non-empty',
                id='an-empty-step',
            ),
            pytest.param(
                {'scale': torch.ones(1, dtype=torch.float64)},
                'code_code_loss needs a 0-d logit scale',
                id='a-scale-that-is-not-0-d',
            ),
            pytest.param(
                {'structural': torch.zeros(3, 5, dtype=torch.float64)},
                'structural must be shaped like the distances',
                id='misaligned-structural-distances',
            ),
            pytest.param(
                {'keep': torch.ones(3, 6)},
                'keep must be a bool mask shaped like the distances',
                id='a-keep-mask-that-is-not-bool',
            ),
        ],
    )
    def test_a_malformed_step_is_refused(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            terms.code_code_loss(**{**_code_code_step(), **overrides})

# -------------------------------------------------------------------------------------------------
# Req 11's radial term
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRadialLoss:
    '''
    Spec 4.1(iii): the mean over anchors of (r_a − ρ · (λ(a) − 1))², so the virtual root sits at o,
    the sectors at ρ and the six-digit codes at 5ρ.
    '''

    @pytest.mark.parametrize('rho', [0.5, 1.0, 2.0])
    def test_the_target_radius_is_rho_times_level_minus_one(self, rho):
        levels = torch.tensor([2, 3, 4, 5, 6])
        # The sectors at ρ, and so on out to the six-digit codes at 5ρ
        steps_out = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        radius = (rho * steps_out).requires_grad_()

        loss = terms.radial_loss(radius, levels, rho)
        loss.backward()

        assert loss.item() == 0.0
        assert torch.count_nonzero(radius.grad) == 0
        # At r = ρ · λ every code sits one step out, so the loss is ρ²
        one_out = terms.radial_loss(rho * levels.double(), levels, rho)
        assert one_out.item() == pytest.approx(rho**2, rel=1e-12)

    def test_the_loss_is_the_mean_squared_gap_to_the_target(self):
        generator = torch.Generator().manual_seed(23)
        levels = torch.randint(2, 7, (9, ), generator=generator)
        radius = 6 * torch.rand(9, generator=generator, dtype=torch.float64)

        loss = terms.radial_loss(radius, levels, 0.75)

        gaps = [r - 0.75 * (level - 1) for r, level in zip(radius.tolist(), levels.tolist())]
        assert loss.item() == pytest.approx(sum(gap**2 for gap in gaps) / 9, rel=1e-12)

    def test_the_radial_loss_passes_gradient_to_the_radii(self):
        radius = torch.tensor([0.5, 2.0, 7.5], requires_grad=True)

        terms.radial_loss(radius, torch.tensor([2, 4, 6]), 1.0).backward()

        # The gradient of the mean of (r − (λ − 1))² is 2 (r − (λ − 1)) / A
        torch.testing.assert_close(radius.grad, torch.tensor([-1 / 3, -2 / 3, 5 / 3]))

    @pytest.mark.parametrize('rho', [0.0, -1.0, math.inf, math.nan])
    def test_a_radial_step_that_is_not_positive_and_finite_is_refused(self, rho):
        with pytest.raises(ValueError, match='radial_step must be a positive finite number'):
            terms.radial_loss(torch.ones(3), torch.tensor([2, 4, 6]), rho)

    @pytest.mark.parametrize(
        ('radius', 'levels'),
        [
            pytest.param(torch.zeros(0), torch.zeros(0, dtype=torch.long), id='an-empty-step'),
            pytest.param(torch.ones(3, 1), torch.tensor([2, 4, 6]), id='radii-that-are-not-1-d'),
            pytest.param(torch.ones(3), torch.tensor([2, 4]), id='misaligned-levels'),
        ],
    )
    def test_a_malformed_step_is_refused(self, radius, levels):
        with pytest.raises(ValueError, match='radial_loss needs non-empty radii'):
            terms.radial_loss(radius, levels, 1.0)

# -------------------------------------------------------------------------------------------------
# Req 11's learned logit scales
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLogitScale:
    '''Spec 4.1: s = exp(θ), from its init and clamped to its range; R7 states 1 in [0.01, 100].'''

    def test_the_stated_defaults_start_at_1_with_one_parameter(self):
        scale = terms.LogitScale(1.0, 0.01, 100.0)

        assert [name for name, _ in scale.named_parameters()] == ['log_scale']
        assert list(scale.buffers()) == []
        assert scale.log_scale.item() == 0.0
        assert scale().ndim == 0
        assert scale().item() == 1.0

    @pytest.mark.parametrize('init', [0.5, 2.0, 8.0])
    def test_the_scale_starts_at_init_anywhere_in_its_range(self, init):
        scale = terms.LogitScale(init, 0.5, 8.0)

        assert scale.log_scale.item() == pytest.approx(math.log(init), rel=1e-6)
        assert scale().item() == pytest.approx(init, rel=1e-6)

    @pytest.mark.parametrize(
        ('theta', 'clamped'),
        [
            pytest.param(math.log(1e4), 100.0, id='above-the-range'),
            pytest.param(math.log(1e-4), 0.01, id='below-the-range'),
        ],
    )
    def test_the_scale_clamps_to_its_range(self, theta, clamped):
        scale = terms.LogitScale(1.0, 0.01, 100.0)
        with torch.no_grad():
            scale.log_scale.fill_(theta)

        assert scale().item() == pytest.approx(clamped, rel=1e-6)

    @pytest.mark.parametrize(
        ('init', 'low', 'high', 'message'),
        [
            pytest.param(1.0, 2.0, 1.0, 'must satisfy 0 < low < high', id='an-empty-range'),
            pytest.param(1.0, 1.0, 1.0, 'must satisfy 0 < low < high', id='a-one-point-range'),
            pytest.param(1.0, 0.0, 10.0, 'must satisfy 0 < low < high', id='a-zero-low'),
            pytest.param(1.0, -1.0, 10.0, 'must satisfy 0 < low < high', id='a-negative-low'),
            pytest.param(0.5, 1.0, 10.0, 'must start inside its range', id='an-init-below'),
            pytest.param(20.0, 1.0, 10.0, 'must start inside its range', id='an-init-above'),
            pytest.param(math.nan, 0.01, 100.0, 'takes finite numbers', id='a-nan-init'),
            pytest.param(1.0, 0.01, math.inf, 'takes finite numbers', id='an-infinite-high'),
        ],
    )
    def test_a_range_that_is_empty_or_not_positive_or_misses_its_init_is_refused(
        self, init, low, high, message
    ):
        with pytest.raises(ValueError, match=message):
            terms.LogitScale(init, low, high)

# -------------------------------------------------------------------------------------------------
# Req 11's terms under autocast
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_terms_and_the_scale_stay_float32_under_cpu_bf16_autocast():
    '''Spec §6 "Precision": float32 inputs give float32 losses under CPU bf16 autocast.'''

    scale = terms.LogitScale(1.0, 0.01, 100.0)
    task, code_code = _task_step(), _code_code_step()

    with torch.autocast('cpu', dtype=torch.bfloat16):
        values = {
            'scale': scale(),
            'task': terms.task_loss(
                task['distances'].float(), scale(), task['candidates'], task['targets']
            ),
            'code_code': terms.code_code_loss(
                code_code['distances'].float(),
                scale(),
                code_code['structural'].float(),
                code_code['keep'],
                1.0,
            ),
            'radial': terms.radial_loss(torch.rand(4), torch.tensor([2, 3, 4, 6]), 1.0),
        }

    dtypes = {name: value.dtype for name, value in values.items()}
    assert dtypes == dict.fromkeys(values, torch.float32)
