# NAICS Embedder Test Suite

## Overview

This directory contains the test suite for the NAICS Hyperbolic Embedding System, covering unit
tests for individual components and integration tests for end-to-end workflows. The test
infrastructure uses pytest with coverage reporting, property-based testing (Hypothesis), and
performance benchmarking.

### Current Status

**Test files:** 53 unit, 2 integration

- ✅ **Well Tested**: Text model pipeline (encoding, MoE, loss, hyperbolic ops, evaluation)
- ✅ **Tested since this file was written**: Data processing, graph model (HGCN), clustering,
  training and validation utilities, CLI commands

**Recent additions:**

- ✅ **test_evaluation.py** - Comprehensive evaluation metrics testing (Issue #49)
- ✅ **test_tokenization_cache.py** - Data loading and caching modules testing (Issue #50)

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Coverage Analysis](#coverage-analysis)
  - [Well-Tested Modules](#well-tested-modules)
  - [Closed Gaps](#closed-gaps)
  - [Open Gaps](#open-gaps)
- [Coverage Goals](#coverage-goals)
- [Test Markers](#test-markers)
- [Adding New Tests](#adding-new-tests)
- [Best Practices](#best-practices)
- [Debugging Failed Tests](#debugging-failed-tests)
- [Continuous Integration](#continuous-integration)
- [Known Issues](#known-issues)

## Test Structure

```text
tests/
├── unit/                      # Unit tests for individual components (53 files)
│   ├── test_hyperbolic.py    # Hyperbolic geometry operations (CRITICAL) ✅
│   ├── test_loss.py          # Loss functions ✅
│   ├── test_moe.py           # Mixture of Experts ✅
│   ├── test_encoder.py       # Multi-channel encoder ✅
│   ├── test_naics_model.py   # PyTorch Lightning module ✅
│   ├── test_evaluation.py    # Text model evaluation metrics ✅
│   ├── test_curriculum.py    # Curriculum scheduling ✅
│   ├── test_hard_negative_mining.py  # Hard negative mining ✅
│   ├── test_false_negative_strategy.py  # False negative mitigation ✅
│   ├── test_tokenization_cache.py  # Tokenization caching ✅
│   ├── test_datamodule.py    # Data module and collation ✅
│   ├── test_streaming_dataset.py  # Streaming dataset utilities ✅
│   ├── test_streaming_sampling.py  # Sampling strategies ✅
│   ├── test_data_distances.py  # Distance computation ✅
│   ├── test_config.py        # Configuration management ✅
│   └── ...
├── integration/              # Integration tests (2 files)
├── fixtures/                 # Test data and fixtures
└── conftest.py              # Shared pytest fixtures

```

## Running Tests

### Run all tests

```bash
uv run pytest tests/
```

### Run specific test file

```bash
uv run pytest tests/unit/test_hyperbolic.py
```

### Run with coverage

```bash
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html
open htmlcov/index.html
```

### Run with verbose output

```bash
uv run pytest tests/ -v
```

### Run only unit tests

```bash
uv run pytest tests/ -m unit
```

### Run in parallel

```bash
uv run pytest tests/ -n auto
```

## Coverage Analysis

### Well-Tested Modules

The following modules have comprehensive test coverage:

#### Text Model Pipeline (Stages 1-3)

1. **text_model/hyperbolic.py** ✅ - `test_hyperbolic.py`
   - LorentzOps (exp/log maps, distances, inner products)
   - HyperbolicProjection (Euclidean → Lorentz)
   - LorentzDistance computation
   - Manifold validity checks
   - Numerical stability tests
   - Property-based tests with Hypothesis

2. **text_model/loss.py** ✅ - `test_loss.py`
   - HyperbolicInfoNCELoss (DCL-based contrastive learning)
   - HierarchyPreservationLoss (distance correlation)
   - StructuralPreferenceLoss (pairwise structural ordering; gradient-direction contracts)

3. **text_model/encoder.py** ✅ - `test_encoder.py`
   - Multi-channel transformer encoders (title, description, examples, exclusions)
   - LoRA adaptation layers
   - Channel-specific encoding

4. **text_model/moe.py** ✅ - `test_moe.py`
   - Top-k gating mechanism
   - Expert routing logic
   - Load balancing loss
   - Batched expert processing

5. **text_model/naics_model.py** ✅ - `test_naics_model.py`
   - NAICSContrastiveModel PyTorch Lightning module
   - Training step (forward + loss)
   - Validation step (metrics)
   - Optimizer configuration

6. **metrics/core.py** ✅ - `test_evaluation.py`
   - EmbeddingEvaluator (distance/similarity computation)
   - RetrievalMetrics (precision@k, recall@k, MAP, NDCG)
   - HierarchyMetrics (cophenetic/Spearman correlation, distortion)
   - EmbeddingStatistics (norm, radius, diversity, collapse detection)
   - NAICSEvaluationRunner (full evaluation pipeline, in `metrics/runner.py`)

7. **text_model/curriculum.py** ✅ - `test_curriculum.py`
   - Dynamic structure-aware curriculum scheduling
   - Difficulty progression

8. **text_model/hard_negative_mining.py** ✅ - `test_hard_negative_mining.py`
   - Hard negative sampling strategies
   - Distance-based selection

9. **text_model/false_negative_strategies.py** ✅ - `test_false_negative_strategy.py`
   - False negative detection strategies
   - Masking logic

#### Data Loading Pipeline

1. **text_model/dataloader/tokenization_cache.py** ✅ - `test_tokenization_cache.py`
    - Cache building from descriptions
    - Cache save/load operations
    - File locking for multi-worker safety
    - Atomic cache operations
    - get_tokens utility function

2. **text_model/dataloader/datamodule.py** ✅ - `test_datamodule.py`
    - collate_fn batching logic
    - Positive level extraction
    - NAICSMapDataset indexing and __getitem__
    - DataLoader shuffle configuration

3. **text_model/dataloader/streaming_dataset.py** ✅ - `test_streaming_dataset.py`
    - Taxonomy utilities
    - Ancestor/descendant generation
    - Matrix loading
    - Sampling weight computation
    - Multi-epoch cache path generation

#### Configuration & Utilities

1. **utils/config.py** ✅ - `test_config.py`
    - Pydantic configuration models
    - YAML loading and parsing
    - Configuration validation

2. **data/compute_distances.py** ✅ - `test_data_distances.py`
    - NAICS tree construction
    - Pairwise distance calculations
    - Distance matrix generation

### Closed Gaps

The gap analysis added in `b1ee4df` (November 2025) flagged nine areas as untested. Each now
has tests:

| Area | Modules | Tests |
|------|---------|-------|
| Graph model (HGCN) | `graph_model/hgcn.py`, `graph_model/dataloader/hgcn_datamodule.py`, `graph_model/dataloader/hgcn_streaming_dataset.py`, `metrics/graph.py` (was `graph_model/evaluation.py`) | `test_hgcn.py`, `test_hgcn_metrics.py`, `test_hgcn_datamodule.py`, `test_hgcn_streaming_dataset.py`, `test_graph_downstream_evaluation.py` |
| Hyperbolic clustering | `text_model/hyperbolic_clustering.py` | `test_hyperbolic_clustering.py` |
| Training utilities | `utils/training.py` | `test_utils_training.py` |
| Validation utilities | `utils/validation.py` | `test_utils_validation.py` |
| Data generation | `data/compute_relations.py`, `data/create_triplets.py`, `data/download_data.py` | `test_data_relations.py`, `test_data_triplets.py`, `test_data_download.py` |
| CLI commands | `cli/commands/data.py`, `cli/commands/tools.py`, `cli/commands/training.py` | `test_cli_commands.py` (data, tools), `test_cli_training.py` (training) |
| Backend and console utilities | `utils/backend.py`, `utils/utilities.py`, `utils/warnings.py`, `utils/console.py` | `test_utils_backend.py`, `test_utils_utilities.py`, `test_warnings.py`, `test_utils_console.py` |
| Tools and visualization | `tools/config_tools.py`, `tools/metrics_tools.py`, `tools/_visualize_metrics.py`, `tools/_investigate_hierarchy.py` | `test_config_tools.py`, `test_metrics_tools_api.py`, `test_visualize_metrics.py`, `test_investigate_hierarchy_tool.py` |
| Integration | Cross-component | `integration/test_stage3_training_step.py` (one training step), `integration/test_distributed_supervision.py` (distributed selection) |

Module paths are relative to `src/naics_embedder/`. The per-module test skeletons that used to
follow here were removed, since the test files are the reference now. The original analysis and
skeletons are in `b1ee4df`.

### Open Gaps

- **Training loops.** No test runs an optimization loop on the text model or HGCN. Tests call
  `training_step` or `backward()` directly, and the `Trainer.fit` runs in `test_datamodule.py`
  drive a stub module to test epoch handling.
- **Whole-system run.** No test runs data generation → text training → HGCN → evaluation end to
  end. Tests described as full-pipeline or end-to-end cover sub-pipelines: graph preprocessing,
  positive sampling, and distributed selection.
- **Coverage numbers.** This README no longer records current coverage. Measure it against
  the [targets](#target-coverage-metrics) with the commands in
  [Measuring Coverage](#measuring-coverage).

## Known Issues

- **Python 3.14:** some tests fail there with the locked torch 2.9.1. Importing `torch._inductor`
  raises `AttributeError: 'typing.Union' object has no attribute '__module__'` in
  `torch/ao/quantization/quantizer/quantizer.py`. The repo pins Python 3.12 in `.python-version`,
  which `uv` uses by default; see [CLAUDE.md](../CLAUDE.md#initial-setup).

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests requiring significant compute
- `@pytest.mark.gpu` - Tests requiring GPU

## Continuous Integration

GitHub Actions automatically runs the full test suite on:

- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches

See `.github/workflows/tests.yml` for CI configuration.

## Coverage Goals

### Target Coverage Metrics

| Module Category | Target |
|-----------------|--------|
| **Critical Math** (hyperbolic.py, loss.py) | >80% |
| **Text Model Pipeline** | >70% |
| **Graph Model (HGCN)** | >70% |
| **Data Generation** | >60% |
| **Training Utilities** | >65% |
| **CLI & Tools** | >50% |
| **Overall Project** | >70% |

### Measuring Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Generate coverage badge (requires coverage-badge)
coverage-badge -o coverage.svg -f

# Check specific module coverage
uv run pytest tests/ --cov=src/naics_embedder.graph_model.hgcn --cov-report=term
```

### Coverage Improvement Roadmap

#### Phase 1: Critical Gaps (Weeks 1-2)

- [x] Add `test_hgcn.py` - Graph model testing
- [x] Add `test_hyperbolic_clustering.py` - Clustering validation
- [ ] Target: Bring overall coverage to >50%

#### Phase 2: Important Gaps (Weeks 3-4)

- [x] Add `test_utils_training.py` - Training utilities
- [x] Add `test_utils_validation.py` - Validation utilities
- [x] Add `test_data_relations.py` - Data relationships
- [ ] Target: Bring overall coverage to >60%

#### Phase 3: Integration & Completeness (Weeks 5-6)

- [ ] Add integration tests for training loops
- [ ] Add end-to-end pipeline tests
- [x] Add CLI command tests
- [ ] Target: Achieve >70% overall coverage

## Best Practices

### General Testing Principles

1. **Test Behavior, Not Implementation**
   - Focus on what the code does, not how it does it
   - Tests should remain valid even if internal implementation changes
   - Example: Test that embeddings are on manifold, not the specific exp_map formula

2. **Follow AAA Pattern** (Arrange, Act, Assert)

   ```python
   def test_hyperbolic_distance_is_positive(self):
       # Arrange: Set up test data
       x = torch.randn(8, 385)
       y = torch.randn(8, 385)

       # Act: Perform the operation
       distance = lorentz_distance(x, y)

       # Assert: Verify the result
       assert torch.all(distance >= 0)
   ```

3. **Use Descriptive Test Names**
   - Good: `test_exponential_map_produces_manifold_points()`
   - Bad: `test_exp_map()`, `test_function1()`
   - Pattern: `test_<what>_<expected_behavior>`

4. **One Assertion Per Test (Generally)**
   - Each test should verify one logical concept
   - Multiple assertions are OK if testing related properties
   - Use `pytest.mark.parametrize` for testing multiple inputs

5. **Test Edge Cases and Boundaries**
   - Empty inputs, single elements, very large inputs
   - Extreme parameter values (curvature = 0.001, curvature = 100)
   - Invalid inputs (should raise appropriate errors)

6. **Use Fixtures for Shared Setup**

   ```python
   @pytest.fixture
   def sample_embeddings():
       '''Generate sample Lorentz embeddings for testing.'''
       return generate_valid_lorentz_embeddings(n=16, dim=384)
   ```

### Testing Hyperbolic Operations

When testing hyperbolic geometry code:

1. **Always Validate Manifold Constraints**

   ```python
   from naics_embedder.text_model.hyperbolic import check_lorentz_manifold_validity

   is_valid, norms, violations = check_lorentz_manifold_validity(embeddings, c=1.0)
   assert is_valid, f'Manifold constraint violated: max={violations.max()}'
   ```

2. **Test Numerical Stability**
   - Test with small curvatures (0.1) and large curvatures (10.0)
   - Test with extreme distance values
   - Check for NaN, Inf in outputs

3. **Verify Geometric Properties**
   - Distance symmetry: d(x, y) = d(y, x)
   - Identity: d(x, x) = 0
   - Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
   - Roundtrip consistency: exp(log(x)) ≈ x

### Testing PyTorch Modules

1. **Test Gradient Flow**

   ```python
   def test_gradients_flow_through_module(self):
       model = MyModule()
       x = torch.randn(8, 10, requires_grad=True)
       loss = model(x).sum()
       loss.backward()
       assert x.grad is not None
       assert not torch.all(x.grad == 0)
   ```

2. **Test with Deterministic Seeds**

   ```python
   import pytorch_lightning as pl

   def test_reproducible_results(self):
       pl.seed_everything(42)
       result1 = model(x)
       pl.seed_everything(42)
       result2 = model(x)
       assert torch.allclose(result1, result2)
   ```

3. **Test Forward and Backward**
   - Forward pass produces correct shapes
   - Backward pass computes gradients
   - Loss decreases over multiple iterations

### Testing Data Processing

1. **Use Synthetic Data**

   ```python
   def test_data_loader_with_synthetic_data(self):
       # Create minimal synthetic dataset
       df = pl.DataFrame({
           'naics_code': ['111110', '111120'],
           'title': ['Industry 1', 'Industry 2']
       })
       dataset = NAICSDataset(df)
       assert len(dataset) == 2
   ```

2. **Test with tmp_path Fixture**

   ```python
   def test_cache_saving(self, tmp_path):
       cache_path = tmp_path / 'test.cache'
       save_cache(cache_path, data)
       assert cache_path.exists()
       loaded = load_cache(cache_path)
       assert loaded == data
   ```

3. **Validate Data Schemas**
   - Check column presence and types
   - Verify data ranges (NAICS codes are strings, distances are floats)

### Using Property-Based Testing

For mathematical functions, use Hypothesis:

```python
from hypothesis import given, settings, strategies as st

@given(
    tangent_vectors=st.lists(
        st.floats(min_value=-10, max_value=10),
        min_size=384,
        max_size=384
    ).map(lambda x: torch.tensor([x]))
)
@settings(max_examples=100)
def test_exp_map_always_produces_manifold_points(self, tangent_vectors):
    '''Property: exp_map always produces valid Lorentz manifold points.'''
    hyp = LorentzOps.exp_map_zero(tangent_vectors, c=1.0)
    is_valid, _, _ = check_lorentz_manifold_validity(hyp, c=1.0)
    assert is_valid
```

### Mocking External Dependencies

Use `unittest.mock` for external resources:

```python
from unittest.mock import Mock, patch

@patch('torch.cuda.is_available')
def test_cuda_detection(self, mock_cuda):
    mock_cuda.return_value = True
    device = get_device()
    assert device == 'cuda'
```

### Performance and Benchmarking

Use `pytest-benchmark` for performance-critical code:

```python
def test_lorentz_distance_performance(benchmark):
    x = torch.randn(1000, 385)
    y = torch.randn(1000, 385)
    result = benchmark(lorentz_distance, x, y, c=1.0)
    assert result.shape == (1000,)
```

## Adding New Tests

### Quick Start Guide

When adding new features, follow this pattern:

1. **Create test file**: `tests/unit/test_<module>.py`
2. **Import relevant code and fixtures**
3. **Create test classes grouped by functionality**
4. **Use descriptive test names**: `test_<what>_<expected_behavior>`
5. **Add docstrings explaining what each test validates**
6. **Use appropriate markers** (`@pytest.mark.unit`, etc.)

### Test File Template

```python
'''
Unit tests for <module_name>.

Tests the <functionality> including <key features>.
'''

import pytest
import torch
from naics_embedder.<module_path> import <classes_to_test>

# -------------------------------------------------------------------------------------------------
# Test Class
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class Test<ClassName>:
    '''Test suite for <ClassName>.'''

    def test_basic_functionality(self):
        '''Test that <feature> works in basic case.'''
        # Arrange
        input_data = create_test_input()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_output

    def test_edge_case_empty_input(self):
        '''Test behavior with empty input.'''
        result = function_under_test([])
        assert result is not None

    @pytest.mark.parametrize('param', [0.1, 1.0, 5.0])
    def test_different_parameters(self, param):
        '''Test with various parameter values.'''
        result = function_under_test(param=param)
        assert result > 0

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class Test<ModuleName>Integration:
    '''Integration tests for <module>.'''

    def test_integration_with_other_module(self):
        '''Test integration between <module1> and <module2>.'''
        # Test end-to-end workflow
```

### Fixtures Guide

Add shared fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def sample_naics_data():
    '''Generate sample NAICS data for testing.'''
    return pl.DataFrame({
        'naics_code': ['111110', '111120', '111130'],
        'title': ['Soybean Farming', 'Oilseed Farming', 'Dry Pea Farming'],
        'description': ['Description 1', 'Description 2', 'Description 3']
    })

@pytest.fixture
def sample_lorentz_embeddings():
    '''Generate valid Lorentz manifold embeddings.'''
    # Implementation that creates valid embeddings
    pass
```

## Debugging Failed Tests

### View detailed error messages

```bash
uv run pytest tests/ -vv
```

### Drop into debugger on failure

```bash
uv run pytest tests/ --pdb
```

### Show print statements

```bash
uv run pytest tests/ -s
```

### Run only failed tests from last run

```bash
uv run pytest tests/ --lf
```

## Performance Benchmarking

Some tests include performance benchmarks using `pytest-benchmark`:

```bash
uv run pytest tests/ --benchmark-only
```

## Summary: Priority Action Items

### For New Contributors

**Start here:**

1. Read this README thoroughly
2. Run existing tests: `uv run pytest tests/ -v`
3. Check coverage: `uv run pytest tests/ --cov=src/naics_embedder --cov-report=html`
4. Pick an [open gap](#open-gaps) and implement tests

### For Maintainers

The areas this section used to prioritize all have tests now (see [Closed Gaps](#closed-gaps)).
What remains is in [Open Gaps](#open-gaps).

### Quick Reference Commands

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/ -m unit

# Run only integration tests
uv run pytest tests/ -m integration

# Run with coverage report
uv run pytest tests/ --cov=src/naics_embedder --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_hyperbolic.py

# Run specific test class
uv run pytest tests/unit/test_hyperbolic.py::TestLorentzOps

# Run specific test method
uv run pytest tests/unit/test_hyperbolic.py::TestLorentzOps::test_exp_log_roundtrip

# Run in parallel (faster)
uv run pytest tests/ -n auto

# Run with verbose output
uv run pytest tests/ -v

# Run only failed tests from last run
uv run pytest tests/ --lf

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Show print statements
uv run pytest tests/ -s

# Generate benchmark report
uv run pytest tests/ --benchmark-only
```

### Test Coverage Quick Stats

| Category | Status |
|----------|--------|
| **Total Source Modules** | 67 (excluding `__init__.py`) |
| **Total Test Files** | 53 unit, 2 integration |
| **Target Coverage** | >70% |

### Critical Missing Tests

All six have since been added:

1. ✅ **`tests/unit/test_hgcn.py`** - Graph model
2. ✅ **`tests/unit/test_hyperbolic_clustering.py`** - Clustering
3. ✅ **`tests/unit/test_utils_training.py`** - Training utilities
4. ✅ **`tests/unit/test_utils_validation.py`** - Validation
5. ✅ **`tests/unit/test_data_relations.py`** - Data generation
6. ✅ **`tests/integration/test_stage3_training_step.py`** - Integration tests

## Contact and Resources

### Getting Help

For questions about the test suite:

- **Documentation**: `/README.md`, `/CLAUDE.md`
- **Issues**: <https://github.com/lowmason/naics-embedder/issues>
- **Discussions**: Use GitHub Discussions for testing strategy questions

### Related Documentation

- **Main README**: Project overview and architecture
- **CLAUDE.md**: AI assistant guide with detailed codebase documentation
- **docs/**: Full MkDocs documentation site
- **pyproject.toml**: Pytest configuration and dependencies

### CI/CD

- **GitHub Actions**: `.github/workflows/tests.yml`
- **Runs on**: Push to `main`, pull requests
- **Includes**: Test suite, coverage reporting, linting

---

**Last Updated**: 2025-11-27

**Document Version**: 2.0 (Comprehensive Coverage Analysis)
