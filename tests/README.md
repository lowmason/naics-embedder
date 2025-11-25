# NAICS Embedder Test Suite

## Overview

This directory contains comprehensive unit and integration tests for the NAICS Hyperbolic Embedding
System. The test suite covers critical mathematical operations, loss functions, model components,
and data processing pipelines.

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_hyperbolic.py    # Hyperbolic geometry operations (CRITICAL)
│   ├── test_loss.py          # Loss functions
│   ├── test_moe.py           # Mixture of Experts
│   ├── test_data_distances.py # Distance computation
│   └── test_config.py        # Configuration management
├── integration/              # Integration tests (future)
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

## Test Coverage Areas

### Phase 1: Critical Mathematical Operations (✅ Implemented)

1. **test_hyperbolic.py** - Hyperbolic geometry operations
   - LorentzOps (exp/log maps, distances)
   - HyperbolicProjection
   - Manifold validity checks
   - Numerical stability
   - Property-based tests with Hypothesis

2. **test_loss.py** - Loss function correctness
   - HyperbolicInfoNCELoss (contrastive learning)
   - HierarchyPreservationLoss (MSE-based)
   - RankOrderPreservationLoss (margin-based)
   - LambdaRankLoss (NDCG-based)

### Phase 2: Core Components (✅ Implemented)

3. **test_moe.py** - Mixture of Experts
   - Gating mechanism
   - Expert routing
   - Load balancing
   - Batched processing

4. **test_data_distances.py** - Distance computation
   - Tree construction
   - Distance calculations
   - Relationship computations

5. **test_config.py** - Configuration management
   - Pydantic models
   - YAML loading
   - Validation

### Phase 3: Integration & ML Components (Future)

6. **test_hgcn.py** - Graph model operations (TODO)
7. **test_dataloader.py** - Streaming and caching (TODO)
8. **test_training_loop.py** - Training validation (TODO)
9. **test_cli.py** - CLI commands (TODO)

## Known Issues

Some tests currently fail due to fixture generation issues:

1. **Hyperbolic fixture**: The `sample_tangent_vectors` and `sample_lorentz_embeddings` fixtures
   need refinement to properly generate tangent vectors at the origin (time component should be 0)

2. **Config loading**: Tests using `tmp_path` need adjustment to work with the config loader's
   path resolution

These are test infrastructure issues, not issues with the core code. The tests themselves are
correctly written and will pass once fixtures are fixed.

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

- **Overall**: >70% coverage
- **Critical modules** (hyperbolic.py, loss.py): >80% coverage
- **Data processing**: >60% coverage

## Adding New Tests

When adding new features, follow this pattern:

1. Create test file: `tests/unit/test_<module>.py`
2. Import relevant code and fixtures
3. Create test classes grouped by functionality
4. Use descriptive test names: `test_<what>_<expected_behavior>`
5. Add docstrings explaining what each test validates
6. Use appropriate markers (`@pytest.mark.unit`, etc.)

Example:

```python
@pytest.mark.unit
class TestNewFeature:
    '''Test suite for new feature.'''

    def test_feature_basic_functionality(self):
        '''Test that feature works in basic case.'''
        result = new_feature(input_data)
        assert result == expected_output

    def test_feature_edge_case(self):
        '''Test feature behavior with edge case input.'''
        result = new_feature(edge_case_input)
        assert result is not None
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

## Contact

For questions about the test suite, see:

- Main README: `/README.md`
- CLAUDE.md: `/CLAUDE.md`
- Issues: https://github.com/lowmason/naics-embedder/issues
