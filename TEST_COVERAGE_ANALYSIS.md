# Test Coverage Analysis and Improvement Recommendations

**Generated:** 2025-12-01

## Executive Summary

The NAICS Embedder project has **strong test coverage** with 81.2% of source files covered by unit
tests. The test suite contains:

- **530+ test functions** across 36 test files
- **~11,000 lines** of test code
- Comprehensive coverage of core ML components (text model, graph model, losses, evaluation)
- Well-structured unit tests with fixtures and proper isolation

However, there are **critical gaps** in testing for data utilities, tools, and integration tests.

---

## Coverage Summary

| Category | Files | Tested | Coverage |
|----------|-------|--------|----------|
| **Total** | 48 | 39 | **81.2%** |
| Text Model | 11 | 11 | **100%** ✓ |
| Graph Model | 8 | 8 | **100%** ✓ |
| Data Processing | 5 | 4 | 80% |
| Utils | 10 | 6 | 60% |
| Tools | 6 | 1 | **16.7%** ⚠️ |
| CLI | 4 | 3 | 75% |
| Metrics | 1 | 1 | 100% ✓ |

---

## Tested Components (39 files)

### ✓ Excellent Coverage

**Text Model (11/11 files)**
- ✓ encoder.py → test_encoder.py (508 lines)
- ✓ moe.py → test_moe.py (389 lines)
- ✓ loss.py → test_loss.py (632 lines)
- ✓ hyperbolic.py → test_hyperbolic.py (463 lines)
- ✓ evaluation.py → test_evaluation.py (811 lines)
- ✓ curriculum.py → test_curriculum.py
- ✓ naics_model.py → test_naics_model.py (767 lines)
- ✓ hard_negative_mining.py → test_hard_negative_mining.py
- ✓ hyperbolic_clustering.py → test_hyperbolic_clustering.py (165 lines)
- ✓ false_negative_strategies.py → test_false_negative_strategy.py
- ✓ dataloader/* → test_datamodule.py (902 lines), test_streaming_dataset.py (740 lines),
  test_tokenization_cache.py (634 lines)

**Graph Model (8/8 files)**
- ✓ hgcn.py → test_hgcn.py (267 lines)
- ✓ evaluation.py → test_hgcn_metrics.py, test_graph_downstream_evaluation.py
- ✓ curriculum/* → test_graph_curriculum.py (838 lines), test_graph_preprocessing.py (425 lines)
- ✓ dataloader/* → test_hgcn_datamodule.py, test_hgcn_streaming_dataset.py

---

## Critical Gaps (9 untested files)

### 🔴 High Priority (Core Functionality)

#### 1. **data/positive_sampling.py** (428 lines)
**Impact:** Critical - Used by both text and graph model training pipelines
**Complexity:** High - Complex taxonomy-based sampling logic

**Missing test coverage:**
- `build_taxonomy()` - Handles merged sectors (31-33, 44-45, 48-49)
- `build_anchor_list()` - Anchor enumeration
- `_linear_skip()` - Descendant finding logic
- `_build_descendants()` - Descendants stratum construction
- `_build_ancestors_6()` - Ancestors stratum construction
- `_build_siblings()` - Siblings stratum construction
- `build_positives()` - Main positive sampling orchestration
- `create_stratified_positives()` - Stratified sampling

**Risks without tests:**
- Silent bugs in taxonomy handling (merged sectors)
- Incorrect stratum weight calculations
- Edge cases in hierarchy traversal
- Sampling distribution issues

**Recommended tests:**
```python
# tests/unit/test_positive_sampling.py

def test_build_taxonomy_handles_merged_sectors():
    """Test that sectors 31-33, 44-45, 48-49 are correctly merged."""

def test_linear_skip_finds_next_level():
    """Test descendant finding at each hierarchy level."""

def test_build_descendants_correct_weights():
    """Test stratum weight calculation (1/num_positives)."""

def test_build_ancestors_for_level_6():
    """Test ancestor chain extraction for leaf nodes."""

def test_create_stratified_positives_distribution():
    """Test stratified sampling produces correct distribution."""

def test_edge_cases_single_child():
    """Test behavior when nodes have single children."""
```

---

#### 2. **utils/naics_hierarchy.py** (88 lines)
**Impact:** High - Used throughout the codebase for hierarchy operations
**Complexity:** Medium - Graph traversal logic

**Missing test coverage:**
- `NaicsHierarchy.__init__()` - Deduplication and parent conflict handling
- `from_relations_parquet()` - Parquet loading and schema validation
- `get_parent()` / `get_children()` / `get_siblings()` - Core traversal
- Caching behavior via `@lru_cache`

**Risks without tests:**
- Silent failures with malformed parquet files
- Incorrect parent-child relationships
- Memory issues with caching
- Edge cases (orphan nodes, cycles)

**Recommended tests:**
```python
# tests/unit/test_naics_hierarchy.py

def test_naics_hierarchy_deduplicates_pairs():
    """Test that duplicate parent-child pairs are handled."""

def test_from_relations_parquet_schema_validation():
    """Test error handling for missing columns."""

def test_get_parent_returns_first_observed():
    """Test conflict resolution when child has multiple parents."""

def test_get_siblings_excludes_self():
    """Test sibling enumeration excludes the input code."""

def test_load_naics_hierarchy_caching():
    """Test LRU cache behavior."""

def test_orphan_nodes():
    """Test behavior with nodes that have no parent."""
```

---

#### 3. **utils/distance_matrix.py** (60 lines)
**Impact:** High - Used in training for hierarchy preservation loss
**Complexity:** Medium - Matrix operations with validation

**Missing test coverage:**
- `load_distance_submatrix()` - Submatrix extraction
- Code alignment and reordering
- Error handling (missing codes, NaN values)
- Column name parsing

**Risks without tests:**
- Misaligned distance matrices (training with wrong ground truth)
- Silent NaN propagation
- Index mismatch errors

**Recommended tests:**
```python
# tests/unit/test_distance_matrix.py

def test_load_distance_submatrix_alignment():
    """Test that submatrix is correctly aligned to node_codes."""

def test_missing_codes_raises_error():
    """Test error when codes are missing from matrix."""

def test_nan_replacement():
    """Test NaN values are replaced with zeros and warning is logged."""

def test_column_name_parsing():
    """Test extraction of codes from column names."""
```

---

### 🟡 Medium Priority (Developer Tools)

#### 4. **tools/_visualize_metrics.py** (550 lines)
**Impact:** Medium - Developer productivity tool
**Complexity:** High - Complex matplotlib visualizations

**Why test this?**
While it's a CLI tool, it contains complex logic for:
- Metric file parsing
- Multi-experiment comparisons
- Plot generation (12 different plot types)
- HTML report generation

**Recommended approach:**
- Integration tests that generate sample outputs
- Snapshot testing for generated HTML/images
- Unit tests for metric parsing logic

---

#### 5. **tools/_investigate_hierarchy.py** (203 lines)
**Impact:** Medium - Debugging tool for hierarchy correlation
**Complexity:** Medium - Statistical analysis

**Recommended tests:**
- Test correlation computation
- Test report generation
- Test with synthetic data

---

#### 6. **tools/metrics_tools.py** (176 lines)
**Impact:** Medium - Metric extraction utilities
**Complexity:** Medium

**Recommended tests:**
- Test metric extraction from checkpoints
- Test CSV export functionality

---

### 🟢 Low Priority (Infrastructure)

#### 7. **tools/config_tools.py** (80 lines)
**Impact:** Low - Config display utility
**Complexity:** Low

**Recommended approach:**
- Simple smoke tests
- Test config rendering

---

#### 8. **utils/warnings.py** (159 lines)
**Impact:** Low - Warning suppression
**Complexity:** Low

**Recommended approach:**
- Test warning suppression
- Test `list_suppressed_warnings()`

---

#### 9. **cli.py** (38 lines)
**Impact:** Low - Thin CLI wrapper
**Complexity:** Very low

**Recommended approach:**
- Integration test for `--help`
- Test version display

---

## Integration Test Gaps

**Current state:** Integration tests directory is **empty** (only `__init__.py`)

### 🔴 Critical: Missing End-to-End Tests

**What's missing:**
1. **Full training pipeline tests**
   - Text model training (1-2 epochs on tiny dataset)
   - HGCN refinement training
   - Checkpoint saving/loading
   - Resume from checkpoint

2. **Data pipeline integration**
   - `naics-embedder data all` (full pipeline)
   - Data validation across steps

3. **Multi-GPU/distributed tests**
   - DDP training (if hardware available)
   - Global batch gathering

4. **Curriculum progression tests**
   - Phase transitions in graph curriculum
   - False negative detection workflow

5. **Evaluation workflow tests**
   - Embedding generation + evaluation
   - Metric computation pipeline

**Recommended approach:**
```python
# tests/integration/test_training_pipeline.py

@pytest.mark.slow
def test_text_model_training_end_to_end(tmp_path):
    """Test complete text model training pipeline."""
    # 1. Generate tiny synthetic dataset
    # 2. Run training for 2 epochs
    # 3. Verify checkpoint is saved
    # 4. Verify metrics are logged
    # 5. Resume from checkpoint

@pytest.mark.slow
def test_hgcn_training_end_to_end(tmp_path):
    """Test complete HGCN training pipeline."""

@pytest.mark.slow
def test_data_pipeline_end_to_end(tmp_path):
    """Test data preparation pipeline."""
```

---

## Test Quality Issues

While coverage is good, consider improving:

### 1. **Edge Case Coverage**

Review existing tests for edge cases:
- Empty datasets
- Single-node graphs
- Extreme curvature values
- NaN/Inf in embeddings
- GPU OOM scenarios

### 2. **Property-Based Testing**

Consider using `hypothesis` for:
- Hyperbolic manifold invariants
- Taxonomy structure properties
- Distance matrix symmetry

Example:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=10, max_size=10))
def test_lorentz_inner_product_invariant(coords):
    """Test Lorentz inner product maintains hyperboloid constraint."""
```

### 3. **Performance/Regression Tests**

Add benchmarks for:
- Tokenization cache hit rate
- Streaming dataset throughput
- Hyperbolic operation speed

---

## Recommended Test Infrastructure Improvements

### 1. **Test Data Fixtures**

Create shared test data fixtures:
```python
# tests/fixtures/synthetic_data.py

def create_tiny_naics_dataset(num_codes=20):
    """Generate minimal NAICS dataset for testing."""

def create_tiny_triplets(num_triplets=100):
    """Generate minimal training triplets."""
```

### 2. **Test Configuration**

Add test-specific configs:
```yaml
# conf/test_config.yaml
# Minimal config for fast testing
model:
  lora_r: 2
  num_experts: 2
data_loader:
  batch_size: 4
training:
  max_epochs: 2
```

### 3. **CI/CD Improvements**

- Add coverage threshold enforcement (e.g., 85%)
- Separate fast/slow test runs
- Generate coverage reports in PRs
- Add mutation testing (e.g., `mutmut`)

---

## Priority Action Items

### Week 1: Critical Gaps
1. ✅ Test `data/positive_sampling.py` (highest impact)
2. ✅ Test `utils/naics_hierarchy.py`
3. ✅ Test `utils/distance_matrix.py`

### Week 2: Integration Tests
4. ✅ Create integration test for text model training
5. ✅ Create integration test for data pipeline
6. ✅ Create integration test for HGCN training

### Week 3: Tools & Quality
7. ✅ Add tests for visualization tools (basic smoke tests)
8. ✅ Add edge case tests to existing test suite
9. ✅ Add property-based tests for hyperbolic operations

### Week 4: Infrastructure
10. ✅ Set up coverage threshold enforcement
11. ✅ Create shared test data fixtures
12. ✅ Add performance benchmarks

---

## Success Metrics

**Target Coverage:**
- Overall: 90%+ (currently 81.2%)
- Critical paths: 100%
- Integration tests: 10+ scenarios

**Quality Metrics:**
- Test execution time: <2 minutes (unit tests)
- Test flakiness: <1%
- Coverage trend: increasing month-over-month

---

## Conclusion

The NAICS Embedder test suite is **strong** for core ML components but has **critical gaps** in:
1. Data sampling utilities (`positive_sampling.py`)
2. Hierarchy utilities (`naics_hierarchy.py`, `distance_matrix.py`)
3. Integration/end-to-end tests (empty)
4. Developer tools (visualization, investigation)

**Immediate priority:** Test the data sampling and hierarchy utilities, as these are used in
production training and silent bugs could corrupt results.

**Secondary priority:** Add integration tests to catch pipeline-level issues that unit tests miss.
