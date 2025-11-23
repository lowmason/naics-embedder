# Verification Report: Issue #20 - Stratified Entailment Sampling

## Issue Requirements Summary

1. **Parent Anchors (2-5 digit)**: 
   - Stratify by immediate child level
   - Dynamic Skip-Level: If anchor has only 1 child, skip to grandchildren
   - Sample at most 1 descendant from each distinct stratum

2. **Leaf Anchors (6-digit)**: 
   - Sample all ancestors (2, 3, 4, 5-digit) as positives

3. **Acceptance Criteria**:
   - Sector 11 → positives from 111, 112, 113, 114, 115 (not just 111)
   - Leaf 541511 → positives: 54, 541, 5415, 54151
   - Linear branches handled correctly (skip-level logic)

## Critical Bugs Found ❌

### Bug 1: `_descendants()` - Wrong Filter Logic (Line 130)

**Location**: `src/naics_embedder/data_loader/streaming_dataset.py:130`

**Issue**: 
```python
.filter(
    pl.col('stratum').is_null()  # ❌ WRONG - filters to keep NULL values only
)
```

**Problem**: 
- This filter keeps only rows where `stratum` is NULL
- It should filter to keep non-null strata: `.filter(pl.col('stratum').is_not_null())`
- Currently, this will return NO results for parent anchors (all valid strata will be filtered out)

**Fix**:
```python
.filter(
    pl.col('stratum').is_not_null()  # ✅ Keep non-null strata
)
```

### Bug 2: `_linear_skip()` - Missing Return Statement (Line 94)

**Location**: `src/naics_embedder/data_loader/streaming_dataset.py:75-94`

**Issue**: 
- If the loop completes without finding a divergent level (all levels have only 1 child), the function implicitly returns `None`
- This will cause issues when `_descendants()` tries to create a list from `None`

**Problem**:
```python
def _linear_skip(anchor: str, taxonomy: pl.DataFrame) -> List[str]:
    # ... loop logic ...
    # If loop completes without returning, function returns None
    # No explicit return statement for edge case
```

**Fix**:
```python
def _linear_skip(anchor: str, taxonomy: pl.DataFrame) -> List[str]:
    lvl = len(anchor)
    anchor_code = f'code_{lvl}'
    codes = [f'code_{i}' for i in range(lvl + 1, 7)]

    for code in codes:
        candidate = (
            taxonomy
            .filter(pl.col(anchor_code).eq(anchor))
            .get_column(code)
            .unique()
            .to_list()
        )

        if lvl == 5:
            return candidate
        elif len(candidate) > 1:
            return sorted(set(candidate))
    
    # ✅ Handle case where all levels have only 1 child (deep linear branch)
    # Return the leaves (code_6) as the final stratum
    return (
        taxonomy
        .filter(pl.col(anchor_code).eq(anchor))
        .get_column('code_6')
        .unique()
        .to_list()
    )
```

## Logic Verification

### ✅ Correct Implementations

1. **`_ancestors()` function (Lines 144-185)**:
   - Correctly handles leaf anchors (level 6)
   - Unpivots code_5, code_4, code_3, code_2
   - Returns all ancestors as positives
   - **Status**: ✅ Correct

2. **Stratification Level Mapping**:
   - Level 2 (Sector) → should stratify by code_3 (Subsector) ✓
   - Level 3 (Subsector) → should stratify by code_4 (Industry Group) ✓
   - Level 4 (Industry Group) → should stratify by code_5 (NAICS Industry) ✓
   - Level 5 (NAICS Industry) → should stratify by code_6 (National Industry) ✓
   - **Status**: ✅ Logic in `_linear_skip()` is correct

3. **Linear Branch Skip Logic**:
   - `_linear_skip()` correctly walks down levels until it finds a divergent level (>1 child)
   - Handles the skip-level requirement
   - **Status**: ✅ Logic is correct (but needs edge case handling)

### ⚠️ Potential Issues

1. **"At Most 1 Per Stratum" Requirement**:
   - **Issue Requirement**: "Sample k positives such that we select at most 1 descendant from each distinct Stratum"
   - **Current Implementation**: Returns ALL codes from each stratum (when exploded)
   - **Analysis**: 
     - The current implementation returns all codes from the divergent stratum
     - For example, if Sector 11 has strata [111, 112, 113, 114, 115], it returns all 5
     - This matches the acceptance criteria for Sector 11 (should return all 5)
     - However, if a stratum has multiple descendants (e.g., 111 has multiple 4-digit codes), it would return all of them
   - **Question**: Does "at most 1 per stratum" mean:
     - Option A: 1 code per stratum level (current implementation - returns all strata)
     - Option B: 1 descendant per stratum code (would need additional sampling)
   - **Status**: ⚠️ **Unclear - may need clarification or additional sampling logic**

2. **Missing k_max Parameter**:
   - The `sample_positives()` function doesn't have a `k_max` parameter
   - Issue mentions "Sample k positives" but there's no limit
   - **Status**: ⚠️ **May be intentional if all positives are needed, or may need to be added**

3. **Performance - Not Fully Vectorized**:
   - `_descendants()` uses a Python loop (lines 115-119) to call `_linear_skip()` for each anchor
   - This is not fully vectorized as requested in the issue
   - **Status**: ⚠️ **Could be optimized but may be acceptable for current scale**

## Acceptance Criteria Verification

### 1. Sector 11 Test ❌

**Expected**: Positives from 111, 112, 113, 114, 115

**Current Logic**:
- Anchor "11" (level 2) → `_linear_skip()` should find code_3 level
- Should return [111, 112, 113, 114, 115] as strata
- After explode, should create 5 rows

**Status**: ❌ **Will FAIL due to Bug 1** (filter keeps NULL instead of non-NULL)
- After fixing Bug 1: ✅ Should work correctly

### 2. Leaf 541511 Test ✅

**Expected**: Positives: 54, 541, 5415, 54151

**Current Logic**:
- `_ancestors()` unpivots code_5, code_4, code_3, code_2
- Should return all ancestors: 54151 (code_5), 5415 (code_4), 541 (code_3), 54 (code_2)

**Status**: ✅ **Should work correctly** (assuming code_5 is "54151" not "5415")

### 3. Linear Branch Test ⚠️

**Expected**: Correctly samples from grandchildren/leaves, ensuring no duplicates

**Current Logic**:
- `_linear_skip()` walks down levels until it finds >1 child
- If all levels have 1 child, it should return leaves (code_6)
- **Issue**: Currently returns `None` if no divergent level found (Bug 2)

**Status**: ⚠️ **Will work after fixing Bug 2, but needs edge case handling**

## Summary

**Status**: ⚠️ **Partially Correct - Has Critical Bugs**

### Critical Issues:
1. ❌ **Bug 1**: `_descendants()` filter keeps NULL instead of non-NULL - **WILL CAUSE ZERO RESULTS**
2. ❌ **Bug 2**: `_linear_skip()` doesn't handle deep linear branches - **WILL RETURN None**

### Minor Issues:
3. ⚠️ **"At most 1 per stratum"** interpretation unclear
4. ⚠️ **Not fully vectorized** (Python loop in `_descendants()`)
5. ⚠️ **Missing k_max parameter** (may be intentional)

### What Works:
- ✅ `_ancestors()` function is correct
- ✅ `_linear_skip()` logic is correct (needs edge case handling)
- ✅ Stratification level mapping is correct
- ✅ Overall structure matches requirements

## Recommended Fixes

### Priority 1 (Critical):
1. **Fix Bug 1**: Change `.filter(pl.col('stratum').is_null())` to `.filter(pl.col('stratum').is_not_null())`
2. **Fix Bug 2**: Add return statement for edge case in `_linear_skip()`

### Priority 2 (Clarification):
3. **Clarify "at most 1 per stratum"** requirement - verify if current behavior is correct
4. **Add edge case handling** for deep linear branches that go all the way to leaves

### Priority 3 (Optimization):
5. **Consider vectorization** of `_descendants()` if performance is an issue
6. **Add k_max parameter** if limiting positives is required

## Testing Recommendations

After fixes, test:
1. Sector 11 → should return 111, 112, 113, 114, 115
2. Leaf 541511 → should return 54, 541, 5415, 54151
3. Linear branch anchor → should return grandchildren/leaves correctly
4. Edge case: Anchor with only 1 descendant path all the way to leaves

