# Verification Report V2: Stratified Entailment Sampling Implementation

## Fixed Issues ✅

1. **`_ancestors` function**: Now correctly receives `anchors` parameter ✓
2. **`sample_positives`**: Now passes `anchors` to `_ancestors` ✓
3. **`_single_parents` line 110**: Now uses `parent` instead of `parents` ✓

## Remaining Critical Bug ❌

### `_single_parents` Function (Line 112)

**Bug**: Return statement is still inside the loop
- The `return` statement on line 112 is inside the `for` loop
- This causes the function to return after processing only the first iteration (`('code_3', 'code_4')`)
- It will never process `('code_4', 'code_5')` or `('code_5', 'code_6')`

**Current Code**:
```python
def _single_parents(taxonomy: pl.DataFrame) -> List[str]:
    single_parents = []
    for col_1, col_2 in [
        ('code_3', 'code_4'),
        ('code_4', 'code_5'),
        ('code_5', 'code_6')
    ]:
        # ... code ...
        single_parents.extend(parent)
        return sorted(set(single_parents))  # ❌ INSIDE LOOP - WRONG!
```

**Fix needed**:
```python
def _single_parents(taxonomy: pl.DataFrame) -> List[str]:
    single_parents = []
    for col_1, col_2 in [
        ('code_3', 'code_4'),
        ('code_4', 'code_5'),
        ('code_5', 'code_6')
    ]:
        # ... code ...
        single_parents.extend(parent)
    
    return sorted(set(single_parents))  # ✅ OUTSIDE LOOP - CORRECT!
```

## Logic Verification

### ✅ Correct Implementations

1. **Stratification by Level** (Lines 144-147):
   - Level 2 (Sector) → code_3 (Subsector) ✓
   - Level 3 (Subsector) → code_4 (Industry Group) ✓
   - Level 4 (Industry Group) → code_5 (NAICS Industry) ✓
   - Level 5 (NAICS Industry) → code_6 (National Industry) ✓

2. **Level 2 & 3 Anchors** (Lines 154-169):
   - Returns all immediate descendants when num_strata > 1 ✓
   - Matches requirement: "all immediate descendants" for Sectors and Subsectors ✓

3. **Leaf Ancestors** (Lines 255-296):
   - Unpivots code_5, code_4, code_3, code_2 ✓
   - Returns all ancestors for 6-digit codes ✓
   - Should correctly return: 54, 541, 5415, 54151 for anchor 541511 ✓

### ⚠️ Logic Issues

#### 1. Linear Branch Handling (Lines 121-150)

**Current Implementation**: Filters out `single_parents` entirely
```python
.filter(
    pl.col('level').lt(6),
    ~pl.col('anchor').is_in(single_parents)  # Just excludes them
)
```

**Issue Requirement**: 
> "Dynamic Skip-Level: If an anchor has only one child at the immediate level (Linear Branch), skip that level and stratify by the grandchild nodes to ensure diversity."

**Gap**: The code excludes linear branches but doesn't implement the skip-level logic. According to the issue, linear branches should:
- Skip the immediate child level
- Stratify by grandchildren instead
- Still be included in the results

**Status**: ❌ **Does not match requirement**

#### 2. 4-Digit Anchor Stratification (Lines 172-220)

**Issue Requirement**: 
> "Industry Group (4-digit) → Stratify by NAICS Industry (5-digit) and Industry (6-digit)"

**Current Implementation**:
- Samples 2 from 5-digit strata (line 188)
- Filters to exclude selected 5-digit codes (line 196)
- Then samples 2 from 6-digit codes per remaining 5-digit (line 209)
- Combines both into final positives

**Analysis**:
- The filter on line 196 (`~pl.col('code_5').is_in(pl.col('stratum'))`) seems incorrect
- It filters OUT the selected 5-digit codes, then tries to sample 6-digit from the remaining ones
- This doesn't match "stratify by both 5-digit AND 6-digit"
- Should probably sample: 1 from each 5-digit stratum, AND 1 from each 6-digit stratum

**Status**: ⚠️ **Logic unclear, may not match requirement**

#### 3. 5-Digit Anchor Sampling (Lines 224-238)

**Issue Requirement**: 
> "random sample of up to 3 iff num(descendants) > 1"

**Current Implementation**:
```python
.filter(pl.col('num_strata').gt(1))
.explode('stratum')  # Returns ALL strata, not up to 3!
```

**Problem**: 
- The code explodes ALL strata, returning all 6-digit descendants
- Should sample up to 3, not return all
- Previous version had sampling logic that was removed

**Status**: ❌ **Does not match requirement - returns all instead of sampling up to 3**

## Acceptance Criteria Verification

### 1. Sector 11 Test
- **Expected**: Positives from 111, 112, 113, 114, 115
- **Current Logic**: 
  - Level 2 anchor "11" should stratify by code_3
  - `descendants_2` should return all unique code_3 values starting with "11"
  - Should work correctly IF `_single_parents` bug is fixed
- **Status**: ⚠️ **Should work after fixing `_single_parents` bug**

### 2. Leaf 541511 Test
- **Expected**: Positives: 54, 541, 5415, 54151
- **Current Logic**: 
  - `_ancestors` unpivots code_5, code_4, code_3, code_2
  - Should return all ancestors
  - Note: Issue says "54151" but code returns code_5 which might be "54151"
- **Status**: ✅ **Should work correctly**

### 3. Linear Branch Test
- **Expected**: Correctly samples from grandchildren/leaves, ensuring no duplicates
- **Current Logic**: Linear branches are filtered out entirely
- **Status**: ❌ **Does not match requirement**

## Summary

**Status**: ⚠️ **Partially correct, but has critical bugs and logic gaps**

### Critical Issues:
1. ❌ **`_single_parents` return statement inside loop** - Will cause incorrect results
2. ❌ **5-digit sampling returns all instead of up to 3** - Performance/logic issue
3. ❌ **Linear branch handling doesn't implement skip-level logic** - Logic gap

### Minor Issues:
4. ⚠️ **4-digit stratification logic unclear** - May not match requirement

### What Works:
- ✅ `_ancestors` function now correctly implemented
- ✅ Stratification levels are correct
- ✅ Level 2 & 3 anchors return all descendants correctly
- ✅ Leaf ancestor sampling structure is correct

## Recommended Fixes

1. **Fix `_single_parents` return statement** (move outside loop)
2. **Fix 5-digit sampling** to return up to 3 instead of all
3. **Implement skip-level logic** for linear branches
4. **Review and fix 4-digit stratification** logic

