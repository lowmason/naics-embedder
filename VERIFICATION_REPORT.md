# Verification Report: Stratified Entailment Sampling Implementation

## Issue #20 Requirements Summary

1. **Parent Anchors (2-5 digit)**: Stratify by immediate child level, skip linear branches, sample at most 1 per stratum
2. **Leaf Anchors (6-digit)**: Sample all ancestors (2, 3, 4, 5-digit)
3. **Acceptance Criteria**:
   - Sector 11 → positives from 111, 112, 113, 114, 115
   - Leaf 541511 → positives: 54, 541, 5415, 54151
   - Linear branches handled correctly

## Critical Bugs Found

### 1. `_single_parents` Function (Lines 75-112)

**Bug 1**: Line 110 uses wrong variable name
```python
single_parents.extend(parents)  # ❌ Should be 'parent', not 'parents'
```

**Bug 2**: Return statement inside loop (Line 112)
- The `return` statement is inside the `for` loop, causing the function to return after the first iteration
- Should be outside the loop

**Fix needed**:
```python
def _single_parents(taxonomy: pl.DataFrame) -> List[str]:
    single_parents = []
    for col_1, col_2 in [
        ('code_3', 'code_4'),
        ('code_4', 'code_5'),
        ('code_5', 'code_6')
    ]:
        parents = (
            taxonomy
            .group_by(col_1)
            .len()
            .filter(pl.col('len').eq(1))
            .get_column(col_1)
            .to_list()
        )
        single_parents.extend(parents)
        
        parent = (
            taxonomy
            .select(col_1, col_2)
            .unique()
            .group_by(col_1)
            .len()
            .filter(pl.col('len').eq(1))
            .get_column(col_1)
            .to_list()
        )
        single_parents.extend(parent)  # ✅ Fixed: 'parent' not 'parents'
    
    return sorted(set(single_parents))  # ✅ Fixed: moved outside loop
```

### 2. `_ancestors` Function (Lines 267-305)

**Bug**: References undefined variable `anchors` (Line 269)
- Function only receives `taxonomy` parameter
- `anchors` is not in scope
- Called from `sample_positives` where `anchors` exists but isn't passed

**Fix needed**:
```python
def _ancestors(anchors: pl.DataFrame, taxonomy: pl.DataFrame) -> pl.DataFrame:
    return (
        anchors  # ✅ Now anchors is a parameter
        .filter(pl.col('level').eq(6))
        # ... rest of function
    )

# In sample_positives:
ancestors = _ancestors(anchors, taxonomy)  # ✅ Pass anchors
```

## Logic Verification

### ✅ Correct Implementations

1. **Stratification by Level** (Lines 144-147):
   - Level 2 → code_3 ✓
   - Level 3 → code_4 ✓
   - Level 4 → code_5 ✓
   - Level 5 → code_6 ✓

2. **Level 2 & 3 Anchors** (Lines 154-169):
   - Returns all immediate descendants when num_strata > 1 ✓
   - Matches requirement for Sectors and Subsectors

3. **Leaf Ancestors** (Lines 267-305, after bug fix):
   - Unpivots code_5, code_4, code_3, code_2 ✓
   - Should return all ancestors for 6-digit codes ✓

### ⚠️ Potential Issues

1. **Linear Branch Handling** (Lines 121-150):
   - Current: Filters out `single_parents` entirely
   - Issue requirement: Should skip the immediate child level and stratify by grandchildren
   - **Gap**: The code excludes linear branches but doesn't implement the skip-level logic

2. **4-Digit Anchor Stratification** (Lines 172-220):
   - Issue requires: Stratify by both 5-digit AND 6-digit
   - Current implementation:
     - Samples 2 from 5-digit strata
     - Then filters and samples from 6-digit
   - **Question**: Does this match "stratify by 5-digit and 6-digit" requirement?
   - The logic seems to sample 2 from 5-digit, then 2 from 6-digit per selected 5-digit
   - This may not be "at most 1 per stratum" as required

3. **5-Digit Anchor Sampling** (Lines 224-250):
   - Samples up to 3 from 6-digit strata ✓
   - But doesn't enforce "at most 1 per stratum" - it samples randomly
   - **Issue**: Should ensure one per distinct stratum, not random sampling

## Acceptance Criteria Check

### ❌ Sector 11 Test
- **Expected**: Positives from 111, 112, 113, 114, 115
- **Current**: Should work if `descendants_2` logic is correct
- **Status**: Needs testing after bug fixes

### ❌ Leaf 541511 Test  
- **Expected**: Positives: 54, 541, 5415, 54151
- **Current**: `_ancestors` function should return these after bug fix
- **Status**: Will fail due to `anchors` bug

### ❌ Linear Branch Test
- **Expected**: Correctly samples from grandchildren/leaves
- **Current**: Linear branches are filtered out, not handled with skip-level logic
- **Status**: **Does not match requirement**

## Recommendations

1. **Fix critical bugs** in `_single_parents` and `_ancestors`
2. **Implement skip-level logic** for linear branches instead of filtering them out
3. **Clarify 4-digit stratification** - verify if current logic matches requirement
4. **Enforce "one per stratum"** for 5-digit anchors instead of random sampling
5. **Add unit tests** for acceptance criteria

## Summary

**Status**: ❌ **Does NOT correctly implement the issue**

**Critical Issues**:
- 2 bugs that will cause runtime errors
- Linear branch handling doesn't match requirements
- Some sampling logic may not enforce "one per stratum" correctly

**Next Steps**:
1. Fix the 2 critical bugs
2. Review and fix linear branch skip-level logic
3. Verify 4-digit and 5-digit sampling matches requirements
4. Test against acceptance criteria

