# Linear Branch Implementation Strategy

## Understanding Linear Branches

A linear branch occurs when a parent has **exactly one child** at the immediate level. According to the issue:

> "Dynamic Skip-Level: If an anchor has only one child at the immediate level (Linear Branch), skip that level and stratify by the grandchild nodes to ensure diversity."

## Mapping by Level

- **Level 2 (Sector) linear**: Has only 1 child at level 3 → Skip to level 4 (code_4)
- **Level 3 (Subsector) linear**: Has only 1 child at level 4 → Skip to level 5 (code_5)  
- **Level 4 (Industry Group) linear**: Has only 1 child at level 5 → Skip to level 6 (code_6)
- **Level 5 (NAICS Industry) linear**: Has only 1 child at level 6 → Cannot skip (already at leaves)

## Implementation Approach

### Step 1: Identify Linear Branches by Level

We need to identify which anchors are linear at each level. The `_single_parents` function currently identifies parents with 1 child, but we need to map this by anchor level.

### Step 2: Handle Linear Branches Separately

Instead of filtering out linear branches, we should:
1. Process non-linear branches normally (current logic)
2. Process linear branches with skip-level logic (new logic)

### Step 3: Skip-Level Stratification

For linear branches:
- Use grandchildren as the stratum (skip immediate children)
- Apply the same sampling rules (all for level 2-3, sample for level 4-5)

## Proposed Code Structure

```python
def _get_linear_branches_by_level(
    anchors: pl.DataFrame,
    taxonomy: pl.DataFrame
) -> Dict[int, List[str]]:
    """
    Identify linear branches by anchor level.
    Returns: {level: [list of linear anchor codes]}
    """
    linear_by_level = {}
    
    # For each level 2-5, find anchors with only 1 child
    for level in [2, 3, 4, 5]:
        child_level = level + 1
        
        # Get all anchors at this level
        level_anchors = anchors.filter(pl.col('level').eq(level))
        
        # Join with taxonomy to find children
        children = (
            level_anchors
            .join(taxonomy, how='cross')
            .filter(
                pl.col(f'code_{child_level}').str.starts_with(pl.col('anchor'))
            )
            .group_by('anchor')
            .agg(
                num_children=pl.col(f'code_{child_level}').n_unique()
            )
            .filter(pl.col('num_children').eq(1))
            .get_column('anchor')
            .to_list()
        )
        
        linear_by_level[level] = children
    
    return linear_by_level


def _descendants_linear(
    anchors: pl.DataFrame,
    taxonomy: pl.DataFrame,
    linear_by_level: Dict[int, List[str]]
) -> pl.DataFrame:
    """
    Handle linear branches with skip-level logic.
    For linear branches, stratify by grandchildren instead of children.
    """
    results = []
    
    for level in [2, 3, 4]:
        if level not in linear_by_level or not linear_by_level[level]:
            continue
            
        grandchild_level = level + 2
        
        # Can't skip beyond level 6
        if grandchild_level > 6:
            continue
        
        linear_anchors = anchors.filter(
            pl.col('level').eq(level),
            pl.col('anchor').is_in(linear_by_level[level])
        )
        
        # Join with taxonomy and filter descendants
        descendants = (
            linear_anchors
            .join(taxonomy, how='cross')
            .with_columns(
                filter_code=pl.when(pl.col('level').eq(2))
                          .then(pl.col('code'))
                          .otherwise(pl.col('code_6'))
            )
            .filter(
                pl.col('filter_code').str.starts_with(pl.col('anchor'))
            )
            .select(
                level=pl.col('level'),
                anchor=pl.col('anchor'),
                stratum=pl.col(f'code_{grandchild_level}')  # Skip to grandchildren
            )
            .unique(maintain_order=True)
        )
        
        # Apply same sampling logic as non-linear
        if level in [2, 3]:
            # All grandchildren when num_strata > 1
            sampled = (
                descendants
                .group_by('level', 'anchor', maintain_order=True)
                .agg(
                    stratum=pl.col('stratum'),
                    num_strata=pl.col('stratum').len()
                )
                .filter(pl.col('num_strata').gt(1))
                .drop('num_strata')
                .explode('stratum')
            )
        elif level == 4:
            # Sample up to 2 from grandchildren
            sampled = (
                descendants
                .group_by('level', 'anchor', maintain_order=True)
                .agg(
                    stratum=pl.col('stratum'),
                    num_strata=pl.col('stratum').len()
                )
                .filter(pl.col('num_strata').gt(1))
                .with_columns(
                    pl.col('stratum').list.sample(2, shuffle=True, seed=42)
                )
                .explode('stratum')
            )
        
        results.append(sampled)
    
    if results:
        return pl.concat(results)
    else:
        return pl.DataFrame(schema={'level': pl.Int8, 'anchor': pl.Utf8, 'stratum': pl.Utf8})


def _descendants(
    anchors: pl.DataFrame, 
    taxonomy: pl.DataFrame,
    single_parents: List[str]
) -> pl.DataFrame:
    """
    Main function - handles both linear and non-linear branches.
    """
    # Identify linear branches by level
    linear_by_level = _get_linear_branches_by_level(anchors, taxonomy)
    
    # Process non-linear branches (current logic, but exclude linear ones)
    linear_anchors_all = []
    for level, anchors_list in linear_by_level.items():
        linear_anchors_all.extend(anchors_list)
    
    # Non-linear descendants (current logic)
    descendants_nonlinear = (
        # ... existing logic but also exclude linear_anchors_all
        anchors
        .filter(
            pl.col('level').lt(6),
            ~pl.col('anchor').is_in(single_parents),
            ~pl.col('anchor').is_in(linear_anchors_all)  # Exclude linear
        )
        # ... rest of existing logic
    )
    
    # Linear descendants (new skip-level logic)
    descendants_linear = _descendants_linear(anchors, taxonomy, linear_by_level)
    
    # Combine both
    return pl.concat([descendants_nonlinear, descendants_linear])
```

## Key Points

1. **Separate identification**: Identify linear branches by level, not just as a flat list
2. **Skip-level stratification**: Use grandchildren (level + 2) as stratum for linear branches
3. **Same sampling rules**: Apply the same sampling logic (all vs. sample) based on anchor level
4. **Level 5 linear**: Cannot skip (grandchildren would be level 7, which doesn't exist)
5. **Combine results**: Merge linear and non-linear results at the end

## Edge Cases

- **Multi-level linear**: If a branch is linear for multiple levels (e.g., 2→3→4 all have 1 child), we only skip the first level
- **Linear at level 5**: These are already at leaves, so we can't skip further - might need special handling
- **Empty results**: Handle case where no linear branches exist


