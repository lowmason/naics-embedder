# NAICS Gemini: Dataclass to Pydantic Migration Guide

This document provides complete instructions for migrating from dataclasses to Pydantic models with YAML configuration files.

## Overview

### What Changed

1. **Added Pydantic Models** to `src/naics_gemini/utils/config.py`:
   - `DownloadConfig` - for data downloading/preprocessing
   - `RelationsConfig` - for relations computation
   - `DistancesConfig` - for distances computation
   - `TripletsConfig` - for triplets generation
   - `TokenizationConfig` - for tokenization caching
   - `StreamingConfig` - for streaming dataset configuration

2. **Created YAML Config Files** in `conf/`:
   - `conf/data_generation/download.yaml`
   - `conf/data_generation/relations.yaml`
   - `conf/data_generation/distances.yaml`
   - `conf/data_generation/triplets.yaml`
   - `conf/data_loader/tokenization.yaml`

3. **Updated Python Files** to use Pydantic configs instead of dataclasses

### Benefits

- ✅ **Type Safety**: Automatic validation with Pydantic
- ✅ **Centralized Config**: All settings in YAML files
- ✅ **Better Documentation**: Field descriptions and defaults
- ✅ **Validation**: Automatic checking of config values
- ✅ **Consistency**: Same pattern across entire codebase

---

## File-by-File Migration Guide

### 1. `src/naics_gemini/data_generation/download_data.py`

**OLD (Dataclass)**:
```python
from dataclasses import asdict, dataclass, field

@dataclass
class Config:
    output_parquet: str = './data/naics_descriptions.parquet'
    url_codes: str = 'https://...'
    # ... more fields
    
    schema_codes: Dict[str, pl.DataType] = field(
        default_factory=lambda: {
            'Seq. No.': pl.UInt32,
            '2022 NAICS US   Code': pl.Utf8,
            '2022 NAICS US Title': pl.Utf8,
        }
    )
```

**NEW (Pydantic)**:
```python
from naics_gemini.utils.config import DownloadConfig

def download_preprocess_data() -> pl.DataFrame:
    # Load config from YAML
    cfg = DownloadConfig.from_yaml('conf/data_generation/download.yaml')
    
    # Use config fields as before
    logger.info(f'Downloading from {cfg.url_codes}')
    
    # Convert schema strings to Polars types
    import polars as pl
    schema_codes = {k: getattr(pl, v) for k, v in cfg.schema_codes.items()}
    
    # Rest of the code remains similar
    df = pl.read_excel(
        source=data,
        sheet_name=cfg.sheet_codes,
        schema_overrides=schema_codes
    ).rename(cfg.rename_codes)
```

**Key Changes**:
1. Remove `@dataclass` and `Config` class definition
2. Import `DownloadConfig` from utils
3. Use `DownloadConfig.from_yaml()` to load config
4. Convert schema strings to Polars types (since YAML can't store Python types)

---

### 2. `src/naics_gemini/data_generation/compute_relations.py`

**OLD**:
```python
from dataclasses import asdict, dataclass, field

@dataclass
class Config:
    input_parquet: str = './data/naics_descriptions.parquet'
    output_parquet: str = './data/naics_relations.parquet'
    relation_id: Dict[str, int] = field(default_factory=lambda: {...})
```

**NEW**:
```python
from naics_gemini.utils.config import RelationsConfig

def calculate_pairwise_relations() -> pl.DataFrame:
    cfg = RelationsConfig.from_yaml('conf/data_generation/relations.yaml')
    
    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    
    # Use cfg.relation_id directly - it's already a dict
    # Rest of code remains the same
```

**Key Changes**:
1. Import `RelationsConfig`
2. Use `.from_yaml()` to load
3. Use `.model_dump_json()` instead of `json.dumps(asdict(cfg))`

---

### 3. `src/naics_gemini/data_generation/compute_distances.py`

**OLD**:
```python
@dataclass
class Config:
    input_parquet: str = './data/naics_descriptions.parquet'
    output_parquet: str = './data/naics_distances.parquet'
```

**NEW**:
```python
from naics_gemini.utils.config import DistancesConfig

def calculate_pairwise_distances() -> pl.DataFrame:
    cfg = DistancesConfig.from_yaml('conf/data_generation/distances.yaml')
    
    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    
    # Rest remains the same
```

---

### 4. `src/naics_gemini/data_generation/create_triplets.py`

**OLD**:
```python
@dataclass
class Config:
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    distances_parquet: str = './data/naics_distances.parquet'
    relations_parquet: str = './data/naics_relations.parquet'
    output_parquet: str = './data/naics_training_pairs'
```

**NEW**:
```python
from naics_gemini.utils.config import TripletsConfig

def generate_training_triplets() -> pl.DataFrame:
    cfg = TripletsConfig.from_yaml('conf/data_generation/triplets.yaml')
    
    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    
    # Rest remains the same
```

---

### 5. `src/naics_gemini/data_loader/tokenization_cache.py`

**OLD**:
```python
from dataclasses import dataclass

@dataclass
class TokenizationConfig:
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    max_length: Optional[int] = None
    output_path: str = './data/token_cache/token_cache.pt'
    
    def __post_init__(self):
        if self.max_length is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.max_length = tokenizer.model_max_length
```

**NEW**:
```python
from naics_gemini.utils.config import TokenizationConfig

def tokenization_cache(
    config_path: str = 'conf/data_loader/tokenization.yaml'
) -> Dict[int, Dict[str, torch.Tensor]]:
    '''Get tokenization cache, loading from disk or building if necessary.'''
    
    # Load config
    cfg = TokenizationConfig.from_yaml(config_path)
    
    # Handle max_length default
    if cfg.max_length is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        cfg.max_length = tokenizer.model_max_length
    
    # Try to load from cache
    cache = load_tokenization_cache(cfg.output_path)
    if cache is not None:
        return cache
    
    # Build cache if it doesn't exist
    cache = build_tokenization_cache(
        cfg.descriptions_parquet, 
        cfg.tokenizer_name, 
        cfg.max_length
    )
    save_tokenization_cache(cache, cfg.output_path)
    
    return cache
```

**Key Changes**:
1. Import `TokenizationConfig`
2. Add `config_path` parameter with default
3. Use `.from_yaml()` to load
4. Handle `max_length` after loading (can't do in Pydantic easily)
5. Remove standalone function `TokenizationConfig()` class

---

### 6. `src/naics_gemini/data_loader/streaming_dataset.py`

**OLD**:
```python
from dataclasses import dataclass, fields

@dataclass
class CurriculumConfig:
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    # ... many more fields
    
    def items(self):
        for f in fields(self):
            if not f.name.endswith('_parquet') and f.name != 'seed':
                v = getattr(self, f.name)
                if v is not None:
                    yield f.name, v
```

**NEW**:
```python
from naics_gemini.utils.config import StreamingConfig

def create_streaming_generator(
    curriculum: StreamingConfig
) -> Iterator[Dict[str, Any]]:
    '''Generate streaming triplets based on curriculum config.'''
    
    # Access config fields directly
    descriptions_parquet = curriculum.descriptions_parquet
    triplets_parquet = curriculum.triplets_parquet
    anchor_level = curriculum.anchor_level
    n_positives = curriculum.n_positives
    n_negatives = curriculum.n_negatives
    
    # Get items for filtering (similar to old .items() method)
    config_dict = curriculum.model_dump(exclude_none=True, exclude={'seed'})
    config_dict = {
        k: v for k, v in config_dict.items()
        if not k.endswith('_parquet')
    }
    
    # Build filters from curriculum
    exprs = []
    for k, v in config_dict.items():
        if isinstance(v, list):
            exprs.append(pl.col(k).is_in(v))       
        if isinstance(v, bool):
            exprs.append(pl.col(k).eq(v))
    
    # Rest of the code remains similar
```

**Key Changes**:
1. Change `CurriculumConfig` to `StreamingConfig`
2. Remove `.items()` method - use `.model_dump()` instead
3. Use `exclude_none=True` to skip None values
4. Filter out parquet paths and seed manually

---

## Complete Updated File Examples

### Updated `download_data.py`

```python
import logging
from pathlib import Path
from io import BytesIO
from typing import Dict, Optional, Set, Tuple

import polars as pl

from naics_gemini.utils.config import DownloadConfig
from naics_gemini.utils.utilities import download_with_retry, parquet_stats

logger = logging.getLogger(__name__)


def _read_xlsx_bytes(
    data: bytes,
    sheet: str,
    schema: Dict[str, str],
    cols: Dict[str, str]
) -> pl.DataFrame:
    '''Read Excel data from bytes into a Polars DataFrame.'''
    
    # Convert schema strings to Polars types
    schema_types = {k: getattr(pl, v) for k, v in schema.items()}
    
    return (
        pl.read_excel(
            BytesIO(data),
            sheet_name=sheet,
            columns=list(schema_types.keys()),
            schema_overrides=schema_types
        )
        .rename(mapping=cols)
    )


def download_preprocess_data() -> pl.DataFrame:
    '''Download and preprocess NAICS data using config from YAML.'''
    
    # Load configuration from YAML
    cfg = DownloadConfig.from_yaml('conf/data_generation/download.yaml')
    
    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')
    
    # Download files
    titles_df = _read_xlsx_bytes(
        download_with_retry(cfg.url_codes),
        cfg.sheet_codes,
        cfg.schema_codes,
        cfg.rename_codes
    )
    
    # ... rest of processing ...
    
    # Write output
    naics_final.write_parquet(cfg.output_parquet)
    
    parquet_stats(
        parquet_df=naics_final,
        message='NAICS codes (text + hierarchy) written to:',
        output_parquet=cfg.output_parquet,
        logger=logger
    )
    
    return naics_final
```

---

### Updated `tokenization_cache.py`

```python
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import polars as pl
import torch
from transformers import AutoTokenizer

from naics_gemini.utils.config import TokenizationConfig

logger = logging.getLogger(__name__)


def tokenization_cache(
    config_path: str = 'conf/data_loader/tokenization.yaml'
) -> Dict[int, Dict[str, torch.Tensor]]:
    '''
    Get tokenization cache, loading from disk or building if necessary.
    
    Args:
        config_path: Path to tokenization config YAML file
        
    Returns:
        Dictionary mapping code indices to tokenized representations
    '''
    
    # Load configuration
    cfg = TokenizationConfig.from_yaml(config_path)
    
    # Handle max_length default (needs tokenizer instance)
    if cfg.max_length is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        cfg.max_length = tokenizer.model_max_length
    
    # Try to load from cache
    cache = load_tokenization_cache(cfg.output_path)
    if cache is not None:
        return cache
    
    # Build cache if it doesn't exist
    cache = build_tokenization_cache(
        cfg.descriptions_parquet, 
        cfg.tokenizer_name, 
        cfg.max_length
    )
    save_tokenization_cache(cache, cfg.output_path)
    
    return cache


def build_tokenization_cache(
    descriptions_path: str,
    tokenizer_name: str,
    max_length: int
) -> Dict[int, Dict[str, torch.Tensor]]:
    '''Build tokenization cache from descriptions file.'''
    
    logger.info('Building tokenization cache...')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # ... rest of implementation ...
    
    return cache
```

---

## Migration Checklist

### Phase 1: Setup (✅ Complete)
- [x] Created enhanced `config.py` with all Pydantic models
- [x] Created YAML config files in `conf/data_generation/`
- [x] Created YAML config files in `conf/data_loader/`

### Phase 2: Update Data Generation Files
- [ ] Update `download_data.py`
  - Remove `@dataclass` Config
  - Import `DownloadConfig`
  - Use `.from_yaml()` to load
  - Convert schema strings to Polars types
  
- [ ] Update `compute_relations.py`
  - Remove `@dataclass` Config
  - Import `RelationsConfig`
  - Use `.from_yaml()` to load
  - Use `.model_dump_json()` for logging
  
- [ ] Update `compute_distances.py`
  - Remove `@dataclass` Config
  - Import `DistancesConfig`
  - Use `.from_yaml()` to load
  
- [ ] Update `create_triplets.py`
  - Remove `@dataclass` Config
  - Import `TripletsConfig`
  - Use `.from_yaml()` to load

### Phase 3: Update Data Loader Files
- [ ] Update `tokenization_cache.py`
  - Remove `@dataclass` TokenizationConfig
  - Import `TokenizationConfig` from utils.config
  - Add `config_path` parameter to main function
  - Use `.from_yaml()` to load
  
- [ ] Update `streaming_dataset.py`
  - Remove `@dataclass` CurriculumConfig
  - Import `StreamingConfig` from utils.config
  - Replace `.items()` with `.model_dump()`
  - Update type hints

### Phase 4: Update datamodule.py
- [ ] Update imports to use `StreamingConfig`
- [ ] Update instantiation to pass config properly

### Phase 5: Testing
- [ ] Test data generation pipeline
- [ ] Test tokenization cache
- [ ] Test streaming dataset
- [ ] Test training pipeline end-to-end

---

## Common Patterns

### Pattern 1: Loading Config

```python
# OLD
cfg = Config()

# NEW
cfg = YourConfig.from_yaml('path/to/config.yaml')
```

### Pattern 2: Accessing Fields

```python
# Same as before!
value = cfg.field_name
```

### Pattern 3: Logging Config

```python
# OLD
import json
from dataclasses import asdict
logger.info(json.dumps(asdict(cfg), indent=2))

# NEW
logger.info(cfg.model_dump_json(indent=2))
```

### Pattern 4: Converting Schemas

```python
# Since YAML can't store Python types directly
schema_dict = {'Code': 'Utf8', 'Value': 'Int64'}

# Convert to Polars types
import polars as pl
schema_types = {k: getattr(pl, v) for k, v in schema_dict.items()}
# Now: schema_types = {'Code': pl.Utf8, 'Value': pl.Int64}
```

### Pattern 5: Filtering Config Items

```python
# OLD (dataclass)
def items(self):
    for f in fields(self):
        if not f.name.endswith('_parquet'):
            v = getattr(self, f.name)
            if v is not None:
                yield f.name, v

# NEW (Pydantic)
config_dict = cfg.model_dump(
    exclude_none=True,
    exclude={'seed', 'descriptions_parquet', 'triplets_parquet'}
)
for k, v in config_dict.items():
    # Process items
```

---

## Validation Benefits

### Automatic Type Checking

```python
# This will raise ValidationError:
cfg = DistancesConfig(
    input_parquet=123  # Wrong type! Should be str
)

# This will raise ValidationError:
cfg = RelationsConfig(
    relation_id={'child': 'one'}  # Wrong value type! Should be int
)
```

### Range Validation

```python
# From TokenizationConfig:
max_length: Optional[int] = Field(default=None)

# This would raise ValidationError if we added gt=0:
cfg = TokenizationConfig(max_length=-1)  # Negative not allowed
```

### Custom Validation

```python
# From Config class:
@field_validator('batch_size')
@classmethod
def warn_large_batch(cls, v: int) -> int:
    if v > 128:
        logger.warning(f'Large batch_size={v} may cause OOM errors')
    return v
```

---

## Troubleshooting

### Issue: ImportError for Pydantic models

**Solution**: Make sure you're importing from the correct module:
```python
from naics_gemini.utils.config import DownloadConfig, RelationsConfig, ...
```

### Issue: Config file not found

**Solution**: Check that YAML files are in correct locations:
```
conf/
├── data_generation/
│   ├── download.yaml
│   ├── relations.yaml
│   ├── distances.yaml
│   └── triplets.yaml
└── data_loader/
    └── tokenization.yaml
```

### Issue: Schema conversion failing

**Solution**: Remember to convert string schema names to Polars types:
```python
schema_dict = cfg.schema_codes  # {'Code': 'Utf8', ...}
schema_types = {k: getattr(pl, v) for k, v in schema_dict.items()}
```

### Issue: .items() method missing

**Solution**: Replace with `.model_dump()`:
```python
# OLD
for k, v in config.items():
    ...

# NEW
for k, v in config.model_dump(exclude_none=True).items():
    ...
```

---

## Next Steps

1. **Copy config.py**: Copy the new `config.py` to your project
2. **Copy YAML files**: Copy all YAML configs to `conf/` directory
3. **Update Python files**: Follow the migration guide for each file
4. **Test incrementally**: Test each component after updating
5. **Run full pipeline**: Verify end-to-end functionality

---

## Summary

This refactoring provides:
- **Better type safety** through Pydantic validation
- **Centralized configuration** in YAML files
- **Improved documentation** with field descriptions
- **Consistency** across all modules
- **Easier testing** with config file overrides

All configuration is now in one place, making it easier to:
- Adjust settings without code changes
- Share configurations between environments
- Version control configuration separately
- Validate configuration before runtime