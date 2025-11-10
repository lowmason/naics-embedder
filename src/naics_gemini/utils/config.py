# -------------------------------------------------------------------------------------------------
# Path configuration
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------------------------------------

class PathsConfig(BaseModel):

    '''File system paths for data, outputs, and checkpoints.'''
    
    data_dir: str = Field(
        default='./data',
        description='Directory containing NAICS data files'
    )
    output_dir: str = Field(
        default='./outputs',
        description='Directory for training outputs and logs'
    )
    checkpoint_dir: str = Field(
        default='./checkpoints',
        description='Directory for model checkpoints'
    )


# -------------------------------------------------------------------------------------------------
# Data Configuration
# -------------------------------------------------------------------------------------------------

class DataConfig(BaseModel):

    '''Data loading and preprocessing configuration.'''
    
    descriptions_path: str = Field(
        default='./data/naics_descriptions.parquet',
        description='Path to NAICS descriptions parquet file'
    )
    triplets_path: str = Field(
        default='./data/naics_training_pairs',
        description='Path to training triplets directory'
    )
    tokenizer_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description='HuggingFace tokenizer name'
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        le=512,
        description='Training batch size'
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description='Number of data loading workers'
    )
    val_split: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description='Validation split fraction'
    )
    
    
    @field_validator('batch_size')
    @classmethod
    def warn_large_batch(cls, v: int) -> int:

        '''Warn about potentially problematic batch sizes.'''

        if v > 128:
            logger.warning(f'Large batch_size={v} may cause OOM errors')
        return v
    

    @field_validator('descriptions_path', 'triplets_path')
    @classmethod
    def validate_paths_exist(cls, v: str) -> str:

        '''Validate that data paths exist (optional check).'''
        
        path = Path(v)
        if not path.exists():
            logger.warning(f'Data path does not exist yet: {v}')
        return v


# -------------------------------------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------------------------------------

class LoRAConfig(BaseModel):

    '''LoRA (Low-Rank Adaptation) configuration.'''
    
    r: int = Field(
        default=8,
        gt=0,
        le=64,
        description='LoRA rank (lower = fewer parameters)'
    )
    alpha: int = Field(
        default=16,
        gt=0,
        description='LoRA scaling factor'
    )
    dropout: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description='LoRA dropout rate'
    )


class MoEConfig(BaseModel):

    '''Mixture of Experts configuration.'''
    
    enabled: bool = Field(
        default=True,
        description='Whether to use MoE layer'
    )
    num_experts: int = Field(
        default=4,
        gt=0,
        le=16,
        description='Number of expert networks'
    )
    top_k: int = Field(
        default=2,
        gt=0,
        description='Number of experts to activate per input'
    )
    hidden_dim: int = Field(
        default=1024,
        gt=0,
        description='Hidden dimension for expert networks'
    )
    load_balancing_coef: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description='Load balancing loss coefficient'
    )
    

    @model_validator(mode='after')
    def validate_top_k_vs_experts(self) -> 'MoEConfig':

        '''Ensure top_k doesn't exceed num_experts.'''
        
        if self.top_k > self.num_experts:
            raise ValueError(
                f'top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})'
            )
        return self


class ModelConfig(BaseModel):

    '''Model architecture configuration.'''
    
    base_model_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description='HuggingFace base model name'
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description='LoRA configuration'
    )
    moe: MoEConfig = Field(
        default_factory=MoEConfig,
        description='Mixture of Experts configuration'
    )
    eval_sample_size: int = Field(
        default=500,
        gt=0,
        le=2125,
        description='Number of codes to sample for evaluation'
    )
    eval_every_n_epochs: int = Field(
        default=1,
        gt=0,
        description='Run evaluation every N epochs'
    )


# -------------------------------------------------------------------------------------------------
# Loss Configuration
# -------------------------------------------------------------------------------------------------

class LossConfig(BaseModel):

    '''Loss function configuration.'''
    
    temperature: float = Field(
        default=0.07,
        gt=0,
        le=1,
        description='Temperature for contrastive loss'
    )
    curvature: float = Field(
        default=1.0,
        gt=0,
        description='Curvature for hyperbolic space'
    )


# -------------------------------------------------------------------------------------------------
# Training Configuration
# -------------------------------------------------------------------------------------------------

class TrainerConfig(BaseModel):

    '''PyTorch Lightning Trainer configuration.'''
    
    max_epochs: int = Field(
        default=10,
        gt=0,
        le=1000,
        description='Maximum number of training epochs'
    )
    accelerator: str = Field(
        default='auto',
        description='Training accelerator (auto, gpu, cpu, mps)'
    )
    devices: int = Field(
        default=1,
        gt=0,
        description='Number of devices to use'
    )
    precision: str = Field(
        default='16-mixed',
        description='Training precision (32, 16-mixed, bf16-mixed)'
    )
    gradient_clip_val: float = Field(
        default=1.0,
        gt=0,
        description='Gradient clipping value'
    )
    accumulate_grad_batches: int = Field(
        default=1,
        gt=0,
        description='Number of batches for gradient accumulation'
    )
    log_every_n_steps: int = Field(
        default=10,
        gt=0,
        description='Log metrics every N steps'
    )
    val_check_interval: float = Field(
        default=1.0,
        gt=0,
        description='Run validation every N epochs (or fraction)'
    )
    

    @field_validator('accelerator')
    @classmethod
    def validate_accelerator(cls, v: str) -> str:

        '''Validate accelerator choice.'''
        
        valid = ['auto', 'gpu', 'cpu', 'mps', 'cuda']
        if v not in valid:
            raise ValueError(f'accelerator must be one of {valid}')
        return v
    

    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v: str) -> str:

        '''Validate precision choice.'''
        
        valid = ['32', '16', '16-mixed', 'bf16', 'bf16-mixed']
        if v not in valid:
            raise ValueError(f'precision must be one of {valid}')
        return v


class TrainingConfig(BaseModel):

    '''Optimizer and training configuration.'''
    
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        lt=1,
        description='Learning rate for optimizer'
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        lt=1,
        description='Weight decay (L2 regularization)'
    )
    warmup_steps: int = Field(
        default=500,
        ge=0,
        description='Number of warmup steps'
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description='PyTorch Lightning Trainer config'
    )


# -------------------------------------------------------------------------------------------------
# Curriculum Configuration
# -------------------------------------------------------------------------------------------------

class CurriculumConfig(BaseModel):
    '''Curriculum learning configuration.'''
    
    name: str = Field(
        default='default',
        description='Curriculum name'
    )
    positive_levels: List[int] = Field(
        default=[4, 5, 6],
        description='NAICS hierarchy levels for positive pairs'
    )
    positive_distance_min: Optional[float] = Field(
        default=None,
        ge=0,
        description='Minimum distance for positive pairs'
    )
    positive_distance_max: float = Field(
        default=2.0,
        gt=0,
        description='Maximum distance for positive pairs'
    )
    max_positives: int = Field(
        default=10,
        gt=0,
        description='Maximum number of positives per anchor'
    )
    difficulty_buckets: List[int] = Field(
        default=[1, 2, 3],
        description='Hardness levels to sample negatives from'
    )
    bucket_percentages: Dict[int, float] = Field(
        default={1: 0.5, 2: 0.3, 3: 0.2},
        description='Percentage of negatives from each bucket'
    )
    k_negatives: int = Field(
        default=8,
        gt=0,
        description='Number of negative samples per positive'
    )
    

    @field_validator('positive_levels')
    @classmethod
    def validate_levels(cls, v: List[int]) -> List[int]:
        '''Validate NAICS levels are in valid range.'''
        if not all(2 <= level <= 6 for level in v):
            raise ValueError('positive_levels must be between 2 and 6')
        return v
    

    @field_validator('difficulty_buckets')
    @classmethod
    def validate_buckets(cls, v: List[int]) -> List[int]:
        '''Validate hardness buckets are in valid range.'''
        if not all(1 <= bucket <= 8 for bucket in v):
            raise ValueError('difficulty_buckets must be between 1 and 8')
        return v
    

    @model_validator(mode='after')
    def validate_bucket_percentages(self) -> 'CurriculumConfig':
        '''Ensure bucket percentages sum to 1.0 and match difficulty_buckets.'''
        # Check all buckets have percentages
        for bucket in self.difficulty_buckets:
            if bucket not in self.bucket_percentages:
                raise ValueError(
                    f'Bucket {bucket} in difficulty_buckets but not in bucket_percentages'
                )
        
        # Check percentages sum to 1.0 (allow small floating point errors)
        total = sum(self.bucket_percentages.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f'bucket_percentages must sum to 1.0, got {total:.4f}'
            )
        
        return self


# -------------------------------------------------------------------------------------------------
# Main Configuration
# -------------------------------------------------------------------------------------------------

class Config(BaseModel):

    '''Main configuration for NAICS training.'''
    
    experiment_name: str = Field(
        default='default',
        description='Experiment name for logging and checkpoints'
    )
    seed: int = Field(
        default=42,
        ge=0,
        description='Random seed for reproducibility'
    )
    paths: PathsConfig = Field(
        default_factory=PathsConfig,
        description='File system paths'
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description='Data loading configuration'
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description='Model architecture configuration'
    )
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description='Loss function configuration'
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description='Training configuration'
    )
    curriculum: CurriculumConfig = Field(
        default_factory=CurriculumConfig,
        description='Curriculum learning configuration'
    )
    

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        curriculum_name: Optional[str] = None
    ) -> 'Config':
        
        '''Load configuration from YAML file.'''

        # Load base config
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f'Config file not found: {yaml_path}')
        
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        # Load curriculum if specified
        if curriculum_name:
            curriculum_path = Path('conf/curriculum') / f'{curriculum_name}.yaml'
            if not curriculum_path.exists():
                raise FileNotFoundError(
                    f'Curriculum not found: {curriculum_path}\n'
                    f'Available curricula: {list(Path("conf/curriculum").glob("*.yaml"))}'
                )
            
            with open(curriculum_path, 'r') as f:
                curriculum_data = yaml.safe_load(f)
            
            if curriculum_data:
                data['curriculum'] = curriculum_data
        
        # Set experiment name from curriculum if not specified
        if 'experiment_name' not in data and 'curriculum' in data:
            data['experiment_name'] = data['curriculum'].get('name', 'default')
        
        # Pydantic automatically validates!
        logger.info(f'Loading config from {yaml_path}')
        if curriculum_name:
            logger.info(f'Using curriculum: {curriculum_name}')
        
        return cls(**data)
    

    def override(self, overrides: Dict[str, Any]) -> 'Config':
        
        '''Apply overrides using dot notation.'''
    
        # Convert to dict
        data = self.model_dump()
        
        for key, value in overrides.items():
            parts = key.split('.')
            current = data
            
            # Navigate to nested dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value
            current[parts[-1]] = value
        
        # Re-validate with new values (Pydantic validates automatically)
        logger.info(f'Applied {len(overrides)} override(s)')
        return Config(**data)
    

    def to_dict(self) -> Dict[str, Any]:
    
        '''Convert config to dictionary.'''
    
        return self.model_dump()
    

    def to_yaml(self, path: str) -> None:

        '''Save config to YAML file.'''
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f'Saved config to {path}')
    

    class ConfigDict:
    
        '''Pydantic v2 configuration.'''
    
        validate_assignment = True  # Validate on attribute assignment
        extra = 'forbid'  # Raise error on unknown fields
        str_strip_whitespace = True  # Strip whitespace from strings


# -------------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------------

def parse_override_value(value: str) -> Any:

    '''Parse override value from string to appropriate type.'''

    try:
        # Handle common cases
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try numeric conversion
        if '.' in value or 'e' in value.lower():
            return float(value)
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try Python literals (lists, dicts, etc.)
        import ast
        return ast.literal_eval(value)
    
    except (ValueError, SyntaxError):
        # Keep as string
        return value


def list_available_curricula() -> List[str]:

    '''List all available curriculum configs.'''
    
    curriculum_dir = Path('conf/curriculum')
    if not curriculum_dir.exists():
        return []
    
    curricula = [
        f.stem for f in curriculum_dir.glob('*.yaml')
        if not f.name.startswith('_')
    ]
    return sorted(curricula)