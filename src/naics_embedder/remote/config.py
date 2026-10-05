'''Strict remote configuration loading and checkout-relative path guards.'''

from pathlib import Path

import yaml

from naics_embedder.utils.config import Config, RemoteConfig
from naics_embedder.utils.training import parse_config_overrides

# -------------------------------------------------------------------------------------------------
# Configuration and protected paths
# -------------------------------------------------------------------------------------------------

def relative_path(root: Path, value: str) -> str:
    '''Return a normalized repo-relative path, refusing traversal and escaping symlinks.'''
    path = Path(value)
    if not value or path.is_absolute() or '\\' in value or any(
        ord(char) < 32 or 127 <= ord(char) <= 159 for char in value
    ):
        raise ValueError(f'expected a repo-relative path: {value!r}')
    if '..' in path.parts:
        raise ValueError(f'expected a repo-relative path: {value!r}')
    try:
        (root / path).resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f'expected a repo-relative path: {value!r}') from error
    return path.as_posix()

def load_remote_config(path: Path) -> RemoteConfig:
    '''Load a required YAML file; unknown settings fail validation.'''
    with path.open() as stream:
        data = yaml.safe_load(stream)
    return RemoteConfig.model_validate({} if data is None else data)

def effective_config(root: Path, path: str, overrides: list[str]) -> Config:
    '''Resolve the pushed training config and apply existing last-wins override parsing.'''
    config_path = root / relative_path(root, path)
    cfg = Config.from_yaml(str(config_path))
    values, invalid = parse_config_overrides(overrides)
    if invalid:
        raise ValueError(f'invalid config override(s): {invalid!r}; expected key=value')
    cfg = cfg.override(values)
    experiment = cfg.experiment_name
    if not experiment.strip() or experiment in ('.', '..') or any(
        char in '/\\' or ord(char) < 32 or 127 <= ord(char) <= 159 for char in experiment
    ):
        raise ValueError('experiment_name must be a single nonempty path component')
    return cfg
