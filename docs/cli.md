# Breaking Up Large CLI Files with Typer

## Overview

Yes, you can definitely break up large Typer CLI files! This guide shows how to refactor your 1100+ line `cli.py` into a modular, maintainable structure.

## New Directory Structure

```
naics_embedder/
├── cli.py                    # Main entry point (~80 lines)
└── cli/
    ├── __init__.py
    └── commands/
        ├── __init__.py
        ├── data.py           # Data generation commands (~120 lines)
        ├── tools.py          # Tools commands (~200 lines)
        └── training.py       # Training commands (~350 lines)
```

## Key Benefits

### 1. **Separation of Concerns**
- Each module handles one logical group of commands
- Easier to find and modify specific functionality
- Reduced cognitive load when working on features

### 2. **Maintainability**
- Changes to data commands don't affect training code
- Each file is ~100-350 lines instead of 1100+
- Easier code reviews and collaboration

### 3. **Testability**
- Test each command group independently
- Mock dependencies at module level
- Easier to write focused unit tests

### 4. **Scalability**
- Add new command groups without touching existing code
- Can further split modules if they grow too large
- Clear pattern for adding new commands

## How It Works

### 1. Main Entry Point (`cli.py`)

The main file stays clean and simple:

```python
import typer
from naics_embedder.cli.commands import data, tools, training

app = typer.Typer(help="NAICS Embedder")

# Add sub-apps (command groups)
app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

# Add individual commands directly
app.command('train')(training.train)
app.command('train-seq')(training.train_sequential)

if __name__ == '__main__':
    app()
```

### 2. Command Modules

Each module creates its own Typer app:

```python
# commands/data.py
import typer

app = typer.Typer(help='Manage and generate project datasets.')

@app.command('preprocess')
def preprocess():
    """Download and preprocess raw data."""
    # Implementation here

@app.command('distances')
def distances():
    """Compute pairwise distances."""
    # Implementation here
```

### 3. Two Patterns for Commands

**Pattern 1: Sub-apps (command groups)**
```python
# In main cli.py
app.add_typer(data.app, name='data')

# Usage:
$ python -m naics_embedder.cli data preprocess
$ python -m naics_embedder.cli data distances
$ python -m naics_embedder.cli data all
```

**Pattern 2: Direct registration**
```python
# In main cli.py
app.command('train')(training.train)

# Usage:
$ python -m naics_embedder.cli train --curriculum 01_text
```

## Migration Guide

### Step 1: Create Directory Structure

```bash
mkdir -p naics_embedder/cli/commands
touch naics_embedder/cli/__init__.py
touch naics_embedder/cli/commands/__init__.py
```

### Step 2: Extract Command Groups

1. **Data commands** → `commands/data.py`
   - Move all `@data_app.command()` functions
   - Keep related imports

2. **Tools commands** → `commands/tools.py`
   - Move all `@tools_app.command()` functions

3. **Training commands** → `commands/training.py`
   - Move `train()` and `train_sequential()` functions
   - These can stay as standalone functions

### Step 3: Update Main CLI

Replace command definitions with imports and registrations:

```python
from naics_embedder.cli.commands import data, tools, training

app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')
app.command('train')(training.train)
app.command('train-seq')(training.train_sequential)
```

### Step 4: Update Package Init

If you have `naics_embedder/__init__.py`, you might want to export the CLI:

```python
from .cli import app as cli_app

__all__ = ['cli_app']
```

## Usage Examples

All commands work exactly the same as before:

```bash
# Data commands
python -m naics_embedder.cli data preprocess
python -m naics_embedder.cli data relations
python -m naics_embedder.cli data distances
python -m naics_embedder.cli data all

# Training commands
python -m naics_embedder.cli train --curriculum 01_text
python -m naics_embedder.cli train-seq --curricula 01_text 02_text

# Tools commands
python -m naics_embedder.cli tools config
python -m naics_embedder.cli tools gpu --auto
python -m naics_embedder.cli tools visualize --stage 02_text
```

## Advanced: Further Splitting

If individual modules get too large, you can split them further:

```
commands/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── relations.py
│   └── distances.py
├── tools/
│   ├── __init__.py
│   ├── config.py
│   ├── gpu.py
│   └── metrics.py
└── training/
    ├── __init__.py
    ├── single.py
    └── sequential.py
```

Then in `commands/data/__init__.py`:

```python
import typer
from . import preprocess, relations, distances

app = typer.Typer(help='Data management commands')

app.command('preprocess')(preprocess.run)
app.command('relations')(relations.run)
app.command('distances')(distances.run)
```

## Best Practices

### 1. Keep Related Code Together
Group commands by domain (data, training, tools), not by technical details.

### 2. Minimal Main File
Your main `cli.py` should only:
- Import command modules
- Register sub-apps/commands
- Define the root Typer app

### 3. Each Module is Self-Contained
Each command module should import its own dependencies.

### 4. Shared Utilities
Put shared code in utility modules:
```python
# utils/console.py
def configure_logging(...): ...

# utils/config.py
def load_config(...): ...
```

### 5. Consistent Naming
- Module names: lowercase, descriptive (`data.py`, `training.py`)
- Command names: match your CLI structure
- Function names: descriptive verbs (`preprocess()`, `visualize()`)

## Common Patterns

### Pattern: Callback for Command Group

Add setup logic for all commands in a group:

```python
app = typer.Typer()

@app.callback()
def callback(verbose: bool = False):
    """Configure logging for all data commands."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
```

### Pattern: Shared Options

Create reusable parameter definitions:

```python
from typing_extensions import Annotated

ConfigOption = Annotated[
    str,
    typer.Option('--config', help='Path to config YAML')
]

@app.command()
def train(config_file: ConfigOption = 'conf/config.yaml'):
    ...

@app.command()
def evaluate(config_file: ConfigOption = 'conf/config.yaml'):
    ...
```

### Pattern: Dynamic Command Registration

Load commands dynamically (useful for plugins):

```python
# In main cli.py
import importlib
import pkgutil
from . import commands

# Discover and register all command modules
for _, name, _ in pkgutil.iter_modules(commands.__path__):
    module = importlib.import_module(f'.commands.{name}', package=__package__)
    if hasattr(module, 'app'):
        app.add_typer(module.app, name=name)
```

## Testing

With modular structure, testing becomes easier:

```python
# tests/test_data_commands.py
from typer.testing import CliRunner
from naics_embedder.cli.commands.data import app

runner = CliRunner()

def test_preprocess():
    result = runner.invoke(app, ['preprocess'])
    assert result.exit_code == 0
    assert 'Preprocessing complete' in result.output
```

## Conclusion

Breaking up your CLI file using Typer's sub-apps is straightforward and provides significant benefits for maintainability. The key is:

1. **Create logical groupings** of related commands
2. **Each module owns a Typer app** for its commands
3. **Main file orchestrates** by importing and registering
4. **Commands work identically** - no breaking changes for users

Your 1100-line file becomes:
- Main file: ~80 lines
- Data commands: ~120 lines
- Tools commands: ~200 lines  
- Training commands: ~350 lines

Much more maintainable! 🎉
