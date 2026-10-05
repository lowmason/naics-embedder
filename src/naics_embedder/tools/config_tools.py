'''
Configuration display tools.

Display the training configuration a run would use: the file, validated over the defaults.
'''

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from naics_embedder.utils.config import Config
from naics_embedder.utils.training import FULL_PRECISION, run_settings

console = Console()

def show_current_config(config_path: Union[str, Path] = './conf/config.yaml') -> bool:
    '''
    Display the training configuration a run would use.

    The file is validated as ``Config`` over the defaults, so every value shown is one a run reads,
    and a file that sets a key the configuration no longer has is refused, not shown (spec 4.5).
    The panel lists the run's name, seed and inputs, then the settings every run records
    (``run_settings``, P21), with the configured accelerator and the precision rule (P31).

    Args:
        config_path: Path to main configuration file

    Returns:
        True when the configuration was displayed. False when an error was printed instead: the
        file is missing, or ``Config`` refuses it. ``tools config`` then exits 1.
    '''

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        console.print(
            f'[bold red]Error:[/bold red] Config file not found: {escape(str(config_path))}'
        )
        return False

    try:
        cfg = Config.from_yaml(config_path)
    except ValidationError as error:
        console.print('[bold red]Error:[/bold red] not a valid configuration:')
        console.print(config_path, markup=False, highlight=False)
        console.print(str(error), markup=False, highlight=False)
        return False

    trainer = cfg.training.trainer
    # A run trains at the configured precision on CUDA, and at full precision elsewhere (P31)
    precision = f'{trainer.precision} on CUDA, {FULL_PRECISION} elsewhere'
    settings = run_settings(cfg, accelerator=trainer.accelerator, precision=precision)

    current_config = [
        f'\n[blue]Configuration ({escape(str(config_path))}):[/blue]\n',
        '[cyan]Run:[/cyan]',
        f'  • [bold]experiment_name:[/bold] {escape(cfg.experiment_name)}',
        f'  • [bold]seed:[/bold] {cfg.seed}',
        f'  • [bold]supervision.manifest_path:[/bold] {escape(str(cfg.supervision.manifest_path))}',
        '  • [bold]descriptions_parquet:[/bold] '
        f'{escape(cfg.data_loader.streaming.descriptions_parquet)}\n',
        '[cyan]Run settings:[/cyan]',
        *[f'  • [bold]{name}:[/bold] {escape(str(value))}' for name, value in settings.items()],
    ]

    console.print(
        Panel(
            '\n'.join(current_config),
            title='[yellow]Current Training Configuration[/yellow]',
            border_style='yellow',
            expand=True,
        )
    )
    return True

def load_config(config_path: str = './conf/config.yaml'):
    '''Load main configuration file.'''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
