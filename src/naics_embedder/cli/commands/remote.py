'''Operator adapters for the verified single-instance remote training workflow.'''

import json
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import typer
from rich.console import Console
from rich.markup import escape
from typing_extensions import Annotated

from naics_embedder.remote.config import load_remote_config
from naics_embedder.remote.transport import SshTransport
from naics_embedder.remote.workflow import RemoteWorkflow
from naics_embedder.utils.config import RemoteConfig

# -------------------------------------------------------------------------------------------------
# Lazy controller construction and presentation
# -------------------------------------------------------------------------------------------------

app = typer.Typer(help='Prepare, train, pull and verify one remote instance.', no_args_is_help=True)
console = Console()

def _transport(host: str, cfg: RemoteConfig) -> SshTransport:
    return SshTransport(host, cfg.repo_dir, cfg.rsync_path)

def _workflow(path: Path) -> RemoteWorkflow:
    root = Path.cwd()
    return RemoteWorkflow(
        root, load_remote_config(path), _transport, lambda: datetime.now(timezone.utc)
    )

@contextmanager
def _operation(ctx: typer.Context) -> Iterator[RemoteWorkflow]:
    try:
        yield _workflow(ctx.obj)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        console.print('[red]Remote refused:[/red] ' + escape(str(error)))
        raise typer.Exit(1) from error

@app.callback()
def remote(
    ctx: typer.Context,
    remote_config: Annotated[Path,
                             typer.Option(
                                 '--remote-config',
                                 help='Transport YAML, including optional GNU rsync path.'
                             )] = Path('conf/remote.yaml'),
) -> None:
    '''Select transport configuration without reading it during help.'''
    ctx.obj = remote_config

# -------------------------------------------------------------------------------------------------
# Five command adapters
# -------------------------------------------------------------------------------------------------

@app.command()
def up(
    ctx: typer.Context,
    host: Annotated[str, typer.Option('--host', help='SSH USER@HOST for the instance.')] = '',
    force: Annotated[bool,
                     typer.Option(
                         '--force', help='Record loss risk or overwrite named instance edits.'
                     )] = False,
    config: Annotated[
        str, typer.Option('--config', help='Repo-relative training YAML.')] = 'conf/config.yaml',
    overrides: Annotated[Optional[list[str]],
                         typer.Argument(help='Trailing key=value training overrides.')] = None,
) -> None:
    '''Push code, bootstrap and verify canonical inputs before becoming ready.'''
    with _operation(ctx) as workflow:
        state = workflow.up(host, config, overrides or [], force)
    console.print('Ready session ' + escape(state.session_id) + ' on ' + escape(state.host))

@app.command()
def train(
    ctx: typer.Context,
    resume: Annotated[
        bool, typer.Option('--resume', help='Exact continuation from last.ckpt only.')] = False,
    config: Annotated[
        str, typer.Option('--config', help='Repo-relative training YAML.')] = 'conf/config.yaml',
    overrides: Annotated[Optional[list[str]],
                         typer.Argument(help='Trailing key=value training overrides.')] = None,
) -> None:
    '''Launch a fresh or exact run; finished runs skip successfully.'''
    with _operation(ctx) as workflow:
        result = workflow.train(resume, config, overrides or [])
    if result.skipped:
        console.print('Skipped: ' + escape(result.reason or 'finished run'))
    else:
        console.print('Launched segment ' + escape(result.segment_id or 'unknown'))

@app.command()
def sync(
    ctx: typer.Context,
    once: Annotated[bool,
                    typer.Option(
                        '--once',
                        help='Perform one verified pull instead of ensuring the detached loop.'
                    )] = False,
) -> None:
    '''Pull coherent artifacts once or ensure the owned Mac sync loop.'''
    with _operation(ctx) as workflow:
        result = workflow.sync(once)
    if result is None:
        console.print('Sync loop ensured')
    else:
        console.print(f'Pulled: {result.pulled}; pending: {result.pending}')

@app.command()
def finish(
    ctx: typer.Context,
    stop_training: Annotated[bool,
                             typer.Option(
                                 '--stop-training',
                                 help='Interrupt the owned training segment before final pull.'
                             )] = False,
    pull_edits: Annotated[bool,
                          typer.Option(
                              '--pull-edits',
                              help='Rescue instance edits outside the Mac working tree.'
                          )] = False,
    abandon: Annotated[bool,
                       typer.Option(
                           '--abandon',
                           help='Close the session with possible loss after the last good sync.'
                       )] = False,
) -> None:
    '''Quiesce syncing and verify termination readiness, or explicitly abandon.'''
    with _operation(ctx) as workflow:
        result = workflow.finish(stop_training, pull_edits, abandon)
    if result.safe:
        console.print('Safe to terminate')
        if result.latest_checkpoint is None:
            console.print(
                'no checkpoint; local SHA-256: no checkpoint; remote SHA-256: no checkpoint'
            )
        else:
            console.print('Checkpoint: ' + escape(result.latest_checkpoint))
            console.print('Local SHA-256: ' + escape(result.local_sha256 or 'no checkpoint'))
            console.print('Remote SHA-256: ' + escape(result.remote_sha256 or 'no checkpoint'))
    elif result.abandoned:
        console.print('Session abandoned. Anything after the last successful sync may be lost.')
    else:
        console.print('Termination readiness was not verified.')
        raise typer.Exit(1)

@app.command()
def status(ctx: typer.Context) -> None:
    '''Read session, process, GPU, pending-transfer and sync-loop observations.'''
    with _operation(ctx) as workflow:
        result = workflow.status()
    console.print(json.dumps(result, indent=2), markup=False, highlight=False)
    if result.get('errors'):
        raise typer.Exit(1)
