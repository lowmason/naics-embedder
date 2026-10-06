# Remote Workflow API

The [operator guide](../remote_workflow.md) describes prerequisites, exact resume, coherent
transport and the pending post-merge real-instance qualification. Five Typer adapters delegate
to `RemoteWorkflow`; transport and domain guards remain below that boundary.

## CLI Adapters

::: naics_embedder.cli.commands.remote

## Workflow Controller

`sync(once=True)` delegates to public `sync_once`, which owns the state lock. `sync(once=False)`
validates the complete session-owned transport config and calls `ensure_loop` while holding
that lock. Both require a ready session. No controller operation silently replaces a missing
persisted config with defaults. `status` observes evidence without repair. `FinishResult.safe`
is the only termination-readiness authority; abandonment remains unsafe.

::: naics_embedder.remote.workflow.RemoteWorkflow

::: naics_embedder.remote.workflow.FinishResult

## Results and Persistent Records

::: naics_embedder.remote.launch.LaunchResult

::: naics_embedder.remote.sync.SyncResult

::: naics_embedder.remote.session.RemoteState

::: naics_embedder.remote.session.RunRecord
