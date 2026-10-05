'''Durable epoch health values and monitor MRR beside the run's checkpoints (spec 4.4, P20).'''

import json
import math
import operator
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

EPOCH_SUMMARY = 'epoch_summary.jsonl'
_HEALTH_KEY = re.compile(
    r'(?:loss/(?:task|code_code|radial|total|load_balancing)|logit_scale/(?:task|code_code)|'
    r'radius/(?:mean|sd)/level_[2-6])\Z'
)

def _epoch(value: Any) -> int:
    '''A non-negative integer epoch; booleans are not epochs.'''

    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'an epoch must be a non-negative integer, not {value!r}') from exc
    if value < 0:
        raise ValueError(f'an epoch must be non-negative, not {value}')
    return value

def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def _validate_row(row: Any) -> Dict[str, Any]:
    '''Check an epoch, its optional MRR and P20's finite health values before a file changes.'''

    if not isinstance(row, dict) or 'epoch' not in row or 'mrr' not in row:
        raise ValueError('an epoch summary row needs epoch and mrr')
    _epoch(row['epoch'])
    mrr = row['mrr']
    if mrr is not None and not (_finite_number(mrr) and 0 <= mrr <= 1):
        raise ValueError(f'an epoch summary mrr must be finite and in [0, 1], or None: {mrr!r}')
    for key, value in row.items():
        if key in ('epoch', 'mrr'):
            continue
        if not _HEALTH_KEY.fullmatch(key) or not _finite_number(value):
            raise ValueError(f'an epoch summary needs a finite P20 health value: {key}={value!r}')
    return row

def _row_lines(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    '''Checked rows and their exact nonblank lines, in increasing epoch order.'''

    lines = []
    previous = -1
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _validate_row(json.loads(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}:{number}: invalid epoch summary: {exc}') from exc
        if row['epoch'] <= previous:
            raise ValueError(
                f'{path}:{number}: epoch {row["epoch"]} would repeat or reorder epochs'
            )
        previous = row['epoch']
        lines.append((line, row))
    return lines

def read_epoch_summary(path: Union[str, Path]) -> List[Dict[str, Any]]:
    '''
    Read the durable summary, in epoch order, skipping blank lines.

    Rows hold ``epoch``, ``mrr`` (None only without a monitor), and the flat P20 health keys as
    Python numbers. No values are rounded or recovered from Lightning's callback metrics.

    Raises:
        ValueError: If a row is invalid, nonfinite, repeats an epoch or is out of order.
        FileNotFoundError: If the summary does not exist.
    '''

    return [row for _, row in _row_lines(Path(path))]

class EpochSummary:
    '''
    Append each completed epoch's MRR and float64 health values to ``epoch_summary.jsonl``.

    The model starts it at fit start and appends after the refresh, monitor and health logs,
    before ModelCheckpoint saves. Exact resume keeps the lines through the restored epoch and
    drops a later interrupted segment's lines, as the monitor file does (spec 4.4).
    '''

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def start(self, *, resumed_epoch: Optional[int]) -> None:
        '''
        Refuse an existing fresh file or atomically prune a resumed file to its checkpoint epoch.

        Raises:
            ValueError: If a fresh file exists, the resumed file is absent, the restored epoch is
                invalid, or any existing summary row is invalid.
        '''

        path = self.path
        if resumed_epoch is None:
            if path.exists():
                raise ValueError(
                    f'{path} already holds an epoch summary: a fresh run needs a new file'
                )
            return
        resumed_epoch = _epoch(resumed_epoch)
        if not path.exists():
            raise ValueError(
                f'an exact resume continues its epoch summary, but {path} does not exist: '
                'bring it with last.ckpt'
            )
        lines = _row_lines(path)
        kept = [line for line, row in lines if row['epoch'] <= resumed_epoch]
        staged = path.with_name(path.name + '.tmp')
        staged.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
        os.replace(staged, path)

    def append(self, *, epoch: int, mrr: Optional[float], health: Mapping[str, float]) -> None:
        '''
        Append one completed epoch without rounding its health values.

        Raises:
            ValueError: If an epoch repeats or reorders the existing file, or a row is invalid.
        '''

        row = _validate_row({**health, 'epoch': _epoch(epoch), 'mrr': mrr})
        lines = _row_lines(self.path) if self.path.exists() else []
        if lines and lines[-1][1]['epoch'] >= row['epoch']:
            raise ValueError(f'epoch {epoch} would repeat or reorder the epoch summary')
        rendered = json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(rendered)
