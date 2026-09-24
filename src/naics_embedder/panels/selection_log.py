'''
The selection log (Req 4; Verification "Selection hygiene").

An append-only JSON-lines file with one record per event: every read of a panel's validation or
test split, and every opening of a test split. It is the evidence that selections read validation
splits only and that each test split was opened once, for the final configuration. A second
opening needs a stated reason and is recorded as ``reopen``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Log
# -------------------------------------------------------------------------------------------------

class SelectionEvent(str, Enum):
    '''What a selection-log record reports.'''

    READ = 'read'
    OPEN = 'open'
    REOPEN = 'reopen'

class SelectionLog:
    '''Append-only JSON-lines log of panel reads and test-split openings.'''

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(
        self,
        event: SelectionEvent,
        *,
        panel: str,
        split: str,
        purpose: str,
        fingerprint: str,
        n_queries: int,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        '''
        Append one record and return it.

        Raises:
            ValueError: If ``purpose`` is blank: every read and opening states why it happens.
        '''

        if not purpose.strip():
            raise ValueError('every selection-log record needs a purpose')
        record = {
            'time': datetime.now(timezone.utc).isoformat(),
            'event': SelectionEvent(event).value,
            'panel': panel,
            'split': split,
            'purpose': purpose,
            'fingerprint': fingerprint,
            'n_queries': n_queries,
            'detail': dict(detail or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
        logger.info(f'Selection log: {record["event"]} {panel}/{split} ({purpose}) -> {self.path}')
        return record

    def records(self) -> List[Dict[str, Any]]:
        '''Every record, oldest first; a log that does not exist yet has none.'''

        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding='utf-8').splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def openings(self, panel: str, fingerprint: str) -> List[Dict[str, Any]]:
        '''The recorded openings (``open`` and ``reopen``) of one panel's test split.'''

        opening = {SelectionEvent.OPEN.value, SelectionEvent.REOPEN.value}
        return [
            record for record in self.records() if record['event'] in opening
            and record['panel'] == panel and record['fingerprint'] == fingerprint
        ]
