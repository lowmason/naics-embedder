'''
The suite's guard on the repository's selection log: no test writes the real log.

The shipped config logs every read to ``./logs/selection_log.jsonl``, relative to the working
directory, which is the repository root when the suite runs. A session fixture in
``tests/conftest.py`` records the log's size (or absence) when the session starts and fails the
session if it changed by the end. These tests check its helpers on logs under ``tmp_path``; none
creates the repository's log.
'''

import re
from pathlib import Path

import pytest
import yaml

from naics_embedder.utils.config import OutcomePanelConfig, RegressorPanelConfig
from tests import conftest as suite

pytestmark = pytest.mark.unit

REPOSITORY = Path(__file__).resolve().parents[2]

def test_the_guard_watches_the_log_the_shipped_configs_name():
    shipped = {
        yaml.safe_load((REPOSITORY / 'conf' / 'data' / name).read_text())['selection_log']
        for name in ('outcome_panel.yaml', 'regressor_panel.yaml')
    }
    defaults = {OutcomePanelConfig().selection_log, RegressorPanelConfig().selection_log}

    assert shipped | defaults == {'./logs/selection_log.jsonl'}
    assert suite.REPOSITORY_SELECTION_LOG == REPOSITORY / 'logs' / 'selection_log.jsonl'

def test_every_test_runs_under_the_guard(request):
    assert 'guard_the_repository_selection_log' in request.fixturenames
    assert request.getfixturevalue('guard_the_repository_selection_log') == (
        suite.REPOSITORY_SELECTION_LOG
    )

def _log(tmp_path: Path, content) -> Path:
    path = tmp_path / 'logs' / 'selection_log.jsonl'
    if content is not None:
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding='utf-8')
    return path

@pytest.mark.parametrize(
    ('content', 'size'),
    [(None, None), ('', 0), ('{"event": "read"}\n', 18)],
    ids=['absent', 'empty', 'one-record'],
)
def test_the_guard_measures_a_log_by_its_size_or_its_absence(tmp_path, content, size):
    assert suite.selection_log_size(_log(tmp_path, content)) == size

@pytest.mark.parametrize(
    'content', [None, '', '{"event": "read"}\n'], ids=['absent', 'empty', 'one-record']
)
def test_the_guard_passes_a_log_the_session_left_alone(tmp_path, content):
    path = _log(tmp_path, content)
    size = suite.selection_log_size(path)

    suite.refuse_a_changed_selection_log(path, size)

@pytest.mark.parametrize(
    ('content', 'change', 'sizes'),
    [
        (None, 'created', 'from absent to 18 bytes'),
        ('{"event": "read"}\n', 'appended', 'from 18 bytes to 36 bytes'),
        ('{"event": "read"}\n', 'deleted', 'from 18 bytes to absent'),
    ],
    ids=['created', 'appended', 'deleted'],
)
def test_the_guard_fails_a_log_the_session_wrote(tmp_path, content, change, sizes):
    path = _log(tmp_path, content)
    size = suite.selection_log_size(path)
    if change == 'deleted':
        path.unlink()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write('{"event": "read"}\n')

    refusal = re.escape(f'the suite wrote {path} (its size went {sizes})')
    with pytest.raises(pytest.fail.Exception, match=refusal):
        suite.refuse_a_changed_selection_log(path, size)
