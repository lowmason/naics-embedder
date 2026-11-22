'''
The project's Claude Code hooks in ``.claude/hooks/``, run the way Claude Code runs them.

``evidence_guard.py`` (PreToolUse) denies ``tools growth-margins`` and ``ruff format``, and asks
before a tools command that reads a real panel. ``format_check.py`` (PostToolUse) runs
``scripts/format_code.sh --check`` on an edited Python file under ``src/`` or ``tests/``. Each test
pipes a hook payload into the script on stdin, then reads its stdout, stderr and exit code.
'''

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / '.claude' / 'hooks'
GUARD = HOOKS / 'evidence_guard.py'
FORMAT_CHECK = HOOKS / 'format_check.py'

# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def run_hook(script, payload, env=None):
    '''Pipe a payload (a dict, or raw text) into a hook script and return the finished process.'''
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(script)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

def guard_decision(command, tool_name='Bash'):
    '''Return the guard's (decision, reason) for a command, or (None, '') when it stays silent.'''
    result = run_hook(GUARD, {'tool_name': tool_name, 'tool_input': {'command': command}})
    assert result.returncode == 0, result.stderr
    if not result.stdout.strip():
        return None, ''
    output = json.loads(result.stdout)['hookSpecificOutput']
    assert output['hookEventName'] == 'PreToolUse'
    return output['permissionDecision'], output['permissionDecisionReason']

# -------------------------------------------------------------------------------------------------
# Evidence guard
# -------------------------------------------------------------------------------------------------

GROWTH_MARGINS_RUNS = [
    'uv run naics-embedder tools growth-margins --codebook c.parquet --output m.json',
    'naics-embedder tools growth-margins',
    '.venv/bin/naics-embedder tools growth-margins',
    'cd /tmp && uv run --locked naics-embedder tools growth-margins',
    'FOO=1 uv run --no-sync naics-embedder tools growth-margins',
    'uv run python -m naics_embedder.cli tools growth-margins',
    "bash -c 'uv run naics-embedder tools growth-margins'",
    'echo start\nuv run naics-embedder tools growth-margins',
    'uv run naics-embedder tools growth-margins 2>&1 | tee log.txt',
    "uv run naics-embedder tools growth-margins # don't rerun",
    'echo hi # a note\nuv run naics-embedder tools growth-margins',
    "cat <<'EOF' > notes.txt\ndon't\nEOF\nuv run naics-embedder tools growth-margins",
    "bash -o pipefail -c 'uv run naics-embedder tools growth-margins'",
    '2>/dev/null uv run naics-embedder tools growth-margins',
    'naics-embedder 2>/dev/null tools growth-margins',
    'uv --directory /home/u/naics-embedder run naics-embedder tools growth-margins',
    'uv run --python python3.12 naics-embedder tools growth-margins',
]

RUFF_FORMAT_RUNS = [
    'uv run ruff format src/',
    'ruff format',
    'uvx ruff format --check src',
    'uv run python -m ruff format x.py',
    'uv run ruff --config pyproject.toml format src/',
    "grep -n '#' CLAUDE.md && uv run ruff format",
    'echo a#b && uv run ruff format',
    "cat <<< 'x' && uv run ruff format",
]

PANEL_READS = [
    'outcome-baseline',
    'regressor-panel',
    'margins',
    'decide',
    'economic-read',
    'outcome-panel',
    'ita-read',
    'sweep',
]

ALLOWED = [
    "grep -n 'tools growth-margins' CLAUDE.md",
    'echo uv run naics-embedder tools growth-margins',
    'git commit -m "never run ruff format"',
    'uv run naics-embedder tools growth-margins --help',
    'uv run naics-embedder tools sweep --help',
    'uv run naics-embedder tools decide -h',
    'uv run naics-embedder tools economic-read --candidate x --dry-run',
    'uv run naics-embedder tools ita-read --dry-run',
    'uv run naics-embedder tools ita-gap --reads r.json',
    'uv run naics-embedder tools radius-report --run r',
    'uv run naics-embedder tools config',
    'uv run naics-embedder --help',
    'uv run ruff check --fix src/',
    './scripts/format_code.sh --check --all',
    'uv --directory /home/u/naics-embedder run --locked --no-sync pytest -n auto',
    'uv run pytest tests/unit/test_cli_commands.py -k growth_margins',
    "echo 'unterminated",
    "git commit -F - <<'EOF'\nuv run naics-embedder tools sweep\nEOF",
    'cat <<-EOF\n\tuv run ruff format\n\tEOF',
    'uv run echo naics-embedder tools growth-margins',
    "env printf '%s\\n' ruff format",
]

@pytest.mark.parametrize('command', GROWTH_MARGINS_RUNS)
def test_guard_denies_every_growth_margins_run(command):
    decision, reason = guard_decision(command)
    assert decision == 'deny'
    assert 'specs/findings/growth-margins.md' in reason

@pytest.mark.parametrize('command', RUFF_FORMAT_RUNS)
def test_guard_denies_ruff_format(command):
    decision, reason = guard_decision(command)
    assert decision == 'deny'
    assert 'scripts/format_code.sh' in reason

@pytest.mark.parametrize('command_name', PANEL_READS)
def test_guard_asks_before_a_panel_read(command_name):
    decision, reason = guard_decision(f'uv run naics-embedder tools {command_name} --name x')
    assert decision == 'ask'
    assert f'tools {command_name}' in reason

@pytest.mark.parametrize('command', ALLOWED)
def test_guard_stays_silent_on_other_commands(command):
    assert guard_decision(command) == (None, '')

def test_guard_ignores_a_dry_run_flag_inside_a_comment():
    decision, _ = guard_decision('uv run naics-embedder tools ita-read # --dry-run')
    assert decision == 'ask'

def test_guard_denial_outranks_a_panel_read_on_the_same_line():
    decision, _ = guard_decision('uv run naics-embedder tools sweep --runs r && uv run ruff format')
    assert decision == 'deny'

def test_guard_reads_the_terminal_panel_tool_payload():
    decision, _ = guard_decision(
        'uv run naics-embedder tools growth-margins',
        tool_name='mcp__terminal__run_in_terminal',
    )
    assert decision == 'deny'

@pytest.mark.parametrize(
    'payload', ['not json', '[]', '{"tool_input": null}', '{"tool_input": {"command": 7}}']
)
def test_guard_allows_a_payload_it_cannot_read(payload):
    result = run_hook(GUARD, payload)
    assert result.returncode == 0
    assert result.stdout == ''

# -------------------------------------------------------------------------------------------------
# Format check
# -------------------------------------------------------------------------------------------------

# Records its arguments beside scripts/, prints one line and exits with $FAKE_FORMAT_EXIT.
FAKE_FORMATTER = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$(dirname "$0")/../formatter_args.txt"
echo 'would reformat (fake)'
exit "${FAKE_FORMAT_EXIT:-0}"
"""

@pytest.fixture
def project(tmp_path):
    '''A checkout with a fake scripts/format_code.sh and one file in each place the hook sorts.'''
    script = tmp_path / 'scripts' / 'format_code.sh'
    script.parent.mkdir()
    script.write_text(FAKE_FORMATTER)
    script.chmod(0o755)
    for rel in ('src/pkg/mod.py', 'tests/unit/test_mod.py', 'docs/notes.py', 'src/pkg/notes.md'):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x = 1\n')
    return tmp_path

def run_format_check(project, file_path, exit_code=0, tool_name='Edit'):
    env = {**os.environ, 'CLAUDE_PROJECT_DIR': str(project), 'FAKE_FORMAT_EXIT': str(exit_code)}
    payload = {'tool_name': tool_name, 'tool_input': {'file_path': str(file_path)}}
    return run_hook(FORMAT_CHECK, payload, env=env)

def formatter_args(project):
    path = project / 'formatter_args.txt'
    return path.read_text().splitlines() if path.exists() else None

def test_format_check_reports_a_failed_check_to_claude(project):
    result = run_format_check(project, project / 'src/pkg/mod.py', exit_code=1)
    assert result.returncode == 2
    assert 'would reformat (fake)' in result.stderr
    assert './scripts/format_code.sh src/pkg/mod.py' in result.stderr
    assert formatter_args(project) == ['--check', 'src/pkg/mod.py']

@pytest.mark.parametrize('tool_name', ['Edit', 'Write'])
def test_format_check_is_silent_when_the_file_is_clean(project, tool_name):
    result = run_format_check(project, project / 'tests/unit/test_mod.py', tool_name=tool_name)
    assert result.returncode == 0
    assert result.stderr == ''
    assert formatter_args(project) == ['--check', 'tests/unit/test_mod.py']

@pytest.mark.parametrize('rel', ['docs/notes.py', 'src/pkg/notes.md', '../elsewhere/mod.py'])
def test_format_check_skips_files_outside_src_and_tests(project, rel):
    result = run_format_check(project, project / rel, exit_code=1)
    assert result.returncode == 0
    assert formatter_args(project) is None

def test_format_check_skips_without_a_project_dir(project):
    env = {key: value for key, value in os.environ.items() if key != 'CLAUDE_PROJECT_DIR'}
    payload = {'tool_name': 'Edit', 'tool_input': {'file_path': str(project / 'src/pkg/mod.py')}}
    result = run_hook(FORMAT_CHECK, payload, env={**env, 'FAKE_FORMAT_EXIT': '1'})
    assert result.returncode == 0
    assert formatter_args(project) is None

def test_format_check_skips_a_checkout_without_the_formatter(project):
    (project / 'scripts' / 'format_code.sh').unlink()
    result = run_format_check(project, project / 'src/pkg/mod.py', exit_code=1)
    assert result.returncode == 0

@pytest.mark.parametrize('payload', ['not json', '[]', '{"tool_input": {"file_path": 3}}'])
def test_format_check_allows_a_payload_it_cannot_read(project, payload):
    env = {**os.environ, 'CLAUDE_PROJECT_DIR': str(project), 'FAKE_FORMAT_EXIT': '1'}
    result = run_hook(FORMAT_CHECK, payload, env=env)
    assert result.returncode == 0
    assert formatter_args(project) is None

# -------------------------------------------------------------------------------------------------
# Both hooks
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('script', [GUARD, FORMAT_CHECK], ids=lambda path: path.name)
def test_hooks_parse_as_python_3_9(script):
    '''Claude Code runs them with the first python3 on PATH, which is 3.9 on macOS.'''
    ast.parse(script.read_text(), feature_version=(3, 9))
