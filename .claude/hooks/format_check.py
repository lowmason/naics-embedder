#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# Format Check (PostToolUse hook)
# -------------------------------------------------------------------------------------------------
'''
PostToolUse check after an Edit or a Write, registered in ``.claude/settings.json``.

When the edited file is a Python file under the project's ``src/`` or ``tests/``, the hook runs
``scripts/format_code.sh --check`` on that one file. A failed check prints the script's output on
stderr and exits 2, so that Claude reads it. Nothing changes on disk.

The hook checks and does not fix, on purpose. ``ruff check --fix`` deletes an unused import
(F401), so a fixer would delete an import that one edit adds and the next edit uses. For the same
reason, an import added before its first use reports F401 until that use lands.

The project is ``$CLAUDE_PROJECT_DIR``, the session's worktree. Claude Code runs the hook with the
first python3 on PATH, which is 3.9 on macOS. So it uses the standard library and 3.9 syntax only.
Anything that it cannot read or run exits 0: the hook must never break an edit.
'''

import json
import os
import subprocess
import sys

CHECKED_DIRS = ('src', 'tests')
TIMEOUT_SECONDS = 90
MAX_OUTPUT_CHARS = 6000

def checked_path(project, file_path):
    '''The file's path relative to the project when the hook checks it, else None.'''
    project = os.path.realpath(project)
    path = os.path.realpath(os.path.join(project, file_path))
    if not path.endswith('.py') or not os.path.isfile(path):
        return None
    parts = os.path.relpath(path, project).split(os.sep)
    if len(parts) < 2 or parts[0] not in CHECKED_DIRS:
        return None
    return '/'.join(parts)

def failure_message(relative, result):
    output = (result.stdout + result.stderr).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + '\n[output cut to its first 6,000 characters]'
    return (
        f'scripts/format_code.sh --check failed for {relative} (exit {result.returncode}).\n'
        f'Fix it with ./scripts/format_code.sh {relative} when this edit sequence is done. '
        'An import added before its first use reports F401 until that use lands.\n\n'
        f'{output}\n'
    )

def main():
    '''Exit 0 when the file passes or is not checked, and 2 with stderr when the check fails.'''
    try:
        payload = json.loads(sys.stdin.read())
        tool_input = payload.get('tool_input') if isinstance(payload, dict) else None
        file_path = tool_input.get('file_path') if isinstance(tool_input, dict) else None
        project = os.environ.get('CLAUDE_PROJECT_DIR')
        if not isinstance(file_path, str) or not project:
            return 0
        relative = checked_path(project, file_path)
        script = os.path.join(project, 'scripts', 'format_code.sh')
        if relative is None or not os.path.isfile(script):
            return 0
        result = subprocess.run(
            ['bash', script, '--check', relative],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except Exception:
        return 0
    if result.returncode == 0:
        return 0
    sys.stderr.write(failure_message(relative, result))
    return 2

if __name__ == '__main__':
    sys.exit(main())
