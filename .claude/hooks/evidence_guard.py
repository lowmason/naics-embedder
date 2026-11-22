#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# Evidence Guard (PreToolUse hook)
# -------------------------------------------------------------------------------------------------
'''
PreToolUse guard for this repository's evidence rules, registered in ``.claude/settings.json``.

Claude Code pipes the hook payload in on stdin. For a shell command, the guard prints one decision:

1. Deny ``naics-embedder tools growth-margins``. The growth margins were fixed once, on
   2026-10-17, and nothing in the code refuses a second run.
2. Deny ``ruff format``. YAPF owns layout, and ``ruff format`` rewrites quotes and blank lines.
3. Ask before a tools command that reads a real panel, because real-data reads need the owner's
   approval. ``--help`` and ``--dry-run`` pass with no prompt.

A deny outranks an ask. Any other command gets no output, and the normal permission flow applies.

The guard catches mistakes, not evasion. It cuts a command into simple commands at ``;``, ``&&``,
``||``, ``|``, ``&``, parentheses and newlines, and it reads the script of ``bash -c`` (and of
``sh`` and ``zsh``). It does not read command substitutions, ``eval``, heredocs, or Python that
calls the CLI itself.

Claude Code runs it with the first python3 on PATH, which is 3.9 on macOS. So it uses the standard
library and 3.9 syntax only. A payload or a command that it cannot read gets no output: the guard
must never break a call that it cannot read.
'''

import json
import os
import re
import shlex
import sys

# -------------------------------------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------------------------------------

GROWTH_MARGINS = 'growth-margins'

# The tools commands that score an arm or a candidate on a real panel, or record a decision.
PANEL_READS = frozenset(
    {
        'outcome-baseline',
        'regressor-panel',
        'margins',
        'decide',
        'economic-read',
        'outcome-panel',
        'ita-read',
        'sweep',
    }
)

# Their --dry-run runs the checks and reads no score.
DRY_RUN_COMMANDS = frozenset({'economic-read', 'ita-read'})

HELP_FLAGS = frozenset({'--help', '-h'})

GROWTH_MARGINS_REASON = (
    'Never run `tools growth-margins` again. The growth margins were fixed once, on 2026-10-17 '
    '(specs/findings/growth-margins.md), and every economic candidate is decided against that '
    'record. Nothing in the code refuses a second run, so this project hook '
    '(.claude/hooks/evidence_guard.py) does. Its --help still runs.'
)

RUFF_FORMAT_REASON = (
    'Never run `ruff format` in this repository. It forces double quotes, which the Q rules '
    'reject, and two blank lines between top-level definitions. YAPF owns layout: run '
    '`./scripts/format_code.sh <paths>` (ruff check --fix, then YAPF) instead.'
)

def panel_read_reason(command):
    '''The prompt text for a tools command that reads a real panel.'''
    reason = (
        f"`tools {command}` reads a real panel, and real-data reads need the owner's approval "
        '(.claude/hooks/evidence_guard.py). Its --help runs with no prompt.'
    )
    if command in DRY_RUN_COMMANDS:
        reason += ' So does its --dry-run, which reads no score.'
    return reason

# -------------------------------------------------------------------------------------------------
# Shell words
# -------------------------------------------------------------------------------------------------

# Programs that run another program after their own options and arguments.
LAUNCHERS = frozenset(
    {
        'uv',
        'uvx',
        'env',
        'command',
        'exec',
        'nice',
        'nohup',
        'time',
        'timeout',
        'xargs',
        'caffeinate',
        'sudo',
    }
)

SHELLS = frozenset({'bash', 'sh', 'zsh'})

# Words that can come before a command without being one.
LEADING_WORDS = frozenset({'!', 'if', 'then', 'elif', 'else', 'while', 'until', 'do', '{'})

PUNCTUATION = '();<>|&\n'
SEPARATOR_CHARS = frozenset(';&|()\n')
ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=')
PYTHON = re.compile(r'python[0-9.]*$')
PYTHON_VALUE_OPTIONS = frozenset({'-W', '-X', '-Q'})
CLI_MODULES = frozenset({'naics_embedder', 'naics_embedder.cli'})
MAX_SHELL_DEPTH = 3

def simple_commands(command):
    '''Cut a command line into the word lists of its simple commands.

    Raises ValueError, as shlex does, for an unclosed quote or a trailing backslash.
    '''
    lexer = shlex.shlex(command.replace('\\\n', ' '), posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace = ' \t\r'
    lexer.whitespace_split = True
    lexer.commenters = ''
    current = []
    for token in lexer:
        if token and SEPARATOR_CHARS.issuperset(token):
            if current:
                yield current
            current = []
        else:
            current.append(token)
    if current:
        yield current

def name(word):
    return os.path.basename(word)

def positionals(words):
    return [word for word in words if not word.startswith('-')]

def shell_script(args):
    '''The script after a shell's -c option, or None.'''
    for index, word in enumerate(args):
        if not word.startswith('-'):
            return None
        if not word.startswith('--') and 'c' in word[1:]:
            return args[index + 1] if index + 1 < len(args) else None
    return None

def python_module(args):
    '''The module after a Python interpreter's -m option and the words after it, or (None, []).'''
    index = 0
    while index < len(args):
        word = args[index]
        if word == '-m' and index + 1 < len(args):
            return args[index + 1], args[index + 2:]
        if not word.startswith('-') or word == '-c':
            break
        index += 2 if word in PYTHON_VALUE_OPTIONS else 1
    return None, []

# -------------------------------------------------------------------------------------------------
# Decisions
# -------------------------------------------------------------------------------------------------

def cli_decision(args):
    '''The decision for naics-embedder's arguments, or None.'''
    words = positionals(args)
    if len(words) < 2 or words[0] != 'tools' or HELP_FLAGS & set(args):
        return None
    command = words[1]
    if command == GROWTH_MARGINS:
        return 'deny', GROWTH_MARGINS_REASON
    if command in PANEL_READS:
        if command in DRY_RUN_COMMANDS and '--dry-run' in args:
            return None
        return 'ask', panel_read_reason(command)
    return None

def ruff_decision(args):
    words = positionals(args)
    if words and words[0] == 'format':
        return 'deny', RUFF_FORMAT_REASON
    return None

def word_decisions(words, depth):
    '''Yield the decisions for one simple command.'''
    while words and (words[0] in LEADING_WORDS or ASSIGNMENT.match(words[0])):
        words = words[1:]
    if not words:
        return
    starts = range(1, len(words)) if name(words[0]) in LAUNCHERS else [0]
    for start in starts:
        program, args = name(words[start]), words[start + 1:]
        if program in SHELLS:
            script = shell_script(args)
            if script is not None and depth < MAX_SHELL_DEPTH:
                yield from line_decisions(script, depth + 1)
        elif program == 'naics-embedder':
            yield cli_decision(args)
        elif program == 'ruff':
            yield ruff_decision(args)
        elif PYTHON.match(program):
            module, module_args = python_module(args)
            if module in CLI_MODULES:
                yield cli_decision(module_args)
            elif module == 'ruff':
                yield ruff_decision(module_args)

def line_decisions(command, depth=0):
    for words in simple_commands(command):
        yield from word_decisions(words, depth)

def decide(command):
    '''The (decision, reason) for a command line, or None. A deny outranks an ask.'''
    decisions = [decision for decision in line_decisions(command) if decision is not None]
    for decision in decisions:
        if decision[0] == 'deny':
            return decision
    reasons = []
    for _, reason in decisions:
        if reason not in reasons:
            reasons.append(reason)
    return ('ask', ' '.join(reasons)) if reasons else None

# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

def main():
    '''Always exit 0. A decision is JSON on stdout. No output lets the call through as usual.'''
    try:
        payload = json.loads(sys.stdin.read())
        tool_input = payload.get('tool_input') if isinstance(payload, dict) else None
        command = tool_input.get('command') if isinstance(tool_input, dict) else None
        if not isinstance(command, str) or not command.strip():
            return 0
        decision = decide(command)
    except Exception:
        return 0
    if decision is not None:
        print(
            json.dumps(
                {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': decision[0],
                        'permissionDecisionReason': decision[1],
                    }
                }
            )
        )
    return 0

if __name__ == '__main__':
    sys.exit(main())
