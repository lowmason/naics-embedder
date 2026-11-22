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

The guard catches mistakes, not evasion. It drops shell comments and heredoc bodies, cuts the rest
into simple commands at ``;``, ``&&``, ``||``, ``|``, ``&``, parentheses and newlines, drops
redirections, and reads the script of ``bash -c`` (and of ``sh`` and ``zsh``). After a launcher
such as ``uv run`` it reads each later word as a possible program (see TEXT_PROGRAMS). It does not
read command substitutions, ``eval``, a heredoc fed to a shell, or Python that calls the CLI itself.

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

# A launcher's scan stops at these programs, because their arguments are text, not a command.
# The scan does not try to find the launcher's child exactly: that needs each launcher's option
# table, and an option value (uv's --directory .../naics-embedder) read as the child would hide
# the real command. So it reads every later word, and a rare false deny is the price.
TEXT_PROGRAMS = frozenset(
    {
        'echo',
        'printf',
        'cat',
        'grep',
        'egrep',
        'rg',
        'sed',
        'awk',
        'head',
        'tail',
        'wc',
        'jq',
        'ls',
        'find',
        'git',
        'gh',
        'pytest',
    }
)

SHELLS = frozenset({'bash', 'sh', 'zsh'})
SHELL_VALUE_OPTIONS = frozenset({'-o', '+o', '-O', '+O'})

# Words that can come before a command without being one.
LEADING_WORDS = frozenset({'!', 'if', 'then', 'elif', 'else', 'while', 'until', 'do', '{'})

PUNCTUATION = '();<>|&\n'
PUNCTUATION_CHARS = frozenset(PUNCTUATION)
SEPARATOR_CHARS = frozenset(';&|()\n')
REDIRECTION_CHARS = frozenset('<>')
WORD_BREAKS = frozenset(' \t\r\n;&|()<>')
ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=')
PYTHON = re.compile(r'python[0-9.]*$')
PYTHON_VALUE_OPTIONS = frozenset({'-W', '-X', '-Q'})
RUFF_VALUE_OPTIONS = frozenset({'--config'})
CLI_MODULES = frozenset({'naics_embedder', 'naics_embedder.cli'})
MAX_SHELL_DEPTH = 3

# A heredoc operator and its delimiter word: 'EOF', "EOF", \EOF or EOF, after << or <<-.
HEREDOC = re.compile(r'<<(-?)[ \t]*(?:\'([^\'\n]*)\'|"([^"\n]*)"|\\?([^\s;&|()<>\'"]+))')

def skip_heredoc_bodies(command, index, pending):
    '''The index just past the bodies of the pending heredocs, the first of which starts at index.

    Each body ends at its delimiter's line. Under <<- the shell strips leading tabs first.
    '''
    for delimiter, strip_tabs in pending:
        while index < len(command):
            newline = command.find('\n', index)
            end = len(command) if newline < 0 else newline
            line = command[index:end]
            index = end + 1
            if (line.lstrip('\t') if strip_tabs else line) == delimiter:
                break
    return min(index, len(command))

def readable_text(command):
    '''The command without its comments and heredoc bodies, and otherwise unchanged.

    A comment is an unquoted # at the start of a word, up to the end of its line. Its newline
    stays, so the next line is still a command of its own. A heredoc body is text for a program's
    stdin, not commands, and prose there often holds an unmatched apostrophe. Quotes and
    backslashes are followed as a POSIX shell reads them, so a # inside quotes or a word stays.
    '''
    out = []
    pending = []  # (delimiter, strip_tabs) for each heredoc whose body starts on the next line
    quote = None
    word_start = True
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == '\\' and quote == '"':
                out.append(command[index:index + 2])
                index += 2
                continue
            quote = None if char == quote else quote
            out.append(char)
            index += 1
            continue
        if char == '\\':
            out.append(command[index:index + 2])
            index += 2
            word_start = False
            continue
        if char in '\'"':
            quote = char
            out.append(char)
            index += 1
            word_start = False
            continue
        if char == '#' and word_start:
            newline = command.find('\n', index)
            index = len(command) if newline < 0 else newline
            continue
        if command.startswith('<<<', index):
            out.append('<<<')
            index += 3
            word_start = True
            continue
        heredoc = HEREDOC.match(command, index) if char == '<' else None
        if heredoc:
            delimiter = next(group for group in heredoc.groups()[1:] if group is not None)
            pending.append((delimiter, heredoc.group(1) == '-'))
            out.append(heredoc.group())
            index = heredoc.end()
            word_start = False
            continue
        out.append(char)
        index += 1
        word_start = char in WORD_BREAKS
        if char == '\n' and pending:
            index = skip_heredoc_bodies(command, index, pending)
            pending = []
    return ''.join(out)

def simple_commands(command):
    '''Cut a command line into the word lists of its simple commands.

    Raises ValueError, as shlex does, for an unclosed quote or a trailing backslash.
    '''
    text = readable_text(command).replace('\\\n', ' ')
    lexer = shlex.shlex(text, posix=True, punctuation_chars=PUNCTUATION)
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

def without_redirections(words):
    '''The words without each redirection operator, the file descriptor before it and its operand.

    shlex cuts 2>/dev/null into 2, > and /dev/null. A digit word just before an operator is taken
    as its file descriptor, so a spaced `cmd 2 > f` loses the argument 2 here, which is harmless.
    '''
    kept = []
    index = 0
    while index < len(words):
        word = words[index]
        if PUNCTUATION_CHARS.issuperset(word) and REDIRECTION_CHARS & set(word):
            if kept and kept[-1].isdigit():
                kept.pop()
            index += 2
            continue
        kept.append(word)
        index += 1
    return kept

def name(word):
    return os.path.basename(word)

def positionals(words, value_options=frozenset()):
    '''The words that are not options, skipping the value after each option in value_options.'''
    found = []
    skip = False
    for word in words:
        if skip:
            skip = False
        elif word.startswith('-'):
            skip = word in value_options
        else:
            found.append(word)
    return found

def shell_script(args):
    '''The script after a shell's -c option, or None. -o and -O take an operand.'''
    index = 0
    while index < len(args):
        word = args[index]
        if word in SHELL_VALUE_OPTIONS:
            index += 2
            continue
        if not word.startswith(('-', '+')) or word in ('-', '--'):
            return None
        if word.startswith('-') and not word.startswith('--') and 'c' in word[1:]:
            return args[index + 1] if index + 1 < len(args) else None
        index += 1
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
    words = positionals(args, RUFF_VALUE_OPTIONS)
    if words and words[0] == 'format':
        return 'deny', RUFF_FORMAT_REASON
    return None

def program_decision(program, args):
    '''The decision for one program and its arguments, or None.'''
    if program == 'naics-embedder':
        return cli_decision(args)
    if program == 'ruff':
        return ruff_decision(args)
    if PYTHON.match(program):
        module, module_args = python_module(args)
        if module in CLI_MODULES:
            return cli_decision(module_args)
        if module == 'ruff':
            return ruff_decision(module_args)
    return None

def word_decisions(words, depth):
    '''Yield the decisions for one simple command.

    After a launcher, every later word may be the child program. The scan stops at a text
    program, at a shell and at the first decision. A target that gives no decision does not stop
    it, because that word may be an option's value.
    '''
    words = without_redirections(words)
    while words and (words[0] in LEADING_WORDS or ASSIGNMENT.match(words[0])):
        words = words[1:]
    if not words:
        return
    starts = range(1, len(words)) if name(words[0]) in LAUNCHERS else [0]
    for start in starts:
        program, args = name(words[start]), words[start + 1:]
        if program in TEXT_PROGRAMS:
            return
        if program in SHELLS:
            script = shell_script(args)
            if script is not None and depth < MAX_SHELL_DEPTH:
                yield from line_decisions(script, depth + 1)
            return
        decision = program_decision(program, args)
        if decision is not None:
            yield decision
            return

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
