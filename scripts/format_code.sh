#!/usr/bin/env bash
#
# Format Python files: ruff (lint autofixes + import sorting), then yapf (layout).
# Both are configured in pyproject.toml. `ruff format` is deliberately not run: it forces two blank
# lines between top-level definitions and double quotes, which would undo yapf's layout.
#
# Usage: ./scripts/format_code.sh [--check] [--all | PATH...]
#   (no paths)  Python files under src/ and tests/ that differ from the merge base with
#               $FORMAT_BASE (default: origin/main), including uncommitted and untracked files
#   PATH...     Only these files or directories
#   --all       All of src/ and tests/. Rewrites every file that drifted while the formatter was
#               missing, so keep it out of feature PRs
#   --check     Change nothing; exit non-zero on ruff issues or files yapf would reformat
#

set -eo pipefail

# Colors only when writing to a terminal
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' NC=''
fi

usage() {
    echo 'Usage: ./scripts/format_code.sh [--check] [--all | PATH...]'
}

check=false
all=false
paths=()
for arg in "$@"; do
    case "$arg" in
        --check) check=true ;;
        --all) all=true ;;
        -h | --help) usage; exit 0 ;;
        -*) echo -e "${RED}Unknown option: $arg${NC}" >&2; usage >&2; exit 2 ;;
        *) paths+=("$arg") ;;
    esac
done

if $all && [ ${#paths[@]} -gt 0 ]; then
    echo -e "${RED}Error: pass either --all or paths, not both${NC}" >&2
    exit 2
fi

if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed or not in PATH${NC}" >&2
    exit 1
fi

# Determine which files to format
if [ ${#paths[@]} -gt 0 ]; then
    targets=("${paths[@]}")
else
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    if $all; then
        targets=(src tests)
    else
        base_ref="${FORMAT_BASE:-origin/main}"
        if ! base="$(git merge-base "$base_ref" HEAD 2> /dev/null)"; then
            echo -e "${RED}Error: no merge base with '$base_ref'${NC}" >&2
            echo 'Pass paths, use --all, or set FORMAT_BASE to another ref.' >&2
            exit 1
        fi
        targets=()
        while IFS= read -r file; do
            targets+=("$file")
        done < <(
            {
                git diff --name-only --diff-filter=d "$base" -- 'src/*.py' 'tests/*.py'
                git ls-files --others --exclude-standard -- 'src/*.py' 'tests/*.py'
            } | sort -u
        )
        if [ ${#targets[@]} -eq 0 ]; then
            echo -e "${GREEN}No Python files under src/ or tests/ changed vs $base_ref.${NC}"
            exit 0
        fi
    fi
fi

echo -e "${YELLOW}Targets:${NC} ${targets[*]}"

if $check; then
    status=0
    echo -e "\n${YELLOW}Running ruff check...${NC}"
    uv run ruff check "${targets[@]}" || status=1
    echo -e "\n${YELLOW}Running yapf --diff...${NC}"
    uv run yapf --diff --recursive --parallel "${targets[@]}" || status=1
    if [ $status -eq 0 ]; then
        echo -e "\n${GREEN}Clean: no lint issues and no formatting changes.${NC}"
    else
        echo -e "\n${RED}Check failed (see output above).${NC}"
    fi
    exit $status
fi

# Apply safe lint fixes (incl. import sorting) silently; whatever is left gets reported at the end
echo -e "\n${YELLOW}Running ruff check --fix...${NC}"
uv run ruff check --fix --exit-zero --silent "${targets[@]}"

echo -e "\n${YELLOW}Running yapf...${NC}"
uv run yapf --in-place --recursive --parallel "${targets[@]}"

echo -e "\n${YELLOW}Running ruff check...${NC}"
if uv run ruff check "${targets[@]}"; then
    echo -e "\n${GREEN}Formatting complete.${NC}"
else
    echo -e "\n${RED}Formatted, but ruff reports issues it cannot auto-fix (see above).${NC}"
    exit 1
fi
