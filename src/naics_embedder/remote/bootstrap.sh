#!/usr/bin/env bash
set -euo pipefail

repo="${1:?repository root required}"
cd "$repo"
step() { printf 'bootstrap: %s\n' "$*" >&2; }

packages=()
command -v tmux >/dev/null 2>&1 || packages+=(tmux)
command -v rsync >/dev/null 2>&1 || packages+=(rsync)
if ((${#packages[@]})); then
    step 'install missing tmux/rsync'
    sudo -n apt-get update >&2
    sudo -n apt-get install -y "${packages[@]}" >&2
fi
command -v tmux >/dev/null 2>&1 || { step 'tmux missing after installation'; exit 1; }
step 'qualify GNU rsync >=3.2'
rsync_banner="$(rsync --version)"
rsync_version="$(printf '%s\n' "$rsync_banner" | sed -nE 's/^rsync[[:space:]]+version ([0-9]+)\.([0-9]+)\.([0-9]+).*/\1 \2 \3/p' | head -1)"
read -r major minor patch <<< "$rsync_version"
if [[ "$rsync_banner" == *openrsync* ]] || [[ ! "$major" =~ ^[0-9]+$ ]] || ((major < 3 || (major == 3 && minor < 2))); then
    step 'GNU rsync >=3.2 required'; exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv="$(command -v uv)"
else
    step 'install official standalone uv'
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --silent --show-error --location https://astral.sh/uv/install.sh -o "$installer" >&2
    UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer" >&2
    uv="$HOME/.local/bin/uv"
fi
uv="$(cd "$(dirname "$uv")" && pwd -P)/$(basename "$uv")"
[[ -x "$uv" ]] || { step 'uv executable missing'; exit 1; }
lock_before="$(sha256sum uv.lock | cut -d ' ' -f 1)"
step "sync --locked using $uv"
"$uv" sync --locked >&2
lock_after="$(sha256sum uv.lock | cut -d ' ' -f 1)"
[[ "$lock_before" == "$lock_after" ]] || { step 'uv.lock changed during sync'; exit 1; }
step 'verify NTP synchronization'
[[ "$(timedatectl show --property=NTPSynchronized --value)" == yes ]] || { step 'NTP synchronization required'; exit 1; }
step 'qualify native BF16 on logical CUDA device 0'
printf '{}' | "$uv" run --locked --no-sync python -m naics_embedder.remote.worker bootstrap --root "$repo" --uv "$uv"
