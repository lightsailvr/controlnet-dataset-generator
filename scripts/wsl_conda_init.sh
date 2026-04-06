#!/bin/bash
# Ensure conda is available in non-interactive WSL shells.
# Sourced by the Electron app before `conda run` commands.

# Already available? Nothing to do.
command -v conda >/dev/null 2>&1 && return 0 2>/dev/null

# 1. Try common install directories
for _d in \
  "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/mambaforge" \
  "/opt/conda" "/usr/local/conda" \
  /home/*/miniconda3 /home/*/miniforge3 /home/*/anaconda3 /home/*/mambaforge
do
  if [ -f "$_d/etc/profile.d/conda.sh" ]; then
    . "$_d/etc/profile.d/conda.sh"
    return 0 2>/dev/null
  fi
done

# 2. Extract conda.sh path from ~/.bashrc (avoids eval which breaks on PATH with parens)
if ! command -v conda >/dev/null 2>&1; then
  _cpath=$(grep 'profile\.d/conda\.sh' "$HOME/.bashrc" 2>/dev/null | head -1 | tr '"' '\n' | grep 'conda\.sh' | head -1)
  if [ -n "$_cpath" ] && [ -f "$_cpath" ]; then
    . "$_cpath"
  fi
fi

# 3. Last resort: find the conda binary path from ~/.bashrc and add its parent to PATH
if ! command -v conda >/dev/null 2>&1; then
  _cbin=$(grep -m1 'bin/conda' "$HOME/.bashrc" 2>/dev/null | tr "'" '\n' | grep 'bin/conda$' | head -1)
  if [ -n "$_cbin" ] && [ -x "$_cbin" ]; then
    _cdir=$(dirname "$_cbin")
    . "$_cdir/../etc/profile.d/conda.sh" 2>/dev/null || export PATH="$_cdir:$PATH"
  fi
fi
