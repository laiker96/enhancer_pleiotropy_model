#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix="$repository/.venv"

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba is required but was not found on PATH" >&2
  exit 1
fi

if [[ -e "$prefix/conda-meta/history" ]]; then
  mamba env update --yes --prefix "$prefix" --file "$repository/environment.yml" --prune
else
  mamba env create --yes --prefix "$prefix" --file "$repository/environment.yml"
fi

"$prefix/bin/python" -m pip install --no-deps --editable "$repository"
"$prefix/bin/python" -m pip list --format=freeze \
  --exclude enhancer-pleiotropy-model > "$repository/environment.lock.txt"

echo "Environment ready: $prefix"
echo "Activate with: mamba activate $prefix"
