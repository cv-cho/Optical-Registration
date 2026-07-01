#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
  echo "Run: bash scripts/setup_macos.sh" >&2
  exit 1
fi

cd "${ROOT}"
"${PYTHON_BIN}" scripts/verify_data_package.py
