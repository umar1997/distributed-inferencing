#!/usr/bin/env bash
# One-time install on a laptop (macOS or Linux): Python environment, packages, and the web page.
# Run from anywhere: ./scripts/install.sh   Safe to run again (e.g. after `git pull`).
# Every step is also written out by hand in docs/LAPTOP_SETUP.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# 1. Python 3.12 (the version every node and the Docker image use)
say "1/4 Looking for Python 3.12"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
      PY="$(command -v "$candidate")"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  if [ "$(uname)" = "Darwin" ]; then
    fail "Python 3.12 not found. Install it with:  brew install python@3.12   (Homebrew: https://brew.sh), then run this again."
  else
    fail "Python 3.12 not found. Install it with:  sudo apt install python3.12 python3.12-venv   then run this again."
  fi
fi
echo "using $PY ($("$PY" --version))"

# 2. The virtual environment: a private folder (.venv) with its own Python and packages,
#    so this project's versions never clash with anything else on the laptop.
say "2/4 Creating the Python environment in .venv"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv || fail "Could not create .venv. On Ubuntu/Debian: sudo apt install python3.12-venv"
else
  echo ".venv already exists, reusing it"
fi
.venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade >/dev/null
.venv/bin/python -m pip install --quiet --upgrade pip

# 3. Packages, at the exact versions in requirements.txt. The PyTorch CPU index gives Linux
#    the CPU-only PyTorch (no multi-GB graphics card libraries); macOS ignores it.
say "3/4 Installing packages (a few minutes the first time; PyTorch is large)"
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python -c "import torch, transformers, fastapi, moe_dist; print('packages OK: torch', torch.__version__, '| transformers', transformers.__version__)"

# 4. The web page (only the master serves it). Needs Node.js; without it the master still works.
say "4/4 Building the web page (ui/)"
if command -v npm >/dev/null; then
  (cd ui && npm ci --no-fund --no-audit && npm run build) && echo "web page built in ui/dist"
else
  echo "Node.js not found: skipped. Only the master shows the web page; to build it there, install"
  echo "Node.js 24 (macOS: brew install node, Linux: https://nodejs.org) and run this script again."
fi

say "Done. Next: ./moe setup   (docs/LAPTOP_SETUP.md: section 2 on the master, section 3 on a worker)"
