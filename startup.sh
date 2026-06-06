#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run jupyter notebook instructions.ipynb
