#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace_dir=$(CDPATH= cd -- "$project_dir/.." && pwd)
current_manifest=${1:-"$project_dir/official/file_manifest_latest.csv"}
report=${2:-"$project_dir/reports/phase2_readiness_latest.json"}
raw_dir=${3:-"$project_dir/raw"}

"$workspace_dir/.venv/bin/kaggle" competitions files \
  -c tree-species-hsi-2026 -v >"$current_manifest"

cd "$project_dir"
"$workspace_dir/.venv/bin/python" -m src.phase2_readiness \
  --baseline-manifest official/file_manifest.csv \
  --current-manifest "$current_manifest" \
  --raw-dir "$raw_dir" \
  --report "$report"
