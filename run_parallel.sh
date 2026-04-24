#!/usr/bin/env bash

set -euo pipefail

# Parallel experiment launcher for carving.py
# Usage:
#   bash run_parallel.sh
# Optional env vars:
#   PARALLEL=3 OUT_ROOT=outputs/experiments DEVICE=cpu MESH_PATH=path/to/model.obj

PARALLEL="${PARALLEL:-32}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments}"
DEVICE="${DEVICE:-cpu}"
MESH_PATH="${MESH_PATH:-}"

mkdir -p "$OUT_ROOT"

CONFIGS=(
  "5 64"
  "10 64"
  "30 96"
  "50 96"
)

run_one() {
  local views="$1"
  local vox="$2"
  local out_dir="$OUT_ROOT/views_${views}_vox_${vox}"
  local mesh_arg=()

  if [[ -n "${MESH_PATH:-}" ]]; then
    mesh_arg=(--mesh-path "$MESH_PATH")
  fi

  echo "[start] views=$views vox=$vox -> $out_dir"
  python carving.py \
    "${mesh_arg[@]}" \
    --out-dir "$out_dir" \
    --num-views "$views" \
    --voxel-resolution "$vox" \
    --image-size 256 \
    --recon-method poisson \
    --poisson-depth 8 \
    --device "$DEVICE"
  echo "[done ] views=$views vox=$vox"
}

export -f run_one
export OUT_ROOT DEVICE

if [[ -n "$MESH_PATH" ]]; then
  export MESH_PATH
fi

for cfg in "${CONFIGS[@]}"; do
  printf "%s\n" "$cfg"
done | xargs -P "$PARALLEL" -n 2 bash -c 'run_one "$@"' _

python - <<PY
import json
from pathlib import Path

root = Path("${OUT_ROOT}")
rows = []
for metrics_path in sorted(root.glob("**/metrics.json")):
    data = json.loads(metrics_path.read_text())
    cfg = data.get("config", {})
    rows.append(
        {
            "run": str(metrics_path.parent),
            "views": cfg.get("num_views"),
            "voxel_resolution": cfg.get("voxel_resolution"),
            "occupied_voxels": data.get("occupied_voxels"),
            "occupancy_ratio": data.get("occupancy_ratio"),
            "runtime_sec": data.get("runtime_sec"),
            "mesh": data.get("outputs", {}).get("mesh"),
        }
    )

summary_path = root / "summary.json"
summary_path.write_text(json.dumps(rows, indent=2))
print(f"Wrote summary: {summary_path}")
for row in rows:
    print(row)
PY
