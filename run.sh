#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash run.sh <train|validate|inference> <config.py> [arguments...]

  train      Start torchrun using config['world_size']; accepts --resume, --work-dir, etc.
  validate   Generate RAW/EMA validation images; requires --checkpoint and --output.
  inference  Generate images for custom prompts; requires --checkpoint, --prompts and --output.

Examples:
  bash run.sh train configs/rfm.py
  bash run.sh validate configs/rfm.py --checkpoint model.pth --output output/validation
  bash run.sh inference configs/rfm.py --checkpoint model.pth --prompts examples/inference_prompts.json --output output/inference

All relative paths are resolved from this repository's root.
Set PYTHON_BIN to select Python and CUDA_VISIBLE_DEVICES to select GPUs.
For train, --launch uses background preparation; --help displays trainer help without starting workers.
HELP
    exit 0
fi

if [[ $# -lt 2 ]]; then
    echo 'Expected an action and config path. Run: bash run.sh --help' >&2
    exit 2
fi

action=$1
config=$2
shift 2
case "$action" in
    train|validate|inference) ;;
    *) echo "Unknown action: $action (expected train, validate or inference)" >&2; exit 2 ;;
esac

RFM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$RFM_ROOT"
if [[ ! -f "$config" ]]; then
    echo "Config file not found: $config" >&2
    exit 2
fi
PYTHON_BIN=${PYTHON_BIN:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

case "$action" in
    train)
        for argument in "$@"; do
            if [[ "$argument" == --launch || "$argument" == -h || "$argument" == --help ]]; then
                exec "$PYTHON_BIN" -m scripts.train_rfm "$config" "$@"
            fi
        done
        world_size=$("$PYTHON_BIN" - "$config" <<'PY'
import runpy
import sys

config = runpy.run_path(sys.argv[1])['config']
world_size = config['world_size']
if type(world_size) is not int or world_size < 1:
    raise SystemExit('config.world_size must be a positive integer')
print(world_size)
PY
        )
        exec "$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$world_size" \
            scripts/train_rfm.py "$config" "$@"
        ;;
    validate)
        exec "$PYTHON_BIN" -m scripts.validate_rfm --config "$config" "$@"
        ;;
    inference)
        exec "$PYTHON_BIN" -m scripts.inference_rfm --config "$config" "$@"
        ;;
esac
