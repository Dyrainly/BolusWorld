set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: GPU=0 bash train_4090d.sh INIT_CHECKPOINT RUN_DIR" >&2
    exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec "${PYTHON:-python}" -u "$project_dir/bolusworld_residual/train_sequence.py" train \
    --run "$2" --init "$1" --batch 4 --epochs 20 --steps 20 --seed 43 \
    --lr 5e-5 --history-every 1 --history-steps 8 --start-fraction .5 \
    --early-val 500 --image-weight .10 --rollout-loss-steps 4 --balance-cohorts
