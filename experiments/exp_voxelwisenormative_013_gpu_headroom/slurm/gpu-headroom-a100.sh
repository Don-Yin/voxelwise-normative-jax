#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --array=0-1
#SBATCH --job-name=exp013-gpu-headroom
#SBATCH --output=experiments/exp_voxelwisenormative_013_gpu_headroom/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_013_gpu_headroom/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue
# Task 0 = float32, task 1 = float64; one A100 each, IXI 450 references, 113 held-out, exp_005's 64 draws.
# Each task fits the 64 draws 8 times: K=1 (bitwise reference), K=1 again (determinism and speed baseline),
# then K=2,4,8,16,32,64 refits per compiled call, stopping at the first out-of-memory K.

# 1.724 s float32 per warm whole-mask fit) and its 72.8 s input read: 8 passes x 64 refits is 1,423 s
# (float64) / 883 s (float32), plus up to 20% more loop steps at large K, ~200 s of compiles, ~130 s of
# host z comparisons and ~90 s of tracing: about 36 min (float64) and 25 min (float32).
# Requested: 2 tasks x 50 min = 1.67 GPU-hours (charged in full); expected use about 1.0 GPU-hour.
# Host RAM: the K=1 reference z (64 x 113 x 165,402 float64 = 9.6 GB) plus one K=64 batch (9.6 GB).
set -euo pipefail

VOXELWISE_HEARTBEAT_SECONDS=180
VOXELWISE_HEARTBEAT_STARTED_AT=$(date +%s)
VOXELWISE_HEARTBEAT_PID=""
voxelwise_start_heartbeat() {
    (
        trap 'kill "$sleep_pid" 2>/dev/null || true; exit 0' HUP INT TERM
        while :; do
            sleep "$VOXELWISE_HEARTBEAT_SECONDS" &
            sleep_pid=$!
            wait "$sleep_pid"
            now=$(date +%s)
            printf '%s HEARTBEAT job=%s elapsed_seconds=%s status=running\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${SLURM_JOB_ID:-unknown}" "$((now - VOXELWISE_HEARTBEAT_STARTED_AT))" >&2
        done
    ) &
    VOXELWISE_HEARTBEAT_PID=$!
}
voxelwise_stop_heartbeat() {
    if [[ -n $VOXELWISE_HEARTBEAT_PID ]]; then
        kill "$VOXELWISE_HEARTBEAT_PID" 2>/dev/null || true
        wait "$VOXELWISE_HEARTBEAT_PID" 2>/dev/null || true
    fi
}
voxelwise_start_heartbeat
trap voxelwise_stop_heartbeat EXIT

precisions=(float32 float64)
precision=${precisions[${SLURM_ARRAY_TASK_ID:?submit as an array}]:?array index must be 0 or 1}

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091

# shellcheck disable=SC1090

command -v nvidia-smi >/dev/null
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# exp_010 ran with JAX's default 75% pool (63.8 GB); 95% lets large K use up to ~76 GB before it fails.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x $SLURM_PROFILE_BIN ]]
ENTRY=experiments/exp_voxelwisenormative_013_gpu_headroom/run.py
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage "sweep-$precision" --script "$ENTRY" -- \
    --no-profile --resources sweep --precision "$precision" --run-name "a100-${SLURM_ARRAY_JOB_ID:?}"
