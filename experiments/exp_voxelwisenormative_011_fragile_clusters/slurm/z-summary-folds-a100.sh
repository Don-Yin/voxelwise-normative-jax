#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=exp011-z-summary-folds-a100
#SBATCH --array=0-4
#SBATCH --output=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue
# per-refit-folds-a100.sh with --z-summary: the same 64 refits of every cross-fit fold (array task k = fold k, one A100
# each), plus one fit without resampling, and z-summary.npz holding that fit's z with every voxel's smallest and largest
# |z| over the 64 refits (3 float64 arrays, about 450 MB per fold), so any |z| threshold can be applied afterwards.
# The flags and counts are those of per-refit-folds-a100.sh; the first refit's seconds no longer hold the compile time,

# reserved per task (1.7 GPU hours at most).

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

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091

# shellcheck disable=SC1090

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x $SLURM_PROFILE_BIN ]]
ENTRY=experiments/exp_voxelwisenormative_011_fragile_clusters/run.py
FOLD=${SLURM_ARRAY_TASK_ID:?submit as an array, --array=0-4}
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage "z-summary-fold-$FOLD" --script "$ENTRY" -- \
    --per-refit --fold "$FOLD" --z-summary --run-name "a100-${SLURM_ARRAY_JOB_ID:?}-z-fold$FOLD" \
    --resources --no-profile
