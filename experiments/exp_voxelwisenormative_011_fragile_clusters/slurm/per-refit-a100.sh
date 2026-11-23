#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=exp011-per-refit-a100
#SBATCH --output=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue
# The 64 refits of fragile-clusters-a100.sh without the five site fits or the cluster ranking, keeping every
# refit's flags for all 113 held-out scans. exp_005 ran 64 refits in 9.8 min on one A100 (7.0 s per fit);
# 20 min covers startup, compilation and the 150 MB bit-packed export.
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
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage per-refit --script "$ENTRY" -- \
    --per-refit --run-name "a100-${SLURM_JOB_ID:?}" --resources --no-profile
