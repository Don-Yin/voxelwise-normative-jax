#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --job-name=exp004-voxel-speed-sweep
#SBATCH --output=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# This allocation covers 9 warm/three-repeat variants, 3 tuning widths and JAX compilations.

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
    if [[ -n "$VOXELWISE_HEARTBEAT_PID" ]]; then
        kill "$VOXELWISE_HEARTBEAT_PID" 2>/dev/null || true
        wait "$VOXELWISE_HEARTBEAT_PID" 2>/dev/null || true
    fi
}
voxelwise_start_heartbeat
trap voxelwise_stop_heartbeat EXIT

cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
# shellcheck disable=SC1091

# shellcheck disable=SC1090

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
ENTRY=experiments/exp_voxelwisenormative_004_speed_headroom/run.py
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage single-gpu --script "$ENTRY" -- full --gpu --no-profile
