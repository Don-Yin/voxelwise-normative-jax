#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=exp006-voxel-scaling-a100
#SBATCH --output=experiments/exp_voxelwisenormative_006_speed_memory_scaling/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_006_speed_memory_scaling/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue
# Nine voxel counts x two fits, each in a fresh process so the device allocator peak belongs to one case.

# about 45 s of startup each plus about 2 min of fitting, so 30 min leaves about 1.8x headroom.

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
ENTRY=experiments/exp_voxelwisenormative_006_speed_memory_scaling/run.py
for n in 1000 2000 5000 10000 20000 32768 50000 100000 165402; do
    for variant in early-f32 baseline; do
        srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage "$variant-n$n" --script "$ENTRY" -- \
            gpu --gpu --no-profile --variant "$variant" --n-voxels "$n"
    done
done
