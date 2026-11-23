#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=exp011-reference-n-a100
#SBATCH --array=0-2
#SBATCH --output=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue
# per-refit-a100.sh with a smaller reference: task k runs run.py --per-refit --reference-n N_k, N = 50, 100, 200
# (k = 0, 1, 2), one A100 each, for the same 113 held-out scans. Optional task 3 (sbatch --array=3) is N = 450, a

# per warm fit, 10 s export. Fit cost grows with N, so N <= 200 should take less; not yet measured.
# Expected at most 10.5 min per task, 0.53 GPU hours for the three; 20 min reserved per task, 1.0 GPU hour at most.
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
REFERENCE_NS=(50 100 200 450)
TASK=${SLURM_ARRAY_TASK_ID:?submit as an array, --array=0-2}
N=${REFERENCE_NS[$TASK]:?array task $TASK has no reference size}
echo "array task $TASK: reference n $N, 113 held-out scans" >&2
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage "reference-n-$N" --script "$ENTRY" -- \
    --per-refit --reference-n "$N" --run-name "a100-${SLURM_ARRAY_JOB_ID:?}-n$N" --resources --no-profile
