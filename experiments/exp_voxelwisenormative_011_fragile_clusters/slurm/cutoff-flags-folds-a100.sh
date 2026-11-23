#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=exp011-cutoff-flags-folds-a100
#SBATCH --array=0-4
#SBATCH --output=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_011_fragile_clusters/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue
# z-summary-folds-a100.sh with --refit-flag-cutoffs: the same 5 folds and z-summary.npz, plus cutoff-flags.npz holding, at
# each cutoff, every one of the 64 refits' flags (abs(z) > T, bit-packed like refit-flags.npz) and the single fit's, so a
# page or a cluster statistic can use any cutoff for one refit as well as for 'kept in all 64' (about 750 MB per fold
# before compression). The job stops, writing nothing, unless per cutoff the AND of the refits' flags is refit_min_abs_z > T
# and their OR refit_max_abs_z > T, and the flags at 1.96 that the same code makes equal the run's own. About 11 min per fold
# (z-summary folds: 624 to 666 s running) plus about 1 min for the flags; 20 min reserved per task (1.7 GPU hours at most).

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
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage "cutoff-flags-fold-$FOLD" --script "$ENTRY" -- \
    --per-refit --fold "$FOLD" --z-summary --refit-flag-cutoffs 2.3,2.5,3.0,3.1,3.5 \
    --run-name "a100-${SLURM_ARRAY_JOB_ID:?}-cutoff-fold$FOLD" \
    --resources --no-profile
