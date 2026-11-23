#!/usr/bin/env bash
# shellcheck shell=bash
# Sourced once per array task after cd to the project root; never submitted by itself.

VOXELWISE_HEARTBEAT_SECONDS=180
VOXELWISE_HEARTBEAT_PID=""
VOXELWISE_HEARTBEAT_STARTED_AT=$(date +%s)

voxelwise_start_heartbeat() {
    (
        sleep_pid=""
        trap 'if [[ -n "$sleep_pid" ]]; then kill "$sleep_pid" 2>/dev/null || true; fi; exit 0' HUP INT TERM
        while :; do
            sleep "$VOXELWISE_HEARTBEAT_SECONDS" &
            sleep_pid=$!
            wait "$sleep_pid"
            now=$(date +%s)
            printf '%s HEARTBEAT job=%s elapsed_seconds=%s status=running\n' \
                "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${SLURM_JOB_ID:-unknown}" \
                "$((now - VOXELWISE_HEARTBEAT_STARTED_AT))" >&2
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

run_case() {
    local modules=$1 stage=$2
    local entry="${BASH_SOURCE[0]%/slurm/runtime.sh}/run.py"
    voxelwise_start_heartbeat
    trap voxelwise_stop_heartbeat EXIT
    # shellcheck disable=SC1091

    # shellcheck disable=SC1090

    set -u

    local threads=1 profile="${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}"
    if [[ "$modules" == gpu ]]; then
        export JAX_PLATFORMS=cuda
        threads=4
    fi
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    [[ -x "$profile" ]]
    srun --cpu-bind=cores "$profile" profile --stage "$stage-${SLURM_ARRAY_TASK_ID:?}" --script "$entry" -- \
        --no-profile --resources "$stage" --index "$SLURM_ARRAY_TASK_ID" --run-name "${SLURM_ARRAY_JOB_ID:?}"
}
