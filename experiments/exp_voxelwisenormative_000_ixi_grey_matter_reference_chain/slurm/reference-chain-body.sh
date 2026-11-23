#!/usr/bin/env bash
# Sourced by the reference-chain stage scripts; it never submits jobs.
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ "${SLURM_SUBMIT_DIR:?}" == "$PWD" ]]
export FSLOUTPUTTYPE=NIFTI_GZ
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
export PYTHONPATH="${IXI_REFERENCE_CHAIN:?set the external reference chain}"
[[ "$(tr -d '\r\n' < "${FSLDIR:?set FSLDIR}/etc/fslversion")" == "6.0.7.18" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

VOXELWISE_HEARTBEAT_STARTED_AT=$(date +%s)
(
    while sleep 180; do
        printf '%s HEARTBEAT job=%s elapsed_seconds=%s status=running\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
            "${SLURM_JOB_ID:-unknown}" "$(($(date +%s) - VOXELWISE_HEARTBEAT_STARTED_AT))" >&2
    done
) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

# the reference campaign's SynthStrip was OOM-killed at 3 GB, its calibration peak was 5,376 MiB.
chain_step() {
    local stage="$1" cpus="$2"
    shift 2
    srun --ntasks=1 --cpus-per-task="$cpus" --mem="$((cpus * 3370))M" --exact --cpu-bind=cores \
        "$SLURM_PROFILE_BIN" profile --stage "ixi-reference-$stage" \
        --script experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/run.py -- --no-profile "$@"
}
