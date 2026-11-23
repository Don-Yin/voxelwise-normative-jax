#!/usr/bin/env bash
# Shell environment for all CPU stages; sourced from the repository root.
set -euo pipefail
umask 077
EXPERIMENT=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference

slurm_fsl_modules() {
    # shellcheck disable=SC1091

    export FSLOUTPUTTYPE=NIFTI_GZ
}

slurm_prepare_env() {
    slurm_fsl_modules
    [[ "$(tr -d '\r\n' < "$FSLDIR/etc/fslversion")" == "6.0.7.18" ]]
    # shellcheck disable=SC1091

    SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
    export SLURM_PROFILE_BIN
    [[ -x "$SLURM_PROFILE_BIN" ]]
    export SLURM_PROFILE_BIN
export PYTHONPATH="${IXI_REFERENCE_CHAIN:?set the external reference chain}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
}

started=$(date +%s)
(
    while sleep 180; do
        printf '%s HEARTBEAT job=%s elapsed_seconds=%s status=running\n' \
            "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${SLURM_JOB_ID:?}" "$(($(date +%s) - started))" >&2
    done
) &
heartbeat=$!
trap 'kill "$heartbeat" 2>/dev/null || true' EXIT
