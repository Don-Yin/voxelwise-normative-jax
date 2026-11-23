#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:08:00
#SBATCH --job-name=exp017-ixi-matrix
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# The local 12-map check took under 1 s; allow time to read all 563 maps on Lustre.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env

recovery=()
if [[ -n ${IXI_RECOVERY_ARRAY:-} ]]; then
    recovery=(--recovery-array "$IXI_RECOVERY_ARRAY" --recovery-sha256 "${IXI_RECOVERY_SHA256:?}")
fi

srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-matrix \
    --script "$EXPERIMENT/run.py" -- --no-profile matrix \
    --array-job "${IXI_ARRAY_JOB:?}" "${recovery[@]}" \
    --workers "${SLURM_CPUS_PER_TASK:?}"
