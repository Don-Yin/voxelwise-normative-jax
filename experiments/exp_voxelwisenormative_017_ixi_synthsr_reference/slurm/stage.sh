#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=exp017-ixi-stage
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# One archive download and eight concurrent FSL crops. Maximum reservation: 6 CPU-hours.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
[[ -f "${IXI_T1_ARCHIVE:?stage the archive using scripts/download-ixi.sh and set IXI_T1_ARCHIVE}" ]]
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-stage-heads \
    --script "$EXPERIMENT/run.py" -- --no-profile stage --workers "${SLURM_CPUS_PER_TASK:?}"
