#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:06:00
#SBATCH --job-name=exp017-ixi-audit-crops
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# Read only the 57 pairs flagged by the hash check (0.1 CPU-h).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-audit-crops \
    --script "$EXPERIMENT/run.py" -- --no-profile audit-crops --rebuilt-dir "${IXI_REBUILT_DIR:?}"
