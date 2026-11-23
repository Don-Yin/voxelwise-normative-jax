#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=00:08:00
#SBATCH --job-name=exp017-ixi-verify-stage
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-verify-heads \
    --script "$EXPERIMENT/run.py" -- --no-profile stage \
    --workers "${SLURM_CPUS_PER_TASK:?}" --repair-from "${IXI_REPAIR_FROM:?}"
