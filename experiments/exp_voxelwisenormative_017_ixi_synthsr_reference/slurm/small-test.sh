#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --job-name=exp017-ixi-test
#SBATCH --array=0-14%12
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# Three exact baseline repeats and twelve first-time full-field SynthSR controls.
# 4 CPUs and 24 GiB give room to measure the installed dense 3D network before sizing production.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-small-test \
    --script "$EXPERIMENT/run.py" -- --no-profile small-test \
    --task "${SLURM_ARRAY_TASK_ID:?}" --threads "${SLURM_CPUS_PER_TASK:?}"
