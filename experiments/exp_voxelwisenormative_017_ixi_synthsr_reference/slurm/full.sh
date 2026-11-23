#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --job-name=exp017-ixi-full
#SBATCH --array=0-562%50
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# Its longest Slurm elapsed time was 13:54; 15% extra rounds to 16 minutes.
# 563 x 4 x 16/60 = 600.53 CPU-hours charged for the requested time.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
staged="$EXPERIMENT/.results/manifests/staged-heads.json"
printf '%s  %s\n' "${IXI_STAGED_SHA256:?}" "$staged" | sha256sum --check
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-full \
    --script "$EXPERIMENT/run.py" -- --no-profile full \
    --task "${SLURM_ARRAY_TASK_ID:?}" --threads "${SLURM_CPUS_PER_TASK:?}"
