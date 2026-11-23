#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3370M
#SBATCH --time=02:00:00
#SBATCH --array=0-46%47
#SBATCH --job-name=ixiref-register
#SBATCH --output=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# kamen() against the IXI template for 563 subjects, 47 shards of <=12; any J <= 0 fails a subject.

# 12 x 396 s = 1.32 h; 2 h limit. 47 tasks x 1 CPU x 2 h = 94 CPU-h.
set -euo pipefail
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:?}/experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/slurm/reference-chain-body.sh"
chain_step "register" 1 register --shard "${SLURM_ARRAY_TASK_ID:?}" --shards 47
