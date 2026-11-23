#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6740M
#SBATCH --time=02:00:00
#SBATCH --array=0-46%47
#SBATCH --job-name=ixiref-prepare
#SBATCH --output=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# SynthStrip + FAST + affine GM -> avg152T1_gray for 563 subjects, 47 shards of <=12.

# 12 x 438 s = 1.46 h; 2 h limit leaves 37% margin. 2 CPUs keep the reference chain's
# 6,740 MiB worker share. 47 tasks x 2 CPU x 2 h = 188 CPU-h.
set -euo pipefail
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:?}/experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/slurm/reference-chain-body.sh"
chain_step "prepare" 2 prepare --shard "${SLURM_ARRAY_TASK_ID:?}" --shards 47
