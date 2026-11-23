#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --array=22-23%1
#SBATCH --job-name=exp010-bench-large
#SBATCH --output=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# N=450 scales to about 5 min if linear; four passes plus prep, compile and headroom.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_010_reference_size_scaling/slurm/runtime.sh
run_case gpu gpu
