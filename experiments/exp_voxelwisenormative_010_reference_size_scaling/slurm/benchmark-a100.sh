#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --array=0-17%2
#SBATCH --job-name=exp010-bench-a100
#SBATCH --output=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# N<=5000 keeps f64 host input below 7 GB; allow for synthetic generation and four fits.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_010_reference_size_scaling/slurm/runtime.sh
run_case gpu gpu
