#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3270M
#SBATCH --time=00:15:00
#SBATCH --array=15-17%3
#SBATCH --job-name=exp010-pcn-synthetic450
#SBATCH --output=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# N<=10k reduces sample widths to 48 or fewer above N=2k; 15 min is a starting limit.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_010_reference_size_scaling/slurm/runtime.sh
run_case cpu cpu
