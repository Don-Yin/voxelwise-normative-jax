#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3270M
#SBATCH --time=00:35:00
#SBATCH --array=30-35%3
#SBATCH --job-name=exp010-pcn-large
#SBATCH --output=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_010_reference_size_scaling/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# gives about 17 min for 24 voxels at N=50k plus startup and prediction; calibrate here.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_010_reference_size_scaling/slurm/runtime.sh
run_case cpu cpu
