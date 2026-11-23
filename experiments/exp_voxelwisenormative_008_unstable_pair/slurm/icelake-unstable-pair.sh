#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:25:00
#SBATCH --job-name=exp08-unstable-pair
#SBATCH --output=experiments/exp_voxelwisenormative_008_unstable_pair/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_008_unstable_pair/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue
# The previous replay fit 395 columns in one batch for 64 draws.
# This job batches the union of at most 113×48 columns, 768 columns per block.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
set -u

export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
ENTRY=experiments/exp_voxelwisenormative_008_unstable_pair/run.py
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage scan --script "$ENTRY" -- --no-profile --baseline "${BASELINE_Z:?}" --baseline-sha256 "${BASELINE_SHA256:?}" --bootstrap "${BOOTSTRAP_ARRAYS:?}" --receipt "${BOOTSTRAP_RECEIPT:?}"
