#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=256120M
#SBATCH --time=01:00:00

#SBATCH --output=experiments/exp_voxelwisenormative_003_batched_jax_fit/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_003_batched_jax_fit/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# warm-up + 3 repeats) timed out in the full stage with about 13 of 76 cores busy, so one repeat here.
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ "${SLURM_SUBMIT_DIR:?}" == "$PWD" ]]
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=76 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRY=experiments/exp_voxelwisenormative_003_batched_jax_fit/run.py
srun --ntasks=1 --cpus-per-task=76 --mem=256120M --exact --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage check --script "$ENTRY" -- check --no-profile
srun --ntasks=1 --cpus-per-task=76 --mem=256120M --exact --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage full --script "$ENTRY" -- full --no-profile --repeats 1
