#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3370M
#SBATCH --time=00:20:00
#SBATCH --job-name=exp07-pcn-stops
#SBATCH --output=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# This job fits 10 voxels in 7 conditions. Its worst case is 70 fits stopped only by SciPy's
# 15,000-evaluation limit (about 0.6 ms each, 11 min) plus the same start-up, so 20 min is enough.
set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
ENTRY=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/run.py
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage stopping-sweep --script "$ENTRY" -- \
    --no-profile stopping-sweep --array "${PCN_ARRAY_JOB:?}" --merge "${PCN_MERGE_JOB:?}" --calibration "${PCN_CALIBRATION_JOB:?}" --audit "${PCN_AUDIT_JOB:?}"
