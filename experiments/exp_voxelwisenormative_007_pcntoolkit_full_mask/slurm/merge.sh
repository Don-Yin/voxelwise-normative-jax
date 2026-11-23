#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3270M
#SBATCH --time=00:04:00
#SBATCH --job-name=exp07-pcn-merge
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

set -eo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
ENTRY=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/run.py
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage merge --script "$ENTRY" -- \
    --no-profile merge --calibration "${PCN_CALIBRATION_JOB:?}" --array "${PCN_ARRAY_JOB:?}" --shards 352
