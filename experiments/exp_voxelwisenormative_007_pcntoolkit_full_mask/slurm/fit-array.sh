#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3270M
#SBATCH --time=00:07:00
#SBATCH --array=0-351%32
#SBATCH --job-name=exp07-pcn-full-mask
#SBATCH --output=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_007_pcntoolkit_full_mask/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# 3,028 MiB cgroup peak. 352 shards of 469–470 columns keep the tested RAM

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
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage fit-mask --script "$ENTRY" -- \
    --no-profile fit-shard --calibration "${PCN_CALIBRATION_JOB:?}" --array "${SLURM_ARRAY_JOB_ID:?}" \
    --index "${SLURM_ARRAY_TASK_ID:?}" --shards 352
