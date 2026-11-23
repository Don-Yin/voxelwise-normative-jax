#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --job-name=exp004-mechanism-timing
#SBATCH --output=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
# shellcheck disable=SC1091

# shellcheck disable=SC1090

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exp=experiments/exp_voxelwisenormative_004_speed_headroom
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage mechanism-timing --script "$exp/src/modules/mechanism_timing.py" -- \
  --no-profile  \
  --output "$exp/.results/json/job-$SLURM_JOB_ID/mechanism-timing.json"
