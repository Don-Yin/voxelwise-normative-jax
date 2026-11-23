#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --job-name=exp004-projection-timing
#SBATCH --output=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_004_speed_headroom/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# This probe imports the same stack and times only one projection on device-resident inputs.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
# shellcheck disable=SC1091

# shellcheck disable=SC1090

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exp=experiments/exp_voxelwisenormative_004_speed_headroom
source_sha=2dd3c995c872c8cdc7409b6602c7778fc96095f1de1ff51a723cff3ff8d31aba
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage projection-timing --script "$exp/src/modules/projection_timing.py" -- \
  --no-profile --input "$exp/.intermediates/mechanism-timing/$source_sha.npz" --sha256 "$source_sha" \
  --output "$exp/.results/json/job-$SLURM_JOB_ID/projection-timing.json"
