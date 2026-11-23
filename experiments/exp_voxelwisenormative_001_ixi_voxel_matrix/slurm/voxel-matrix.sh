#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6740M
#SBATCH --time=01:00:00
#SBATCH --job-name=ixi-voxel-matrix
#SBATCH --output=experiments/exp_voxelwisenormative_001_ixi_voxel_matrix/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_001_ixi_voxel_matrix/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# 563 published IXI maps -> one float32 subjects x voxels matrix (~0.5 GB) with mask and covariates.
# Reads ~1 GB of gzipped NIfTI and hashes every file. 2 CPU x 1 h = 2 CPU-h.
set -euo pipefail
umask 007
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ "${SLURM_SUBMIT_DIR:?}" == "$PWD" ]]
# Cheap guard: the published subject table.
TABLE=${IXI_DERIVATIVES:?set the derivative directory}/subjects.tsv
TABLE_SHA256=${IXI_SUBJECTS_SHA256:?set the manifest digest}
[[ $(sha256sum < "$TABLE" | cut -d' ' -f1) == "$TABLE_SHA256" ]]
SLURM_PROFILE_BIN=${SLURM_PROFILE_BIN:-$PWD/scripts/slurm-profile.sh}
export SLURM_PROFILE_BIN
[[ -x "$SLURM_PROFILE_BIN" ]]
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRY=experiments/exp_voxelwisenormative_001_ixi_voxel_matrix/run.py
srun --ntasks=1 --cpus-per-task=2 --mem=6740M --exact --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-voxel-matrix --script "$ENTRY" -- --no-profile --derivatives "${IXI_DERIVATIVES:?}" --subjects-sha256 "$TABLE_SHA256" --template-sha256 "${IXI_TEMPLATE_SHA256:?}"
