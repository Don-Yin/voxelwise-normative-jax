#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=215680M
#SBATCH --time=02:00:00
#SBATCH --job-name=ixiref-template
#SBATCH --output=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# IXI-only study template (fslvbm_2_template -n) from all 563 prepared subjects.
# Second-pass fnirt per subject: calibration kamen fsl_reg took 217-395 s at 745 MiB, one core.
# 563 x 395 s / 64 workers = 58 min worst case, plus two 563-volume merges; 2 h limit.
# 64 CPU x 2 h = 128 CPU-h.
set -euo pipefail
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:?}/experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/slurm/reference-chain-body.sh"
chain_step "template" 64 template --workers 64
