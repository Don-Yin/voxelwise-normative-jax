#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=26960M
#SBATCH --time=01:00:00
#SBATCH --job-name=ixiref-calibration
#SBATCH --output=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue

# Four subjects (Guys, HH, IOP and IXI058, the old chain's fold case), one 2-CPU step each:
# SynthStrip + FAST + affine GM -> avg152T1_gray + kamen() against the MNI152 GM prior.
# Measures per-stage seconds and peak RSS to size the 563-subject arrays. 8 CPU x 1 h = 8 CPU-h.
set -euo pipefail
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:?}/experiments/exp_voxelwisenormative_000_ixi_grey_matter_reference_chain/slurm/reference-chain-body.sh"
pids=()
for rank in 0 1 2 3; do
    chain_step "calibration-$rank" 2 calibrate --shard "$rank" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
