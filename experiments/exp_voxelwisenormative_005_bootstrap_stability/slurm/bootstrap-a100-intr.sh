#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --qos=intr
#SBATCH --job-name=exp005-bootstrap-intr
#SBATCH --output=experiments/exp_voxelwisenormative_005_bootstrap_stability/.results/slurm-out/%x-%j.out
#SBATCH --error=experiments/exp_voxelwisenormative_005_bootstrap_stability/.results/slurm-out/%x-%j.err
#SBATCH --no-requeue
# Same calibrated workload as bootstrap-a100.sh; the shorter-priority queue
# is explicitly for rehearsing the live GPU demo. Do not reserve a second GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
# Execute the calibrated stage under this existing allocation, not as a new job.
exec bash experiments/exp_voxelwisenormative_005_bootstrap_stability/slurm/bootstrap-a100.sh
