#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --job-name=exp017-ixi-recover
#SBATCH --array=2,12,42,65,77,97,112-113,125-126,412,425,427,429,434,437,443,447
#SBATCH --output=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.out
#SBATCH --error=experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/.results/slurm-out/%x-%A_%a.err
#SBATCH --no-requeue

# Reuse its resources; the longest successful task took 12:57. No complete subject reruns.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source experiments/exp_voxelwisenormative_017_ixi_synthsr_reference/slurm/common.sh
slurm_prepare_env
manifest="$EXPERIMENT/.intermediates/manifests/full-${IXI_ARRAY_JOB:?}-missing.tsv"
staged="$EXPERIMENT/.results/manifests/staged-heads.json"
printf '%s  %s\n' "${IXI_RECOVERY_SHA256:?}" "$manifest" | sha256sum --check
printf '%s  %s\n' "${IXI_STAGED_SHA256:?}" "$staged" | sha256sum --check
subject=$(awk -F '\t' -v n="${SLURM_ARRAY_TASK_ID:?}" '$1==n {print $2}' "$manifest")
[[ "$subject" =~ ^IXI[0-9]{3}-(Guys|HH|IOP)-[0-9]{4}-T1$ ]]
[[ ! -e "$EXPERIMENT/.results/maps/full/sr/$subject" && ! -e "$EXPERIMENT/.intermediates/full/sr/$subject" ]]
srun --cpu-bind=cores "$SLURM_PROFILE_BIN" profile --stage ixi-recover \
    --script "$EXPERIMENT/run.py" -- --no-profile full \
    --task "${SLURM_ARRAY_TASK_ID:?}" --threads "${SLURM_CPUS_PER_TASK:?}"
