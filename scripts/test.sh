#!/usr/bin/env bash
# Each experiment has its own local `src` package; test each in a process of its own.
set -euo pipefail
cd "$(dirname "$0")/.."
python=${VOXELWISE_PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
status=0
if [[ ${1:-} == --agreement-only ]]; then
    "$python" -B -c 'from importlib.util import find_spec; raise SystemExit(0 if find_spec("pcntoolkit") else "install the pcntoolkit extra to run the paired check")'
    experiment=(experiments/exp_voxelwisenormative_007_*/)
    PYTHONPATH="$PWD/${experiment[0]}tests:$PWD/${experiment[0]}:$PWD" \
        "$python" -B -m pytest -q -s "${experiment[0]}tests/test_gradient_benchmark.py" \
        -k test_two_voxel_jax_agreement
    exit
fi
[[ $# == 0 ]] || exit 2
PYTHONPATH="$PWD" "$python" -B -m pytest -q shared/hugging_face_jobs/tests \
    shared/voxelwisenormative/tests scripts/tests || status=1
for folder in experiments/exp_voxelwisenormative_*/; do
    [[ -d ${folder}tests ]] || continue
    printf '\n%s\n' "${folder%/}"
    PYTHONPATH="$PWD/${folder}tests:$PWD/$folder:$PWD" \
        "$python" -B -m pytest -q -s -rs "${folder}tests" || status=1
done
exit "$status"
