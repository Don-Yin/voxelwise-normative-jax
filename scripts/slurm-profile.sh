#!/usr/bin/env bash
# Profile the Python worker started by each Slurm step. Run from the project root.
set -euo pipefail
[[ ${1-} == profile ]] || { printf 'usage: slurm-profile.sh profile --stage NAME --script PATH -- ARGS\n' >&2; exit 2; }
shift
stage='' kind='' entry=''
while (($#)); do
    case $1 in
        --stage) stage=${2:?missing stage}; shift 2 ;;
        --script|--module) kind=$1; entry=${2:?missing entry}; shift 2 ;;
        --) shift; break ;;
        *) printf 'unexpected profiler option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $stage =~ ^[a-zA-Z0-9_-]+$ && -n $entry ]]
python=${VOXELWISE_PYTHON:-python}
profile_root=${VOXELWISE_PROFILE_ROOT:-}
if [[ $kind == --script && $entry =~ ^(experiments/exp_[^/]+)/ ]]; then
    export PYTHONPATH="$PWD/${BASH_REMATCH[1]}:$PWD${PYTHONPATH:+:$PYTHONPATH}"
    profile_root="$PWD/${BASH_REMATCH[1]}/.results/profile"
fi
if [[ -z $profile_root ]]; then
    printf 'set VOXELWISE_PROFILE_ROOT for module profiling\n' >&2
    exit 2
fi
mkdir -p "$profile_root"
output=$(mktemp "$profile_root/$stage-${SLURM_JOB_ID:-local}-XXXXXXXX")
# Let the Python profiler render its normal HTML report using a file extension.
mv "$output" "$output.html"
if [[ $kind == --module ]]; then
    exec "$python" -B -m pyinstrument -r html -o "$output.html" -m "$entry" "$@"
fi
exec "$python" -B -m pyinstrument -r html -o "$output.html" "$entry" "$@"
