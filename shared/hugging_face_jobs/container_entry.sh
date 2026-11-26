#!/usr/bin/env bash
# shellcheck shell=bash
# The shared in-container entry of the Hugging Face A100 jobs.
#
#   container_entry.sh [--check-only] [--allow-unpinned a,b] --run-name NAME [--script PATH] <experiment-dir> [args...]
#
# NAME must start with hugging-face-a100-, so these results never mix with Slurm's a100-<jobid> ones. The script
# (default run.py) runs inside the experiment directory as `python SCRIPT --no-profile ARGS`; @RUN_NAME@ in ARGS becomes
# NAME. HF_PROFILE_RUN=1 keeps the default profiler and installs pyinstrument 5.1.3; exp_018 and 019 default to it. Nothing goes
# through slurm-profile. Mounts: /code (read-only, build_input.py's directory) and /out (the
# read-write bucket); the jobs run on a writable copy of /code, and the copy's .results/ is copied to /out/NAME/.
# --check-only installs the packages and imports every experiment but fits nothing; it is the cpu-basic check. It is not
# called --dry-run because `hf jobs run` takes that flag for itself wherever it stands and would not pass it on.
# The environment variables of each experiment are those of its Slurm script; the thread count is the CPUs the container
# has where Slurm used SLURM_CPUS_PER_TASK. HF_LOCAL_TEST=1 (tests on a Mac only) skips the venv, pip and GPU checks.
# A Hugging Face job has no Slurm id and none is made up: any SLURM_* variable stops the entry. The guards of real IXI input
# (shared/voxelwisenormative/batch_job.py) accept the marker HUGGING_FACE_JOB_ID instead, which this script sets from the
# platform's own JOB_ID when that is 24 lowercase hexadecimal characters; nothing else sets it.
set -euo pipefail

CODE=${HF_CODE_DIR:-/code}
OUT=${HF_OUT_DIR:-/out}
WORK=${HF_WORK_DIR:-/work}
VENV=${HF_VENV_DIR:-/opt/hugging-face-a100-venv}
PREFIX=${HF_RUN_PREFIX:-hugging-face-a100-}
LOCAL_TEST=${HF_LOCAL_TEST:-0}
BASE_PYTHON=${HF_PYTHON:-python3}
HEARTBEAT_SECONDS=180
STARTED=$(date +%s)
step=start
copied=0
heartbeat_pid=""

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '%s INFO %s\n' "$(stamp)" "$*" >&2; }
die() { printf '%s ERROR %s\n' "$(stamp)" "$*" >&2; exit 2; }

start_heartbeat() {
    (
        trap 'kill "$sleep_pid" 2>/dev/null || true; exit 0' HUP INT TERM
        while :; do
            sleep "$HEARTBEAT_SECONDS" &
            sleep_pid=$!
            wait "$sleep_pid"
            memory=""
            [[ -r /sys/fs/cgroup/memory.current ]] && memory=" cgroup_memory_mib=$(($(< /sys/fs/cgroup/memory.current) / 1048576))"
            [[ -r /sys/fs/cgroup/memory.peak ]] && memory+=" cgroup_peak_mib=$(($(< /sys/fs/cgroup/memory.peak) / 1048576))"
            printf '%s HEARTBEAT run=%s elapsed_seconds=%s step=%s%s\n' "$(stamp)" "${run_name:-unknown}" \
                "$(($(date +%s) - STARTED))" "$step" "$memory" >&2
        done
    ) &
    heartbeat_pid=$!
}

# shellcheck disable=SC2329 # called through the EXIT trap below
on_exit() {
    local status=$1
    if [[ -n $heartbeat_pid ]]; then
        kill "$heartbeat_pid" 2>/dev/null || true
        wait "$heartbeat_pid" 2>/dev/null || true
    fi
    if ((status != 0 && copied == 0)) && [[ -n ${run_name:-} && -d $OUT && ! -e $OUT/$run_name ]]; then
        mkdir -p "$OUT/$run_name" && printf 'entry stopped with status %s in step %s\n' "$status" "$step" > "$OUT/$run_name/ENTRY-FAILED.txt"
    fi
    log "entry finished with status $status after $(($(date +%s) - STARTED)) s"
}
trap 'on_exit $?' EXIT

dry_run=0 allow_unpinned="" run_name="" script=run.py
while (($#)); do
    case $1 in
        --check-only) dry_run=1; shift ;;
        --allow-unpinned) allow_unpinned=${2:?--allow-unpinned needs a comma-separated list}; shift 2 ;;
        --run-name) run_name=${2:?--run-name needs a value}; shift 2 ;;
        --script) script=${2:?--script needs a path}; shift 2 ;;
        --) shift; break ;;
        -*) die "unknown option $1" ;;
        *) break ;;
    esac
done
experiment=${1:?usage: container_entry.sh [--check-only] --run-name NAME [--script PATH] <experiment-dir> [args...]}
shift
args=("$@")
experiment=${experiment#experiments/}
experiment=${experiment%/}
[[ $experiment =~ ^exp_voxelwisenormative_[0-9]{3}_[a-z0-9_]+$ ]] || die "not an experiment directory name: $experiment"
[[ $run_name == "$PREFIX"* && $run_name =~ ^[a-z0-9][a-z0-9-]*$ && ${#run_name} -le 64 ]] \
    || die "run name '$run_name' must start with $PREFIX and use at most 64 lowercase letters, digits and hyphens"
[[ $script =~ ^[A-Za-z0-9_./-]+$ && $script != *..* ]] || die "script must be a relative path inside the experiment: $script"
args=("${args[@]//@RUN_NAME@/$run_name}")

slurm_variables=$(compgen -e | grep '^SLURM_' | tr '\n' ' ' || true)
[[ -z $slurm_variables ]] || die "refusing to run with ${slurm_variables% } set: a Hugging Face job has no Slurm id and none may be made up"
unset HUGGING_FACE_JOB_ID
if [[ ${JOB_ID:-} =~ ^[0-9a-f]{24}$ ]]; then export HUGGING_FACE_JOB_ID=$JOB_ID; fi

# The environment of each experiment's Slurm script (see the file named on each row).
# `needs` are the packages the experiment imports whose pin requirements-a100.txt may still mark UNRESOLVED.
# Threads: the CPUs this container may use, as the Slurm scripts used SLURM_CPUS_PER_TASK. nproc says 96 on an a100-large whose
# cgroup allows 12 (hugging-face-a100-mechanism-timing host record), so the platform's CPU_CORES comes first, then the cgroup quota.
cpus=${CPU_CORES:-}
if [[ ! $cpus =~ ^[1-9][0-9]*$ ]]; then
    quota=max period=""
    { read -r quota period < /sys/fs/cgroup/cpu.max; } 2> /dev/null || true
    if [[ $quota =~ ^[0-9]+$ && $period =~ ^[1-9][0-9]*$ ]]; then
        cpus=$(((quota + period - 1) / period))
    else
        cpus=$(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN)
    fi
fi
threads=$cpus
expected_gpus=1
profile_default=0
ixi=0 # 1: the job always opens real IXI input, so a real run needs the marker
unset JAX_PLATFORMS XLA_PYTHON_CLIENT_MEM_FRACTION
case $experiment in
    exp_voxelwisenormative_004_speed_headroom) threads=1; needs=" numpy " ;; # slurm/mechanism-timing-a100.sh: all four thread variables 1
    exp_voxelwisenormative_010_reference_size_scaling) threads=4; needs=" numpy "; export JAX_PLATFORMS=cuda ;; # slurm/runtime.sh, GPU case
    exp_voxelwisenormative_011_fragile_clusters)
        needs=" numpy nibabel "; ixi=1; export JAX_PLATFORMS=cuda ;;
    exp_voxelwisenormative_013_gpu_headroom) needs=" numpy "; ixi=1; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 ;; # gpu-headroom-a100.sh
    exp_voxelwisenormative_018_four_gpu_refits)
        threads=1; needs=" numpy scipy "; ixi=1; expected_gpus=4; profile_default=1; export JAX_PLATFORMS=cuda ;;
    exp_voxelwisenormative_019_zbin_stability)
        threads=4; needs=" numpy scipy "; ixi=1; profile_default=1; export JAX_PLATFORMS=cuda ;;
    *) die "no environment row for $experiment" ;;
esac
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads NUMEXPR_NUM_THREADS=$threads
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

log "run $run_name: $experiment/$script ${args[*]:-} (dry run: $dry_run, threads: $threads)"
step=preflight
[[ -d $CODE/experiments/$experiment && -f $CODE/INVENTORY.tsv ]] || die "$CODE is not a build_input.py directory with $experiment"
[[ -f $CODE/experiments/$experiment/$script ]] || die "$CODE/experiments/$experiment/$script does not exist"
[[ -d $OUT && -w $OUT ]] || die "$OUT is not a writable mount"
[[ ! -e $OUT/$run_name ]] || die "refusing to overwrite $OUT/$run_name"
if ((LOCAL_TEST == 0 && dry_run == 0)); then
    gpus=$(nvidia-smi -L) || die "nvidia-smi failed: this job needs the A100"
    [[ $(grep -c '^GPU .*A100' <<< "$gpus") -eq $expected_gpus && $(wc -l <<< "$gpus") -eq $expected_gpus ]] \
        || die "expected exactly $expected_gpus A100 GPU(s), nvidia-smi -L says: $gpus"
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        CUDA_VISIBLE_DEVICES=0
        for ((index = 1; index < expected_gpus; index++)); do CUDA_VISIBLE_DEVICES+=,$index; done
    fi
    export CUDA_VISIBLE_DEVICES
    if ((ixi)); then
        [[ -n ${HUGGING_FACE_JOB_ID:-} ]] || die "this job opens real IXI input, which its guard allows only with the Hugging Face job marker; JOB_ID is '${JOB_ID:-}', not 24 lowercase hexadecimal characters"
    fi
fi
log "Hugging Face job marker: ${HUGGING_FACE_JOB_ID:-none} (platform JOB_ID '${JOB_ID:-}')"
start_heartbeat

step=copy
SECONDS=0
rm -rf "${WORK:?}/code" "$WORK/meta"
mkdir -p "$WORK/meta"
cp -R "$CODE" "$WORK/code"
chmod -R u+w "$WORK/code"
"$BASE_PYTHON" "$WORK/code/shared/hugging_face_jobs/build_input.py" --verify "$WORK/code" --quiet
log "copied and verified $CODE against INVENTORY.tsv in $SECONDS s"

step=install
mapfile -t pins < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' -e 's/[[:space:]]*#.*//' "$WORK/code/shared/hugging_face_jobs/requirements-a100.txt")
run_flags=(--no-profile)
if [[ ${HF_PROFILE_RUN:-$profile_default} == 1 ]]; then
    pins+=(pyinstrument==5.1.3)
    run_flags=()
fi
mapfile -t unresolved < <(sed -n 's/^# UNRESOLVED \([A-Za-z0-9_.-]*\).*/\1/p' "$WORK/code/shared/hugging_face_jobs/requirements-a100.txt")
unpinned=()
for name in "${unresolved[@]}"; do
    if ((dry_run)) || [[ ,$allow_unpinned, == *,"$name",* ]]; then
        unpinned+=("$name")
    elif [[ $needs == *" $name "* ]]; then
        die "the Slurm version of $name is not recorded in requirements-a100.txt; pin it there, or pass --allow-unpinned $name"
    fi
done
if ((LOCAL_TEST)); then
    PY=$BASE_PYTHON
    log "HF_LOCAL_TEST=1: no venv, no pip, no GPU checks; using $PY"
else
    "$BASE_PYTHON" -m venv "$VENV"
    PY=$VENV/bin/python
    SECONDS=0
    "$PY" -m pip install --quiet --only-binary=:all: "${pins[@]}" "${unpinned[@]}"
    log "pip installed ${pins[*]} ${unpinned[*]:-} in $SECONDS s"
fi

step=host-record
export PYTHONPATH=$WORK/code${PYTHONPATH:+:$PYTHONPATH}
host_flags=() host_env=()
if ((dry_run || LOCAL_TEST)); then host_flags+=(--dry-run) host_env+=(JAX_PLATFORMS=cpu); fi # no GPU here: JAX on CPU
env "${host_env[@]}" "$PY" "$WORK/code/shared/hugging_face_jobs/host_record.py" start "$WORK/meta/hugging-face-host.json" \
    --run-name "$run_name" --experiment "$experiment" --entry-args "$script ${args[*]:-}" \
    --unpinned "$(IFS=,; echo "${unpinned[*]:-}")" "${host_flags[@]}"

results=$WORK/code/experiments/$experiment/.results
status=0
if ((dry_run)); then
    step=dry-run
    if ((LOCAL_TEST == 0)); then # the marker and the environment name the guards expect, in the real container
        "$PY" - <<'MARKER' || { log "the Hugging Face job marker check failed"; status=1; }
import pathlib
import sys

from shared.voxelwisenormative.batch_job import batch_job, expected_environment

job = batch_job()
assert job is not None and job.platform == "hugging_face", f"no valid Hugging Face job marker: {job}"
name = pathlib.Path(sys.prefix).name
assert name == expected_environment(), f"environment {name} is not the {expected_environment()} the guards expect"
print(f"marker ok: Hugging Face job {job.job_id}, environment {name}", file=sys.stderr)
MARKER
    fi
    for folder in "$WORK"/code/experiments/exp_*/; do
        (
            cd "$folder"
            for entry in run.py src/modules/mechanism_timing.py; do
                if [[ -f $entry ]]; then "$PY" "$entry" --help > /dev/null; fi
            done
            "$PY" - <<'IMPORTS'
import importlib
import pathlib
import sys

folder = pathlib.Path.cwd()
sys.path[:0] = [str(folder), str(folder.parents[1])]
for path in sorted((folder / "src" / "modules").glob("*.py")):
    if path.stem != "__init__":
        importlib.import_module(f"src.modules.{path.stem}")
        print(f"imported {folder.name} src.modules.{path.stem}", file=sys.stderr)
IMPORTS
        ) || { log "dry run failed in $folder"; status=1; }
    done
else
    step=run
    cd "$WORK/code/experiments/$experiment"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    set +e
    "$PY" "$script" "${run_flags[@]}" "${args[@]}" 2>&1 | tee "$WORK/run.log"
    status=${PIPESTATUS[0]}
    set -e
    log "$script exited with status $status"
fi

step=copy-out
"$PY" "$WORK/code/shared/hugging_face_jobs/host_record.py" finish "$WORK/meta/hugging-face-host.json" --exit-code "$status"
mkdir -p "$results/json/$run_name" "$results/logs"
cp "$WORK"/meta/* "$results/json/$run_name/"
if [[ -f $WORK/run.log ]]; then cp "$WORK/run.log" "$results/logs/$run_name.log"; fi
mkdir -p "$OUT/$run_name"
cp -R "$results/." "$OUT/$run_name/"
copied=1
log "wrote $OUT/$run_name:"
(cd "$OUT/$run_name" && find . -type f -exec wc -c {} + | sort -k2 >&2)
exit "$status"
