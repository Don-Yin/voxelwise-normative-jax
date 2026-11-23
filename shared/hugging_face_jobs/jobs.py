"""The Hugging Face A100 jobs: one row per job, the Slurm script whose parameters it keeps, and its hf command.

    python shared/hugging_face_jobs/jobs.py --synthetic-input DIR --ixi-input DIR   # the two build_input.py profiles

Prints every command and the most each can cost; nothing here submits anything. The arguments are those of the Slurm
script named in each row; container_entry.sh adds `--no-profile` and replaces @RUN_NAME@. The timeout is the Slurm
`--time` plus 10 minutes (pip install and the copy to /out). `--allow-unpinned` appears for exactly the packages that
requirements-a100.txt still marks UNRESOLVED and the experiment needs; it disappears when the Slurm versions are pinned.
`hf jobs run` takes its own options wherever they stand in the line (`--dry-run` after the image never reaches the
container), so no argument after the image may be one of HF_OPTIONS; tests/test_jobs.py checks that.
"""

from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from shared.hugging_face_jobs.bucket import bucket_uri

IMAGE = "python:3.14.3"  # the Python recorded in the Slurm receipts
FLAVOR, PRICE_PER_HOUR = "a100-large", 2.50  # `hf jobs hardware`, 2026-11-21: 1x A100 80 GB, 12 vCPU, 142 GB
CPU_FLAVOR, CPU_PRICE_PER_HOUR = "cpu-basic", 0.01
EXTRA_MINUTES = 10
DRY_RUN_MINUTES = 15
ENTRY = "/code/shared/hugging_face_jobs/container_entry.sh"
BUCKET_MOUNT = '"${OUTPUT_BUCKET:?set --bucket}":/out'
PREFIX = "hugging-face-a100-"
REQUIREMENTS = Path(__file__).with_name("requirements-a100.txt")
EXP004 = "experiments/exp_voxelwisenormative_004_speed_headroom"
EXP010 = "experiments/exp_voxelwisenormative_010_reference_size_scaling"
EXP011 = "experiments/exp_voxelwisenormative_011_fragile_clusters"
EXP013 = "experiments/exp_voxelwisenormative_013_gpu_headroom"
# The packages each experiment imports whose pin may still be UNRESOLVED; container_entry.sh holds the same table.
NEEDS = {EXP004: {"numpy"}, EXP010: {"numpy"}, EXP011: {"numpy", "nibabel"}, EXP013: {"numpy"}}
# Which build_input.py profile each experiment's job mounts: SYN has no IXI file (1 MiB), IXI adds the two artifacts (361 MiB).
INPUT_OF = {EXP004: "SYN", EXP010: "SYN", EXP011: "IXI", EXP013: "IXI"}
# Every option of `hf jobs run` in huggingface_hub 2.4.0 (`hf jobs run --help`, 2026-11-21).
HF_OPTIONS = frozenset({
    "-e", "--env", "-s", "--secrets", "--name", "-l", "--label", "-v", "--volume", "--env-file", "--secrets-file", "--flavor",
    "--timeout", "--attempts", "-d", "--detach", "--dry-run", "--expose", "--expose-public", "--ssh", "--network-group",
    "--network-alias", "--resource-group-id", "--namespace", "--token", "--format", "--json", "-q", "--quiet", "-h", "--help"})


def unresolved() -> set[str]:
    """Packages whose Slurm version requirements-a100.txt does not yet record."""
    return set(re.findall(r"^# UNRESOLVED (\S+?):?\s", REQUIREMENTS.read_text(), re.MULTILINE))


@dataclass(frozen=True)
class Job:
    """One hf job: run name (also its folder in /out), experiment, script arguments and the Slurm script it copies."""

    group: str
    name: str
    experiment: str
    args: tuple[str, ...]
    slurm_script: str
    slurm_minutes: int
    script: str = "run.py"

    @property
    def timeout_minutes(self) -> int:
        """Slurm `--time` plus the container's own setup."""
        return self.slurm_minutes + EXTRA_MINUTES

    @property
    def max_cost(self) -> float:
        """Dollars if the job runs to its timeout; the job is billed only while running."""
        return self.timeout_minutes / 60 * PRICE_PER_HOUR

    def command(self) -> str:
        """The complete `hf jobs run` command, as a shell snippet using $H, $IN_SYN and $IN_IXI."""
        flags = [f"--run-name {self.name}"]
        if self.script != "run.py":
            flags.append(f"--script {self.script}")
        if allowed := sorted(unresolved() & NEEDS[self.experiment]):
            flags.append(f"--allow-unpinned {','.join(allowed)}")
        args = " ".join(shlex.quote(a) for a in self.args)
        return (f"$H jobs run --flavor {FLAVOR} --timeout {self.timeout_minutes}m -d --name {self.name} \\\n"
                f'  -v "$IN_{INPUT_OF[self.experiment]}":/code -v {BUCKET_MOUNT} \\\n'
                f"  {IMAGE} bash {ENTRY} {' '.join(flags)} \\\n"
                f"  {self.experiment} {args}").rstrip()


def refit_args(*options: str) -> tuple[str, ...]:
    """exp_011 per-refit arguments as in its Slurm scripts."""
    return ("--per-refit", *options, "--run-name", "@RUN_NAME@", "--resources")


FOLDS = f"{EXP011}/slurm/per-refit-folds-a100.sh"
Z_SUMMARY = f"{EXP011}/slurm/z-summary-folds-a100.sh"
CUTOFF_FLAGS = f"{EXP011}/slurm/cutoff-flags-folds-a100.sh"
CUTOFFS = "2.3,2.5,3.0,3.1,3.5"  # --refit-flag-cutoffs: the additional cutoffs above 1.96
REFERENCE_N = f"{EXP011}/slurm/reference-n-a100.sh"
HEADROOM = f"{EXP013}/slurm/gpu-headroom-a100.sh"
LARGE = f"{EXP010}/slurm/benchmark-large-a100.sh"
MECHANISM = f"{EXP004}/slurm/mechanism-timing-a100.sh"
JOBS = (
    Job("1 calibration: exp_004 mechanism timing", f"{PREFIX}mechanism-timing", EXP004,
        ("--output", ".results/json/@RUN_NAME@/mechanism-timing.json"), MECHANISM, 3,
        script="src/modules/mechanism_timing.py"),
    *(Job("2 exp_011 reference-n, the small reference group", f"{PREFIX}reference-n{n}", EXP011,
          refit_args("--reference-n", str(n)), REFERENCE_N, 20) for n in (50, 100, 200)),
    *(Job("3 exp_013 gpu-headroom", f"{PREFIX}gpu-headroom-{precision}", EXP013,
          ("--resources", "sweep", "--precision", precision, "--run-name", "@RUN_NAME@"), HEADROOM, 50)
      for precision in ("float32", "float64")),
    *(Job("4 exp_010 bench-large, synthetic n = 50,000", f"{PREFIX}bench-large-n50000-{precision}", EXP010,
          ("--resources", "gpu", "--index", str(index), "--run-name", "@RUN_NAME@"), LARGE, 55)
      for index, precision in ((22, "float32"), (23, "float64"))),
    *(Job("5 exp_011 per-refit folds", f"{PREFIX}per-refit-fold{fold}", EXP011, refit_args("--fold", str(fold)), FOLDS, 20)
      for fold in (1, 2, 3, 4)),
    Job("5 exp_011 per-refit folds", f"{PREFIX}per-refit-fold0-rerun", EXP011, refit_args("--fold", "0"), FOLDS, 20),
    *(Job("6 exp_011 z summary: the fit without resampling and each voxel's smallest and largest |z| of the 64 refits",
          f"{PREFIX}z-summary-fold{fold}", EXP011, refit_args("--fold", str(fold), "--z-summary"), Z_SUMMARY, 20)
      for fold in range(5)),
    *(Job("7 exp_011 cutoff flags: the z summary and every refit's flags at stricter cutoffs",
          f"{PREFIX}cutoff-flags-fold{fold}", EXP011,
          refit_args("--fold", str(fold), "--z-summary", "--refit-flag-cutoffs", CUTOFFS), CUTOFF_FLAGS, 20)
      for fold in range(5)),
)


def check_command(name: str, input_key: str, experiment: str) -> str:
    """The cpu-basic check of the mounts, the copy, the packages, the imports, the write to /out and the job marker."""
    return (f"$H jobs run --flavor {CPU_FLAVOR} --timeout {DRY_RUN_MINUTES}m -d --name {name} \\\n"
            f'  -v "$IN_{input_key}":/code -v {BUCKET_MOUNT} \\\n'
            f"  {IMAGE} bash {ENTRY} --check-only --run-name {name} {experiment}")


# The first check mounts the synthetic directory; the second the IXI one, with the code that reads the marker; the third
# is the second on the directory with --z-summary, the fourth the same on the directory with --refit-flag-cutoffs (a run
# name is used once: /out/<name> is never overwritten).
CHECKS = ((f"{PREFIX}check-only", "SYN", EXP013), (f"{PREFIX}check-only-ixi", "IXI", EXP011),
          (f"{PREFIX}check-only-z-summary", "IXI", EXP011), (f"{PREFIX}check-only-cutoff-flags", "IXI", EXP011))


def render(synthetic_input: str, ixi_input: str, only: str = "", *, bucket: str | None = None) -> str:
    """Every command grouped, with the most each group can cost; `only` keeps the checks and jobs whose name contains it."""
    cost = DRY_RUN_MINUTES * CPU_PRICE_PER_HOUR / 60
    lines = ['H="uvx --from huggingface_hub hf"', f"IN_SYN={shlex.quote(synthetic_input)}", f"IN_IXI={shlex.quote(ixi_input)}"]
    if bucket is not None:
        lines.append(f"OUTPUT_BUCKET={shlex.quote(bucket_uri(bucket))}")
    lines.append("")
    checks = [check for check in CHECKS if only in check[0]]
    jobs = [job for job in JOBS if only in job.name]
    for name, key, experiment in checks:
        note = (f"# cpu-basic check on the {key} directory (installs, imports, job marker, writes /out; fits nothing): "
                f"at most ${cost:.3f}")
        lines += [note, check_command(name, key, experiment), ""]
    total = 0.0
    for group in dict.fromkeys(job.group for job in jobs):
        members = [job for job in jobs if job.group == group]
        group_cost = sum(job.max_cost for job in members)
        total += group_cost
        lines.append(f"# {group}: {len(members)} job(s), at most ${group_cost:.2f}")
        for job in members:
            lines += [(f"# {job.name}: --timeout {job.timeout_minutes}m = Slurm {job.slurm_minutes} min + {EXTRA_MINUTES}, "
                       f"at most ${job.max_cost:.2f} (copies {job.slurm_script.rsplit('/', 1)[1]})"), job.command()]
        lines.append("")
    lines.append(f"# total, {len(jobs)} {FLAVOR} jobs at their timeouts: ${total:.2f}, plus {len(checks)} cpu-basic checks")
    return "\n".join(lines)


def main() -> None:
    """Print the commands; submit nothing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-input", default="<build_input.py --profile synthetic directory>")
    parser.add_argument("--ixi-input", default="<build_input.py --profile ixi directory>")
    parser.add_argument("--only", default="", help="print only the checks and jobs whose run name contains this, e.g. z-summary")
    parser.add_argument("--bucket", required=True, help="output bucket as namespace/name")
    args = parser.parse_args()
    try:
        text = render(args.synthetic_input, args.ixi_input, args.only, bucket=args.bucket)
    except ValueError as error:
        parser.error(str(error))
    print(text)


if __name__ == "__main__":
    main()
