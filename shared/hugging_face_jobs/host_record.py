"""Host record of a Hugging Face job: GPU and driver, CPU and vCPUs, memory, Python and package versions.

    python host_record.py start <file> --run-name N --experiment E --entry-args "..." [--dry-run] [--unpinned a,b]
    python host_record.py finish <file> --exit-code C

`start` runs after the packages are installed and, unless --dry-run, refuses any machine whose JAX does not see an
A100, before the job spends its hours. `finish` adds the end time and the exit code of run.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)
PACKAGES = ("jax", "jaxlib", "numpy", "scipy", "nibabel", "ml_dtypes", "opt_einsum", "jax-cuda12-plugin",
            "jax-cuda12-pjrt", "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cusolver-cu12",
            "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-nvjitlink-cu12")
ENV_KEYS = ("JOB_ID", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CPU_CORES", "XLA_PYTHON_CLIENT_MEM_FRACTION")
INVENTORY = Path(__file__).resolve().parents[2] / "INVENTORY.tsv"
FIELDS = "index,name,driver_version,memory.total,clocks.max.sm,power.limit"


def run(*command: str) -> str | None:
    """Standard output of a bounded command, or None when it is absent or fails."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def gpu() -> dict:
    """nvidia-smi's own words: devices, driver, and the CUDA version of its header."""
    table = run("nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits")
    if table is None:
        return {"available": False}
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", run("nvidia-smi") or "")
    devices = [dict(zip(FIELDS.split(","), (cell.strip() for cell in line.split(",")), strict=True))
               for line in table.strip().splitlines()]
    return {"available": True, "devices": devices, "nvidia_smi_cuda_version": cuda.group(1) if cuda else None}


def read_lines(path: str) -> list[str]:
    """Lines of a proc or cgroup file; empty when it does not exist."""
    try:
        return Path(path).read_text().splitlines()
    except OSError:
        return []


def cpu() -> dict:
    """CPU model, logical CPUs seen, those this process may use, memory, and the cgroup limits."""
    model = next((line.split(":", 1)[1].strip() for line in read_lines("/proc/cpuinfo")
                  if line.startswith("model name")), platform.processor() or None)
    total = next((int(line.split()[1]) * 1024 for line in read_lines("/proc/meminfo") if line.startswith("MemTotal")), None)
    affinity = getattr(os, "sched_getaffinity", None)  # absent on macOS
    return {"model": model, "vcpus_visible": os.cpu_count(), "vcpus_usable": len(affinity(0)) if affinity else None,
            "mem_total_bytes": total, "cgroup_cpu_max": next(iter(read_lines("/sys/fs/cgroup/cpu.max")), None),
            "cgroup_memory_max": next(iter(read_lines("/sys/fs/cgroup/memory.max")), None), "machine": platform.machine()}


def versions() -> dict:
    """Installed versions of the packages the jobs depend on; None when a package is not installed."""
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def code_hashes() -> dict:
    """sha256 of every code file this job ran, from the verified inventory of the copied input directory."""
    if not INVENTORY.is_file():
        return {"inventory": None}
    rows = [line.split("\t") for line in INVENTORY.read_text().splitlines()[1:]]
    return {"inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
            "files": {path: sha for path, _, sha, role in rows if role in ("code", "hugging-face")}}


def jax_view() -> dict:
    """What JAX itself reports: backend and devices (importing it starts no computation)."""
    try:
        import jax
    except ImportError as error:
        return {"error": f"cannot import jax: {error}"}
    try:
        devices = jax.devices()
        return {"version": jax.__version__, "default_backend": jax.default_backend(),
                "devices": [{"platform": d.platform, "kind": d.device_kind} for d in devices]}
    except Exception as error:  # noqa: BLE001 - any backend failure is the record's finding, not a crash
        return {"version": jax.__version__, "error": f"jax.devices() failed: {type(error).__name__}: {error}"}


def start(args: argparse.Namespace) -> None:
    """Write the record; a real job needs an A100 that JAX can use."""
    seen = jax_view()
    if not args.dry_run:
        devices = seen.get("devices", [])
        if not devices or devices[0]["platform"] != "gpu" or "A100" not in devices[0]["kind"]:
            raise RuntimeError(f"this job requires an NVIDIA A100 that JAX can use, JAX reports {seen}")
    installed = versions()
    freeze = "".join(f"{name}=={version}\n" for name, version in installed.items() if version)
    args.file.with_name("pip-freeze.txt").write_text(freeze)
    record = {"schema_version": 1, "run_name": args.run_name, "experiment": args.experiment, "entry_args": args.entry_args,
              "dry_run": args.dry_run, "started_utc": datetime.now(UTC).isoformat(), "gpu": gpu(), "cpu": cpu(),
              "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                         "executable": Path(sys.executable).name, "prefix": Path(sys.prefix).name,
                         "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED"))},
              "packages": installed, "jax_reports": seen, "pip_freeze_file": "pip-freeze.txt",
              "unpinned_in_dry_run": [name for name in args.unpinned.split(",") if name],
              "environment": {key: value for key in ENV_KEYS if (value := os.environ.get(key))
                              and re.fullmatch(r"[a-z0-9.,-]{1,40}", value)},
              "code_sha256": code_hashes(),
              # No Slurm here: every SLURM_* variable must be absent. The marker is the platform's own JOB_ID, set by
              # container_entry.sh, which the real-IXI guards accept instead of a Slurm id.
              "slurm_variables": sorted(key for key in os.environ if key.startswith("SLURM_")),
              "platform_job_id": os.environ.get("JOB_ID"), "hugging_face_job": os.environ.get("HUGGING_FACE_JOB_ID")}
    args.file.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    logger.info("host record %s: gpu=%s cpu=%s vcpus=%s", args.file, [d["name"] for d in record["gpu"].get("devices", [])],
                record["cpu"]["model"], record["cpu"]["vcpus_usable"])


def finish(args: argparse.Namespace) -> None:
    """Add the end time, the elapsed seconds and run.py's exit code."""
    record = json.loads(args.file.read_text())
    end = datetime.now(UTC)
    record.update({"finished_utc": end.isoformat(), "exit_code": args.exit_code,
                   "elapsed_seconds": (end - datetime.fromisoformat(record["started_utc"])).total_seconds()})
    args.file.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Dispatch the two subcommands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    first = sub.add_parser("start")
    first.add_argument("file", type=Path)
    first.add_argument("--run-name", required=True)
    first.add_argument("--experiment", required=True)
    first.add_argument("--entry-args", required=True)
    first.add_argument("--dry-run", action="store_true")
    first.add_argument("--unpinned", default="")
    last = sub.add_parser("finish")
    last.add_argument("file", type=Path)
    last.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    (start if args.stage == "start" else finish)(args)


if __name__ == "__main__":
    main()
