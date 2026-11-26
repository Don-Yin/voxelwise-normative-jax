"""Stage an explicit file allowlist and checksum inventory for a Hugging Face job.

The default stages code only. Experiments 004 and 010 generate synthetic arrays in memory.
Select an IXI profile to stage a locally prepared, manifest-checked cohort and reference.
The 018 and 019 profiles mount only their experiment and omit the mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = "experiments"
CODE_PREFIX = "experiments/exp_voxelwisenormative_"
UTILS = ("src/__init__.py", "src/utils/__init__.py", "src/utils/paths.py", "src/utils/profiling.py",
         "src/modules/__init__.py")
CODE = {
    "004_speed_headroom": (*UTILS, "src/modules/mechanism_timing.py"),
    "010_reference_size_scaling": ("run.py", *UTILS, "src/modules/gpu.py"),
    "011_fragile_clusters": ("run.py", *UTILS, "src/modules/clusters.py", "src/modules/inputs.py",
                             "src/modules/publish.py", "src/modules/reference_size.py", "src/modules/study.py",
                             "src/modules/z_summary.py", "src/modules/reader-facing-axial-grid.json"),
    "013_gpu_headroom": ("run.py", *UTILS, "src/utils/receipts.py", "src/modules/gpu_samples.py",
                         "src/modules/sweep.py"),
}
KERNEL = ("__init__", "artifacts", "basis", "batch_job", "batched_blr", "cohort", "early_newton", "receipts")
REFIT = ("refit_batch", "refit_inputs", "refit_trace")
REFERENCE = ("reference_bootstrap", "reference_cohort", "reference_fit", "reference_grid")
SHARED_CODE = (*KERNEL, *REFIT, *REFERENCE, "bootstrap_score", "ixi_folds", "template_grid")
JOB_FILES = ("shared/hugging_face_jobs/container_entry.sh", "shared/hugging_face_jobs/host_record.py",
             "shared/hugging_face_jobs/build_input.py", "shared/hugging_face_jobs/requirements-a100.txt")
ARTIFACTS = {
    "ixi-grey-matter-voxel-matrix-v1": ("manifest.json", "gm.npy", "covariates.tsv", "mask.nii.gz"),
    "pcntoolkit-blr-reference-n1000-v1": ("manifest.json", "arrays.npz"),
}
MASKLESS_IXI = {artifact: tuple(name for name in names if name != "mask.nii.gz") for artifact, names in ARTIFACTS.items()}


class StagingProfile(NamedTuple):
    """Explicit experiment files, shared modules and artifact payloads for one mount."""

    code: dict[str, tuple[str, ...]]
    modules: tuple[str, ...]
    artifacts: dict[str, tuple[str, ...]]


PROFILES = {
    "synthetic": StagingProfile(CODE, SHARED_CODE, {}),
    "ixi": StagingProfile(CODE, SHARED_CODE, ARTIFACTS),
    "four-gpu-refits": StagingProfile({"018_four_gpu_refits": ("run.py", *UTILS, "src/modules/compare.py")},
                                     (*KERNEL, *REFIT), MASKLESS_IXI),
    "zbin-stability": StagingProfile({"019_zbin_stability": ("run.py", *UTILS, "src/modules/zbins.py")},
                                     (*KERNEL, *REFERENCE), MASKLESS_IXI),
}
INVENTORY = "INVENTORY.tsv"
PRIVATE_SOURCE = re.compile(r"/(?:Users|home|rds)/|hf://|(?:uk|bio)bank", re.IGNORECASE)
SUBJECT = re.compile(r"IXI\d{3}-(HH|Guys|IOP)-\d{4}-T1")
JUNK = (".results", ".intermediates", "__pycache__", ".git", ".pytest_cache", ".ruff_cache")
MATRIX_SHAPE = (563, 165_402)


def sha256(path: Path) -> str:
    """sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(root: Path, profile: str = "synthetic") -> list[tuple[str, Path, str]]:
    """(destination, source, role) for every file; a missing source is an error naming it."""
    code, modules, artifacts = PROFILES[profile]
    shared = ("shared/__init__.py", *(f"shared/voxelwisenormative/{name}.py" for name in modules))
    rows = [(f"{CODE_PREFIX}{name}/{rel}", root / f"{CODE_PREFIX}{name}/{rel}", "code")
            for name, files in code.items() for rel in files]
    rows += [(rel, root / rel, "code") for rel in shared]
    rows += [(rel, root / rel, "hugging-face") for rel in JOB_FILES]
    rows += [(f"{EXPERIMENTS}/shared/artifacts/{artifact}/{name}", root / EXPERIMENTS / "shared/artifacts" / artifact / name,
              "data") for artifact, names in artifacts.items() for name in names]
    missing = [str(source) for _, source, _ in rows
               if not source.is_file() or any(p.is_symlink() for p in (source, *source.parents) if p != root and root in p.parents)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} planned files are missing: {missing}")
    return rows


def check_names(destinations: list[str]) -> None:
    """Accept only relative inventory paths outside generated data and cache trees."""
    bad = [d for d in destinations if Path(d).is_absolute() or not Path(d).parts
           or any(part in (*JUNK, '..') for part in Path(d).parts)]
    if bad:
        raise ValueError(f"forbidden file names in the input list: {bad}")


def check_ixi_provenance(root: Path) -> list[str]:
    """Check the declared IXI origin and reference parent without a private receipt."""
    folder = root / EXPERIMENTS / "shared/artifacts"
    lines = []
    matrix = folder / "ixi-grey-matter-voxel-matrix-v1"
    record = read_manifest(matrix)
    if record.get("source_dataset") != "https://brain-development.org/ixi-dataset/":
        raise ValueError("matrix manifest must declare the official IXI Dataset page as source_dataset")
    reference = read_manifest(folder / "pcntoolkit-blr-reference-n1000-v1")
    if reference["input_artifact"] != "ixi-grey-matter-voxel-matrix-v1":
        raise ValueError(f"reference artifact input is {reference['input_artifact']!r}, not the IXI matrix")
    lines.append("pcntoolkit-blr-reference-n1000-v1 manifest.json input_artifact: ixi-grey-matter-voxel-matrix-v1")
    for artifact in ARTIFACTS:
        path = folder / artifact / "manifest.json"
        if PRIVATE_SOURCE.search(path.read_text()):
            raise ValueError(f"{path} contains a private source path or dataset")
        lines.append(f"{artifact}: manifest contains no private source path")
    with (matrix / "covariates.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    odd = [r["subject"] for r in rows if not SUBJECT.fullmatch(r["subject"])]
    if len(rows) != MATRIX_SHAPE[0] or odd:
        raise ValueError(f"covariates.tsv: {len(rows)} rows, subjects that are not IXI identifiers: {odd[:5]}")
    lines.append(f"covariates.tsv: {len(rows)} subjects, every identifier matches IXI###-(HH|Guys|IOP)-####-T1")
    return lines


def read_manifest(folder: Path) -> dict:
    """Read one artifact's JSON object, including its path in errors."""
    path = folder / "manifest.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError(f"{path}: expected a manifest object with a files list")
    return value


def check_ixi_data(root: Path, artifacts: dict[str, tuple[str, ...]]) -> list[str]:
    """Hashes against the manifests and the matrix header; nothing is loaded whole."""
    import numpy as np

    lines = []
    for artifact, names in artifacts.items():
        base = root / EXPERIMENTS / "shared/artifacts" / artifact
        listed = {item["path"]: item.get("sha256") for item in read_manifest(base)["files"]}
        for name in names[1:]:
            if listed.get(name) is None or sha256(base / name) != listed[name]:
                raise ValueError(f"{base / name} does not match the sha256 in its manifest: {listed.get(name)}")
            lines.append(f"{artifact}/{name}: sha256 equals manifest {listed[name][:12]}")
    matrix = np.load(root / EXPERIMENTS / "shared/artifacts/ixi-grey-matter-voxel-matrix-v1/gm.npy", mmap_mode="r",
                     allow_pickle=False)  # memory-mapped: only the header is read
    if matrix.shape != MATRIX_SHAPE or matrix.dtype != np.float32:
        raise ValueError(f"gm.npy header is {matrix.shape} {matrix.dtype}, expected {MATRIX_SHAPE} float32")
    lines.append(f"gm.npy header: {matrix.shape} {matrix.dtype}")
    return lines


def write_inventory(out: Path, rows: list[tuple[str, Path, str]]) -> None:
    """INVENTORY.tsv: path, bytes, sha256 and role of every file."""
    with (out / INVENTORY).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256", "role"))
        for destination, _, role in sorted(rows):
            file = out / destination
            writer.writerow((destination, file.stat().st_size, sha256(file), role))


def verify(out: Path) -> list[dict]:
    """Rows of the inventory, after re-hashing the directory against it and refusing any file it does not list."""
    with (out / INVENTORY).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    listed = {row["path"] for row in rows}
    check_names([r["path"] for r in rows])
    entries = tuple(out.rglob("*"))
    if len(listed) != len(rows) or any(p.is_symlink() for p in entries):
        raise ValueError("duplicate inventory entry or symlink in mount")
    present = {p.relative_to(out).as_posix() for p in entries if p.is_file()} - {INVENTORY}
    if present != listed:
        raise ValueError(f"{out}: unlisted {sorted(present - listed)}, missing {sorted(listed - present)}")
    for row in rows:
        if sha256(out / row["path"]) != row["sha256"] or (out / row["path"]).stat().st_size != int(row["bytes"]):
            raise ValueError(f"{out / row['path']} differs from its inventory row {row}")
    return rows


def report(rows: list[dict], *, quiet: bool = False) -> None:
    """The per-file list (unless quiet) and the totals."""
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["role"]] = totals.get(row["role"], 0) + int(row["bytes"])
        if not quiet:
            print(f"{row['role']:<15}{int(row['bytes']):>12,}  {row['sha256'][:12]}  {row['path']}")
    total = sum(totals.values())
    print(f"files: {len(rows)}  total: {total:,} bytes = {total / 2**20:.1f} MiB  by role: "
          + ", ".join(f"{role} {size / 2**20:.2f} MiB" for role, size in sorted(totals.items())))


def build(out: Path, root: Path = ROOT, profile: str = "synthetic") -> list[dict]:
    """Copy the planned files into an empty `out`, write and verify the inventory; returns its rows."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a directory a job may be mounting must never change, use a new one")
    rows = plan(root, profile)
    check_names([destination for destination, _, _ in rows])
    if artifacts := PROFILES[profile].artifacts:
        for line in check_ixi_provenance(root) + check_ixi_data(root, artifacts):
            logger.info("%s", line)
    for destination, source, _ in rows:
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_inventory(out, rows)
    inventory = verify(out)
    check_names([row["path"] for row in inventory])
    return inventory


def main() -> None:
    """Parse the destination, build or verify, and print the inventory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="new or empty directory to fill; with --verify, the directory to check")
    parser.add_argument("--verify", action="store_true", help="only re-check an existing directory against its inventory")
    parser.add_argument("--quiet", action="store_true", help="print only the totals line")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="synthetic",
                        help="choose an explicit code and artifact allowlist (synthetic stages no IXI files)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report(verify(args.output) if args.verify else build(args.output, profile=args.profile), quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
