"""Extract public IXI images and reproduce the exact published cropped heads."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..utils.integrity import HEAD_SHA, head_rows, sha256, write_json
from ..utils.paths import ARCHIVE, INTERMEDIATES, STAGED_INDEX

logger = logging.getLogger(__name__)
ARCHIVE_BYTES = 4_840_816_640


def select_members(tar: tarfile.TarFile, subjects: set[str]) -> dict[str, tarfile.TarInfo]:
    """Bind each expected raw file once, without extracting paths from tar metadata."""
    wanted = {f"{subject}.nii.gz": subject for subject in subjects}
    found = {}
    for member in tar:
        name = Path(member.name).name
        if name not in wanted:
            continue
        subject = wanted[name]
        if not member.isfile() or subject in found or member.size <= 0:
            raise ValueError(f"archive has invalid or duplicate member {member.name!r}")
        found[subject] = member
    if missing := subjects.difference(found):
        raise ValueError(f"archive lacks {len(missing)} required members: {sorted(missing)}")
    return found


def head_commands(source: Path, folder: Path) -> tuple[list[str], list[str]]:
    """Reorient the volume and crop it to a 170 mm brain region."""
    reoriented = str(folder / "reoriented")
    return (
        ["fslreorient2std", str(source), reoriented],
        ["robustfov", "-i", reoriented, "-r", str(folder / "cropped"),
         "-m", str(folder / "crop_to_full.mat"), "-b", "170"],
    )


def pin_crop(source: Path, target: Path, expected: str) -> bool:
    """Accept a gzip OS-byte conversion only if all bytes then match the published hash."""
    data = bytearray(source.read_bytes())
    original = hashlib.sha256(data).hexdigest()
    changed = original != expected
    if changed:
        if len(data) < 18 or data[:4] != b"\x1f\x8b\x08\x00" or data[9] != 3:
            raise ValueError(f"{source}: unexpected gzip header or SHA-256 {original}, expected {expected}")
        data[9] = 19  # Historic Mac FSL zlib OS identifier; the DEFLATE stream starts at byte 10.
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{source}: SHA-256 {original}, after gzip OS-byte change {actual}, expected {expected}")
    if source != target or changed:
        partial = target.with_suffix(".partial")
        with partial.open("xb") as f:
            f.write(data)
        partial.replace(target)
    return changed


def stage_one(
    subject: str, member: tarfile.TarInfo, folder: Path, expected: str, previous: Path | None,
) -> dict[str, Any]:
    """Extract one file, prepare and check it, and bind its inputs and outputs."""
    start = time.monotonic()
    folder.mkdir()
    archive_source = f"{ARCHIVE.resolve()}\n{member.name}\n"
    if previous is None:
        prepare_head(subject, member, folder)
        source = folder
    else:
        source = previous / subject
        if (source / "source.txt").read_text() != archive_source:
            raise ValueError(f"{source}: previous head does not bind the same archive member {member.name}")
        shutil.copyfile(source / 'reoriented.nii.gz', folder / 'reoriented.nii.gz')
    (folder / "source.txt").write_text(archive_source)
    corrected = pin_crop(source / "cropped.nii.gz", folder / "cropped.nii.gz", expected)
    result = {
        "subject": subject, "folder": str(folder), "archive_member": member.name,
        "cropped_sha256": expected, "reoriented_sha256": sha256(folder / "reoriented.nii.gz"),
        "gzip_os_byte_changed": corrected, "prepared_in": str(source),
        "seconds": round(time.monotonic() - start, 3),
    }
    write_json(folder / "head.json", result)
    return result


def prepare_head(subject: str, member: tarfile.TarInfo, folder: Path) -> None:
    """Extract one raw image and run the original two FSL commands."""
    raw = folder / f"{subject}.nii.gz"
    with ARCHIVE.open("rb") as f, raw.open("xb") as out:
        f.seek(member.offset_data)
        data = f.read(member.size)
        if len(data) != member.size:
            raise ValueError(f"{ARCHIVE}: short read for {member.name}: {len(data)}/{member.size}")
        out.write(data)
    with (folder / "head.log").open("w") as log:
        for command in head_commands(raw, folder):
            subprocess.run(command, env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def stage(workers: int, repair_from: int | None) -> None:
    """Prepare or verify the full set; any unequal crop blocks the stage receipt."""
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError(f"invalid stage worker count {workers}")
    if STAGED_INDEX.exists():
        raise FileExistsError(f"verified staging already published at {STAGED_INDEX}")
    rows = head_rows()
    previous = None
    if repair_from is None:
        for tool in ("fslreorient2std", "robustfov"):
            if not shutil.which(tool):
                raise FileNotFoundError(f"{tool} is not on PATH")
    else:
        previous = INTERMEDIATES / "staging" / str(repair_from)
        prior = json.loads((previous / "stage-report.json").read_text())
        if (prior["job"] != str(repair_from) or prior["manifest_sha256"] != HEAD_SHA
                or set(prior["heads"]) | set(prior["failures"]) != set(rows)
                or Path(prior["archive"]) != ARCHIVE.resolve()):
            raise ValueError(f"{previous}: prior attempt does not cover the frozen staging plan")
    size = ARCHIVE.stat().st_size
    if size != ARCHIVE_BYTES:
        raise ValueError(f"{ARCHIVE}: {size} bytes, expected {ARCHIVE_BYTES}")
    with tarfile.open(ARCHIVE, "r:") as tar:
        members = select_members(tar, set(rows))
    job = os.environ.get("SLURM_JOB_ID")
    attempt = job if job and job.isascii() and job.isdecimal() else f"local-{time.time_ns()}"
    base = INTERMEDIATES / "staging" / attempt
    base.mkdir(parents=True)
    completed, failures = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(stage_one, s, members[s], base / s, row["cropped_sha256"], previous): s
                   for s, row in rows.items()}
        for i, item in enumerate(as_completed(pending), start=1):
            subject = pending[item]
            try:
                completed[subject] = item.result()
                logger.info("[%d/%d] %s head verified", i, len(rows), subject)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                failures[subject] = str(exc)
                logger.error("[%d/%d] %s head failed: %s", i, len(rows), subject, exc)
    report = {"job": job, "repair_from": repair_from, "manifest_sha256": HEAD_SHA,
              "archive": str(ARCHIVE.resolve()),
              "n_gzip_os_byte_changes": sum(r["gzip_os_byte_changed"] for r in completed.values()),
              "heads": {s: completed[s] for s in rows if s in completed}, "failures": failures}
    write_json(base / "stage-report.json", report)
    if failures:
        raise ValueError(f"{len(failures)}/{len(rows)} heads failed; see {base / 'stage-report.json'}")
    write_json(STAGED_INDEX, report)
    logger.info("staged and verified %s/%s public IXI heads: %s", len(completed), len(rows), STAGED_INDEX)


def load_staged(rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Require a passing receipt for the entire frozen cohort before any registration."""
    index = json.loads(STAGED_INDEX.read_text())
    heads = index["heads"]
    if index["manifest_sha256"] != HEAD_SHA or index["failures"] or set(heads) != set(rows):
        raise ValueError(f"{STAGED_INDEX}: staging receipt does not cover the frozen cohort")
    return heads
