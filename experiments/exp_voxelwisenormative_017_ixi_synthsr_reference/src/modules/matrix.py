"""Build a verified subject-by-voxel matrix in the frozen benchmark row and voxel order."""

from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.integrity import N_SUBJECTS, expect, head_rows, sha256, write_json
from ..utils.paths import INTERMEDIATES, RESULTS, STAGED_INDEX
from .audit_crops import staged_digest
from .compare import MAP_NAME, MASK, MASK_SHA, load_image, map_values
from .matrix_qc import measure, save_qc
from .reference import BASE, CONTROLS
from .staging import load_staged

logger = logging.getLogger(__name__)
COVARIATES = MASK.with_name("covariates.tsv")
COVARIATES_SHA = os.getenv('IXI_COVARIATES_SHA256', 'cae73119c2e9693f1a36420132cddb8de4c517b9361e4d5de75cffcb82edc1d9')
N_VOXELS = 165_402


def covariate_order(expected: set[str]) -> list[str]:
    """Use the published covariates file unchanged for subject identity and order."""
    expect(COVARIATES, COVARIATES_SHA)
    with COVARIATES.open(newline="") as f:
        subjects = [r["subject"] for r in csv.DictReader(f, delimiter="\t")]
    if len(subjects) != N_SUBJECTS or set(subjects) != expected:
        raise ValueError(f"{COVARIATES}: subject set does not match the frozen head manifest")
    return subjects


def read_receipt(path: Path) -> dict[str, Any]:
    """Fail with the path if a receipt cannot be parsed."""
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"{path}: {err}") from err
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def map_receipt(folder: Path, subject: str, task: int, array_job: int, source: Path) -> dict[str, Any]:
    """Bind a map to the requested array, subject, source, and successful registration."""
    path = folder / "receipt.json"
    receipt = read_receipt(path)
    expected = {"subject": subject, "route": "sr", "status": "complete", "array_job": str(array_job),
                "array_task": str(task), "threads_synthsr": 4, "source": str(source)}
    if mismatches := {k: (receipt.get(k), v) for k, v in expected.items() if receipt.get(k) != v}:
        raise ValueError(f"{path}: receipt values differ from the requested array {mismatches}")
    if receipt["fold_check"]["nonpositive_voxels"] != 0 or not (folder / MAP_NAME).is_file():
        raise ValueError(f"{path}: registration has a nonpositive Jacobian or no final map")
    return receipt


def missing_note(subject: str, array_job: int, error: Exception) -> str:
    """Retain the producing worker's concrete failure if it saved one."""
    path = RESULTS / "manifests" / "full-failures" / f"{subject}.json"
    if path.is_file():
        try:
            failed = read_receipt(path)
            if failed.get("array_job") == str(array_job) and failed.get("subject") == subject:
                return f"{failed['error_type']}: {failed['error']}"
        except (ValueError, KeyError) as exc:
            return f"{error}; unreadable failure receipt: {exc}"
    return str(error)


def gather_maps(root: Path, people: list[str], tasks: dict[str, int], jobs: dict[str, int],
                staged: dict[str, Any]) -> tuple[dict[str, dict], dict[str, str]]:
    """Inspect the complete physical input set before opening any image payloads."""
    failures, receipts = {}, {}
    actual = {p.name for p in root.iterdir()} if root.is_dir() else set()
    for extra in sorted(actual - set(people)):
        failures[extra] = f"unexpected subject directory {root / extra}"
    for subject in people:
        source = Path(staged[subject]["folder"]) / "reoriented.nii.gz"
        try:
            receipts[subject] = map_receipt(root / subject, subject, tasks[subject], jobs[subject], source)
        except (ValueError, KeyError, TypeError) as err:
            failures[subject] = missing_note(subject, jobs[subject], err)
    return receipts, failures


def vector(path: Path, receipt: dict, grid, mask: np.ndarray) -> np.ndarray:
    """Load the same C-order mask selection as the original published matrix."""
    expect(path, receipt["map_sha256"])
    return map_values(path, grid)[mask].astype(np.float32)


def concatenate(root: Path, people: list[str], receipts: dict[str, dict], grid, mask: np.ndarray,
                workers: int) -> tuple[np.ndarray, dict[str, str]]:
    """Assemble rows in published order, independent of worker completion order."""
    matrix = np.empty((len(people), int(mask.sum())), dtype=np.float32)
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(vector, root / s / MAP_NAME, receipts[s], grid, mask): i for i, s in enumerate(people)}
        for n, result in enumerate(as_completed(pending), 1):
            i = pending[result]
            try:
                matrix[i] = result.result()
            except (OSError, ValueError, KeyError, TypeError) as err:
                failures[people[i]] = str(err)
            if n % 25 == 0 or n == len(people):
                logger.info("[%d/%d] mapped rows; %d failures", n, len(people), len(failures))
    return matrix, failures


def verify_small_test(matrix: np.ndarray, people: list[str], root: Path, mask: np.ndarray) -> None:
    """Independently check each saved pilot row by indexing flat image data."""
    columns = np.flatnonzero(mask.ravel(order="C"))
    for i, subject in enumerate(people):
        image = load_image(root / subject / MAP_NAME)
        expected = np.asarray(image.dataobj, dtype=np.float32).ravel(order="C")[columns]
        if not np.array_equal(matrix[i], expected):
            raise ValueError(f"row {i} for {subject} does not equal the map's frozen-mask voxels")
    logger.info("%d/%d saved rows exactly matched the voxels of their original maps", len(people), len(people))


def require_complete(path: Path, report: dict[str, Any], failures: dict[str, str]) -> None:
    """Record every bad subject and stop before a partial matrix can be saved."""
    if failures:
        report.update(status="incomplete", failures=failures)
        write_json(path, report)
        raise ValueError(f"{len(failures)} map checks failed; subjects and reasons: {path}")


def recovery_subjects(primary: int, tasks: dict[str, int], digest: str) -> set[str]:
    """Bind the replacement array to the exact failed task indices before reading maps."""
    path = INTERMEDIATES / "manifests" / f"full-{primary}-missing.tsv"
    expect(path, digest)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    people = {r["subject"] for r in rows}
    if not rows or len(people) != len(rows) or any(str(tasks.get(r["subject"])) != r["task"] for r in rows):
        raise ValueError(f"{path}: recovery list does not bind distinct original task indices")
    return people


def build_matrix(array_job: int | None, workers: int, recovery_array: int | None,
                 recovery_sha256: str | None, pilot_job: int | None) -> None:
    """Build and audit a full candidate; omit the shared artifact until it is cleared."""
    job = pilot_job if array_job is None else array_job
    if not 1 <= workers <= 4 or (pilot_job is None) == (array_job is None) or job is None or job <= 0:
        raise ValueError('provide one positive --array-job or --pilot-job and 1–4 workers')
    original = head_rows()
    people = covariate_order(set(original))
    stage_sha = staged_digest()
    staged = load_staged(original)
    recovered = {s for s in original if staged[s].get('recovery') == 'exact public source copy'}
    count = int(os.getenv('IXI_EXPECTED_RECOVERED', '0'))
    if len(recovered) != count:
        raise ValueError(f'{STAGED_INDEX}: expected {count} original public crop copies, got {len(recovered)}')
    small = pilot_job is not None
    name = f"{'small-test' if small else 'full'}-{job}"
    root = RESULTS / "maps/full/sr"
    tasks = {s: i for i, s in enumerate(original)}
    if small:
        root = RESULTS / 'maps/small-test/sr'
        people = [s for s in people if s in CONTROLS]
        tasks = {s: i for i, s in enumerate(CONTROLS, len(BASE))}
    jobs = dict.fromkeys(people, job)
    if recovery_array is not None or recovery_sha256 is not None:
        if small or recovery_array is None or recovery_sha256 is None or recovery_array <= 0 or recovery_array == job:
            raise ValueError(f"invalid recovery binding: primary={job} recovery={recovery_array} sha={recovery_sha256}")
        jobs.update(dict.fromkeys(recovery_subjects(job, tasks, recovery_sha256), recovery_array))
    receipts, failures = gather_maps(root, people, tasks, jobs, staged)
    report_path = RESULTS / "manifests" / f"matrix-{name}.json"
    report: dict[str, Any] = {
        "array_job": job, "recovery_array": recovery_array, "recovery_list_sha256": recovery_sha256,
        "n_expected": len(people), "n_found": len(receipts), "status": "checking",
        "staged_input_sha256": stage_sha, "covariates_sha256": COVARIATES_SHA, "mask_sha256": MASK_SHA,
        "rows": [{"subject": s, "task": tasks[s], "array_job": jobs[s], "job": receipts[s]["job"],
                  "map": str(root / s / MAP_NAME), "map_sha256": receipts[s]["map_sha256"]}
                 for s in people if s in receipts],
    }
    require_complete(report_path, report, failures)
    expect(MASK, MASK_SHA)
    grid = load_image(MASK)
    mask = np.asarray(grid.dataobj).astype(bool)
    if mask.sum() != N_VOXELS:
        raise ValueError(f"{MASK}: expected {N_VOXELS} mask voxels, found {mask.sum()}")
    matrix, failures = concatenate(root, people, receipts, grid, mask, workers)
    require_complete(report_path, report, failures)
    if array_job is not None and len(people) != N_SUBJECTS:
        raise ValueError(f"full matrix needs {N_SUBJECTS} rows; got {len(people)}")
    quality = save_qc(RESULTS / "tables", f"map-qc-{name}", measure(matrix, people, recovered))
    output = INTERMEDIATES / "matrix" / name / "gm.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix, allow_pickle=False)
    if small:
        verify_small_test(np.load(output, mmap_mode="r", allow_pickle=False), people, root, mask)
    report.update(status="review" if quality["flagged_subjects"] else "verified", shape=list(matrix.shape),
                  dtype=str(matrix.dtype), matrix=str(output), matrix_sha256=sha256(output), qc=quality)
    write_json(report_path, report)
    logger.info("saved %s %s matrix; %d quality-control flags; receipt: %s", name, matrix.shape,
                len(quality["flagged_subjects"]), report_path)
    if not small and quality["flagged_subjects"]:
        raise ValueError(f"quality review needed for {quality['flagged_subjects']}; see {report_path}")
