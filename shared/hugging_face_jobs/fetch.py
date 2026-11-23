"""Fetch a finished Hugging Face job's results, check them, and place them in the experiment's `.results`.

    python shared/hugging_face_jobs/fetch.py RUN_NAME [--job-id ID] [--source DIR]

RUN_NAME is the folder `container_entry.sh` wrote under `/out`. The folder is synced to a partial directory, checked,
and only then placed; a failed check places nothing. Placing means:

1. a byte-for-byte archive at `<experiment>/.results/hugging-face/RUN_NAME/`, with FETCHED.json (job id, time, sha256 of every file);
2. every file of the archive hard-linked to the same relative path under `<experiment>/.results/` (the tree a Slurm
   job writes there), so `collect` and the figure scripts read it as they read a Slurm run; nothing is overwritten,
   and a path that already exists with other content stops the placement before anything is linked;
3. for exp_011 also the flat folder `<experiment>/.results/fetch-RUN_NAME/` (result.json, refit-flags.npz, z-summary.npz and
   cutoff-flags.npz when the run has them, the host record and the log), the layout `jobs/rank_removed.py`, `tests/accept_fetch.py` and the figure
   scripts read for `fetch-<job>`.

The checks: the host record says the run finished with exit code 0 on one A100 (and, with --job-id, that the platform's
JOB_ID is that job); the experiment's receipts agree with the log and carry no Slurm id. `--source DIR` uses an already
fetched folder instead of the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from shared.hugging_face_jobs.bucket import bucket_uri

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = re.compile(r"hugging-face-a100-[a-z0-9][a-z0-9-]{0,45}")
HOST_FILE = "hugging-face-host.json"
ENVIRONMENT = "hugging-face-a100-venv"  # the venv container_entry.sh builds; shared/voxelwisenormative/batch_job.py names it too
ARCHIVE_RECORD = "FETCHED.json"
EXP004, EXP010, EXP011, EXP013 = (f"exp_voxelwisenormative_{name}" for name in (
    "004_speed_headroom", "010_reference_size_scaling", "011_fragile_clusters", "013_gpu_headroom"))
SLURM_FIELDS = ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id")
Z_SUMMARY = "z-summary.npz"  # exp_011 --z-summary; src/modules/z_summary.py writes it
Z_ARRAYS = ("original_z", "refit_min_abs_z", "refit_max_abs_z", "held_out_cohort_rows_zero_based")
CUTOFF_FLAGS = "cutoff-flags.npz"  # exp_011 --refit-flag-cutoffs (with --z-summary); the same module writes it
CUTOFF_ARRAYS = ("cutoffs", "refit_flags_packed", "original_flags_packed", "held_out_cohort_rows_zero_based")
LOG_TITLES = {"posterior": "rotated_posterior_pass", "optimizer early stop": "early_optimizer",
              "optimizer fixed cap": "fixed_optimizer"}


class FetchError(Exception):
    """A check failed; nothing was placed."""


def sha256(path: Path) -> str:
    """sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """A JSON object, with the file named when it is missing or not one."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FetchError(f"cannot read {path}: {error}") from error
    if not isinstance(record, dict):
        raise FetchError(f"{path} is not a JSON object")
    return record


def expect(condition: bool, message: str) -> None:
    """Raise FetchError with the message unless the condition holds."""
    if not condition:
        raise FetchError(message)


def check_host(folder: Path, run: str, job_id: str | None) -> tuple[dict, list[str]]:
    """The host record of a run that finished on an A100: returns it with the notes of what was checked."""
    expect(not (folder / "ENTRY-FAILED.txt").exists(), f"{run}: the entry script failed, see ENTRY-FAILED.txt")
    host = read_json(folder / "json" / run / HOST_FILE)
    notes = []
    expect(host.get("run_name") == run, f"{run}: the host record is of run {host.get('run_name')!r}")
    expect(host.get("dry_run") is False, f"{run}: a --check-only run has no results to place")
    expect(host.get("exit_code") == 0, f"{run}: run.py exited with {host.get('exit_code')!r}, not 0")
    devices = host.get("gpu", {}).get("devices", [])
    expect(len(devices) == 1 and "A100" in devices[0]["name"], f"{run}: expected one A100, the host record has {devices}")
    platform_job = host.get("environment", {}).get("JOB_ID")
    expect(job_id is None or platform_job == job_id, f"{run}: the platform's JOB_ID is {platform_job!r}, not {job_id!r}")
    marker = host.get("hugging_face_job")
    expect(marker is None or marker == platform_job, f"{run}: marker {marker!r} differs from JOB_ID {platform_job!r}")
    prefix = host.get("python", {}).get("prefix", "")
    expect(Path(prefix).name == ENVIRONMENT, f"{run}: ran in {prefix!r}, not in an environment named {ENVIRONMENT}")
    expect(not host.get("slurm_variables"), f"{run}: SLURM_* variables were set: {host.get('slurm_variables')}")
    notes.append(f"host record: exit 0 on {devices[0]['name']}, driver {devices[0]['driver_version']}, JOB_ID {platform_job}, {ENVIRONMENT}"
                 + ("" if marker else " (v2 record: no hugging_face_job marker; its note about a SLURM_JOB_ID stand-in is stale)"))
    return host, notes


def same_marker(found: str | None, host: dict, where: str) -> None:
    """A receipt names the job exactly as the host record does: a v3 record has the id, a v2 record has none."""
    expect(found == host.get("hugging_face_job"), f"{where}: hugging_face_job is {found!r}, the host record says {host.get('hugging_face_job')!r}")


def log_numbers(log: Path, pattern: str) -> list[tuple[str, ...]]:
    """Every match of `pattern` in the run log, as tuples of its groups."""
    expect(log.is_file(), f"missing log {log}")
    return [m.groups() for m in map(re.compile(pattern).search, log.read_text().splitlines()) if m]


def check_exp004(folder: Path, run: str) -> list[str]:
    """mechanism-timing.json against the log's 21 timings, and each stored median."""
    receipt = read_json(folder / "json" / run / "mechanism-timing.json")
    expect(receipt.get("smoke_cpu") is False and receipt.get("slurm_job_id") is None, f"{run}: not a real, Slurm-free receipt")
    rows = log_numbers(folder / "logs" / f"{run}.log", r"^\[(\d+)/(\d+)\] (.+): ([0-9.]+) s$")
    for title, key in LOG_TITLES.items():
        logged = [float(value) for _, _, name, value in rows if name == title]
        expect(len(logged) == receipt["repeats"], f"{run}: {len(logged)} log lines for {title}, {receipt['repeats']} repeats")
        expect([f"{v:.6f}" for v in logged] == [f"{v:.6f}" for v in receipt["seconds"][key]], f"{run}: log and receipt differ for {key}")
    for key, values in receipt["seconds"].items():
        expect(statistics.median(values) == receipt["medians_seconds"][key], f"{run}: median of {key} is not the stored one")
    return ["mechanism-timing.json: 21 log timings equal the receipt to 6 decimals, medians recomputed"]


def check_exp010(folder: Path, run: str, host: dict) -> list[str]:
    """Each `json/<stage>/<run>/*.json` receipt: no Slurm id, medians recomputed, timings equal the log's."""
    notes = []
    receipts = sorted((folder / "json").glob(f"*/{run}/*.json"))
    expect(bool(receipts), f"{run}: no json/<stage>/{run}/*.json receipt")
    log = (folder / "logs" / f"{run}.log").read_text()
    for path in receipts:
        row = read_json(path)
        expect(row.get("schema_version") == 1 and all(row.get(key) is None for key in SLURM_FIELDS), f"{path}: not a Slurm-free schema 1 receipt")
        same_marker(row.get("hugging_face_job"), host, str(path))
        if row.get("workload") == "batched-early":
            expect("A100" in row["device"], f"{path}: device {row['device']}")
            expect(statistics.median(row["seconds_repeats"]) == row["seconds_median"], f"{path}: seconds_median is not the median of seconds_repeats")
            timed = [float(v) for v in re.findall(r" timed ([0-9.]+)s$", log, re.MULTILINE)]
            warm = [float(v) for v in re.findall(r" warmup ([0-9.]+)s$", log, re.MULTILINE)]
            expect([f"{v:.3f}" for v in timed] == [f"{v:.3f}" for v in row["seconds_repeats"]], f"{path}: log timings {timed} differ from the receipt")
            expect(len(warm) == 1 and f"{warm[0]:.3f}" == f"{row['warmup_seconds']:.3f}", f"{path}: log warm-up {warm} differs from the receipt")
            rss = re.search(r"peak_rss_self=([0-9.]+)MiB", log)
            expect(rss is not None and abs(float(rss.group(1)) - row["host_peak_rss_bytes"] / 2**20) < 0.06, f"{path}: log peak RSS differs from the receipt")
            notes.append(f"{path.name}: median {row['seconds_median']:.3f} s of {len(timed)} timed passes, warm-up {row['warmup_seconds']:.3f} s, "
                         f"log lines and peak RSS equal the receipt")
    return notes


def flag_limit(run: str, receipt: dict) -> str:
    """The T of the receipt's `abs(z) > T` rule, as written."""
    rule = re.fullmatch(r"abs\(z\) > ([0-9.]+)", receipt.get("flag_rule", ""))
    if rule is None:
        raise FetchError(f"{run}: cannot read the flag rule {receipt.get('flag_rule')!r}")
    return rule.group(1)


def need_numpy(run: str, file: str):
    """numpy, or the reason a check of `file` cannot run."""
    try:
        import numpy as np
    except ImportError as error:
        raise FetchError(f"{run}: checking {file} needs numpy") from error
    return np


def check_cutoff_flags(folder: Path, run: str, receipt: dict, cutoffs: list[float]) -> str:
    """cutoff-flags.npz against z-summary.npz and refit-flags.npz: the job's own checks, repeated on the fetched files.

    Per cutoff T the AND of the refits' flags must be `refit_min_abs_z > T`, their OR `refit_max_abs_z > T`, and the single
    fit's flags `abs(original_z) > T`; the run's own flags at the receipt's limit must AND and OR to the same at that limit
    (the job's third check compares those flags with the ones its cutoff code makes; only they are stored here), and each
    stricter cutoff must flag a subset of what the next looser one does, down to those flags.
    """
    np = need_numpy(run, CUTOFF_FLAGS)
    limit = float(flag_limit(run, receipt))
    expect(len(set(cutoffs)) == len(cutoffs) and all(cutoff > limit for cutoff in cutoffs), f"{run}: cutoffs {cutoffs} are not distinct values above {limit}")

    def pack(flags):
        return np.packbits(flags, axis=-1, bitorder="little")

    with (np.load(folder / "arrays" / run / CUTOFF_FLAGS, allow_pickle=False) as cut, np.load(folder / "arrays" / run / Z_SUMMARY, allow_pickle=False) as z,
          np.load(folder / "arrays" / run / "refit-flags.npz", allow_pickle=False) as own):
        expect(list(cut.files) == list(CUTOFF_ARRAYS), f"{run}: {CUTOFF_FLAGS} holds {cut.files}, expected {list(CUTOFF_ARRAYS)}")
        scans, voxels = own["recurrence_counts"].shape
        refits, width = receipt["bootstrap_refits"], (voxels + 7) // 8
        stored, single = cut["refit_flags_packed"], cut["original_flags_packed"]
        expect(cut["cutoffs"].dtype == np.float64 and cut["cutoffs"].tolist() == cutoffs, f"{run}: stored cutoffs {cut['cutoffs'].tolist()} are not the arguments' {cutoffs}")
        expect(stored.dtype == single.dtype == np.uint8 and stored.shape == (len(cutoffs), refits, scans, width) and single.shape == (len(cutoffs), scans, width),
               f"{run}: flag arrays are {stored.dtype} {stored.shape} and {single.dtype} {single.shape}, expected uint8 {(len(cutoffs), refits, scans, width)} and {(len(cutoffs), scans, width)}")
        expect(np.array_equal(cut[CUTOFF_ARRAYS[3]], own[CUTOFF_ARRAYS[3]]) and np.array_equal(z[Z_ARRAYS[3]], own[CUTOFF_ARRAYS[3]]),
               f"{run}: {CUTOFF_FLAGS}, {Z_SUMMARY} and refit-flags.npz hold different scans")
        smallest, largest, original = z["refit_min_abs_z"], z["refit_max_abs_z"], z["original_z"]
        for at, flags, one in [(limit, own["refit_flags_packed"], None), *((c, stored[k], single[k]) for k, c in enumerate(cutoffs))]:
            expect(np.array_equal(np.bitwise_and.reduce(flags, axis=0), pack(smallest > at)), f"{run}: at {at:g} the AND of the refits' flags is not refit_min_abs_z > {at:g}")
            expect(np.array_equal(np.bitwise_or.reduce(flags, axis=0), pack(largest > at)), f"{run}: at {at:g} the OR of the refits' flags is not refit_max_abs_z > {at:g}")
            expect(one is None or np.array_equal(one, pack(np.abs(original) > at)), f"{run}: at {at:g} the single fit's flags are not abs(original_z) > {at:g}")
        looser = own["refit_flags_packed"]
        for k in sorted(range(len(cutoffs)), key=cutoffs.__getitem__):
            expect(not (stored[k] & ~looser).any(), f"{run}: cutoff {cutoffs[k]:g} flags something its next looser cutoff does not")
            looser = stored[k]
    return (f"{CUTOFF_FLAGS}: {len(cutoffs)} cutoffs {','.join(f'{c:g}' for c in cutoffs)}, uint8 {stored.shape}; per cutoff AND == refit_min_abs_z > T, "
            f"OR == refit_max_abs_z > T, single fit == abs(original_z) > T; the same AND and OR at {limit:g} on refit-flags.npz; nested in {limit:g}")


def check_z_summary(folder: Path, run: str, receipt: dict) -> str:
    """z-summary.npz again, here: float64 arrays of the flags' shape, min <= max, and `min |z| > T` equal to the run's own counts."""
    np = need_numpy(run, Z_SUMMARY)
    limit = flag_limit(run, receipt)
    with (np.load(folder / "arrays" / run / Z_SUMMARY, allow_pickle=False) as z,
          np.load(folder / "arrays" / run / "refit-flags.npz", allow_pickle=False) as flags):
        expect(list(z.files) == list(Z_ARRAYS), f"{run}: {Z_SUMMARY} holds {z.files}, expected {list(Z_ARRAYS)}")
        counts, rows = flags["recurrence_counts"], flags["held_out_cohort_rows_zero_based"]
        arrays = {key: z[key] for key in Z_ARRAYS[:3]}
        expect(all(a.dtype == np.float64 and a.shape == counts.shape for a in arrays.values()),
               f"{run}: the z arrays are not float64 of shape {counts.shape}: {[(a.dtype, a.shape) for a in arrays.values()]}")
        expect(all(np.isfinite(a).all() for a in arrays.values()), f"{run}: a z array holds a NaN or an infinity")
        expect(np.array_equal(z[Z_ARRAYS[3]], rows), f"{run}: {Z_SUMMARY} and refit-flags.npz hold different scans")
        smallest, largest = arrays["refit_min_abs_z"], arrays["refit_max_abs_z"]
        expect(bool((smallest >= 0).all() and (smallest <= largest).all()), f"{run}: the smallest |z| is negative or above the largest")
        every = counts == receipt["bootstrap_refits"]
        differ = int(((smallest > float(limit)) != every).sum())
        expect(differ == 0, f"{run}: refit_min_abs_z > {limit} and recurrence_counts == {receipt['bootstrap_refits']} differ in {differ:,} entries")
        expect(receipt["z_summary"]["check"]["above_threshold_in_every_refit"] == int(every.sum()), f"{run}: the receipt's own check counts another {int(every.sum()):,}")
    return (f"{Z_SUMMARY}: float64 {counts.shape} x 3, min <= max, refit_min_abs_z > {limit} equals recurrence_counts == "
            f"{receipt['bootstrap_refits']} in all {counts.size:,} entries ({int(every.sum()):,} flagged in every refit)")


def check_exp011(folder: Path, run: str, host: dict) -> list[str]:
    """result.json against every file it lists (sha256, bytes), the run's own arguments, and the job ids."""
    receipt = read_json(folder / "json" / run / "result.json")
    npz = folder / "arrays" / run / "refit-flags.npz"
    outputs = receipt.get("outputs", {})
    expect("refit-flags.npz" in outputs, f"{run}: result.json lists no refit-flags.npz: {list(outputs)}")
    for name, record in outputs.items():
        path = folder / "arrays" / run / name
        expect(path.is_file() and (sha256(path), path.stat().st_size) == (record.get("sha256"), record.get("bytes")), f"{run}: {name} does not match result.json {record}")
    runtime = receipt["runtime"]
    expect(receipt.get("mode") == "ixi-cuda-refits" and "A100" in runtime.get("device", ""), f"{run}: not a real A100 per-refit receipt")
    expect(runtime.get("slurm_job") is None, f"{run}: runtime.slurm_job is {runtime.get('slurm_job')!r}")
    same_marker(runtime.get("hugging_face_job"), host, f"{run} runtime")
    args = host["entry_args"].split()
    fold = int(args[args.index("--fold") + 1]) if "--fold" in args else 0
    expect(receipt["fold"] == fold, f"{run}: receipt fold {receipt['fold']}, arguments say {fold}")
    if "--reference-n" in args:
        expect(receipt.get("reference_n") == int(args[args.index("--reference-n") + 1]), f"{run}: reference_n differs from the arguments")
    expect(receipt["bootstrap_refits"] == 64 and receipt["held_out_scans"] > 0, f"{run}: unexpected refit or scan count")
    seconds = receipt["refit_seconds"]
    notes = [(f"result.json: {', '.join(outputs)} sha256 and bytes equal the receipt (refit-flags.npz {npz.stat().st_size:,}); fold {fold}, "
              f"{receipt['references']} references, {receipt['held_out_scans']} held-out scans, 64 refits "
              f"(first {seconds[0]:.1f} s, warm median {statistics.median(seconds[1:]):.2f} s)")]
    asked = "--z-summary" in args
    expect(asked == (Z_SUMMARY in outputs) == ("z_summary" in receipt), f"{run}: --z-summary is {asked} in the arguments, {Z_SUMMARY in outputs} in the outputs and {'z_summary' in receipt} in the receipt")
    cutoffs = [float(value) for value in args[args.index("--refit-flag-cutoffs") + 1].split(",")] if "--refit-flag-cutoffs" in args else None
    in_receipt = "cutoff_flags" in receipt.get("z_summary", {})
    expect((cutoffs is not None) == (CUTOFF_FLAGS in outputs) == in_receipt and (cutoffs is None or asked),
           f"{run}: --refit-flag-cutoffs is {cutoffs is not None} in the arguments, {CUTOFF_FLAGS in outputs} in the outputs and {in_receipt} in the receipt (it needs --z-summary)")
    if in_receipt:
        entry = receipt["z_summary"]["cutoff_flags"]
        expect(entry["cutoffs"] == cutoffs and entry["check"]["entries_that_differ"] == 0, f"{run}: the receipt's cutoffs {entry['cutoffs']} or its own check disagree with the arguments' {cutoffs}")
    return notes + ([check_z_summary(folder, run, receipt)] if asked else []) + ([check_cutoff_flags(folder, run, receipt, cutoffs)] if cutoffs else [])


def check_exp013(folder: Path, run: str, host: dict) -> list[str]:
    """Each precision's receipt: no Slurm id, a sampler that ended cleanly and left its samples, the passes' statuses."""
    notes = []
    receipts = sorted((folder / "json" / run).glob("float*.json"))
    expect(any(not p.name.endswith("-trace.json") for p in receipts), f"{run}: no float32.json or float64.json receipt")
    for path in (p for p in receipts if not p.name.endswith("-trace.json")):
        row = read_json(path)
        expect(all(row.get(key) is None for key in SLURM_FIELDS), f"{path}: Slurm ids are set")
        same_marker(row.get("hugging_face_job"), host, str(path))
        sampler = row["gpu_sampler"]
        if sampler is None or sampler["exit_before_stop"] is not None or sampler["stderr"]:
            raise FetchError(f"{path}: the nvidia-smi sampler is missing or failed: {sampler}")
        samples = folder / sampler["path"]
        expect(samples.is_file() and samples.stat().st_size > 0, f"{path}: sampler output {samples} is missing or empty")
        statuses = {entry["label"]: entry["status"] for entry in row["passes"]}
        notes.append(f"{path.name}: sampler ended cleanly ({samples.stat().st_size:,} bytes); passes {statuses}")
    return notes


def check(folder: Path, run: str, job_id: str | None) -> tuple[dict, list[str]]:
    """Every check that applies to the run's experiment; returns the host record and the notes."""
    host, notes = check_host(folder, run, job_id)
    experiment = host["experiment"]
    expect((folder / "logs" / f"{run}.log").is_file(), f"{run}: logs/{run}.log is missing")
    checks = {EXP004: lambda: check_exp004(folder, run), EXP010: lambda: check_exp010(folder, run, host),
              EXP011: lambda: check_exp011(folder, run, host), EXP013: lambda: check_exp013(folder, run, host)}
    expect(experiment in checks, f"{run}: no checks for {experiment}")
    return host, notes + checks[experiment]()


def files_of(folder: Path) -> list[Path]:
    """Relative paths of every file under folder, sorted."""
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())


def link(source: Path, target: Path) -> None:
    """Hard-link source at target (no second copy on disk); copy when the file system refuses."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def require_no_clash(tree: Path, results: Path) -> None:
    """Stop unless every file of `tree` is absent from `results` or already there with the same content."""
    clashes = [rel for rel in files_of(tree) if rel.name != ARCHIVE_RECORD
               and (results / rel).exists() and sha256(results / rel) != sha256(tree / rel)]
    expect(not clashes, f"refusing to overwrite {len(clashes)} existing files under {results}: {[str(c) for c in clashes[:5]]}")


def merge(archive: Path, results: Path) -> int:
    """Link every archived file into `results` at its relative path, leaving what is already there; returns the count."""
    placed = 0
    for rel in files_of(archive):
        if rel.name != ARCHIVE_RECORD and not (results / rel).exists():
            link(archive / rel, results / rel)
            placed += 1
    return placed


def flat_view(archive: Path, results: Path, run: str) -> Path:
    """exp_011's `fetch-<run>` folder: the files its rank_removed, accept_fetch and figure scripts read, side by side."""
    folder = results / f"fetch-{run}"
    expect(not folder.exists(), f"{folder} exists")
    pairs = [(f"json/{run}/result.json", "result.json"), (f"arrays/{run}/refit-flags.npz", "refit-flags.npz"),
             (f"json/{run}/{HOST_FILE}", HOST_FILE), (f"logs/{run}.log", f"{run}.log")]
    pairs += [(f"arrays/{run}/{name}", name) for name in (Z_SUMMARY, CUTOFF_FLAGS) if (archive / "arrays" / run / name).is_file()]
    for source, name in pairs:
        link(archive / source, folder / name)
    return folder


def place(folder: Path, run: str, host: dict, notes: list[str], root: Path = ROOT, *, origin: str = "local") -> list[str]:
    """Archive, merge and (exp_011) flat view; returns what was done."""
    results = root / "experiments" / host["experiment"] / ".results"
    archive = results / "hugging-face" / run
    expect(not archive.exists(), f"{archive} exists: already fetched")
    expect(not (results / f"fetch-{run}").exists(), f"{results / f'fetch-{run}'} exists")
    require_no_clash(folder, results)  # before anything is written, so a refusal leaves nothing behind
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, archive)
    record = {"run_name": run, "job_id": host["environment"].get("JOB_ID"), "source": origin,
              "fetched_utc": datetime.now(UTC).isoformat(), "checks": notes,
              "files": {str(rel): {"bytes": (archive / rel).stat().st_size, "sha256": sha256(archive / rel)} for rel in files_of(archive)}}
    (archive / ARCHIVE_RECORD).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    done = [f"archived {len(record['files'])} files at {archive}"]
    done.append(f"linked {merge(archive, results)} files into {results}")
    if host["experiment"] == EXP011:
        done.append(f"flat folder {flat_view(archive, results, run)}")
    return done


def sync(origin: str, destination: Path) -> None:
    """Sync a run from the caller's bucket."""
    command = ["uvx", "--from", "huggingface_hub", "hf", "sync", origin, str(destination)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
    expect(done.returncode == 0, f"bucket sync returned {done.returncode}")
    expect(any(destination.rglob("*")), "bucket run is empty or missing")


def fetch(run: str, job_id: str | None = None, source: Path | None = None, root: Path = ROOT,
          *, bucket: str | None = None) -> list[str]:
    """Sync (or read `source`), check, place; returns the notes of the checks and of the placement."""
    expect(RUN_NAME.fullmatch(run) is not None, f"run name must look like hugging-face-a100-<name>: {run}")
    expect((source is None) != (bucket is None), "provide one of --source and --bucket")
    origin = "local" if bucket is None else f"{bucket_uri(bucket)}/{run}"
    with tempfile.TemporaryDirectory(prefix=f"{run}-") as scratch:
        folder = Path(scratch) / run
        if source is None:
            sync(origin, folder)
        else:
            shutil.copytree(source, folder)
        host, notes = check(folder, run, job_id)
        return notes + place(folder, run, host, notes, root, origin=origin)


def main() -> None:
    """Fetch one run; print each check and each placement, or the first failed check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name")
    parser.add_argument("--job-id", help="the Hugging Face job id; must equal the host record's JOB_ID")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="an already fetched folder to verify")
    inputs.add_argument("--bucket", help="source bucket as namespace/name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        for line in fetch(args.run_name, args.job_id, args.source, bucket=args.bucket):
            logger.info("%s", line)
    except (FetchError, ValueError) as error:
        logger.error("not placed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
