"""Join each precision's receipt, GPU samples and device traces into one table per run."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from shared.voxelwisenormative.refit_trace import device_busy

from .gpu_samples import read_samples, summarize

logger = logging.getLogger(__name__)
PRECISIONS = ("float32", "float64")


def sampler_ok(report: dict) -> None:
    """A sampler that died or complained invalidates the utilisation columns."""
    if report["exit_before_stop"] is not None or report["stderr"]:
        raise ValueError(f"nvidia-smi sampler failed: exit={report['exit_before_stop']}, stderr={report['stderr']!r}")


def pass_row(receipt: dict, entry: dict, baseline: float, samples: dict | None) -> dict:
    """One table row: effective time, loop padding, memory, utilisation and agreement with K=1."""
    row = {"precision": receipt["precision"], "label": entry["label"], "k": entry["k"], "status": entry["status"]}
    if entry["status"] != "ok":
        return row
    batches, refits = entry["batches"], receipt["refits"]
    plan = receipt["compile"][str(entry["k"])].values()
    computed = sum(b["computed_voxel_steps"] for b in batches)
    row.update({
        "seconds_per_refit": entry["seconds_per_refit"], "speedup_vs_k1": baseline / entry["seconds_per_refit"],
        "device_seconds_per_refit": sum(b["compute_seconds"] for b in batches) / refits,
        "gather_seconds_per_refit": sum(b["gather_seconds"] for b in batches) / refits,
        "computed_voxel_steps_per_refit": computed / refits,
        "padding_factor": computed / sum(b["member_voxel_steps"] for b in batches),
        "active_fraction": sum(b["active_voxel_steps"] for b in batches) / computed,
        "xla_temp_bytes_max": max(p["temp_size_in_bytes"] for p in plan),
        "xla_argument_output_bytes_max": max(p["argument_size_in_bytes"] + p["output_size_in_bytes"] for p in plan),
        "compile_seconds": sum(p["compile_seconds"] for p in plan) if entry["label"] != "repeat" else 0.0,
        "allocator_peak_bytes_cumulative": (entry["memory_after"] or {}).get("peak_bytes_in_use")})
    if samples is not None:
        row.update(summarize(samples, [b["device_window"] for b in batches]))
    if entry["agreement_with_k1"] is not None:
        row.update(entry["agreement_with_k1"])
    return row


def receipt_folder(results: Path, run_name: str, precision: str) -> Path:
    """json/<run>/ when both precisions share one run (a Slurm array job), else json/<run>-<precision>/.

    The second form is how two Hugging Face jobs, one per precision, are named (hugging-face-a100-gpu-headroom-float32).
    """
    shared = results / "json" / run_name
    return shared if (shared / f"{precision}.json").exists() else results / "json" / f"{run_name}-{precision}"


def collect(results: Path, run_name: str) -> Path:
    """Write tables/<run>/sweep.csv and trace.csv from every precision present for the run."""
    rows, traces = [], []
    for precision in PRECISIONS:
        folder = receipt_folder(results, run_name, precision)
        path = folder / f"{precision}.json"
        if not path.exists():
            logger.info("no %s receipt for %s", precision, run_name)
            continue
        receipt = json.loads(path.read_text())
        sampler = receipt["gpu_sampler"]
        samples = None
        if sampler is not None:
            sampler_ok(sampler)
            samples = read_samples(results / sampler["path"], sampler["utc_offset_seconds"])
        # Speed baseline: the second, fully warm K=1 pass; the first K=1 pass is only the bitwise reference.
        baseline = next(p["seconds_per_refit"] for p in receipt["passes"] if p["label"] == "repeat")
        rows.extend(pass_row(receipt, entry, baseline, samples) for entry in receipt["passes"])
        trace_path = folder / f"{precision}-trace.json"
        if trace_path.exists():
            for item in json.loads(trace_path.read_text())["traces"]:
                busy = device_busy(results / item["perfetto"])
                threads = busy.pop("device_threads")
                traces.append({"precision": precision, **{k: v for k, v in item.items() if k != "perfetto"},
                               **busy, "device_threads": json.dumps(threads)})
    if not rows:
        raise FileNotFoundError(f"no float32.json or float64.json receipts for run {run_name} under {results / 'json'}")
    out = results / "tables" / run_name
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("sweep.csv", rows), ("trace.csv", traces)):
        if not table:
            continue
        fields = list(dict.fromkeys(key for row in table for key in row))
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        logger.info("wrote %s (%d rows)", out / name, len(table))
    return out
