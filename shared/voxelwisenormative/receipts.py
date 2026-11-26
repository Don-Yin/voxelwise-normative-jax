"""Small helpers for process-level memory, metadata and complete result receipts."""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import sys
from pathlib import Path

from .batch_job import hugging_face_job

logger = logging.getLogger(__name__)


def peak_rss_bytes() -> int:
    """Process RSS high-water mark; Linux reports KiB and macOS reports bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def job_identity() -> dict:
    """Identify the machine, interpreter, Slurm task or Hugging Face job."""
    hugging_face = hugging_face_job()
    return {"machine": platform.machine(), "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            **({"hugging_face_job": hugging_face} if hugging_face else {})}


def runtime() -> dict:
    """Record job identity and thread limit for the reference-size fits."""
    return {**job_identity(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}


def publish(path: Path, report: dict) -> Path:
    """Persist a finite, complete receipt, without replacing a completed result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"result already exists: {path}")
    tmp = path.with_suffix(f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote %s", path)
    return path
