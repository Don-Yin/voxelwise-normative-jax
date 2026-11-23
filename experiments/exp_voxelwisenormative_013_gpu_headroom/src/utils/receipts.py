"""Process metadata and complete, never-overwritten JSON receipts."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import resource
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
ENVIRONMENT_KEYS = ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS",
                    "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")


def peak_rss_bytes() -> int:
    """Process RSS high-water mark; Linux reports KiB and macOS reports bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def runtime() -> dict:
    """Identify the interpreter, the Slurm task or Hugging Face job and the environment that shapes JAX's allocator."""
    # run.py imports this module before the project root is on sys.path
    from shared.voxelwisenormative.batch_job import hugging_face_job

    hugging_face = hugging_face_job()  # a Hugging Face job leaves every slurm_* field empty and is recorded here instead
    return {"machine": platform.machine(), "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
            **({"hugging_face_job": hugging_face} if hugging_face else {})}


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
