"""Run-name validation and the sweep's environment metadata."""

from __future__ import annotations

import os
import re

RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
ENVIRONMENT_KEYS = ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS",
                    "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")


def runtime() -> dict:
    """Record job identity and the allowlisted environment that shapes JAX's allocator."""
    # run.py imports RUN_NAME before the project root is on sys.path.
    from shared.voxelwisenormative.receipts import job_identity

    return {**job_identity(), "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}}
