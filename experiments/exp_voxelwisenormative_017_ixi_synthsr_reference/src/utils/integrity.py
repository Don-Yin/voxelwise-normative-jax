"""Verify frozen inputs and publish small JSON receipts atomically."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .paths import HEAD_MANIFEST

HEAD_SHA = os.getenv('IXI_HEAD_MANIFEST_SHA256', '')
N_SUBJECTS = 563


def sha256(path: Path) -> str:
    """Stream the digest of the bytes on disk."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def expect(path: Path, expected: str) -> str:
    """Return the digest only if it matches the frozen reference."""
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual}, expected {expected}")
    return actual


def write_json(path: Path, value: Any) -> None:
    """Publish a complete receipt only after the write succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    partial.replace(path)


def head_rows() -> dict[str, dict[str, str]]:
    """Load the 563 original participants in manifest order."""
    if not re.fullmatch(r'[a-f0-9]{64}', HEAD_SHA):
        raise ValueError('set IXI_HEAD_MANIFEST_SHA256 to the locally prepared manifest digest')
    expect(HEAD_MANIFEST, HEAD_SHA)
    with HEAD_MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    result = {row["subject"]: row for row in rows}
    if len(rows) != N_SUBJECTS or len(result) != N_SUBJECTS:
        raise ValueError(f"{HEAD_MANIFEST}: expected {N_SUBJECTS} unique rows, got {len(result)}/{len(rows)}")
    for subject in result:
        if not re.fullmatch(r"IXI\d{3}-(Guys|HH|IOP)-\d{4}-T1", subject):
            raise ValueError(f"{HEAD_MANIFEST}: invalid subject {subject!r}")
    return result
