"""Convert the caller's Hugging Face bucket name to a storage URI."""
from __future__ import annotations

import re


def bucket_uri(name: str) -> str:
    """Require an explicit namespace/name, with no shell or mount syntax."""
    field = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    if not re.fullmatch(rf"{field}/{field}", name) or ".." in name:
        raise ValueError("bucket must be namespace/name")
    return "hf://buckets/" + name
