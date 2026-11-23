"""Run the published example rather than a second copy of its code."""

import os
from pathlib import Path
import re
import subprocess
import sys


def test_readme_example(tmp_path):
    """Check an installed package from outside the checkout on the CPU."""
    readme = Path(__file__).resolve().parents[2] / "README.md"
    examples = re.findall(r"<!-- readme-example -->\n```python\n(.*?)\n```\n<!-- /readme-example -->",
                          readme.read_text(encoding="utf-8"), re.DOTALL)
    assert len(examples) == 1, "expected one marked Python example in README.md"
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-c", examples[0]],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "(16, 2)"
