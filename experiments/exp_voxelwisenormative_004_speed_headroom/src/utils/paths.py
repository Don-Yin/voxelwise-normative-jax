"""single source of truth for this experiment's filesystem paths."""

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS = EXPERIMENT_DIR / ".results"


def add_project_root() -> None:
    """resolve shared code from this checkout when run as a script."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
