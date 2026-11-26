"""single source of truth for this experiment's filesystem paths."""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
RESULTS = EXPERIMENT_DIR / ".results"
