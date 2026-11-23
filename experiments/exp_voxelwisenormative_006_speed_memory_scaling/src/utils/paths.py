"""single source of truth for this experiment's filesystem paths."""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
RESULTS = EXPERIMENT_DIR / ".results"
INTERMEDIATES = EXPERIMENT_DIR / ".intermediates"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
ARTIFACT = PROJECT_ROOT / "experiments" / "shared" / "artifacts" / "voxelwise-speed-memory-scaling-v1"
