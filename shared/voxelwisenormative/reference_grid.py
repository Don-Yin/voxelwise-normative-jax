"""The reference-size grid, independent bootstrap chains and Slurm index order."""

from __future__ import annotations

from dataclasses import dataclass

MASK_VOXELS = 165_402
N_HELD_OUT = 113
IXI_N = (25, 50, 100, 200, 450)
SYNTHETIC_N = (450, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000)
PRECISIONS = ("float32", "float64")
CHAINS = (20261121, 20261122, 20261123)
COLUMNS_SEED = 20261121
SYNTHETIC_SEED = 20261120
TRIALS = 64
MIN_RECUR = 52
Z_THRESHOLD = 1.96


@dataclass(frozen=True)
class Cell:
    """One dataset and training size; prediction size stays fixed."""

    dataset: str
    n: int

    @property
    def label(self) -> str:
        """Stable path key, sortable within the dataset."""
        return f"{self.dataset}-n{self.n:05d}"

    @property
    def cpu_widths(self) -> tuple[int, int, int]:
        """Brackets for the monolithic peak extrapolation within a 1-CPU RAM limit."""
        if self.n <= 450:
            return 16, 64, 256
        if self.n <= 2_000:
            return 8, 32, 96
        if self.n <= 10_000:
            return 4, 16, 48
        return 4, 12, 24


REAL = tuple(Cell("ixi", n) for n in IXI_N)
BENCHMARK = REAL + tuple(Cell("synthetic", n) for n in SYNTHETIC_N)
CPU_CASES = tuple((c, w) for c in BENCHMARK for w in c.cpu_widths)
GPU_CASES = tuple((c, p) for c in BENCHMARK for p in PRECISIONS)
BOOT_CASES = tuple((c, k) for c in REAL for k in range(3 if c.n <= 100 else 1))
V_SWEEP = (1_000, 10_000, 100_000, MASK_VOXELS)


def case_at(cases: tuple, index: int):
    """Refuse negative and out-of-range Slurm task indices."""
    if not 0 <= index < len(cases):
        raise ValueError(f"invalid task index {index}; expected 0..{len(cases) - 1}")
    return cases[index]
