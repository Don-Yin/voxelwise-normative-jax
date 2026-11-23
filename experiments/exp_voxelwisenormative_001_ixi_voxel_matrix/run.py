"""IXI grey matter as one subjects x voxels matrix for the voxelwise normative benchmark.

Reads the published reference-chain maps (563 IXI, gm-modulated-s8), checks every file against
subjects.tsv and the template receipt, and writes:
  gm.npy          float32 (subjects, voxels), rows in subjects.tsv order, columns in mask C order
  mask.nii.gz     uint8 analysis mask on the template grid
  covariates.tsv  subject, age, sex, site
  receipt.json    input and output SHA-256, mask rule, shape
Mask rule: template grey matter >= 0.2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / ".results"
THRESHOLD = 0.2
N_SUBJECTS = 563

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(EXPERIMENT_DIR.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """parse command line arguments for this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivatives", type=Path, required=True)
    parser.add_argument("--subjects-sha256", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--no-profile", action="store_true", help="skip the default pyinstrument html report")
    parser.add_argument("--resources", action="store_true", help="log wall, cpu, and peak rss for this run")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Check inputs, build the mask and the matrix, then publish them with a receipt."""
    import nibabel as nib
    import numpy as np

    root = args.derivatives
    table = root / "subjects.tsv"
    template_path = root / "template" / "template_GM.nii.gz"
    if sha256(table) != args.subjects_sha256 or sha256(template_path) != args.template_sha256:
        raise RuntimeError("subjects.tsv or template differs from the supplied digest")
    with open(table, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    valid = {r['subject'] for r in rows if r['status'] == 'complete'
             and re.fullmatch(r'IXI\d{3}-(Guys|HH|IOP)-\d{4}-T1', r['subject'])}
    if len(rows) != N_SUBJECTS or len(valid) != N_SUBJECTS:
        raise RuntimeError('expected 563 distinct complete IXI subjects')

    def load(path: Path) -> nib.Nifti1Image:
        image = nib.load(path)
        if not isinstance(image, nib.Nifti1Image):
            raise RuntimeError(f"not NIfTI-1: {path}")
        return image

    template = load(template_path)
    mask = np.asarray(template.dataobj, dtype=np.float32) >= THRESHOLD
    n_voxels = int(mask.sum())
    logger.info("mask: %d voxels (template GM >= %.1f)", n_voxels, THRESHOLD)

    gm = np.empty((len(rows), n_voxels), dtype=np.float32)
    for i, row in enumerate(rows):
        path = root / row["subject"] / "gm-modulated-s8.nii.gz"
        if sha256(path) != row["gm_modulated_s8_sha256"]:
            raise RuntimeError(f"map changed: {row['subject']}")
        image = load(path)
        if image.shape != template.shape or not np.allclose(image.get_qform(), template.get_qform()):
            raise RuntimeError(f"grid differs: {row['subject']}")
        gm[i] = np.asarray(image.dataobj, dtype=np.float32)[mask]
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            logger.info("[%d/%d] loaded", i + 1, len(rows))
    if not np.isfinite(gm).all():
        raise RuntimeError("non-finite grey matter values")

    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gm.npy": lambda p: np.save(p, gm),
        "mask.nii.gz": lambda p: nib.save(nib.Nifti1Image(mask.astype(np.uint8), template.affine, template.header), p),
        "covariates.tsv": lambda p: p.write_text(
            "subject\tage\tsex\tsite\n" + "".join(f"{r['subject']}\t{r['age']}\t{r['sex']}\t{r['site']}\n" for r in rows)
        ),
    }
    receipt = {
        "source_dataset": "https://brain-development.org/ixi-dataset/",
        "inputs": {"subjects_tsv": table.name, "subjects_sha256": args.subjects_sha256,
                   "template": template_path.name, "template_sha256": args.template_sha256},
        "mask_rule": f"template_GM >= {THRESHOLD}",
        "shape": [len(rows), n_voxels],
        "outputs": {},
    }
    for name, write in outputs.items():
        partial = RESULTS / f".{name}.partial{''.join(Path(name).suffixes)}"
        write(partial)
        partial.replace(RESULTS / name)
        receipt["outputs"][name] = sha256(RESULTS / name)
    (RESULTS / 'json').mkdir(exist_ok=True)
    (RESULTS / 'json' / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    logger.info("wrote %s, shape %s, mean GM %.4f", RESULTS, gm.shape, float(gm.mean()))


if __name__ == "__main__":
    args = parse_args()
    try:
        from src.utils.profiling import profiled, run_resources  # pyright: ignore[reportMissingImports]
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def run_resources(*, enabled: bool = True, python_peak: bool = False):
            """stdlib-only fallback when the tier-1 helper is not present."""
            del enabled, python_peak
            yield

        @contextmanager
        def profiled(out: Path | None):
            """lazy pyinstrument fallback when the tier-1 helper is not present."""
            if out is None:
                yield
                return
            from pyinstrument import Profiler

            profiler = Profiler()
            profiler.start()
            try:
                yield
            finally:
                profiler.stop()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(profiler.output_html())

    with (
        run_resources(enabled=args.resources),
        profiled(None if args.no_profile else RESULTS / "profile" / "run.html"),
    ):
        main(args)
