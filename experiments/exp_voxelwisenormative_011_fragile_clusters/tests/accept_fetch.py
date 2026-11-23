"""Accept a fetched real run of exp_011 before anyone uses its numbers (not a unit test: it needs the fetched files).

    python tests/accept_fetch.py --help

Checks, each written independently of src/modules:
  receipt    sha256 and size of the three outputs equal result.json; mode, shapes and counts are the expected ones;
  draws      sha256 of exp_005's sampled_cohort_rows (as '<i8') equals the receipt's draw digest;
  exp_005    held-out row 0: all 64 recurrence counts and refits 1 to 3 equal exp_005's saved bits;
  cpu replay for all 113 scans on the seven planes' columns, recurrence counts and refits 1 to 3 equal bootstrap-explainer's
             CPU replay (a different code path from the GPU job);
  full-site  the full-site flags equal |z| > 1.96 of exp_003's z_test.npy, up to float rounding at the cutoff;
  tables     bootstrap-ranking.tsv and scanner-ranking.tsv recomputed from flags.npz with scipy.ndimage.label.
Writes acceptance.json beside the files; any failed check stops the script.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
REFERENCE = ROOT / "experiments/shared/artifacts/pcntoolkit-blr-reference-n1000-v1/arrays.npz"
MASK = ROOT / "experiments/shared/artifacts/ixi-grey-matter-voxel-matrix-v1/mask.nii.gz"
N_SCANS, N_VOXELS, N_REFITS, KEEP = 113, 165_402, 64, 52
FACES = ndimage.generate_binary_structure(3, 1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=N_VOXELS, bitorder="little").astype(bool)


class Grid:
    """Largest face-connected component of a boolean mask-column vector, and its voxels on given axial slices."""

    def __init__(self) -> None:
        image = nib.load(MASK)
        assert isinstance(image, nib.Nifti1Image)
        self.mask = np.asarray(image.dataobj) != 0
        assert int(self.mask.sum()) == N_VOXELS

    def largest(self, columns: np.ndarray, slices: list[int]) -> tuple[int, list[int]]:
        volume = np.zeros(self.mask.shape, dtype=bool)
        volume[self.mask] = columns
        labels = np.zeros(volume.shape, dtype=np.int32)
        if ndimage.label(volume, structure=FACES, output=labels) == 0:
            return 0, [0] * len(slices)
        sizes = np.bincount(labels.ravel())[1:]
        chosen = int(np.flatnonzero(sizes == sizes.max())[0]) + 1  # first label in C order among equal sizes
        return int(sizes.max()), [int((labels[:, :, k] == chosen).sum()) for k in slices]


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder holding result.json, flags.npz and the two tables")
    parser.add_argument('--bootstrap-arrays', type=Path, required=True)
    parser.add_argument('--baseline-z', type=Path, required=True)
    parser.add_argument('--planes-dir', type=Path, required=True, help='separately generated CPU replay shards')
    args = parser.parse_args()
    folder = args.folder
    folder = folder if folder.is_absolute() else EXP / folder
    receipt = json.loads((folder / "result.json").read_text())
    report: dict[str, object] = {"folder": folder.name, "run": receipt["run_name"]}

    for name, info in receipt["outputs"].items():
        assert digest(folder / name) == info["sha256"] and (folder / name).stat().st_size == info["bytes"], name
    assert receipt["mode"] == "ixi-cuda" and receipt["held_out_scans"] == N_SCANS
    assert (receipt["voxel_columns"], receipt["references"], receipt["bootstrap_refits"]) == (N_VOXELS, 450, N_REFITS)
    assert receipt["flag_rule"] == "abs(z) > 1.96" and receipt["recurring_count_minimum"] == KEEP
    assert receipt["component_connectivity"] == 6
    report["receipt"] = "sha256 and size of 3 outputs equal result.json; mode ixi-cuda; 113 x 165402; 64 refits"

    with np.load(folder / "flags.npz", allow_pickle=False) as arrays:
        counts, first3, site = arrays["recurrence_counts"], unpack(arrays["first_three_flags_packed"]), unpack(arrays["site_flags_packed"])
        names = [str(n) for n in arrays["site_conditions"]]
        held_out = arrays["held_out_cohort_rows_zero_based"]
    assert counts.shape == (N_SCANS, N_VOXELS) and counts.max() <= N_REFITS and first3.shape == (3, N_SCANS, N_VOXELS)
    assert names == [c["name"] for c in receipt["site_conditions"]] == ["full-site", "no-site", "without-guys", "without-hh", "without-iop"]

    with np.load(args.bootstrap_arrays, allow_pickle=False) as old:
        sampled, bits = old["sampled_cohort_rows"], unpack(old["packed_trial_flags"])
    assert hashlib.sha256(np.ascontiguousarray(sampled, dtype="<i8").tobytes()).hexdigest() == receipt["bootstrap_cohort_rows_sha256"]
    report["draws"] = "exp_005 sampled_cohort_rows hash equals the receipt's draw digest"
    assert bits.shape == (N_REFITS, N_VOXELS)
    np.testing.assert_array_equal(counts[0], bits.sum(axis=0))
    np.testing.assert_array_equal(first3[:, 0], bits[:3])
    report["exp_005_row_0"] = "64 recurrence counts and refits 1 to 3 equal exp_005's saved bits on all 165,402 voxels"

    shards = sorted(glob.glob(str(args.planes_dir / 'rall-planes-*-of-*.npz')))
    if not shards:
        raise FileNotFoundError(f'{args.planes_dir}: no separately generated CPU replay shards')
    covered = off_counts = off_flags = 0
    for path in shards:
        with np.load(path) as part:
            cols, replay = part["columns"], part["flags"]
            assert replay.shape[:2] == (N_REFITS, N_SCANS)
            gap = counts[:, cols].astype(int) - replay.sum(axis=0)
            assert np.abs(gap).max() <= 1, f"{Path(path).name}: CPU and GPU counts differ by {np.abs(gap).max()}"
            off_counts += int(np.count_nonzero(gap))
            off_flags += int(np.count_nonzero(first3[:, :, cols] != replay[:3]))
            covered += len(cols)
    # The two devices round z differently; a voxel whose |z| sits on 1.96 can flip in a single refit.
    assert off_counts <= 1e-4 * N_SCANS * covered and off_flags <= 1e-4 * 3 * N_SCANS * covered, (off_counts, off_flags)
    report["cpu_replay"] = {"scans": N_SCANS, "plane_columns": covered, "counts_differing_by_1": off_counts,
                            "of_counts": N_SCANS * covered, "refit_1_to_3_flags_differing": off_flags}

    magnitude = np.abs(np.load(args.baseline_z, allow_pickle=False).astype(np.float64))
    wrong = site[0] != (magnitude > 1.96)
    nearest = float(np.abs(magnitude[wrong] - 1.96).max()) if wrong.any() else 0.0
    assert nearest < 1e-3, f"{int(wrong.sum())} full-site flags differ from exp_003 by up to {nearest}"
    report["full_site"] = {"flags_differing_from_exp_003_of_18.7M": int(wrong.sum()), "largest_distance_to_cutoff": nearest}

    grid = Grid()
    receipt_slices = receipt["axial_indices_zero_based"]
    rows = []
    for i in range(N_SCANS):
        unstable = (counts[i] > 0) & (counts[i] < KEEP)
        fragile = first3[0, i] & unstable
        size, on = grid.largest(fragile, receipt_slices)
        rows.append((size, int(fragile.sum()), i, int(unstable.sum()), on))
    rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
    table = read_table(folder / "bootstrap-ranking.tsv")
    assert len(table) == N_SCANS
    for rank, (size, flags, i, ever, on) in enumerate(rows, start=1):
        got = table[rank - 1]
        assert (int(got["rank"]), int(got["held_out_row_zero_based"]), int(got["largest_cluster_voxels"]),
                int(got["first_unstable_voxels"]), int(got["ever_unstable_voxels"]),
                got["largest_cluster_axial_voxels"]) == (rank, i, size, flags, ever, ",".join(map(str, on))), f"rank {rank}"
    report["bootstrap_ranking"] = "all 113 rows recomputed from flags.npz (rank, row, largest cluster, counts, slice voxels)"

    scanner = {(r["contrast"], int(r["held_out_row_zero_based"])): r for r in read_table(folder / "scanner-ranking.tsv")}
    assert len(scanner) == 4 * N_SCANS
    order = []
    for cond in names[1:]:
        control = {"no-site": "full-site"}.get(cond, "no-site")
        assert scanner[(cond, 0)]["compared_with"] == control
        for i in range(N_SCANS):
            gained, lost = site[names.index(cond), i] & ~site[names.index(control), i], site[names.index(control), i] & ~site[names.index(cond), i]
            (gs, gon), (ls, lon) = grid.largest(gained, receipt_slices), grid.largest(lost, receipt_slices)
            got = scanner[(cond, i)]
            assert (int(got["largest_flip_cluster_voxels"]), int(got["flipped_voxels"]), int(got["new_flags"]), int(got["removed_flags"]),
                    int(got["new_largest_cluster_voxels"]), int(got["removed_largest_cluster_voxels"]),
                    got["new_largest_cluster_axial_voxels"], got["removed_largest_cluster_axial_voxels"]) == (
                max(gs, ls), int(gained.sum() + lost.sum()), int(gained.sum()), int(lost.sum()), gs, ls,
                ",".join(map(str, gon)), ",".join(map(str, lon))), f"{cond} row {i}"
            order.append((-max(gs, ls), -int(gained.sum() + lost.sum()), i, cond))
    ranks = sorted(read_table(folder / "scanner-ranking.tsv"), key=lambda r: int(r["rank"]))
    expected = sorted(order, key=lambda r: (r[0], r[1], r[2]))
    assert [(r["contrast"], int(r["held_out_row_zero_based"])) for r in ranks[:25]] == [(c, i) for _, _, i, c in expected[:25]]
    report["scanner_ranking"] = "all 452 contrast rows recomputed from site flags; top 25 order equals the table"
    split = np.flatnonzero(np.load(REFERENCE)["test"])
    np.testing.assert_array_equal(held_out, split)
    report["held_out_rows"] = "the 113 cohort rows equal the pcntoolkit reference split's test rows"
    (folder / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
