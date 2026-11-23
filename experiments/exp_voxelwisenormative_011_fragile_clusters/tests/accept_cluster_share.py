"""Check a finished cluster-share run against the fetched arrays with code that shares nothing with src/modules.

    PYTHONPATH="$PWD" micromamba run -n voxelwise-normative-jax python \
        experiments/exp_voxelwisenormative_011_fragile_clusters/tests/accept_cluster_share.py <run-name> [--people 1]

1. Every file listed in the run's result.json (its outputs and the fetched inputs) has the recorded SHA-256 and size.
2. In every fold, each person's clusters cover exactly the voxels with |z| > T of the single fit (T: the run's cutoff), at every connectivity.
3. The first --people people of each fold are labelled again by plain flood fill (naive_stability.py) and compared cluster by
   cluster: sign, voxels, core voxels, voxels in each refit (exact) and each refit's best Jaccard (to 1e-6, the stored float32).
4. recurrence_sum equals the sum of refit_voxels for every cluster.
5. Every number the readme and the figures quote (counts, pooled and per-person shares, volume shares, Jaccard bands, per-refit
   counts, size curves, fold spread) is recomputed from the stored per-cluster columns in plain Python and compared with result.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import naive_stability as naive
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT), str(EXPERIMENT.parents[1])]
# restated here on purpose: a change in the code under test must show up as a difference
SIZE_CLASSES = {"all": 1, "ge10": 10, "ge50": 50, "ge100": 100}
RULES = ("no_core", "vanishes", "loses_half", "jaccard_not_trusted", "jaccard_not_stable", "low_reselection")
BANDS = ("not_trusted", "doubtful", "stable", "highly_stable")
EPS = 1e-9


def check(ok: bool, message: str) -> None:
    """Stop with the offending values named."""
    if not ok:
        raise SystemExit(f"FAILED: {message}")


def agree(found: Any, wanted: Any) -> bool:
    """Equal: integers and lists exactly, floats to 1e-12, None only to None, tuples against the json's {q1, median, q3}."""
    if isinstance(wanted, dict):  # the json's {q1, median, q3}
        return agree(found, tuple(wanted[k] for k in ("q1", "median", "q3")))
    if found is None or wanted is None:
        return found is wanted
    if isinstance(found, (tuple, list)):
        return len(found) == len(wanted) and all(agree(a, b) for a, b in zip(found, wanted, strict=True))
    return abs(found - wanted) <= 1e-12 * max(1.0, abs(wanted))


def quartiles(values: Sequence[float]) -> tuple[float, float, float] | None:
    """Quartiles by linear interpolation (the 'inclusive' method), without numpy."""
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0], values[0]
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q2, q3


def readings(c: dict, i: int) -> dict[str, bool]:
    """The six readings of cluster i, written out literally (an exact fraction for the 2/3 rule)."""
    size, each, mean = int(c["voxels"][i]), [int(k) for k in c["refit_voxels"][i]], float(c["jaccard_mean"][i])
    return {"no_core": int(c["core_voxels"][i]) == 0, "vanishes": min(each) == 0, "loses_half": sum(2 * k < size for k in each) >= 4,
            "jaccard_not_trusted": mean < 0.6 - EPS, "jaccard_not_stable": mean < 0.75 - EPS,
            "low_reselection": Fraction(sum(each), 64 * size) <= Fraction(2, 3)}


def summary(rows: list[tuple[int, int, bool]]) -> dict:
    """Counts, shares and volume of one set of clusters from (person, voxels, fragile) rows."""
    people: dict[int, list[int]] = {}
    for person, size, bad in rows:
        t = people.setdefault(person, [0, 0, 0, 0])  # clusters, fragile clusters, voxels, voxels in fragile clusters
        t[0] += 1
        t[1] += bad
        t[2] += size
        t[3] += size if bad else 0
    fragile, volume, fragile_volume = sum(t[1] for t in people.values()), sum(t[2] for t in people.values()), sum(t[3] for t in people.values())
    return {"clusters": len(rows), "fragile": fragile, "pooled_share": fragile / len(rows), "people_with_clusters": len(people),
            "per_person_share": quartiles([t[1] / t[0] for t in people.values()]), "flagged_voxels": volume,
            "fragile_voxels": fragile_volume, "fragile_volume_share": fragile_volume / volume,
            "per_person_fragile_volume_share": quartiles([t[3] / t[2] for t in people.values()])}


def same_clusters(c: dict, own: np.ndarray, plain: dict) -> bool:
    """Stored rows `own` and the flood fill's clusters agree: integer facts exactly, Jaccard values to float32 precision."""
    def group(items):
        groups: dict[tuple, list] = {}
        for key, mean, each in items:
            groups.setdefault(key, []).append((mean, np.asarray(each, dtype=float)))
        return {key: sorted(v, key=lambda t: t[0]) for key, v in groups.items()}

    got = group(((int(c["sign"][i]), int(c["voxels"][i]), int(c["core_voxels"][i]), tuple(int(k) for k in c["refit_voxels"][i])),
                 float(c["jaccard_mean"][i]), c["refit_jaccard"][i]) for i in np.flatnonzero(own))
    want = group(((s, v["voxels"], v["core_voxels"], tuple(v["refit_voxels"])), v["jaccard_mean"], v["refit_jaccard"]) for (s, _), v in plain.items())
    return got.keys() == want.keys() and all(
        len(got[k]) == len(want[k]) and all(abs(a[0] - b[0]) < EPS and np.allclose(a[1], b[1], atol=1e-6) for a, b in zip(got[k], want[k], strict=True))
        for k in got)


def check_files(results: Path, result: dict, sha256) -> None:
    """Check 1: every output and input file has the digest and size result.json records."""
    records = [*result["outputs"].values(), *(f for run in result["inputs"].values() for f in run["files"].values())]
    for record in records:
        path = results / record["path"]
        check((sha256(path), path.stat().st_size) == (record["sha256"], record["bytes"]), f"{path} differs from result.json")
    print(f"1. {len(records)} files match their recorded sha256 and size")


def check_numbers(result: dict, n: int, c: dict) -> dict:
    """Check 5 for one connectivity; returns the independent pooled shares by (scope, rule, class)."""
    block, size, person, fold = result[f"connectivity_{n}"], c["voxels"].tolist(), c["cohort_row"].tolist(), c["fold"].tolist()
    flags = [readings(c, i) for i in range(len(size))]
    mean, jaccard = c["jaccard_mean"].tolist(), c["refit_jaccard"].tolist()
    dissolved = [sum(j <= 0.5 for j in row) for row in jaccard]
    recovered = [sum(j > 0.75 for j in row) for row in jaccard]
    pooled, bins = {}, [(s - 1).bit_length() for s in size]
    lower, upper = block["size_curve"]["size_from"], block["size_curve"]["size_to"]
    check(upper == [2**b for b in range(len(upper))] and lower == [1] + [2 ** (b - 1) + 1 for b in range(1, len(upper))], f"{n}: size bins differ")
    check(max(bins) == len(upper) - 1, f"{n}: the last size bin is not the one holding the largest cluster")
    for scope, data in block["scopes"].items():
        in_scope = [i for i in range(len(size)) if scope == "all" or fold[i] == int(scope.removeprefix("fold"))]
        check(len({person[i] for i in in_scope}) == data["people"], f"{n} {scope}: people differ")
        for cls, low in SIZE_CLASSES.items():
            picked = [i for i in in_scope if size[i] >= low]
            for rule in RULES:
                s = summary([(person[i], size[i], flags[i][rule]) for i in picked])
                pooled[scope, rule, cls] = s["pooled_share"]
                found = data["readings"][rule]["size_classes"][cls]
                check(all(agree(v, found[k]) for k, v in s.items()) and agree(5 * s["pooled_share"], found["pooled_x_of_5"]), f"{n} {scope} {rule} {cls}: {found} vs {s}")
            edges = [0.6, 0.75, 0.85]
            band = [sum(mean[i] >= e - EPS for e in edges) for i in picked]
            want = {"clusters": len(picked), **{name: band.count(k) for k, name in enumerate(BANDS)}}
            found = data["jaccard_bands"][cls]
            check(all(found[k] == v for k, v in want.items()) and all(agree(want[k] / len(picked), found["share"][k]) for k in BANDS), f"{n} {scope} bands {cls}")
            for kind, counts in (("dissolutions", dissolved), ("recoveries", recovered)):
                values, found = [counts[i] for i in picked], data["jaccard_counts"][cls][kind]
                check(found["histogram"] == [values.count(k) for k in range(65)] and agree(statistics.fmean(values), found["mean"])
                      and agree(quartiles(values), found), f"{n} {scope} {kind} {cls}")
        for rule in RULES:
            found = block["size_curve"]["scopes"][scope][rule]
            total = [sum(bins[i] == b for i in in_scope) for b in range(len(upper))]
            bad = [sum(bins[i] == b and flags[i][rule] for i in in_scope) for b in range(len(upper))]
            check(found["clusters"] == total and found["fragile"] == bad and all(agree(f / t if t else None, s)
                  for f, t, s in zip(bad, total, found["share"], strict=True)), f"{n} {scope} {rule}: size curve")
        for kind, counts in (("dissolutions", dissolved), ("recoveries", recovered)):
            found = block["refit_count_curve"]["scopes"][scope][kind]
            for b in range(len(upper)):
                values = [counts[i] for i in in_scope if bins[i] == b]
                q = quartiles(values)
                check(agree(statistics.fmean(values) if values else None, found["mean"][b])
                      and all(agree(None if q is None else q[k], found[name][b]) for k, name in enumerate(("q1", "median", "q3"))), f"{n} {scope} {kind} bin {b}")
    return pooled


def check_spread(result: dict, n: int, pooled: dict) -> None:
    """Fold spread of one connectivity from the independent pooled shares."""
    spread, folds = result[f"connectivity_{n}"]["fold_spread"], result["folds"]
    check(spread["folds"] == folds, f"{n}: fold spread lists other folds")
    for rule in RULES:
        for cls in SIZE_CLASSES:
            by_fold, everyone, found = [pooled[f"fold{k}", rule, cls] for k in folds], pooled["all", rule, cls], spread["readings"][rule][cls]
            want = {"pooled_share_by_fold": by_fold, "min": min(by_fold), "max": max(by_fold), "range": max(by_fold) - min(by_fold),
                    "range_x_of_5": 5 * (max(by_fold) - min(by_fold)), "all_people": everyone,
                    "largest_distance_from_all": max(abs(v - everyone) for v in by_fold)}
            check(all(agree(v, found[k]) for k, v in want.items()), f"{n} fold spread {rule} {cls}")


def load_fold(results: Path, result: dict, fold: int, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fold's cohort rows, single-fit z and packed refit flags at `threshold`, read straight from the files the run lists."""
    files = result["inputs"][f"fold{fold}"]["files"]
    with np.load(results / files["z-summary.npz"]["path"]) as archive:
        z, rows = archive["original_z"], archive["held_out_cohort_rows_zero_based"]
    if threshold == 1.96:
        with np.load(results / files["refit-flags.npz"]["path"]) as archive:
            return rows, z, archive["refit_flags_packed"]
    with np.load(results / files["cutoff-flags.npz"]["path"]) as archive:
        return rows, z, archive["refit_flags_packed"][list(archive["cutoffs"]).index(threshold)]


def main() -> None:
    """Run the five checks on one run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--people", type=int, default=1, help="people per fold to re-measure by flood fill")
    args = parser.parse_args()
    from src.modules.inputs import (
        read_mask,  # imported here: sys.path is extended above
    )

    from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve, sha256

    results = EXPERIMENT / ".results"
    result = json.loads((results / "json" / args.run / "result.json").read_text())
    threshold = result["threshold"]
    check(result["refits"] == 64 and f"z > {threshold:g}" in result["definitions"]["cluster"], f"refits {result['refits']} or the documented cutoff differ from {threshold}")
    check(result["definitions"]["size_classes_min_voxels"] == SIZE_CLASSES, "documented size classes differ from the ones checked")
    connectivities = sorted(int(k.removeprefix("connectivity_")) for k in result if k.startswith("connectivity_"))
    check(connectivities == [6, 18, 26], f"connectivities {connectivities}")
    check_files(results, result, sha256)
    mask, _, _ = read_mask(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
    stored = {n: dict(np.load(results / "arrays" / args.run / f"clusters-{n}.npz")) for n in connectivities}
    for n, c in stored.items():
        check(np.array_equal(c["recurrence_sum"], c["refit_voxels"].sum(axis=1, dtype=np.int64)), f"{n}: recurrence_sum differs from the refit voxels")
    print("4. recurrence_sum equals the sum of refit_voxels for every cluster at every connectivity")
    for fold in result["folds"]:
        rows, z, packed = load_fold(results, result, fold, threshold)
        flagged = (np.abs(z) > threshold).sum(axis=1)
        for n, c in stored.items():
            mine = c["fold"] == fold
            covered = np.bincount(c["person"][mine], weights=c["voxels"][mine], minlength=len(flagged))
            check(np.array_equal(covered, flagged), f"fold {fold}, {n} neighbours: cluster voxels differ from the single-fit flags")
            check(np.array_equal(c["cohort_row"][mine], rows[c["person"][mine]]), f"fold {fold}: cohort rows differ")
        for p in range(args.people):
            refits = np.unpackbits(packed[:, p, :], axis=-1, count=int(mask.sum()), bitorder="little").view(bool)
            for n, c in stored.items():
                plain = naive.person(mask, z[p], refits, refits.sum(axis=0), n, threshold)
                check(same_clusters(c, (c["fold"] == fold) & (c["person"] == p), plain), f"fold {fold} person {p}, {n} neighbours: differs from the flood fill")
        print(f"2./3. fold {fold}: clusters cover the single-fit flags at {len(stored)} connectivities; {args.people} person(s) re-labelled by flood fill")
    for n, c in stored.items():
        pooled = check_numbers(result, n, c)
        if "fold_spread" in result[f"connectivity_{n}"]:
            check_spread(result, n, pooled)
        print(f"5. {n} neighbours: {len(c['voxels']):,} clusters; counts, shares, volume, bands, per-refit counts, curves and spread recomputed and equal")
    print("ACCEPTED", args.run)


if __name__ == "__main__":
    main()
