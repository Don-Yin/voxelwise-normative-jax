"""The five fragility rules, the size classes and the shares they give, over per-cluster columns (numpy only, no files)."""
from __future__ import annotations

import numpy as np

from .inputs import N_REFITS, N_SCANS

HALF_REFITS = 4  # rule 3 (the project's own): in at least this many refits fewer than half of a cluster's voxels are flagged
BANDS = (0.6, 0.75, 0.85)  # rule 4, fpc manual p. 44 (not the paper): mean best Jaccard < 0.6 not to be trusted, < 0.75 doubtful, < 0.85 stable
DISSOLVED_AT_MOST, RECOVERED_ABOVE = 0.5, 0.75  # rule 4, Hennig 2007 p. 7: per refit; the paper sets no cut-off on how many refits
RESELECTED_ABOVE = (2, 3)  # rule 5, Roels 2016 thesis ch. 5: mean of recurrence_counts / 64 over the cluster above 2/3, the author's preset
SLACK = 1e-12  # a mean that is exactly a band edge counts as on the upper side, whatever the rounding
SIZE_CLASSES = {"all": 1, "ge10": 10, "ge50": 50, "ge100": 100}  # smallest cluster of each class, in voxels (8 mm3 each)
PROJECT, FPC, ROELS = "the project's own definition", "Hennig 2007; bands from the fpc manual p. 44, not from the paper", "Roels 2016 thesis ch. 5"
READINGS = {
    "no_core": {"rule": 1, "source": PROJECT, "fragile_if": "none of the cluster's voxels is flagged in all 64 refits"},
    "vanishes": {"rule": 2, "source": PROJECT, "fragile_if": "in at least one refit none of the cluster's voxels is flagged"},
    "loses_half": {"rule": 3, "source": PROJECT,
                   "fragile_if": f"in at least {HALF_REFITS} of the 64 refits fewer than half of the cluster's voxels are flagged"},
    "jaccard_not_trusted": {"rule": 4, "source": FPC, "fragile_if": f"mean over 64 refits of the best Jaccard overlap is below {BANDS[0]}"},
    "jaccard_not_stable": {"rule": 4, "source": FPC, "fragile_if": f"the same mean is below {BANDS[1]} (doubtful or not to be trusted)"},
    "low_reselection": {"rule": 5, "source": f"{ROELS}; 2/3 is the author's preset",
                        "fragile_if": "mean over the cluster's voxels of recurrence_counts / 64 is at most 2/3"},
}


def band_of(mean: np.ndarray) -> np.ndarray:
    """0 not to be trusted, 1 doubtful, 2 stable, 3 highly stable: the fpc manual's bands of a cluster's mean best Jaccard."""
    return np.searchsorted(np.asarray(BANDS) - SLACK, mean, side="right")


def readings(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Which clusters each reading of the five rules calls fragile."""
    inside, size, band = c["refit_voxels"].astype(np.int64), c["voxels"].astype(np.int64), band_of(c["jaccard_mean"])
    if inside.shape[1] != N_REFITS:
        raise ValueError(f"expected per-refit counts for {N_REFITS} refits, got {inside.shape}")
    return {"no_core": c["core_voxels"] == 0,
            "vanishes": (inside == 0).any(axis=1),
            "loses_half": (2 * inside < size[:, None]).sum(axis=1) >= HALF_REFITS,
            "jaccard_not_trusted": band == 0,
            "jaccard_not_stable": band <= 1,
            "low_reselection": RESELECTED_ABOVE[1] * c["recurrence_sum"].astype(np.int64) <= RESELECTED_ABOVE[0] * N_REFITS * size}


def quartiles(values: np.ndarray) -> dict[str, float] | None:
    """First quartile, median and third quartile (linear interpolation), or None when there are no values."""
    if not len(values):
        return None
    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return {"q1": q1, "median": median, "q3": q3}


def times_five(stats: dict[str, float] | None) -> dict[str, float] | None:
    """The same numbers as 'x of 5 clusters'."""
    return None if stats is None else {key: 5 * value for key, value in stats.items()}


def shares(size: np.ndarray, person: np.ndarray, bad: np.ndarray) -> dict:
    """Pooled and per-person fragile share of one set of clusters, and the share of its voxels that sit in fragile ones.

    `person` is the cohort row of each cluster's owner; people without a cluster in the set are left out of the
    per-person numbers.
    """
    total = np.bincount(person, minlength=N_SCANS)
    fragile = np.bincount(person[bad], minlength=N_SCANS)
    volume = np.bincount(person, weights=size, minlength=N_SCANS)
    fragile_volume = np.bincount(person[bad], weights=size[bad], minlength=N_SCANS)
    have = total > 0
    pooled = float(fragile.sum() / total.sum()) if total.sum() else None
    per_person = quartiles(fragile[have] / total[have])
    return {"clusters": int(total.sum()), "fragile": int(fragile.sum()), "pooled_share": pooled,
            "pooled_x_of_5": None if pooled is None else 5 * pooled, "people_with_clusters": int(have.sum()),
            "per_person_share": per_person, "per_person_x_of_5": times_five(per_person),
            "flagged_voxels": int(volume.sum()), "fragile_voxels": int(fragile_volume.sum()),
            "fragile_volume_share": float(fragile_volume.sum() / volume.sum()) if volume.sum() else None,
            "per_person_fragile_volume_share": quartiles(fragile_volume[have] / volume[have])}


def band_counts(mean: np.ndarray) -> dict:
    """Clusters in each of the fpc manual's bands of the mean best Jaccard, as counts and as shares of the set."""
    found = np.bincount(band_of(mean), minlength=4)
    names = ("not_trusted", "doubtful", "stable", "highly_stable")
    return {"clusters": len(mean), **{name: int(n) for name, n in zip(names, found, strict=True)},
            "share": {name: float(n / len(mean)) if len(mean) else None for name, n in zip(names, found, strict=True)}}


def count_summary(counts: np.ndarray) -> dict | None:
    """Distribution over clusters of 'in how many of the 64 refits': histogram from 0 to 64, mean and quartiles."""
    stats = quartiles(counts)
    if stats is None:
        return None
    return {"histogram": np.bincount(counts, minlength=N_REFITS + 1).tolist(), "mean": float(counts.mean()), **stats}


def size_bins(largest: int) -> tuple[np.ndarray, np.ndarray]:
    """Doubling size bins: 1, 2, 3 to 4, 5 to 8, ... as (smallest, largest) voxels, until `largest` is inside the last."""
    upper = 2 ** np.arange(int(np.ceil(np.log2(max(largest, 2)))) + 1)
    return np.r_[1, upper[:-1] + 1], upper


def curve(size: np.ndarray, bad: np.ndarray, upper: np.ndarray) -> dict[str, list]:
    """Clusters, fragile clusters and their share in each size bin (null share where the bin is empty)."""
    index = np.searchsorted(upper, size)
    total, fragile = np.bincount(index, minlength=len(upper)), np.bincount(index[bad], minlength=len(upper))
    return {"clusters": total.tolist(), "fragile": fragile.tolist(),
            "share": [float(f / t) if t else None for f, t in zip(fragile, total, strict=True)]}


def count_curve(counts: np.ndarray, size: np.ndarray, upper: np.ndarray) -> dict[str, list]:
    """Mean and quartiles of a per-refit count in each size bin (null where the bin is empty)."""
    index, out = np.searchsorted(upper, size), {"mean": [], "q1": [], "median": [], "q3": []}
    for b in range(len(upper)):
        s = count_summary(counts[index == b])
        for key, values in out.items():
            values.append(None if s is None else s[key])
    return out


def summarise(c: dict[str, np.ndarray], folds: list[int]) -> dict:
    """Per scope (each fold, then all): shares by reading and size class, Jaccard bands and counts, size curves, fold spread."""
    flags = readings(c)
    per_refit = {"dissolutions": (c["refit_jaccard"] <= DISSOLVED_AT_MOST).sum(axis=1),
                 "recoveries": (c["refit_jaccard"] > RECOVERED_ABOVE).sum(axis=1)}
    lower, upper = size_bins(int(c["voxels"].max()))
    scopes = {f"fold{k}": c["fold"] == k for k in folds} | {"all": np.ones(len(c["fold"]), dtype=bool)}
    bins = {"size_from": lower.tolist(), "size_to": upper.tolist()}
    out: dict = {"scopes": {}, "size_curve": {**bins, "scopes": {}}, "refit_count_curve": {**bins, "scopes": {}}}
    for name, scope in scopes.items():
        picks = {cls: scope & (c["voxels"] >= low) for cls, low in SIZE_CLASSES.items()}
        out["scopes"][name] = {
            "people": len(np.unique(c["cohort_row"][scope])),
            "readings": {rule: {"size_classes": {cls: shares(c["voxels"][pick], c["cohort_row"][pick], bad[pick]) for cls, pick in picks.items()}}
                         for rule, bad in flags.items()},
            "jaccard_bands": {cls: band_counts(c["jaccard_mean"][pick]) for cls, pick in picks.items()},
            "jaccard_counts": {cls: {kind: count_summary(v[pick]) for kind, v in per_refit.items()} for cls, pick in picks.items()}}
        out["size_curve"]["scopes"][name] = {rule: curve(c["voxels"][scope], bad[scope], upper) for rule, bad in flags.items()}
        out["refit_count_curve"]["scopes"][name] = {kind: count_curve(v[scope], c["voxels"][scope], upper) for kind, v in per_refit.items()}
    if len(folds) > 1:
        out["fold_spread"] = fold_spread(out["scopes"], folds)
    return out


def fold_spread(scopes: dict, folds: list[int]) -> dict:
    """How far the folds' pooled shares lie apart, per reading and size class (fraction of clusters, and the same as x of 5)."""
    spread: dict = {rule: {} for rule in READINGS}
    for rule, cls in ((r, k) for r in READINGS for k in SIZE_CLASSES):
        by_fold = [scopes[f"fold{k}"]["readings"][rule]["size_classes"][cls]["pooled_share"] for k in folds]
        everyone = scopes["all"]["readings"][rule]["size_classes"][cls]["pooled_share"]
        if None in by_fold or everyone is None:
            raise ValueError(f"a fold has no cluster of class {cls} under {rule}; pooled shares by fold {by_fold}, all people {everyone}")
        spread[rule][cls] = {"pooled_share_by_fold": by_fold, "min": min(by_fold), "max": max(by_fold),
                             "range": max(by_fold) - min(by_fold), "range_x_of_5": 5 * (max(by_fold) - min(by_fold)),
                             "all_people": everyone, "largest_distance_from_all": max(abs(v - everyone) for v in by_fold)}
    return {"folds": folds, "readings": spread}
