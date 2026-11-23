"""What the synthetic runs leave behind, as digests, to prove that new options leave the old outputs alone.

`digest` ran on the code from before `--z-summary` existed and its fixed result is EXPECTED_OUTPUTS;
test_z_summary.py runs it again on the current code and requires the same answer.  Arrays are compared by their bytes,
dtype, shape and order; a receipt by its text without the fields that change on every run (wall-clock seconds and the
size and checksum of the compressed archive, which depend on the zlib build).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest import mock

import numpy as np
from src.modules.study import run, run_per_refit

WALL_CLOCK = {"refit_seconds", "seconds_before_export", "fit_seconds"}
FIXED_ZIP_TIME = time.struct_time((2026, 1, 1, 0, 0, 0, 3, 1, 0))
RUNS = (("main", {}), ("refits", {}), ("fold-2", {"fold": 2}), ("n24", {"reference_n": 24}))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def steady(value):
    """The receipt's value with the wall-clock fields and the compressed archive's checksum and size blanked."""
    if isinstance(value, dict):
        return {key: None if key in WALL_CLOCK or key in {"sha256", "bytes"} else steady(item) for key, item in value.items()}
    if isinstance(value, list):
        return [steady(item) for item in value]
    return value


def digest(results: Path) -> dict[str, object]:
    """One entry per file below `results`: array digests for an npz, the steady text for a receipt, bytes otherwise."""
    out: dict[str, object] = {}
    for path in sorted(p for p in results.rglob("*") if p.is_file()):
        name = path.relative_to(results).as_posix()
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                out[name] = {"keys": list(archive.files),
                             "arrays": {key: [str(archive[key].dtype), list(archive[key].shape),
                                              sha(np.ascontiguousarray(archive[key]).tobytes())] for key in archive.files}}
        elif path.name == "result.json":
            out[name] = {"keys": list(json.loads(path.read_text())),
                         "steady_text_sha256": sha(json.dumps(steady(json.loads(path.read_text())), indent=2).encode())}
        else:
            out[name] = sha(path.read_bytes())
    return out


def synthetic_runs(results: Path) -> None:
    """The four runs, with the zip time fixed so that the archives are the same bytes on every run of one machine."""
    with mock.patch.object(time, "localtime", return_value=FIXED_ZIP_TIME):
        run(results, "synthetic-main", 32768, use_fixture=True)
        for name, options in RUNS[1:]:
            run_per_refit(results, f"synthetic-{name}", 32768, use_fixture=True, **options)


# Fixed outputs of the synthetic runs before the new option.
EXPECTED_OUTPUTS = {
  "arrays/synthetic-fold-2/refit-flags.npz": {
    "keys": [
      "refit_flags_packed",
      "recurrence_counts",
      "held_out_cohort_rows_zero_based"
    ],
    "arrays": {
      "refit_flags_packed": [
        "uint8",
        [
          64,
          6,
          40
        ],
        "51b9c0660493b7307d29f4897740b382053e961613b73300014e6ed81ab011c7"
      ],
      "recurrence_counts": [
        "uint8",
        [
          6,
          320
        ],
        "c37bd22476a4a86883d040606be7a1111f880c56bf6b5a904f7ae090de12230a"
      ],
      "held_out_cohort_rows_zero_based": [
        "int32",
        [
          6
        ],
        "6b0a79e41cdfac4c653ea957247b0d3e922450b6ed1a0c653654f8f18a091c79"
      ]
    }
  },
  "arrays/synthetic-main/flags.npz": {
    "keys": [
      "recurrence_counts",
      "first_three_flags_packed",
      "site_flags_packed",
      "site_conditions",
      "held_out_cohort_rows_zero_based"
    ],
    "arrays": {
      "recurrence_counts": [
        "uint8",
        [
          4,
          320
        ],
        "1220a74dfa13b4b8aeebc4f0e8ec3f7ba70f39851f20657c7256237a1f98af24"
      ],
      "first_three_flags_packed": [
        "uint8",
        [
          3,
          4,
          40
        ],
        "2188172d29ef42af66a9a27725423baf60a151339d8eae4bffad2a11550ecf88"
      ],
      "site_flags_packed": [
        "uint8",
        [
          5,
          4,
          40
        ],
        "23b946f59c7b6e16062ef2aa47b0c21b1bc1ad678745090e5ac21324ef9e6ccb"
      ],
      "site_conditions": [
        "<U12",
        [
          5
        ],
        "b8f93b7232813180a0049685ae3cd43cf30067beff52f67c7707641c41b6f7c0"
      ],
      "held_out_cohort_rows_zero_based": [
        "int32",
        [
          4
        ],
        "1cf37c9507affafe5f304a568f40306718e5507697d76a4d054b4f64e543e6dc"
      ]
    }
  },
  "arrays/synthetic-n24/refit-flags.npz": {
    "keys": [
      "refit_flags_packed",
      "recurrence_counts",
      "held_out_cohort_rows_zero_based",
      "original_flags_packed",
      "reference_cohort_rows_zero_based"
    ],
    "arrays": {
      "refit_flags_packed": [
        "uint8",
        [
          64,
          4,
          40
        ],
        "22612d3940a04f5a3e19fd1b0a9da7de6f32366160972d4491fdb3a72ba317fd"
      ],
      "recurrence_counts": [
        "uint8",
        [
          4,
          320
        ],
        "1220a74dfa13b4b8aeebc4f0e8ec3f7ba70f39851f20657c7256237a1f98af24"
      ],
      "held_out_cohort_rows_zero_based": [
        "int32",
        [
          4
        ],
        "1cf37c9507affafe5f304a568f40306718e5507697d76a4d054b4f64e543e6dc"
      ],
      "original_flags_packed": [
        "uint8",
        [
          4,
          40
        ],
        "7721afac6108e5ff801a538bceb0332561611cc42893ad7fc14671ac2c317f92"
      ],
      "reference_cohort_rows_zero_based": [
        "int32",
        [
          24
        ],
        "a26f2589bc817e205aed8ed29161a2538dbe40952ed97c98974e90b4b056d4b4"
      ]
    }
  },
  "arrays/synthetic-refits/refit-flags.npz": {
    "keys": [
      "refit_flags_packed",
      "recurrence_counts",
      "held_out_cohort_rows_zero_based"
    ],
    "arrays": {
      "refit_flags_packed": [
        "uint8",
        [
          64,
          4,
          40
        ],
        "22612d3940a04f5a3e19fd1b0a9da7de6f32366160972d4491fdb3a72ba317fd"
      ],
      "recurrence_counts": [
        "uint8",
        [
          4,
          320
        ],
        "1220a74dfa13b4b8aeebc4f0e8ec3f7ba70f39851f20657c7256237a1f98af24"
      ],
      "held_out_cohort_rows_zero_based": [
        "int32",
        [
          4
        ],
        "1cf37c9507affafe5f304a568f40306718e5507697d76a4d054b4f64e543e6dc"
      ]
    }
  },
  "json/synthetic-fold-2/result.json": {
    "keys": [
      "mode",
      "runtime",
      "held_out_scans",
      "voxel_columns",
      "references",
      "bootstrap_refits",
      "bootstrap_seed",
      "bootstrap_cohort_rows_sha256",
      "optimizer_iterations",
      "chunk",
      "flag_rule",
      "design",
      "fold",
      "folds",
      "reference_cohort_rows_sha256",
      "mask_column_order",
      "packed_axis",
      "packed_bitorder",
      "refit_seconds",
      "seconds_before_export",
      "schema_version",
      "run_name",
      "outputs"
    ],
    "steady_text_sha256": "7ad802ce5f9d3e314d4cbc33fb6c971c2ab5439c91e90c76baf65bf0532f3ad9"
  },
  "json/synthetic-main/result.json": {
    "keys": [
      "mode",
      "runtime",
      "held_out_scans",
      "voxel_columns",
      "references",
      "bootstrap_refits",
      "bootstrap_seed",
      "bootstrap_cohort_rows_sha256",
      "optimizer_iterations",
      "chunk",
      "flag_rule",
      "recurring_count_minimum",
      "component_connectivity",
      "mask_column_order",
      "axial_mm",
      "axial_indices_zero_based",
      "refit_seconds",
      "site_conditions",
      "seconds_before_export",
      "schema_version",
      "run_name",
      "packed_axis",
      "packed_bitorder",
      "outputs"
    ],
    "steady_text_sha256": "59ee356e1c1e9f3ff7e1ae88f9b50dc8f748ed9f478972943c651fb184e8dceb"
  },
  "json/synthetic-n24/result.json": {
    "keys": [
      "mode",
      "runtime",
      "held_out_scans",
      "voxel_columns",
      "references",
      "bootstrap_refits",
      "bootstrap_seed",
      "bootstrap_cohort_rows_sha256",
      "optimizer_iterations",
      "chunk",
      "flag_rule",
      "design",
      "fold",
      "folds",
      "reference_cohort_rows_sha256",
      "mask_column_order",
      "packed_axis",
      "packed_bitorder",
      "refit_seconds",
      "seconds_before_export",
      "reference_n",
      "reference_pool",
      "reference_subset_rule",
      "reference_subset_seed",
      "reference_age_range",
      "reference_pool_age_range",
      "reference_age_quartiles",
      "reference_site_counts",
      "draw_site_minimum",
      "original_fit",
      "schema_version",
      "run_name",
      "outputs"
    ],
    "steady_text_sha256": "5418df065b12d6716991c826cd0eb6aa08f80c955d21ecbb626db0ee258ff5e0"
  },
  "json/synthetic-refits/result.json": {
    "keys": [
      "mode",
      "runtime",
      "held_out_scans",
      "voxel_columns",
      "references",
      "bootstrap_refits",
      "bootstrap_seed",
      "bootstrap_cohort_rows_sha256",
      "optimizer_iterations",
      "chunk",
      "flag_rule",
      "design",
      "fold",
      "folds",
      "reference_cohort_rows_sha256",
      "mask_column_order",
      "packed_axis",
      "packed_bitorder",
      "refit_seconds",
      "seconds_before_export",
      "schema_version",
      "run_name",
      "outputs"
    ],
    "steady_text_sha256": "92c20073625aed16010b81e0b6d2e659f52e457e0012fe8e9a3268afaff3b6ec"
  },
  "tables/synthetic-main/bootstrap-ranking.tsv": "cb0d81b0fe904faa75d7af7c7336afadc3d8ab9be2c8aaeff1a8373ae2bab94d",
  "tables/synthetic-main/scanner-ranking.tsv": "e39a50fa3bb77ecc8466945516d759c0f64b0cf86bec2fd54cef89acaa9d5838"
}
