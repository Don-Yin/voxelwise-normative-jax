"""Run the published grey-matter chain with and without the installed SynthSR v1 model."""

from __future__ import annotations

import csv
import importlib
import logging
import os
import time
from pathlib import Path

from ..utils.integrity import expect, head_rows, sha256, write_json
from ..utils.paths import FS_HOME, INTERMEDIATES, PUBLISHED, REFERENCE_REPO, RESULTS
from .staging import head_commands, load_staged

logger = logging.getLogger(__name__)
BASE = ("IXI017-Guys-0698-T1", "IXI165-HH-1589-T1", "IXI331-IOP-0892-T1")
CONTROLS = (
    "IXI017-Guys-0698-T1", "IXI065-Guys-0744-T1", "IXI122-Guys-0773-T1",
    "IXI129-Guys-0775-T1", "IXI165-HH-1589-T1", "IXI214-HH-1636-T1",
    "IXI275-HH-1803-T1", "IXI292-IOP-0877-T1", "IXI315-IOP-0888-T1",
    "IXI331-IOP-0892-T1", "IXI563-IOP-1153-T1", "IXI613-HH-2734-T1",
)
PUBLISHED_SHA = os.getenv('IXI_SUBJECTS_SHA256', 'ce285979cffac4f245614280a20f0fbd5233fd83965a95ffb2dd083216df1684')
TEMPLATE_SHA = os.getenv('IXI_TEMPLATE_SHA256', '4b7d84dd0e8a244fb96dc8b957b618c971df75ea0d6a55d877db18a65afbe279')
SYNTHSR_SHA = "2fd59e96196388360eba95254fb6dfc9eb9eb8638018b590575e47e0a387f255"
CHAIN_SHA = {
    "commands.py": "79722bbf105da8d6bd6fae9ca7899d48db1cd6363b1fb219b8f5f88cfb39a926",
    "folds.py": "11e95dfddac0511a8535a92d6e1d7c1cfa796f1fae03184a9691b5fe193b872d",
    "images.py": "c70b1dd3e13c9eecaef05536dda07d98ae0a9dae6e22ed58075e6a22f2c27baa",
    "registration.py": "cf370ae737298f59b97692a5d9d96cd501b8622bd5cfececc0b468f91931d2e1",
    "segmentation.py": "49f1c6bf6bcd1031037ce0e06e10958913ddc48435b556c2a1edbb35e2272a90",
    "synthstrip.py": "fbc3e47ed53b7b78acfc24687519ed7f8d56a8eccf24a6b3477cc526b64f6482",
    "synthstrip_model.py": "6570f5c4bab2e1e55696a51b734a37b683fb98fab9d02a5b2cff57f1545bf554",
}


def published_hashes() -> dict[str, str]:
    """Read reference output digests from the frozen publication."""
    path = PUBLISHED / "subjects.tsv"
    expect(path, PUBLISHED_SHA)
    with path.open(newline="") as f:
        return {r["subject"]: r["gm_modulated_s8_sha256"] for r in csv.DictReader(f, delimiter="\t")
                if r["status"] == "complete"}


def chain_tool(module: str, name: str):
    """Resolve the shared function only from the expected external reference repository."""
    imported = importlib.import_module(f"experiments.shared.vbm_chain.{module}")
    expected = REFERENCE_REPO / f"experiments/shared/vbm_chain/{module}.py"
    origin = imported.__file__
    if origin is None or Path(origin).resolve() != expected.resolve():
        raise ImportError(f"{name}: imported from {origin}, expected {expected}")
    return getattr(imported, name)


def check_tools(route: str) -> Path:
    """Check versions and weights before starting any heavy transform."""
    folder = REFERENCE_REPO / "experiments/shared/vbm_chain"
    for file, digest in CHAIN_SHA.items():
        expect(folder / file, digest)
    expect(FS_HOME / "models/synthstrip.1.pt", "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33")
    if route == "sr":
        expect(FS_HOME / "models/synthsr_v10_210712.h5", SYNTHSR_SHA)
        expect(FS_HOME / "bin/mri_synthsr", "eebcf9dac4164457fd4acf71faf6843ad1062330bf66cf9906a10831e3ff2840")
        expect(FS_HOME / "python/scripts/mri_synthsr", "865bb5eff99f91e57264e7290333754664edbc63f2b57bba76d1f6392b5ee771")
    template = PUBLISHED / "template/template_GM.nii.gz"
    expect(template, TEMPLATE_SHA)
    return template


def synthesize(source: Path, folder: Path, threads: int, commands: list[dict]) -> Path:
    """Call the installed CPU vendor command and use the same crop as the Mac workflow."""
    run_command = chain_tool("commands", "run_command")
    folder.mkdir()
    synthetic = folder / "synthetic.nii.gz"
    run_command(["env", "-u", "LD_LIBRARY_PATH", f"FREESURFER_HOME={FS_HOME}", "CUDA_VISIBLE_DEVICES=-1",
                 str(FS_HOME / "bin/mri_synthsr"), "--i", str(source), "--o", str(synthetic),
                 "--v1", "--cpu", "--threads", str(threads), "--disable_sharpening"], commands)
    # The vendor script can return zero after catching a failed prediction.
    if not synthetic.is_file() or synthetic.stat().st_size == 0:
        raise RuntimeError(f"mri_synthsr produced no file {synthetic} for {source}")
    for argv in head_commands(synthetic, folder):
        run_command(argv, commands)
    return folder / "cropped.nii.gz"


def one_subject(subject: str, route: str, threads: int, cohort: str = "small-test") -> None:
    """Bind one public IXI row, run the selected route and save its measured receipt."""
    start = time.monotonic()
    rows = head_rows()
    staged = load_staged(rows)[subject]
    stage_dir = Path(staged["folder"])
    head = stage_dir / "cropped.nii.gz"
    expect(head, rows[subject]["cropped_sha256"])
    source = stage_dir / "reoriented.nii.gz" if route == "sr" else head
    if route == "sr":
        expect(source, staged["reoriented_sha256"])
    expected = published_hashes()[subject] if route == "base" else None
    template = check_tools(route)
    folder = INTERMEDIATES / cohort / route / subject
    folder.mkdir(parents=True)
    (folder / "source.txt").write_text(f"{source.resolve()}\n")
    logger.info("%s %s starts with %s", subject, route, source.resolve())
    commands: list[dict] = []
    if route == "sr":
        head = synthesize(source, folder / "synthsr", threads, commands)
    prepared = chain_tool("segmentation", "prepare_native_gm")
    run_command = chain_tool("commands", "run_command")
    gm = folder / "GM.nii.gz"
    prepared(head, folder, commands, extractor="synthstrip")
    (folder / "brain_pve_1.nii.gz").replace(gm)
    if route == "base":
        fsl = Path(os.environ["FSLDIR"])
        run_command(["fsl_reg", str(gm), str(fsl / "data/standard/tissuepriors/avg152T1_gray"),
                     str(folder / "GM_to_T"), "-a"], commands)
    registration = chain_tool("registration", "kamen")(gm, template, folder / "kamen")
    commands += registration["commands"]
    actual_map = folder / "kamen/gm-modulated-s8.nii.gz"
    map_sha = sha256(actual_map)
    nonpositive = registration["fold_check"]["nonpositive_voxels"]
    status = "complete" if (expected is None or expected == map_sha) and nonpositive == 0 else "failed"
    out = RESULTS / "maps" / cohort / route / subject
    out.mkdir(parents=True)
    (out / "source.txt").write_text(f"{source.resolve()}\n")
    map_path = out / actual_map.name
    os.link(actual_map, map_path)
    summary = {
        "subject": subject, "route": route, "status": status, "source": str(source), "map": str(map_path),
        "map_sha256": map_sha, "expected_map_sha256": expected,
        "fold_check": registration["fold_check"], "seconds": round(time.monotonic() - start, 3),
        "cpu_seconds_commands": round(sum(c["cpu_seconds"] for c in commands), 3),
        "peak_process_rss_mib": round(max(c["peak_rss_mib"] for c in commands), 3), "commands": commands,
        "threads_synthsr": threads if route == "sr" else None, "job": os.environ.get("SLURM_JOB_ID"),
        "array_job": os.environ.get("SLURM_ARRAY_JOB_ID"), "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json(out / "receipt.json", summary)
    if status != "complete":
        raise ValueError(f"{subject} {route}: SHA-256 {map_sha} expected {expected}; nonpositive J: {nonpositive}")
    logger.info("%s %s verified in %.3f s, peak command RSS %.3f MiB", subject, route,
                summary["seconds"], summary["peak_process_rss_mib"])


def small_test(task: int, threads: int) -> None:
    """Process one of the 3 published verification people or 12 SynthSR controls."""
    planned = [(s, "base") for s in BASE] + [(s, "sr") for s in CONTROLS]
    if not 0 <= task < len(planned) or threads < 1:
        raise ValueError(f"invalid small-test task {task}, threads={threads}")
    subject, route = planned[task]
    one_subject(subject, route, threads)


def full_reference(task: int, threads: int) -> None:
    """Process exactly one frozen IXI row, with no reuse of small-test outputs."""
    subjects = tuple(head_rows())
    if not 0 <= task < len(subjects) or threads != 4:
        raise ValueError(f"invalid full-reference task {task}, threads={threads}; expected 0..562 and 4 threads")
    subject = subjects[task]
    try:
        one_subject(subject, "sr", threads, "full")
    except Exception as error:
        write_json(RESULTS / "manifests" / "full-failures" / f"{subject}.json", {
            "subject": subject, "task": task, "array_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
