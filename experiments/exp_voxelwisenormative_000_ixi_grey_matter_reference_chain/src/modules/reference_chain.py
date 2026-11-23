"""Build a symmetric IXI-only grey-matter template and register each head to it."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..utils.paths import EXPERIMENT_DIR, RESULTS

REFERENCE_REPO = Path(os.environ['IXI_REFERENCE_CHAIN']).resolve(strict=True)
INPUTS = Path(os.environ['IXI_PREPARED']).resolve(strict=True)
MANIFEST_SHA256 = os.environ['IXI_HEAD_MANIFEST_SHA256']
OUT = RESULTS / "ixi-reference-chain"
CALIBRATION = ["IXI002-Guys-0828-T1", "IXI012-HH-1211-T1", "IXI035-IOP-0873-T1", "IXI058-Guys-0726-T1"]
# Pin the separately supplied reference implementation and reject changes.
CHAIN_SHA256 = {
    "commands.py": "79722bbf105da8d6bd6fae9ca7899d48db1cd6363b1fb219b8f5f88cfb39a926",
    "folds.py": "11e95dfddac0511a8535a92d6e1d7c1cfa796f1fae03184a9691b5fe193b872d",
    "images.py": "c70b1dd3e13c9eecaef05536dda07d98ae0a9dae6e22ed58075e6a22f2c27baa",
    "registration.py": "cf370ae737298f59b97692a5d9d96cd501b8622bd5cfececc0b468f91931d2e1",
    "segmentation.py": "49f1c6bf6bcd1031037ce0e06e10958913ddc48435b556c2a1edbb35e2272a90",
    "synthstrip.py": "fbc3e47ed53b7b78acfc24687519ed7f8d56a8eccf24a6b3477cc526b64f6482",
    "synthstrip_model.py": "6570f5c4bab2e1e55696a51b734a37b683fb98fab9d02a5b2cff57f1545bf554",
}
LOGGER = logging.getLogger("reference_chain")


def sha256(path: Path) -> str:
    try:
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    except OSError as error:
        raise OSError(f'{path}: cannot hash reference input: {error}') from error


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'{path}: cannot read the stage receipt: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON record')
    return value


for name, expected in CHAIN_SHA256.items():
    if sha256(REFERENCE_REPO / "experiments/shared/vbm_chain" / name) != expected:
        raise RuntimeError(f"shared vbm_chain changed: {name}")
sys.path.insert(0, str(REFERENCE_REPO))
# These modules are supplied by the separately installed reference chain.
from experiments.shared.campaign_io import reference  # noqa: E402  # pyright: ignore[reportMissingImports]
from experiments.shared.vbm_chain.commands import run_command  # noqa: E402  # pyright: ignore[reportMissingImports]
from experiments.shared.vbm_chain.images import header_grid, same_grid  # noqa: E402  # pyright: ignore[reportMissingImports]
from experiments.shared.vbm_chain.registration import kamen  # noqa: E402  # pyright: ignore[reportMissingImports]
from experiments.shared.vbm_chain.segmentation import prepare_native_gm  # noqa: E402  # pyright: ignore[reportMissingImports]


def fsl(name: str) -> str:
    return str(Path(os.environ["FSLDIR"]) / name)


def subjects() -> list[dict]:
    manifest = INPUTS / "manifest.tsv"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("head manifest changed")
    lines = manifest.read_text().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]
    ids = {row['subject'] for row in rows}
    if len(rows) != 563 or len(ids) != 563 or any(not re.fullmatch(r'IXI\d{3}-(Guys|HH|IOP)-\d{4}-T1', s) for s in ids):
        raise RuntimeError(f'{manifest}: expected 563 distinct IXI subject IDs')
    return rows


def finish(path: Path, value: dict, start: float) -> dict:
    """Add timing and job identity, then publish the receipt atomically."""
    value |= {"seconds": round(time.monotonic() - start, 3), "job": os.environ.get("SLURM_JOB_ID")}
    write_json(path, value)
    return value


def write_json(path: Path, value: dict) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(partial, path)


def prepare(row: dict, base: Path) -> dict:
    """SynthStrip + FAST through the shared chain, then fslvbm's affine GM -> avg152T1_gray."""
    folder = base / row["subject"]
    head = INPUTS / row['subject'] / 'cropped.nii.gz'
    record = f"{head.resolve()}\n{row['cropped_sha256']}\n"
    if sha256(head) != row['cropped_sha256']:
        raise RuntimeError(f"head hash differs: {row['subject']}")
    if (folder / 'prepare.json').is_file():
        if (folder / 'source.txt').read_text() != record:
            raise RuntimeError(f'{folder}: differs from the prepared source')
        return read_json(folder / 'prepare.json')
    folder.mkdir(parents=True)
    (folder / 'source.txt').write_text(record)
    start, commands = time.monotonic(), []
    prepared = prepare_native_gm(head, folder, commands, extractor="synthstrip")
    gm = folder / "GM.nii.gz"
    os.replace(folder / "brain_pve_1.nii.gz", gm)
    run_command(["fsl_reg", str(gm), fsl("data/standard/tissuepriors/avg152T1_gray"),
                 str(folder / "GM_to_T"), "-a"], commands)
    return finish(folder / "prepare.json", {"subject": row["subject"], "head": prepared["head"],
                  "gm": reference(gm), "extractor": prepared["extractor"], "commands": commands}, start)


def symmetric_mean(images: list[Path], out: Path, commands: list) -> None:
    """Average both orientations of the registered grey-matter images."""
    stack = out.parent / f"{out.name}_4D"
    run_command(["fslmerge", "-t", str(stack), *map(str, images)], commands)
    run_command(["fslmaths", str(stack), "-Tmean", str(out)], commands)
    run_command(["fslswapdim", str(out), "-x", "y", "z", f"{out}_flipped"], commands)
    run_command(["fslmaths", str(out), "-add", f"{out}_flipped", "-div", "2", str(out)], commands)
    # avg152T1_gray is Analyze without orientation; its voxels are the MNI152 2 mm grid.
    run_command(["fslcpgeom", fsl("data/standard/MNI152_T1_2mm"), str(out)], commands)
    Path(f"{stack}.nii.gz").unlink()


def nonlinear_to_init(job: tuple[str, str]) -> dict:
    subject, init = job
    folder = OUT / subject
    out = folder / "GM_to_T_init"
    if Path(f"{out}.nii.gz").is_file() and (folder / "init.json").is_file():
        return read_json(folder / 'init.json')
    commands: list = []
    run_command(["fsl_reg", str(folder / "GM.nii.gz"), init, str(out), "-fnirt",
                 "--config=GM_2_MNI152GM_2mm.cnf"], commands)
    result = {"subject": subject, "commands": commands}
    write_json(folder / "init.json", result)
    return result


def template(workers: int) -> dict:
    start, commands = time.monotonic(), []
    listed = sorted(r["subject"] for r in subjects() if (OUT / r["subject"] / "prepare.json").is_file())
    if len(listed) != 563:
        raise RuntimeError(f"template needs every prepared subject; prepared={len(listed)}")
    folder = OUT / "template"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    init = folder / "template_GM_init"
    symmetric_mean([OUT / s / "GM_to_T.nii.gz" for s in listed], init, commands)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        second = list(pool.map(nonlinear_to_init, [(s, str(init)) for s in listed]))
    final = folder / "template_GM"
    symmetric_mean([OUT / s / "GM_to_T_init.nii.gz" for s in listed], final, commands)
    same_grid(header_grid(Path(f"{final}.nii.gz")), header_grid(Path(fsl("data/standard/MNI152_T1_2mm_brain_mask.nii.gz"))))
    (folder / "template_list.txt").write_text("\n".join(listed) + "\n")
    result = {"recipe": "fslvbm_2_template -n, using the full IXI cohort",
              "list_rule": "every prepared IXI subject (one healthy group; no balancing needed)",
              "n_list": len(listed), "template_list": reference(folder / "template_list.txt"),
              "template_init": reference(Path(f"{init}.nii.gz")),
              "template": reference(Path(f"{final}.nii.gz")), "commands": commands,
              "second_pass_seconds": [sum(c["seconds"] for c in s["commands"]) for s in second]}
    return finish(folder / "receipt.json", result, start)


def register(row: dict, base: Path, template_path: Path) -> dict:
    folder = base / row["subject"]
    if (folder / 'register.json').is_file():
        report = read_json(folder / 'register.json')
        if report['template'] != reference(template_path):
            raise RuntimeError(f'{folder}: registration is bound to another template')
        return report
    if (folder / 'kamen').exists():
        raise FileExistsError(f'{folder}: partial registration remains; inspect it before rerunning')
    start = time.monotonic()
    result = kamen(folder / "GM.nii.gz", template_path, folder / "kamen")
    folds = result['fold_check']['nonpositive_voxels']
    output = folder / 'kamen' / 'gm-modulated-s8.nii.gz'
    report = finish(folder / 'register.json', {**result, 'subject': row['subject'], 'template': reference(template_path),
                    'status': 'complete' if folds == 0 else 'fold-failed', 'map_sha256': sha256(output)}, start)
    if folds != 0:
        raise ValueError(f'{folder}: {folds} nonpositive Jacobian voxels')
    return report


def shard_rows(shard: int, shards: int) -> list[dict]:
    rows = subjects()
    if not 0 <= shard < shards:
        raise ValueError(f"invalid shard {shard}/{shards}")
    return rows[shard::shards]


def run_rows(step, rows: list[dict]) -> None:
    failed = []
    for i, row in enumerate(rows, 1):
        try:
            step(row)
            LOGGER.info("[%d/%d] %s done", i, len(rows), row["subject"])
        except Exception:
            LOGGER.exception("[%d/%d] %s failed", i, len(rows), row["subject"])
            failed.append(row["subject"])
    if failed:
        raise SystemExit(f"failed subjects: {failed}")


def publish(destination: Path) -> None:
    """Gather a completed cohort and verify the original maps before copying to a new local folder."""
    template_path = OUT / 'template' / 'template_GM.nii.gz'
    template_digest = sha256(template_path)
    template_receipt = read_json(OUT / 'template/receipt.json')
    if template_digest != template_receipt['template']['sha256']:
        raise ValueError('template differs from the recorded digest')
    maps = []
    for row in subjects():
        folder = OUT / row['subject']
        receipt = read_json(folder / 'register.json')
        source = folder / 'kamen/gm-modulated-s8.nii.gz'
        if receipt['status'] != 'complete' or receipt['template']['sha256'] != template_digest or sha256(source) != receipt['map_sha256']:
            raise ValueError(f'{folder}: incomplete registration or inconsistent digest')
        maps.append((row, source, receipt['map_sha256']))
    destination.mkdir(parents=True, mode=0o700)
    (destination / 'template').mkdir()
    shutil.copyfile(template_path, destination / 'template' / template_path.name)
    fields = ('subject', 'ixi_id', 'site', 'sex', 'age', 'status', 'gm_modulated_s8_sha256')
    with (destination / 'subjects.tsv').open('x', newline='') as table:
        writer = csv.writer(table, delimiter='\t', lineterminator='\n')
        writer.writerow(fields)
        for row, source, digest in maps:
            folder = destination / row['subject']
            folder.mkdir()
            shutil.copyfile(source, folder / source.name)
            writer.writerow((*[row[k] for k in fields[:5]], 'complete', digest))
    LOGGER.info('subjects sha256=%s template sha256=%s', sha256(destination / 'subjects.tsv'), template_digest)


def run(args) -> None:
    """Run one registration stage or gather its verified output."""
    if args.mode == 'publish':
        publish(args.derivatives)
        return
    os.environ.setdefault('VOXELWISE_PROFILE_ROOT', str(RESULTS / 'profile'))
    if not os.environ.get("SLURM_PROFILE_BIN"):
        raise RuntimeError("SLURM_PROFILE_BIN is required by the shared SynthStrip step")
    if args.mode == "calibrate":
        # Timing only: the real template does not exist yet, so kamen targets the MNI152 GM prior
        # on the MNI152 2 mm grid (same fnirt config and grid as the real run).
        base = OUT.parent / "ixi-reference-chain-calibration"
        row = {r["subject"]: r for r in subjects()}[CALIBRATION[args.shard]]
        target = base / "target" / f"prior-{args.shard}.nii.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            run_command(["fslmaths", fsl("data/standard/tissuepriors/avg152T1_gray"), str(target),
                         "-odt", "float"], [])
            run_command(["fslcpgeom", fsl("data/standard/MNI152_T1_2mm"), str(target)], [])
        run_rows(lambda r: (prepare(r, base), register(r, base, target)), [row])
    elif args.mode == "prepare":
        OUT.mkdir(parents=True, exist_ok=True)
        run_rows(lambda r: prepare(r, OUT), shard_rows(args.shard, args.shards))
    elif args.mode == "template":
        template(args.workers)
    else:
        receipt = read_json(OUT / 'template/receipt.json')
        template_path = Path(receipt["template"]["path"])
        if sha256(template_path) != receipt["template"]["sha256"]:
            raise RuntimeError("template changed after its receipt")
        run_rows(lambda r: register(r, OUT, template_path), shard_rows(args.shard, args.shards))
