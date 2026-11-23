"""Prepare and checksum public IXI T1 crops using FSL 6.0.7.18."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

MEMBER = re.compile(r"IXI(\d{3})-(Guys|HH|IOP)-\d{4}-T1\.nii\.gz")
COHORT_SIZE = 563


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def match_members(archive: tarfile.TarFile, covariates: dict[int, tuple[str, float]]) -> list:
    """Select verified regular files with matched metadata; ignore unrelated archive members."""
    found = {}
    for entry in archive:
        match = MEMBER.fullmatch(Path(entry.name).name)
        if match is None:
            continue
        ixi_id = int(match[1])
        if not entry.isfile() or entry.size <= 0 or ixi_id in found:
            raise ValueError(f"unexpected or duplicate T1 member: {entry.name}")
        found[ixi_id] = (entry, match)
    return [found[i] for i in sorted(found.keys() & covariates.keys())]


def demographics(book: Path) -> dict[int, tuple[str, float]]:
    """Read the age and sex columns of the IXI workbook."""
    import pandas as pd

    cols = ['IXI_ID', 'SEX_ID (1=m, 2=f)', 'AGE']
    table = pd.read_excel(book).filter(items=cols)
    if list(table.columns) != cols:
        raise ValueError(f"expected {cols} in the IXI demographic workbook")
    data = {}
    for subject, sex, age in table.dropna().itertuples(index=False, name=None):
        if int(subject) in data or sex not in (1, 2) or not 0 < age <= 120:
            raise ValueError(f"invalid or repeated demographic row for IXI ID {subject}")
        data[int(subject)] = ('M' if sex == 1 else 'F', float(age))
    return data


def prepare(archive_path: Path, book: Path, out: Path) -> None:
    fsl_dir = os.environ.get('FSLDIR')
    if not fsl_dir or (Path(fsl_dir) / 'etc/fslversion').read_text().strip() != '6.0.7.18':
        raise RuntimeError('supply FSLDIR for FSL 6.0.7.18')
    if any(shutil.which(name) is None for name in ('fslreorient2std', 'robustfov')):
        raise RuntimeError('put fslreorient2std and robustfov on PATH')
    covariates = demographics(book)
    with tarfile.open(archive_path, mode='r:') as archive:
        matches = match_members(archive, covariates)
        if len(matches) != COHORT_SIZE:
            raise ValueError(f'expected {COHORT_SIZE} T1 images with demographics, found {len(matches)}')
        out.mkdir(parents=True, mode=0o700)
        rows = []
        env = {**os.environ, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        for i, (entry, match) in enumerate(matches, start=1):
            ixi_id = int(match[1])
            subject = match[0].removesuffix('.nii.gz')
            folder = out / subject
            folder.mkdir(mode=0o700)
            raw, oriented, cropped = (folder / f'{name}.nii.gz' for name in ('raw', 'reoriented', 'cropped'))
            member = archive.extractfile(entry)
            if member is None:
                raise ValueError(f'cannot read archive member {entry.name}')
            with member, raw.open('xb') as f:
                shutil.copyfileobj(member, f)
            subprocess.run(['fslreorient2std', str(raw), str(oriented)], env=env, check=True)
            subprocess.run(['robustfov', '-i', str(oriented), '-r', str(cropped), '-b', '170'], env=env, check=True)
            sex, age = covariates[ixi_id]
            rows.append((subject, ixi_id, match[2], sex, age, digest(cropped)))
            print(f'[{i}/{len(matches)}] prepared {subject}', flush=True)
    manifest = out / 'manifest.tsv'
    with manifest.open('x', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(('subject', 'ixi_id', 'site', 'sex', 'age', 'cropped_sha256'))
        writer.writerows(rows)
    print(f'{manifest}: sha256 {digest(manifest)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new local directory for T1 crops')
    args = parser.parse_args()
    prepare(args.archive, args.metadata, args.out)


if __name__ == '__main__':
    main()
