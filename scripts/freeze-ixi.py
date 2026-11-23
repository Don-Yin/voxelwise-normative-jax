"""Freeze fresh IXI matrix or PCNtoolkit comparison artifacts locally; keep these outside Git."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import ARTIFACTS, PCNTOOLKIT_REFERENCE, VOXEL_MATRIX, resolve, sha256


MATRIX_FILES = ('gm.npy', 'mask.nii.gz', 'covariates.tsv')
REF_KEYS = {'Phi_train', 'Phi_test', 'y_train', 'y_test', 'test', 'z_test', 'nlZ', 'hyp', 'voxels'}


def read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'{path}: cannot read producer record: {error}') from error
    if not isinstance(record, dict):
        raise ValueError(f'{path}: expected a JSON record')
    return record


def describe(stage: str, source: Path) -> tuple[str, tuple[str, ...], dict]:
    if stage == 'matrix':
        receipt = read_record(source / 'json/receipt.json')
        if receipt.get('source_dataset') != 'https://brain-development.org/ixi-dataset/':
            raise ValueError('wrong dataset in the producer matrix receipt')
        for file in MATRIX_FILES:
            if sha256(source / file) != receipt['outputs'][file]:
                raise ValueError(f'{source / file}: checksum differs from the producer receipt')
        extra = {'source_dataset': receipt['source_dataset']}
        return VOXEL_MATRIX, MATRIX_FILES, extra
    timing = read_record(source / 'timing.json')
    if timing.get('pcntoolkit') != '1.4.0' or timing.get('l_bfgs_b_epsilon') != 1e-7:
        raise ValueError('expected the converged PCNtoolkit 1.4.0 reference (epsilon 1e-7)')
    expected = resolve(VOXEL_MATRIX, 'gm.npy')['gm.npy']
    if timing.get('matrix_sha256') != sha256(expected):
        raise ValueError('reference fit matrix does not match the frozen IXI matrix')
    with np.load(source / 'arrays.npz', allow_pickle=False) as ref:
        if not REF_KEYS.issubset(ref.files) or ref['Phi_train'].shape != (450, 12):
            raise ValueError('unexpected reference fields or training design')
        width = ref['voxels'].size
        if width not in (20, 1000) or ref['z_test'].shape != (113, width) or not np.isfinite(ref['z_test']).all():
            raise ValueError('expected a complete 20- or 1,000-column reference')
    artifact = PCNTOOLKIT_REFERENCE if width == 1000 else 'pcntoolkit-blr-reference-n20-v1'
    return artifact, ('arrays.npz',), {'input_artifact': VOXEL_MATRIX}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('matrix', 'reference'))
    parser.add_argument('source', type=Path, help='fresh output directory of the producer')
    args = parser.parse_args()
    artifact, files, extra = describe(args.stage, args.source)
    producer = 'exp_voxelwisenormative_001_ixi_voxel_matrix' if args.stage == 'matrix' else 'exp_voxelwisenormative_002_pcntoolkit_cpu_baseline'
    dest = ARTIFACTS / artifact
    dest.mkdir(parents=True, mode=0o700)
    listed = []
    for file in files:
        shutil.copyfile(args.source / file, dest / file)
        listed.append({'path': file, 'storage': 'local-ignored', 'sha256': sha256(dest / file)})
    (dest / 'README.md').write_text(f'# {artifact}\n\nLocal output of {producer}. Regenerate with `scripts/freeze-ixi.py`.\n')
    manifest = dict(schema_version=1, artifact_id=artifact, producer_experiment=producer, files=listed, **extra)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'froze {artifact}')


if __name__ == '__main__':
    main()
