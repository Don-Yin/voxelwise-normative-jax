"""A clean checkout can stage the cloud code without any imaging data."""
from pathlib import Path

import pytest

from shared.hugging_face_jobs.bucket import bucket_uri
from shared.hugging_face_jobs.build_input import build, verify


def test_code_only_bundle(tmp_path):
    mount = tmp_path / 'mount'
    rows = build(mount)
    assert rows == verify(mount)
    assert {row['role'] for row in rows} == {'code', 'hugging-face'}
    assert not any(row['path'].endswith(('.npz', '.npy', '.tsv', '.gz')) for row in rows)
    assert not (mount / 'experiments/shared').exists()
    file = mount / rows[0]['path']
    file.write_bytes(file.read_bytes() + b'!')
    with pytest.raises(ValueError, match='differs'):
        verify(mount)


def test_bucket_namespace_and_name():
    assert bucket_uri('example-team/example-bucket').endswith('/example-team/example-bucket')
    for value in ('', 'namespace', '../other', 'space/bad name', 'name/with:mount'):
        with pytest.raises(ValueError, match='namespace/name'):
            bucket_uri(value)
