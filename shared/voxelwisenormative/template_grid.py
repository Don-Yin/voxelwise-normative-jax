"""Model mask and voxel-vector extraction for maps already on the template grid."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .artifacts import VOXEL_MATRIX, resolve


def _image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI-1 image with a voxel-to-world transform."""
    img = nib.load(path)
    if not isinstance(img, nib.Nifti1Image) or img.affine is None:
        raise ValueError(f"expected NIfTI-1 with an affine in {path}")
    return img, img.affine


class TemplateGrid:
    """Extract C-order model vectors from files on the pinned IXI mask grid."""

    def __init__(self) -> None:
        path = resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"]
        image, self.affine = _image(path)
        self.mask = np.asarray(image.dataobj) > 0

    def vector(self, path: Path) -> np.ndarray:
        """Read masked float32 voxel values and widen them to float64 for scoring."""
        image, affine = _image(path)
        if image.shape != self.mask.shape or not np.allclose(affine, self.affine, atol=1e-4):
            raise ValueError(f"{path} is not on the model grid: shape {image.shape}")
        return np.asarray(image.dataobj, dtype=np.float32)[self.mask].astype(np.float64)

    def valid(self, path: Path) -> np.ndarray:
        """Return scanned field of view at the model voxels."""
        return self.vector(path) > 0
