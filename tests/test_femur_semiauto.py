import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from autoanatomy.engine.femur_semiauto import run_femur_graphcut
from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_BACKGROUND, ANNOTATION_FOREGROUND


def _synthetic_ct_with_bone_blob(shape=(48, 48, 48), seed=0):
    rng = np.random.default_rng(seed)
    vol = np.full(shape, -1000.0, dtype=np.float32) + rng.normal(0, 20, shape).astype(np.float32)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    blob = dist < 12
    vol[blob] = 800 + rng.normal(0, 50, shape)[blob]
    return vol.astype(np.float32)


def test_run_femur_graphcut_produces_a_binary_mask_end_to_end():
    """Real graph-cut solve through the vendored FemurSegmentation pipeline
    and the compiled GraphCutSupport extension -- not mocked. Confirms the
    ITK/vendoring/dtype plumbing actually works, not just that it's wired up."""
    shape = (48, 48, 48)
    vol = _synthetic_ct_with_bone_blob(shape)
    img = nib.Nifti1Image(vol, np.eye(4))

    with tempfile.TemporaryDirectory() as tmp:
        scan_path = Path(tmp) / "scan.nii.gz"
        nib.save(img, str(scan_path))

        annotation = np.zeros(shape, dtype=np.uint8)
        annotation[20:24, 20:24, 20:24] = ANNOTATION_FOREGROUND
        annotation[2:6, 2:6, 2:6] = ANNOTATION_BACKGROUND
        annotation[40:44, 40:44, 40:44] = ANNOTATION_BACKGROUND

        out_path = Path(tmp) / "result.nii.gz"
        result = run_femur_graphcut(scan_path, annotation, img.affine, img.header, out_path)

        assert result.shape == shape
        assert set(np.unique(result).tolist()) <= {0, 1}
        assert (result > 0).any(), "graph-cut produced an all-background result"
        assert out_path.exists()


def test_run_femur_graphcut_requires_both_foreground_and_background_seeds():
    shape = (16, 16, 16)
    vol = _synthetic_ct_with_bone_blob(shape)
    img = nib.Nifti1Image(vol, np.eye(4))

    with tempfile.TemporaryDirectory() as tmp:
        scan_path = Path(tmp) / "scan.nii.gz"
        nib.save(img, str(scan_path))

        annotation = np.zeros(shape, dtype=np.uint8)
        annotation[5:8, 5:8, 5:8] = ANNOTATION_FOREGROUND  # no background seed at all

        with pytest.raises(ValueError, match="foreground and .* background"):
            run_femur_graphcut(scan_path, annotation, img.affine, img.header, Path(tmp) / "out.nii.gz")


def test_run_femur_graphcut_rejects_a_shape_mismatch():
    shape = (16, 16, 16)
    vol = _synthetic_ct_with_bone_blob(shape)
    img = nib.Nifti1Image(vol, np.eye(4))

    with tempfile.TemporaryDirectory() as tmp:
        scan_path = Path(tmp) / "scan.nii.gz"
        nib.save(img, str(scan_path))

        wrong_shape_annotation = np.zeros((8, 8, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            run_femur_graphcut(scan_path, wrong_shape_annotation, img.affine, img.header, Path(tmp) / "out.nii.gz")
