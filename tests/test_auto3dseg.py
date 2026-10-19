import hashlib
import json
import zipfile

import nibabel as nib
import numpy as np
import pytest
import torch

from autoanatomy.engine import auto3dseg, auto3dseg_weights


def _config(**overrides):
    config = {
        "normalize_mode": "range", "intensity_bounds": [-60, 1700],
        "resample_resolution": [0.45703125, 0.459, 0.458015625],
        "roi_size": [16, 16, 16],
    }
    config.update(overrides)
    return config


class _PointwiseModel(torch.nn.Module):
    """A deterministic ten-class classifier for testing geometry/export only."""

    def forward(self, image):
        labels = ((image[:, 0] - 0.2) * 10).floor().long().clamp(0, 9)
        return torch.nn.functional.one_hot(labels, 10).movedim(-1, 1).float()


def test_remap_excludes_lateral_pterygoid_and_other_whole_head_labels():
    source = np.arange(10, dtype=np.uint8).reshape(1, 2, 5)
    actual = auto3dseg.remap_labels(source)
    np.testing.assert_array_equal(actual.ravel(), [0, 1, 2, 0, 3, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(
        auto3dseg.remap_labels(source, ["medial_pterygoid"]).ravel(),
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    )


def test_normalization_matches_published_sigmoid_and_does_not_clip_ct():
    from monai.data import MetaTensor
    data = np.array([-1000, -60, 0, 820, 1700, 3000, 40, 100], dtype=np.float32).reshape(1, 2, 2, 2)
    image = MetaTensor(data, affine=np.diag([0.45703125, 0.459, 0.458015625, 1]))
    result = auto3dseg._transforms(_config(crop_foreground=False))({"image": image, "source": image})
    expected = torch.sigmoid(torch.tensor((data + 60) / 1760 * 2 - 1))
    torch.testing.assert_close(result["image"].as_tensor(), expected)


@pytest.mark.parametrize("orientation_ras", [False, True])
def test_prediction_restores_original_shape_affine_and_crop(orientation_ras):
    from autoanatomy.engine.progress import report_progress
    data = np.full((43, 45, 39), -1000, dtype=np.float32)
    data[18:25, 19:26, 16:23] = 400
    affine = np.array([[-0.8, 0, 0, 28], [0, 0.7, 0, -11], [0, 0, 1.1, 6], [0, 0, 0, 1]])
    image = nib.Nifti1Image(data, affine)
    progress = []
    with report_progress(lambda *args: progress.append(args)):
        result = auto3dseg.predict_image(
            image, _PointwiseModel(), _config(orientation_ras=orientation_ras), "cpu", quiet=True,
        )
    assert result.shape == image.shape
    np.testing.assert_array_equal(result.affine, image.affine)
    assert result.get_data_dtype() == np.dtype("uint8")
    assert set(np.unique(result.dataobj)) <= {0, 1, 2, 3}
    assert np.count_nonzero(result.dataobj) > 0
    assert not np.asanyarray(result.dataobj)[:4].any()
    assert progress[0][0] == 0
    assert progress[-1][0] == progress[-1][1] > 0
    assert all(p[3] is None for p in progress)


def test_archive_with_bad_checksum_never_installs(tmp_path):
    archive = tmp_path / "download.zip"
    archive.write_bytes(b"incomplete or modified download")
    destination = tmp_path / "model"
    with pytest.raises(ValueError, match="checksum"):
        auto3dseg_weights.install_archive(archive, destination)
    assert not destination.exists()


def test_archive_rejects_unexpected_paths_even_with_matching_digest(tmp_path, monkeypatch):
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../model.pt", b"not a model")
        bundle.writestr("labels.csv", "LabelValue,Name")
    monkeypatch.setattr(auto3dseg_weights, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Unexpected files"):
        auto3dseg_weights.install_archive(archive, tmp_path / "cache")
    assert not (tmp_path / "model.pt").exists()


@pytest.mark.parametrize("ml", [False, True])
def test_api_exports_selected_masks_and_original_geometry(tmp_path, monkeypatch, ml):
    from autoanatomy.engine.api import segment
    from autoanatomy.engine.nifti_ext_header import load_multilabel_nifti

    monkeypatch.setattr(auto3dseg, "download_weights", lambda **kwargs: tmp_path / "model.pt")
    monkeypatch.setattr(auto3dseg, "_load_model", lambda *args: (_PointwiseModel(), _config()))
    image = nib.Nifti1Image(np.full((17, 19, 21), 100, dtype=np.float32), np.diag([0.46, 0.46, 0.46, 1]))
    output = tmp_path / ("muscles.nii.gz" if ml else "masks")
    report = tmp_path / "run.json"
    result, stats = segment(
        image, output, task="auto3dseg_muscles", device="cpu", ml=ml,
        roi_subset=["masseter", "medial_pterygoid"], statistics=True, quiet=True,
        statistics_exclude_masks_at_border=False, report=report,
    )
    assert result.shape == image.shape
    assert set(stats) == {"masseter", "medial_pterygoid"}
    if ml:
        saved, labels = load_multilabel_nifti(output)
        assert labels == {1: "masseter", 3: "medial_pterygoid"}
        assert saved.shape == image.shape
    else:
        assert sorted(p.name for p in output.glob("*.nii.gz")) == ["masseter.nii.gz", "medial_pterygoid.nii.gz"]
        saved = nib.load(output / "masseter.nii.gz")
        assert np.count_nonzero(saved.dataobj) > 0
    np.testing.assert_allclose(saved.affine, image.affine, rtol=1e-6, atol=1e-6)
    assert json.loads(report.read_text())["model"]["laterality"] == "combined"


def test_unsupported_fast_mode_fails_before_download(monkeypatch):
    from autoanatomy.engine.api import segment
    monkeypatch.setattr(auto3dseg, "download_weights", lambda **kwargs: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="fixed 0.5 mm"):
        segment("does-not-exist.nii.gz", task="auto3dseg_muscles", fast=True, device="cpu")


def test_invalid_scan_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(auto3dseg, "download_weights", lambda **kwargs: pytest.fail("unexpected download"))
    archive = tmp_path / "scan.zip"
    archive.write_bytes(b"unsupported input")
    with pytest.raises(ValueError, match="NIfTI file or a DICOM directory"):
        auto3dseg.segment_auto3dseg(archive)


def test_mirroring_preserves_predictions_and_counts_completed_tiles():
    from autoanatomy.engine.progress import report_progress
    image = torch.linspace(0, 1, 16 ** 3).reshape(1, 1, 16, 16, 16)
    model = _PointwiseModel()
    direct = auto3dseg._predict(model, image, _config(), "cpu", quiet=True)
    progress = []
    with report_progress(lambda *args: progress.append(args)):
        mirrored = auto3dseg._predict(model, image, _config(), "cpu", mirroring=True)
    torch.testing.assert_close(direct, mirrored)
    assert [p[:2] for p in progress] == [(0, 1), (1, 1)]


@pytest.mark.skipif(not auto3dseg_weights.weights_cached(), reason="Auto3DSeg checkpoint not cached")
def test_real_checkpoint_loads_strictly_and_runs_ten_class_forward():
    # A smaller spatial patch tests the real fully convolutional network and
    # published weights without making the suite run a full clinical CT.
    model, config = auto3dseg._load_model(auto3dseg_weights.download_weights(quiet=True), "cpu")
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 1, 32, 32, 32))
    assert tuple(prediction.shape) == (1, 10, 32, 32, 32)
    assert torch.isfinite(prediction).all()
    assert config["roi_size"] == [224, 224, 144]
