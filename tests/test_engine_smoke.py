from pathlib import Path

import pytest


def test_engine_modules_import_cleanly():
    from autoanatomy.engine import api, class_map, config, nnunet_runner, registry, weights  # noqa: F401


def test_unsupported_task_raises():
    from autoanatomy.engine.api import segment

    with pytest.raises(ValueError, match="dental_segmentator"):
        segment(input=Path("does_not_matter.nii.gz"), output=None, task="total")


def _weights_cached(*dataset_dirs) -> bool:
    from autoanatomy.engine.config import get_weights_dir

    weights_dir = get_weights_dir()
    return all((weights_dir / d).exists() for d in dataset_dirs)


@pytest.mark.skipif(not _weights_cached("Dataset112_DentalSegmentator_v100"),
                     reason="model weights not downloaded (run `autoanatomy download-weights`)")
def test_real_segmentation_produces_all_five_masks(tmp_path):
    from autoanatomy.engine.api import segment
    from autoanatomy.engine.class_map import class_map

    atlas = Path(__file__).parent / "reference_files" / "ct_brain_atlas_1mm.nii.gz"
    if not atlas.exists():
        pytest.skip("reference CT volume not available")

    import torch
    device = "gpu" if torch.cuda.is_available() else "cpu"
    segment(input=atlas, output=tmp_path, task="dental_segmentator", device=device, quiet=True)

    for name in class_map["dental_segmentator"].values():
        assert (tmp_path / f"{name}.nii.gz").exists()


@pytest.mark.skipif(not _weights_cached("Dataset777_head_muscles_492subj"),
                     reason="model weights not downloaded (run `autoanatomy download-weights`)")
def test_real_head_muscles_segmentation_produces_all_eleven_masks(tmp_path):
    from autoanatomy.engine.api import segment
    from autoanatomy.engine.class_map import class_map

    atlas = Path(__file__).parent / "reference_files" / "ct_brain_atlas_1mm.nii.gz"
    if not atlas.exists():
        pytest.skip("reference CT volume not available")

    import torch
    device = "gpu" if torch.cuda.is_available() else "cpu"
    segment(input=atlas, output=tmp_path, task="head_muscles", device=device, quiet=True)

    for name in class_map["head_muscles"].values():
        assert (tmp_path / f"{name}.nii.gz").exists()
