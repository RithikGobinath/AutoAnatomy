from pathlib import Path

import pytest


def test_engine_modules_import_cleanly():
    from autoanatomy.engine import api, class_map, config, nnunet_runner, registry, weights  # noqa: F401


def test_unsupported_task_raises():
    from autoanatomy.engine.api import segment

    with pytest.raises(ValueError, match="craniofacial_structures"):
        segment(input=Path("does_not_matter.nii.gz"), output=None, task="total")


def _weights_cached() -> bool:
    from autoanatomy.engine.config import get_weights_dir

    weights_dir = get_weights_dir()
    return (weights_dir / "Dataset115_mandible").exists() and \
        (weights_dir / "Dataset298_TotalSegmentator_total_6mm_1559subj").exists()


@pytest.mark.skipif(not _weights_cached(), reason="model weights not downloaded (run `autoanatomy download-weights`)")
def test_real_segmentation_produces_all_seven_masks(tmp_path):
    from autoanatomy.engine.api import segment
    from autoanatomy.engine.class_map import class_map

    atlas = Path(__file__).parent / "reference_files" / "ct_brain_atlas_1mm.nii.gz"
    if not atlas.exists():
        pytest.skip("reference CT volume not available")

    import torch
    device = "gpu" if torch.cuda.is_available() else "cpu"
    segment(input=atlas, output=tmp_path, task="craniofacial_structures", device=device, quiet=True)

    for name in class_map["craniofacial_structures"].values():
        assert (tmp_path / f"{name}.nii.gz").exists()
