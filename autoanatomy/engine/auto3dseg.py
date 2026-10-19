"""MONAI Auto3DSeg whole-head inference for three bilateral muscle groups.

    Checkpoint and reference preprocessing:
    https://github.com/lassoan/SlicerMONAIAuto3DSeg

The original ten-way classifier is always evaluated before filtering outputs;
excluding a structure from export must never turn it into another muscle.
"""
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import tempfile
import time

import nibabel as nib
import numpy as np
import torch

from autoanatomy.engine.auto3dseg_weights import download_weights
from autoanatomy.engine.class_map import class_map

TASK = "auto3dseg_muscles"
SOURCE_LABELS = {1: 1, 2: 2, 4: 3}
OVERLAP = 0.625


def _load_model(checkpoint_path, device):
    from monai.networks.nets import SegResNetDS

    # This checkpoint is a tensor state_dict plus primitive configuration data.
    # Do not use ConfigParser to execute expressions stored in a downloaded file.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    network = dict(config["network"])
    if network.pop("_target_") != "SegResNetDS":
        raise ValueError("Unsupported Auto3DSeg checkpoint architecture.")
    if network.get("norm") == "INSTANCE_NVFUSER":
        # MONAI's portable fallback has the same parameters and normalization;
        # the optional NVIDIA Apex implementation is not available on all hosts.
        network["norm"] = "INSTANCE"
    model = SegResNetDS(**network)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    return model, config


def _transforms(config):
    from monai.transforms import (
        Compose, CropForegroundd, EnsureTyped, Lambdad, Orientationd,
        ScaleIntensityRanged, Spacingd,
    )

    if config.get("normalize_mode") != "range":
        raise ValueError("Unsupported intensity normalization in the Auto3DSeg checkpoint.")
    lower, upper = config["intensity_bounds"]
    transforms = [
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
        ScaleIntensityRanged(
            keys="image", a_min=lower, a_max=upper, b_min=-1.0, b_max=1.0, clip=False,
        ),
        Lambdad(keys="image", func=torch.sigmoid),
    ]
    if config.get("orientation_ras", False):
        transforms.append(Orientationd(keys=("image", "source"), axcodes="RAS"))
    if config.get("crop_foreground", True):
        # Reference code crops the original CT (>0), before its normalization.
        transforms.append(CropForegroundd(
            keys="image", source_key="source", margin=10, allow_smaller=True,
        ))
    spacing = np.asarray(config["resample_resolution"], dtype=float)
    transforms.append(Spacingd(
        keys="image", pixdim=spacing, mode="bilinear", dtype=np.float32,
        min_pixdim=spacing * 0.75, max_pixdim=spacing * 1.25,
    ))
    return Compose(transforms)


def _predict(model, image, config, device, mirroring=False, quiet=False):
    from monai.data.utils import dense_patch_slices
    from monai.inferers import SlidingWindowInferer
    from autoanatomy.engine.progress import has_progress_callback, notify_progress

    roi = tuple(config["roi_size"])
    shape = tuple(max(a, b) for a, b in zip(image.shape[2:], roi))
    interval = tuple(max(int(size * (1 - OVERLAP)), 1) for size in roi)
    total = len(dense_patch_slices(shape, roi, interval))
    completed = 0
    variants = [()] if not mirroring else [
        tuple(axis + 2 for axis, enabled in enumerate(bits) if enabled)
        for bits in itertools.product((False, True), repeat=3)
    ]
    notify_progress(0, total, "Auto3DSeg muscle segmentation")

    def predict_tile(tile):
        nonlocal completed
        prediction = None
        for axes in variants:
            sample = torch.flip(tile, axes) if axes else tile
            result = model(sample)
            if isinstance(result, (tuple, list)):
                result = result[0]
            if axes:
                result = torch.flip(result, axes)
            prediction = result if prediction is None else prediction + result
        completed += 1
        notify_progress(completed, total, "Auto3DSeg muscle segmentation")
        return prediction / len(variants)

    # Keep the full-resolution accumulator in system RAM on every device.
    # Only one real inference patch at a time needs GPU memory.
    inferer = SlidingWindowInferer(
        roi_size=roi, sw_batch_size=1, overlap=OVERLAP, mode="gaussian",
        sw_device=device, device="cpu", progress=not quiet and not has_progress_callback(),
    )
    amp = torch.autocast("cuda") if torch.device(device).type == "cuda" else nullcontext()
    with torch.inference_mode(), amp:
        logits = inferer(image, predict_tile)
    return logits.argmax(dim=1).to(torch.uint8)[0]


def remap_labels(source, roi_subset=None):
    """Map source labels 1/2/4 to the public 1/2/3 muscle label map."""
    result = np.zeros(source.shape, dtype=np.uint8)
    for source_id, output_id in SOURCE_LABELS.items():
        name = class_map[TASK][output_id]
        if roi_subset is None or name in roi_subset:
            result[source == source_id] = output_id
    return result


def predict_image(image, model, config, device, roi_subset=None, mirroring=False, quiet=False):
    """Run inference and restore the exact original voxel array and affine."""
    from monai.data import MetaTensor
    from monai.transforms import Invertd, KeepLargestConnectedComponent

    source = MetaTensor(
        np.asarray(image.dataobj, dtype=np.float32)[None], affine=image.affine,
    )
    transform = _transforms(config)
    data = transform({"image": source, "source": source})
    prediction = _predict(
        model, data["image"].as_tensor()[None], config, device, mirroring, quiet,
    )
    data["pred"] = MetaTensor(prediction[None], affine=data["image"].affine)
    restored = Invertd(
        keys="pred", orig_keys="image", transform=transform, nearest_interp=True,
        to_tensor=True,
    )(data)["pred"]
    restored = KeepLargestConnectedComponent(
        applied_labels=list(SOURCE_LABELS), independent=True, num_components=2,
    )(restored)
    result = remap_labels(restored[0].cpu().numpy(), roi_subset)
    if result.shape != image.shape:
        raise RuntimeError("Auto3DSeg did not restore the original scan dimensions.")
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(result, image.affine, header)


def segment_auto3dseg(
    input, output=None, ml=False, roi_subset=None, device="cpu", quiet=False,
    verbose=False, statistics=False, skip_saving=False, remove_small_blobs=False,
    mirroring=False, report=None, statistics_exclude_masks_at_border=True,
    stats_aggregation="mean", statistics_normalized_intensities=False,
    statistics_extra=False,
):
    started = time.time()
    if roi_subset is not None:
        if not isinstance(roi_subset, list) or not roi_subset:
            raise ValueError("roi_subset must be a nonempty list of Auto3DSeg muscle names.")
        unknown = set(roi_subset) - set(class_map[TASK].values())
        if unknown:
            raise ValueError(f"Unknown Auto3DSeg structures: {', '.join(sorted(unknown))}")
    output = Path(output) if output is not None else None
    with tempfile.TemporaryDirectory(prefix="auto3dseg-input-") as tmp:
        if isinstance(input, nib.Nifti1Image):
            image = input
        elif str(input).lower().endswith((".nii", ".nii.gz")):
            image = nib.load(input)
        else:
            if not Path(input).is_dir():
                raise ValueError("Auto3DSeg input must be a NIfTI file or a DICOM directory.")
            from autoanatomy.engine.dicom_io import dcm_to_nifti
            nifti_path = Path(tmp) / "ct.nii.gz"
            dcm_to_nifti(Path(input), nifti_path, tmp_dir=Path(tmp), verbose=verbose)
            image = nib.load(nifti_path)
        if len(image.shape) != 3 or any(size < 1 for size in image.shape):
            raise ValueError("Auto3DSeg requires a nonempty 3D CT volume.")
        checkpoint_path = download_weights(quiet=quiet)
        if not quiet:
            print("Auto3DSeg: masseter, temporalis and medial pterygoid (both sides together).")
        model, config = _load_model(checkpoint_path, device)
        if verbose:
            print(f"Auto3DSeg target spacing: {config['resample_resolution']} mm")
            print(f"Auto3DSeg inference patch: {config['roi_size']}; device: {device}")
        segmentation = predict_image(
            image, model, config, device, roi_subset, bool(mirroring), quiet,
        )
        data = np.asanyarray(segmentation.dataobj)
        if remove_small_blobs:
            from scipy import ndimage
            threshold = 200.0 if remove_small_blobs is True else float(remove_small_blobs)
            voxel_volume = abs(float(np.linalg.det(image.affine[:3, :3])))
            for label in class_map[TASK]:
                components, _ = ndimage.label(data == label)
                sizes = np.bincount(components.ravel()) * voxel_volume
                small = sizes < threshold
                small[0] = False
                data[small[components]] = 0
        selected = {k: v for k, v in class_map[TASK].items()
                    if roi_subset is None or v in roi_subset}
        if output is not None and not skip_saving:
            if ml:
                from autoanatomy.engine.nifti_ext_header import save_multilabel_nifti
                output.parent.mkdir(parents=True, exist_ok=True)
                save_multilabel_nifti(segmentation, output, selected)
            else:
                output.mkdir(parents=True, exist_ok=True)
                for label, name in selected.items():
                    mask = nib.Nifti1Image(
                        (data == label).astype(np.uint8), image.affine, segmentation.header,
                    )
                    nib.save(mask, output / f"{name}.nii.gz")
        stats = None
        if statistics:
            from autoanatomy.engine.statistics import get_basic_statistics
            stats_path = None
            if isinstance(statistics, (str, Path)):
                stats_path = Path(statistics)
            elif output is not None:
                stats_path = (output.parent if ml else output) / "statistics.json"
            if stats_path is not None:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats = get_basic_statistics(
                data, image, stats_path, quiet, TASK, statistics_exclude_masks_at_border,
                roi_subset, metric=stats_aggregation,
                normalized_intensities=statistics_normalized_intensities,
                extra_metrics=statistics_extra,
            )
    if report is not None:
        from autoanatomy.engine.api import build_run_report
        manifest = build_run_report(
            input, output, TASK, device, False, False, ml, "nifti", roi_subset,
            time.time() - started,
        )
        manifest["model"] = {
            "name": "Whole Head Segmentation", "version": "1.0.1", "engine": "MONAI Auto3DSeg",
            "target_spacing_mm": config["resample_resolution"], "laterality": "combined",
        }
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return (segmentation, stats) if statistics else segmentation
