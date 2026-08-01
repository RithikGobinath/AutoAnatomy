"""Semi-automatic femur segmentation via graph-cut, using manually-painted
foreground/background seeds (see autoanatomy.tui.widgets.slice_viewer's
annotation mode) instead of a trained deep-learning model.

Wraps RiccardoBiondi/FemurSegmentation's own graph-cut pipeline (vendored
under autoanatomy/third_party/femur_segmentation -- see the README and
LICENSE there). Unlike the nnU-Net tasks in this codebase, this task has no
downloadable checkpoint: the "model" is a per-scan interactive graph-cut
solve seeded by the slices the user actually painted, so it needs no
training data and works on anatomy the deep-learning tasks never saw.

Note: the underlying maxflow-v3.01 solver this links against is licensed
"for research purposes only" (see
third_party/femur_segmentation/native_src/maxflow-v3.01/README.TXT) --
distinct from, and more restrictive than, the commercial_models license
gate the rest of this codebase uses for nnU-Net tasks. Do not offer this
task in any commercial deployment without resolving that separately.
"""
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_BACKGROUND, ANNOTATION_FOREGROUND

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "femur_segmentation"
_PATCH_MARKER = "_autoanatomy_femur_vendor_patch"


def _ensure_vendor_on_path() -> None:
    if getattr(sys.modules[__name__], _PATCH_MARKER, False):
        return
    for p in (str(_VENDOR_ROOT), str(_VENDOR_ROOT / "lib")):
        if p not in sys.path:
            sys.path.insert(0, p)
    setattr(sys.modules[__name__], _PATCH_MARKER, True)


def run_femur_graphcut(
    scan_path: Path,
    annotation_mask: np.ndarray,
    scan_affine: np.ndarray,
    scan_header,
    output_path: Path,
    smoothing: str | None = None,
    smooth_size: int = 1,
) -> np.ndarray:
    """Run the graph-cut femur segmentation.

    scan_path: the CT NIfTI file the annotation was painted against (must be
        a real file on disk -- convert a DICOM series first).
    annotation_mask: same shape as the scan volume, values 0 (unlabeled),
        ANNOTATION_FOREGROUND, or ANNOTATION_BACKGROUND -- exactly what
        SliceViewer.annotation_mask holds after painting.
    scan_affine/scan_header: from the nibabel image the scan array came
        from, so the seed volume written to disk lines up voxel-for-voxel.
    output_path: where the resulting binary femur mask NIfTI is written.
    smoothing: None, "gaussian", "open_close", or "shrink_grow" (see
        FemurSegmentation.filters -- applied to the post-processed result).

    Returns the result as a (X, Y, Z) numpy array in the scan's own voxel
    grid, matching nibabel's usual orientation (this function always reads
    the result back via nibabel after ITK writes it, so callers never need
    to reason about ITK's internal axis order).
    """
    _ensure_vendor_on_path()

    import itk

    from FemurSegmentation.IOManager import ImageReader, VolumeWriter
    from FemurSegmentation.boneness import Boneness
    from FemurSegmentation.links import GraphCutLinks
    from FemurSegmentation.utils import array2image, cast_image, image2array
    from FemurSegmentation.filters import (
        binary_curvature_flow, binary_threshold, connected_components,
        dilate, erode, execute_pipeline, iterative_hole_filling,
        itk_binary_morphological_closing, itk_binary_morphological_opening,
        itk_discrete_gaussian, relabel_components,
    )
    from GraphCutSupport import RunGraphCut

    if annotation_mask.shape != scan_header.get_data_shape():
        raise ValueError(
            f"annotation_mask shape {annotation_mask.shape} doesn't match "
            f"the scan's shape {scan_header.get_data_shape()}"
        )
    if not np.any(annotation_mask == ANNOTATION_FOREGROUND) or not np.any(annotation_mask == ANNOTATION_BACKGROUND):
        raise ValueError(
            "annotation_mask needs at least one foreground and one background "
            "painted voxel -- graph-cut has nothing to seed from otherwise."
        )

    with tempfile.TemporaryDirectory(prefix="autoanatomy_femur_") as tmp:
        seed_path = Path(tmp) / "seed.nii.gz"
        nib.save(nib.Nifti1Image(annotation_mask.astype(np.int16), scan_affine, scan_header), str(seed_path))

        reader = ImageReader()
        image = reader(path=str(scan_path), image_type=itk.Image[itk.F, 3])
        init = reader(path=str(seed_path), image_type=itk.Image[itk.SS, 3])

        roi = binary_threshold(image, upper_thr=3000, lower_thr=-400)
        cond, info = image2array(init)
        # itk.GetImageFromArray only supports a limited set of pixel types
        # (no plain int64/"signed long") -- copy cond's own dtype (itk.SS's
        # numpy equivalent) rather than building a fresh array with
        # np.where's default int dtype.
        obj = cond.copy()
        obj[cond == ANNOTATION_FOREGROUND] = 1
        obj[cond != ANNOTATION_FOREGROUND] = 0
        bkg = cond.copy()
        bkg[cond != ANNOTATION_BACKGROUND] = 0
        bkg[cond == ANNOTATION_BACKGROUND] = 1
        obj = array2image(obj, info)
        bkg = array2image(bkg, info)

        boneness = Boneness(image=image, scales=[.5, .75], roi=roi)
        ms_bones = boneness.computeBonenessMeasure()

        gc_links = GraphCutLinks(
            image=image, boneness=ms_bones, roi=roi, obj=obj, bkg=bkg,
            sigma=.25, Lambda=100, bone_ms_thr=.3,
        )
        (cost_sink_flatten, cost_source_flatten, cost_vx, centers_vx, neighbors_vx,
         total_neighbors, cost_from_center, cost_to_center) = gc_links.getLinks()

        result = RunGraphCut(
            gc_links.total_vx,
            np.ascontiguousarray(cost_vx, dtype=np.uint32),
            np.ascontiguousarray(cost_source_flatten, dtype=np.uint32),
            np.ascontiguousarray(cost_sink_flatten, dtype=np.uint32),
            total_neighbors,
            np.ascontiguousarray(centers_vx, dtype=np.uint32),
            np.ascontiguousarray(neighbors_vx, dtype=np.uint32),
            np.ascontiguousarray(cost_from_center, dtype=np.uint32),
            np.ascontiguousarray(cost_to_center, dtype=np.uint32),
        )

        label_id_image = gc_links.vx_id
        label_id_image[gc_links.vx_id != -1] = result
        label_id_image[gc_links.vx_id == -1] = 0
        label_id_image = np.asarray(label_id_image, dtype=np.uint8)
        labeled = array2image(label_id_image, info)

        labeled = cast_image(labeled, itk.SS)
        cc = execute_pipeline(connected_components(labeled))
        cc = relabel_components(cc)
        labeled = binary_threshold(cc, upper_thr=2, lower_thr=0, out_type=itk.F)
        labeled = execute_pipeline(binary_curvature_flow(image=labeled))
        labeled = binary_threshold(labeled, upper_thr=2., lower_thr=.9, out_type=itk.SS)
        labeled = iterative_hole_filling(
            image=labeled, max_iter=5, radius=1, majority_threshold=1, bkg_val=0, fgr_val=1,
        )
        labeled = execute_pipeline(labeled)

        if smoothing == "gaussian":
            labeled = execute_pipeline(itk_discrete_gaussian(labeled, smooth_size))
        elif smoothing == "open_close":
            labeled = execute_pipeline(itk_binary_morphological_opening(labeled, radius=smooth_size))
            labeled = execute_pipeline(itk_binary_morphological_closing(labeled, radius=smooth_size))
        elif smoothing == "shrink_grow":
            labeled = execute_pipeline(erode(labeled, radius=smooth_size))
            labeled = execute_pipeline(dilate(labeled, radius=smooth_size))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        VolumeWriter(path=str(output_path), image=labeled).write()

    result_img = nib.load(str(output_path))
    return np.asanyarray(result_img.dataobj)
