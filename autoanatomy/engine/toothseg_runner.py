"""
Custom two-branch inference pipeline for the "toothseg" task: individual,
FDI-numbered tooth instance segmentation.

Adapted from ToothSeg (Apache License 2.0):
    van Nistelrooij N., Kraemer L., Kempers S., Beyer M., Bolelli F., Xi T.,
    Berge S., Heiland M., Maier-Hein K.H., Vinayahalingam S., Isensee F.
    "ToothSeg: Robust Tooth Instance Segmentation and Numbering in CBCT using
    Deep Learning and Self-Correction." IEEE J Biomed Health Inform, 2026.
    https://github.com/MIC-DKFZ/ToothSeg

Unlike the other three AutoAnatomy tasks, this does not go through
nnunet_runner.nnUNet_predict_image (a single-checkpoint, class-map-driven
wrapper). It runs two independent nnU-Net v2 checkpoints -- a 33-class
semantic branch (background + 32 FDI tooth positions) and a 3-class
border/core instance branch -- via the lower-level nnUNetv2_predict
primitive, then reproduces ToothSeg's own published postprocessing:
border-core connected-components instance extraction, followed by
dynamic-programming FDI tooth numbering that uses the semantic branch's
softmax probabilities to resolve merged/split teeth and pick the most
likely tooth-number sequence per dental arch.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from scipy.stats import multivariate_normal

from autoanatomy.engine.class_map import class_map, TOOTHSEG_SEMANTIC_FDI_ORDER
from autoanatomy.engine.weights import download_pretrained_weights

SEMANTIC_TASK_ID = 121
SEMANTIC_TRAINER = "nnUNetTrainer_onlyMirror01_DASegOrd0"
SEMANTIC_CONFIG = "3d_fullres_resample_torch_256_bs8_ctnorm"
SEMANTIC_FOLD = 5
# Verified from this checkpoint's own plans.json (isotropic on all 3 axes).
SEMANTIC_SPACING_MM = 0.3

INSTANCE_TASK_ID = 123
INSTANCE_TRAINER = "nnUNetTrainer"
INSTANCE_CONFIG = "3d_fullres_resample_torch_192_bs8_ctnorm"
INSTANCE_FOLD = 5

_FDI_DISTR_PATH = Path(__file__).parent / "toothseg_data" / "fdi_pair_distrs.json"

# ToothSeg's own defaults (border_core_to_instances.py __main__ block):
# volumes in mm^3, roughly 2000 voxels at 0.2mm isotropic spacing.
_SMALL_CENTER_THRESHOLD_MM3 = 16
_ISOLATED_BORDER_THRESHOLD_MM3 = 0


def _run_branch(img_in: Nifti1Image, task_id, trainer, config, fold, device, quiet):
    """Run one ToothSeg nnU-Net branch on an in-memory image via the standard
    file-based nnU-Net pipeline (resamples back to img_in's own resolution).

    Only used for the instance (border/core) branch now -- the semantic
    branch needs per-class probabilities, which is handled separately by
    _run_semantic_branch_lowmem to avoid materializing a 33-channel array at
    full scan resolution (see run_toothseg's docstring / the OOM this used to
    cause).

    Returns (segmentation: np.ndarray uint8, affine: np.ndarray).
    """
    from autoanatomy.engine.nnunet_runner import nnUNetv2_predict

    download_pretrained_weights(task_id)

    with tempfile.TemporaryDirectory(prefix="toothseg_branch_", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        dir_in = tmp_dir / "in"
        dir_out = tmp_dir / "out"
        dir_in.mkdir()
        dir_out.mkdir()
        nib.save(img_in, dir_in / "s01_0000.nii.gz")

        # Single case per run -- a second background export worker process
        # buys nothing here and only adds its own baseline memory overhead.
        nnUNetv2_predict(
            dir_in, dir_out, task_id, model=config, folds=[fold], trainer=trainer,
            tta=False, plans="nnUNetPlans", device=device, quiet=quiet,
            num_threads_nifti_save=1,
        )

        seg_img = nib.load(str(dir_out / "s01.nii.gz"))
        seg_data = np.asarray(seg_img.dataobj).astype(np.uint8)
        affine = seg_img.affine.copy()

        return seg_data, affine


def _run_semantic_branch_lowmem(img_in: Nifti1Image, instance_seg: np.ndarray, instance_affine: np.ndarray,
                                device, quiet, verbose):
    """Run the semantic branch and align the (already-computed) instance
    segmentation to its working resolution -- WITHOUT going through nnU-Net's
    own resample-probabilities-back-to-original-resolution export path.

    For a 33-class probability volume, that export needs several full-scan-
    resolution float32 arrays alive at once (the resampled probabilities, a
    zero-filled full-size canvas to un-crop into, etc) -- for a typical head
    CT this is tens of GB and is what was crashing with
    "Segmentation export worker died... insufficient available CPU RAM".

    The actual algorithm (_assign_fdi_labels) only ever looks up
    probabilities at specific instance-voxel locations, never needs a dense
    full-resolution volume. So instead: get raw probabilities at the
    network's own (cropped + resampled) working resolution directly via
    predict_logits_from_preprocessed_data (skipping nnU-Net's own export
    step entirely), and resample the SINGLE-CHANNEL instance map to match
    that resolution instead -- ~33x cheaper than the reverse.

    Returns (probabilities: (33,x,y,z) float32 at the network's working
    resolution, aligned_instance_seg: (x,y,z) same resolution,
    undo_to_original: callable that maps a single-channel array at that same
    working resolution back to instance_seg's original resolution/orientation).
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    download_pretrained_weights(SEMANTIC_TASK_ID)

    use_gpu = str(device).startswith("cuda") or str(device) == "gpu"
    torch_device = torch.device("cuda") if use_gpu else torch.device("cpu")

    model_folder = get_output_folder(SEMANTIC_TASK_ID, SEMANTIC_TRAINER, "nnUNetPlans", SEMANTIC_CONFIG)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=use_gpu, device=torch_device,
        verbose=verbose, verbose_preprocessing=verbose, allow_tqdm=not quiet,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=[SEMANTIC_FOLD], checkpoint_name="checkpoint_final.pth",
    )

    with tempfile.TemporaryDirectory(prefix="toothseg_semantic_lowmem_", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        img_path = tmp_dir / "s01_0000.nii.gz"
        seg_path = tmp_dir / "s01_instances.nii.gz"
        nib.save(img_in, img_path)
        # uint16: instance IDs are small integers, but not guaranteed < 256
        # once split-instance IDs are assigned upstream.
        nib.save(nib.Nifti1Image(instance_seg.astype(np.uint16), instance_affine), seg_path)

        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=verbose)
        data, seg, properties = preprocessor.run_case(
            [str(img_path)], str(seg_path), predictor.plans_manager,
            predictor.configuration_manager, predictor.dataset_json,
        )

        if not quiet:
            print(f"  Semantic branch working resolution: {data.shape[1:]} "
                 f"(native scan: {instance_seg.shape})")

        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
        probabilities = torch.softmax(logits.float(), dim=0).numpy().astype(np.float32)
        # crop_to_nonzero marks seg voxels outside the image's own nonzero
        # mask as -1 (an "ignore" label meant for training loss masking) --
        # not applicable here, so fold it back into background (0).
        aligned_instance_seg = np.maximum(seg[0], 0).astype(np.uint16)

    plans_manager = predictor.plans_manager
    configuration_manager = predictor.configuration_manager

    # preprocessor.run_case loads/writes images via SimpleITK (nnU-Net's own
    # convention), whose GetArrayFromImage returns arrays as (Z, Y, X) --
    # the reverse of nibabel's (X, Y, Z). Every array above (data/seg/
    # probabilities/aligned_instance_seg) is still in that (Z, Y, X) axis
    # order. Everywhere else in this file (instance_affine, _to_canonical,
    # _assign_fdi_labels, the final Nifti1Image) is nibabel/(X, Y, Z)-based,
    # so convert here, once, at this SimpleITK/nibabel boundary.
    probabilities = np.ascontiguousarray(probabilities.transpose(0, 3, 2, 1))
    aligned_instance_seg = np.ascontiguousarray(aligned_instance_seg.transpose(2, 1, 0))

    def undo_to_original(model_res_array: np.ndarray) -> np.ndarray:
        """Map a single-channel (X, Y, Z)-ordered array at the working
        resolution above back to instance_seg's original shape/orientation.
        Mirrors nnU-Net's own export-time resample-then-uncrop-then-
        transpose-back, just applied to a single-channel array instead of
        the full probability volume (and converting back to SimpleITK's
        (Z, Y, X) convention first, since that's what properties/
        configuration_manager below are all expressed in)."""
        model_res_zyx = np.ascontiguousarray(model_res_array.transpose(2, 1, 0))

        spacing_transposed = [properties["spacing"][i] for i in plans_manager.transpose_forward]
        current_spacing = configuration_manager.spacing
        if len(current_spacing) < len(properties["shape_after_cropping_and_before_resampling"]):
            current_spacing = [spacing_transposed[0], *current_spacing]

        # int16, not uint16: torch's CPU backend doesn't implement basic
        # comparisons (">") for uint16, which resample_torch_fornnunet needs
        # internally. Instance/FDI label values here are always comfortably
        # within int16's range.
        resampled = configuration_manager.resampling_fn_seg(
            model_res_zyx[None].astype(np.int16),
            properties["shape_after_cropping_and_before_resampling"],
            current_spacing, spacing_transposed,
        )
        resampled = np.asarray(resampled)[0]

        full = np.zeros(properties["shape_before_cropping"], dtype=resampled.dtype)
        full = insert_crop_into_image(full, resampled, properties["bbox_used_for_cropping"])
        full = full.transpose(plans_manager.transpose_backward)
        return np.ascontiguousarray(full.transpose(2, 1, 0))

    return probabilities, aligned_instance_seg, undo_to_original


def _border_core_to_instances(border_core: np.ndarray, spacing) -> np.ndarray:
    """background=0, core=1, border=2 -> connected-component instance IDs (1..N)."""
    from acvl_utils.instance_segmentation.instance_as_semantic_seg import (
        convert_semantic_to_instanceseg, postprocess_instance_segmentation,
    )
    instance_seg = convert_semantic_to_instanceseg(
        border_core, np.asarray(spacing),
        small_center_threshold=_SMALL_CENTER_THRESHOLD_MM3,
        isolated_border_as_separate_instance_threshold=_ISOLATED_BORDER_THRESHOLD_MM3,
    )
    return postprocess_instance_segmentation(instance_seg)


def _load_normals():
    with open(_FDI_DISTR_PATH) as f:
        pair_dists = json.load(f)
    normals = []
    for i in range(32):
        row = []
        for j in range(32):
            if i // 16 != j // 16:
                row.append(None)
                continue
            row.append(multivariate_normal(
                mean=pair_dists["means"][i][j][:2],
                cov=np.array(pair_dists["covs"][i][j])[:2, :2],
            ))
        normals.append(row)
    return normals


def _determine_sequence(centroids):
    idxs = np.full(centroids.shape[0], -1, dtype=int)
    inverse = np.full(centroids.shape[0], -1, dtype=int)
    first_idx = centroids[:, 1].argmin()
    idxs[0] = first_idx
    inverse[first_idx] = 0
    for i in range(1, centroids.shape[0]):
        dists = np.linalg.norm(centroids[inverse == -1] - centroids[idxs[i - 1]], axis=-1)
        next_idx = np.nonzero(inverse == -1)[0][dists.argmin()]
        idxs[i] = next_idx
        inverse[next_idx] = i
    return idxs, inverse


def _determine_transition_probabilities(normals, centroids, is_arch_lower, seq_idxs):
    index = np.arange(16, 32) if is_arch_lower else np.arange(16)
    trans_log_probs = np.zeros((max(centroids.shape[0] - 1, 0), 16, 16))
    for i, (idx1, idx2) in enumerate(zip(seq_idxs[:-1], seq_idxs[1:])):
        offsets = centroids[idx2] - centroids[idx1]
        for j in range(16):
            for k in range(16):
                trans_log_probs[i, j, k] = normals[index[j]][index[k]].logpdf(offsets[:2])
    return trans_log_probs


def _dynamic_programming(tooth_probs, seq_idxs, trans_log_probs, tooth_factor=4.0):
    """Returns, for each arch-local slot in sequence order, the most likely
    tooth position (0-15) accounting for both per-tooth class probability and
    the learned pairwise spatial-offset distributions between neighbours."""
    tooth_log_probs = np.log(tooth_probs)

    q = np.zeros_like(tooth_log_probs)
    q[0] = -tooth_factor * tooth_log_probs[seq_idxs[0]]
    p = np.zeros_like(q, dtype=int)
    p[0] = np.arange(16)

    for i in range(1, tooth_probs.shape[0]):
        for j in range(16):
            prev_costs = q[i - 1]
            trans_costs = -trans_log_probs[i - 1, :, j].copy()
            costs = prev_costs + trans_costs
            m = costs.min()
            q[i, j] = m - tooth_factor * tooth_log_probs[seq_idxs[i], j]
            p[i, j] = costs.argmin()

    path = [int(q[-1].argmin())]
    for i in range(tooth_probs.shape[0] - 1):
        path.insert(0, int(p[-1 - i, path[0]]))
    return np.array(path)


def _assign_fdi_labels(raw_instance_seg: np.ndarray, semantic_probs: np.ndarray, spacing, normals) -> np.ndarray:
    """
    raw_instance_seg: (X,Y,Z) connected-component instance IDs, 0=background.
    semantic_probs: (33, X,Y,Z) softmax probabilities, same shape/orientation.
    Returns an (X,Y,Z) uint8 array of final FDI tooth numbers (0 = none).
    """
    instances, inverse = np.unique(raw_instance_seg, return_inverse=True)
    cc_seg = inverse.reshape(raw_instance_seg.shape)

    # Split any single connected component that actually straddles multiple
    # confidently-predicted tooth classes (merged-teeth case), and build a
    # per-split-instance centroid + mean class-probability vector.
    inst_centroids = np.zeros((0, 3))
    inst_probs = np.zeros((0, 33))
    split_seg = np.zeros_like(cc_seg)
    for inst_idx in range(1, instances.shape[0]):
        inst_mask = cc_seg == inst_idx
        voxel_probs = semantic_probs[:, inst_mask]
        class_idxs = voxel_probs.argmax(0)
        scores = np.zeros(33)
        for class_idx in np.nonzero(voxel_probs.mean(1) >= 0.1)[0]:
            if not np.any(class_idxs == class_idx):
                continue
            scores[class_idx] = voxel_probs[class_idx, class_idxs == class_idx].mean()

        if (scores[1:] >= 0.95).sum() <= 1:
            split_idxs = np.zeros(int(inst_mask.sum()), dtype=int)
        else:
            candidate_classes = np.nonzero(scores[1:] >= 0.95)[0] + 1
            split_idxs = semantic_probs[candidate_classes][:, inst_mask].argmax(0)

        voxel_idxs = np.column_stack(np.nonzero(inst_mask))
        for split_idx in np.unique(split_idxs):
            sel = split_idxs == split_idx
            centroid = voxel_idxs[sel].mean(0) * spacing
            inst_centroids = np.concatenate((inst_centroids, [centroid]))
            prob_dist = voxel_probs[:, sel].mean(1)
            inst_probs = np.concatenate((inst_probs, [prob_dist]))
            split_seg[tuple(voxel_idxs[sel].T)] = split_seg.max() + 1

    if inst_centroids.shape[0] == 0:
        return np.zeros_like(cc_seg, dtype=np.uint8)

    is_background = inst_probs[:, 0] >= 0.95
    kept_idxs = np.nonzero(~is_background)[0]
    inst_centroids = inst_centroids[kept_idxs]
    inst_probs = inst_probs[kept_idxs]

    # positions[i] = semantic-channel index (1-32) assigned to kept instance i
    positions = np.zeros(inst_centroids.shape[0], dtype=int)
    is_inst_lower = inst_probs[:, 17:].sum(-1) > inst_probs[:, 1:17].sum(-1)
    for is_arch_lower in (False, True):
        if not np.any(is_arch_lower == is_inst_lower):
            continue
        arch_idxs = np.nonzero(is_arch_lower == is_inst_lower)[0]
        arch_centroids = inst_centroids[arch_idxs]
        arch_probs = inst_probs[arch_idxs]
        arch_probs = arch_probs[:, 17:] if is_arch_lower else arch_probs[:, 1:17]
        arch_probs = arch_probs / arch_probs.sum(axis=1, keepdims=True)

        seq_idxs, _ = _determine_sequence(arch_centroids)
        trans_probs = _determine_transition_probabilities(normals, arch_centroids, is_arch_lower, seq_idxs)
        path = _dynamic_programming(arch_probs, seq_idxs, trans_probs)

        arch_positions = path + 16 * is_arch_lower + 1
        positions[arch_idxs[seq_idxs]] = arch_positions

    # Rebuild the full (split-instance-id -> semantic position 1-32, or 0) map,
    # re-inserting the background instances we filtered out above as 0.
    position_map = np.zeros(is_background.shape[0] + 1, dtype=int)
    position_map[kept_idxs + 1] = positions

    fdi_lookup = np.array([0] + list(TOOTHSEG_SEMANTIC_FDI_ORDER), dtype=np.uint8)
    final_seg = fdi_lookup[position_map[split_seg]]
    return final_seg.astype(np.uint8)


def _to_canonical(data: np.ndarray, affine: np.ndarray, is_multichannel: bool = False):
    """Reorient a (X,Y,Z) or (C,X,Y,Z) array to RAS-closest canonical, matching
    what the spatial reasoning in the FDI-assignment step assumes."""
    orientation = nib.io_orientation(affine)
    if is_multichannel:
        reoriented = np.stack([nib.apply_orientation(data[c], orientation) for c in range(data.shape[0])])
    else:
        reoriented = nib.apply_orientation(data, orientation)
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(orientation, data.shape[-3:])
    return reoriented, canonical_affine, orientation


def _invert_orientation(ornt: np.ndarray) -> np.ndarray:
    """Invert an nibabel orientation transform (as returned by io_orientation),
    i.e. return the ornt that undoes apply_orientation(data, ornt)."""
    inverse = np.zeros_like(ornt)
    for out_axis in range(ornt.shape[0]):
        in_axis, flip = ornt[out_axis]
        inverse[int(in_axis)] = [out_axis, flip]
    return inverse


def run_toothseg(
    input_img: Nifti1Image,
    device="cuda",
    nr_threads_resampling=1,
    quiet=False,
    verbose=False,
):
    """Run the full ToothSeg pipeline on an already-loaded, already-oriented
    CT image. Returns a Nifti1Image whose voxel values are FDI tooth numbers
    (0 = background), matching class_map["toothseg"]."""
    if not quiet:
        print("Running ToothSeg instance branch (border/core)...")
    instance_border_core, instance_affine = _run_branch(
        input_img, INSTANCE_TASK_ID, INSTANCE_TRAINER, INSTANCE_CONFIG, INSTANCE_FOLD, device, quiet,
    )

    if not quiet:
        print("Converting border/core prediction to per-tooth instances...")
    instance_spacing = np.asarray(nib.affines.voxel_sizes(instance_affine))
    raw_instances = _border_core_to_instances(instance_border_core, instance_spacing)

    if not quiet:
        print("Running ToothSeg semantic branch (individual tooth classes)...")
    semantic_probs, aligned_instances, undo_to_original = _run_semantic_branch_lowmem(
        input_img, raw_instances, instance_affine, device, quiet, verbose,
    )
    # The semantic branch's working resolution is isotropic (0.3mm on every
    # axis for this checkpoint), so axis order doesn't matter here.
    semantic_spacing = np.full(3, SEMANTIC_SPACING_MM, dtype=float)

    # aligned_instances and semantic_probs came from the exact same
    # preprocessing call (run_case), so they already share the same grid --
    # reuse instance_affine's orientation (sign/permutation only, which
    # nib.io_orientation reads) since this dataset's transpose is identity.
    raw_instances_canon, canon_affine, orientation = _to_canonical(aligned_instances, instance_affine)
    semantic_probs_canon, _, _ = _to_canonical(semantic_probs, instance_affine, is_multichannel=True)

    if not quiet:
        print("Assigning FDI tooth numbers...")
    normals = _load_normals()
    fdi_seg_canon = _assign_fdi_labels(raw_instances_canon, semantic_probs_canon, semantic_spacing, normals)

    # Undo the canonical reorientation, then map back from the semantic
    # branch's working resolution to the scan's own original resolution.
    fdi_seg_model_res = nib.apply_orientation(fdi_seg_canon, _invert_orientation(orientation))
    fdi_seg = undo_to_original(fdi_seg_model_res)

    return nib.Nifti1Image(fdi_seg.astype(np.uint8), instance_affine)
