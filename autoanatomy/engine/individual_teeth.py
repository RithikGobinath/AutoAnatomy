"""Individual-tooth segmentation (32 permanent teeth, FDI numbering).

Combines two ToothSeg nnU-Net v2 checkpoints (Hanisch et al., MIC-DKFZ;
Apache-2.0; trained on the ToothFairy2 Challenge dataset):

- Dataset121 (semantic): one of 32 FDI-ordered tooth classes per voxel.
- Dataset123 (instance): border/core format, decoded into separate tooth
  instances via connected components.

Each instance is then relabeled with its majority semantic class -- the same
two-model combination ToothSeg's own published inference recipe uses
(inference_toothfairy2.sh + assign_majority_tooth_labels.py). Only that one
capability was ported: no SimpleITK dependency (plain nibabel instead), no
DP-based self-correction (assign_mincost_tooth_labels.py), and none of
ToothSeg's other classes (maxilla/mandible/mandibular canal already live in
dental_segmentator).

Unlike this build's other tasks, the two checkpoints use different trainers
and different training spacings (0.3mm semantic / 0.2mm instance). nnU-Net's
own per-model preprocessing resamples each checkpoint's input independently
and resamples its output back to the original image's resolution on export
(see nnunetv2.inference.export_prediction), so -- unlike every other task in
this codebase -- no explicit pre/post resample step is needed here, and the
two models' outputs already share the original image's voxel grid.
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from acvl_utils.instance_segmentation.instance_as_semantic_seg import (
    convert_semantic_to_instanceseg,
    postprocess_instance_segmentation,
)

from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.nifti_ext_header import add_label_map_to_nifti
from autoanatomy.engine.statistics import get_basic_statistics
from autoanatomy.engine.weights import nostdout

SEMANTIC_TASK_ID = 121
SEMANTIC_TRAINER = "nnUNetTrainer_onlyMirror01_DASegOrd0"
SEMANTIC_MODEL = "3d_fullres_resample_torch_256_bs8_ctnorm"

INSTANCE_TASK_ID = 123
INSTANCE_TRAINER = "nnUNetTrainer"
INSTANCE_MODEL = "3d_fullres_resample_torch_192_bs8_ctnorm"

# Both checkpoints only ship a single published fold.
FOLD = 5


def _assign_majority_tooth_labels(semantic_seg: np.ndarray, instance_seg: np.ndarray) -> np.ndarray:
    """Port of ToothSeg's assign_majority_tooth_labels.py (allow_bg=True,
    isolated_semsegs_as_new_instances=False, min_tooth_volume=0 -- the exact
    settings its own published inference_toothfairy2.sh uses).

    For each predicted tooth instance, relabel it with whichever of the 32
    FDI-ordered semantic classes (or background) is most common among that
    instance's voxels. An instance whose voxels are mostly semantic
    background is dropped.
    """
    output = np.zeros_like(semantic_seg)
    num_classes = int(semantic_seg.max()) + 1
    for instance_id in np.unique(instance_seg):
        if instance_id == 0:
            continue
        mask = instance_seg == instance_id
        freq = np.bincount(semantic_seg[mask], minlength=num_classes)
        predicted_label = int(np.argmax(freq))
        if predicted_label != 0:
            output[mask] = predicted_label
    return output


def run_individual_teeth(input, output=None, multilabel_image=True, roi_subset=None,
                         output_type="nifti", nora_tag="None", quiet=False, verbose=False,
                         skip_saving=False, device="cuda", debug=False,
                         nr_threads_resampling=1, nr_threads_saving=6, tta=False,
                         statistics=False):
    """Run the individual_teeth task. Mirrors nnUNet_predict_image's return
    signature (multilabel segmentation, original CT image, stats-or-None) so
    engine/api.py's segment() can use it as a drop-in replacement for that
    call for this one task.
    """
    from autoanatomy.engine.nnunet_runner import nnUNetv2_predict

    if output_type != "nifti":
        raise ValueError("task individual_teeth only supports output_type='nifti' in this build.")

    if not isinstance(input, Nifti1Image):
        input = Path(input)
        if not (str(input).endswith(".nii") or str(input).endswith(".nii.gz")):
            raise ValueError("task individual_teeth only supports NIfTI input in this build.")
        img_in_orig = nib.load(input)
    else:
        img_in_orig = input

    if output is not None:
        output = Path(output)

    st = time.time()
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_individual_teeth_", ignore_cleanup_errors=True) as tmp_folder:
        tmp_dir = Path(tmp_folder)
        nib.save(nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine), tmp_dir / "s01_0000.nii.gz")

        # low_memory=True unconditionally (not threaded from the caller, unlike
        # this codebase's other low_memory uses): both checkpoints resample
        # the *entire* input to a fine native spacing (0.2mm instance / 0.3mm
        # semantic) regardless of the input volume's real field of view, so
        # the resulting per-case array can be large enough that nnU-Net's
        # normal background export-worker pool dies from the OS killing it
        # for low memory when it tries to pickle that array across to a
        # worker process -- confirmed in practice (RuntimeError:
        # "Segmentation export worker died") even on a full-head CT with a
        # GPU available. Running export synchronously in-process is slower
        # but avoids that failure mode entirely.
        if not quiet: print("Predicting tooth instances...")
        instance_dir = tmp_dir / "instance"
        instance_dir.mkdir()
        with nostdout(verbose):
            nnUNetv2_predict(tmp_dir, instance_dir, INSTANCE_TASK_ID, model=INSTANCE_MODEL, folds=[FOLD],
                             trainer=INSTANCE_TRAINER, tta=tta,
                             num_threads_preprocessing=nr_threads_resampling,
                             num_threads_nifti_save=nr_threads_saving, device=device, quiet=quiet,
                             low_memory=True)
        border_core = nib.load(instance_dir / "s01.nii.gz")
        spacing = border_core.header.get_zooms()
        border_core_data = np.asanyarray(border_core.dataobj).astype(np.uint8)

        if not quiet: print("Predicting tooth classes...")
        semantic_dir = tmp_dir / "semantic"
        semantic_dir.mkdir()
        with nostdout(verbose):
            nnUNetv2_predict(tmp_dir, semantic_dir, SEMANTIC_TASK_ID, model=SEMANTIC_MODEL, folds=[FOLD],
                             trainer=SEMANTIC_TRAINER, tta=tta,
                             num_threads_preprocessing=nr_threads_resampling,
                             num_threads_nifti_save=nr_threads_saving, device=device, quiet=quiet,
                             low_memory=True)
        semantic_data = np.asanyarray(nib.load(semantic_dir / "s01.nii.gz").dataobj).astype(np.uint8)
        if not quiet: print(f"  Predicted in {time.time() - st:.2f}s")

        if not quiet: print("Assigning tooth labels to instances...")
        instance_seg = convert_semantic_to_instanceseg(border_core_data, spacing=spacing)
        instance_seg = postprocess_instance_segmentation(instance_seg.astype(np.uint32))
        img_data = _assign_majority_tooth_labels(semantic_data, instance_seg).astype(np.uint8)

    label_map = dict(class_map["individual_teeth"])
    if roi_subset is not None:
        label_map = {k: v for k, v in label_map.items() if v in roi_subset}
        img_data *= np.isin(img_data, list(label_map.keys()))

    new_header = img_in_orig.header.copy()
    new_header.set_data_dtype(np.uint8)
    img_out = nib.Nifti1Image(img_data, img_in_orig.affine, new_header)
    img_out = add_label_map_to_nifti(img_out, label_map)

    stats = None
    if statistics:
        if not quiet: print("Calculating statistics...")
        st_stats = time.time()
        stats_file = None
        if isinstance(statistics, (str, Path)):
            stats_file = Path(statistics).absolute()
        elif output is not None:
            stats_dir = output.parent if multilabel_image else output
            stats_dir.mkdir(exist_ok=True, parents=True)
            stats_file = stats_dir / "statistics.json"
        stats = get_basic_statistics(img_data, img_in_orig, stats_file, quiet, "individual_teeth",
                                     True, roi_subset)
        if not quiet: print(f"  calculated in {time.time() - st_stats:.2f}s")

    if output is not None and not skip_saving:
        if not quiet: print("Saving segmentations...")
        out_dir = output.parent if multilabel_image else output
        out_dir.mkdir(exist_ok=True, parents=True)
        if multilabel_image:
            nib.save(img_out, output)
            if nora_tag != "None":
                import subprocess
                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output} --addtag atlas", shell=True)
        else:
            for k, v in label_map.items():
                nib.save(nib.Nifti1Image((img_data == k).astype(np.uint8), img_out.affine, new_header),
                         out_dir / f"{v}.nii.gz")

    return img_out, img_in_orig, stats
