# Changelog

## 0.4.0

- Add the `individual_teeth` task: the 32 permanent teeth as separate masks,
  named by FDI number (`tooth_11` ... `tooth_48`). `dental_segmentator` and
  its lumped `upper_teeth`/`lower_teeth` output are unchanged.
- Built on the ToothSeg nnU-Net checkpoints (MIC-DKFZ, Apache-2.0, trained on
  the ToothFairy2 dataset). A semantic model and a border-core instance model
  are combined: instances are decoded by connected components and each is
  labelled by its majority semantic class. Only the individual-tooth output
  was taken from ToothSeg.
- Run both models with sequential in-process export, because full-volume
  resampling to 0.2/0.3 mm can otherwise kill the export worker.
- The checkpoints (about 920 MB) download from Zenodo on first use.
- Merge the 0.3.0 release line (Auto3DSeg muscles, femur semi-auto) with the
  local main branch.

This release does not claim an independently measured accuracy result for
`individual_teeth`; it was smoke-tested end to end but not validated against
labelled dental scans.

## 0.3.0

- Add a dedicated **Auto3DSeg** home-screen section and muscle workflow.
- Add the `auto3dseg_muscles` CLI/Python task for **masseter, temporalis and
  medial pterygoid**, with both sides combined in each muscle mask.
- Run the released MONAI Whole Head Segmentation 0.5 mm v1.0.1 checkpoint
  at its saved approximately 0.46 mm target spacing. Preserve its intensity
  normalization, spacing tolerance, Gaussian sliding-window overlap and
  two-component muscle cleanup, and restore the original scan geometry.
- Support selected structures, binary or multilabel NIfTI export, statistics,
  GPU/CPU devices, optional mirroring, and live completed-tile progress.
- Verify downloaded archives/checkpoints and load tensor weights without
  evaluating checkpoint configuration expressions.
- Include the terminal GUI stylesheet in installed packages.

The checkpoint is downloaded from its publisher on first use. This release
does not claim an independently measured improvement in muscle accuracy.
