# Changelog

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
