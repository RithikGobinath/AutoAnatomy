# AutoAnatomy

Craniofacial CT segmentation — jaw, teeth, skull, mandibular canal, chewing
and neck muscles — with a terminal GUI, built around a real nnU-Net inference
engine and MONAI Auto3DSeg.

> **Status: under construction.** The segmentation engine is real (real
> downloaded nnU-Net weights, real GPU/CPU inference, real DICOM/NIfTI I/O).
> The product surface around it — batch runs, 3D mesh export, additional
> craniofacial tasks — is scaffolded but not all wired up yet. Those are
> marked `[Phase 2]` wherever they show up in the TUI.

## What it segments today

**`dental_segmentator`** (5 classes, 0.43x0.31x0.43mm native spacing) — the
default task, tuned for boundary accuracy: a finer 0.25 sliding-window step
size and the smoothest resampling path. Mirroring test-time augmentation is
available but off by default (roughly 8x slower for a small Dice gain) —
toggle with `--mirroring`/`--no-mirroring` or the TUI's Advanced Settings.
Validated on both conventional CT and CBCT (Dot et al., J Dentistry 2024):

| Label | Structure |
|-------|-----------|
| 1 | upper_skull |
| 2 | mandible |
| 3 | upper_teeth |
| 4 | lower_teeth |
| 5 | mandibular_canal |

**`head_muscles`** (11 classes, 0.75x0.75x1.0mm) — chewing/swallowing
muscles: masseter, temporalis, pterygoids, tongue, digastric.

**`headneck_muscles`** (23 classes, 0.75x0.75x1.0mm) — complementary neck
posture/swallowing muscles, zero overlap with `head_muscles`:
sternocleidomastoid, pharyngeal constrictors, trapezius, platysma, levator
scapulae, scalenes, sterno-thyroid, thyrohyoid, prevertebral. Per upstream's
own docs this wasn't trained on the full TotalSegmentator dataset, so expect
it to be less robust than the other two tasks. **Requires a paid license for
commercial use** (free for non-commercial use) — see [NOTICE](NOTICE).

`dental_segmentator` and `head_muscles` need no license (free for commercial
and non-commercial use alike); `headneck_muscles` is the one exception above.

## Auto3DSeg

The home screen has a separate **Auto3DSeg** section. Choose **Muscle
segmentation**, load a conventional CT scan, then select the outputs:

| Label | Output |
|---|---|
| 1 | `masseter` |
| 2 | `temporalis` |
| 3 | `medial_pterygoid` |

Each mask includes **both left and right sides**. The additional
`auto3dseg_muscles` task runs the pretrained MONAI **Whole Head Segmentation
0.5 mm, v1.0.1** model. Its saved target spacing is approximately
**0.457 × 0.459 × 0.458 mm**. The model's spacing tolerance is preserved;
exports return to the original scan's dimensions and affine. Finer model
spacing alone does not demonstrate better boundary accuracy.

```bash
autoanatomy segment -i scan.nii.gz -o out/ -ta auto3dseg_muscles
autoanatomy segment -i scan.nii.gz -o muscles.nii.gz -ta auto3dseg_muscles --ml
autoanatomy segment -i scan.nii.gz -o out/ -ta auto3dseg_muscles --structures masseter,temporalis
```

The first run downloads the 323 MB checkpoint from the
[Slicer MONAI Auto3DSeg model release](https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/tag/Models).
The archive and checkpoint are checked against pinned SHA-256 hashes; weights
are loaded with PyTorch's restricted tensor loader. The original full classifier
runs before filtering to the three requested muscles. This is an additional
task; `head_muscles` continues to provide its separate left/right structures.
See [NOTICE](NOTICE) for attribution and checkpoint terms.

## Quick start

```bash
# 1. Set up an isolated environment, install deps, download real model weights
./scripts/setup_env.sh        # or scripts\setup_env.ps1 on Windows

# 2. Launch the terminal GUI
.venv/bin/autoanatomy-gui      # or .venv\Scripts\autoanatomy-gui.exe on Windows

# ...or drive it headlessly
.venv/bin/autoanatomy segment -i scan.nii.gz -o out/ --device gpu
.venv/bin/autoanatomy list-structures
.venv/bin/autoanatomy check
```

First run downloads the model checkpoint(s) for whichever task(s) you select
from GitHub releases / Zenodo.

## How it works

1. **DentalSegmentator inference.** The real nnU-Net model (`nnUNetTrainer`,
   task 112) runs directly on the full input scan at its native
   0.43x0.31x0.43mm spacing — no cropping, 0.25 sliding-window step size,
   mirroring TTA available via `--mirroring` (off by default).
2. **Postprocess + export.** Masks are resampled back to the original scan
   geometry (via nnU-Net's own probability-based resampling, not a manual
   nearest-neighbor pass) and written out as NIfTI (per-structure or
   multilabel).

See [`autoanatomy/engine/api.py`](autoanatomy/engine/api.py) for the full
pipeline.

## Project layout

```
autoanatomy/
  engine/   nnU-Net segmentation engine
  cli/      headless CLI (`autoanatomy`)
  tui/      Textual terminal GUI (`autoanatomy-gui`)
scripts/    environment setup
tests/      pytest suite (class map correctness + real inference smoke test)
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attribution.
