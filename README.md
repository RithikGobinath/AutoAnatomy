# AutoAnatomy

Craniofacial CT segmentation — jaw, teeth, skull, mandibular canal, chewing
muscles — with a terminal GUI, built around a real nnU-Net inference engine.

> **Status: under construction.** The segmentation engine is real (real
> downloaded nnU-Net weights, real GPU/CPU inference, real DICOM/NIfTI I/O).
> The product surface around it — batch runs, 3D mesh export, additional
> craniofacial tasks — is scaffolded but not all wired up yet. Those are
> marked `[Phase 2]` wherever they show up in the TUI.

## What it segments today

**`dental_segmentator`** (5 classes, 0.43x0.31x0.43mm native spacing) — the
default task, tuned for maximum accuracy: mirroring test-time augmentation,
a finer 0.25 sliding-window step size, and the smoothest resampling path.
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

No license required — free for non-commercial and commercial use alike.

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
   0.43x0.31x0.43mm spacing — no cropping, mirroring TTA enabled, 0.25
   sliding-window step size.
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
