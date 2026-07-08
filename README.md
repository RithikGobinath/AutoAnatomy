# AutoAnatomy

Craniofacial CT segmentation — mandible, teeth, skull, head, sinuses — with a
terminal GUI. A focused fork of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
stripped down to a single task and rebuilt around it.

> **Status: under construction.** The segmentation engine is real (real
> downloaded nnU-Net weights, real GPU/CPU inference, real DICOM/NIfTI I/O).
> The product surface around it — batch runs, DICOM-SEG/RT-STRUCT/STL export,
> the other TotalSegmentator tasks — is scaffolded but not all wired up yet.
> Those are marked `[Phase 2]` wherever they show up in the TUI.

## What it segments today

One task, `craniofacial_structures` (7 classes, 0.5mm isotropic):

| Label | Structure |
|-------|-----------|
| 1 | mandible |
| 2 | teeth_lower |
| 3 | skull |
| 4 | head |
| 5 | sinus_maxillary |
| 6 | sinus_frontal |
| 7 | teeth_upper |

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

First run downloads two real model checkpoints from GitHub releases: the
craniofacial_structures model itself (~230MB) and a rough whole-body model
(~135MB) used internally to crop the scan down to the skull region before the
high-resolution model runs.

## How it works

1. **Rough crop.** A 6mm low-res pass of the upstream "total" model locates
   the skull so the expensive 0.5mm model only has to run on a small ROI.
2. **craniofacial_structures inference.** The real nnU-Net model
   (`nnUNetTrainer_DASegOrd0_NoMirroring`, task 115) segments the cropped
   region at 0.5mm.
3. **Postprocess + export.** Masks are resampled back to the original scan
   geometry and written out as NIfTI (per-structure or multilabel).

See [`autoanatomy/engine/api.py`](autoanatomy/engine/api.py) for the full
pipeline.

## Project layout

```
autoanatomy/
  engine/   nnU-Net inference engine (stripped from TotalSegmentator)
  cli/      headless CLI (`autoanatomy`)
  tui/      Textual terminal GUI (`autoanatomy-gui`)
scripts/    environment setup
tests/      pytest suite (class map correctness + real inference smoke test)
```

## Attribution

Derived from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
by Jakob Wasserthal and contributors, Apache 2.0. See [NOTICE](NOTICE).
