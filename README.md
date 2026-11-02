# AutoAnatomy

AutoAnatomy segments craniofacial structures and head and neck muscles in CT scans. It provides a terminal interface and a command-line interface built on nnU-Net and MONAI Auto3DSeg. Input can be a NIfTI scan or a DICOM folder; segmentations are exported as NIfTI files.

The project is under development. Some features shown in the terminal interface, including batch runs, 3D mesh export, and additional craniofacial tasks, are planned but not yet available.

## Segmentation tasks

| Task | Structures | Classes |
| --- | --- | ---: |
| `dental_segmentator` | Upper skull, mandible, upper and lower teeth, mandibular canal | 5 |
| `head_muscles` | Chewing and swallowing muscles, including masseter, temporalis, pterygoids, tongue, and digastric | 11 |
| `headneck_muscles` | Neck and swallowing muscles, including sternocleidomastoid, pharyngeal constrictors, trapezius, and scalenes | 23 |
| `auto3dseg_muscles` | Masseter, temporalis, and medial pterygoid; each mask combines both sides | 3 |

Run `autoanatomy list-structures` for the complete list of labels. The terminal interface also includes a semi-automatic femur segmentation workflow using painted foreground and background seeds. It requires a NIfTI input; see its [third-party documentation](autoanatomy/third_party/femur_segmentation/README.md) for usage and licensing details.

The dental task uses the DentalSegmentator checkpoint at its native 0.43 × 0.31 × 0.43 mm spacing. Mirroring test-time augmentation is off by default and can be enabled with `--mirroring` or in Advanced Settings. The `head_muscles` and `headneck_muscles` models use 0.75 × 0.75 × 1.0 mm spacing. The latter was not trained on the full upstream TotalSegmentator dataset and may be less robust.

The Auto3DSeg task uses the MONAI Whole Head Segmentation 0.5 mm v1.0.1 checkpoint, with an approximately 0.457 × 0.459 × 0.458 mm target spacing. It runs the full classifier before selecting the three muscle outputs. Outputs are restored to the original scan geometry. Model spacing alone does not establish boundary accuracy.

## Getting started

Python 3.10 or later is required. The setup scripts create a virtual environment, install the package and dependencies, and download model weights. An NVIDIA GPU can accelerate inference; CPU inference is also supported. Segmentation requires at least 5 GB of free disk space.

On Linux or macOS:

```bash
./scripts/setup_env.sh
.venv/bin/autoanatomy-gui
```

On Windows, in PowerShell:

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\autoanatomy-gui.exe
```

The setup scripts download all registered task models. Model weights are also downloaded when needed if they are not cached. The Auto3DSeg checkpoint is downloaded from the [Slicer MONAI Auto3DSeg model release](https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/tag/Models); its archive and checkpoint are verified against pinned SHA-256 hashes.

## Command-line usage

The examples below assume the virtual environment is activated. Use `--device cpu` if a supported GPU is unavailable.

```bash
# Check the device, cached weights, and disk space
autoanatomy check

# List tasks and their structures
autoanatomy list-structures

# Save one NIfTI file per structure
autoanatomy segment -i scan.nii.gz -o out/ --device gpu

# Save selected Auto3DSeg muscles
autoanatomy segment -i scan.nii.gz -o out/ -ta auto3dseg_muscles --structures masseter,temporalis

# Save a single multilabel NIfTI file
autoanatomy segment -i scan.nii.gz -o muscles.nii.gz -ta auto3dseg_muscles --ml
```

Use `autoanatomy segment --help` for options including multiple tasks, statistics, memory usage, and postprocessing.

## Project layout

```text
autoanatomy/engine/   Segmentation and inference
autoanatomy/cli/      Command-line interface
autoanatomy/tui/      Terminal interface
scripts/              Environment setup
tests/                Automated tests
```

## License and attribution

AutoAnatomy is licensed under the [Apache License 2.0](LICENSE). Third-party components and model weights have their own terms; see [NOTICE](NOTICE). In particular, `headneck_muscles` requires a commercial license for commercial use, and the Auto3DSeg checkpoint's terms should be confirmed with its publisher. The femur graph-cut solver is licensed for research purposes; see its [documentation](autoanatomy/third_party/femur_segmentation/README.md).
