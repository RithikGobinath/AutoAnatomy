import argparse
import shutil
import sys
from pathlib import Path

from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.registry import TASKS, ROADMAP_TASKS, format_classes_table

# See autoanatomy/tui/screens/run_progress.py for why this guard exists:
# nnU-Net's background export workers can hang/crash on a disk-full write in
# a way that's hard to catch cleanly, so refuse to start below this threshold.
MIN_FREE_DISK_GB = 5


def cmd_segment(args):
    # Imported lazily: this pulls in torch/nnunetv2, which is slow and
    # unnecessary for --list-structures / check / download-weights.
    from autoanatomy.engine.api import totalsegmentator, validate_device_type_api

    try:
        validate_device_type_api(args.device)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    output = Path(args.output)
    _, _, free_bytes = shutil.disk_usage(str(output.anchor or "/"))
    free_gb = free_bytes / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        print(
            f"error: only {free_gb:.1f} GB free on {output.anchor!r}. "
            f"Need at least {MIN_FREE_DISK_GB} GB of scratch space for a full-resolution run.",
            file=sys.stderr,
        )
        return 1

    totalsegmentator(
        input=Path(args.input),
        output=output,
        task=args.task,
        ml=args.ml,
        device=args.device,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    return 0


def cmd_list_structures(args):
    print(format_classes_table("craniofacial_structures"))
    print()
    print("Planned for later phases (not yet segmentable in this build):")
    for t in ROADMAP_TASKS:
        print(f"  - {t}")
    return 0


def cmd_check(args):
    print("AutoAnatomy system check")
    print("-" * 40)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
        print(f"torch:          {torch.__version__}")
        print(f"CUDA available: {cuda_ok}")
        print(f"GPU:            {device_name}")
    except ImportError:
        print("torch:          NOT INSTALLED")

    import shutil
    from autoanatomy.engine.config import get_totalseg_dir, get_weights_dir

    home_dir = get_totalseg_dir()
    weights_dir = get_weights_dir()
    print(f"home dir:       {home_dir}")
    print(f"weights dir:    {weights_dir}")

    craniofacial_cached = (weights_dir / "Dataset115_mandible").exists()
    crop_model_cached = (weights_dir / "Dataset298_TotalSegmentator_total_6mm_1559subj").exists()
    print(f"weights cached: craniofacial_structures={craniofacial_cached}  crop-model={crop_model_cached}")

    total, used, free = shutil.disk_usage(home_dir.anchor or "/")
    print(f"disk free:      {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")
    return 0


def cmd_download_weights(args):
    from autoanatomy.engine.weights import download_pretrained_weights

    print("Downloading craniofacial_structures model (task 115)...")
    download_pretrained_weights(115)
    print("Downloading rough skull-cropping model (task 298)...")
    download_pretrained_weights(298)
    print("Done.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="autoanatomy",
        description="Segment craniofacial structures (mandible, teeth, skull, head, sinuses) in a CT scan.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_seg = sub.add_parser(
        "segment",
        help="Run craniofacial segmentation on a CT scan",
        description="Argument names match upstream TotalSegmentator's CLI (-i/-o/-ta) so existing "
                     "commands drop in unchanged, e.g.: "
                     "autoanatomy segment -i t.nii.gz -o out\\ -ta craniofacial_structures",
    )
    p_seg.add_argument("-i", "--input", required=True, help="CT NIfTI file or DICOM folder")
    p_seg.add_argument("-o", "--output", required=True, help="Output directory (or file path if --ml)")
    p_seg.add_argument("-ta", "--task", default="craniofacial_structures", choices=TASKS,
                        help="Segmentation task (default: craniofacial_structures -- the only task this build supports)")
    p_seg.add_argument("--ml", action="store_true", help="Write a single multilabel NIfTI instead of one file per structure")
    p_seg.add_argument("--device", default="gpu", help="gpu | cpu | mps | gpu:X (default: gpu)")
    p_seg.add_argument("-q", "--quiet", action="store_true")
    p_seg.add_argument("-v", "--verbose", action="store_true")
    p_seg.set_defaults(func=cmd_segment)

    p_list = sub.add_parser("list-structures", help="List the structures this build can segment")
    p_list.set_defaults(func=cmd_list_structures)

    p_check = sub.add_parser("check", help="Report GPU/CUDA, weight cache, and disk status")
    p_check.set_defaults(func=cmd_check)

    p_dl = sub.add_parser("download-weights", help="Download the craniofacial_structures and crop models")
    p_dl.set_defaults(func=cmd_download_weights)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
