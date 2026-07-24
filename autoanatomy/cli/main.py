import argparse
import shutil
import sys
from pathlib import Path

from autoanatomy.engine.registry import TASKS, ROADMAP_TASKS, format_classes_table, get_task_classes

# See autoanatomy/tui/screens/run_progress.py for why this guard exists:
# nnU-Net's background export workers can hang/crash on a disk-full write in
# a way that's hard to catch cleanly, so refuse to start below this threshold.
MIN_FREE_DISK_GB = 5


def resampling_order(value):
    ivalue = int(value)
    if not 0 <= ivalue <= 5:
        raise argparse.ArgumentTypeError("resampling order must be between 0 and 5")
    return ivalue


def _ml_output_path(output, task, multi_task):
    """A single task keeps the exact literal path given (zero behavior change
    for existing single-task scripts). With more than one task, suffix the
    filename per task so the second task's --ml run can't silently overwrite
    the first's multilabel file. Mirrors tui/screens/run_progress.py."""
    if not multi_task:
        return output
    return output.with_name(f"{output.stem.removesuffix('.nii')}_{task}.nii.gz")


def cmd_segment(args):
    # Imported lazily: this pulls in torch/nnunetv2, which is slow and
    # unnecessary for --list-structures / check / download-weights.
    from autoanatomy.engine.api import segment, validate_device_type_api

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

    tasks = list(dict.fromkeys(args.task))  # de-dupe, preserve order

    # {task: roi_subset_or_None}, only for tasks that will actually run.
    run_plan = {}
    if args.structures:
        requested = [s.strip() for s in args.structures.split(",") if s.strip()]
        unclaimed = set(requested)
        for task in tasks:
            valid = set(get_task_classes(task).values())
            matched = [s for s in requested if s in valid]
            unclaimed -= set(matched)
            if matched:
                run_plan[task] = matched
            else:
                print(f"skipping {task}: none of the requested --structures belong to it", file=sys.stderr)
        if unclaimed:
            print(
                f"error: unknown structure(s) for the selected task(s): {', '.join(sorted(unclaimed))}",
                file=sys.stderr,
            )
            return 1
        if not run_plan:
            print("error: no selected task has any of the requested --structures", file=sys.stderr)
            return 1
    else:
        run_plan = {task: None for task in tasks}

    multi_task = len(run_plan) > 1
    for task, roi_subset in run_plan.items():
        task_output = _ml_output_path(output, task, multi_task) if args.ml else output
        segment(
            input=Path(args.input),
            output=task_output,
            task=task,
            ml=args.ml,
            device=args.device,
            quiet=args.quiet,
            verbose=args.verbose,
            statistics=args.statistics,
            remove_small_blobs=args.remove_small_blobs,
            nr_thr_resamp=args.resample_threads,
            nr_thr_saving=args.saving_threads,
            resampling_order=args.resampling_order,
            roi_subset=roi_subset,
        )
    return 0


def cmd_list_structures(args):
    for i, task in enumerate(TASKS):
        if i > 0:
            print()
        print(format_classes_table(task))
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
    head_muscles_cached = (weights_dir / "Dataset777_head_muscles_492subj").exists()
    crop_model_cached = (weights_dir / "Dataset298_TotalSegmentator_total_6mm_1559subj").exists()
    print(
        f"weights cached: craniofacial_structures={craniofacial_cached}  "
        f"head_muscles={head_muscles_cached}  crop-model={crop_model_cached}"
    )

    total, used, free = shutil.disk_usage(home_dir.anchor or "/")
    print(f"disk free:      {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")
    return 0


def cmd_download_weights(args):
    from autoanatomy.engine.weights import download_pretrained_weights

    print("Downloading craniofacial_structures model (task 115)...")
    download_pretrained_weights(115)
    print("Downloading head_muscles model (task 777)...")
    download_pretrained_weights(777)
    print("Downloading rough skull-cropping model (task 298)...")
    download_pretrained_weights(298)
    print("Done.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="autoanatomy",
        description="Segment craniofacial structures and head muscles in a CT scan.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_seg = sub.add_parser(
        "segment",
        help="Run segmentation on a CT scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Examples:\n"
                     "  autoanatomy segment -i t.nii.gz -o out\\ -ta craniofacial_structures\n"
                     "  autoanatomy segment -i t.nii.gz -o out\\ -ta craniofacial_structures head_muscles",
    )
    p_seg.add_argument("-i", "--input", required=True, help="CT NIfTI file or DICOM folder")
    p_seg.add_argument("-o", "--output", required=True, help="Output directory (or file path if --ml)")
    p_seg.add_argument("-ta", "--task", nargs="+", default=["craniofacial_structures"], choices=TASKS,
                        help="One or more segmentation tasks to run, space-separated (default: "
                             "craniofacial_structures). Run 'autoanatomy list-structures' to see every "
                             "task and the structures it produces.")
    p_seg.add_argument("--structures", metavar="NAME[,NAME...]",
                        help="Only save these structures (comma-separated names, matched against whichever "
                             "selected task(s) they belong to). Default: all structures for every selected task.")
    p_seg.add_argument("--ml", action="store_true", help="Write a single multilabel NIfTI instead of one file per structure")
    p_seg.add_argument("--device", default="gpu", help="gpu | cpu | mps | gpu:X (default: gpu)")
    p_seg.add_argument("-q", "--quiet", action="store_true")
    p_seg.add_argument("-v", "--verbose", action="store_true")

    p_seg.add_argument("--statistics", action="store_true",
                        help="Compute per-structure volume/intensity stats and write statistics.json next to the output")
    p_seg.add_argument("--remove-small-blobs", nargs="?", const=200.0, default=False, type=float, metavar="MM3",
                        help="Postprocessing cleanup: drop disconnected blobs smaller than MM3 (default 200mm^3 if given with no value)")
    p_seg.add_argument("--resample-threads", type=int, default=1, metavar="N",
                        help="Threads used for image resampling (default: 1)")
    p_seg.add_argument("--saving-threads", type=int, default=6, metavar="N",
                        help="Threads used for writing output files (default: 6)")
    p_seg.add_argument("--resampling-order", type=resampling_order, default=3, metavar="0-5",
                        help="Interpolation order for resampling -- higher is smoother but slower (default: 3)")
    p_seg.set_defaults(func=cmd_segment)

    p_list = sub.add_parser("list-structures", help="List the structures this build can segment")
    p_list.set_defaults(func=cmd_list_structures)

    p_check = sub.add_parser("check", help="Report GPU/CUDA, weight cache, and disk status")
    p_check.set_defaults(func=cmd_check)

    p_dl = sub.add_parser("download-weights", help="Download all task models and the shared crop model")
    p_dl.set_defaults(func=cmd_download_weights)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
