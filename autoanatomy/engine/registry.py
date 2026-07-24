"""
Machine-readable registry of AutoAnatomy's segmentation tasks.

This module is the single source of truth for which segmentation tasks exist,
which anatomical classes each one outputs, whether it requires a license and
whether it operates on CT or MR images.

It intentionally only depends on the pure-data maps in ``class_map`` (no
torch, no model weights), so it can be imported and queried instantly. This
powers the ``list-structures`` and ``check`` subcommands of the main CLI,
letting humans and automation discover the tool's capabilities without
reading the source code.
"""
import importlib.metadata

from autoanatomy.engine.class_map import class_map, commercial_models


# Selectable tasks, in the order they are offered on the command line.
# The "total" entry stays in class_map.py only as an internal dependency of
# the skull-cropping step (see engine/api.py) -- it is deliberately NOT
# selectable here, since it's infrastructure, not a product task.
TASKS = [
    "craniofacial_structures",
    "head_muscles",
    "dental_segmentator",
    "toothseg",
]

# Other segmentation tasks planned for later phases. Not yet selectable --
# surfaced in the TUI's roadmap panel as "coming soon". ("teeth" is no longer
# listed here -- the "toothseg" task above now covers individual-tooth
# segmentation, via a different published model than upstream's "teeth" task.)
ROADMAP_TASKS = [
    "head_glands_cavities", "headneck_bones_vessels",
]

# Every task's class map uses small (or, for toothseg, FDI-numbered) integer
# label IDs that overlap across tasks. Structure *names* never collide, so
# results dicts keyed by name are always safe to merge across tasks -- but
# the slice-viewer overlay and the results table's color swatches are keyed
# by label ID, so combining more than one task's raw IDs would make unrelated
# structures share a color. This offset exists only to disambiguate that
# combined *display*; it is never written into the actual saved segmentation
# NIfTI files, which always use each task's own native label IDs. Applied
# uniformly (even when only one task is being shown) so display behavior
# never depends on how many tasks happen to be selected in a given run.
TASK_DISPLAY_OFFSET = {
    "craniofacial_structures": 0,
    "head_muscles": 100,
    "dental_segmentator": 200,
    # toothseg's own label IDs are already the two-digit FDI numbers (11-48),
    # so no offset is needed to keep them clear of the ranges above.
    "toothseg": 0,
}

# Tasks that operate on MR images but whose name does not end in "_mr".
_MR_TASKS_WITHOUT_SUFFIX = set()  # none in this build


def package_version():
    """Installed AutoAnatomy version, or None if not installed (e.g. run from source)."""
    try:
        return importlib.metadata.version("autoanatomy")
    except importlib.metadata.PackageNotFoundError:
        return None


def task_modality(task):
    """Return "MR" or "CT" for a task name."""
    if task.endswith("_mr") or task in _MR_TASKS_WITHOUT_SUFFIX:
        return "MR"
    return "CT"


def requires_license(task):
    """Whether the task needs a license (free for non-commercial use)."""
    return task in commercial_models


def get_task_classes(task):
    """Return the {label_index: class_name} map a task outputs.

    Raises KeyError for an unknown task.
    """
    try:
        return dict(class_map[task])
    except KeyError:
        raise KeyError(f"Unknown task: {task!r}. Valid tasks: {', '.join(TASKS)}")


def list_tasks():
    """Summary of every selectable task: name, modality, license flag, #classes."""
    return [
        {
            "name": t,
            "modality": task_modality(t),
            "license_required": requires_license(t),
            "num_classes": len(get_task_classes(t)),
        }
        for t in TASKS
    ]


def task_registry():
    """Full machine-readable capability map for all selectable tasks (JSON-serializable)."""
    return {
        "autoanatomy_version": package_version(),
        "tasks": {
            t: {
                "modality": task_modality(t),
                "license_required": requires_license(t),
                "classes": {str(idx): name for idx, name in get_task_classes(t).items()},
            }
            for t in TASKS
        },
    }


def format_tasks_table():
    """Human-readable table of all selectable tasks."""
    rows = list_tasks()
    name_w = max(len("TASK"), max(len(r["name"]) for r in rows))
    header = f"{'TASK'.ljust(name_w)}  MODALITY  LICENSE  CLASSES"
    lines = [header, "-" * len(header)]
    for r in rows:
        lic = "yes" if r["license_required"] else "no"
        lines.append(f"{r['name'].ljust(name_w)}  {r['modality'].ljust(8)}  {lic.ljust(7)}  {r['num_classes']}")
    lines.append("")
    lines.append(f"{len(rows)} task(s). None require a license in this build.")
    return "\n".join(lines)


def format_classes_table(task):
    """Human-readable index->name listing of the classes a task outputs."""
    classes = get_task_classes(task)
    lic = "license required" if requires_license(task) else "open license"
    lines = [f"Task '{task}'  [{task_modality(task)}, {lic}, {len(classes)} classes]", ""]
    for idx in sorted(classes):
        lines.append(f"{str(idx).rjust(4)}  {classes[idx]}")
    return "\n".join(lines)
