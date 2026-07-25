from pathlib import Path

from textual.app import App


class AutoAnatomyApp(App):
    """AutoAnatomy - craniofacial CT segmentation, terminal edition.

    Real engine underneath (real nnU-Net inference, real GPU/CPU, real NIfTI
    I/O). Screens for features we haven't built yet are clearly marked
    "Phase 2" rather than faked.
    """

    CSS_PATH = "app.tcss"
    TITLE = "AutoAnatomy"
    SUB_TITLE = "craniofacial segmentation — under construction"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    # Shared run state, read/written by screens as the user moves through the flow.
    scan_path: Path | None = None
    # {task_name: roi_subset_or_None} -- only entries for tasks the user
    # included this run. None means "all structures for that task" (today's
    # no-op-filter default); a list means only those structure names.
    selected_tasks: dict = {}
    device: str = "gpu"
    ml: bool = False
    output_dir: Path | None = None
    result_volumes_mm3: dict = {}
    result_voxel_counts: dict = {}

    # Advanced settings -- all real, thread straight through to engine.api.segment().
    statistics: bool = False
    remove_small_blobs: bool = False
    nr_thr_resamp: int = 1
    nr_thr_saving: int = 6
    resampling_order: int = 3
    # Off by default -- see engine/nnunet_runner.py's low_memory branch.
    # Sending a prediction to the normal background export pool means
    # pickling a full copy of it to hand to a worker process; for a
    # high-resolution volume that copy alone can raise a bare MemoryError on
    # Windows even when the prediction itself succeeded. This runs export
    # synchronously in the same process instead -- slower, but avoids that
    # failure mode entirely rather than just making it less likely.
    low_memory: bool = False
    # Off by default -- running more than one task at once means every
    # task's model is loaded into memory (and, on GPU, VRAM) simultaneously
    # instead of one at a time.
    parallel_tasks: bool = False
    # Caps how many of those run truly simultaneously when parallel_tasks is
    # on (see engine/parallel_runner.py's DEFAULT_MAX_PARALLEL_TASKS) -- an
    # uncapped run is what causes real out-of-memory crashes with several
    # tasks selected, since each one loads its own full model + scan copy.
    max_parallel_tasks: int | None = None
    statistics_path: Path | None = None

    def on_mount(self) -> None:
        from autoanatomy.tui.screens.splash import SplashScreen
        self.push_screen(SplashScreen())


def run():
    AutoAnatomyApp().run()


if __name__ == "__main__":
    run()
