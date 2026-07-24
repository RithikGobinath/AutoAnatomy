import io
import shutil
import sys
import threading
import traceback
from contextlib import contextmanager

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, Static

# Full-resolution real CT scans can need several GB of scratch space for
# resampled/cropped intermediates (a 0.5mm isotropic crop of a real head CT
# is easily 500MB-2GB by itself, times several pipeline stages). Below this
# threshold, nnU-Net's background export workers can hang or crash outright
# on a failed write in a way our try/except can't catch (it runs in a
# separate process) -- so refuse to start rather than risk that.
MIN_FREE_DISK_GB = 5


class _StreamToLog(io.TextIOBase):
    """Redirects the engine's real print() progress output into the RichLog widget.

    sys.stdout is process-global, not thread-local, so redirecting it for the
    worker thread also affects the main thread for as long as the redirect is
    active. If anything on the main thread prints during that window, route it
    straight to the real stdout instead of call_from_thread -- that call is
    only valid when invoked from a thread other than the app's own.
    """

    def __init__(self, screen: "RunProgressScreen", worker_thread: threading.Thread, real_stdout):
        self.screen = screen
        self.worker_thread = worker_thread
        self.real_stdout = real_stdout
        self._buf = ""

    def write(self, text: str) -> int:
        if threading.current_thread() is not self.worker_thread:
            return self.real_stdout.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.screen.app.call_from_thread(self.screen.log_line, line)
        return len(text)

    def flush(self) -> None:
        pass


def _ml_output_path(output_dir, task: str, multi_task: bool):
    """A single task keeps the exact literal path the user gave (zero behavior
    change for existing single-task scripts). With more than one task selected,
    suffix the filename per task so the second task's --ml run can't silently
    overwrite the first's multilabel file."""
    if not multi_task:
        return output_dir
    return output_dir.with_name(f"{output_dir.stem.removesuffix('.nii')}_{task}.nii.gz")


class RunProgressScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back (does not cancel run)")]

    def compose(self) -> ComposeResult:
        tasks = list(self.app.selected_tasks)
        title = " + ".join(tasks)
        yield Header()
        with Vertical(classes="panel"):
            yield Static(f"Running {title} segmentation...", classes="section-title", id="status-title")
            yield Static(
                "First run downloads real model weights for each selected task plus the "
                "skull-cropping model — this can take a while.",
                classes="hint",
            )
            yield RichLog(id="progress-log", wrap=True, highlight=False)
            yield Button("Back to Home", id="back-btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_segmentation()

    def log_line(self, line: str) -> None:
        self.query_one("#progress-log", RichLog).write(line)

    @work(thread=True)
    def run_segmentation(self) -> None:
        from autoanatomy.engine.api import segment

        app = self.app
        tasks = list(app.selected_tasks)
        multi_task = len(tasks) > 1

        _, _, free_bytes = shutil.disk_usage(str(app.output_dir.anchor or "/"))
        free_gb = free_bytes / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            app.call_from_thread(
                self._on_failure,
                f"Refusing to start: only {free_gb:.1f} GB free on "
                f"{app.output_dir.anchor}. Need at least {MIN_FREE_DISK_GB} GB "
                "of scratch space for a full-resolution run. Free up space and try again.",
            )
            return

        stream = _StreamToLog(self, threading.current_thread(), sys.stdout)
        try:
            with _redirect_stdout(stream):
                for i, task in enumerate(tasks):
                    if multi_task:
                        app.call_from_thread(self.log_line, f"=== Running {i + 1}/{len(tasks)}: {task} ===")
                    output = _ml_output_path(app.output_dir, task, multi_task) if app.ml else app.output_dir
                    segment(
                        input=app.scan_path,
                        output=output,
                        task=task,
                        roi_subset=app.selected_tasks[task],
                        ml=app.ml,
                        device=app.device,
                        quiet=False,
                        verbose=True,
                        statistics=app.statistics,
                        remove_small_blobs=app.remove_small_blobs,
                        nr_thr_resamp=app.nr_thr_resamp,
                        nr_thr_saving=app.nr_thr_saving,
                        resampling_order=app.resampling_order,
                    )
            if app.statistics:
                # Only meaningful for the last task run when --ml is used across
                # multiple tasks (each writes its own statistics.json into the
                # shared non-ml output dir, so the last one wins either way).
                stats_dir = app.output_dir.parent if app.ml else app.output_dir
                stats_path = stats_dir / "statistics.json"
                app.statistics_path = stats_path if stats_path.exists() else None
            self._collect_results()
            app.call_from_thread(self._on_success)
        except Exception:
            tb = traceback.format_exc()
            app.call_from_thread(self._on_failure, tb)

    def _collect_results(self) -> None:
        import nibabel as nib
        import numpy as np

        from autoanatomy.engine.class_map import class_map

        app = self.app
        tasks = list(app.selected_tasks)
        multi_task = len(tasks) > 1
        voxel_counts = {}
        volumes_mm3 = {}

        for task in tasks:
            cmap = class_map[task]
            roi_subset = app.selected_tasks[task]
            output = _ml_output_path(app.output_dir, task, multi_task) if app.ml else app.output_dir

            if app.ml:
                if not output.exists():
                    continue
                img = nib.load(str(output))
                data = np.asanyarray(img.dataobj)
                voxel_vol = float(np.prod(img.header.get_zooms()))
                for label_id, name in cmap.items():
                    if roi_subset is not None and name not in roi_subset:
                        continue
                    voxels = int((data == label_id).sum())
                    voxel_counts[name] = voxels
                    volumes_mm3[name] = voxels * voxel_vol
            else:
                for label_id, name in cmap.items():
                    mask_path = output / f"{name}.nii.gz"
                    if not mask_path.exists():
                        continue
                    img = nib.load(str(mask_path))
                    data = np.asanyarray(img.dataobj)
                    voxel_vol = float(np.prod(img.header.get_zooms()))
                    voxels = int(data.sum())
                    voxel_counts[name] = voxels
                    volumes_mm3[name] = voxels * voxel_vol

        app.result_voxel_counts = voxel_counts
        app.result_volumes_mm3 = volumes_mm3

    def _on_success(self) -> None:
        self.query_one("#status-title", Static).update("[green]Segmentation complete.[/green]")
        self.query_one("#back-btn", Button).disabled = False
        from autoanatomy.tui.screens.results import ResultsScreen
        self.app.switch_screen(ResultsScreen())

    def _on_failure(self, traceback_text: str) -> None:
        self.query_one("#status-title", Static).update("[red]Segmentation failed.[/red]")
        self.log_line(traceback_text)
        self.query_one("#back-btn", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            from autoanatomy.tui.screens.home import HomeScreen
            self.app.switch_screen(HomeScreen())


@contextmanager
def _redirect_stdout(stream):
    old = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old
