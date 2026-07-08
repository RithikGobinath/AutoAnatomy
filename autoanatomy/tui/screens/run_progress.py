import io
import shutil
import sys
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
    """Redirects the engine's real print() progress output into the RichLog widget."""

    def __init__(self, screen: "RunProgressScreen"):
        self.screen = screen
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.screen.app.call_from_thread(self.screen.log_line, line)
        return len(text)

    def flush(self) -> None:
        pass


class RunProgressScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back (does not cancel run)")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="panel"):
            yield Static("Running craniofacial segmentation...", classes="section-title", id="status-title")
            yield Static(
                "First run downloads real model weights (craniofacial_structures + the "
                "skull-cropping model) — this can take a while.",
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
        from autoanatomy.engine.api import totalsegmentator

        app = self.app

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

        stream = _StreamToLog(self)
        try:
            with _redirect_stdout(stream):
                totalsegmentator(
                    input=app.scan_path,
                    output=app.output_dir,
                    task="craniofacial_structures",
                    ml=app.ml,
                    device=app.device,
                    statistics=True,
                    quiet=False,
                    verbose=True,
                )
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
        cmap = class_map["craniofacial_structures"]
        voxel_counts = {}
        volumes_mm3 = {}

        if app.ml:
            # Single multilabel file: app.output_dir is the file path itself.
            if not app.output_dir.exists():
                return
            img = nib.load(str(app.output_dir))
            data = np.asanyarray(img.dataobj)
            voxel_vol = float(np.prod(img.header.get_zooms()))
            for label_id, name in cmap.items():
                voxels = int((data == label_id).sum())
                voxel_counts[name] = voxels
                volumes_mm3[name] = voxels * voxel_vol
        else:
            for label_id, name in cmap.items():
                mask_path = app.output_dir / f"{name}.nii.gz"
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
