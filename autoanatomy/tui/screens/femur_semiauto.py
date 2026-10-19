import sys
import threading
import traceback
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, RichLog, Static

from autoanatomy.tui.screens.run_progress import _redirect_stdout, _StreamToLog
from autoanatomy.tui.widgets.slice_viewer import (
    ANNOTATION_BACKGROUND, ANNOTATION_FOREGROUND, FEMUR_RESULT_DISPLAY_ID, SliceViewer,
)

# FemurSegmentation's own docs recommend labeling roughly this many slices
# for a reliable graph-cut solve -- shown as guidance, not enforced (the
# engine's only hard requirement is >=1 foreground + >=1 background voxel).
RECOMMENDED_MIN_SLICES = 6


class FemurPaintScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back (does not cancel a running segmentation)")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(classes="panel", id="femur-controls"):
                yield Static("Femur semi-automatic segmentation (graph-cut)", classes="section-title")
                yield Static(f"Scan: [b]{self.app.scan_path}[/b]")
                yield Static(
                    "Paint foreground (femur) and background seeds on a handful of slices, then run.\n"
                    "click+drag: paint   (f)/(b): switch label   (c): clear this slice\n"
                    "([)/(]) : scrub slices   (tab): cycle axis",
                    classes="hint",
                )
                yield Static(
                    "[dim]Note: this task's graph-cut solver (maxflow-v3.01) is licensed for "
                    "research purposes only -- see autoanatomy/third_party/femur_segmentation/README.md "
                    "before using this outside personal/research use.[/dim]"
                )
                yield Static("", id="annotation-status")
                with Horizontal():
                    yield Button("Run Segmentation", id="run-btn", variant="primary")
                    yield Button("Clear all seeds", id="clear-all-btn")
                yield Button("Back to Home", id="home-btn")
                yield RichLog(id="femur-log", wrap=True, highlight=False)
            yield SliceViewer(id="slice-viewer")
        yield Footer()

    def on_mount(self) -> None:
        self._scan_array = None
        self._scan_affine = None
        self._scan_header = None
        self._segmentation_running = False
        self._load_scan()
        self.set_interval(0.3, self._update_status)

    def log_line(self, line: str) -> None:
        self.query_one("#femur-log", RichLog).write(line)

    def _load_scan(self) -> None:
        scan_path = self.app.scan_path
        if scan_path is None:
            self.log_line("[red]No scan loaded -- go back and load one first.[/red]")
            return
        if scan_path.is_dir():
            self.log_line(
                "[red]This screen needs a NIfTI file (.nii/.nii.gz), not a DICOM folder -- "
                "convert it first, then load the converted file.[/red]"
            )
            return
        try:
            import nibabel as nib
            import numpy as np

            img = nib.load(str(scan_path))
            volume = np.asanyarray(img.dataobj).astype(np.float32)
        except Exception as e:
            self.log_line(f"[red]Failed to read scan: {e}[/red]")
            return

        self._scan_array = volume
        self._scan_affine = img.affine
        self._scan_header = img.header

        sv = self.query_one("#slice-viewer", SliceViewer)
        sv.set_volume(volume)
        sv.enable_annotation_mode()
        sv.focus()

    def _update_status(self) -> None:
        sv = self.query_one("#slice-viewer", SliceViewer)
        if sv.annotation_mask is None:
            return
        n = sv.annotated_slice_count()
        status = self.query_one("#annotation-status", Static)
        if n == 0:
            status.update("No seeds painted yet.")
        elif n < RECOMMENDED_MIN_SLICES:
            status.update(f"{n} slice(s) annotated -- {RECOMMENDED_MIN_SLICES}+ recommended for a reliable result.")
        else:
            status.update(f"[green]{n} slice(s) annotated.[/green]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "home-btn":
            from autoanatomy.tui.screens.home import HomeScreen
            self.app.switch_screen(HomeScreen())
        elif event.button.id == "clear-all-btn":
            sv = self.query_one("#slice-viewer", SliceViewer)
            if sv.volume is not None:
                sv.enable_annotation_mode()
        elif event.button.id == "run-btn":
            self._start_segmentation()

    def _start_segmentation(self) -> None:
        if self._segmentation_running:
            return
        if self._scan_array is None:
            self.log_line("[red]No scan loaded.[/red]")
            return
        sv = self.query_one("#slice-viewer", SliceViewer)
        if sv.annotation_mask is None:
            self.log_line("[red]No seeds painted.[/red]")
            return
        import numpy as np
        if not np.any(sv.annotation_mask == ANNOTATION_FOREGROUND) or not np.any(
            sv.annotation_mask == ANNOTATION_BACKGROUND
        ):
            self.log_line(
                "[red]Paint at least one foreground (f) and one background (b) seed before running.[/red]"
            )
            return
        self._segmentation_running = True
        self.query_one("#run-btn", Button).disabled = True
        self.run_segmentation(sv.annotation_mask.copy())

    @work(thread=True)
    def run_segmentation(self, annotation_mask) -> None:
        from autoanatomy.engine.femur_semiauto import run_femur_graphcut

        app = self.app
        output_dir = app.output_dir if getattr(app, "output_dir", None) else Path.home() / "AutoAnatomy_output"
        output_path = Path(output_dir) / "femur_graphcut_result.nii.gz"

        stream = _StreamToLog(self, threading.current_thread(), sys.stdout)
        try:
            with _redirect_stdout(stream):
                result = run_femur_graphcut(
                    app.scan_path, annotation_mask, self._scan_affine, self._scan_header, output_path,
                )
            app.call_from_thread(self._on_success, result, output_path)
        except Exception:
            tb = traceback.format_exc()
            app.call_from_thread(self._on_failure, tb)

    def _on_success(self, result, output_path: Path) -> None:
        self._segmentation_running = False
        self.query_one("#run-btn", Button).disabled = False
        self.log_line(f"[green]Done.[/green] Saved to {output_path}")

        sv = self.query_one("#slice-viewer", SliceViewer)
        sv.disable_annotation_mode()
        display_mask = (result > 0).astype(result.dtype) * FEMUR_RESULT_DISPLAY_ID
        sv.set_volume(self._scan_array, display_mask)

    def _on_failure(self, traceback_text: str) -> None:
        self._segmentation_running = False
        self.query_one("#run-btn", Button).disabled = False
        self.log_line("[red]Segmentation failed:[/red]")
        self.log_line(traceback_text)
