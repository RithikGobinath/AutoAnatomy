import re
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, Collapsible, Footer, Header, Input, RadioButton, RadioSet, Static,
)

from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.registry import TASKS

DEFAULT_BLOB_THRESHOLD_MM3 = 200
DEFAULT_RESAMPLE_THREADS = 1
DEFAULT_SAVING_THREADS = 6
DEFAULT_RESAMPLING_ORDER = 3


def _structure_checkbox_id(name: str) -> str:
    return "struct-" + re.sub(r"[^a-zA-Z0-9_-]", "-", name)


class ConfigureRunScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="panel scroll-wrapper"):
            yield Static("Configure segmentation", classes="section-title")
            yield Static(f"Scan: [b]{self.app.scan_path}[/b]")

            yield Static("\nTasks (any combination -- each run independently, results are combined):")
            for i, task in enumerate(TASKS):
                # Preserve today's default: only the first task runs unless you
                # opt in to more -- a second task roughly doubles run time.
                yield Checkbox(f"Include {task}", value=(i == 0), id=f"task-enable-{i}")
                with Vertical(id=f"structures-container-{i}", classes="task-structures"):
                    for widget in self._build_structure_checkboxes(task):
                        yield widget
            yield Static("[dim]Unchecking a structure still runs the full model for its task -- it only "
                         "changes which output files are saved.[/dim]")
            yield Static("", id="structures-error")

            yield Static("\nDevice:")
            with RadioSet(id="device-set"):
                yield RadioButton("gpu (recommended if available)", value=True, id="dev-gpu")
                yield RadioButton("cpu", id="dev-cpu")

            yield Checkbox("Save as a single multilabel NIfTI file (--ml)", id="ml-checkbox")

            yield Static("\nOutput (directory, or a .nii.gz file path if --ml is checked):", id="output-label")
            default_out = str(Path.home() / "AutoAnatomy_output")
            yield Input(value=default_out, id="output-input")

            yield Static(
                "\n[dim]Resampling resolution is fixed per task by the model. --fast is not "
                "supported for either task in this build (fixed model constraint).[/dim]"
            )

            with Collapsible(title="Advanced settings", collapsed=True):
                yield Checkbox("Compute statistics.json (per-structure volume/intensity)", id="statistics-checkbox")

                yield Checkbox("Remove small disconnected blobs (postprocessing cleanup)", id="blobs-checkbox")
                with Horizontal(classes="advanced-row"):
                    yield Static("  Threshold (mm³):", classes="hint")
                    yield Input(value=str(DEFAULT_BLOB_THRESHOLD_MM3), id="blobs-threshold-input", type="number")

                with Horizontal(classes="advanced-row"):
                    yield Static("  Resample threads:", classes="hint")
                    yield Input(value=str(DEFAULT_RESAMPLE_THREADS), id="resample-threads-input", type="integer")
                with Horizontal(classes="advanced-row"):
                    yield Static("  Saving threads:", classes="hint")
                    yield Input(value=str(DEFAULT_SAVING_THREADS), id="saving-threads-input", type="integer")
                with Horizontal(classes="advanced-row"):
                    yield Static("  Resampling order (0-5):", classes="hint")
                    yield Input(value=str(DEFAULT_RESAMPLING_ORDER), id="resampling-order-input", type="integer")

                yield Checkbox(
                    "Run tasks in parallel (faster with multiple tasks selected -- "
                    "uses more RAM/VRAM at once)",
                    id="parallel-tasks-checkbox",
                )

            with Vertical():
                yield Button("Run Segmentation", id="run-btn", variant="primary")
                yield Button("Back", id="back-btn")
        yield Footer()

    def _build_structure_checkboxes(self, task: str) -> list[Checkbox]:
        return [
            Checkbox(name, value=True, id=_structure_checkbox_id(name))
            for name in class_map[task].values()
        ]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "run-btn":
            self._start_run()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "ml-checkbox":
            return
        out_input = self.query_one("#output-input", Input)
        if event.value and not out_input.value.endswith((".nii", ".nii.gz")):
            out_input.value = str(Path.home() / "AutoAnatomy_output" / "output.nii.gz")
        elif not event.value and out_input.value.endswith((".nii", ".nii.gz")):
            out_input.value = str(Path.home() / "AutoAnatomy_output")

    def _num(self, input_id: str, cast, default):
        raw = self.query_one(f"#{input_id}", Input).value.strip()
        try:
            return cast(raw)
        except (ValueError, TypeError):
            return default

    def _start_run(self) -> None:
        error = self.query_one("#structures-error", Static)
        selected_tasks = {}

        for i, task in enumerate(TASKS):
            if not self.query_one(f"#task-enable-{i}", Checkbox).value:
                continue

            checked_names = [
                name for name in class_map[task].values()
                if self.query_one(f"#{_structure_checkbox_id(name)}", Checkbox).value
            ]
            if not checked_names:
                error.update(f"[red]'{task}' is included but has no structures selected -- "
                             "either uncheck the task or select at least one structure.[/red]")
                return

            all_names = list(class_map[task].values())
            selected_tasks[task] = None if len(checked_names) == len(all_names) else checked_names

        if not selected_tasks:
            error.update("[red]Include at least one task.[/red]")
            return

        self.app.selected_tasks = selected_tasks

        device_set = self.query_one("#device-set", RadioSet)
        pressed = device_set.pressed_button
        self.app.device = "cpu" if pressed and pressed.id == "dev-cpu" else "gpu"

        self.app.ml = self.query_one("#ml-checkbox", Checkbox).value

        out_raw = self.query_one("#output-input", Input).value.strip().strip('"')
        self.app.output_dir = Path(out_raw) if out_raw else Path.home() / "AutoAnatomy_output"

        self.app.statistics = self.query_one("#statistics-checkbox", Checkbox).value

        if self.query_one("#blobs-checkbox", Checkbox).value:
            self.app.remove_small_blobs = self._num("blobs-threshold-input", float, DEFAULT_BLOB_THRESHOLD_MM3)
        else:
            self.app.remove_small_blobs = False

        self.app.nr_thr_resamp = self._num("resample-threads-input", int, DEFAULT_RESAMPLE_THREADS)
        self.app.nr_thr_saving = self._num("saving-threads-input", int, DEFAULT_SAVING_THREADS)
        order = self._num("resampling-order-input", int, DEFAULT_RESAMPLING_ORDER)
        self.app.resampling_order = min(5, max(0, order))
        self.app.parallel_tasks = self.query_one("#parallel-tasks-checkbox", Checkbox).value

        from autoanatomy.tui.screens.run_progress import RunProgressScreen
        self.app.push_screen(RunProgressScreen())
