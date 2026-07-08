from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, RadioButton, RadioSet, Static

from autoanatomy.engine.class_map import class_map


class ConfigureRunScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="panel"):
            yield Static("Configure segmentation", classes="section-title")
            yield Static(f"Scan: [b]{self.app.scan_path}[/b]")

            yield Static("\nStructures that will be segmented (fixed for task craniofacial_structures):")
            structures = ", ".join(class_map["craniofacial_structures"].values())
            yield Static(f"  {structures}", classes="hint")

            yield Static("\nDevice:")
            with RadioSet(id="device-set"):
                yield RadioButton("gpu (recommended if available)", value=True, id="dev-gpu")
                yield RadioButton("cpu", id="dev-cpu")

            yield Static("\nOutput directory:")
            default_out = str(Path.home() / "AutoAnatomy_output")
            yield Input(value=default_out, id="output-input")

            yield Static(
                "\n[dim]0.5mm isotropic resolution, fixed by the model. --fast is not "
                "supported for this task (matches upstream TotalSegmentator).[/dim]"
            )

            with Vertical():
                yield Button("Run Segmentation", id="run-btn", variant="primary")
                yield Button("Back", id="back-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "run-btn":
            self._start_run()

    def _start_run(self) -> None:
        device_set = self.query_one("#device-set", RadioSet)
        pressed = device_set.pressed_button
        self.app.device = "cpu" if pressed and pressed.id == "dev-cpu" else "gpu"

        out_raw = self.query_one("#output-input", Input).value.strip().strip('"')
        self.app.output_dir = Path(out_raw) if out_raw else Path.home() / "AutoAnatomy_output"

        from autoanatomy.tui.screens.run_progress import RunProgressScreen
        self.app.push_screen(RunProgressScreen())
