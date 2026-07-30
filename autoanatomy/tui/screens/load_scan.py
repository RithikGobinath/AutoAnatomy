from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static


class LoadScanScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="panel scroll-wrapper"):
            yield Static("Load a CT scan", classes="section-title")
            yield Static(
                "Enter the path to a CT NIfTI file (.nii / .nii.gz) or a folder of DICOM slices.",
                classes="hint",
            )
            yield Input(placeholder=r"C:\path\to\scan.nii.gz", id="path-input")
            yield Static("", id="scan-info")
            with Vertical():
                yield Button("Load", id="load-btn", variant="primary")
                yield Button("Back", id="back-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load-btn":
            self._load()
        elif event.button.id == "back-btn":
            self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._load()

    def _load(self) -> None:
        info = self.query_one("#scan-info", Static)
        raw = self.query_one("#path-input", Input).value.strip().strip('"')
        if not raw:
            info.update("[red]Enter a path first.[/red]")
            return

        path = Path(raw)
        if not path.exists():
            info.update(f"[red]Path does not exist: {path}[/red]")
            return

        try:
            if path.is_dir():
                n_files = len(list(path.glob("*")))
                info.update(
                    f"[green]DICOM folder found[/green] — {n_files} files.\n"
                    "[dim]Will be converted to NIfTI before segmentation.[/dim]"
                )
            else:
                import nibabel as nib
                img = nib.load(str(path))
                shape = img.shape
                zooms = img.header.get_zooms()
                info.update(
                    f"[green]NIfTI loaded[/green]\n"
                    f"shape: {shape}   spacing: {tuple(round(z, 2) for z in zooms)} mm"
                )
        except Exception as e:
            info.update(f"[red]Failed to read scan: {e}[/red]")
            return

        self.app.scan_path = path
        from autoanatomy.tui.screens.configure_run import ConfigureRunScreen
        self.app.push_screen(ConfigureRunScreen())
