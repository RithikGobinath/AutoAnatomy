from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from autoanatomy.tui.widgets.system_status import SystemStatus


class SettingsScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="panel scroll-wrapper"):
            yield Static("Settings", classes="section-title")
            yield SystemStatus()

            from autoanatomy.engine.config import get_totalseg_dir, get_weights_dir
            yield Static(
                f"\nHome directory:    {get_totalseg_dir()}\n"
                f"Weights directory: {get_weights_dir()}\n"
                f"Default device:    {self.app.device}\n",
                classes="hint",
            )
            from autoanatomy.engine.registry import TASKS, requires_license
            licensed_tasks = [t for t in TASKS if requires_license(t)]
            if licensed_tasks:
                license_text = (
                    "License / cloud sync: dental_segmentator and head_muscles need no "
                    f"license. {', '.join(licensed_tasks)} requires a paid license for "
                    "commercial use (free for non-commercial use) -- see NOTICE."
                )
            else:
                license_text = (
                    "License / cloud sync: not applicable — every task in this build "
                    "is free for non-commercial use, no license needed."
                )
            yield Static(license_text, classes="hint")
            yield Button("Back", id="back-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
