from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

ROADMAP = [
    ("teeth", "77 classes — individual teeth (FDI notation), pulp, canals, implants"),
    ("head_glands_cavities", "eyes, optic nerves, parotid/submandibular glands, nasal cavity"),
    ("head_muscles", "masseter, temporalis, pterygoids, tongue"),
    ("headneck_bones_vessels", "larynx, hyoid, zygomatic arch, carotid/jugular vessels"),
]


class HomeScreen(Screen):
    BINDINGS = [("n", "new_run", "New Segmentation"), ("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="panel"):
            yield Static("AutoAnatomy", classes="title-banner")
            yield Static("What do you want to do?", classes="section-title")
            with Horizontal(id="home-actions"):
                yield Button("New Segmentation  (n)", id="new-run", variant="primary")
                yield Button("Settings", id="settings")
                yield Button("Quit", id="quit", variant="error")

        with Vertical(classes="panel"):
            yield Static("Roadmap — other TotalSegmentator tasks (coming soon)", classes="section-title")
            with Horizontal():
                for name, desc in ROADMAP:
                    with Vertical(classes="roadmap-card"):
                        yield Static(name, classes="roadmap-card-title")
                        yield Static(desc)
                        yield Static("[Phase 2]", classes="phase2-badge")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-run":
            self.action_new_run()
        elif event.button.id == "settings":
            from autoanatomy.tui.screens.settings import SettingsScreen
            self.app.push_screen(SettingsScreen())
        elif event.button.id == "quit":
            self.app.exit()

    def action_new_run(self) -> None:
        from autoanatomy.tui.screens.load_scan import LoadScanScreen
        self.app.push_screen(LoadScanScreen())
