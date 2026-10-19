from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

ROADMAP = [
    ("teeth", "77 classes — individual teeth (FDI notation), pulp, canals, implants"),
    ("head_glands_cavities", "eyes, optic nerves, parotid/submandibular glands, nasal cavity"),
    ("headneck_bones_vessels", "larynx, hyoid, zygomatic arch, carotid/jugular vessels"),
]


class HomeScreen(Screen):
    BINDINGS = [
        ("n", "new_run", "New Segmentation"),
        ("f", "femur_run", "Femur (semi-auto)"),
        ("a", "auto3dseg_run", "Auto3DSeg"),
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="scroll-wrapper"):
            with Vertical(classes="panel"):
                yield Static("AutoAnatomy", classes="title-banner")
                yield Static("What do you want to do?", classes="section-title")
                with Horizontal(id="home-actions"):
                    yield Button("New Segmentation  (n)", id="new-run", variant="primary")
                    yield Button("Femur (semi-auto)  (f)", id="femur-run")
                    yield Button("Settings", id="settings")
                    yield Button("Quit", id="quit", variant="error")

            with Vertical(classes="panel", id="auto3dseg-section"):
                yield Static("Auto3DSeg", classes="section-title")
                yield Static("Masseter, temporalis and medial pterygoid")
                yield Static("Each muscle mask includes both left and right sides.", classes="hint")
                yield Button("Muscle segmentation  (a)", id="auto3dseg-run", variant="primary")

            with Vertical(classes="panel"):
                yield Static("Roadmap — additional craniofacial tasks (coming soon)", classes="section-title")
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
        elif event.button.id == "femur-run":
            self.action_femur_run()
        elif event.button.id == "auto3dseg-run":
            self.action_auto3dseg_run()
        elif event.button.id == "settings":
            from autoanatomy.tui.screens.settings import SettingsScreen
            self.app.push_screen(SettingsScreen())
        elif event.button.id == "quit":
            self.app.exit()

    def action_new_run(self) -> None:
        from autoanatomy.tui.screens.load_scan import LoadScanScreen
        self.app.push_screen(LoadScanScreen())

    def action_femur_run(self) -> None:
        from autoanatomy.tui.screens.femur_semiauto import FemurPaintScreen
        from autoanatomy.tui.screens.load_scan import LoadScanScreen
        self.app.push_screen(LoadScanScreen(next_screen_factory=FemurPaintScreen))

    def action_auto3dseg_run(self) -> None:
        from autoanatomy.tui.screens.configure_run import ConfigureRunScreen
        from autoanatomy.tui.screens.load_scan import LoadScanScreen
        self.app.push_screen(LoadScanScreen(
            next_screen_factory=lambda: ConfigureRunScreen(auto3dseg_only=True),
        ))
