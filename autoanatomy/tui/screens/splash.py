from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

from autoanatomy.tui.widgets.system_status import SystemStatus

BANNER = r"""
   _         _        _                _
  /_\  _   _| |_ ___  /_\  _ __   __ _| |_ ___  _ __ ___  _   _
 //_\\| | | | __/ _ \//_\\| '_ \ / _` | __/ _ \| '_ ` _ \| | | |
/  _  \ |_| | || (_) /  _  \ | | | (_| | || (_) | | | | | | |_| |
\_/ \_/\__,_|\__\___/\_/ \_/_| |_|\__,_|\__\___/|_| |_| |_|\__, |
                                                             |___/
"""


class SplashScreen(Screen):
    BINDINGS = [("enter", "continue_", "Continue"), ("space", "continue_", "Continue")]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(classes="panel"):
                yield Static(BANNER, classes="title-banner")
                yield Static(
                    "Craniofacial CT segmentation — mandible · teeth · skull · "
                    "mandibular canal · chewing + neck muscles",
                    classes="subtitle",
                )
                yield SystemStatus()
                yield Static(
                    "\n[b]Real engine[/b]: this build downloads and runs the actual nnU-Net "
                    "DentalSegmentator (task 112), head_muscles (task 777), and headneck_muscles "
                    "(tasks 778+779, license required for commercial use) models on real "
                    "GPU/CPU hardware. Auto3DSeg adds masseter, temporalis and medial "
                    "pterygoid segmentation using the MONAI whole-head model.\n",
                    classes="hint",
                )
                yield Static("[dim]Press ENTER or SPACE to continue...[/dim]")
        yield Footer()

    def action_continue_(self) -> None:
        from autoanatomy.tui.screens.home import HomeScreen
        self.app.switch_screen(HomeScreen())
