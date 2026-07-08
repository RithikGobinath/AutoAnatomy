import shutil

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static


class SystemStatus(Static):
    """Live GPU / CUDA / disk / weight-cache status bar. Reads real state, not decoration."""

    DEFAULT_CSS = """
    SystemStatus {
        height: 3;
        border: round $primary;
        padding: 0 1;
    }
    SystemStatus Horizontal {
        height: 1fr;
        align: left middle;
    }
    SystemStatus .status-item {
        margin-right: 3;
    }
    SystemStatus .ok {
        color: $success;
    }
    SystemStatus .bad {
        color: $error;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(id="gpu-status", classes="status-item")
            yield Static(id="weights-status", classes="status-item")
            yield Static(id="disk-status", classes="status-item")

    def on_mount(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        self._update_gpu()
        self._update_weights()
        self._update_disk()

    def _update_gpu(self) -> None:
        widget = self.query_one("#gpu-status", Static)
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                widget.update(f"[b]GPU[/b] [green]{name}[/green]")
            else:
                widget.update("[b]GPU[/b] [red]not available (CPU fallback)[/red]")
        except ImportError:
            widget.update("[b]GPU[/b] [red]torch not installed[/red]")

    def _update_weights(self) -> None:
        widget = self.query_one("#weights-status", Static)
        try:
            from autoanatomy.engine.config import get_weights_dir
            weights_dir = get_weights_dir()
            craniofacial_ok = (weights_dir / "Dataset115_mandible").exists()
            crop_ok = (weights_dir / "Dataset298_TotalSegmentator_total_6mm_1559subj").exists()
            if craniofacial_ok and crop_ok:
                widget.update("[b]Models[/b] [green]cached[/green]")
            else:
                widget.update("[b]Models[/b] [yellow]not downloaded[/yellow]")
        except Exception:
            widget.update("[b]Models[/b] [red]unknown[/red]")

    def _update_disk(self) -> None:
        widget = self.query_one("#disk-status", Static)
        try:
            _, _, free = shutil.disk_usage("/")
            free_gb = free / 1e9
            color = "red" if free_gb < 5 else ("yellow" if free_gb < 20 else "green")
            widget.update(f"[b]Disk free[/b] [{color}]{free_gb:.1f} GB[/{color}]")
        except Exception:
            widget.update("[b]Disk free[/b] [red]unknown[/red]")
