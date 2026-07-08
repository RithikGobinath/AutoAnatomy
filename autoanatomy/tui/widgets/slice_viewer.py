import numpy as np
from rich.style import Style
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

# One color per craniofacial_structures label (1-7). Chosen to be distinguishable
# in a 256/truecolor terminal.
LABEL_COLORS = {
    1: (230, 25, 75),    # mandible
    2: (60, 180, 75),    # teeth_lower
    3: (255, 225, 25),   # skull
    4: (70, 130, 240),   # head
    5: (245, 130, 48),   # sinus_maxillary
    6: (145, 30, 180),   # sinus_frontal
    7: (0, 200, 200),    # teeth_upper
}

HU_WINDOW_MIN = -200
HU_WINDOW_MAX = 1200


class SliceViewer(Static):
    """Renders a real axial/coronal/sagittal slice of a NIfTI volume (+ optional
    label mask overlay) as truecolor terminal blocks. Not a placeholder -- this
    reads actual numpy arrays produced by the segmentation engine."""

    DEFAULT_CSS = """
    SliceViewer {
        width: 1fr;
        height: 1fr;
        border: round $primary;
        content-align: center middle;
    }
    """

    axis = reactive(2)
    slice_idx = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volume: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def set_volume(self, volume: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.volume = volume
        self.mask = mask
        self.axis = 2
        self.slice_idx = volume.shape[2] // 2
        self.refresh()

    def scrub(self, delta: int) -> None:
        if self.volume is None:
            return
        max_idx = self.volume.shape[self.axis] - 1
        self.slice_idx = max(0, min(max_idx, self.slice_idx + delta))

    def cycle_axis(self) -> None:
        if self.volume is None:
            return
        self.axis = (self.axis + 1) % 3
        self.slice_idx = self.volume.shape[self.axis] // 2

    def watch_slice_idx(self, _old: int, _new: int) -> None:
        self.refresh()

    def watch_axis(self, _old: int, _new: int) -> None:
        self.refresh()

    def _take_slice(self, arr: np.ndarray) -> np.ndarray:
        idx = min(self.slice_idx, arr.shape[self.axis] - 1)
        if self.axis == 0:
            return arr[idx, :, :]
        if self.axis == 1:
            return arr[:, idx, :]
        return arr[:, :, idx]

    def render(self) -> Text:
        if self.volume is None:
            return Text("No scan loaded yet.", justify="center")

        img = self._take_slice(self.volume)
        mask = self._take_slice(self.mask) if self.mask is not None else None

        width = max(20, self.size.width - 4)
        height = max(10, self.size.height - 2)

        h, w = img.shape
        row_idx = np.linspace(0, h - 1, min(height, h)).astype(int)
        col_idx = np.linspace(0, w - 1, min(width, w)).astype(int)
        img_ds = img[np.ix_(row_idx, col_idx)]
        mask_ds = mask[np.ix_(row_idx, col_idx)] if mask is not None else None

        clipped = np.clip(img_ds, HU_WINDOW_MIN, HU_WINDOW_MAX)
        gray = ((clipped - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN) * 255).astype(np.uint8)

        text = Text()
        for r in range(gray.shape[0]):
            for c in range(gray.shape[1]):
                label = int(mask_ds[r, c]) if mask_ds is not None else 0
                if label in LABEL_COLORS:
                    color = LABEL_COLORS[label]
                else:
                    g = int(gray[r, c])
                    color = (g, g, g)
                text.append("  ", style=Style(bgcolor=f"rgb({color[0]},{color[1]},{color[2]})"))
            text.append("\n")

        axis_name = ["sagittal", "coronal", "axial"][self.axis]
        text.append(f"\n{axis_name}  slice {self.slice_idx}/{self.volume.shape[self.axis] - 1}"
                     f"   ([/][ ]/[)] scrub  (tab) cycle axis", style="dim")
        return text
