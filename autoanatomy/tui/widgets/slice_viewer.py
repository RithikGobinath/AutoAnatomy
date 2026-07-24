import numpy as np
from rich.style import Style
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

# Keyed by *display* ID (registry.TASK_DISPLAY_OFFSET + a task's native label
# ID), not the raw label ID written into the actual NIfTI files. Multiple
# tasks can be shown combined in one overlay/table, so each task gets its own
# offset range to keep colors from colliding between unrelated structures
# that happen to share a raw ID (e.g. craniofacial's "mandible"=1 and
# head_muscles' "masseter_right"=1). Applied uniformly even for a single
# task, so display never depends on how many tasks are selected this run.
# craniofacial_structures: offset 0 -> IDs 1-7. head_muscles: offset 100 ->
# IDs 101-111. Extend both this and TASK_DISPLAY_OFFSET if a task with more
# labels ships.
LABEL_COLORS = {
    1: (230, 25, 75),    # red
    2: (60, 180, 75),    # green
    3: (255, 225, 25),   # yellow
    4: (70, 130, 240),   # blue
    5: (245, 130, 48),   # orange
    6: (145, 30, 180),   # purple
    7: (0, 200, 200),    # cyan
    # Deliberately a different palette from 1-7 above, not a repeat -- these
    # can be shown in the same combined overlay/table as craniofacial_structures.
    101: (255, 182, 193),  # pink
    102: (0, 128, 128),    # teal
    103: (128, 128, 0),    # olive
    104: (240, 50, 230),   # magenta
    105: (210, 245, 60),   # lime
    106: (170, 110, 40),   # brown
    107: (0, 128, 255),    # azure
    108: (128, 0, 0),      # maroon
    109: (170, 255, 195),  # mint
    110: (220, 190, 255),  # lavender
    111: (255, 215, 0),    # gold
    # dental_segmentator: offset 200 -> IDs 201 (upper_skull), 202 (mandible),
    # 205 (mandibular_canal). A third distinct palette.
    201: (0, 191, 255),    # deep sky blue
    202: (255, 105, 180),  # hot pink
    205: (154, 205, 50),   # yellow green
}

# toothseg: offset 0 -> IDs are the real two-digit FDI tooth numbers
# (11-18, 21-28, 31-38, 41-48), so they're generated into LABEL_COLORS
# separately rather than hand-picked -- 32 individually-numbered teeth is too
# many for a hand-picked palette, and any reasonable, distinct set is fine
# here since teeth aren't colored to convey meaning beyond "which one".
def _toothseg_colors():
    import colorsys
    colors = {}
    fdi_numbers = (
        list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(range(41, 49))
    )
    for i, fdi in enumerate(fdi_numbers):
        hue = i / len(fdi_numbers)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors[fdi] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


LABEL_COLORS.update(_toothseg_colors())

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
