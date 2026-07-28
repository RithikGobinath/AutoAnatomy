import numpy as np
from rich.style import Style
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

# Keyed by *display* ID (registry.TASK_DISPLAY_OFFSET + a task's native label
# ID), not the raw label ID written into the actual NIfTI files. Multiple
# tasks can be shown combined in one overlay/table, so each task gets its own
# offset range to keep colors from colliding between unrelated structures
# that happen to share a raw ID. Applied uniformly even for a single task, so
# display never depends on how many tasks are selected this run.
# head_muscles: offset 100 -> IDs 101-111. dental_segmentator: offset 200 ->
# IDs 201-205. headneck_muscles: offset 300 -> IDs 301-323 (generated below,
# too many structures to hand-pick distinct colors for). Extend this and
# TASK_DISPLAY_OFFSET if a task with more labels ships.
LABEL_COLORS = {
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
    # dental_segmentator: IDs 201 (upper_skull), 202 (mandible),
    # 203 (upper_teeth), 204 (lower_teeth), 205 (mandibular_canal).
    201: (0, 191, 255),    # deep sky blue
    202: (255, 105, 180),  # hot pink
    203: (255, 165, 0),    # orange
    204: (50, 205, 50),    # lime green
    205: (154, 205, 50),   # yellow green
}


def _headneck_muscles_colors():
    """headneck_muscles has 23 structures (display IDs 301-323) -- too many
    to hand-pick distinct colors for, so spread them evenly around the hue
    wheel instead. Any reasonably distinct set is fine here since these
    structures aren't colored to convey clinical meaning, just "which one"."""
    import colorsys
    colors = {}
    for i in range(23):
        hue = i / 23
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors[301 + i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


LABEL_COLORS.update(_headneck_muscles_colors())

HU_WINDOW_MIN = -200
HU_WINDOW_MAX = 1200

_HALF_BLOCK = "▀"  # upper half block -- foreground = top pixel, background = bottom pixel


def _block_reduce(img: np.ndarray, mask, out_h: int, out_w: int):
    """Downsample `img` (grayscale, area-averaged per output cell) and `mask`
    (categorical, majority vote among non-background labels present in each
    cell) from native resolution to an out_h x out_w grid.

    Unlike np.linspace-based point sampling (the previous approach), every
    source pixel contributes to some output cell -- a thin one-voxel-wide
    structure can no longer disappear just because it didn't land on a
    sampled index, and the grayscale background is properly anti-aliased
    instead of looking speckled at low output resolution.
    """
    h, w = img.shape
    out_h = max(1, min(out_h, h))
    out_w = max(1, min(out_w, w))
    row_edges = np.linspace(0, h, out_h + 1).astype(int)
    col_edges = np.linspace(0, w, out_w + 1).astype(int)

    img_out = np.zeros((out_h, out_w), dtype=np.float64)
    mask_out = np.zeros((out_h, out_w), dtype=mask.dtype) if mask is not None else None

    for i in range(out_h):
        r0, r1 = row_edges[i], max(row_edges[i] + 1, row_edges[i + 1])
        for j in range(out_w):
            c0, c1 = col_edges[j], max(col_edges[j] + 1, col_edges[j + 1])
            img_out[i, j] = img[r0:r1, c0:c1].mean()
            if mask is not None:
                mblock = mask[r0:r1, c0:c1]
                nonzero = mblock[mblock != 0]
                if nonzero.size:
                    vals, counts = np.unique(nonzero, return_counts=True)
                    mask_out[i, j] = vals[np.argmax(counts)]
    return img_out, mask_out


def _pixel_rgb(gray_value, label) -> tuple:
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    g = int(gray_value)
    return (g, g, g)


class SliceViewer(Static):
    """Renders a real axial/coronal/sagittal slice of a NIfTI volume (+ optional
    label mask overlay) as truecolor terminal blocks. Not a placeholder -- this
    reads actual numpy arrays produced by the segmentation engine.

    Uses half-block (▀) rendering: each character cell shows two
    independent colors (foreground = top image pixel, background = bottom),
    roughly doubling vertical resolution over a naive one-color-per-cell
    approach. Combined with dropping the old "2 characters per pixel" width
    padding (a half-cell is already close to square in most monospace
    fonts), this fits roughly 4x as much real image detail in the same
    terminal area.
    """

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

        width = max(20, self.size.width - 2)
        height_chars = max(10, self.size.height - 2)
        # Each character row renders two image rows via half-block glyphs, so
        # sample twice as many image rows as there are character rows.
        height = height_chars * 2

        gray_full, mask_ds = _block_reduce(img, mask, height, width)
        clipped = np.clip(gray_full, HU_WINDOW_MIN, HU_WINDOW_MAX)
        gray = ((clipped - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN) * 255).astype(np.uint8)

        n_rows = gray.shape[0]
        if n_rows % 2 == 1:
            # Odd row count (possible when the source image has fewer rows
            # than 2*height_chars) -- duplicate the last row so every
            # character row still has a full top+bottom pair to draw.
            gray = np.vstack([gray, gray[-1:]])
            if mask_ds is not None:
                mask_ds = np.vstack([mask_ds, mask_ds[-1:]])
            n_rows += 1

        text = Text()
        for r in range(0, n_rows, 2):
            for c in range(gray.shape[1]):
                top_label = int(mask_ds[r, c]) if mask_ds is not None else 0
                bot_label = int(mask_ds[r + 1, c]) if mask_ds is not None else 0
                top = _pixel_rgb(gray[r, c], top_label)
                bot = _pixel_rgb(gray[r + 1, c], bot_label)
                style = Style(color=f"rgb({top[0]},{top[1]},{top[2]})",
                             bgcolor=f"rgb({bot[0]},{bot[1]},{bot[2]})")
                text.append(_HALF_BLOCK, style=style)
            text.append("\n")

        axis_name = ["sagittal", "coronal", "axial"][self.axis]
        text.append(f"\n{axis_name}  slice {self.slice_idx}/{self.volume.shape[self.axis] - 1}"
                     f"   ([/][ ]/[)] scrub  (tab) cycle axis", style="dim")
        return text
