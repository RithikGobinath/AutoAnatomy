import numpy as np
from rich.style import Style
from rich.text import Text
from textual import events
from textual.reactive import reactive
from textual.widgets import Static

# Annotation (graph-cut seed) label values -- distinct from the segmentation
# result LABEL_COLORS namespace below (those are keyed by *display id*,
# hundreds+; these are the raw 0/1/2 values painted directly into a seed
# mask, matching RiccardoBiondi/FemurSegmentation's own --init mask
# convention: 0 = unlabeled, 1 = foreground/femur, 2 = background).
ANNOTATION_UNLABELED = 0
ANNOTATION_FOREGROUND = 1
ANNOTATION_BACKGROUND = 2

ANNOTATION_COLORS = {
    ANNOTATION_FOREGROUND: (255, 60, 60),   # bright red seed paint
    ANNOTATION_BACKGROUND: (60, 140, 255),  # bright blue seed paint
}

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


def _compute_edges(h: int, w: int, out_h: int, out_w: int):
    """Bin edges mapping an h x w source image onto an out_h x out_w grid --
    shared by _block_reduce_with_edges (rendering) and SliceViewer's mouse
    handling (so a click maps back to exactly the source region that pixel's
    color was actually averaged from)."""
    out_h = max(1, min(out_h, h))
    out_w = max(1, min(out_w, w))
    row_edges = np.linspace(0, h, out_h + 1).astype(int)
    col_edges = np.linspace(0, w, out_w + 1).astype(int)
    return row_edges, col_edges


def _block_reduce_with_edges(img, mask, row_edges: np.ndarray, col_edges: np.ndarray):
    """Same reduction as _block_reduce, but reusing precomputed bin edges --
    lets multiple categorical layers (segmentation mask, annotation seed
    mask) be reduced onto the exact same grid as the grayscale image."""
    out_h = len(row_edges) - 1
    out_w = len(col_edges) - 1
    img_out = np.zeros((out_h, out_w), dtype=np.float64) if img is not None else None
    mask_out = np.zeros((out_h, out_w), dtype=mask.dtype) if mask is not None else None

    for i in range(out_h):
        r0, r1 = row_edges[i], max(row_edges[i] + 1, row_edges[i + 1])
        for j in range(out_w):
            c0, c1 = col_edges[j], max(col_edges[j] + 1, col_edges[j + 1])
            if img is not None:
                img_out[i, j] = img[r0:r1, c0:c1].mean()
            if mask is not None:
                mblock = mask[r0:r1, c0:c1]
                nonzero = mblock[mblock != 0]
                if nonzero.size:
                    vals, counts = np.unique(nonzero, return_counts=True)
                    mask_out[i, j] = vals[np.argmax(counts)]
    return img_out, mask_out


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
    row_edges, col_edges = _compute_edges(*img.shape, out_h, out_w)
    return _block_reduce_with_edges(img, mask, row_edges, col_edges)


def _pixel_rgb(gray_value, label, annotation_label=ANNOTATION_UNLABELED) -> tuple:
    # Annotation seed paint takes visual priority -- it's the thing the user
    # is actively editing, and must stay visible even over an existing
    # segmentation-result overlay.
    if annotation_label in ANNOTATION_COLORS:
        return ANNOTATION_COLORS[annotation_label]
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

    can_focus = True  # needed to receive key events for f/b/c while annotating

    axis = reactive(2)
    slice_idx = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volume: np.ndarray | None = None
        self.mask: np.ndarray | None = None

        # Annotation (graph-cut seed) state -- see enable_annotation_mode().
        # annotation_mask is the same shape as volume: 0 everywhere except
        # the handful of slices the user has actually painted.
        self.annotation_mask: np.ndarray | None = None
        self.annotation_enabled: bool = False
        self.paint_label: int = ANNOTATION_FOREGROUND
        self._dragging: bool = False

        # Set on every render(); used to map a mouse click's character-cell
        # position back to the exact source-image region that cell's color
        # was reduced from. A terminal mouse event only reports which
        # character cell was clicked -- not a sub-cell pixel position -- so a
        # single click paints the *entire* source region (both the top and
        # bottom image-rows) that character's half-block glyph represents.
        self._row_edges: np.ndarray | None = None
        self._col_edges: np.ndarray | None = None

    def set_volume(self, volume: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.volume = volume
        self.mask = mask
        self.axis = 2
        self.slice_idx = volume.shape[2] // 2
        self.refresh()

    def enable_annotation_mode(self) -> None:
        """Start graph-cut seed painting. Requires set_volume() to have been
        called first. Safe to call again to just reset all painted seeds."""
        if self.volume is None:
            raise RuntimeError("enable_annotation_mode() requires set_volume() first")
        self.annotation_mask = np.zeros(self.volume.shape, dtype=np.uint8)
        self.annotation_enabled = True
        self.refresh()

    def disable_annotation_mode(self) -> None:
        self.annotation_enabled = False
        self.refresh()

    def set_paint_label(self, label: int) -> None:
        if label not in (ANNOTATION_FOREGROUND, ANNOTATION_BACKGROUND):
            raise ValueError(f"paint_label must be ANNOTATION_FOREGROUND or ANNOTATION_BACKGROUND, got {label!r}")
        self.paint_label = label

    def clear_annotation_slice(self) -> None:
        """Erase every seed painted on the currently-displayed slice only."""
        if self.annotation_mask is None:
            return
        idx = min(self.slice_idx, self.annotation_mask.shape[self.axis] - 1)
        if self.axis == 0:
            self.annotation_mask[idx, :, :] = ANNOTATION_UNLABELED
        elif self.axis == 1:
            self.annotation_mask[:, idx, :] = ANNOTATION_UNLABELED
        else:
            self.annotation_mask[:, :, idx] = ANNOTATION_UNLABELED
        self.refresh()

    def annotated_slice_count(self) -> int:
        """How many distinct slices (along the current axis) have at least
        one painted seed voxel -- used to show progress against the ~6-7
        slices FemurSegmentation's own docs recommend labeling."""
        if self.annotation_mask is None:
            return 0
        has_label = self.annotation_mask != ANNOTATION_UNLABELED
        other_axes = tuple(a for a in range(3) if a != self.axis)
        return int(np.count_nonzero(has_label.any(axis=other_axes)))

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

    def _paint_full_res_region(self, r0: int, r1: int, c0: int, c1: int) -> None:
        """Write self.paint_label into the annotation mask's *current* slice,
        over the full-resolution source region [r0:r1, c0:c1] -- the region
        one clicked character cell was reduced from (see _row_edges/_col_edges)."""
        idx = min(self.slice_idx, self.annotation_mask.shape[self.axis] - 1)
        if self.axis == 0:
            self.annotation_mask[idx, r0:r1, c0:c1] = self.paint_label
        elif self.axis == 1:
            self.annotation_mask[r0:r1, idx, c0:c1] = self.paint_label
        else:
            self.annotation_mask[r0:r1, c0:c1, idx] = self.paint_label

    def _paint_at_screen_pos(self, x: int, y: int) -> None:
        """x, y are character-cell coordinates relative to this widget's own
        content region (Textual delivers mouse events already offset this
        way). y indexes character *rows*; each one covers two image rows
        (see _row_edges, built at 2x the character-row count)."""
        if not self.annotation_enabled or self.annotation_mask is None:
            return
        if self._row_edges is None or self._col_edges is None:
            return
        char_row, char_col = y, x
        top_row = char_row * 2
        bot_row = top_row + 1
        if char_col < 0 or char_col + 1 >= len(self._col_edges):
            return
        if top_row < 0 or bot_row + 1 >= len(self._row_edges):
            return
        r0 = self._row_edges[top_row]
        r1 = self._row_edges[bot_row + 1]
        c0 = self._col_edges[char_col]
        c1 = self._col_edges[char_col + 1]
        if r1 <= r0 or c1 <= c0:
            return
        self._paint_full_res_region(r0, r1, c0, c1)
        self.refresh()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if not self.annotation_enabled:
            return
        self._dragging = True
        self.capture_mouse()
        self._paint_at_screen_pos(event.x, event.y)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self.annotation_enabled or not self._dragging:
            return
        self._paint_at_screen_pos(event.x, event.y)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()

    def on_key(self, event: events.Key) -> None:
        if not self.annotation_enabled:
            return
        if event.key == "f":
            self.set_paint_label(ANNOTATION_FOREGROUND)
            self.refresh()
            event.stop()
        elif event.key == "b":
            self.set_paint_label(ANNOTATION_BACKGROUND)
            self.refresh()
            event.stop()
        elif event.key == "c":
            self.clear_annotation_slice()
            event.stop()

    def render(self) -> Text:
        if self.volume is None:
            return Text("No scan loaded yet.", justify="center")

        img = self._take_slice(self.volume)
        mask = self._take_slice(self.mask) if self.mask is not None else None
        annotation = self._take_slice(self.annotation_mask) if self.annotation_mask is not None else None

        width = max(20, self.size.width - 2)
        height_chars = max(10, self.size.height - 2)
        # Each character row renders two image rows via half-block glyphs, so
        # sample twice as many image rows as there are character rows.
        height = height_chars * 2

        row_edges, col_edges = _compute_edges(*img.shape, height, width)
        self._row_edges, self._col_edges = row_edges, col_edges

        gray_full, mask_ds = _block_reduce_with_edges(img, mask, row_edges, col_edges)
        _, annotation_ds = _block_reduce_with_edges(None, annotation, row_edges, col_edges)
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
            if annotation_ds is not None:
                annotation_ds = np.vstack([annotation_ds, annotation_ds[-1:]])
            n_rows += 1

        text = Text()
        for r in range(0, n_rows, 2):
            for c in range(gray.shape[1]):
                top_label = int(mask_ds[r, c]) if mask_ds is not None else 0
                bot_label = int(mask_ds[r + 1, c]) if mask_ds is not None else 0
                top_ann = int(annotation_ds[r, c]) if annotation_ds is not None else ANNOTATION_UNLABELED
                bot_ann = int(annotation_ds[r + 1, c]) if annotation_ds is not None else ANNOTATION_UNLABELED
                top = _pixel_rgb(gray[r, c], top_label, top_ann)
                bot = _pixel_rgb(gray[r + 1, c], bot_label, bot_ann)
                style = Style(color=f"rgb({top[0]},{top[1]},{top[2]})",
                             bgcolor=f"rgb({bot[0]},{bot[1]},{bot[2]})")
                text.append(_HALF_BLOCK, style=style)
            text.append("\n")

        axis_name = ["sagittal", "coronal", "axial"][self.axis]
        text.append(f"\n{axis_name}  slice {self.slice_idx}/{self.volume.shape[self.axis] - 1}"
                     f"   ([/][ ]/[)] scrub  (tab) cycle axis", style="dim")
        if self.annotation_enabled:
            paint_name = "femur" if self.paint_label == ANNOTATION_FOREGROUND else "background"
            text.append(f"\n[paint: {paint_name}]  {self.annotated_slice_count()} slice(s) annotated"
                         f"   (click+drag) paint  (f/b) switch label  (c) clear slice", style="dim")
        return text
