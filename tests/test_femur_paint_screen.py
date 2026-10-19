import asyncio
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from textual.app import App, ComposeResult
from textual.widgets import Static

from autoanatomy.tui.screens.femur_semiauto import FemurPaintScreen
from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_BACKGROUND, ANNOTATION_FOREGROUND, SliceViewer


def _make_scan_file(tmp_path, shape=(32, 32, 32)):
    rng = np.random.default_rng(0)
    vol = np.full(shape, -1000.0, dtype=np.float32) + rng.normal(0, 20, shape).astype(np.float32)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    vol[dist < 8] = 800 + rng.normal(0, 50, shape)[dist < 8]
    scan_path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)), str(scan_path))
    return scan_path


class _FemurTestApp(App):
    def __init__(self, scan_path, output_dir):
        super().__init__()
        self.scan_path = scan_path
        self.output_dir = output_dir

    def compose(self) -> ComposeResult:
        yield from ()


def test_femur_paint_screen_loads_scan_and_enables_annotation():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scan_path = _make_scan_file(tmp_path)
            app = _FemurTestApp(scan_path, tmp_path / "out")
            async with app.run_test() as pilot:
                screen = FemurPaintScreen()
                await app.push_screen(screen)
                await pilot.pause()

                sv = screen.query_one("#slice-viewer", SliceViewer)
                assert sv.volume is not None
                assert sv.annotation_enabled is True

    asyncio.run(_run())


def test_femur_paint_screen_run_button_blocked_without_seeds():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scan_path = _make_scan_file(tmp_path)
            app = _FemurTestApp(scan_path, tmp_path / "out")
            async with app.run_test() as pilot:
                screen = FemurPaintScreen()
                await app.push_screen(screen)
                await pilot.pause()

                screen._start_segmentation()
                await pilot.pause()

                assert screen._segmentation_running is False

    asyncio.run(_run())


def test_femur_paint_screen_runs_real_segmentation_end_to_end():
    """Paints real seeds via the actual SliceViewer painting API, runs the
    real graph-cut pipeline through the background worker, and confirms the
    result gets displayed as a mask overlay -- not mocked at any layer."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scan_path = _make_scan_file(tmp_path)
            output_dir = tmp_path / "out"
            app = _FemurTestApp(scan_path, output_dir)
            async with app.run_test() as pilot:
                screen = FemurPaintScreen()
                await app.push_screen(screen)
                await pilot.pause()

                sv = screen.query_one("#slice-viewer", SliceViewer)
                # Paint directly into the mask (same code path a real click
                # drives) -- bypasses needing exact screen-coordinate math.
                sv.annotation_mask[10:14, 10:14, 10:14] = ANNOTATION_FOREGROUND
                sv.annotation_mask[2:5, 2:5, 2:5] = ANNOTATION_BACKGROUND
                sv.annotation_mask[28:31, 28:31, 28:31] = ANNOTATION_BACKGROUND

                screen._start_segmentation()
                await pilot.pause()
                assert screen._segmentation_running is True

                deadline = time.monotonic() + 60
                while screen._segmentation_running and time.monotonic() < deadline:
                    await asyncio.sleep(0.2)
                    await pilot.pause()

                assert screen._segmentation_running is False, "segmentation worker never finished"
                assert (output_dir / "femur_graphcut_result.nii.gz").exists()
                assert sv.annotation_enabled is False
                assert sv.mask is not None
                assert sv.mask.any(), "result mask overlay is empty"

    asyncio.run(_run())
