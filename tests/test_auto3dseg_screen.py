import asyncio
from pathlib import Path

from textual.app import App
from textual.widgets import Checkbox, Static

from autoanatomy.engine.registry import TASKS
from autoanatomy.tui.screens.configure_run import ConfigureRunScreen
from autoanatomy.tui.screens.home import HomeScreen
from autoanatomy.tui.screens.load_scan import LoadScanScreen
from autoanatomy.tui.screens.results import ResultsScreen
from autoanatomy.tui.widgets.slice_viewer import SliceViewer


class _Host(App):
    scan_path = Path("scan.nii.gz")


def test_home_auto3dseg_button_opens_dedicated_configuration():
    async def run():
        app = _Host()
        async with app.run_test(size=(120, 48)) as pilot:
            await app.push_screen(HomeScreen())
            assert app.screen.query_one("#auto3dseg-section").query_one(Static).render().plain == "Auto3DSeg"
            await pilot.click("#auto3dseg-run")
            assert isinstance(app.screen, LoadScanScreen)
            screen = app.screen.next_screen_factory()
            assert screen.available_tasks == ["auto3dseg_muscles"]
            await app.push_screen(screen)
            await pilot.pause()
            i = TASKS.index("auto3dseg_muscles")
            assert screen.query_one(f"#task-enable-{i}", Checkbox).value
            assert len(screen.query(".task-structures Checkbox")) == 3
            assert screen.query_one("#struct-masseter", Checkbox).value
            assert screen.query_one("#struct-temporalis", Checkbox).value
            assert screen.query_one("#struct-medial_pterygoid", Checkbox).value
    asyncio.run(run())


def test_results_preview_preserves_auto3dseg_display_ids_above_255(tmp_path):
    import nibabel as nib
    import numpy as np

    class PreviewHost(App):
        scan_path = tmp_path / "scan.nii.gz"
        selected_tasks = {"auto3dseg_muscles": None}
        output_dir = tmp_path
        ml = False

        def compose(self):
            yield SliceViewer(id="slice-viewer")

    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[3:5, 3:5, 3:5] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), PreviewHost.scan_path)
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "medial_pterygoid.nii.gz")

    async def run():
        app = PreviewHost()
        async with app.run_test():
            ResultsScreen._load_preview(app)
            mask = app.query_one(SliceViewer).mask
            assert mask is not None
            assert mask.max() == 403
    asyncio.run(run())
