from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from autoanatomy.tui.widgets.slice_viewer import SliceViewer
from autoanatomy.tui.widgets.structure_table import StructureTable


class ResultsScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("[", "scrub_back", "Prev slice"),
        ("]", "scrub_fwd", "Next slice"),
        ("tab", "cycle_axis", "Cycle axis"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with VerticalScroll(classes="panel scroll-wrapper"):
                yield Static("Results", classes="section-title")
                yield StructureTable(id="structure-table")
                if self.app.statistics_path is not None:
                    yield Static(f"\n[dim]Full stats: {self.app.statistics_path}[/dim]")
                yield Static("\nExport:", classes="section-title")
                with Vertical():
                    yield Button("Save NIfTI masks (done)", id="export-nifti", disabled=True)
                    yield Button("3D mesh (STL)  [Phase 2]", id="export-stl", disabled=True, variant="warning")
                yield Button("Back to Home", id="home-btn")
            with Vertical(classes="panel"):
                yield Static("Slice viewer", classes="section-title")
                yield SliceViewer(id="slice-viewer")
        yield Footer()

    def on_mount(self) -> None:
        from autoanatomy.engine.class_map import class_map
        from autoanatomy.engine.registry import TASK_DISPLAY_OFFSET

        # Combine every selected task's classes into one {display_id: name}
        # dict. The offset only disambiguates the table/overlay -- it's never
        # applied to the actual saved NIfTI files, which always use each
        # task's own native label IDs (see _load_preview and registry.py).
        combined_map = {}
        for task, roi_subset in self.app.selected_tasks.items():
            offset = TASK_DISPLAY_OFFSET[task]
            for label_id, name in class_map[task].items():
                if roi_subset is not None and name not in roi_subset:
                    continue
                combined_map[offset + label_id] = name

        table = self.query_one("#structure-table", StructureTable)
        table.load_volumes(
            combined_map,
            self.app.result_volumes_mm3,
            self.app.result_voxel_counts,
        )
        self._load_preview()

    def _load_preview(self) -> None:
        scan_path = self.app.scan_path
        if scan_path is None or scan_path.is_dir() or not str(scan_path).endswith((".nii", ".nii.gz")):
            return
        try:
            import nibabel as nib
            import numpy as np

            from autoanatomy.engine.class_map import class_map
            from autoanatomy.engine.registry import TASK_DISPLAY_OFFSET
            from autoanatomy.tui.screens.run_progress import _ml_output_path

            img = nib.load(str(scan_path))
            volume = np.asanyarray(img.dataobj)

            multi_task = len(self.app.selected_tasks) > 1
            combined_mask = np.zeros(volume.shape, dtype=np.uint8)

            for task, roi_subset in self.app.selected_tasks.items():
                offset = TASK_DISPLAY_OFFSET[task]
                if self.app.ml:
                    output = _ml_output_path(self.app.output_dir, task, multi_task)
                    if not output.exists():
                        continue
                    m = nib.load(str(output))
                    data = np.asanyarray(m.dataobj)
                    if data.shape != volume.shape:
                        continue
                    for label_id, name in class_map[task].items():
                        if roi_subset is not None and name not in roi_subset:
                            continue
                        combined_mask[data == label_id] = offset + label_id
                else:
                    for label_id, name in class_map[task].items():
                        mask_path = self.app.output_dir / f"{name}.nii.gz"
                        if not mask_path.exists():
                            continue
                        m = nib.load(str(mask_path))
                        data = np.asanyarray(m.dataobj)
                        if data.shape == volume.shape:
                            combined_mask[data > 0] = offset + label_id

            self.query_one("#slice-viewer", SliceViewer).set_volume(volume, combined_mask)
        except Exception:
            pass

    def action_scrub_back(self) -> None:
        self.query_one("#slice-viewer", SliceViewer).scrub(-1)

    def action_scrub_fwd(self) -> None:
        self.query_one("#slice-viewer", SliceViewer).scrub(1)

    def action_cycle_axis(self) -> None:
        self.query_one("#slice-viewer", SliceViewer).cycle_axis()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "home-btn":
            from autoanatomy.tui.screens.home import HomeScreen
            self.app.switch_screen(HomeScreen())
