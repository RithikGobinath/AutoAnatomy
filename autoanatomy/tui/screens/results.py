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
        table = self.query_one("#structure-table", StructureTable)
        table.load_volumes(
            class_map["craniofacial_structures"],
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

            img = nib.load(str(scan_path))
            volume = np.asanyarray(img.dataobj)

            combined_mask = np.zeros(volume.shape, dtype=np.uint8)
            if self.app.ml:
                if self.app.output_dir.exists():
                    m = nib.load(str(self.app.output_dir))
                    data = np.asanyarray(m.dataobj)
                    if data.shape == volume.shape:
                        combined_mask = data.astype(np.uint8)
            else:
                for label_id, name in class_map["craniofacial_structures"].items():
                    mask_path = self.app.output_dir / f"{name}.nii.gz"
                    if mask_path.exists():
                        m = nib.load(str(mask_path))
                        data = np.asanyarray(m.dataobj)
                        if data.shape == volume.shape:
                            combined_mask[data > 0] = label_id

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
