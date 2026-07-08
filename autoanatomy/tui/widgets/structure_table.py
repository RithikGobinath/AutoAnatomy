from textual.widgets import DataTable

from autoanatomy.tui.widgets.slice_viewer import LABEL_COLORS


class StructureTable(DataTable):
    """Per-structure volume table, populated from real segmentation output."""

    def on_mount(self) -> None:
        self.add_columns("Structure", "Voxels", "Volume (mm³)")
        self.cursor_type = "row"

    def load_volumes(self, class_map: dict, volumes_mm3: dict, voxel_counts: dict) -> None:
        self.clear()
        for label_id, name in sorted(class_map.items()):
            color = LABEL_COLORS.get(label_id)
            swatch = f"[rgb({color[0]},{color[1]},{color[2]})]■[/] " if color else "  "
            voxels = voxel_counts.get(name, 0)
            volume = volumes_mm3.get(name, 0.0)
            self.add_row(f"{swatch}{name}", f"{voxels:,}", f"{volume:,.1f}")
