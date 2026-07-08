from pathlib import Path

from textual.app import App


class AutoAnatomyApp(App):
    """AutoAnatomy - craniofacial CT segmentation, terminal edition.

    Real engine underneath (real nnU-Net inference, real GPU/CPU, real NIfTI
    I/O). Screens for features we haven't built yet are clearly marked
    "Phase 2" rather than faked.
    """

    CSS_PATH = "app.tcss"
    TITLE = "AutoAnatomy"
    SUB_TITLE = "craniofacial segmentation — under construction"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    # Shared run state, read/written by screens as the user moves through the flow.
    scan_path: Path | None = None
    device: str = "gpu"
    ml: bool = False
    output_dir: Path | None = None
    result_volumes_mm3: dict = {}
    result_voxel_counts: dict = {}

    # Advanced settings -- all real, thread straight through to engine.api.segment().
    statistics: bool = False
    remove_small_blobs: bool = False
    robust_crop: bool = False
    nr_thr_resamp: int = 1
    nr_thr_saving: int = 6
    resampling_order: int = 3
    statistics_path: Path | None = None

    def on_mount(self) -> None:
        from autoanatomy.tui.screens.splash import SplashScreen
        self.push_screen(SplashScreen())


def run():
    AutoAnatomyApp().run()


if __name__ == "__main__":
    run()
