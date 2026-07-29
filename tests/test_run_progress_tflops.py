import asyncio

from textual.app import App, ComposeResult
from textual.widgets import ProgressBar, Static

from autoanatomy.tui.screens.run_progress import RunProgressScreen

# RunProgressScreen._reset_progress/_on_progress only touch self.query_one(...)
# for the three widgets below -- exercising them against a bare App hosting
# those same widgets (instead of mounting the real Screen) sidesteps Textual's
# on_mount lifecycle, which would otherwise kick off a real segmentation run
# (RunProgressScreen.on_mount -> self.run_segmentation()) the moment the
# screen is pushed.


class _FakeProgressHost(App):
    def compose(self) -> ComposeResult:
        yield Static("", id="progress-phase")
        yield ProgressBar(id="task-progress")
        yield Static("", id="tflops-readout")


def test_tflops_readout_shows_a_value_between_two_progress_calls():
    async def _run():
        app = _FakeProgressHost()
        async with app.run_test() as pilot:
            RunProgressScreen._reset_progress(app, "Preparing...")
            # First call only establishes the timing baseline -- no rate yet.
            RunProgressScreen._on_progress(app, 1, 10, None, 1e12)
            await pilot.pause()
            assert str(app.query_one("#tflops-readout", Static).render()) == ""

            RunProgressScreen._on_progress(app, 2, 10, None, 1e12)
            await pilot.pause()
            text = str(app.query_one("#tflops-readout", Static).render())
            assert "TFLOPS" in text

    asyncio.run(_run())


def test_tflops_readout_stays_blank_when_flops_unmeasured():
    async def _run():
        app = _FakeProgressHost()
        async with app.run_test() as pilot:
            RunProgressScreen._reset_progress(app, "Preparing...")
            RunProgressScreen._on_progress(app, 1, 10, None, None)
            await pilot.pause()
            RunProgressScreen._on_progress(app, 2, 10, None, None)
            await pilot.pause()

            assert str(app.query_one("#tflops-readout", Static).render()) == ""

    asyncio.run(_run())


def test_reset_progress_clears_tflops_readout():
    async def _run():
        app = _FakeProgressHost()
        async with app.run_test() as pilot:
            RunProgressScreen._reset_progress(app, "Preparing...")
            RunProgressScreen._on_progress(app, 1, 10, None, 1e12)
            RunProgressScreen._on_progress(app, 2, 10, None, 1e12)
            await pilot.pause()
            assert str(app.query_one("#tflops-readout", Static).render()) != ""

            RunProgressScreen._reset_progress(app, "Preparing next task...")
            assert str(app.query_one("#tflops-readout", Static).render()) == ""

    asyncio.run(_run())
