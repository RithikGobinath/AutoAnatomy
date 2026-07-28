import asyncio

import numpy as np

from autoanatomy.tui.widgets.slice_viewer import _HALF_BLOCK, _block_reduce


def test_block_reduce_area_averages_grayscale():
    img = np.arange(16).reshape(4, 4).astype(float)
    out, mask_out = _block_reduce(img, None, 2, 2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    assert np.allclose(out, expected)
    assert mask_out is None


def test_block_reduce_mask_majority_vote_prefers_nonbackground():
    mask = np.array([
        [0, 0, 5, 5],
        [5, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)
    img = np.zeros((4, 4))
    _, mask_out = _block_reduce(img, mask, 2, 2)
    # top-left quadrant: three label-5 voxels, one label-0 -- majority (of the
    # non-background voxels) wins, not a plain majority-including-background.
    assert mask_out[0, 0] == 5
    # bottom-right quadrant is all background.
    assert mask_out[1, 1] == 0


def test_block_reduce_preserves_thin_structures():
    # A single "hot" voxel in an otherwise empty large mask, at a position that
    # a naive point-sampling downsample could easily skip entirely -- this is
    # exactly the failure mode block-averaging exists to fix.
    big_mask = np.zeros((100, 100), dtype=np.uint8)
    big_mask[37, 62] = 7
    big_img = np.zeros((100, 100))
    _, small_mask = _block_reduce(big_img, big_mask, 10, 10)
    assert (small_mask == 7).any()


def test_block_reduce_output_shape_caps_at_source_size():
    # Requesting a larger output grid than the source image has pixels should
    # never crash or fabricate resolution that isn't there.
    img = np.zeros((5, 5))
    out, _ = _block_reduce(img, None, 50, 50)
    assert out.shape == (5, 5)


def test_slice_viewer_renders_half_block_glyphs():
    from textual.app import App, ComposeResult

    from autoanatomy.tui.widgets.slice_viewer import SliceViewer

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield SliceViewer(id="sv")

    async def _run():
        app = _TestApp()
        async with app.run_test(size=(80, 40)) as pilot:
            sv = app.query_one("#sv", SliceViewer)

            vol = np.zeros((32, 32, 32), dtype=np.float32)
            vol[10:20, 10:20, 10:20] = 200
            mask = np.zeros((32, 32, 32), dtype=np.uint8)
            mask[14:18, 14:18, 14:18] = 201  # dental_segmentator's "upper_skull" display id

            sv.set_volume(vol, mask)
            await pilot.pause()

            text = sv.render()
            assert _HALF_BLOCK in text.plain

            # The mask overlay color for display id 201 (deep sky blue) must
            # show up somewhere in the rendered spans -- confirms the overlay
            # survives the half-block + block-reduce pipeline end to end, not
            # just that *some* half-blocks were drawn.
            found = any(
                "0,191,255" in str(getattr(span.style, "color", ""))
                or "0,191,255" in str(getattr(span.style, "bgcolor", ""))
                for span in text.spans
            )
            assert found, "mask overlay color not found in rendered output"

    asyncio.run(_run())


def test_slice_viewer_shows_placeholder_before_volume_loaded():
    from textual.app import App, ComposeResult

    from autoanatomy.tui.widgets.slice_viewer import SliceViewer

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield SliceViewer(id="sv")

    async def _run():
        app = _TestApp()
        async with app.run_test(size=(80, 40)) as pilot:
            sv = app.query_one("#sv", SliceViewer)
            await pilot.pause()
            text = sv.render()
            assert "No scan loaded yet." in text.plain

    asyncio.run(_run())


def _make_annotatable_app():
    from textual.app import App, ComposeResult

    from autoanatomy.tui.widgets.slice_viewer import SliceViewer

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield SliceViewer(id="sv")

    return _TestApp


def test_enable_annotation_mode_requires_a_volume_first():
    from autoanatomy.tui.widgets.slice_viewer import SliceViewer

    sv = SliceViewer()
    try:
        sv.enable_annotation_mode()
        assert False, "should have raised without a loaded volume"
    except RuntimeError:
        pass


def test_real_mouse_click_paints_the_annotation_mask():
    from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_FOREGROUND

    async def _run():
        app = _make_annotatable_app()()
        async with app.run_test(size=(80, 40)) as pilot:
            from autoanatomy.tui.widgets.slice_viewer import SliceViewer
            sv = app.query_one("#sv", SliceViewer)
            vol = np.zeros((64, 64, 64), dtype=np.float32)
            sv.set_volume(vol)
            sv.enable_annotation_mode()
            await pilot.pause()
            sv.render()  # populate _row_edges/_col_edges before a real click needs them

            assert sv.annotation_mask.sum() == 0
            await pilot.click("#sv", offset=(10, 10))
            await pilot.pause()

            painted = int((sv.annotation_mask == ANNOTATION_FOREGROUND).sum())
            assert painted > 0, "a real click through Textual's event pipeline painted nothing"
            assert sv.annotated_slice_count() == 1

    asyncio.run(_run())


def test_real_key_presses_switch_label_and_clear():
    from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_BACKGROUND, ANNOTATION_FOREGROUND

    async def _run():
        app = _make_annotatable_app()()
        async with app.run_test(size=(80, 40)) as pilot:
            from autoanatomy.tui.widgets.slice_viewer import SliceViewer
            sv = app.query_one("#sv", SliceViewer)
            vol = np.zeros((64, 64, 64), dtype=np.float32)
            sv.set_volume(vol)
            sv.enable_annotation_mode()
            await pilot.pause()
            sv.render()
            sv.focus()
            await pilot.pause()

            assert sv.paint_label == ANNOTATION_FOREGROUND
            await pilot.press("b")
            await pilot.pause()
            assert sv.paint_label == ANNOTATION_BACKGROUND

            await pilot.press("f")
            await pilot.pause()
            assert sv.paint_label == ANNOTATION_FOREGROUND

            sv._paint_at_screen_pos(5, 5)
            assert sv.annotation_mask.sum() > 0
            await pilot.press("c")
            await pilot.pause()
            assert sv.annotation_mask.sum() == 0

    asyncio.run(_run())


def test_key_presses_are_noop_when_annotation_mode_is_off():
    from autoanatomy.tui.widgets.slice_viewer import ANNOTATION_FOREGROUND

    async def _run():
        app = _make_annotatable_app()()
        async with app.run_test(size=(80, 40)) as pilot:
            from autoanatomy.tui.widgets.slice_viewer import SliceViewer
            sv = app.query_one("#sv", SliceViewer)
            vol = np.zeros((64, 64, 64), dtype=np.float32)
            sv.set_volume(vol)
            sv.focus()
            await pilot.pause()
            # annotation_enabled is False -- switching label must be a no-op
            assert sv.paint_label == ANNOTATION_FOREGROUND
            await pilot.press("b")
            await pilot.pause()
            assert sv.paint_label == ANNOTATION_FOREGROUND, "label should not change while annotation mode is off"

    asyncio.run(_run())


def test_disable_annotation_mode_stops_painting():
    async def _run():
        app = _make_annotatable_app()()
        async with app.run_test(size=(80, 40)) as pilot:
            from autoanatomy.tui.widgets.slice_viewer import SliceViewer
            sv = app.query_one("#sv", SliceViewer)
            vol = np.zeros((64, 64, 64), dtype=np.float32)
            sv.set_volume(vol)
            sv.enable_annotation_mode()
            await pilot.pause()
            sv.render()
            sv.disable_annotation_mode()

            sv._paint_at_screen_pos(5, 5)
            assert sv.annotation_mask.sum() == 0, "painting should be a no-op once annotation mode is disabled"

    asyncio.run(_run())


def test_clear_annotation_slice_only_clears_the_current_slice():
    async def _run():
        app = _make_annotatable_app()()
        async with app.run_test(size=(80, 40)) as pilot:
            from autoanatomy.tui.widgets.slice_viewer import SliceViewer
            sv = app.query_one("#sv", SliceViewer)
            vol = np.zeros((64, 64, 64), dtype=np.float32)
            sv.set_volume(vol)
            sv.enable_annotation_mode()
            await pilot.pause()
            sv.render()

            sv._paint_at_screen_pos(5, 5)
            sv.scrub(5)
            sv.render()
            sv._paint_at_screen_pos(5, 5)
            assert sv.annotated_slice_count() == 2

            sv.clear_annotation_slice()  # clears only the *current* (post-scrub) slice
            assert sv.annotated_slice_count() == 1

    asyncio.run(_run())
