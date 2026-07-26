"""Real-time progress reporting for nnU-Net's sliding-window prediction loop.

nnU-Net already tracks genuine per-patch progress internally (see
_internal_predict_sliding_window_return_logits in the vendored
nnunetv2/inference/predict_from_raw_data.py): it knows the real total number
of spatial tiles for this volume up front, and increments a real counter as
each tile finishes, via a tqdm progress bar. This module taps that existing
counter by swapping in a tqdm subclass for nnU-Net's own module-level `tqdm`
import, so every pbar.update() call nnU-Net already makes also notifies
whichever callback is currently registered via report_progress().

This is purely observational. The subclass still calls real tqdm's own
__init__/update underneath -- nothing about what gets predicted, how many
folds/mirrors run, or the sliding-window step size is touched here. The only
behavior change is that tqdm's own console rendering is redirected to a
null sink while a callback is active, since the caller is presumably
rendering the same real numbers itself instead (e.g. a Textual ProgressBar)
rather than wanting two competing renderers. Note this is NOT done via
tqdm's own `disable=True`: tqdm's update() returns immediately without
touching its internal counter at all when disabled (confirmed by reading
its source), which would silently break the exact counter this module
exists to observe -- redirecting `file` instead leaves every bit of tqdm's
own counting logic running exactly as it always has.
"""
import contextvars
from contextlib import contextmanager

from tqdm import tqdm as _real_tqdm

_active_callback = contextvars.ContextVar("autoanatomy_progress_callback", default=None)

_PATCH_MARKER = "_autoanatomy_progress_patch"


class _NullWriter:
    """Discards tqdm's own console writes without touching its internal
    counting logic (unlike disable=True, which would -- see module docstring)."""

    def write(self, _text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


_NULL_WRITER = _NullWriter()


class _CallbackTqdm(_real_tqdm):
    """Drop-in tqdm replacement that also reports to whichever callback is
    currently registered via report_progress(). Behaves exactly like real
    tqdm otherwise, except its own console output is redirected to a null
    sink while a callback is active (see module docstring)."""

    def __init__(self, *args, **kwargs):
        # tqdm doesn't expose `desc` back as a readable attribute after
        # __init__ (only set_description()/set_description_str(), which are
        # write-only) -- so capture the description it was actually
        # constructed with here, straight from the same args/kwargs nnU-Net
        # itself passed in.
        self._autoanatomy_desc = kwargs.get("desc") if "desc" in kwargs else (args[1] if len(args) > 1 else None)
        if _active_callback.get() is not None:
            kwargs["file"] = _NULL_WRITER
        super().__init__(*args, **kwargs)
        self._report()

    def update(self, n=1):
        result = super().update(n)
        self._report()
        return result

    def _report(self):
        callback = _active_callback.get()
        if callback is None:
            return
        try:
            callback(self.n, self.total or 0, self._autoanatomy_desc)
        except Exception:
            # A bug in a UI-side callback must never break a real segmentation run.
            pass


@contextmanager
def report_progress(callback):
    """Route nnU-Net's own real per-patch progress counter to `callback` for
    the duration of the `with` block.

    callback(completed: int, total: int, desc: str | None) is invoked every
    time nnU-Net's sliding-window loop finishes another tile (or the
    background export step finishes another case) -- both real counts, not
    estimates. Only observes an in-process run: does not span the separate
    OS processes started by --parallel-tasks.
    """
    token = _active_callback.set(callback)
    try:
        yield
    finally:
        _active_callback.reset(token)


def patch_tqdm_progress() -> bool:
    """Swap nnU-Net's own module-level `tqdm` import for _CallbackTqdm, so
    its existing progress bars (sliding-window prediction, background export
    collection) also notify report_progress()'s callback.

    Returns True if this call installed the patch, False if it was already
    installed (idempotent, safe to call from multiple import sites).
    """
    import nnunetv2.inference.predict_from_raw_data as predict_module

    if getattr(predict_module, _PATCH_MARKER, False):
        return False

    predict_module.tqdm = _CallbackTqdm
    setattr(predict_module, _PATCH_MARKER, True)
    return True
