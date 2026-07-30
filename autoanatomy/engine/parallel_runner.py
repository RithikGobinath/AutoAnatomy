"""
Run multiple independent segmentation tasks concurrently, each in its own
process, instead of one after another.

Opt-in only (see --parallel-tasks on the CLI / the TUI's "Run tasks in
parallel" checkbox): running several tasks back to back is the long-standing
default and stays that way unless explicitly requested, since running
multiple tasks at once means multiple models' memory footprints (and, on
GPU, VRAM) all at once rather than one at a time.
"""
import contextlib
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_one_task(kwargs: dict) -> tuple:
    """Runs in a fresh child process (spawned, so it re-imports everything --
    this is unavoidable on Windows and is what keeps each task's torch/
    nnU-Net state fully isolated from the others).

    Captures this task's own stdout as one block rather than streaming it
    live, since a child process's prints don't route through the parent's
    own sys.stdout redirection.

    Also redirects stderr to a throwaway buffer (discarded, not returned):
    engine/progress.py's report_progress() callback is a contextvars.ContextVar,
    which does not cross the process boundary a spawned worker introduces --
    so in this child process nnU-Net's own tqdm bars run undiverted and write
    real \\r-based progress text straight to the real stderr, which is
    inherited from the parent and corrupts whatever terminal UI (e.g. the
    Textual TUI) is rendered there. There's no live per-tile bar for this
    parallel path anyway (see run_progress.py), so this output was never
    meant to be shown -- just silence it at the source instead of leaking it.
    """
    from autoanatomy.engine.api import segment

    task_name = kwargs.get("task", "?")
    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            segment(**kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return task_name, buf.getvalue(), error


# Unbounded (one process per selected task, all launched at once) is how
# this used to run and is a real way to exhaust system RAM: every worker
# loads its own full copy of torch + nnU-Net + the input scan into memory
# simultaneously, with no relationship to how much RAM is actually free.
# Default to a conservative cap rather than "as many as you selected" --
# still faster than fully sequential for most task counts, but bounded.
DEFAULT_MAX_PARALLEL_TASKS = 2


def run_tasks_concurrently(task_kwargs_list: list, on_task_done=None, max_workers=None):
    """Run segment(**kwargs) for each kwargs dict in task_kwargs_list, up to
    max_workers at once (each in its own process), queueing the rest.

    max_workers=None uses DEFAULT_MAX_PARALLEL_TASKS. Pass an explicit int
    (e.g. len(task_kwargs_list), to fully restore the old unbounded
    behavior) if you know you have the RAM/VRAM for it.

    on_task_done(task_name, captured_output, error_or_none), if given, is
    called as each task finishes -- in completion order, not submission
    order.

    Raises RuntimeError listing every task that failed, once all tasks (not
    just the first failure) have had a chance to finish.
    """
    if max_workers is None:
        max_workers = DEFAULT_MAX_PARALLEL_TASKS
    max_workers = max(1, min(max_workers, len(task_kwargs_list)))

    ctx = multiprocessing.get_context("spawn")
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_one_task, kwargs) for kwargs in task_kwargs_list]
        for future in as_completed(futures):
            task_name, output, error = future.result()
            if on_task_done is not None:
                on_task_done(task_name, output, error)
            if error is not None:
                failures.append((task_name, error))

    if failures:
        details = "; ".join(f"{name}: {err}" for name, err in failures)
        raise RuntimeError(f"{len(failures)} task(s) failed: {details}")
