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
    """
    from autoanatomy.engine.api import segment

    task_name = kwargs.get("task", "?")
    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            segment(**kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return task_name, buf.getvalue(), error


def run_tasks_concurrently(task_kwargs_list: list, on_task_done=None):
    """Run segment(**kwargs) for each kwargs dict in task_kwargs_list in its
    own process, concurrently.

    on_task_done(task_name, captured_output, error_or_none), if given, is
    called as each task finishes -- in completion order, not submission
    order.

    Raises RuntimeError listing every task that failed, once all tasks (not
    just the first failure) have had a chance to finish.
    """
    ctx = multiprocessing.get_context("spawn")
    failures = []
    with ProcessPoolExecutor(max_workers=len(task_kwargs_list), mp_context=ctx) as executor:
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
