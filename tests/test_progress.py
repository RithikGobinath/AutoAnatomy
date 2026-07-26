from autoanatomy.engine.progress import patch_tqdm_progress, report_progress


def test_patch_is_idempotent():
    assert patch_tqdm_progress() in (True, False)
    # Whether or not this process already patched it, a second call must be a no-op.
    assert patch_tqdm_progress() is False


def test_callback_receives_real_incrementing_counts():
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    calls = []
    with report_progress(lambda completed, total, desc: calls.append((completed, total, desc))):
        bar = prd.tqdm(total=5, desc=None)
        for _ in range(5):
            bar.update(1)
        bar.close()

    assert [c[0] for c in calls] == [0, 1, 2, 3, 4, 5]
    assert all(c[1] == 5 for c in calls)
    assert all(c[2] is None for c in calls)


def test_desc_is_reported_correctly():
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    calls = []
    with report_progress(lambda completed, total, desc: calls.append((completed, total, desc))):
        bar = prd.tqdm(total=1, desc="Collecting results")
        bar.update(1)
        bar.close()

    assert calls[-1] == (1, 1, "Collecting results")


def test_no_callback_registered_leaves_tqdm_untouched(capsys):
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    # No report_progress() context active -- must behave exactly like stock
    # tqdm, including actually writing its console bar (to stderr).
    bar = prd.tqdm(total=3, desc=None, disable=False, file=__import__("sys").stderr)
    for _ in range(3):
        bar.update(1)
    bar.close()

    captured = capsys.readouterr()
    assert captured.err != ""


def test_callback_suppresses_console_output_without_breaking_the_counter(capsys):
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    calls = []
    with report_progress(lambda completed, total, desc: calls.append(completed)):
        bar = prd.tqdm(total=3, desc=None, disable=False)
        for _ in range(3):
            bar.update(1)
        bar.close()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert calls[-1] == 3
