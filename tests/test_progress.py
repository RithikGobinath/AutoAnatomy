from types import SimpleNamespace

from autoanatomy.engine.progress import (
    measure_flops_per_tile,
    patch_tqdm_progress,
    report_progress,
    set_flops_per_unit,
)


def test_patch_is_idempotent():
    assert patch_tqdm_progress() in (True, False)
    # Whether or not this process already patched it, a second call must be a no-op.
    assert patch_tqdm_progress() is False


def test_callback_receives_real_incrementing_counts():
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    calls = []
    with report_progress(lambda completed, total, desc, flops_per_unit: calls.append((completed, total, desc))):
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
    with report_progress(lambda completed, total, desc, flops_per_unit: calls.append((completed, total, desc))):
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
    with report_progress(lambda completed, total, desc, flops_per_unit: calls.append(completed)):
        bar = prd.tqdm(total=3, desc=None, disable=False)
        for _ in range(3):
            bar.update(1)
        bar.close()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert calls[-1] == 3


def test_flops_per_unit_is_passed_through_the_callback():
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    set_flops_per_unit(123.0)
    calls = []
    with report_progress(lambda completed, total, desc, flops_per_unit: calls.append(flops_per_unit)):
        bar = prd.tqdm(total=1, desc=None)
        bar.update(1)
        bar.close()
    set_flops_per_unit(None)

    # _CallbackTqdm reports once on construction (n=0) and once per update().
    assert calls == [123.0, 123.0]


def test_flops_per_unit_defaults_to_none_when_unset():
    patch_tqdm_progress()
    import nnunetv2.inference.predict_from_raw_data as prd

    calls = []
    with report_progress(lambda completed, total, desc, flops_per_unit: calls.append(flops_per_unit)):
        bar = prd.tqdm(total=1, desc=None)
        bar.update(1)
        bar.close()

    assert calls == [None, None]


def _make_fake_predictor(use_mirroring, allowed_mirroring_axes):
    import torch
    import torch.nn as nn

    network = nn.Sequential(nn.Conv3d(1, 2, kernel_size=3, padding=1))
    return SimpleNamespace(
        network=network,
        configuration_manager=SimpleNamespace(patch_size=(4, 4, 4)),
        num_input_channels=1,
        use_mirroring=use_mirroring,
        allowed_mirroring_axes=allowed_mirroring_axes,
    )


def test_measure_flops_per_tile_returns_a_real_positive_number():
    predictor = _make_fake_predictor(use_mirroring=False, allowed_mirroring_axes=())
    flops = measure_flops_per_tile(predictor)
    assert flops is not None
    assert flops > 0


def test_measure_flops_per_tile_multiplies_by_mirror_variants():
    no_mirror = measure_flops_per_tile(_make_fake_predictor(use_mirroring=False, allowed_mirroring_axes=(0, 1, 2)))
    with_mirror = measure_flops_per_tile(_make_fake_predictor(use_mirroring=True, allowed_mirroring_axes=(0, 1, 2)))
    assert no_mirror is not None and with_mirror is not None
    assert with_mirror == no_mirror * 8


def test_measure_flops_per_tile_returns_none_on_failure():
    broken_predictor = SimpleNamespace(network=object())  # missing everything measure_flops_per_tile needs
    assert measure_flops_per_tile(broken_predictor) is None
