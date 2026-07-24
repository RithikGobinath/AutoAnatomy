import argparse
from unittest import mock

import pytest

from autoanatomy.cli.main import cmd_segment


def _args(**overrides):
    ns = argparse.Namespace(
        input="scan.nii.gz",
        output="out",
        task=["craniofacial_structures"],
        structures=None,
        ml=False,
        device="cpu",
        quiet=False,
        verbose=False,
        statistics=False,
        remove_small_blobs=False,
        resample_threads=1,
        saving_threads=6,
        resampling_order=3,
        parallel_tasks=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def segment_calls():
    calls = []

    def fake_segment(**kwargs):
        calls.append(kwargs)

    with mock.patch("autoanatomy.engine.api.segment", side_effect=fake_segment), \
         mock.patch("autoanatomy.engine.api.validate_device_type_api", return_value=None), \
         mock.patch("shutil.disk_usage", return_value=(0, 0, 100 * 1e9)):
        yield calls


def test_single_task_is_unchanged_from_pre_multitask_behavior(segment_calls):
    rc = cmd_segment(_args())
    assert rc == 0
    assert len(segment_calls) == 1
    assert segment_calls[0]["task"] == "craniofacial_structures"
    assert segment_calls[0]["roi_subset"] is None
    assert str(segment_calls[0]["output"]) == "out"


def test_single_task_ml_keeps_literal_output_path(segment_calls):
    rc = cmd_segment(_args(ml=True, output="out.nii.gz"))
    assert rc == 0
    assert len(segment_calls) == 1
    assert str(segment_calls[0]["output"]) == "out.nii.gz"


def test_multi_task_runs_both_with_no_structures_filter(segment_calls):
    rc = cmd_segment(_args(task=["craniofacial_structures", "head_muscles"]))
    assert rc == 0
    assert len(segment_calls) == 2
    assert {c["task"] for c in segment_calls} == {"craniofacial_structures", "head_muscles"}
    assert all(c["roi_subset"] is None for c in segment_calls)
    assert all(str(c["output"]) == "out" for c in segment_calls)


def test_multi_task_ml_suffixes_output_per_task(segment_calls):
    rc = cmd_segment(_args(task=["craniofacial_structures", "head_muscles"], ml=True, output="out.nii.gz"))
    assert rc == 0
    outs = sorted(str(c["output"]) for c in segment_calls)
    assert outs == ["out_craniofacial_structures.nii.gz", "out_head_muscles.nii.gz"]


def test_multi_task_structures_split_per_task(segment_calls):
    rc = cmd_segment(_args(
        task=["craniofacial_structures", "head_muscles"],
        structures="mandible,masseter_left,tongue",
    ))
    assert rc == 0
    by_task = {c["task"]: c["roi_subset"] for c in segment_calls}
    assert by_task["craniofacial_structures"] == ["mandible"]
    assert sorted(by_task["head_muscles"]) == ["masseter_left", "tongue"]


def test_multi_task_structures_matching_only_one_task_skips_the_other(segment_calls):
    rc = cmd_segment(_args(task=["craniofacial_structures", "head_muscles"], structures="mandible"))
    assert rc == 0
    assert len(segment_calls) == 1
    assert segment_calls[0]["task"] == "craniofacial_structures"
    assert segment_calls[0]["roi_subset"] == ["mandible"]


def test_unknown_structure_name_is_an_error(segment_calls):
    rc = cmd_segment(_args(
        task=["craniofacial_structures", "head_muscles"],
        structures="mandible,not_a_real_structure",
    ))
    assert rc == 1
    assert segment_calls == []


def test_structures_matching_no_selected_task_is_an_error(segment_calls):
    rc = cmd_segment(_args(task=["craniofacial_structures"], structures="masseter_left"))
    assert rc == 1
    assert segment_calls == []


def test_repeated_task_on_cli_is_deduped(segment_calls):
    rc = cmd_segment(_args(task=["head_muscles", "head_muscles"]))
    assert rc == 0
    assert len(segment_calls) == 1
