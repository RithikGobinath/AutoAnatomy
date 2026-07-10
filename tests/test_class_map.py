from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.registry import (
    TASK_DISPLAY_OFFSET,
    TASKS,
    get_task_classes,
    requires_license,
    task_modality,
)

EXPECTED_CRANIOFACIAL = {
    1: "mandible",
    2: "teeth_lower",
    3: "skull",
    4: "head",
    5: "sinus_maxillary",
    6: "sinus_frontal",
    7: "teeth_upper",
}

EXPECTED_HEAD_MUSCLES = {
    1: "masseter_right",
    2: "masseter_left",
    3: "temporalis_right",
    4: "temporalis_left",
    5: "lateral_pterygoid_right",
    6: "lateral_pterygoid_left",
    7: "medial_pterygoid_right",
    8: "medial_pterygoid_left",
    9: "tongue",
    10: "digastric_right",
    11: "digastric_left",
}


def test_craniofacial_structures_matches_upstream():
    assert class_map["craniofacial_structures"] == EXPECTED_CRANIOFACIAL


def test_head_muscles_matches_upstream():
    assert class_map["head_muscles"] == EXPECTED_HEAD_MUSCLES


def test_selectable_tasks():
    assert TASKS == ["craniofacial_structures", "head_muscles"]


def test_no_task_requires_a_license():
    for task in TASKS:
        assert requires_license(task) is False


def test_all_tasks_are_ct():
    for task in TASKS:
        assert task_modality(task) == "CT"


def test_get_task_classes_returns_expected_counts():
    assert len(get_task_classes("craniofacial_structures")) == 7
    assert len(get_task_classes("head_muscles")) == 11


def test_total_class_map_kept_for_skull_cropping():
    # engine/api.py crops to the skull region using class_map["total"]["skull"]
    # before running the craniofacial/head_muscles models -- this must stay intact.
    assert class_map["total"][91] == "skull"


def test_task_display_offset_covers_every_task_with_no_overlap():
    assert set(TASK_DISPLAY_OFFSET) == set(TASKS)

    # Every task's offset + native label ID range must be disjoint from every
    # other task's, or combining multiple tasks in the results table/overlay
    # (tui/screens/results.py) would collide two unrelated structures onto the
    # same display ID and color (see tui/widgets/slice_viewer.py LABEL_COLORS).
    used_display_ids = set()
    for task in TASKS:
        offset = TASK_DISPLAY_OFFSET[task]
        for label_id in class_map[task]:
            display_id = offset + label_id
            assert display_id not in used_display_ids, (
                f"{task}'s label {label_id} collides with another task at display id {display_id}"
            )
            used_display_ids.add(display_id)


def test_structure_names_unique_across_tasks():
    # tui/screens/run_progress.py merges per-structure result dicts across
    # every selected task keyed by structure *name* -- this only works safely
    # if no two tasks share a name.
    seen = set()
    for task in TASKS:
        for name in class_map[task].values():
            assert name not in seen, f"{name!r} appears in more than one task's class map"
            seen.add(name)
