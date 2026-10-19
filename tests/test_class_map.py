from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.registry import (
    TASK_DISPLAY_OFFSET,
    TASKS,
    get_task_classes,
    requires_license,
    task_modality,
)

EXPECTED_DENTAL_SEGMENTATOR = {
    1: "upper_skull",
    2: "mandible",
    3: "upper_teeth",
    4: "lower_teeth",
    5: "mandibular_canal",
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

EXPECTED_INDIVIDUAL_TEETH = {
    1: "tooth_11", 2: "tooth_12", 3: "tooth_13", 4: "tooth_14",
    5: "tooth_15", 6: "tooth_16", 7: "tooth_17", 8: "tooth_18",
    9: "tooth_21", 10: "tooth_22", 11: "tooth_23", 12: "tooth_24",
    13: "tooth_25", 14: "tooth_26", 15: "tooth_27", 16: "tooth_28",
    17: "tooth_31", 18: "tooth_32", 19: "tooth_33", 20: "tooth_34",
    21: "tooth_35", 22: "tooth_36", 23: "tooth_37", 24: "tooth_38",
    25: "tooth_41", 26: "tooth_42", 27: "tooth_43", 28: "tooth_44",
    29: "tooth_45", 30: "tooth_46", 31: "tooth_47", 32: "tooth_48",
}

EXPECTED_HEADNECK_MUSCLES = {
    1: "sternocleidomastoid_right",
    2: "sternocleidomastoid_left",
    3: "superior_pharyngeal_constrictor",
    4: "middle_pharyngeal_constrictor",
    5: "inferior_pharyngeal_constrictor",
    6: "trapezius_right",
    7: "trapezius_left",
    8: "platysma_right",
    9: "platysma_left",
    10: "levator_scapulae_right",
    11: "levator_scapulae_left",
    12: "anterior_scalene_right",
    13: "anterior_scalene_left",
    14: "middle_scalene_right",
    15: "middle_scalene_left",
    16: "posterior_scalene_right",
    17: "posterior_scalene_left",
    18: "sterno_thyroid_right",
    19: "sterno_thyroid_left",
    20: "thyrohyoid_right",
    21: "thyrohyoid_left",
    22: "prevertebral_right",
    23: "prevertebral_left",
}


def test_dental_segmentator_matches_upstream():
    assert class_map["dental_segmentator"] == EXPECTED_DENTAL_SEGMENTATOR


def test_head_muscles_matches_upstream():
    assert class_map["head_muscles"] == EXPECTED_HEAD_MUSCLES


def test_headneck_muscles_matches_upstream():
    assert class_map["headneck_muscles"] == EXPECTED_HEADNECK_MUSCLES


def test_individual_teeth_matches_upstream():
    assert class_map["individual_teeth"] == EXPECTED_INDIVIDUAL_TEETH


def test_selectable_tasks():
    assert TASKS == ["dental_segmentator", "head_muscles", "headneck_muscles", "individual_teeth", "auto3dseg_muscles"]


def test_auto3dseg_muscles_exposes_only_the_requested_groups():
    assert class_map["auto3dseg_muscles"] == {
        1: "masseter", 2: "temporalis", 3: "medial_pterygoid",
    }


def test_only_headneck_muscles_requires_a_license():
    # dental_segmentator, head_muscles and individual_teeth are all free for
    # commercial and non-commercial use. headneck_muscles is the one
    # exception in this build -- upstream requires a paid license for
    # commercial use.
    assert requires_license("dental_segmentator") is False
    assert requires_license("head_muscles") is False
    assert requires_license("headneck_muscles") is True
    assert requires_license("individual_teeth") is False


def test_all_tasks_are_ct():
    for task in TASKS:
        assert task_modality(task) == "CT"


def test_get_task_classes_returns_expected_counts():
    assert len(get_task_classes("dental_segmentator")) == 5
    assert len(get_task_classes("head_muscles")) == 11
    assert len(get_task_classes("headneck_muscles")) == 23
    assert len(get_task_classes("individual_teeth")) == 32


def test_total_class_map_kept_for_crop_infrastructure():
    # engine/api.py's generic crop-generation step reads class_map["total"]
    # for any task whose branch sets crop != None -- this must stay intact
    # even though no selectable task currently uses it.
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
