from autoanatomy.engine.class_map import class_map
from autoanatomy.engine.registry import TASKS, get_task_classes, requires_license, task_modality

EXPECTED_CRANIOFACIAL = {
    1: "mandible",
    2: "teeth_lower",
    3: "skull",
    4: "head",
    5: "sinus_maxillary",
    6: "sinus_frontal",
    7: "teeth_upper",
}


def test_craniofacial_structures_matches_upstream():
    assert class_map["craniofacial_structures"] == EXPECTED_CRANIOFACIAL


def test_only_craniofacial_structures_is_selectable():
    assert TASKS == ["craniofacial_structures"]


def test_craniofacial_structures_is_free():
    assert requires_license("craniofacial_structures") is False


def test_craniofacial_structures_is_ct():
    assert task_modality("craniofacial_structures") == "CT"


def test_get_task_classes_returns_seven_structures():
    assert len(get_task_classes("craniofacial_structures")) == 7


def test_total_class_map_kept_for_skull_cropping():
    # engine/api.py crops to the skull region using class_map["total"]["skull"]
    # before running the craniofacial model -- this must stay intact.
    assert class_map["total"][91] == "skull"
