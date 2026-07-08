class_map = {

    # Kept because the craniofacial_structures pipeline crops to the skull
    # region first, using a rough low-res whole-scan segmentation (see
    # engine/api.py crop step).
    "total": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right",
        25: "sacrum",
        26: "vertebrae_S1",
        27: "vertebrae_L5",
        28: "vertebrae_L4",
        29: "vertebrae_L3",
        30: "vertebrae_L2",
        31: "vertebrae_L1",
        32: "vertebrae_T12",
        33: "vertebrae_T11",
        34: "vertebrae_T10",
        35: "vertebrae_T9",
        36: "vertebrae_T8",
        37: "vertebrae_T7",
        38: "vertebrae_T6",
        39: "vertebrae_T5",
        40: "vertebrae_T4",
        41: "vertebrae_T3",
        42: "vertebrae_T2",
        43: "vertebrae_T1",
        44: "vertebrae_C7",
        45: "vertebrae_C6",
        46: "vertebrae_C5",
        47: "vertebrae_C4",
        48: "vertebrae_C3",
        49: "vertebrae_C2",
        50: "vertebrae_C1",
        51: "heart",
        52: "aorta",
        53: "pulmonary_vein",
        54: "brachiocephalic_trunk",
        55: "subclavian_artery_right",
        56: "subclavian_artery_left",
        57: "common_carotid_artery_right",
        58: "common_carotid_artery_left",
        59: "brachiocephalic_vein_left",
        60: "brachiocephalic_vein_right",
        61: "atrial_appendage_left",
        62: "superior_vena_cava",
        63: "inferior_vena_cava",
        64: "portal_vein_and_splenic_vein",
        65: "iliac_artery_left",
        66: "iliac_artery_right",
        67: "iliac_vena_left",
        68: "iliac_vena_right",
        69: "humerus_left",
        70: "humerus_right",
        71: "scapula_left",
        72: "scapula_right",
        73: "clavicula_left",
        74: "clavicula_right",
        75: "femur_left",
        76: "femur_right",
        77: "hip_left",
        78: "hip_right",
        79: "spinal_cord",
        80: "gluteus_maximus_left",
        81: "gluteus_maximus_right",
        82: "gluteus_medius_left",
        83: "gluteus_medius_right",
        84: "gluteus_minimus_left",
        85: "gluteus_minimus_right",
        86: "autochthon_left",
        87: "autochthon_right",
        88: "iliopsoas_left",
        89: "iliopsoas_right",
        90: "brain",
        91: "skull",
        92: "rib_left_1",
        93: "rib_left_2",
        94: "rib_left_3",
        95: "rib_left_4",
        96: "rib_left_5",
        97: "rib_left_6",
        98: "rib_left_7",
        99: "rib_left_8",
        100: "rib_left_9",
        101: "rib_left_10",
        102: "rib_left_11",
        103: "rib_left_12",
        104: "rib_right_1",
        105: "rib_right_2",
        106: "rib_right_3",
        107: "rib_right_4",
        108: "rib_right_5",
        109: "rib_right_6",
        110: "rib_right_7",
        111: "rib_right_8",
        112: "rib_right_9",
        113: "rib_right_10",
        114: "rib_right_11",
        115: "rib_right_12",
        116: "sternum",
        117: "costal_cartilages"
    },

    # AutoAnatomy's actual product task: mandible, teeth, skull, head, sinuses
    "craniofacial_structures": {
        1: "mandible",
        2: "teeth_lower",
        3: "skull",
        4: "head",
        5: "sinus_maxillary",
        6: "sinus_frontal",
        7: "teeth_upper"
    }
}

# No task in this build requires a commercial license (craniofacial_structures
# and the "total" rough-crop model are both free). Kept as an empty dict so
# engine/weights.py's import contract still holds.
commercial_models = {}

# --- Supporting maps for the multi-part nnU-Net "total" ensemble machinery ---
# Kept from upstream because engine/nnunet_runner.py imports these unconditionally.
# Not exercised by the craniofacial_structures pipeline itself (which uses single-
# model task IDs), but the rough-crop step can fall back to related total-task
# infrastructure, so these stay intact for fidelity with upstream behavior.
class_map_5_parts = {

    # 24 classes
    "class_map_part_organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right"
    },

    # 26 classes
    "class_map_part_vertebrae": {
        1: "sacrum",
        2: "vertebrae_S1",
        3: "vertebrae_L5",
        4: "vertebrae_L4",
        5: "vertebrae_L3",
        6: "vertebrae_L2",
        7: "vertebrae_L1",
        8: "vertebrae_T12",
        9: "vertebrae_T11",
        10: "vertebrae_T10",
        11: "vertebrae_T9",
        12: "vertebrae_T8",
        13: "vertebrae_T7",
        14: "vertebrae_T6",
        15: "vertebrae_T5",
        16: "vertebrae_T4",
        17: "vertebrae_T3",
        18: "vertebrae_T2",
        19: "vertebrae_T1",
        20: "vertebrae_C7",
        21: "vertebrae_C6",
        22: "vertebrae_C5",
        23: "vertebrae_C4",
        24: "vertebrae_C3",
        25: "vertebrae_C2",
        26: "vertebrae_C1"
    },

    # 18
    "class_map_part_cardiac": {
        1: "heart",
        2: "aorta",
        3: "pulmonary_vein",
        4: "brachiocephalic_trunk",
        5: "subclavian_artery_right",
        6: "subclavian_artery_left",
        7: "common_carotid_artery_right",
        8: "common_carotid_artery_left",
        9: "brachiocephalic_vein_left",
        10: "brachiocephalic_vein_right",
        11: "atrial_appendage_left",
        12: "superior_vena_cava",
        13: "inferior_vena_cava",
        14: "portal_vein_and_splenic_vein",
        15: "iliac_artery_left",
        16: "iliac_artery_right",
        17: "iliac_vena_left",
        18: "iliac_vena_right"
    },

    # 23
    "class_map_part_muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "spinal_cord",
        12: "gluteus_maximus_left",
        13: "gluteus_maximus_right",
        14: "gluteus_medius_left",
        15: "gluteus_medius_right",
        16: "gluteus_minimus_left",
        17: "gluteus_minimus_right",
        18: "autochthon_left",
        19: "autochthon_right",
        20: "iliopsoas_left",
        21: "iliopsoas_right",
        22: "brain",
        23: "skull"
    },

    # 26 classes
    "class_map_part_ribs": {
        1: "rib_left_1",
        2: "rib_left_2",
        3: "rib_left_3",
        4: "rib_left_4",
        5: "rib_left_5",
        6: "rib_left_6",
        7: "rib_left_7",
        8: "rib_left_8",
        9: "rib_left_9",
        10: "rib_left_10",
        11: "rib_left_11",
        12: "rib_left_12",
        13: "rib_right_1",
        14: "rib_right_2",
        15: "rib_right_3",
        16: "rib_right_4",
        17: "rib_right_5",
        18: "rib_right_6",
        19: "rib_right_7",
        20: "rib_right_8",
        21: "rib_right_9",
        22: "rib_right_10",
        23: "rib_right_11",
        24: "rib_right_12",
        25: "sternum",
        26: "costal_cartilages"
    },

    "test": {1: "carpal"}
}


class_map_parts_mr = {

    "class_map_part_organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_left",
        11: "lung_right",
        12: "esophagus",
        13: "small_bowel",
        14: "duodenum",
        15: "colon",
        16: "urinary_bladder",
        17: "prostate",
        18: "sacrum",
        19: "vertebrae",
        20: "intervertebral_discs",
        21: "spinal_cord",
        22: "heart",
        23: "aorta",
        24: "inferior_vena_cava",
        25: "portal_vein_and_splenic_vein",
        26: "iliac_artery_left",
        27: "iliac_artery_right",
        28: "iliac_vena_left",
        29: "iliac_vena_right"
    },

    "class_map_part_muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "gluteus_maximus_left",
        12: "gluteus_maximus_right",
        13: "gluteus_medius_left",
        14: "gluteus_medius_right",
        15: "gluteus_minimus_left",
        16: "gluteus_minimus_right",
        17: "autochthon_left",
        18: "autochthon_right",
        19: "iliopsoas_left",
        20: "iliopsoas_right",
        21: "brain"
    }
}


class_map_parts_headneck_muscles = {

    "class_map_part_muscles_1": {
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
        11: "levator_scapulae_left"
    },

    "class_map_part_muscles_2": {
        1: "anterior_scalene_right",
        2: "anterior_scalene_left",
        3: "middle_scalene_right",
        4: "middle_scalene_left",
        5: "posterior_scalene_right",
        6: "posterior_scalene_left",
        7: "sterno_thyroid_right",
        8: "sterno_thyroid_left",
        9: "thyrohyoid_right",
        10: "thyrohyoid_left",
        11: "prevertebral_right",
        12: "prevertebral_left"
    }
}

class_map_5_parts_total_v3 = {part_name: part_map.copy() for part_name, part_map in class_map_5_parts.items()}
class_map_5_parts_total_v3["class_map_part_vertebrae"][2] = "vertebrae_L6"


map_taskid_to_partname_ct = {
    291: "class_map_part_organs",
    292: "class_map_part_vertebrae",
    293: "class_map_part_cardiac",
    294: "class_map_part_muscles",
    295: "class_map_part_ribs",

    517: "test",
}

map_taskid_to_partname_ct_v3 = {
    831: "class_map_part_organs",
    832: "class_map_part_vertebrae",
    833: "class_map_part_cardiac",
    834: "class_map_part_muscles",
    835: "class_map_part_ribs",
}

map_taskid_to_partname_mr = {
    850: "class_map_part_organs",
    851: "class_map_part_muscles"
}

map_taskid_to_partname_headneck_muscles = {
    778: "class_map_part_muscles_1",
    779: "class_map_part_muscles_2"
}
