# # inverse mapping
# TOTALSEG_NAME2ID = {name: idx for idx, name in TOTALSEG_ID2NAME.items()}
TS2GROUP = {
    1: 13,  # spleen -> Spleen
    2: 8,   # kidney_right -> Kidney
    3: 8,   # kidney_left -> Kidney
    4: 11,  # gallbladder -> Gall bladder
    5: 10,  # liver -> Liver
    6: 9,   # stomach -> Stomach
    7: 12,  # pancreas -> Pancreas
    8: 7,   # adrenal_gland_right -> Adrenal gland
    9: 7,   # adrenal_gland_left -> Adrenal gland

    10: 5,  # lung_upper_lobe_left -> Lung
    11: 5,  # lung_lower_lobe_left -> Lung
    12: 5,  # lung_upper_lobe_right -> Lung
    13: 5,  # lung_middle_lobe_right -> Lung
    14: 5,  # lung_lower_lobe_right -> Lung

    15: 3,  # esophagus -> Esophagus
    16: 4,  # trachea -> Trachea
    17: 0,  # thyroid_gland -> (no group)

    18: 15, # small_bowel -> Small bowel
    19: 15, # duodenum -> Small bowel
    20: 14, # colon -> Colon
    21: 16, # urinary_bladder -> Urinary bladder
    22: 0,  # prostate -> (no group)

    # kidney cysts grouped into Kidney
    23: 8,  # kidney_cyst_left -> Kidney (assumption)
    24: 8,  # kidney_cyst_right -> Kidney (assumption)

    25: 32, # sacrum -> Sacrum
    26: 32, # vertebrae_S1 -> Sacrum (assumption; sacral)
    27: 23, # vertebrae_L5 -> Lumbar vertebrae (assumption: include L5)
    28: 23, # vertebrae_L4 -> Lumbar vertebrae
    29: 23, # vertebrae_L3 -> Lumbar vertebrae
    30: 23, # vertebrae_L2 -> Lumbar vertebrae
    31: 23, # vertebrae_L1 -> Lumbar vertebrae

    # Thoracic vertebrae T12–T1
    32: 24, # vertebrae_T12 -> Thoracic vertebrae
    33: 24, # T11
    34: 24, # T10
    35: 24, # T9
    36: 24, # T8
    37: 24, # T7
    38: 24, # T6
    39: 24, # T5
    40: 24, # T4
    41: 24, # T3
    42: 24, # T2
    43: 24, # T1

    # Cervical vertebrae C7–C1
    44: 25, # vertebrae_C7 -> Cervical vertebrae
    45: 25, # C6
    46: 25, # C5
    47: 25, # C4
    48: 25, # C3
    49: 25, # C2
    50: 25, # C1

    51: 6,  # heart -> Heart
    52: 17, # aorta -> Aorta

    # No pulmonary_artery label; use Pulmonary vessels group
    53: 20, # pulmonary_vein -> Pulmonary artery group (assumption: pulmonary vessels)

    # brachiocephalic trunk is an aortic branch; group with Aorta
    54: 17, # brachiocephalic_trunk -> Aorta (assumption)

    55: 0,  # subclavian_artery_right -> (no group)
    56: 0,  # subclavian_artery_left -> (no group)
    57: 0,  # common_carotid_artery_right -> (no group)
    58: 0,  # common_carotid_artery_left -> (no group)
    59: 0,  # brachiocephalic_vein_left -> (no group)
    60: 0,  # brachiocephalic_vein_right -> (no group)

    # appendage is part of heart
    61: 6,  # atrial_appendage_left -> Heart (assumption)
    62: 0,  # superior_vena_cava -> (no group)

    63: 18, # inferior_vena_cava -> Inferior vena cava
    64: 19, # portal_vein_and_splenic_vein -> Portal vein and splenic vein

    65: 21, # iliac_artery_left -> Iliac artery
    66: 21, # iliac_artery_right -> Iliac artery
    67: 22, # iliac_vena_left -> Iliac vena
    68: 22, # iliac_vena_right -> Iliac vena

    69: 27, # humerus_left -> Humerus
    70: 27, # humerus_right -> Humerus

    71: 28, # scapula_left -> Scapula
    72: 28, # scapula_right -> Scapula

    73: 29, # clavicula_left -> Clavicula
    74: 29, # clavicula_right -> Clavicula

    75: 30, # femur_left -> Femur
    76: 30, # femur_right -> Femur

    77: 31, # hip_left -> Hip
    78: 31, # hip_right -> Hip

    79: 0,  # spinal_cord -> (no group)

    80: 33, # gluteus_maximus_left -> Gluteus
    81: 33, # gluteus_maximus_right -> Gluteus
    82: 33, # gluteus_medius_left -> Gluteus
    83: 33, # gluteus_medius_right -> Gluteus
    84: 33, # gluteus_minimus_left -> Gluteus
    85: 33, # gluteus_minimus_right -> Gluteus

    86: 35, # autochthon_left -> Autochthon
    87: 35, # autochthon_right -> Autochthon

    88: 34, # iliopsoas_left -> Iliopsoas
    89: 34, # iliopsoas_right -> Iliopsoas

    90: 2,  # brain -> Brain

    # no explicit "face" label; use skull as Face
    91: 1,  # skull -> Face (assumption)

    # ribs
    92: 26, # rib_left_1 -> Rib
    93: 26,
    94: 26,
    95: 26,
    96: 26,
    97: 26,
    98: 26,
    99: 26,
    100: 26,
    101: 26,
    102: 26,
    103: 26,
    104: 26, # rib_right_1 -> Rib
    105: 26,
    106: 26,
    107: 26,
    108: 26,
    109: 26,
    110: 26,
    111: 26,
    112: 26,
    113: 26,
    114: 26,
    115: 26,

    # I also group sternum / cartilages into Rib (chest wall) for simplicity
    116: 26, # sternum -> Rib group (assumption)
    117: 26, # costal_cartilages -> Rib group (assumption)
}


GROUPIDX2NAME = {
    1:  "Face",
    2:  "Brain",
    3:  "Esophagus",
    4:  "Trachea",
    5:  "Lung",
    6:  "Heart",
    7:  "Adrenal gland",
    8:  "Kidney",
    9:  "Stomach",
    10: "Liver",
    11: "Gall bladder",
    12: "Pancreas",
    13: "Spleen",
    14: "Colon",
    15: "Small bowel",
    16: "Urinary bladder",
    17: "Aorta",
    18: "Inferior vena cava",
    19: "Portal vein and splenic vein",
    20: "Pulmonary artery",
    21: "Iliac artery",
    22: "Iliac vena",
    23: "Lumbar vertebrae",
    24: "Thoracic vertebrae",
    25: "Cervical vertebrae",
    26: "Rib",
    27: "Humerus",
    28: "Scapula",
    29: "Clavicula",
    30: "Femur",
    31: "Hip",
    32: "Sacrum",
    33: "Gluteus",
    34: "Iliopsoas",
    35: "Autochthon",
}
