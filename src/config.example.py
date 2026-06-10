"""
config.example.py — Template de configuration (copier vers config.py).

    cp src/config.example.py src/config.py
"""

from typing import Literal

DATASET: Literal["student-mat", "student-por", "data_global"] = "data_global"

DATASET_PATHS = {
    "student-mat": "student-mat.csv",
    "student-por": "student-por.csv",
    "data_global": "data_global.csv",
}

SHARED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet",
]

SUBJECT_COLUMNS = [
    "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "higher", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2", "G3",
]

INCLUDE_VARS: dict[str, bool] = {
    "school": True, "sex": True, "age": True, "address": True,
    "famsize": True, "Pstatus": True, "Medu": True, "Fedu": True,
    "Mjob": True, "Fjob": True, "reason": True, "nursery": True,
    "internet": True,
    "guardian": True, "traveltime": True, "studytime": True,
    "failures": True, "schoolsup": True, "famsup": True,
    "paid": True, "activities": True, "higher": True,
    "romantic": True, "famrel": True, "freetime": True,
    "goout": True, "Dalc": True, "Walc": True, "health": True,
    "absences": True, "G1": False, "G2": False, "G3": False,
}

FOCUS_VAR = "G3"
DROPOUT_VALUE = 0
DROPOUT_LABELS = {True: "Décrochage (G3=0)", False: "Non-décrochage (G3>0)"}

TARGET_VARS = ["G3"]
TARGET_TYPES: dict[str, str] = {}
ACTIONABLE_VARS = [
    "studytime", "schoolsup", "paid", "activities",
    "goout", "Dalc", "Walc", "absences", "internet", "higher",
]

AVERAGE_MAT_POR = False

PAIRS_TO_AVERAGE = [
    ("Dalc.m", "Dalc.p", "Dalc"), ("Walc.m", "Walc.p", "Walc"),
    ("goout.m", "goout.p", "goout"), ("freetime.m", "freetime.p", "freetime"),
    ("studytime.m", "studytime.p", "studytime"), ("failures.m", "failures.p", "failures"),
    ("absences.m", "absences.p", "absences"), ("traveltime.m", "traveltime.p", "traveltime"),
    ("famrel.m", "famrel.p", "famrel"), ("health.m", "health.p", "health"),
    ("G1.m", "G1.p", "G1"), ("G2.m", "G2.p", "G2"), ("G3.m", "G3.p", "G3"),
]

# Importer les helpers depuis config.py une fois copié, ou dupliquer :
# from config import apply_preprocessing, resolve_columns, results_suffix, ...
