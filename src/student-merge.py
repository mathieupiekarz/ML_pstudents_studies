#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MERGE_KEYS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]

# Same column order as R merge() with suffixes .x / .y
R_COLUMNS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
    "guardian.x",
    "traveltime.x",
    "studytime.x",
    "failures.x",
    "schoolsup.x",
    "famsup.x",
    "paid.x",
    "activities.x",
    "higher.x",
    "romantic.x",
    "famrel.x",
    "freetime.x",
    "goout.x",
    "Dalc.x",
    "Walc.x",
    "health.x",
    "absences.x",
    "G1.x",
    "G2.x",
    "G3.x",
    "guardian.y",
    "traveltime.y",
    "studytime.y",
    "failures.y",
    "schoolsup.y",
    "famsup.y",
    "paid.y",
    "activities.y",
    "higher.y",
    "romantic.y",
    "famrel.y",
    "freetime.y",
    "goout.y",
    "Dalc.y",
    "Walc.y",
    "health.y",
    "absences.y",
    "G1.y",
    "G2.y",
    "G3.y",
]


def main() -> None:
    d1 = pd.read_csv(DATA_DIR / "student-mat.csv", sep=",")
    d2 = pd.read_csv(DATA_DIR / "student-por.csv", sep=",")
    d3 = pd.merge(d1, d2, on=MERGE_KEYS, suffixes=(".x", ".y"))
    d3 = d3[R_COLUMNS]
    print(len(d3))  # 382 students


if __name__ == "__main__":
    main()
