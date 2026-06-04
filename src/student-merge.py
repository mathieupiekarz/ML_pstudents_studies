#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "data_global.csv"

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

# Same column order as R merge() with suffixes .m / .p
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
    "guardian.m",
    "traveltime.m",
    "studytime.m",
    "failures.m",
    "schoolsup.m",
    "famsup.m",
    "paid.m",
    "activities.m",
    "higher.m",
    "romantic.m",
    "famrel.m",
    "freetime.m",
    "goout.m",
    "Dalc.m",
    "Walc.m",
    "health.m",
    "absences.m",
    "G1.m",
    "G2.m",
    "G3.m",
    "guardian.p",
    "traveltime.p",
    "studytime.p",
    "failures.p",
    "schoolsup.p",
    "famsup.p",
    "paid.p",
    "activities.p",
    "higher.p",
    "romantic.p",
    "famrel.p",
    "freetime.p",
    "goout.p",
    "Dalc.p",
    "Walc.p",
    "health.p",
    "absences.p",
    "G1.p",
    "G2.p",
    "G3.p",
]


def main() -> None:
    d1 = pd.read_csv(DATA_DIR / "student-mat.csv", sep=",")
    d2 = pd.read_csv(DATA_DIR / "student-por.csv", sep=",")
    d3 = pd.merge(d1, d2, on=MERGE_KEYS, suffixes=(".m", ".p"))
    d3 = d3[R_COLUMNS]
    d3.to_csv(OUTPUT_CSV, index=False, sep=",")
    print(len(d3))  # 382 students


if __name__ == "__main__":
    main()
