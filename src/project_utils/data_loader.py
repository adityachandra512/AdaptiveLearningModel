import pandas as pd
import os

RAW_DATA_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/data/raw"
PROCESSED_DATA_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/data/processed"

def load_raw_student_data():
    """Loads and merges student-mat and student-por datasets."""

    mat_path = os.path.join(RAW_DATA_DIR, "student-mat.csv")
    por_path = os.path.join(RAW_DATA_DIR, "student-por.csv")

    d1 = pd.read_csv(mat_path, sep=";")
    d2 = pd.read_csv(por_path, sep=";")

    merge_cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
    ]

    d3 = pd.merge(d1, d2, on=merge_cols)

    print("Merged Dataset Size:", len(d3))

    return d3
