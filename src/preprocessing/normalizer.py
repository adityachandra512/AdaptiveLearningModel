import pandas as pd
import numpy as np
import os

PROCESSED_DATA_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/data/processed"

class Normalizer:
    """
    Applies normalization on numerical columns.
    Supports Min-Max and Z-score normalization.
    """

    def __init__(self, method="minmax"):
        self.method = method
        self.stats = {}  # store min,max or mean,std for inference

    def fit(self, df, numeric_cols):
        """Compute stats needed for normalization."""
        self.stats = {}

        for col in numeric_cols:
            if self.method == "minmax":
                self.stats[col] = {
                    "min": df[col].min(),
                    "max": df[col].max()
                }

            elif self.method == "zscore":
                self.stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std()
                }

    def transform(self, df):
        """Apply normalization using computed stats."""
        df_norm = df.copy()

        for col, values in self.stats.items():

            if self.method == "minmax":
                df_norm[col] = (df[col] - values["min"]) / (values["max"] - values["min"] + 1e-9)

            elif self.method == "zscore":
                df_norm[col] = (df[col] - values["mean"]) / (values["std"] + 1e-9)

        return df_norm

    def fit_transform(self, df, numeric_cols):
        """Shortcut for training + transforming."""
        self.fit(df, numeric_cols)
        return self.transform(df)


def normalize_dataset(df=None, input_file="student_merged.csv", method="minmax", save=True):
    """
    Load processed dataset and normalize numerical columns.
    Saves output as: data/processed/student_normalized.csv
    
    Args:
        df: Optional dataframe to normalize. If None, loads from input_file.
        input_file: CSV file to load if df is None.
        method: Normalization method ("minmax" or "zscore").
        save: Whether to save the result to file.
    
    Returns:
        Normalized data as numpy array.
    """
    if df is None:
        path = os.path.join(PROCESSED_DATA_DIR, input_file)
        df = pd.read_csv(path)

    # select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("Numeric columns:", len(numeric_cols))

    normalizer = Normalizer(method=method)
    df_norm = normalizer.fit_transform(df, numeric_cols)

    if save:
        # save output
        out_path = os.path.join(PROCESSED_DATA_DIR, "student_normalized.csv")
        df_norm.to_csv(out_path, index=False)
        print("Normalized dataset saved to:", out_path)

    return df_norm.values
