import pandas as pd
import numpy as np
import os

PROCESSED_DATA_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/data/processed"

class Binarizer:
    """
    Converts dataset to binary (0/1) format.
    Required for Hebbian learning + ART1 network.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.one_hot_cols = []  # Keep track of expanded columns

    def numeric_to_binary(self, df, numeric_cols):
        """Convert numeric values to 0/1 using threshold."""
        df_bin = df.copy()

        for col in numeric_cols:
            df_bin[col] = (df[col] >= self.threshold).astype(int)

        return df_bin

    def categorical_to_onehot(self, df, categorical_cols):
        """Convert categorical features to one-hot binary features."""
        df_bin = df.copy()
        df_bin = pd.get_dummies(df_bin, columns=categorical_cols, prefix_sep="_")

        # store new one-hot columns (convert column names to strings to handle integer column names)
        self.one_hot_cols = [c for c in df_bin.columns if any(str(prefix) in str(c) for prefix in categorical_cols)]

        return df_bin

    def binarize(self, df):
        """Automatic binarization pipeline."""

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df_1 = self.numeric_to_binary(df, numeric_cols)
        df_2 = self.categorical_to_onehot(df_1, categorical_cols)

        return df_2


def generate_binary_dataset(data=None, input_file="student_normalized.csv", threshold=0.5, save=True):
    """
    Convert normalized dataset into binary format.
    Saves output as data/processed/student_binary.csv
    
    Args:
        data: Optional numpy array or dataframe to binarize. If None, loads from input_file.
        input_file: CSV file to load if data is None.
        threshold: Threshold for binarization (values >= threshold become 1).
        save: Whether to save the result to file.
    
    Returns:
        Binary data as numpy array.
    """
    if data is None:
        path = os.path.join(PROCESSED_DATA_DIR, input_file)
        df = pd.read_csv(path)
    elif isinstance(data, np.ndarray):
        # Convert numpy array to dataframe for binarizer
        df = pd.DataFrame(data)
    else:
        df = data

    binarizer = Binarizer(threshold=threshold)
    df_binary = binarizer.binarize(df)

    if save:
        out_path = os.path.join(PROCESSED_DATA_DIR, "student_binary.csv")
        df_binary.to_csv(out_path, index=False)
        print("Binary dataset saved to:", out_path)

    return df_binary.values
