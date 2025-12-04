import os
import pandas as pd
from src.project_utils.data_loader import load_raw_student_data

PROCESSED_DATA_DIR = "/home/dgxuser16/NTL/norman/Aditya/AdaptiveLearningSystem/data/processed"

def save_processed_dataset():
    data = load_raw_student_data()

    # Handle categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        
        data = data.drop(columns=categorical_cols)
        data = pd.concat([data, encoded_df], axis=1)

    output_path = os.path.join(PROCESSED_DATA_DIR, "student_merged.csv")
    data.to_csv(output_path, index=False)

    print("Processed dataset saved to:", output_path)
