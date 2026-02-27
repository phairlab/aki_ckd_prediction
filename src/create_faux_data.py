
"""
Creates scrambled (fake) data by shuffling each column independently.
This destroys relationships between columns while preserving data types and value distributions.

Run this script ON the secure server.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input paths (real data)
COHORT_PATH = "/data/kidney/Sacha/newdata/cohort.csv"
IN_HOSP_LABS_PATH = "/data/kidney/Hing/in-hosp labs.csv"
IN_HOSP_VARS_PATH = "/data/kidney/Hing/in-hosp vars.csv"

# Output directory for fake data
OUTPUT_DIR = Path("/data/kidney/Sacha/aki_ckd_prediction/nonsense_data")


def scramble_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scramble each column independently - shuffles values within each column
    so that row relationships are destroyed but data types and distributions preserved.
    """
    scrambled = df.copy()
    
    for col in scrambled.columns:
        # Get non-null indices
        non_null_mask = scrambled[col].notna()
        non_null_values = scrambled.loc[non_null_mask, col].values.copy()
        
        # Shuffle the non-null values
        np.random.shuffle(non_null_values)
        
        # Put shuffled values back
        scrambled.loc[non_null_mask, col] = non_null_values
    
    return scrambled


def create_faux_datasets():
    """
    Load real datasets, scramble each column, and save them.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "cohort.csv": COHORT_PATH,
        "in-hosp_labs.csv": IN_HOSP_LABS_PATH,
        "in-hosp_vars.csv": IN_HOSP_VARS_PATH,
    }
    
    for output_name, input_path in datasets.items():
        print(f"\nProcessing: {input_path}")
        
        if not Path(input_path).exists():
            print(f"  WARNING: File not found, skipping.")
            continue
        
        try:
            df = pd.read_csv(input_path)
            print(f"  Original shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Limit to max 1000 rows
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42).reset_index(drop=True)
                print(f"  Sampled to: {df.shape}")
            
            scrambled_df = scramble_dataframe(df)
            
            output_path = OUTPUT_DIR / output_name
            scrambled_df.to_csv(output_path, index=False)
            print(f"  Saved scrambled data to: {output_path}")
            
        except Exception as e:
            print(f"  ERROR processing file: {e}")
    
    print(f"\n{'='*50}")
    print(f"Scrambled data saved to: {OUTPUT_DIR}")
    print("These files can be safely transferred off the secure server.")


if __name__ == "__main__":
    create_faux_datasets()