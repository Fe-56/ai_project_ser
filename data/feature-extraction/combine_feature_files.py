import pandas as pd
import os
from tqdm import tqdm

def combine_feature_files(features_dir, output_file):
    """
    Combine all feature CSV files from a directory into a single CSV file
    
    Args:
        features_dir (str): Directory containing the feature CSV files
        output_file (str): Path to save the combined CSV file
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]
    
    if not csv_files:
        raise ValueError(f"No feature CSV files found in {features_dir}")
    
    print(f"Found {len(csv_files)} feature files to combine")
    
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Read each CSV file and append to the list
    for csv_file in tqdm(csv_files, desc="Combining feature files"):
        file_path = os.path.join(features_dir, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined dataframe
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined {len(csv_files)} files into: {output_file}")
    print(f"Total rows in combined file: {len(combined_df)}")

def main():
    # Set up paths
    features_dir = "/teamspace/studios/this_studio/AI-Project--Speech-Emotion-Recognition/data/temp-features"
    output_file = os.path.join("/teamspace/studios/this_studio/AI-Project--Speech-Emotion-Recognition/data", "combined_features.csv")
    
    try:
        combine_feature_files(features_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 