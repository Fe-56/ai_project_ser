import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def combine_feature_files_streaming(features_dir, output_file):
    csv_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]

    if not csv_files:
        raise ValueError(f"No feature CSV files found in {features_dir}")

    print(f"Found {len(csv_files)} feature files to combine")

    file_paths = [os.path.join(features_dir, f) for f in csv_files]
    first_file = True

    for file_path in file_paths:
        try:
            for chunk in pd.read_csv(file_path, chunksize=100_000):
                chunk.to_csv(output_file, mode='a', index=False, header=first_file)
                first_file = False
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"\nSuccessfully combined files into: {output_file}")

def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def combine_feature_files(features_dir, output_file):
    csv_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]

    if not csv_files:
        raise ValueError(f"No feature CSV files found in {features_dir}")
    
    print(f"Found {len(csv_files)} feature files to combine")
    
    file_paths = [os.path.join(features_dir, f) for f in csv_files]
    dfs = []

    # Use a thread pool to read files concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_csv_file, fp): fp for fp in file_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSV files"):
            df = future.result()
            if df is not None:
                dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined dataframe
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined {len(dfs)} files into: {output_file}")
    print(f"Total rows in combined file: {len(combined_df)}")

def main():
    features_dir = "/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/data/temp-features"
    output_file = os.path.join("/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/data", "combined_features.csv")
    
    try:
        # combine_feature_files(features_dir, output_file)
        combine_feature_files_streaming(features_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
