import pandas as pd
import os
from extract_features import OpenSMILEFeatureExtractor
from tqdm import tqdm

def extract_features_from_csv(csv_path, audio_column, output_dir=None):
    """
    Extract features from audio files listed in a CSV and combine them into a single CSV file
    
    Args:
        csv_path (str): Path to the CSV file containing audio file paths
        audio_column (str): Name of the column containing audio file paths
        output_dir (str, optional): Directory to save feature files. If None, will use the same directory as the CSV
        
    Returns:
        str: Path to the combined feature CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Validate that the audio column exists
    if audio_column not in df.columns:
        raise ValueError(f"Column '{audio_column}' not found in CSV file")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the feature extractor
    extractor = OpenSMILEFeatureExtractor()
    
    # Process each audio file
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_file = os.path.join('/teamspace/studios/this_studio/AI-Project--Speech-Emotion-Recognition/data', row[audio_column][2:].replace('\\','/'))
        
        # Skip if the audio file doesn't exist
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
            continue
            
        # Generate temporary output filename
        temp_output_file = os.path.join(output_dir, f"temp_{row['Id']}_features.csv")
        output_file = os.path.join(output_dir, f"{row['Id']}_features.csv")

        if os.path.exists(output_file):
            print(f"Skipping audio file as it already exists: {audio_file}")
            continue
        
        # Extract features
        feature_file = extractor.extract_features(audio_file, temp_output_file)
        
        if feature_file:
            # Read the feature file into a dataframe with semicolon separator
            feature_df = pd.read_csv(feature_file, sep=';')
            # Add audio path column
            feature_df[audio_column] = row[audio_column]
            feature_df.drop('name', axis=1, inplace=True)
            feature_df.to_csv(output_file)
            # Remove temporary file
            os.remove(temp_output_file)
    
    # Combine all feature dataframes
    # combined_df = pd.concat(feature_dfs, ignore_index=True)
    
    # # Save combined features to a single CSV file
    # output_file = os.path.join(output_dir, "combined_train_features.csv")
    # combined_df.to_csv(output_file, index=False)
    
    # return output_file

def main():
    # Set up paths
    csv_path = f"/teamspace/studios/this_studio/AI-Project--Speech-Emotion-Recognition/data/speech_dataset.csv"
    audio_column = "Filepath"
    output_dir = os.path.join(os.path.dirname(csv_path), 'temp-features')
    
    try:
        print(f"Starting feature extraction from {csv_path}")
        print(f"Looking for audio files in column: {audio_column}")
        
        extract_features_from_csv(csv_path, audio_column, output_dir)
        
        print(f"\nSuccessfully processed all files")
        # print("Combined feature file saved as:", output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 