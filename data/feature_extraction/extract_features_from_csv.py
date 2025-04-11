import pandas as pd
import os
from extract_features import OpenSMILEFeatureExtractor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_audio(row_dict, audio_column, base_data_path, output_dir):
    """
    Worker function to process a single audio file
    """
    try:
        row_id = row_dict['Id']
        audio_rel_path = row_dict[audio_column][2:].replace('\\', '/')
        audio_file = os.path.join(base_data_path, audio_rel_path)
        output_file = os.path.join(output_dir, f"{row_id}_features.csv")
        temp_output_file = os.path.join(output_dir, f"temp_{row_id}_features.csv")

        if not os.path.exists(audio_file):
            return f"Warning: Audio file not found: {audio_file}"

        if os.path.exists(output_file):
            return f"Skipping already processed file: {audio_file}"

        # Local instance per process to avoid shared state issues
        extractor = OpenSMILEFeatureExtractor()

        feature_file = extractor.extract_features(audio_file, temp_output_file)

        if feature_file:
            feature_df = pd.read_csv(feature_file, sep=';')
            feature_df[audio_column] = row_dict[audio_column]
            feature_df.drop('name', axis=1, inplace=True)
            feature_df.to_csv(output_file, index=False)
            os.remove(temp_output_file)
        
        return f"Processed {audio_file}"
    except Exception as e:
        return f"Error processing {row_dict[audio_column]}: {e}"

def extract_features_from_csv(csv_path, audio_column, output_dir=None):
    df = pd.read_csv(csv_path)
    
    if audio_column not in df.columns:
        raise ValueError(f"Column '{audio_column}' not found in CSV file")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'features')
    os.makedirs(output_dir, exist_ok=True)

    base_data_path = '/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/data'

    # Convert each row to a dictionary
    rows = df.to_dict(orient='records')

    # Setup multiprocessing pool
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(process_audio, row, audio_column, base_data_path, output_dir)
            for row in rows
        ]

        for future in tqdm(futures, desc="Extracting features", total=len(futures)):
            msg = future.result()
            if msg:
                print(msg)

def main():
    csv_path = "/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/data/speech_dataset.csv"
    audio_column = "Filepath"
    output_dir = os.path.join(os.path.dirname(csv_path), 'temp-features')

    try:
        print(f"Starting feature extraction from {csv_path}")
        extract_features_from_csv(csv_path, audio_column, output_dir)
        print("Successfully processed all files")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
