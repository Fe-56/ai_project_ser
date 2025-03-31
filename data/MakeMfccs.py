import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tinytag import TinyTag
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
import pickle
import time
import functools
import torch

# Set parameters for parallel and batch processing with checkpoints
NUM_PROCESSES = None
BATCH_SIZE = 1000

# Set consistent audio processing parameters
TARGET_SR = 22050  # Target sample rate
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length (samples)
N_MELS = 128  # Number of mel bands
FMIN = 20  # Minimum frequency
FMAX = 8000  # Maximum frequency
POWER = 2.0  # Power for mel spectrogram (2.0 = power spectrogram)
WINDOW_TYPE = 'hann'  # Window function type
WINDOW_SIZE = 2048  # Window size (samples)
# The resolution of the mel spectrogram image (multiply by 100)
FIG_SIZE = (2.24, 2.24)
SAVE_CSV = False
FOLDER_NAME = 'mfccs'
N_MFCC = 40

# Create necessary directories
os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


# Find max audio duration to pad all audios to the same length
def find_max_duration(paths):
    max_duration = 0
    for path in tqdm(paths, desc="Finding max duration"):
        try:
            tag = TinyTag.get(path)
            max_duration = max(max_duration, tag.duration)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return max_duration


# Function to process a single file
def create_mfccs(path, target_sr, max_duration, hop_length, n_mfcc):
    try:
        # Load audio file and resample
        y, sr = librosa.load(path, sr=target_sr, res_type='kaiser_best')

        # Normalize audio
        y = librosa.util.normalize(y)

        # Compute hop length based on target_sr if not provided
        if hop_length is None:
            hop_length = int(0.01 * sr)  # 10ms hop

        # Generate MFCCs with consistent parameters
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCCs (min-max or z-score)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)

        length = int(max_duration * (target_sr / hop_length))

        # Zero-pad or truncate to length
        if mfccs.shape[1] < length:
            # Pad with zeros on the right (axis=1 is time)
            pad_width = length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :length]

        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]  # Remove extension
        mfcc_path = os.path.join(FOLDER_NAME, f"{filename}_mfcc.pt")
        mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32)
        torch.save(mfcc_tensor, mfcc_path)

        return {'path': path, 'mfcc_path': mfcc_path, 'status': 'success'}
    except Exception as e:
        error_msg = f"Error processing {path}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'path': path, 'mfcc_path': None, 'status': 'error', 'error': str(e)}


def save_checkpoint(batch_id, results, checkpoint_dir='checkpoints'):
    """Save processing results to a checkpoint file"""
    checkpoint_file = os.path.join(
        checkpoint_dir, f'batch_{batch_id}_checkpoint.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results, f)
    return checkpoint_file


def load_checkpoint(checkpoint_file):
    """Load results from a checkpoint file"""
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def get_all_checkpoint_results(checkpoint_dir='checkpoints'):
    """Load all checkpoint results"""
    all_results = []
    checkpoint_files = [f for f in os.listdir(
        checkpoint_dir) if f.endswith('_checkpoint.pkl')]

    for cf in checkpoint_files:
        try:
            results = load_checkpoint(os.path.join(checkpoint_dir, cf))
            all_results.extend(results)
            print(f"Loaded {len(results)} results from {cf}")
        except Exception as e:
            print(f"Error loading checkpoint {cf}: {e}")

    return all_results


def create_mfccs_in_batches(paths, max_duration, batch_size=BATCH_SIZE, n_processes=NUM_PROCESSES, resume=True, hop_length=HOP_LENGTH):
    """Create mfccs in batches with checkpointing"""
    if n_processes is None:
        n_processes = max(1, int(cpu_count() * 0.75))

    print(f"Using {n_processes} processes")

    # Load existing results if resuming
    processed_paths = set()
    all_results = []

    if resume:
        try:
            existing_results = get_all_checkpoint_results()
            all_results = existing_results

            # Track which paths have already been processed
            for result in existing_results:
                processed_paths.add(result['path'])

            print(
                f"Resuming from checkpoints: {len(processed_paths)} files already processed")
        except Exception as e:
            print(f"Error loading checkpoints: {e}. Starting fresh.")
            all_results = []

    # Filter out already processed paths
    paths_to_process = [p for p in paths if p not in processed_paths]
    print(f"{len(paths_to_process)} files left to process")

    # Create a partial function with the fixed parameters
    process_func = functools.partial(
        create_mfccs,
        target_sr=TARGET_SR,
        max_duration=max_duration,
        hop_length=hop_length,
        n_mfcc=N_MFCC
    )

    # Process in batches
    for batch_idx, i in enumerate(range(0, len(paths_to_process), batch_size)):
        batch_paths = paths_to_process[i:i+batch_size]
        if not batch_paths:
            continue

        batch_id = int(time.time())  # Use timestamp as batch ID
        print(f"Processing batch {batch_idx+1}/{(len(paths_to_process)-1)//batch_size + 1} "
              f"({len(batch_paths)} files)")

        # Process batch in parallel
        with Pool(processes=n_processes, maxtasksperchild=100) as pool:
            batch_results = list(tqdm(
                pool.imap(process_func, batch_paths),
                total=len(batch_paths),
                desc=f"Batch {batch_idx+1}"
            ))

        # Save batch results to checkpoint
        save_checkpoint(batch_id, batch_results)
        all_results.extend(batch_results)

        # Print batch summary
        success_count = sum(
            1 for r in batch_results if r['status'] == 'success')
        error_count = len(batch_results) - success_count
        print(
            f"Batch {batch_idx+1} complete: {success_count} succeeded, {error_count} failed")

    return all_results


def main():
    # Print mfcc parameters
    print("\nMFCC Parameters:")
    print(f"Sample Rate: {TARGET_SR} Hz")
    print(f"FFT Window Size: {N_FFT}")
    print(
        f"Hop Length: {HOP_LENGTH} samples ({HOP_LENGTH/TARGET_SR*1000:.1f} ms)")
    print(f"Window Type: {WINDOW_TYPE}")
    print(f"Mel Bands: {N_MELS}")
    print(f"Frequency Range: {FMIN} Hz - {FMAX} Hz")
    print(f"Power: {POWER}")
    print("")

    # Load datasets
    print("Loading datasets...")
    train = pd.read_csv('train_dataset.csv')
    val = pd.read_csv('val_dataset.csv')
    test = pd.read_csv('test_dataset.csv')
    train = train[['Filepath', 'Emotion']]
    val = val[['Filepath', 'Emotion']]
    test = test[['Filepath', 'Emotion']]

    # Fix file paths if needed
    train['Filepath'] = train['Filepath'].str.replace('\\', '/')
    test['Filepath'] = test['Filepath'].str.replace('\\', '/')
    val['Filepath'] = val['Filepath'].str.replace('\\', '/')

    all_paths = pd.concat(
        [train['Filepath'], val['Filepath'], test['Filepath']], ignore_index=True)
    max_duration = find_max_duration(all_paths)
    print(f"Maximum duration of audio: {max_duration}s")

    # Generate melspectrograms for the train set
    # Get the list of paths to process
    train_paths = train['Filepath'].tolist()

    # Process the training data
    print(f"Processing {len(train_paths)} training files...")
    results = create_mfccs_in_batches(
        paths=train_paths,
        max_duration=max_duration,
        batch_size=BATCH_SIZE,  # Adjust based on your dataset size and memory constraints
        n_processes=NUM_PROCESSES,  # Will use 75% of available cores by default
        resume=True,  # Set to False to start fresh and ignore checkpoints
    )

    if SAVE_CSV:
        # Create a mapping from paths to spectrograms
        path_to_spec = {r['path']: r['mfcc_path'] for r in results}

        # Update the dataframes
        train['MfccPath'] = train['Filepath'].map(path_to_spec)

        # Save the updated dataframes
        train = train[['Filepath', 'MfccPath', 'Emotion']]
        train.to_csv('mfcc_train_dataset.csv', index=False)

    # Calculate and print statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print("\nProcessing complete!")
    print(f"Total files processed: {len(results)}")
    print(
        f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Failed: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"Results saved to mfcc_train_dataset.csv")

    # Generate melspectrograms for the val set
    val_paths = val['Filepath'].tolist()
    print(f"Processing {len(val_paths)} test files...")
    val_results = create_mfccs_in_batches(
        paths=val_paths,
        max_duration=max_duration,
        batch_size=BATCH_SIZE,  # Adjust based on your dataset size and memory constraints
        n_processes=NUM_PROCESSES,  # Will use 75% of available cores by default
        resume=True,  # Set to False to start fresh and ignore checkpoints
    )

    if SAVE_CSV:
        # Create a mapping from paths to spectrograms
        val_path_to_spec = {r['path']: r['mfcc_path']
                            for r in val_results}

        # Update the test dataframe
        val['MfccPath'] = val['Filepath'].map(val_path_to_spec)

        # Save the updated test dataframe
        val = val[['Filepath', 'MfccPath', 'Emotion']]
        val.to_csv('mfcc_val_dataset.csv', index=False)

    # Calculate and print statistics
    val_success_count = sum(1 for r in val_results if r['status'] == 'success')
    val_error_count = sum(1 for r in val_results if r['status'] == 'error')

    print("\nTest processing complete!")
    print(f"Total test files processed: {len(val_results)}")
    print(
        f"Successful: {val_success_count} ({val_success_count/len(val_results)*100:.1f}%)")
    print(
        f"Failed: {val_error_count} ({val_error_count/len(val_results)*100:.1f}%)")
    print(f"Results saved to mfcc_val_dataset.csv")

    # Generate melspectrograms for the test set
    test_paths = test['Filepath'].tolist()
    print(f"Processing {len(test_paths)} test files...")
    test_results = create_mfccs_in_batches(
        paths=test_paths,
        max_duration=max_duration,
        batch_size=BATCH_SIZE,  # Adjust based on your dataset size and memory constraints
        n_processes=NUM_PROCESSES,  # Will use 75% of available cores by default
        resume=True,  # Set to False to start fresh and ignore checkpoints
    )

    if SAVE_CSV:
        # Create a mapping from paths to spectrograms
        test_path_to_spec = {r['path']: r['mfcc_path']
                             for r in test_results}

        # Update the test dataframe
        test['MfccPath'] = test['Filepath'].map(test_path_to_spec)

        # Save the updated test dataframe
        test = test[['Filepath', 'MfccPath', 'Emotion']]
        test.to_csv('mfcc_test_dataset.csv', index=False)

    # Calculate and print statistics
    test_success_count = sum(
        1 for r in test_results if r['status'] == 'success')
    test_error_count = sum(1 for r in test_results if r['status'] == 'error')

    print("\nTest processing complete!")
    print(f"Total test files processed: {len(test_results)}")
    print(
        f"Successful: {test_success_count} ({test_success_count/len(test_results)*100:.1f}%)")
    print(
        f"Failed: {test_error_count} ({test_error_count/len(test_results)*100:.1f}%)")
    print(f"Results saved to mfcc_test_dataset.csv")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
