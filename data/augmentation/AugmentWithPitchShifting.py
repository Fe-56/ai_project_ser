import os
import pandas as pd
import numpy as np
import librosa
from tinytag import TinyTag
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
import pickle
import time
import functools
from OfflineDataAugmentation import OfflineDataAugmentation
import soundfile as sf

# Set parameters for parallel and batch processing with checkpoints
NUM_PROCESSES = None
BATCH_SIZE = 1000

SAVE_CSV = True
FOLDER_NAME = 'augmented_pitch'
N_STEPS_RANGE = (-2, 2)

# Create necessary directories
os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Function to process a single file


def augment_speech(path):
    try:
        # Load audio file and resample
        y, sr = librosa.load(path)

        # Normalize audio
        y = librosa.util.normalize(y)

        y = OfflineDataAugmentation.pitch_shift(
            y, sr=sr, n_steps_range=N_STEPS_RANGE)

        filename = os.path.basename(path)
        if filename.endswith('.wav'):
            filename = filename.replace('.wav', '_pitch.wav')
        elif filename.endswith('.mp4'):
            filename = filename.replace('.mp4', '_pitch.wav')

        pitch_path = os.path.join(FOLDER_NAME, filename)
        sf.write(pitch_path, y, sr)

        return {'path': path, 'pitch_path': pitch_path, 'status': 'success'}
    except Exception as e:
        error_msg = f"Error processing {path}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {'path': path, 'pitch_path': None, 'status': 'error', 'error': str(e)}


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


def augment_speech_in_batches(paths, batch_size=BATCH_SIZE, n_processes=NUM_PROCESSES, resume=True):
    """Create melspectrograms in batches with checkpointing"""
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
        augment_speech
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
    # Load datasets
    print("Loading train dataset...")
    train = pd.read_csv('../train_dataset.csv')
    train = train[['Filepath', 'Emotion']]

    # Fix file paths if needed
    train['Filepath'] = train['Filepath'].str.replace(
        '\\', '/').str.replace('./', '../')

    # Get the list of paths to process
    train_paths = train['Filepath'].tolist()

    # Process the training data
    print(f"Processing {len(train_paths)} training files...")
    results = augment_speech_in_batches(
        paths=train_paths,
        batch_size=BATCH_SIZE,  # Adjust based on your dataset size and memory constraints
        n_processes=NUM_PROCESSES,  # Will use 75% of available cores by default
        resume=True,  # Set to False to start fresh and ignore checkpoints
    )

    if SAVE_CSV:
        # Create a mapping from paths to time-stretched speech
        path_to_pitch = {r['path']: r['pitch_path'] for r in results}

        # Update the dataframes
        train['Pitchpath'] = train['Filepath'].map(path_to_pitch)

        # Save the updated dataframes
        train = train[['Filepath', 'Pitchpath', 'Emotion']]
        train.to_csv('augmented_pitch_train_dataset.csv', index=False)

    # Calculate and print statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print("\nProcessing complete!")
    print(f"Total files processed: {len(results)}")
    print(
        f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Failed: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"Results saved to augmented_pitch_train_dataset.csv")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
