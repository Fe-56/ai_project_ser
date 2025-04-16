import pandas as pd
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Read augmented datasets
noise_df = pd.read_csv('augmented_noise_train_dataset.csv')
pitch_df = pd.read_csv('augmented_pitch_train_dataset.csv')
time_df = pd.read_csv('augmented_time_train_dataset.csv')

# Read original dataset from parent directory
original_df = pd.read_csv('../train_dataset.csv')

# Fix file paths if needed
original_df['Filepath'] = original_df['Filepath'].str.replace(
    '\\', '/').str.replace('./', '../')

# Combine all datasets, keeping only Filepath and Emotion columns
noise_df = noise_df.rename(columns={'Filepath': 'Originalpath'})
noise_df = noise_df.rename(columns={'Noisepath': 'Filepath'})
pitch_df = pitch_df.rename(columns={'Filepath': 'Originalpath'})
pitch_df = pitch_df.rename(columns={'Pitchpath': 'Filepath'})
time_df = time_df.rename(columns={'Filepath': 'Originalpath'})
time_df = time_df.rename(columns={'Timepath': 'Filepath'})
combined_df = pd.concat([
    original_df[['Filepath', 'Emotion']],
    noise_df[['Filepath', 'Emotion']],
    pitch_df[['Filepath', 'Emotion']],
    time_df[['Filepath', 'Emotion']]
], ignore_index=True)

# Save the combined dataset
combined_df.to_csv('augmented_combined_train_dataset.csv', index=False)

print("Combined dataset created successfully!")
print(f"Total samples: {len(combined_df)}")
