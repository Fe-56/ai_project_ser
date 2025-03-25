# Custom dataset class for loading precomputed Melspectrogram tensors
from torch.utils.data import Dataset
import os
import torch
import pandas as pd


class MelspectrogramTensorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with Melspectrogram paths and labels.
            root_dir (string): Directory with all the Melspectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(csv_file)  # Read the CSV into a DataFrame
        self.data_frame = df[['Tensorpath', 'Emotion']]
        self.root_dir = root_dir
        self.transform = transform

        # Sort unique labels before mapping
        unique_labels = sorted(df['Emotion'].unique())
        self.label_map = {label: idx for idx,
                          label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the path to the Melspectrogram and the emotion label (string)
        # First column is path, second column is label (emotion as string)
        tensor_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        emotion_label_str = self.data_frame.iloc[idx, 1]

        # Convert the emotion string to its corresponding integer
        emotion_label = self.label_map[emotion_label_str]

        # Load the Melspectrogram tensor
        tensor = torch.load(tensor_path)

        # Apply transformations if any
        if self.transform:
            tensor = self.transform(tensor)

        # Convert emotion label to tensor
        emotion_label = torch.tensor(emotion_label, dtype=torch.long)

        return tensor, emotion_label
