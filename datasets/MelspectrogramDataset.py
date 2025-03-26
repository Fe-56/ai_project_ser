import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

# Custom dataset class for loading Melspectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with Melspectrogram paths and labels.
            root_dir (string): Directory with all the Melspectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(csv_file)  # Read the CSV into a DataFrame
        self.data_frame = df[['Melspectrogrampath', 'Emotion']]
        self.root_dir = root_dir
        self.transform = transform
        
        # Sort unique labels before mapping
        unique_labels = sorted(df['Emotion'].unique())  
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the path to the Melspectrogram and the emotion label (string)
        # First column is path, second column is label (emotion as string)
        mel_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        emotion_label_str = self.data_frame.iloc[idx, 1]

        # Convert the emotion string to its corresponding integer
        emotion_label = self.label_map[emotion_label_str]

        # Load the Melspectrogram
        mel_image = Image.open(mel_path)
        mel_image = mel_image.convert("RGB")

        # Apply transformations if any
        if self.transform:
            mel_image = self.transform(mel_image)
            
        # Convert emotion label to tensor
        emotion_label = torch.tensor(emotion_label, dtype=torch.long)

        return mel_image, emotion_label    