import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# Custom dataset class, pad to maximum length in the training batch
class SpeechEmotionDatasetDynamicPad(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = '../data/' + self.df.iloc[idx]['Filepath']
        label = self.df.iloc[idx]['Emotion']

        # Load audio file
        speech, sr = librosa.load(audio_path, sr=16000)

        # Preprocess audio
        inputs = self.processor(speech, sampling_rate=16000, return_tensors='pt')

        input_values = inputs.input_values.squeeze()
        return {'input_values': input_values, 'labels': torch.tensor(label, dtype=torch.long)}