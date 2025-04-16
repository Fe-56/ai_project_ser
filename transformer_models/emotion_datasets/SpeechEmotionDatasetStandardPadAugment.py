import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# Custom dataset class, standard padding of audios to 5s long
class SpeechEmotionDatasetStandardPadAugment(Dataset):
    # Max_length = 5s, 80000 because sampling rate is 16000
    def __init__(self, df, processor, root_dir, max_length=80000):
        self.df = df
        self.processor = processor
        self.root_dir = root_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.df.iloc[idx]['Filepath'])
        label = self.df.iloc[idx]['Emotion']

        # Load audio file
        speech, sr = librosa.load(audio_path, sr=16000)

        # Pad speech to required length
        speech = np.pad(speech, (0, self.max_length -
                        len(speech)), mode='constant')

        # Preprocess audio
        inputs = self.processor(speech, sampling_rate=16000, return_tensors='pt',
                                padding=True, truncate=True, max_length=self.max_length)

        input_values = inputs.input_values.squeeze()
        return {'input_values': input_values, 'labels': torch.tensor(label, dtype=torch.long)}