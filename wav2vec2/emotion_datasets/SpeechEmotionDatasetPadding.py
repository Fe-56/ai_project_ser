import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# Custom dataset class, pads audios to 4s long
class SpeechEmotionDatasetPadding(Dataset):
    # Max_length = 4s, 64000 because sampling rate is 16000
    def __init__(self, df, processor, max_length=64000):
        self.df = df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = '../data/' + self.df.iloc[idx]['Filepath']
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