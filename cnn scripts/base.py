# # # import pandas as pd
# # # from datasets import Dataset, Image

# # # df_train = pd.read_csv('../data/melspectrogram_train_dataset.csv')

# # # print(df_train)

# # import torch
# # print(torch.__version__)
# # print(torch.cuda.is_available())

# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import soundfile as sf  # For saving the audio

# def repeat_pad_audio(y, desired_length):
#     """
#     Repeat and pad the audio signal 'y' until it reaches the desired_length (in samples).
#     """
#     if len(y) >= desired_length:
#         return y[:desired_length]
#     # Calculate how many full repetitions of y we need
#     full_reps = int(np.floor(desired_length / len(y)))
#     # Calculate the number of samples needed from the next repetition
#     remainder = desired_length - full_reps * len(y)
#     # Repeat the audio and add the remaining part
#     y_padded = np.concatenate([np.tile(y, full_reps), y[:remainder]])
#     return y_padded

# # Load your 5-second audio clip at a target sample rate (e.g., 16000 Hz)
# y, sr = librosa.load('YAF_back_angry.wav', sr=16000)

# # Calculate the desired total number of samples for 11 seconds
# target_duration = 11  # seconds
# target_length = int(target_duration * sr)

# # Apply repetitive padding to create an 11-second audio clip
# y_padded = repeat_pad_audio(y, target_length)

# # Save the padded audio to a file so you can listen to it
# sf.write('padded_audio.wav', y_padded, sr)
# print("Padded audio saved as 'padded_audio.wav'.")

# # Extract the mel spectrogram from the padded audio
# melspectrogram = librosa.feature.melspectrogram(
#     y=y_padded, 
#     sr=sr, 
#     n_fft=2048, 
#     hop_length=512, 
#     n_mels=128
# )
# melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# # Plot and display the mel spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(melspectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
# plt.title('Mel Spectrogram of 11s Padded Audio')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

from tinytag import TinyTag
import pandas as pd

def get_duration(filepath):
    tag = TinyTag.get(filepath)
    return tag.duration

df = pd.read_csv('../data/speech_dataset.csv')
print(df)
print(len(df))

paths = df['Filepath'].tolist()
paths = ["../data/" + x for x in paths]

tinyl = []

# Apply the function to add a new duration column
for x in paths:
    if x == "../data/./dataset/tess\YAF_fear\YAF_neat_fear.wav":
        continue
    print(x)
    tinyl.append(get_duration(x))

print(len(tinyl), max(tinyl))