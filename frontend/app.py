import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
import torchvision.models as models
import torch.nn as nn
import tempfile  


model_weights = "../cnn scripts/resnet50/best_resnet50.pth"

class SpeechCNN(nn.Module):
    def __init__(self):
        super(SpeechCNN, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 9) 

    def forward(self, x):
        return self.model(x)

def load_model():
    num_classes = 9  
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.normalize(y)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=target_sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) 

    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  
    mel_spec_db = np.repeat(mel_spec_db, 3, axis=0)  

    mel_spec_resized = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
    mel_spec_resized = torch.nn.functional.interpolate(mel_spec_resized, size=(224, 224), mode="bilinear")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    mel_spec_resized = (mel_spec_resized - mean) / std

    return mel_spec_resized

st.title("Speech Emotion Recognition from Audio")
st.write("Upload an audio file to predict its emotion.")
labelmap = {0: "Anger", 1: "Bored", 2: "Disgust", 3: "Fear", 4: "Happy", 
            5: "Neutral", 6: "Question", 7: "Sad", 8: "Surprise"}

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_audio_path = temp_file.name
    y, sr = librosa.load(temp_audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 1.0:
        st.error("File is too short! Please upload an audio file longer than 1 second.")
    elif duration > 4.0:
        st.error("File is too long! Please upload an audio file shorter than 4 seconds.")
    else:
        input_tensor = preprocess_audio(temp_audio_path)

        with torch.no_grad():
            output = model(input_tensor)
             # check the raw output logits to see bias
            print("Raw model output:", output.numpy()) 
            print("Predicted class index:", output.argmax(dim=1).item())

        predicted_class = output.argmax(dim=1).item()
        predicted_emotion = labelmap.get(predicted_class, "Unknown")
        st.write(f"Predicted Emotion: {predicted_emotion}")

        st.audio(uploaded_file, format="audio/mp3")    