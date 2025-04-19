from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import tempfile

from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    WavLMForSequenceClassification
)

from MetaLearner import MetaFFNN

app = Flask(__name__)
CORS(app)

# Emotion label map
labelmap = {
    0: "Anger", 1: "Bored", 2: "Disgust", 3: "Fear",
    4: "Happy", 5: "Neutral", 6: "Question", 7: "Sad", 8: "Surprise"
}

processor1 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor2 = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")

model1 = Wav2Vec2ForSequenceClassification.from_pretrained('./checkpoint-22112', num_labels=9)
model2 = WavLMForSequenceClassification.from_pretrained('./checkpoint-55280', num_labels=9)

meta_model = MetaFFNN(input_dim=18, hidden_dim=128, output_dim=9)
meta_model.load_state_dict(torch.load('./best_meta_ffnn_model.pt', map_location=torch.device('cpu')))

model1.eval()
model2.eval()
meta_model.eval()

def preprocess_audio(audio_path):
    print("inside preprocess audio")
    print("audio_path: ", audio_path)
    y, sr = librosa.load(audio_path, sr=16000)
    segment_length = 5 * sr 
    segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
    segments = [np.pad(seg, (0, max(0, segment_length - len(seg))), mode='constant') for seg in segments]
    return segments, sr

@app.route('/predict_emotion', methods=['POST'])
def predict():
    print("inside predict")
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_file.read())
        audio_path = temp_file.name

    segments, sr = preprocess_audio(audio_path)
    print("segments: ", segments)

    inputs1 = processor1(segments, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs2 = processor2(segments, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits1 = model1(inputs1.input_values).logits 
        logits2 = model2(inputs2.input_values).logits 

        meta_features = torch.cat([logits1, logits2], dim=-1) 

        meta_outputs = meta_model(meta_features)
        probs = F.softmax(meta_outputs, dim=1)
        preds = torch.argmax(meta_outputs, dim=1)

    predictions = []
    for idx, prob_vec in enumerate(probs):
        top_probs, top_indices = torch.topk(prob_vec, k=3)
        top_emotions = []
        for i in range(3):
            top_emotions.append({
                "emotion": labelmap[top_indices[i].item()],
                "confidence": round(top_probs[i].item() * 100, 2)
            })

        predictions.append({
            "start_time": idx * 5,
            "end_time": (idx + 1) * 5,
            "top_emotions": top_emotions
        })
    print("predictions:", predictions)

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
