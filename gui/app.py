from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa
import numpy as np
import tempfile
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("./checkpoint-9477")
labelmap = {0: "Anger", 1: "Bored", 2: "Disgust", 3: "Fear", 4: "Happy", 
            5: "Neutral", 6: "Question", 7: "Sad", 8: "Surprise"}

def preprocess_audio(audio_path):
    print("audio_path: ", audio_path)
    y, sr = librosa.load(audio_path, sr=16000)
    segment_length = 4 * sr 
    segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
    
    segments = [np.pad(seg, (0, max(0, segment_length - len(seg))), mode='constant') for seg in segments]
    
    return segments

def predict_emotion(segment):
    print("segment: ", segment)
    
    # Process the input segment
    input_tensor = processor(segment, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        output = model(input_tensor)

    logits = output.logits  # Raw model outputs
    probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
    predicted_class = logits.argmax(dim=1).item()  # Get predicted label
    confidence = probabilities[0, predicted_class].item()  # Get confidence score

    return {
        "emotion": labelmap.get(predicted_class, "Unknown"),
        "confidence": round(confidence * 100, 2)  # Convert to percentage
    }

@app.route('/predict_emotion', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_file.read())
        audio_path = temp_file.name

    segments = preprocess_audio(audio_path)
    
    predictions = []
    for idx, segment in enumerate(segments):
        result = predict_emotion(segment)  # Now returns emotion + confidence
        start_time = idx * 4 
        end_time = start_time + 4
        predictions.append({
            'start_time': start_time,
            'end_time': end_time,
            'emotion': result['emotion'],
            'confidence': result['confidence']
        })
        print("predictions:", predictions)
    
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)