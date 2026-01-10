from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.utils import pad_sequences
import re
import numpy as np

app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return render_template('index.html')
# --- CONFIGURATION ---
MAX_LEN = 100  # Must match the 'max_len' used in training
# ---------------------

# Load Assets
print("Loading model and tokenizer...")
model = tf.keras.models.load_model("hate_speech_model.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def clean_text(text):
    """Must match the training cleaning logic EXACTLY"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
@app.route('/')
def home():
    return "<h1>Backend is active! Open index.html to use the app.</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '')

    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Preprocess
    cleaned = clean_text(raw_text)
    
    # 2. Tokenize & Pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    # 3. Predict
    prediction = model.predict(padded)[0][0]
    
    # 4. Interpret (Threshold 0.5)
    result = "OFFENSIVE/HATE" if prediction > 0.5 else "NON-OFFENSIVE"
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

    return jsonify({
        'raw_text': raw_text,
        'prediction': result,
        'confidence': f"{confidence * 100:.2f}%",
        'score': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)