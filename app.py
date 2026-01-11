import numpy as np
import onnxruntime as ort
import pickle
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MAX_LEN = 100 

# --- LOAD ASSETS (ONNX) ---
print("Loading ONNX model...")
# Load the ONNX model
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- HELPER FUNCTIONS ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def pad_sequences_numpy(sequences, maxlen, padding='post'):
    """Lightweight replacement for tensorflow.keras.utils.pad_sequences"""
    padded = np.zeros((len(sequences), maxlen), dtype='float32')
    for i, seq in enumerate(sequences):
        if not seq: continue
        trunc = seq[:maxlen]
        if padding == 'post':
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
    return padded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '')

    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Preprocess
    cleaned = clean_text(raw_text)
    
    # 2. Tokenize
    seq = tokenizer.texts_to_sequences([cleaned])
    
    # 3. Pad (Using NumPy instead of Keras)
    padded = pad_sequences_numpy(seq, maxlen=MAX_LEN, padding='post')

    # 4. Predict (Using ONNX Runtime)
    # ONNX requires inputs as a dictionary
    prediction = sess.run([output_name], {input_name: padded})[0][0][0]
    
    # 5. Interpret
    result = "OFFENSIVE/HATE" if prediction > 0.5 else "NON-OFFENSIVE"
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

    return jsonify({
        'raw_text': raw_text,
        'prediction': result,
        'confidence': f"{confidence * 100:.2f}%",
        'score': float(prediction)
    })

if __name__ == '__main__':
    # Render assigns the PORT env var, but 5000 is default fallback
    app.run(host='0.0.0.0', port=5000)