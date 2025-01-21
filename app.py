from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load the emotion classifier
classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

@app.route('/')
def home():
    return "Emotion Classifier API - Send a POST request to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    results = classifier(text)
    return jsonify(results[0])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
