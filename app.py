from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "efficientnet_b0_final_v2.h5"
CLASSES_PATH = "classes_v2.json"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✓ Model loaded")

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

DISEASE_INFO = {
    "amd": "Age-related Macular Degeneration",
    "cataract": "Cataract",
    "diabetes": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "hypertension": "Hypertensive Retinopathy",
    "myopia": "High Myopia",
    "normal": "Normal Retina",
    "other": "Other Abnormalities"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        
        # Read and preprocess
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img)[0]
        
        # Format results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            score = float(predictions[i])
            results.append({
                'disease': DISEASE_INFO.get(class_name, class_name),
                'probability': f"{score*100:.1f}%",
                'risk': 'high' if score > 0.6 else ('medium' if score > 0.3 else 'low')
            })
        
        # Sort by probability
        results.sort(key=lambda x: float(x['probability'].strip('%')), reverse=True)
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)