from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

app = Flask(__name__)

# ============================================================
#                    CONFIGURATION
# ============================================================
# UPDATE THESE PATHS TO YOUR ACTUAL FILES
MODEL_PATH = r'C:\Users\shrav\OneDrive\Desktop\major-project\model\efficientnet_b0_final.h5'
LABELS_PATH = r'C:\Users\shrav\OneDrive\Desktop\major-project\model\classes.json'

# Disease information database
DISEASE_INFO = {
    "amd": {
        "name": "Age-related Macular Degeneration",
        "description": "AMD damages the macula (central part of retina), causing loss of central vision. Common in people over 50.",
        "symptoms": "Blurred central vision, straight lines appear wavy, difficulty recognizing faces",
        "action": "Schedule urgent appointment with retinal specialist"
    },
    "cataract": {
        "name": "Cataract",
        "description": "Clouding of the eye's natural lens, leading to decreased vision. Most common in older adults.",
        "symptoms": "Cloudy or blurry vision, faded colors, glare sensitivity, poor night vision",
        "action": "Consult ophthalmologist for evaluation and possible surgery"
    },
    "diabetes": {
        "name": "Diabetic Retinopathy",
        "description": "Diabetes damages blood vessels in the retina, potentially causing vision loss if untreated.",
        "symptoms": "Floaters, blurred vision, dark areas in vision, difficulty with colors",
        "action": "See retinal specialist immediately. Monitor blood sugar levels closely"
    },
    "glaucoma": {
        "name": "Glaucoma",
        "description": "Increased eye pressure damages the optic nerve, leading to progressive vision loss.",
        "symptoms": "Peripheral vision loss, tunnel vision, eye pain, halos around lights",
        "action": "Urgent ophthalmologist visit. Get intraocular pressure test and visual field test"
    },
    "hypertension": {
        "name": "Hypertensive Retinopathy",
        "description": "High blood pressure damages retinal blood vessels, potentially affecting vision.",
        "symptoms": "Blurred vision, headaches, double vision (in severe cases)",
        "action": "See your primary care physician to control blood pressure. Follow up with eye doctor"
    },
    "myopia": {
        "name": "High Myopia (Pathological)",
        "description": "Severe nearsightedness causing elongated eyeball and stretched retina, increasing risk of complications.",
        "symptoms": "Very blurry distance vision, eye strain, increased floaters",
        "action": "Regular eye exams to monitor retinal health. Discuss corrective options with optometrist"
    },
    "normal": {
        "name": "Normal Retina",
        "description": "Your retina appears healthy with no signs of disease or abnormalities.",
        "symptoms": "No concerning symptoms detected",
        "action": "Maintain regular eye checkups every 1-2 years"
    },
    "other": {
        "name": "Other Retinal Abnormalities",
        "description": "Detected abnormalities that don't fit standard categories. Requires professional evaluation.",
        "symptoms": "Varies depending on specific condition",
        "action": "Consult an ophthalmologist for comprehensive examination"
    }
}

# ============================================================
#                    LOAD MODEL
# ============================================================
print("Loading model...")
try:
    # Try loading with compile=False to avoid optimizer issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully (compile=False)")
except Exception as e:
    print(f"First attempt failed: {e}")
    print("Trying alternative loading method...")
    # Alternative: Load with custom objects
    from tensorflow.keras.layers import InputLayer
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'InputLayer': InputLayer},
        compile=False
    )
    print("Model loaded with custom objects")

with open(LABELS_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)
print(f"Classes loaded: {CLASS_NAMES}")

# ============================================================
#                    HELPER FUNCTIONS
# ============================================================
def get_risk_level(score):
    """Determine risk level based on prediction score"""
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    else:
        return "high"

def preprocess_image(img):
    """Preprocess image for EfficientNet model"""
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================================
#                    ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # Prepare results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            score = float(predictions[i])
            disease_data = DISEASE_INFO.get(class_name, {
                "name": class_name,
                "description": "No description available",
                "symptoms": "Unknown",
                "action": "Consult an eye care professional"
            })
            
            results.append({
                'class_key': class_name,
                'name': disease_data['name'],
                'score': score,
                'percentage': round(score * 100, 1),
                'risk_level': get_risk_level(score),
                'description': disease_data['description'],
                'symptoms': disease_data['symptoms'],
                'action': disease_data['action']
            })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Primary diagnosis
        primary = results[0]
        
        return jsonify({
            'success': True,
            'primary_diagnosis': primary,
            'all_results': results
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)