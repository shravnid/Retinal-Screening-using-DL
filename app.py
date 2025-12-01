from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os

app = Flask(__name__)

# ============================================================
#                    CONFIGURATION
# ============================================================
# Auto-detect environment
if os.environ.get('RENDER'):
    MODEL_PATH = 'efficientnet_b0_final.h5'
    LABELS_PATH = 'classes.json'
else:
    MODEL_PATH = r'C:\Users\shrav\OneDrive\Desktop\major-project\model\efficientnet_b0_final.h5'
    LABELS_PATH = r'C:\Users\shrav\OneDrive\Desktop\major-project\model\classes.json'

# Disease database with complete info
DISEASE_INFO = {
    "amd": {
        "name": "Age-related Macular Degeneration",
        "description": "AMD damages the macula (central retina), causing central vision loss. Most common in people over 50.",
        "symptoms": "Blurred central vision, straight lines appear wavy, difficulty recognizing faces",
        "severity": "high",
        "action": "Schedule urgent appointment with retinal specialist",
        "general_advice": [
            "Schedule comprehensive eye exam within 1-2 weeks",
            "Monitor vision daily with Amsler grid",
            "Protect eyes from UV light with sunglasses",
            "Avoid smoking - it doubles AMD risk"
        ],
        "specific_advice": [
            "Ask about anti-VEGF injections for wet AMD",
            "Consider AREDS2 vitamin supplements (consult doctor first)",
            "Discuss OCT imaging to monitor progression",
            "Inquire about low vision rehabilitation services"
        ]
    },
    "cataract": {
        "name": "Cataract",
        "description": "Clouding of the eye's natural lens, leading to decreased vision. Treatable with surgery.",
        "symptoms": "Cloudy/blurry vision, faded colors, glare sensitivity, poor night vision",
        "severity": "medium",
        "action": "Consult ophthalmologist for evaluation",
        "general_advice": [
            "Schedule eye exam to assess severity",
            "Update eyeglass prescription if needed",
            "Use brighter lighting for reading",
            "Wear anti-glare sunglasses outdoors"
        ],
        "specific_advice": [
            "Discuss cataract surgery options if vision affects daily life",
            "Ask about premium intraocular lens (IOL) options",
            "Inquire about laser-assisted cataract surgery",
            "Learn about recovery time and post-op care"
        ]
    },
    "diabetes": {
        "name": "Diabetic Retinopathy",
        "description": "Diabetes damages retinal blood vessels, potentially causing vision loss. Early detection is critical.",
        "symptoms": "Floaters, blurred vision, dark areas, difficulty with colors",
        "severity": "high",
        "action": "See retinal specialist immediately",
        "general_advice": [
            "Control blood sugar levels (HbA1c under 7%)",
            "Monitor blood pressure (target <130/80)",
            "Get dilated eye exam at least annually",
            "Maintain healthy diet and exercise routine"
        ],
        "specific_advice": [
            "Ask about laser photocoagulation treatment",
            "Discuss anti-VEGF injection therapy",
            "Inquire about diabetic macular edema screening",
            "Consider continuous glucose monitoring system",
            "Work with endocrinologist to optimize diabetes management"
        ]
    },
    "glaucoma": {
        "name": "Glaucoma",
        "description": "Increased eye pressure damages the optic nerve, causing progressive, irreversible vision loss.",
        "symptoms": "Peripheral vision loss, tunnel vision, eye pain, halos around lights",
        "severity": "high",
        "action": "Urgent ophthalmologist visit required",
        "general_advice": [
            "Get comprehensive eye exam including pressure check",
            "Never skip prescribed eye drop medications",
            "Avoid activities that increase eye pressure",
            "Inform family members (glaucoma can be hereditary)"
        ],
        "specific_advice": [
            "Ask about intraocular pressure (IOP) target for your case",
            "Discuss pressure-lowering eye drops options",
            "Inquire about laser trabeculoplasty (SLT)",
            "Learn about surgical options if medications insufficient",
            "Request visual field test and OCT imaging"
        ]
    },
    "hypertension": {
        "name": "Hypertensive Retinopathy",
        "description": "High blood pressure damages retinal blood vessels. Controlling BP prevents progression.",
        "symptoms": "Blurred vision, headaches, double vision in severe cases",
        "severity": "medium",
        "action": "Control blood pressure, see eye doctor",
        "general_advice": [
            "Monitor blood pressure daily at home",
            "Follow DASH diet (low sodium, high fruits/vegetables)",
            "Exercise 30 minutes daily (with doctor approval)",
            "Limit alcohol and quit smoking"
        ],
        "specific_advice": [
            "Work with primary care physician to optimize BP medications",
            "Target blood pressure below 130/80 mmHg",
            "Get kidney function tests (hypertension affects kidneys too)",
            "Schedule follow-up retinal exam in 3-6 months",
            "Ask about calcium channel blockers or ACE inhibitors"
        ]
    },
    "myopia": {
        "name": "High Myopia (Pathological)",
        "description": "Severe nearsightedness with elongated eyeball and stretched retina. Increases risk of complications.",
        "symptoms": "Very blurry distance vision, eye strain, increased floaters",
        "severity": "medium",
        "action": "Regular eye exams to monitor retinal health",
        "general_advice": [
            "Get annual comprehensive eye exams",
            "Update corrective lens prescription regularly",
            "Take breaks during near work (20-20-20 rule)",
            "Ensure proper lighting when reading"
        ],
        "specific_advice": [
            "Discuss myopia control options (atropine drops, ortho-k)",
            "Ask about risk of retinal detachment",
            "Inquire about LASIK/PRK eligibility",
            "Learn warning signs of retinal tears (flashes, floaters)",
            "Consider ICL surgery for very high myopia"
        ]
    },
    "normal": {
        "name": "Normal Retina",
        "description": "No significant abnormalities detected. Eyes appear healthy with no signs of disease.",
        "symptoms": "No concerning symptoms detected",
        "severity": "low",
        "action": "Maintain regular eye checkups",
        "general_advice": [
            "Schedule routine eye exams every 1-2 years",
            "Protect eyes from UV with quality sunglasses",
            "Maintain healthy lifestyle (diet, exercise, sleep)",
            "Stay hydrated and limit screen time"
        ],
        "specific_advice": [
            "Continue current eye care routine",
            "Report any sudden vision changes immediately",
            "Consider blue light filtering glasses for screens",
            "Keep emergency eye care contact information handy"
        ]
    },
    "other": {
        "name": "Other Retinal Abnormalities",
        "description": "Detected abnormalities that don't fit standard categories. Requires professional evaluation.",
        "symptoms": "Varies depending on specific condition",
        "severity": "medium",
        "action": "Consult ophthalmologist for comprehensive exam",
        "general_advice": [
            "Schedule comprehensive eye exam within 1 week",
            "Bring any previous eye exam records",
            "Document any vision changes or symptoms",
            "Avoid eye strain until evaluated"
        ],
        "specific_advice": [
            "Request detailed retinal imaging (OCT, fundus photography)",
            "Ask about referral to retinal specialist if needed",
            "Discuss potential causes of abnormalities",
            "Inquire about follow-up timeline and monitoring plan"
        ]
    }
}

# ============================================================
#                    LOAD MODEL (OPTIMIZED)
# ============================================================
print("🚀 Loading OptiScan AI...")

# Optimize TensorFlow for faster inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_jit(True)  # Enable XLA compilation for speed

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Warm up the model with a dummy prediction for faster first run
    dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
    _ = model.predict(dummy_input, verbose=0)
    print("✓ Model loaded and warmed up")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    model = None

with open(LABELS_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

# ============================================================
#                    HELPER FUNCTIONS
# ============================================================
def get_likelihood(score):
    """Classify prediction into likelihood categories"""
    if score >= 0.7:
        return "likely", "#ef4444"  # Red
    elif score >= 0.4:
        return "possible", "#f59e0b"  # Yellow
    elif score >= 0.15:
        return "unlikely", "#22c55e"  # Green
    else:
        return "very_unlikely", "#22c55e"  # Green

def preprocess_image(img):
    """Optimized preprocessing"""
    img = img.resize((224, 224), Image.BILINEAR)  # Faster resize
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

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
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model unavailable'}), 500
    
    try:
        # Fast image processing
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_img = preprocess_image(img)
        
        # Fast prediction
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # Build results with likelihood classification
        results = []
        seen_diseases = set()
        
        for i, class_name in enumerate(CLASS_NAMES):
            score = float(predictions[i])
            disease = DISEASE_INFO[class_name]
            likelihood, color = get_likelihood(score)
            
            # Skip if already seen (safety check)
            if class_name in seen_diseases:
                continue
            seen_diseases.add(class_name)
            
            results.append({
                'key': class_name,
                'name': disease['name'],
                'description': disease['description'],
                'symptoms': disease['symptoms'],
                'score': score,
                'percentage': round(score * 100, 1),
                'likelihood': likelihood,
                'color': color,
                'severity': disease['severity'],
                'action': disease['action'],
                'general_advice': disease['general_advice'],
                'specific_advice': disease['specific_advice']
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Get primary diagnosis
        primary = results[0]
        
        # If top 2 are same disease (safety), replace second with normal
        if len(results) > 1 and results[0]['key'] == results[1]['key']:
            normal_disease = DISEASE_INFO['normal']
            likelihood, color = get_likelihood(0.0)
            results[1] = {
                'key': 'normal',
                'name': normal_disease['name'],
                'description': normal_disease['description'],
                'symptoms': normal_disease['symptoms'],
                'score': 0.0,
                'percentage': 0.0,
                'likelihood': likelihood,
                'color': color,
                'severity': normal_disease['severity'],
                'action': normal_disease['action'],
                'general_advice': normal_disease['general_advice'],
                'specific_advice': normal_disease['specific_advice']
            }
        
        return jsonify({
            'success': True,
            'primary': primary,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)