import os
import sys

# Set environment variables to help TensorFlow operate cleanly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
    import keras
    # Disable eager execution defaults but keep graph mode for compatibility
    TF_AVAILABLE = True
    print("✓ TensorFlow imported successfully")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"✗ TensorFlow not found: {e}")

from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = None

def load_model():
    global model
    if model is None:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required but not available!")
        
        model_path = 'crop_model.keras'
        
        # Check if file exists first
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            print(f"Attempting to load model: {model_path}")
            
            # Try loading the model
            custom_objects = {}
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=custom_objects
            )
            model.trainable = False
            print(f"✓ Model loaded successfully from {model_path}!")
            return model
                
        except Exception as e:
            error_msg = f"Failed to load model from {model_path}. Error: {e}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)

# Class names for disease detection (17 classes from crop_detection1.ipynb)
CLASS_NAMES = [
    'Corn - Common Rust',
    'Corn - Gray Leaf Spot',
    'Corn - Healthy',
    'Corn - Northern Leaf Blight',
    'Potato - Early Blight',
    'Potato - Healthy',
    'Potato - Late Blight',
    'Rice - Brown Spot',
    'Rice - Healthy',
    'Rice - Leaf Blast',
    'Rice - Neck Blast',
    'Sugarcane - Bacterial Blight',
    'Sugarcane - Healthy',
    'Sugarcane - Red Rot',
    'Wheat - Brown Rust',
    'Wheat - Healthy',
    'Wheat - Yellow Rust'
]

# Disease recommendations database
DISEASE_RECOMMENDATIONS = {
    'Common Rust': {
        'description': 'A fungal disease producing reddish-brown pustules on corn leaves.',
        'treatment': ['Apply fungicides if severe', 'Plant resistant varieties'],
        'prevention': ['Use resistant hybrids', 'Remove volunteer corn plants', 'Practice crop rotation']
    },
    'Gray Leaf Spot': {
        'description': 'A fungal disease causing rectangular lesions on corn leaves.',
        'treatment': ['Apply foliar fungicides', 'Remove crop residue'],
        'prevention': ['Rotate crops', 'Tillage to bury residue', 'Plant resistant hybrids']
    },
    'Northern Leaf Blight': {
        'description': 'A fungal disease causing long, elliptical gray-green lesions on corn.',
        'treatment': ['Apply fungicides at first sign', 'Remove crop debris after harvest'],
        'prevention': ['Plant resistant hybrids', 'Rotate crops', 'Bury or remove crop residue']
    },
    'Early Blight': {
        'description': 'A fungal disease causing dark concentric rings on older potato leaves.',
        'treatment': ['Apply fungicides containing chlorothalonil or mancozeb', 'Remove infected leaves'],
        'prevention': ['Rotate crops', 'Mulch to prevent soil splash', 'Water at base of plants']
    },
    'Late Blight': {
        'description': 'A devastating fungal disease that can destroy potato crops quickly.',
        'treatment': ['Apply fungicides immediately', 'Remove and destroy infected plants'],
        'prevention': ['Use certified disease-free seed', 'Avoid overhead irrigation', 'Ensure good air circulation']
    },
    'Brown Spot': {
        'description': 'A fungal disease causing oval spots with brown margins on rice or wheat.',
        'treatment': ['Apply fungicides', 'Balance nitrogen fertilisation'],
        'prevention': ['Use clean seeds', 'Ensure proper drainage', 'Maintain soil fertility']
    },
    'Leaf Blast': {
        'description': 'A fungal disease causing diamond-shaped lesions on rice leaves.',
        'treatment': ['Apply systemic fungicides', 'Avoid excessive nitrogen'],
        'prevention': ['Plant resistant varieties', 'Regulate water levels', 'Clean farm equipment']
    },
    'Neck Blast': {
        'description': 'The most severe form of rice blast affecting the neck of the panicle.',
        'treatment': ['Apply preventive fungicides before flowering', 'Remove infected straw'],
        'prevention': ['Avoid high humidity', 'Plant during recommended window', 'Use resistant cultivars']
    },
    'Bacterial Blight': {
        'description': 'A bacterial disease causing water-soaked streaks on sugarcane or rice.',
        'treatment': ['No chemical cure; remove infected plants', 'Apply copper-based sprays early'],
        'prevention': ['Use certified healthy setts', 'Avoid flooding', 'Field sanitation']
    },
    'Red Rot': {
        'description': 'A fungal disease causing reddening of sugarcane internal tissues.',
        'treatment': ['Destroy infected crops', 'No effective chemical control once established'],
        'prevention': ['Use healthy planting material', 'Rotate crops', 'Proper drainage']
    },
    'Brown Rust': {
        'description': 'A fungal disease producing small orange-brown pustules on wheat leaves.',
        'treatment': ['Apply triazole fungicides', 'Monitor crops regularly'],
        'prevention': ['Plant resistant varieties', 'Destroy volunteer plants', 'Early sowing']
    },
    'Yellow Rust': {
        'description': 'A fungal disease producing yellow pustules in stripes on wheat leaves.',
        'treatment': ['Apply fungicides promptly', 'Choose cool-weather resistant varieties'],
        'prevention': ['Monitor during cool/damp weather', 'Resistant cultivars', 'Crop rotation']
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_disease(image_path):
    """Predict disease from image - returns consistent results for same image"""
    model = load_model()
    
    # Open and resize image
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    image = image.resize((128, 128))
    
    # Normalize the image consistently
    input_arr = np.array(image, dtype=np.float32) / 255.0
    input_arr = np.array([input_arr])
    
    # Make prediction
    prediction = model.predict(input_arr, verbose=0)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    
    # Validate index is within range
    if result_index >= len(CLASS_NAMES):
        print(f"ERROR: Prediction index {result_index} is out of range for {len(CLASS_NAMES)} classes")
        result_index = 0
    
    print(f"Prediction: {CLASS_NAMES[result_index]} with {confidence:.2f}% confidence")
    return result_index, confidence

@app.route('/')
def index():
    """Home page - Introduction"""
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Disease detection page"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict disease
            result_index, confidence = predict_disease(filepath)
            diagnosis = CLASS_NAMES[result_index]
            plant, disease = diagnosis.split(' - ')
            
            # Store in session for recommendations page
            session['last_detection'] = {
                'image': filename,
                'plant': plant,
                'disease': disease,
                'confidence': confidence,
                'diagnosis': diagnosis
            }
            
            return redirect(url_for('recommendations'))
    
    return render_template('detect.html')

@app.route('/recommendations')
def recommendations():
    """AI Recommendations page"""
    detection = session.get('last_detection')
    if not detection:
        flash('Please upload an image first', 'warning')
        return redirect(url_for('detect'))
    
    disease_name = detection['disease']
    recommendations = None
    
    # Get recommendations for the disease
    for key in DISEASE_RECOMMENDATIONS:
        if key.lower() in disease_name.lower():
            recommendations = DISEASE_RECOMMENDATIONS[key]
            break
    
    # Default recommendations if disease not found
    if not recommendations:
        if 'Healthy' in disease_name:
            recommendations = {
                'description': 'Your plant appears to be healthy! No disease detected.',
                'treatment': ['Continue regular care', 'Monitor for any changes', 'Maintain good growing conditions'],
                'prevention': ['Ensure proper watering', 'Provide adequate nutrients', 'Monitor for pests regularly']
            }
        else:
            recommendations = {
                'description': 'Disease detected. Consult with a local agricultural expert for specific treatment.',
                'treatment': ['Remove affected leaves', 'Improve air circulation', 'Consider organic or chemical treatments'],
                'prevention': ['Practice crop rotation', 'Use disease-resistant varieties', 'Maintain plant health']
            }
    
    return render_template('recommendations.html', detection=detection, recommendations=recommendations)

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    """User contribution page"""
    if request.method == 'POST':
        plant_name = request.form.get('plant_name')
        disease_name = request.form.get('disease_name')
        description = request.form.get('description')
        
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"contrib_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Save contribution data (in production, save to database)
            contribution = {
                'plant': plant_name,
                'disease': disease_name,
                'description': description,
                'image': filename,
                'timestamp': timestamp
            }
            
            flash('Thank you for your contribution! Your submission will be reviewed.', 'success')
            return redirect(url_for('contribute'))
    
    return render_template('contribute.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    print('Crop Shield AI - Crop Disease Detection System Running')
