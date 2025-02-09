from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Get absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to CSV files
crop_csv_path = os.path.join(BASE_DIR, 'Crop_recommendation.csv')
fertilizer_csv_path = os.path.join(BASE_DIR, 'Fertilizer.csv')

# Load crop dataset
if not os.path.exists(crop_csv_path):
    raise FileNotFoundError(f"Could not find {crop_csv_path}")

df = pd.read_csv(crop_csv_path)
X = df.drop('label', axis=1)
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Load fertilizer dataset
if os.path.exists(fertilizer_csv_path):
    fertilizer_data = pd.read_csv(fertilizer_csv_path)
    print("Fertilizer.csv loaded successfully!")
else:
    print(f"Error: Fertilizer.csv not found at {fertilizer_csv_path}")
    fertilizer_data = None  # Avoid app crash if file isn't found

# Route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Crop recommendation prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        features_scaled = scaler.transform([features])
        prediction = rf_model.predict(features_scaled)
        
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({'error': str(e)}), 400

# Fertilizer suggestion route
@app.route('/suggest_fertilizer', methods=['POST'])
def suggest_fertilizer():
    try:
        # Check if fertilizer data is loaded
        if fertilizer_data is None:
            return jsonify({'error': 'Fertilizer data not available'}), 500

        # Get request data
        data = request.get_json()
        print("Received data for fertilizer:", data)
        
        # Extract soil features
        soil_features = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph'])
        }
        
        print("Soil features:", soil_features)
        
        # Find the best matching fertilizer
        best_match = None
        min_difference = float('inf')
        
        for _, fertilizer in fertilizer_data.iterrows():
            try:
                difference = (
                    abs(fertilizer['N_content'] - soil_features['N']) +
                    abs(fertilizer['P_content'] - soil_features['P']) +
                    abs(fertilizer['K_content'] - soil_features['K']) +
                    abs(fertilizer['temperature_requirement'] - soil_features['temperature']) +
                    abs(fertilizer['humidity_requirement'] - soil_features['humidity']) +
                    abs(fertilizer['pH_requirement'] - soil_features['ph'])
                )
                
                if difference < min_difference:
                    min_difference = difference
                    best_match = fertilizer['name']
                    
            except Exception as e:
                print(f"Error processing fertilizer row: {str(e)}")
                continue
        
        if best_match is None:
            return jsonify({'error': 'Could not find suitable fertilizer'}), 400
            
        print("Best match found:", best_match)
        
        return jsonify({
            'fertilizer_suggestion': best_match,
            'soil_features': soil_features
        })
        
    except Exception as e:
        print("Error in fertilizer suggestion:", str(e))
        return jsonify({'error': str(e)}), 400

# Debug route to check files in the directory
@app.route('/debug-files')
def debug_files():
    files = os.listdir(BASE_DIR)
    return jsonify({"files_in_directory": files})

# Check static images
@app.route('/check_images')
def check_images():
    static_dir = app.static_folder
    images_dir = os.path.join(static_dir, 'images')
    
    if not os.path.exists(images_dir):
        return "Images directory doesn't exist!"
    
    files = os.listdir(images_dir)
    return f"Files in images directory: {files}"

# Run the Flask app
if __name__ == '__main__':
    app.run()
