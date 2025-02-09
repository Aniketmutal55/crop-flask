from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load crop dataset
csv_path = 'Crop_recommendation.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}")

# Load and prepare the crop 
df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Load fertilizer dataset
try:
    fertilizer_data = pd.read_csv('Fertilizer.csv')
    print("Fertilizer data loaded successfully")
    print("Columns in fertilizer data:", fertilizer_data.columns.tolist())
except Exception as e:
    print(f"Error loading Fertilizer.csv: {str(e)}")
    fertilizer_data = None

# Add this route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/suggest_fertilizer', methods=['POST'])
def suggest_fertilizer():
    try:
        # Check if fertilizer data is loaded
        if fertilizer_data is None:
            return jsonify({'error': 'Fertilizer data not available'}), 500

        # Get data from request
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
        
        # Find most suitable fertilizer based on soil conditions
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

@app.route('/check_images')
def check_images():
    static_dir = app.static_folder
    images_dir = os.path.join(static_dir, 'images')
    
    if not os.path.exists(images_dir):
        return "Images directory doesn't exist!"
    
    files = os.listdir(images_dir)
    return f"Files in images directory: {files}"

if __name__ == '__main__':
    app.run() 