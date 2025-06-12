"""
Prediction API for Predictive Maintenance System

This module provides a Flask API for real-time failure predictions.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_columns = None

def load_model_and_scaler():
    """
    Load the trained model and scaler.
    """
    global model, scaler, feature_columns
    
    try:
        model = joblib.load('models/best_model.pkl')
        print("Model loaded successfully")
        
        # Try to load scaler (may not exist if not used)
        try:
            scaler = joblib.load('models/scaler.pkl')
            print("Scaler loaded successfully")
        except FileNotFoundError:
            print("Scaler not found, proceeding without scaling")
        
        # Load feature columns from a sample data file
        try:
            sample_data = pd.read_csv('data/processed/engineered_features.csv', nrows=1)
            feature_columns = [col for col in sample_data.columns if col not in ['failure', 'timestamp']]
            print(f"Feature columns loaded: {len(feature_columns)} features")
        except FileNotFoundError:
            print("Sample data not found, using default feature columns")
            feature_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(data):
    """
    Preprocess input data for prediction.
    
    Args:
        data (dict): Input sensor data
        
    Returns:
        np.array: Preprocessed data ready for prediction
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in df.columns:
                # For missing engineered features, use simple defaults
                if 'rolling' in col or 'lag' in col or 'roc' in col:
                    df[col] = 0.0  # Default value for engineered features
                elif col in ['temperature', 'vibration', 'pressure', 'rotation_speed']:
                    df[col] = data.get(col, 0.0)
                else:
                    df[col] = 0.0
        
        # Select only the required columns in the correct order
        df = df[feature_columns]
        
        # Apply scaling if scaler is available
        if scaler is not None:
            df_scaled = scaler.transform(df)
            return df_scaled
        else:
            return df.values
            
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/')
def home():
    """
    Home endpoint with API information.
    """
    return jsonify({
        "message": "Predictive Maintenance API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Make failure prediction",
            "/health": "GET - Check API health",
            "/model_info": "GET - Get model information"
        }
    })

@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route('/model_info')
def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": type(model).__name__,
        "feature_count": len(feature_columns),
        "features": feature_columns[:10],  # Show first 10 features
        "scaler_loaded": scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make failure prediction based on sensor data.
    
    Expected input format:
    {
        "temperature": 75.5,
        "vibration": 0.6,
        "pressure": 102.3,
        "rotation_speed": 1850
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['temperature', 'vibration', 'pressure', 'rotation_speed']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        if processed_data is None:
            return jsonify({"error": "Error preprocessing input data"}), 500
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Determine risk level
        failure_probability = prediction_proba[1]
        if failure_probability < 0.3:
            risk_level = "Low"
        elif failure_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Generate recommendations
        recommendations = generate_recommendations(data, failure_probability)
        
        # Prepare response
        response = {
            "prediction": {
                "failure_predicted": bool(prediction),
                "failure_probability": float(failure_probability),
                "no_failure_probability": float(prediction_proba[0]),
                "risk_level": risk_level
            },
            "input_data": data,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

def generate_recommendations(sensor_data, failure_probability):
    """
    Generate maintenance recommendations based on sensor data and failure probability.
    
    Args:
        sensor_data (dict): Current sensor readings
        failure_probability (float): Predicted failure probability
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Temperature-based recommendations
    if sensor_data.get('temperature', 0) > 80:
        recommendations.append("High temperature detected. Check cooling system.")
    
    # Vibration-based recommendations
    if sensor_data.get('vibration', 0) > 0.8:
        recommendations.append("High vibration detected. Inspect bearings and alignment.")
    
    # Pressure-based recommendations
    if sensor_data.get('pressure', 0) < 95:
        recommendations.append("Low pressure detected. Check for leaks in the system.")
    
    # Speed-based recommendations
    rotation_speed = sensor_data.get('rotation_speed', 0)
    if rotation_speed < 1700 or rotation_speed > 1900:
        recommendations.append("Rotation speed outside normal range. Check motor and drive system.")
    
    # Failure probability-based recommendations
    if failure_probability > 0.7:
        recommendations.append("URGENT: High failure probability. Schedule immediate maintenance.")
    elif failure_probability > 0.5:
        recommendations.append("Moderate failure risk. Schedule maintenance within 24 hours.")
    elif failure_probability > 0.3:
        recommendations.append("Elevated failure risk. Monitor closely and schedule preventive maintenance.")
    else:
        recommendations.append("System operating normally. Continue regular monitoring.")
    
    return recommendations

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple data points.
    
    Expected input format:
    {
        "data": [
            {"temperature": 75.5, "vibration": 0.6, "pressure": 102.3, "rotation_speed": 1850},
            {"temperature": 82.1, "vibration": 0.9, "pressure": 98.7, "rotation_speed": 1780}
        ]
    }
    """
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "No data array provided"}), 400
        
        data_points = request_data['data']
        results = []
        
        for i, data_point in enumerate(data_points):
            try:
                # Preprocess input
                processed_data = preprocess_input(data_point)
                
                if processed_data is None:
                    results.append({
                        "index": i,
                        "error": "Error preprocessing input data"
                    })
                    continue
                
                # Make prediction
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]
                failure_probability = prediction_proba[1]
                
                # Determine risk level
                if failure_probability < 0.3:
                    risk_level = "Low"
                elif failure_probability < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                results.append({
                    "index": i,
                    "prediction": {
                        "failure_predicted": bool(prediction),
                        "failure_probability": float(failure_probability),
                        "risk_level": risk_level
                    },
                    "input_data": data_point
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": f"Prediction error: {str(e)}"
                })
        
        return jsonify({
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model and scaler on startup
    if load_model_and_scaler():
        print("Starting Predictive Maintenance API...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Please ensure the model file exists.")
        print("Run model_training.py first to train and save the model.")

