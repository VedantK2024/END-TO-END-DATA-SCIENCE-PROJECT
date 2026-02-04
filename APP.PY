"""
Flask API for House Price Prediction
"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessors
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            # API request
            data = request.get_json()
        else:
            # Form request
            data = request.form.to_dict()
        
        # Create feature dictionary
        features = {
            'area': float(data['area']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'stories': int(data['stories']),
            'mainroad': data['mainroad'],
            'guestroom': data['guestroom'],
            'basement': data['basement'],
            'hotwaterheating': data['hotwaterheating'],
            'airconditioning': data['airconditioning'],
            'parking': int(data['parking']),
            'prefarea': data['prefarea'],
            'furnishingstatus': data['furnishingstatus']
        }
        
        # Encode categorical features
        for col in label_encoders.keys():
            features[col] = label_encoders[col].transform([features[col]])[0]
        
        # Create feature array in correct order
        feature_array = np.array([[features[name] for name in feature_names]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(feature_scaled)[0]
        
        if request.is_json:
            return jsonify({
                'success': True,
                'predicted_price': float(prediction),
                'formatted_price': f"${prediction:,.2f}"
            })
        else:
            return render_template('result.html', 
                                 prediction=f"${prediction:,.2f}",
                                 features=data)
    
    except Exception as e:
        if request.is_json:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        else:
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    Expected JSON format:
    {
        "area": 2500,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }
    """
    return predict()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Ridge Regression',
        'version': '1.0'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
