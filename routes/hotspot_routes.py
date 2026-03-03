import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask import Blueprint, render_template
from flask_login import login_required

#Set up the Blueprint
hotspot_bp = Blueprint("hotspot", __name__)

#Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__)) 
base_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))



# Load global assets
scaler = joblib.load(os.path.join(base_path, 'feature_scaler.joblib'))
with open(os.path.join(base_path, 'gn_name_mapping.json'), 'r') as f:
    gn_map = {int(k): v for k, v in json.load(f).items()}

#The logic
def generate_risk_scores(crime_type):
    #Load Master feature list
    with open(os.path.join(base_path, 'feature_list.json'), 'r') as f:
        master_features = json.load(f)
    scaler_features = [f for f in master_features if f != 'gn_encoded']

    #Data Prep
    inference_df = pd.read_parquet(os.path.join(base_path, 'inference_data_latest.parquet'))
    clean_df = inference_df.drop_duplicates(subset=['gn_encoded'], keep='last').copy()
    
    #Scaling
    scaled_values = scaler.transform(clean_df[scaler_features])
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_features, index=clean_df.index)
    scaled_df['gn_encoded'] = clean_df['gn_encoded'].values
    
    #Model setup
    model = joblib.load(os.path.join(base_path, 'models', f'model_{crime_type}.pkl'))
    model_features = joblib.load(os.path.join(base_path, 'model features', f'{crime_type}_features.pkl'))
    
    #Predict
    risk_scores = model.predict_proba(scaled_df[model_features])[:, 1]
    
    results = pd.DataFrame({
        'gn_encoded': clean_df['gn_encoded'].values,
        'risk_score': risk_scores
    })
    results['gn_name'] = results['gn_encoded'].map(lambda x: gn_map.get(int(x), f"ID {x}"))
    
    return results.sort_values(by='risk_score', ascending=False).to_dict(orient='records')

#Routes
@hotspot_bp.route("/")
@login_required
def index():
    
    return render_template("hotspot.html")

@hotspot_bp.route('/predict', methods=['GET'])
@login_required
def predict():
    """API endpont used by map_visualizer.js to get current risk data"""
    crime_type = request.args.get('type', 'burglary').lower()
    try:
        data = generate_risk_scores(crime_type)
        return jsonify({"status": "success", "predictions": data})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


