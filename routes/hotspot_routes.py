import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask import Blueprint, render_template, session
from flask_login import login_required
from sqlalchemy import text, bindparam 
from extensions import db # db is imported from extensions

#Set up the Blueprint
hotspot_bp = Blueprint("hotspot", __name__)

#Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__)) 
base_path = os.path.abspath(os.path.join(current_dir, ".."))



# Load global assets
scaler = joblib.load(os.path.join(base_path,'models','feature_scaler.joblib'))
with open(os.path.join(base_path, 'static', 'gn_div_info','gn_name_mapping.json'), 'r') as f:
    gn_map = {int(k): v for k, v in json.load(f).items()}

#The logic
def generate_risk_scores(crime_type):
    
    #Load Master feature list
    with open(os.path.join(base_path, 'models','feature_h_list.json'), 'r') as f:
        master_features = json.load(f)
    scaler_features = [f for f in master_features if f != 'gn_encoded']

    #Data Prep
    inference_df = pd.read_parquet(os.path.join(base_path,'models', 'inference_data_latest.parquet'))
    clean_df = inference_df.drop_duplicates(subset=['gn_encoded'], keep='last').copy()
    
    #Scaling
    scaled_values = scaler.transform(clean_df[scaler_features])
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_features, index=clean_df.index)
    scaled_df['gn_encoded'] = clean_df['gn_encoded'].values
    
    #Model setup
    model_path = os.path.join(base_path, 'models', f'model_{crime_type}.pkl')
    model = joblib.load(model_path)

    feature_file = os.path.join(base_path, 'models', f'{crime_type}_features.pkl')
    model_features = joblib.load(feature_file)
    
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

    #Store the selection in the session
    session['selected_crime_type'] = crime_type
    try:
        #Get the predictions
        predictions = generate_risk_scores(crime_type)
        df = pd.DataFrame(predictions)

        #Fetch actual names from the database 
        gn_pcodes = df['gn_name'].tolist()

    
        query = text("""
            SELECT "admin4Pcode" AS pcode,
                   "admin4Name_en" AS actual_name
            FROM gn_division_info
            WHERE "admin4Pcode" IN :codes
        """).bindparams(bindparam("codes", expanding=True))

        with db.engine.connect() as conn:
            name_data = conn.execute(query, {"codes": gn_pcodes}).mappings().all()

        #Merge the names back into the results
        if name_data:
            names_df = pd.DataFrame(name_data)
            df = df.merge(names_df, left_on='gn_name', right_on='pcode', how='left')

            # Map PCODE to technical ID for the map and Name for the chart
            df['display_name'] = df['actual_name'].fillna(df['gn_name'])
            df = df.rename(columns={'gn_name': 'pcode_id'})

        else:
            # Fallback if DB query fails
            df['display_name'] = df['gn_name']
            df['pcode_id'] = df['gn_name'] 


        # Return both the readable name for charts AND the pcode for the map
        final_data = df[['display_name', 'pcode_id', 'risk_score']].to_dict(orient='records')
        return jsonify({"status": "success", "predictions": final_data})
 

    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({"status": "failed", "error": str(e)}), 500


