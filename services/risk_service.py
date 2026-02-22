import joblib
import json
import pandas as pd
import numpy as np
import os
import shap
from extensions import db
from sqlalchemy import text


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR,"models")

# load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# load the models
CRIME_MODELS = {}
for crime in ["burglary" ,"theft" ,"vehicle" ,"robbery" ,"drugs", "stabbing"]:
    CRIME_MODELS[crime] = joblib.load(
        os.path.join(MODEL_DIR, f"rf_{crime}.pkl")
    )

# load feature order
with open("models/feature_list.json") as f:
    FEATURE_ORDER = json.load(f)


def run_risk_factor_pipeline(crime_type, features):
    """
    Runs ML inference and generate SHAP explanation
    """

    if crime_type not in CRIME_MODELS:
        raise ValueError("Invalid crime type")

    # convert to dataframe - (ml models requires tabular input, feature order must match training)
    X = pd.DataFrame([features])
    X = X[FEATURE_ORDER]

    # scale features
    X_scaled = scaler.transform(X)

    # PREDICT
    model = CRIME_MODELS[crime_type]
    # predict_proba - returns the probabilities of crime and no crime
    # float -  converts Numpy float in to python float(JSON safe)
    # this is the risk score shown on the dashboard
    risk = float(model.predict_proba(X_scaled)[0][1])

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shape_values(X_scaled)

    shap_values_pos = shap_values[:,:,1]

    # get top 3 drivers
    shap_row = shap_values_pos[0]
    feature_importance = pd.DataFrame({
        "feature":FEATURE_ORDER,
        "shap_value":shap_row
    })

    top_drivers = (
        feature_importance
        .reindex(feature_importance.shap_values.abs().sort_values(ascending=False).index)
        .head(3)
    )

    return {
        "risk_score":round(risk, 3),
        "top_features":top_drivers["feature"].tolist(),
        "shap_values":top_drivers["shap_values"].tolist()
    }

    # SHAP

    # explainer = shap.TreeExplainer(rf_model)

    # shap_values = explainer.shap_values(X_test_scaled)

    # shap_values_pos = shap_values[:,:,1]
    # base_value = explainer.expected_value[1]

    # global_summary = get_global_summary_table(shap_values_pos, X_test_scaled.columns)

    # dirrectional_effect = get_global_directional_effect_table(shap_values_pos, X_test_scaled)


    # local_results = []

    # for i in range(len(X_test_scaled)):
    #     top_drivers = get_local_top_drivers(shap_values_pos[i],X_test_scaled.iloc[i])


# fetch the GN list 
def get_all_gns():
    query = text("""
        SELECT DISTINCT gn_division
        FROM crime_data
        ORDER BY gn_pcode    
    """)

    with db.engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    return [row[0] for row in rows]




# fetch gn level environmental, socio-economic, demographic,... data
def fetch_gn_features(gn_division):
    query = text("""
        SELECT
            GN_population,
            AVG_Household_income,
            Unemployement_Rate,
            Builiding Density,
            Road_Density,
            distance_to_station_km
        FROM crime_data,
        WHERE gn_division = :gn,
        ORDER BY year DESC
        LIMIT 1
    """)

    with db.engine.connect() as conn():
        row = conn.execute(query, {"gn":gn_division}),mappings().first()

    if not row:
        raise ValueError("GN not found")
    
    return dict(row)





# gn_division :  ['Welata' 'Katukele West' 'Penideniya' 'Aniwatta East' 'Aruppala East'
#  'Suduhumpala West' 'Mahaiyawa' 'Boowelikada' 'Asgiriya' 'Thennakumbura'
#  'Bowala' 'Mahanuwara' 'Thalwatta' 'Ampitiya Udagama North'
#  'Thalathuoya East' 'Poorna Watta West' 'Bogambara' 'Ihala Katukele'
#  'Suduhumpala East' 'Bowalawatta' 'Poorna Watta East' 'Ihala Dodamwala'
#  'Sirimalwatta West' 'Sirimalwatta East' 'Ampitiya Pallegama' 'Batagalla'
#  'Uda Bowala' 'Kotagepitiya' 'Watapuluwa West' 'Hanthana Pedesa'
#  'Ampitiya North' 'Nagasthenna' 'Bahirawa Kanda' 'Getambe' 'Nittawela'
#  'Pallegunnepana North' 'Lewella' 'Mahaweli Uyana' 'Hippola'
#  'Pitakanda Gama' 'Deiyannewela' 'Ampitiya South' 'Kendakaduwa'
#  'Watapuluwa South' 'Kumburegedara' 'Kundasale North' 'Watapuluwa'
#  'Pallegunnepana South' 'Dehiwatta' 'Palle Peradeniya' 'Katukele'
#  'Ogastawatta' 'Aniwatta West' 'Meddegama' 'Inguruwatta' 'Wattegama North'
#  'Kalugalawatta' 'Malwatta' 'Pitakanda' 'Mulgampala'
#  'Gurudeniya Dambawela' 'Wewathenna' 'Godagandeniya' 'Haloluwa'
#  'Uduwelawatta Colony' 'Uda Eriyagama East' 'Arangala South'
#  'Wathurakumbura' 'Mapanawathura' 'Thalathuoya West' 'Ulpathakumbura']


            # land_area_density,
            # road_density,
            # log_population,
            # distance_km,
            # year,
            # week_sin,
            # week_cos,
        