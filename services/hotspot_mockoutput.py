import pandas as pd
import os

DATA_PATH = os.path.join(os.getcwd(), "data", "CrimeData_Final.csv")

def get_hotspot_data(crime_type="drugs"):
    """
    Generates hotspot predictions based on historical crime data.
    Returns a dictionary with crime_type and a list of predictions.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Filter by crime type if valid
    # Check if 'crime' column exists
    if 'crime' in df.columns:
        # Normalize crime types in data
        df['crime'] = df['crime'].astype(str).str.lower().str.strip()
        
        target_crime = crime_type.lower().strip()
        # Filter dataframe
        crime_df = df[df['crime'] == target_crime].copy()
        
        # If no data found for this crime, fallback to all or handle gracefully
        if crime_df.empty:
            # Fallback: use all data but warn (or just return empty)
            # For now, let's return based on all crimes but with lower scores if specific not found, 
            # or better, just return empty list to show no risk.
            pass 
    else:
        crime_df = df.copy()

    if crime_df.empty:
        return {
            "crime_type": crime_type,
            "predictions": []
        }

    # Group by GN PCode (gn_pcode)
    # The standard column seems to be 'gn_pcode' based on CSV view
    if 'gn_pcode' not in crime_df.columns:
         raise ValueError("CSV missing 'gn_pcode' column")

    # Calculate risk score based on frequency
    # A simple heuristic: count of incidents per GN
    gn_counts = crime_df['gn_pcode'].value_counts().reset_index()
    gn_counts.columns = ['gn_name', 'count']
    
    # Normalize to 0-1 range for risk_score
    max_count = gn_counts['count'].max()
    if max_count > 0:
        gn_counts['risk_score'] = gn_counts['count'] / max_count
    else:
        gn_counts['risk_score'] = 0.0

    # Sort by risk score descending
    gn_counts = gn_counts.sort_values('risk_score', ascending=False)
    
    # Format for output
    predictions = gn_counts[['gn_name', 'risk_score']].to_dict(orient="records")
    
    return {
        "crime_type": crime_type,
        "predictions": predictions
    }
