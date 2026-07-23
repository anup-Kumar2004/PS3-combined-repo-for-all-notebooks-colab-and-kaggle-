import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_models():
    models = {}

    models['soh_model']    = joblib.load(BASE_DIR / 'models' / 'soh_model_combined.pkl')
    models['soh_scaler']   = joblib.load(BASE_DIR / 'models' / 'scaler_soh_combined.pkl')
    models['soh_features'] = joblib.load(BASE_DIR / 'models' / 'soh_feature_cols_combined.pkl')

    models['rul_model']    = joblib.load(BASE_DIR / 'models' / 'rf_rul_combined.pkl')
    models['rul_scaler']   = joblib.load(BASE_DIR / 'models' / 'scaler_rul_combined.pkl')
    models['rul_features'] = joblib.load(BASE_DIR / 'models' / 'rul_feature_cols.pkl')

    models['battery_data'] = pd.read_csv(BASE_DIR / 'data' / 'features_final.csv')

    return models