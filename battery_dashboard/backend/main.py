from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from pathlib import Path

from model_loader import load_models
from schemas import SOHRequest, SOHResponse, RULResponse, BatteryHistoryResponse

app = FastAPI(title="Battery Health Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
frontend_dir = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

print("Loading models...")
models = load_models()
raw_df = models["battery_data"]
print("Computing temporal features for NASA batteries...")

def compute_temporal_features(df):
    df = df.sort_values(['battery_id', 'cycle_number']).copy()

    for col in ['duration', 'v_mean', 't_mean', 'i_mean']:
        df[f'{col}_roll5'] = (
            df.groupby('battery_id')[col]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df[f'{col}_trend'] = df[col] - df[f'{col}_roll5']

    df['SOH_prev'] = df.groupby('battery_id')['SOH'].shift(1)
    df['SOH_change'] = df['SOH'] - df['SOH_prev']
    df['SOH_roll5'] = (
        df.groupby('battery_id')['SOH']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df = df.dropna(subset=['SOH_prev']).reset_index(drop=True)
    return df

df_nasa = compute_temporal_features(raw_df)
df_nasa['dataset'] = 'NASA'

hust_path = BASE_DIR / 'data' / 'hust_features_final.csv'
df_hust = pd.read_csv(hust_path)
df_hust['dataset'] = 'HUST'

df = pd.concat([df_nasa, df_hust], ignore_index=True)
NASA_EOL_BATTERIES = {'B0005', 'B0006', 'B0007', 'B0018', 'B0039'}
print(f"Combined: {len(df_nasa)} NASA cycles + {len(df_hust)} HUST cycles")

# precompute EOL cycle per battery for RUL conversion
eol_map = {}
for bat in df['battery_id'].unique():
    bat_df = df[df['battery_id'] == bat].sort_values('cycle_number')
    eol_rows = bat_df[bat_df['SOH'] <= 0.80]
    if len(eol_rows) > 0:
        eol_cycle = int(eol_rows['cycle_number'].iloc[0])
    else:
        eol_cycle = int(bat_df['cycle_number'].max()) + 50
    min_cycle = int(bat_df['cycle_number'].min())
    eol_map[bat] = {'eol_cycle': eol_cycle, 'max_rul': max(1, eol_cycle - min_cycle)}

print("Ready.")

def get_health_status(soh):
    if soh >= 0.90:
        return "Healthy", "#2ecc71"
    elif soh >= 0.85:
        return "Monitor", "#f39c12"
    elif soh >= 0.80:
        return "Replace Soon", "#e74c3c"
    else:
        return "End of Life", "#8e44ad"

def rul_norm_to_cycles(battery_id, rul_norm):
    info = eol_map.get(battery_id, {'max_rul': 150})
    return int(round(float(rul_norm) * info['max_rul']))

@app.get("/")
def serve_frontend():
    return FileResponse(str(frontend_dir / "index.html"))

@app.get("/api/batteries")
def get_batteries():
    batteries = sorted(df["battery_id"].unique().tolist())
    return {"batteries": batteries}

@app.get("/api/battery/{battery_id}")
def get_battery_history(battery_id: str):
    bat_data = df[df["battery_id"] == battery_id].sort_values("cycle_number")
    if len(bat_data) == 0:
        raise HTTPException(status_code=404, detail="Battery not found")

    dataset  = bat_data["dataset"].iloc[0]
    is_nasa  = dataset == "NASA"
    has_rul  = (not is_nasa) or (battery_id in NASA_EOL_BATTERIES)

    soh_feature_cols = models["soh_features"]
    rul_feature_cols = models["rul_features"]

    # SOH prediction
    missing_soh = [c for c in soh_feature_cols if c not in bat_data.columns]
    if not missing_soh:
        X_soh        = bat_data[soh_feature_cols].copy()
        X_soh_scaled = models["soh_scaler"].transform(X_soh)
        soh_pred     = np.clip(models["soh_model"].predict(X_soh_scaled), 0, 1).tolist()
    else:
        soh_pred = [None] * len(bat_data)

    # RUL predicted
    if has_rul:
        missing_rul = [c for c in rul_feature_cols if c not in bat_data.columns]
        if not missing_rul:
            X_rul        = bat_data[rul_feature_cols].copy()
            X_rul_scaled = models["rul_scaler"].transform(X_rul)
            rul_norm     = np.clip(models["rul_model"].predict(X_rul_scaled), 0, 1)
            rul_pred     = [rul_norm_to_cycles(battery_id, v) for v in rul_norm]
        else:
            rul_pred = [None] * len(bat_data)
    else:
        rul_pred = [None] * len(bat_data)

    # RUL actual — compute from EOL cycle
    if has_rul and battery_id in eol_map:
        eol_cycle = eol_map[battery_id]['eol_cycle']
        rul_actual = [
            max(0, eol_cycle - int(c))
            for c in bat_data["cycle_number"]
        ]
    else:
        rul_actual = [None] * len(bat_data)

    return BatteryHistoryResponse(
        battery_id=battery_id,
        cycles=bat_data["cycle_number"].tolist(),
        soh_actual=bat_data["SOH"].tolist(),
        soh_predicted=soh_pred,
        rul_predicted=rul_pred,
        rul_actual=rul_actual,
        has_rul=has_rul,
        dataset=dataset,
    )


@app.post("/api/predict/soh", response_model=SOHResponse)
def predict_soh(req: SOHRequest):
    feature_cols = models["soh_features"]
    row = {col: getattr(req, col, 0.0) for col in feature_cols}
    X = pd.DataFrame([row])[feature_cols]
    X_scaled = models["soh_scaler"].transform(X)
    soh = float(np.clip(models["soh_model"].predict(X_scaled)[0], 0, 1))
    status, color = get_health_status(soh)
    return SOHResponse(
        battery_id=req.battery_id,
        cycle_number=req.cycle_number,
        soh_predicted=round(soh, 4),
        health_status=status,
        health_color=color,
    )

@app.get("/api/predict/rul/{battery_id}/{cycle_number}")
def predict_rul(battery_id: str, cycle_number: int):
    bat_data = df[df["battery_id"] == battery_id].sort_values("cycle_number")
    if len(bat_data) == 0:
        raise HTTPException(status_code=404, detail="Battery not found")

    row = bat_data[bat_data["cycle_number"] <= cycle_number]
    row = row.iloc[[-1]] if len(row) > 0 else bat_data.iloc[[0]]

    rul_feature_cols = models["rul_features"]
    missing = [c for c in rul_feature_cols if c not in row.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    X = row[rul_feature_cols]
    X_scaled = models["rul_scaler"].transform(X)
    rul_norm = float(np.clip(models["rul_model"].predict(X_scaled)[0], 0, 1))
    rul_cycles = rul_norm_to_cycles(battery_id, rul_norm)

    return RULResponse(
        battery_id=battery_id,
        cycle_number=cycle_number,
        rul_cycles=rul_cycles,
        days_light=round(rul_cycles / 0.5, 1),
        days_average=round(rul_cycles / 1.0, 1),
        days_heavy=round(rul_cycles / 2.0, 1),
    )