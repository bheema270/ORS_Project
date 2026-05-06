# ============================================================
# FINAL: Climate-driven NDVI Prediction (Random Forest - SAVED MODEL)
# ============================================================

import os
import time
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import joblib   # ✅ NEW

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

start_total = time.time()

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
log("Loading data...")

DATA_DIR = "similipal_pixelwise"
files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

log(f"Raw data shape: {df.shape}")

# ------------------------------------------------------------
# 2. TIME SERIES AGGREGATION
# ------------------------------------------------------------
log("Aggregating to time-series...")

df["date"] = pd.to_datetime(df["date"])

cols = [
    "date",
    "NDVI",
    "LST_Day_C","LST_Night_C","LST_DTR",
    "rainfall_mm_8day",
    "ET_mm8day","PET_mm8day","ET_stress"
]

ts = df[cols].groupby("date").mean().reset_index()

# Time features
ts["year"]  = ts["date"].dt.year
ts["month"] = ts["date"].dt.month
ts["doy"]   = ts["date"].dt.dayofyear

log(f"After aggregation: {ts.shape}")

# ------------------------------------------------------------
# 3. SMOOTHING
# ------------------------------------------------------------
log("Smoothing NDVI...")

ts["NDVI"] = ts["NDVI"].rolling(window=5, center=True).mean()
ts = ts.dropna()

# ------------------------------------------------------------
# 4. LAG FEATURES
# ------------------------------------------------------------
log("Creating lag features...")

ts = ts.sort_values("date")

ts["NDVI_lag1"] = ts["NDVI"].shift(1)
ts["NDVI_lag2"] = ts["NDVI"].shift(2)

ts = ts.dropna()

log(f"After lag: {ts.shape}")

# ------------------------------------------------------------
# 5. FEATURES
# ------------------------------------------------------------
FEATURES = [
    "NDVI_lag1","NDVI_lag2",
    "LST_Day_C","LST_Night_C","LST_DTR",
    "rainfall_mm_8day",
    "ET_mm8day","PET_mm8day","ET_stress",
    "month","doy"
]

TARGET = "NDVI"

# ------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ------------------------------------------------------------
log("Splitting train/test...")

train = ts[ts["year"] <= 2020]
test  = ts[ts["year"] > 2020]

X_train = train[FEATURES]
y_train = train[TARGET]

X_test  = test[FEATURES]
y_test  = test[TARGET]

log(f"Train size: {len(train)}")
log(f"Test size : {len(test)}")

# ------------------------------------------------------------
# 7. MODEL
# ------------------------------------------------------------
log("Training Random Forest model...")

t0 = time.time()

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

log(f"Training time: {(time.time()-t0):.2f} sec")

# ------------------------------------------------------------
# ✅ SAVE MODEL
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/rf_model.pkl"
joblib.dump(model, MODEL_PATH)

log(f"Model saved → {MODEL_PATH}")

# ------------------------------------------------------------
# 8. PREDICTION
# ------------------------------------------------------------
log("Predicting...")

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 9. EVALUATION
# ------------------------------------------------------------
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== FINAL MODEL PERFORMANCE =====")
print(f"R²   : {r2:.3f}")
print(f"RMSE : {rmse:.4f}")

# ------------------------------------------------------------
# 10. SAVE OUTPUT
# ------------------------------------------------------------
os.makedirs("results_final", exist_ok=True)

out = test.copy()
out["NDVI_pred"] = y_pred
out.to_csv("results_final/predictions.csv", index=False)

log("Saved predictions → results_final/predictions.csv")

# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------------------------
log("Computing feature importance...")

imp = pd.Series(model.feature_importances_, index=FEATURES)
imp = imp.sort_values(ascending=False)

imp_percent = (imp / imp.sum()) * 100

print("\n===== FEATURE IMPORTANCE =====")
for f, val in imp_percent.items():
    print(f"{f:20s} → {val:6.2f}%")

print(f"\n[INSIGHT] Most important driver: {imp.index[0]}")

# ------------------------------------------------------------
# 12. TOTAL TIME
# ------------------------------------------------------------
log(f"TOTAL TIME: {(time.time()-start_total):.2f} sec")