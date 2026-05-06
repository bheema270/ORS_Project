# ============================================================
# FINAL PLOTS (Model Comparison + Insights)
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("results_final/predictions.csv")

os.makedirs("results_final/plots", exist_ok=True)

# -----------------------------
# V1: Predicted vs Actual (Best Model)
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(df["NDVI"], df["NDVI_pred"], s=20, alpha=0.6)

lims = [df["NDVI"].min(), df["NDVI"].max()]
plt.plot(lims, lims, 'r--')

r2 = r2_score(df["NDVI"], df["NDVI_pred"])

plt.title(f"Predicted vs Actual NDVI (R² = {r2:.3f})")
plt.xlabel("Actual NDVI")
plt.ylabel("Predicted NDVI")
plt.grid()

plt.savefig("results_final/plots/V1_scatter.png", dpi=150)
plt.close()


# -----------------------------
# V2: Time Series Comparison
# -----------------------------
df_sorted = df.sort_values("date")

plt.figure(figsize=(10,4))
plt.plot(df_sorted["date"], df_sorted["NDVI"], label="Actual", linewidth=2)
plt.plot(df_sorted["date"], df_sorted["NDVI_pred"], label="Predicted", linestyle="--")

plt.xticks(rotation=45)
plt.title("NDVI Time Series (Actual vs Predicted)")
plt.legend()
plt.grid()

plt.savefig("results_final/plots/V2_timeseries.png", dpi=150)
plt.close()


# -----------------------------
# V3: Residual Distribution
# -----------------------------
res = df["NDVI"] - df["NDVI_pred"]

plt.figure(figsize=(6,4))
plt.hist(res, bins=30)

plt.title("Residual Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid()

plt.savefig("results_final/plots/V3_residuals.png", dpi=150)
plt.close()


# -----------------------------
# V4: Feature Importance Plot
# -----------------------------
imp_df = pd.read_csv("results_final/feature_importance.csv")

plt.figure(figsize=(6,5))
plt.barh(imp_df["Feature"], imp_df["Contribution_%"])
plt.gca().invert_yaxis()

plt.title("Feature Importance (%)")
plt.xlabel("Contribution (%)")

plt.savefig("results_final/plots/V4_feature_importance.png", dpi=150)
plt.close()


# -----------------------------
# V5: Climate vs NDVI (Insight)
# -----------------------------
plt.figure(figsize=(6,5))
plt.scatter(df["rainfall_mm_8day"], df["NDVI"], s=15, alpha=0.6)

plt.title("NDVI vs Rainfall")
plt.xlabel("Rainfall")
plt.ylabel("NDVI")
plt.grid()

plt.savefig("results_final/plots/V5_ndvi_vs_rain.png", dpi=150)
plt.close()


print("✅ All plots saved in results_final/plots/")