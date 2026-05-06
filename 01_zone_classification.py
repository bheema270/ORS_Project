# ============================================================
# FILE 1: 01_zone_classification.py (UPDATED)
# Zone Classification — Similipal Core Area Phenological Study
# ============================================================

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────

DATA_DIR = "similipal_pixelwise"   # 🔥 folder containing 2001–2024 CSVs
OUTPUT_ZONES = "zone_labels.csv"
PLOT_DIR = "plots/zones"

OTHERS_PCT = 1.0
NDWI_WEIGHT = 0.5
NDMI_WEIGHT = 0.5

ZONE_COLORS = {"Wet": "#1a78c2", "Dry": "#d4701a", "Others": "#888888"}

os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 1. LOAD ALL CSV FILES
# ────────────────────────────────────────────────────────────

print("[INFO] Loading multi-year data...")

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

if len(files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

df_list = []
for f in files:
    temp = pd.read_csv(f, parse_dates=["date"])
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print(f"       Loaded {len(files)} files")
print(f"       {len(df):,} records | "
      f"{df['date'].dt.year.nunique()} years | "
      f"{df[['lon','lat']].drop_duplicates().shape[0]} unique pixels")

# ────────────────────────────────────────────────────────────
# 2. PER-PIXEL STATISTICS
# ────────────────────────────────────────────────────────────

print("[INFO] Computing per-pixel moisture statistics...")

pixel_stats = (
    df.groupby(["lon", "lat"])
    .agg(
        mean_NDWI=("NDWI", "mean"),
        std_NDWI=("NDWI", "std"),
        mean_NDMI=("NDMI", "mean"),
        std_NDMI=("NDMI", "std"),
        mean_NDVI=("NDVI", "mean"),
        mean_EVI=("EVI", "mean"),
        mean_LST=("LST_Day_C", "mean"),
        n_obs=("NDVI", "count"),
    )
    .reset_index()
)

# ────────────────────────────────────────────────────────────
# 3. NORMALIZATION + MOISTURE COMPOSITE
# ────────────────────────────────────────────────────────────

def minmax_norm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0

pixel_stats["norm_NDWI"] = minmax_norm(pixel_stats["mean_NDWI"])
pixel_stats["norm_NDMI"] = minmax_norm(pixel_stats["mean_NDMI"])

pixel_stats["moisture_composite"] = (
    NDWI_WEIGHT * pixel_stats["norm_NDWI"] +
    NDMI_WEIGHT * pixel_stats["norm_NDMI"]
)

# ────────────────────────────────────────────────────────────
# 4. CLASSIFY ZONES
# ────────────────────────────────────────────────────────────

print(f"[INFO] Classifying zones (Others = {OTHERS_PCT:.1f}%)")

low_pct = 50.0 - OTHERS_PCT / 2.0
high_pct = 50.0 + OTHERS_PCT / 2.0

low_thresh = np.percentile(pixel_stats["moisture_composite"], low_pct)
high_thresh = np.percentile(pixel_stats["moisture_composite"], high_pct)

def classify_zone(val):
    if val > high_thresh:
        return "Wet"
    elif val < low_thresh:
        return "Dry"
    else:
        return "Others"

pixel_stats["zone"] = pixel_stats["moisture_composite"].apply(classify_zone)

# Summary
zone_counts = pixel_stats["zone"].value_counts()
total_px = len(pixel_stats)

print("\nZone distribution:")
for z in ["Wet", "Dry", "Others"]:
    n = zone_counts.get(z, 0)
    print(f"{z:8s}: {n:6d} ({100*n/total_px:.1f}%)")

print(f"\nThresholds → Dry < {low_thresh:.4f} < Others < {high_thresh:.4f} < Wet")

# ────────────────────────────────────────────────────────────
# 5. SAVE OUTPUT
# ────────────────────────────────────────────────────────────

zone_out = pixel_stats[[
    "lon", "lat", "zone",
    "moisture_composite",
    "mean_NDWI", "mean_NDMI",
    "mean_NDVI", "mean_EVI",
    "mean_LST", "n_obs"
]]

zone_out.to_csv(OUTPUT_ZONES, index=False)

print(f"\n[INFO] Saved → {OUTPUT_ZONES}")

# ────────────────────────────────────────────────────────────
# 6. VISUALIZATION (KEY ONE ONLY — CLEAN)
# ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

for zone in ["Dry", "Wet", "Others"]:
    grp = pixel_stats[pixel_stats["zone"] == zone]
    ax.scatter(
        grp["lon"], grp["lat"],
        c=ZONE_COLORS[zone],
        s=15, alpha=0.7,
        label=zone
    )

ax.set_title("Zone Classification (Similipal Core)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/zone_map.png", dpi=150)
plt.close()

print(f"[PLOT] zone_map.png")

# ────────────────────────────────────────────────────────────
# DONE
# ────────────────────────────────────────────────────────────

print("\n[DONE] Zone classification complete.")
print("Next → Run: 02_phenology_analysis.py")