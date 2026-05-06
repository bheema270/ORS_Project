# ============================================================
# FILE 2: 02_phenology_analysis.py
# Comprehensive Phenological Analysis — Similipal Core Area
# Study period: 2001–2024 | Phenological window: March→March
# ============================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
#  • Merges zone labels from 01_zone_classification.py
#  • Re-maps calendar dates to a March-1-anchored phenological DOY
#  • Spatially aggregates all variables per (zone, phenoYear, phenoDOY)
#  • Smooths NDVI with Savitzky-Golay filter
#  • Detects SOS / EOS / LOS with zone-specific amplitude thresholds
#  • Runs Mann-Kendall + Sen's slope on all phenological metrics
#  • Produces every standard visualisation published in phenology papers
#
# OUTPUT FOLDERS
#  plots/phenology/A_*   → Seasonal climatological profiles
#  plots/phenology/B_*   → Inter-annual phenological trends
#  plots/phenology/C_*   → Heatmaps & anomaly analyses
#  plots/phenology/D_*   → Climate–phenology relationships
#  plots/phenology/E_*   → Statistical summaries (MK, violin, CV)
#  plots/phenology/F_*   → Ecosystem fluxes (GPP, ET, LST, PsnNet)
#  phenology_metrics.csv → Per-(zone, year) table of all metrics
#  mannkendall_results.csv
#
# USAGE
#   python 02_phenology_analysis.py
# ============================================================

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from scipy import stats

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
ZONES_FILE  = "zone_labels.csv"          # from 01_zone_classification.py
OUT_DIR     = "plots/phenology"
RESULTS_CSV = "phenology_metrics.csv"
MK_CSV      = "mannkendall_results.csv"

FIRST_YEAR  = 2001
LAST_YEAR   = 2024

# Zone-specific amplitude thresholds for SOS/EOS detection
# Dry deciduous sheds its leaves fully and rapidly (needs higher threshold to catch true flush)
# Wet/Moist deciduous retains deeper canopy longer (needs lower threshold for its long season)
THRESHOLDS  = {"Wet": 0.20, "Dry": 0.35, "Others": 0.25}

# Savitzky-Golay smoothing (applied within each phenological year)
SG_WINDOW   = 7    # must be odd; 7 × 8-day = ~56-day smoothing window
SG_POLY     = 3

# Colour palette (consistent with file 1)
ZONE_COLORS = {"Wet": "#1a78c2", "Dry": "#d4701a", "Others": "#888888"}
ZONES_MAIN  = ["Wet", "Dry"]

# Month tick positions (phenoDOY 1 = March 1)
MONTH_TICKS  = [1, 32, 62, 93, 123, 154, 185, 215, 246, 276, 307, 336]
MONTH_LABELS = ["Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb"]

os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def pheno_year_doy(date_series):
    """
    Convert calendar dates to phenological year and DOY.
    Phenological year starts March 1 (phenoDOY = 1).
    Jan–Feb dates belong to the *previous* phenological year.
    """
    year_arr = date_series.dt.year.values
    month_arr = date_series.dt.month.values
    pheno_year = np.where(month_arr >= 3, year_arr, year_arr - 1)
    march1 = pd.to_datetime(
        [f"{y}-03-01" for y in pheno_year], format="%Y-%m-%d"
    )
    pheno_doy = (date_series.values - march1.values).astype("timedelta64[D]").astype(int) + 1
    return pheno_year, pheno_doy


def smooth_sg(y, window=SG_WINDOW, poly=SG_POLY):
    """Savitzky-Golay, gracefully handles short series."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < window or n < poly + 2:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=poly)


def detect_sos_eos(doys, ndvi_smooth, threshold_frac):
    """
    Amplitude-fraction method for SOS and EOS detection.

    Parameters
    ----------
    doys           : 1-D array of phenoDOY values (sorted ascending)
    ndvi_smooth    : smoothed NDVI at those DOYs
    threshold_frac : fraction of seasonal amplitude above baseline
                     (Wet ≈ 0.35, Dry ≈ 0.20)

    Returns
    -------
    sos_doy, eos_doy, peak_doy, peak_ndvi, baseline, amplitude
    """
    nan6 = (np.nan,) * 6
    if len(doys) < 5:
        return nan6

    baseline  = np.nanpercentile(ndvi_smooth, 10)
    peak_ndvi = np.nanmax(ndvi_smooth)
    amplitude = peak_ndvi - baseline

    if amplitude < 0.05:        # essentially flat time series
        return nan6

    threshold = baseline + threshold_frac * amplitude
    peak_idx  = np.nanargmax(ndvi_smooth)
    peak_doy  = doys[peak_idx]

    # SOS = first ascending crossing of threshold before the peak
    sos_doy = np.nan
    for i in range(peak_idx):
        if ndvi_smooth[i] >= threshold:
            sos_doy = doys[i]
            break

    # EOS = last point above threshold after the peak (descending limb)
    eos_doy = np.nan
    for i in range(len(doys) - 1, peak_idx, -1):
        if ndvi_smooth[i] >= threshold:
            eos_doy = doys[i]
            break

    return sos_doy, eos_doy, peak_doy, peak_ndvi, baseline, amplitude


def mann_kendall(series):
    """
    Simple Mann-Kendall trend test.
    Returns (trend_str, z, p_value, S_statistic)
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return "no trend", 0.0, 1.0, 0

    S = int(sum(
        np.sign(x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    ))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = (S - np.sign(S)) / np.sqrt(var_s)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    trend = "increasing" if z > 0 else ("decreasing" if z < 0 else "no trend")
    return trend, round(z, 3), round(p, 4), S


def sens_slope(years, values):
    """Theil-Sen slope estimator (units per year)."""
    arr = np.column_stack([years, values])
    arr = arr[~np.isnan(arr[:, 1])]
    if len(arr) < 2:
        return np.nan
    yr, vl = arr[:, 0], arr[:, 1]
    slopes = [
        (vl[j] - vl[i]) / (yr[j] - yr[i])
        for i in range(len(yr) - 1)
        for j in range(i + 1, len(yr))
        if yr[j] != yr[i]
    ]
    return float(np.nanmedian(slopes)) if slopes else np.nan


def add_linreg_trend(ax, years, values, color="black", lw=2.0):
    """Add OLS trend line; return (slope, p_value)."""
    mask = ~np.isnan(np.asarray(values, float))
    yr = np.asarray(years, float)[mask]
    vl = np.asarray(values, float)[mask]
    if len(yr) < 4:
        return np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(yr, vl)
    xfit = np.array([yr.min(), yr.max()])
    ax.plot(xfit, slope * xfit + intercept, "--", color=color,
            lw=lw, alpha=0.85)
    return slope, p


def set_month_xaxis(ax):
    """Replace numeric phenoDOY x-axis with month labels."""
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlabel("Month (Mar → Feb)", fontsize=10)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ════════════════════════════════════════════════════════════
# 1. LOAD DATA & MERGE ZONES (MULTI-FILE)
# ════════════════════════════════════════════════════════════

import os
import glob
import numpy as np
import pandas as pd

print("[INFO] Loading multi-year data ...")

DATA_DIR = "similipal_pixelwise"

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

# ── Merge zones ─────────────────────────────────────────────
zones = pd.read_csv(ZONES_FILE)

df = df.merge(zones[["lon", "lat", "zone"]], on=["lon", "lat"], how="left")
df["zone"] = df["zone"].fillna("Others")

# ── Phenological year & DOY ──────────────────────────────────
print("[INFO] Computing phenological year & DOY (March 1 = phenoDOY 1) ...")

df["pheno_year"], df["pheno_doy"] = pheno_year_doy(df["date"])

df = df[(df["pheno_year"] >= FIRST_YEAR) & (df["pheno_year"] <= LAST_YEAR)]

print(f"       {len(df):,} records | "
      f"phenoYears {FIRST_YEAR}–{LAST_YEAR} | "
      f"zones: {df['zone'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════
# 2. ZONE-WISE SPATIAL AGGREGATION (VERY IMPORTANT FIRST)
# ════════════════════════════════════════════════════════════
print("[INFO] Aggregating spatially by zone ...")

AGG_COLS = [
    "NDVI", "EVI", "NDWI", "NDMI",
    "LST_Day_C", "LST_Night_C", "LST_DTR",
    "ET_mm8day", "LE_MJm2day", "PET_mm8day", "ET_stress",
    "GPP_gCm2_8day", "PsnNet_gCm2_8day", "rainfall_mm_8day",
]

ts = (
    df.groupby(["zone", "pheno_year", "pheno_doy"])[AGG_COLS]
    .median()
    .reset_index()
    .sort_values(["zone", "pheno_year", "pheno_doy"])
    .reset_index(drop=True)
)

# ════════════════════════════════════════════════════════════
# 3. FIX TIME SERIES (CONTINUOUS DOY + INTERPOLATION)
# ════════════════════════════════════════════════════════════

print("[INFO] Fixing missing values & ensuring continuous DOY...")

full_doy = np.arange(1, 366)

def reindex_group(grp):
    zone_val = grp["zone"].iloc[0]
    year_val = grp["pheno_year"].iloc[0]

    grp = grp.groupby("pheno_doy").mean(numeric_only=True)
    grp = grp.reindex(full_doy)

    # restore identifiers properly
    grp["zone"] = zone_val
    grp["pheno_year"] = year_val

    grp = grp.ffill().bfill()

    return grp

ts = (
    ts.groupby(["zone", "pheno_year"], group_keys=False)
    .apply(reindex_group)
    .reset_index()
    .rename(columns={"index": "pheno_doy"})
)
print("[DEBUG] ts columns:", ts.columns)

# Final interpolation (smooth gaps)
for col in AGG_COLS:
    ts[col] = ts.groupby(["zone", "pheno_year"])[col].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

# Clip NDVI to valid range
if "NDVI" in ts.columns:
    ts["NDVI"] = ts["NDVI"].clip(0, 1)

print("[INFO] Missing data handled.")

# ════════════════════════════════════════════════════════════
# 4. READY FOR SMOOTHING (NEXT STEP)
# ════════════════════════════════════════════════════════════
print("[INFO] Time series ready for smoothing and phenology extraction.")

# ════════════════════════════════════════════════════════════
# 3.  SMOOTH NDVI (Savitzky-Golay)
# ════════════════════════════════════════════════════════════
print("[INFO] Smoothing NDVI with Savitzky-Golay ...")

def sg_smooth_group(grp):
    grp = grp.sort_values("pheno_doy").copy()
    grp["NDVI_smooth"] = smooth_sg(grp["NDVI"].values)
    return grp

ts = (
    ts.groupby(["zone", "pheno_year"], group_keys=False)
    .apply(sg_smooth_group)
    .reset_index(drop=True)
)

# ════════════════════════════════════════════════════════════
# 4.  SOS / EOS / LOS DETECTION
# ════════════════════════════════════════════════════════════
print("[INFO] Detecting SOS, EOS, LOS ...")

records = []
for (zone, pyear), grp in ts.groupby(["zone", "pheno_year"]):
    grp   = grp.sort_values("pheno_doy")
    doys  = grp["pheno_doy"].values
    ndvi_s = grp["NDVI_smooth"].values
    thr   = THRESHOLDS.get(zone, 0.25)

    sos, eos, pk_doy, pk_ndvi, baseline, amplitude = detect_sos_eos(
        doys, ndvi_s, thr
    )
    los = (eos - sos) if (not np.isnan(eos) and not np.isnan(sos)) else np.nan

    # Integrated NDVI (season-long vegetation productivity proxy)
    int_ndvi = np.trapz(ndvi_s, doys) if len(doys) > 1 else np.nan

    # Climate aggregates
    pre_sos_rain = grp.loc[grp["pheno_doy"] <= 60, "rainfall_mm_8day"].sum()
    season_rain  = grp["rainfall_mm_8day"].sum()
    mean_lst_day = grp["LST_Day_C"].mean()
    mean_lst_nit = grp["LST_Night_C"].mean()
    mean_dtr     = grp["LST_DTR"].mean()
    mean_et      = grp["ET_mm8day"].mean()
    mean_et_stress = grp["ET_stress"].mean()
    total_gpp    = grp["GPP_gCm2_8day"].sum()
    total_psn    = grp["PsnNet_gCm2_8day"].sum()
    mean_ndwi    = grp["NDWI"].mean()
    mean_ndmi    = grp["NDMI"].mean()
    mean_evi     = grp["EVI"].mean()

    records.append(dict(
        zone=zone, pheno_year=pyear,
        SOS=sos, EOS=eos, LOS=los,
        peak_doy=pk_doy, peak_ndvi=pk_ndvi,
        baseline_ndvi=baseline, amplitude=amplitude,
        integrated_NDVI=int_ndvi,
        pre_sos_rain_mm=pre_sos_rain, season_rain_mm=season_rain,
        mean_LST_Day=mean_lst_day,  mean_LST_Night=mean_lst_nit,
        mean_DTR=mean_dtr,
        mean_ET=mean_et, mean_ET_stress=mean_et_stress,
        annual_GPP=total_gpp, annual_PsnNet=total_psn,
        mean_NDWI=mean_ndwi, mean_NDMI=mean_ndmi, mean_EVI=mean_evi,
    ))

metrics = pd.DataFrame(records)
metrics.to_csv(RESULTS_CSV, index=False)
print(f"[INFO] Phenology metrics → {RESULTS_CSV}")
print("\n  Per-zone summary of key metrics:")
print(
    metrics.groupby("zone")[["SOS","EOS","LOS","peak_ndvi","integrated_NDVI"]]
    .agg(["mean","std"]).round(2).to_string()
)

# ════════════════════════════════════════════════════════════
# DECADE LABEL (used in boxplots)
# ════════════════════════════════════════════════════════════
DECADES = {"2001–2010": (2001,2010), "2011–2020": (2011,2020), "2021–2024": (2021,2024)}
def assign_decade(yr):
    for label, (s, e) in DECADES.items():
        if s <= yr <= e: return label
    return "Other"
metrics["decade"] = metrics["pheno_year"].apply(assign_decade)

# ════════════════════════════════════════════════════════════
# PHENOLOGICAL VARIABLE DEFINITIONS (used in loops)
# ════════════════════════════════════════════════════════════
PHENO_VARS = [
    ("SOS",           "SOS (phenoDOY)",          "Start of Season"),
    ("EOS",           "EOS (phenoDOY)",           "End of Season"),
    ("LOS",           "LOS (days)",               "Length of Season"),
    ("peak_doy",      "Peak DOY (phenoDOY)",       "Peak Timing"),
    ("peak_ndvi",     "Peak NDVI",                "Peak NDVI"),
    ("integrated_NDVI","Integrated NDVI",         "Seasonal Productivity"),
]

years_all = sorted(metrics["pheno_year"].unique())

# ════════════════════════════════════════════════════════════
# ██████████  VIZ BLOCK A — SEASONAL PROFILES  ██████████
# ════════════════════════════════════════════════════════════
print("\n[INFO] [A] Plotting seasonal profiles ...")

# ── Long-term mean climatology ────────────────────────────────
clim = (
    ts.groupby(["zone", "pheno_doy"])["NDVI_smooth"]
    .agg(mean_ndvi="mean", std_ndvi="std")
    .reset_index()
)

# A1: Long-term mean seasonal profile ± 1 SD
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ["Wet", "Dry", "Others"]:
    sub = clim[clim["zone"] == zone].sort_values("pheno_doy")
    ax.plot(sub["pheno_doy"], sub["mean_ndvi"],
            color=ZONE_COLORS[zone], lw=2.5, label=zone)
    ax.fill_between(sub["pheno_doy"],
                    sub["mean_ndvi"] - sub["std_ndvi"],
                    sub["mean_ndvi"] + sub["std_ndvi"],
                    color=ZONE_COLORS[zone], alpha=0.15)
set_month_xaxis(ax)
ax.set_ylabel("Mean NDVI", fontsize=11)
ax.set_title(
    "Long-term Mean Seasonal NDVI Profile — Similipal Core (2001–2024)\n"
    "Shaded region = ±1 SD of inter-annual variability",
    fontsize=13
)
ax.legend(fontsize=10)
ax.set_xlim(1, 365)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A1_seasonal_climatology.png", dpi=150)
plt.close()
print("  A1_seasonal_climatology.png")

# A2: Spaghetti plot — individual year curves per zone
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
cmap_years = cm.get_cmap("plasma", len(years_all))

for ax, zone in zip(axes, ZONES_MAIN):
    sub_ts = ts[ts["zone"] == zone]
    for i, yr in enumerate(years_all):
        ydata = sub_ts[sub_ts["pheno_year"] == yr].sort_values("pheno_doy")
        ax.plot(ydata["pheno_doy"], ydata["NDVI_smooth"],
                color=cmap_years(i), lw=0.9, alpha=0.55)
    mean_c = clim[clim["zone"] == zone].sort_values("pheno_doy")
    ax.plot(mean_c["pheno_doy"], mean_c["mean_ndvi"],
            color="black", lw=2.5, label="2001–2024 mean")
    set_month_xaxis(ax)
    ax.set_ylabel("NDVI", fontsize=11)
    ax.set_title(f"{zone} Zone — Annual NDVI Profiles", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(1, 365)
    ax.grid(True, alpha=0.3)

sm = cm.ScalarMappable(cmap="plasma",
     norm=mcolors.Normalize(vmin=years_all[0], vmax=years_all[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01, shrink=0.9)
cbar.set_label("Phenological Year", fontsize=10)
plt.suptitle(
    "Inter-annual NDVI Seasonal Profiles — Similipal Core (2001–2024)\n"
    "Colour gradient: early years (purple) → recent years (yellow)",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A2_spaghetti_ndvi.png", dpi=150)
plt.close()
print("  A2_spaghetti_ndvi.png")

# A3: Multi-variable climatological seasonality
climate_clim_vars = [
    ("rainfall_mm_8day",  "Rainfall (mm/8d)",       "#4a90d9"),
    ("ET_mm8day",         "ET (mm/8d)",              "#2ca02c"),
    ("LST_Day_C",         "LST Day (°C)",            "#d62728"),
    ("GPP_gCm2_8day",     "GPP (gC/m²/8d)",         "#9467bd"),
]
fig, axes = plt.subplots(2, 2, figsize=(17, 10))
for ax, (var, ylabel, color) in zip(axes.flat, climate_clim_vars):
    for zone in ZONES_MAIN:
        sub = ts[ts["zone"] == zone].groupby("pheno_doy")[var].mean()
        ax.plot(sub.index, sub.values,
                color=ZONE_COLORS[zone], lw=2, label=zone)
    set_month_xaxis(ax)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Mean Seasonal {ylabel}", fontsize=11)
    ax.set_xlim(1, 365)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle(
    "Climatological Seasonal Profiles — Climate Variables by Zone\n"
    "Similipal Core (2001–2024 mean)",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A3_climate_seasonality.png", dpi=150)
plt.close()
print("  A3_climate_seasonality.png")

# A4: NDVI + Rainfall dual-axis
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub  = ts[ts["zone"] == zone].groupby("pheno_doy")[
        ["NDVI_smooth", "rainfall_mm_8day"]
    ].mean()
    ax2  = ax.twinx()
    ln1, = ax.plot(sub.index, sub["NDVI_smooth"],
                   color=ZONE_COLORS[zone], lw=2.5, label="NDVI")
    ln2, = ax2.plot(sub.index, sub["rainfall_mm_8day"],
                    color="#4a90d9", lw=2, ls="--", label="Rainfall")
    ax.set_ylabel("NDVI", fontsize=11, color=ZONE_COLORS[zone])
    ax2.set_ylabel("Rainfall (mm/8day)", fontsize=11, color="#4a90d9")
    ax.set_title(f"{zone} Zone — NDVI & Rainfall Seasonality", fontsize=12)
    set_month_xaxis(ax)
    ax.set_xlim(1, 365)
    ax.legend(handles=[ln1, ln2], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
plt.suptitle(
    "NDVI vs Rainfall Seasonal Co-variation\nSimilipal Core (2001–2024 mean)",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A4_ndvi_rainfall_dual.png", dpi=150)
plt.close()
print("  A4_ndvi_rainfall_dual.png")

# A5: NDVI + LST dual-axis
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = ts[ts["zone"] == zone].groupby("pheno_doy")[
        ["NDVI_smooth", "LST_Day_C", "LST_Night_C"]
    ].mean()
    ax2 = ax.twinx()
    ln1, = ax.plot(sub.index, sub["NDVI_smooth"],
                   color=ZONE_COLORS[zone], lw=2.5, label="NDVI")
    ln2, = ax2.plot(sub.index, sub["LST_Day_C"],
                    color="#d62728", lw=2, ls="--", label="LST Day")
    ln3, = ax2.plot(sub.index, sub["LST_Night_C"],
                    color="#ff7f0e", lw=1.5, ls=":", label="LST Night")
    ax.set_ylabel("NDVI", fontsize=11, color=ZONE_COLORS[zone])
    ax2.set_ylabel("LST (°C)", fontsize=11, color="#d62728")
    ax.set_title(f"{zone} Zone — NDVI & LST Seasonality", fontsize=12)
    set_month_xaxis(ax)
    ax.set_xlim(1, 365)
    ax.legend(handles=[ln1, ln2, ln3], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
plt.suptitle(
    "NDVI vs Land Surface Temperature Seasonal Profile\nSimilipal Core (2001–2024 mean)",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A5_ndvi_lst_dual.png", dpi=150)
plt.close()
print("  A5_ndvi_lst_dual.png")

# A6: EVI & NDMI seasonal profile
fig, ax = plt.subplots(figsize=(13, 5))
evi_clim = ts.groupby(["zone","pheno_doy"])[["EVI","NDMI"]].mean().reset_index()
for zone in ZONES_MAIN:
    sub = evi_clim[evi_clim["zone"]==zone].sort_values("pheno_doy")
    ax.plot(sub["pheno_doy"], sub["EVI"],
            color=ZONE_COLORS[zone], lw=2, label=f"{zone} EVI")
    ax.plot(sub["pheno_doy"], sub["NDMI"],
            color=ZONE_COLORS[zone], lw=1.5, ls="--",
            label=f"{zone} NDMI")
set_month_xaxis(ax)
ax.set_ylabel("Index Value", fontsize=11)
ax.set_title("EVI & NDMI Seasonal Profiles — Similipal Core", fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.set_xlim(1, 365)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/A6_evi_ndmi_seasonal.png", dpi=150)
plt.close()
print("  A6_evi_ndmi_seasonal.png")


# ════════════════════════════════════════════════════════════
# ██  VIZ BLOCK B — INTER-ANNUAL PHENOLOGICAL TRENDS  ██
# ════════════════════════════════════════════════════════════
print("\n[INFO] [B] Plotting phenological metric trends ...")

# B1-B6: One figure per metric (Wet / Dry side by side)
for var, ylabel, title in PHENO_VARS:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, zone in zip(axes, ZONES_MAIN):
        sub  = metrics[metrics["zone"] == zone].sort_values("pheno_year")
        vals = sub[var].values
        yrs  = sub["pheno_year"].values
        ax.scatter(yrs, vals, color=ZONE_COLORS[zone], s=60, zorder=3, edgecolors="white", lw=0.5)
        ax.plot(yrs, vals, color=ZONE_COLORS[zone], lw=1, alpha=0.45)

        mask = ~np.isnan(vals)
        if mask.sum() >= 4:
            slope, p = add_linreg_trend(ax, yrs[mask], vals[mask],
                                        color="black", lw=2.0)
            trend, z, pval, _ = mann_kendall(vals[mask])
            ss = sens_slope(yrs[mask], vals[mask])
            info = (f"OLS slope: {slope:+.2f}/yr  (p={p:.3f} {sig_stars(p)})\n"
                    f"MK: {trend}  z={z:.2f}, p={pval:.3f}\n"
                    f"Sen's slope: {ss:+.2f}/yr")
            ax.text(0.03, 0.97, info, transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.set_xlabel("Phenological Year", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{zone} Zone", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(FIRST_YEAR - 0.5, LAST_YEAR + 0.5)
    plt.suptitle(
        f"{title} Trend — Similipal Core Area (2001–2024)\n"
        f"Threshold: Wet={THRESHOLDS['Wet']}, Dry={THRESHOLDS['Dry']} (amplitude fraction)",
        fontsize=13
    )
    plt.tight_layout()
    fname = var.lower()
    plt.savefig(f"{OUT_DIR}/B_{fname}_trend.png", dpi=150)
    plt.close()
    print(f"  B_{fname}_trend.png")

# B7: Combined SOS + EOS + LOS on one figure (all zones)
fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
for ax, (var, ylabel, title) in zip(axes, PHENO_VARS[:3]):
    for zone in ZONES_MAIN:
        sub  = metrics[metrics["zone"] == zone].sort_values("pheno_year")
        ax.plot(sub["pheno_year"], sub[var], "o-",
                color=ZONE_COLORS[zone], lw=1.5, ms=5, label=zone)
        mask = ~sub[var].isna()
        if mask.sum() >= 4:
            add_linreg_trend(ax,
                sub["pheno_year"][mask].values,
                sub[var][mask].values,
                color=ZONE_COLORS[zone], lw=2.0)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Phenological Year", fontsize=11)
plt.suptitle(
    "Phenological Shifts — SOS, EOS, Length of Season (2001–2024)\n"
    "Similipal Core Area | Dashed lines = OLS trend",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/B_sos_eos_los_combined.png", dpi=150)
plt.close()
print("  B_sos_eos_los_combined.png")

# B8: Decadal boxplots for SOS / EOS / LOS
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
decade_labels = list(DECADES.keys())

for ax, (var, ylabel, title) in zip(axes, PHENO_VARS[:3]):
    x_pos = np.arange(len(decade_labels))
    width = 0.35
    for i, zone in enumerate(ZONES_MAIN):
        sub = metrics[metrics["zone"] == zone]
        groups = [sub[sub["decade"] == d][var].dropna().values for d in decade_labels]
        offset = (i - 0.5) * width
        bp = ax.boxplot(
            groups,
            positions=x_pos + offset,
            widths=width * 0.85,
            patch_artist=True,
            boxprops    =dict(facecolor=ZONE_COLORS[zone], alpha=0.65),
            medianprops =dict(color="black", lw=2),
            whiskerprops=dict(color=ZONE_COLORS[zone]),
            capprops    =dict(color=ZONE_COLORS[zone]),
            flierprops  =dict(marker="o", color=ZONE_COLORS[zone], alpha=0.5, ms=4),
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(decade_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

handles = [Patch(facecolor=ZONE_COLORS[z], label=z) for z in ZONES_MAIN]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11)
plt.suptitle(
    "Decadal Shifts in Phenological Metrics — Similipal Core\n"
    "(2001–2010 / 2011–2020 / 2021–2024)",
    fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/B_decadal_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  B_decadal_boxplots.png")


# ════════════════════════════════════════════════════════════
# ██  VIZ BLOCK C — HEATMAPS & ANOMALIES  ██
# ════════════════════════════════════════════════════════════
print("\n[INFO] [C] Plotting heatmaps and anomalies ...")

# C1: NDVI Phenolandscape (Year × phenoDOY)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, zone in zip(axes, ZONES_MAIN):
    sub    = ts[ts["zone"] == zone]
    pivot  = sub.pivot_table(
        index="pheno_year", columns="pheno_doy",
        values="NDVI_smooth", aggfunc="mean"
    )
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn",
        vmin=0.2, vmax=0.9,
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.max() + 0.5, pivot.index.min() - 0.5]
    )
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8, rotation=30)
    ax.set_xlabel("Month (Mar → Feb)", fontsize=10)
    ax.set_ylabel("Phenological Year", fontsize=10)
    ax.set_title(f"{zone} Zone — NDVI Phenolandscape", fontsize=12)
    plt.colorbar(im, ax=ax, label="NDVI", shrink=0.85)

plt.suptitle(
    "NDVI Phenolandscape — Similipal Core (2001–2024)\n"
    "Year × Phenological DOY | Phenological year starts March 1",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/C1_phenolandscape_heatmap.png", dpi=150)
plt.close()
print("  C1_phenolandscape_heatmap.png")

# C2: NDVI Anomaly Heatmap (departure from long-term DOY mean)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, zone in zip(axes, ZONES_MAIN):
    sub   = ts[ts["zone"] == zone]
    pivot = sub.pivot_table(
        index="pheno_year", columns="pheno_doy",
        values="NDVI_smooth", aggfunc="mean"
    )
    doy_mean = pivot.mean(axis=0)
    anomaly  = pivot.subtract(doy_mean, axis=1)
    im = ax.imshow(
        anomaly.values, aspect="auto", cmap="RdBu",
        vmin=-0.15, vmax=0.15,
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.max() + 0.5, pivot.index.min() - 0.5]
    )
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8, rotation=30)
    ax.set_xlabel("Month (Mar → Feb)", fontsize=10)
    ax.set_ylabel("Phenological Year", fontsize=10)
    ax.set_title(f"{zone} Zone — NDVI Anomaly", fontsize=12)
    plt.colorbar(im, ax=ax, label="NDVI Anomaly (Blue=above / Red=below mean)", shrink=0.85)

plt.suptitle(
    "NDVI Anomaly from Long-term Mean — Similipal Core (2001–2024)\n"
    "Each cell = departure from the 24-year mean NDVI for that phenoDOY",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/C2_ndvi_anomaly_heatmap.png", dpi=150)
plt.close()
print("  C2_ndvi_anomaly_heatmap.png")

# C3: Annual NDVI anomaly bar chart
annual_ndvi = (
    ts.groupby(["zone", "pheno_year"])["NDVI_smooth"]
    .mean()
    .reset_index(name="mean_NDVI")
)
for zone in ZONES_MAIN:
    mask      = annual_ndvi["zone"] == zone
    gm        = annual_ndvi.loc[mask, "mean_NDVI"].mean()
    annual_ndvi.loc[mask, "anomaly"] = annual_ndvi.loc[mask, "mean_NDVI"] - gm

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
for ax, zone in zip(axes, ZONES_MAIN):
    sub = annual_ndvi[annual_ndvi["zone"] == zone].sort_values("pheno_year")
    colors = [ZONE_COLORS[zone] if v >= 0 else "#aaaaaa" for v in sub["anomaly"]]
    ax.bar(sub["pheno_year"], sub["anomaly"], color=colors, alpha=0.8, edgecolor="black", lw=0.3)
    ax.axhline(0, color="black", lw=1)
    add_linreg_trend(ax, sub["pheno_year"].values, sub["anomaly"].values, color="darkred", lw=1.5)
    ax.set_ylabel("NDVI Anomaly", fontsize=10)
    ax.set_title(f"{zone} Zone", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
axes[-1].set_xlabel("Phenological Year", fontsize=11)
plt.suptitle(
    "Annual Mean NDVI Anomaly — Similipal Core (2001–2024)\n"
    "Positive = above average greenness | Dashed = trend",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/C3_annual_ndvi_anomaly.png", dpi=150)
plt.close()
print("  C3_annual_ndvi_anomaly.png")

# C4: SOS / EOS / LOS heatmap (Year × Zone)
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, (var, ylabel, title) in zip(axes, PHENO_VARS[:3]):
    sub   = metrics[metrics["zone"].isin(ZONES_MAIN)]
    pivot = sub.pivot_table(index="pheno_year", columns="zone", values=var)
    im    = ax.imshow(
        pivot.values, aspect="auto", cmap="YlOrRd",
        extent=[-0.5, len(pivot.columns) - 0.5,
                pivot.index.max() + 0.5, pivot.index.min() - 0.5]
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_ylabel("Phenological Year", fontsize=10)
    ax.set_title(f"{title}", fontsize=12)
    plt.colorbar(im, ax=ax, label=ylabel, shrink=0.85)
    # Annotate values
    for i, yr in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[yr, col]
            if not np.isnan(val):
                ax.text(j, yr, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="black")
plt.suptitle("Phenological Metrics Heatmap — Year × Zone — Similipal Core", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/C4_phenometric_heatmap.png", dpi=150)
plt.close()
print("  C4_phenometric_heatmap.png")

# C5: Rainfall anomaly heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, zone in zip(axes, ZONES_MAIN):
    sub   = ts[ts["zone"] == zone]
    pivot = sub.pivot_table(
        index="pheno_year", columns="pheno_doy",
        values="rainfall_mm_8day", aggfunc="mean"
    )
    doy_mean = pivot.mean(axis=0)
    anomaly  = pivot.subtract(doy_mean, axis=1)
    im = ax.imshow(
        anomaly.values, aspect="auto", cmap="BrBG",
        vmin=-20, vmax=20,
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.max() + 0.5, pivot.index.min() - 0.5]
    )
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8, rotation=30)
    ax.set_xlabel("Month (Mar → Feb)", fontsize=10)
    ax.set_ylabel("Phenological Year", fontsize=10)
    ax.set_title(f"{zone} Zone — Rainfall Anomaly (mm/8d)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Rainfall Anomaly (mm/8d)", shrink=0.85)

plt.suptitle("Rainfall Anomaly Heatmap — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/C5_rainfall_anomaly_heatmap.png", dpi=150)
plt.close()
print("  C5_rainfall_anomaly_heatmap.png")


# ════════════════════════════════════════════════════════════
# ██  VIZ BLOCK D — CLIMATE–PHENOLOGY RELATIONSHIPS  ██
# ════════════════════════════════════════════════════════════
print("\n[INFO] [D] Plotting climate–phenology relationships ...")

# D1: Correlation matrix
corr_cols = [
    "SOS","EOS","LOS","peak_doy","peak_ndvi","integrated_NDVI",
    "pre_sos_rain_mm","season_rain_mm","mean_LST_Day","mean_LST_Night",
    "mean_DTR","mean_ET","mean_ET_stress","annual_GPP","mean_NDWI","mean_NDMI",
]
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, zone in zip(axes, ZONES_MAIN):
    sub  = metrics[metrics["zone"] == zone][corr_cols].dropna()
    corr = sub.corr()
    im   = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols, fontsize=7)
    ax.set_title(f"{zone} Zone — Correlation Matrix", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.4:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(val) >= 0.7 else "black")
plt.suptitle("Phenology–Climate Correlation Matrix — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/D1_correlation_matrix.png", dpi=150)
plt.close()
print("  D1_correlation_matrix.png")

# D2: SOS vs climate drivers (scatter + regression)
climate_drivers = [
    ("pre_sos_rain_mm", "Pre-SOS Rainfall (mm)"),
    ("mean_LST_Day",    "Mean Seasonal LST Day (°C)"),
    ("mean_ET",         "Mean Seasonal ET (mm/8d)"),
]
fig, axes = plt.subplots(len(ZONES_MAIN), len(climate_drivers),
                          figsize=(16, 9), sharey="row")
for row, zone in enumerate(ZONES_MAIN):
    sub = metrics[metrics["zone"] == zone].dropna(subset=["SOS"])
    for col, (driver, xlabel) in enumerate(climate_drivers):
        ax = axes[row, col]
        sc = ax.scatter(sub[driver], sub["SOS"],
                        c=sub["pheno_year"], cmap="viridis",
                        s=55, zorder=3, edgecolors="white", lw=0.3)
        mask = sub[driver].notna() & sub["SOS"].notna()
        if mask.sum() >= 4:
            slope, intercept, r, p, _ = stats.linregress(
                sub[driver][mask], sub["SOS"][mask])
            xfit = np.linspace(sub[driver][mask].min(), sub[driver][mask].max(), 50)
            ax.plot(xfit, slope * xfit + intercept, "r--", lw=1.8,
                    label=f"r={r:.2f}, p={p:.3f} {sig_stars(p)}")
            ax.legend(fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("SOS (phenoDOY)" if col == 0 else "", fontsize=9)
        ax.set_title(f"{zone} — SOS vs {xlabel.split('(')[0].strip()}", fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle("SOS vs Climate Drivers — Similipal Core (2001–2024)\n"
             "Colour = phenological year (early=purple, recent=yellow)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/D2_sos_vs_climate.png", dpi=150)
plt.close()
print("  D2_sos_vs_climate.png")

# D3: EOS vs season rainfall
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = metrics[metrics["zone"] == zone].dropna(subset=["EOS", "season_rain_mm"])
    sc  = ax.scatter(sub["season_rain_mm"], sub["EOS"],
                     c=sub["pheno_year"], cmap="viridis", s=60,
                     edgecolors="white", lw=0.3)
    mask = sub["season_rain_mm"].notna()
    if mask.sum() >= 4:
        slope, intercept, r, p, _ = stats.linregress(
            sub["season_rain_mm"][mask], sub["EOS"][mask])
        xfit = np.linspace(sub["season_rain_mm"][mask].min(),
                           sub["season_rain_mm"][mask].max(), 50)
        ax.plot(xfit, slope * xfit + intercept, "r--", lw=2,
                label=f"r={r:.2f}, p={p:.3f} {sig_stars(p)}")
    ax.set_xlabel("Season Rainfall (mm)", fontsize=11)
    ax.set_ylabel("EOS (phenoDOY)", fontsize=11)
    ax.set_title(f"{zone} Zone — EOS vs Season Rainfall", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.85)
plt.suptitle("EOS vs Season Rainfall — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/D3_eos_vs_rainfall.png", dpi=150)
plt.close()
print("  D3_eos_vs_rainfall.png")

# D4: Peak NDVI vs season rainfall
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = metrics[metrics["zone"] == zone].dropna(subset=["peak_ndvi", "season_rain_mm"])
    sc  = ax.scatter(sub["season_rain_mm"], sub["peak_ndvi"],
                     c=sub["pheno_year"], cmap="plasma", s=60,
                     edgecolors="white", lw=0.3)
    mask = sub["season_rain_mm"].notna()
    if mask.sum() >= 4:
        slope, intercept, r, p, _ = stats.linregress(
            sub["season_rain_mm"][mask], sub["peak_ndvi"][mask])
        xfit = np.linspace(sub["season_rain_mm"][mask].min(),
                           sub["season_rain_mm"][mask].max(), 50)
        ax.plot(xfit, slope * xfit + intercept, "r--", lw=2,
                label=f"r={r:.2f}, p={p:.3f} {sig_stars(p)}")
    ax.set_xlabel("Season Rainfall (mm)", fontsize=11)
    ax.set_ylabel("Peak NDVI", fontsize=11)
    ax.set_title(f"{zone} Zone — Peak NDVI vs Rainfall", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.85)
plt.suptitle("Peak NDVI vs Season Rainfall — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/D4_peakndvi_vs_rainfall.png", dpi=150)
plt.close()
print("  D4_peakndvi_vs_rainfall.png")

# D5: LOS vs mean LST Day
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = metrics[metrics["zone"] == zone].dropna(subset=["LOS", "mean_LST_Day"])
    sc  = ax.scatter(sub["mean_LST_Day"], sub["LOS"],
                     c=sub["pheno_year"], cmap="inferno", s=60,
                     edgecolors="white", lw=0.3)
    mask = sub["mean_LST_Day"].notna()
    if mask.sum() >= 4:
        slope, intercept, r, p, _ = stats.linregress(
            sub["mean_LST_Day"][mask], sub["LOS"][mask])
        xfit = np.linspace(sub["mean_LST_Day"][mask].min(),
                           sub["mean_LST_Day"][mask].max(), 50)
        ax.plot(xfit, slope * xfit + intercept, "b--", lw=2,
                label=f"r={r:.2f}, p={p:.3f} {sig_stars(p)}")
    ax.set_xlabel("Mean Seasonal LST Day (°C)", fontsize=11)
    ax.set_ylabel("LOS (days)", fontsize=11)
    ax.set_title(f"{zone} Zone — LOS vs Mean LST", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.85)
plt.suptitle("Length of Season vs Mean LST — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/D5_los_vs_lst.png", dpi=150)
plt.close()
print("  D5_los_vs_lst.png")


# ════════════════════════════════════════════════════════════
# ██  VIZ BLOCK E — STATISTICAL SUMMARIES  ██
# ════════════════════════════════════════════════════════════
print("\n[INFO] [E] Computing Mann-Kendall and producing summaries ...")

mk_records = []
for zone in ["Wet", "Dry", "Others"]:
    sub_m = metrics[metrics["zone"] == zone].sort_values("pheno_year")
    for var, ylabel, title in PHENO_VARS:
        vals   = sub_m[var].dropna().values
        yrs_v  = sub_m.loc[sub_m[var].notna(), "pheno_year"].values
        if len(vals) < 4:
            continue
        trend, z, p, S = mann_kendall(vals)
        ss = sens_slope(yrs_v, vals)
        mk_records.append(dict(
            Zone=zone, Variable=title, var_key=var,
            MK_trend=trend, Z_score=z, p_value=p, S_stat=S,
            Sens_slope_per_yr=round(ss, 4) if not np.isnan(ss) else np.nan,
        ))

mk_df = pd.DataFrame(mk_records)
mk_df.to_csv(MK_CSV, index=False)
print(f"  MK results → {MK_CSV}")
print(mk_df.to_string(index=False))

# E1: MK Z-score summary heatmap
vars_order  = [t for _, _, t in PHENO_VARS]
zones_order = ["Wet", "Dry"]
Z_mat = np.full((len(zones_order), len(vars_order)), np.nan)
P_mat = np.ones((len(zones_order), len(vars_order)))
S_mat = np.full((len(zones_order), len(vars_order)), np.nan)

for i, zone in enumerate(zones_order):
    for j, (_, _, title) in enumerate(PHENO_VARS):
        row = mk_df[(mk_df["Zone"] == zone) & (mk_df["Variable"] == title)]
        if not row.empty:
            Z_mat[i, j] = row["Z_score"].values[0]
            P_mat[i, j] = row["p_value"].values[0]
            S_mat[i, j] = row["Sens_slope_per_yr"].values[0]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(Z_mat, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(len(vars_order)))
ax.set_xticklabels(vars_order, rotation=25, ha="right", fontsize=10)
ax.set_yticks(range(len(zones_order)))
ax.set_yticklabels(zones_order, fontsize=12)

for i in range(len(zones_order)):
    for j in range(len(vars_order)):
        p  = P_mat[i, j]
        ss = S_mat[i, j]
        stars = sig_stars(p)
        txt   = f"{ss:+.2f}/yr\n{stars}" if not np.isnan(ss) else ""
        fc    = "white" if abs(Z_mat[i, j]) > 1.5 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=fc)

plt.colorbar(im, ax=ax, label="MK Z-score  (−/+ = decreasing/increasing)")
ax.set_title(
    "Mann-Kendall Trend Summary — Similipal Core Area (2001–2024)\n"
    "Cell text: Sen's slope / significance stars (* p<0.05, ** p<0.01, *** p<0.001)",
    fontsize=12
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/E1_mannkendall_summary.png", dpi=150)
plt.close()
print("  E1_mannkendall_summary.png")

# E2: Violin plots — SOS / EOS / LOS
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, (var, ylabel, title) in zip(axes, PHENO_VARS[:3]):
    data   = [metrics[metrics["zone"] == z][var].dropna().values for z in ZONES_MAIN]
    parts  = ax.violinplot(data, positions=range(len(ZONES_MAIN)),
                           showmedians=True, showextrema=True)
    for pc, zone in zip(parts["bodies"], ZONES_MAIN):
        pc.set_facecolor(ZONE_COLORS[zone])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(ZONES_MAIN)))
    ax.set_xticklabels(ZONES_MAIN, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
plt.suptitle(
    "Distribution of Phenological Metrics by Zone (2001–2024)\n"
    "Similipal Core — Violin Plots",
    fontsize=13
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/E2_violin_plots.png", dpi=150)
plt.close()
print("  E2_violin_plots.png")

# E3: Coefficient of Variation of NDVI
cv_data = (
    ts.groupby(["zone", "pheno_doy"])["NDVI_smooth"]
    .agg(mean_ndvi="mean", std_ndvi="std")
    .reset_index()
)
cv_data["cv_pct"] = cv_data["std_ndvi"] / cv_data["mean_ndvi"] * 100

fig, ax = plt.subplots(figsize=(13, 5))
for zone in ["Wet", "Dry", "Others"]:
    sub = cv_data[cv_data["zone"] == zone].sort_values("pheno_doy")
    ax.plot(sub["pheno_doy"], sub["cv_pct"],
            color=ZONE_COLORS[zone], lw=2, label=zone)
set_month_xaxis(ax)
ax.set_ylabel("CV of NDVI (%)", fontsize=11)
ax.set_title(
    "Inter-annual Coefficient of Variation in NDVI\n"
    "Similipal Core (2001–2024) — higher CV = more inter-annual variability",
    fontsize=13
)
ax.legend(fontsize=10)
ax.set_xlim(1, 365)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/E3_ndvi_cv_seasonal.png", dpi=150)
plt.close()
print("  E3_ndvi_cv_seasonal.png")

# E4: Amplitude trend
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ZONES_MAIN:
    sub = metrics[metrics["zone"] == zone].sort_values("pheno_year")
    ax.plot(sub["pheno_year"], sub["amplitude"], "o-",
            color=ZONE_COLORS[zone], lw=1.5, ms=5, label=zone)
    mask = ~sub["amplitude"].isna()
    if mask.sum() >= 4:
        add_linreg_trend(ax, sub["pheno_year"][mask].values,
                         sub["amplitude"][mask].values,
                         color=ZONE_COLORS[zone], lw=1.8)
ax.set_xlabel("Phenological Year", fontsize=11)
ax.set_ylabel("Seasonal NDVI Amplitude", fontsize=11)
ax.set_title("Seasonal NDVI Amplitude Trend — Similipal Core (2001–2024)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/E4_amplitude_trend.png", dpi=150)
plt.close()
print("  E4_amplitude_trend.png")

# E5: Smoothing example — one zone, multiple years
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
example_years = [2005, 2010, 2015, 2020, 2023]
for ax, zone in zip(axes, ZONES_MAIN):
    sub_ts = ts[ts["zone"] == zone]
    for yr in example_years:
        yd = sub_ts[sub_ts["pheno_year"] == yr].sort_values("pheno_doy")
        ax.plot(yd["pheno_doy"], yd["NDVI"], "--",
                alpha=0.45, lw=1, color=ZONE_COLORS[zone])
        ax.plot(yd["pheno_doy"], yd["NDVI_smooth"],
                lw=2, label=str(yr))
    set_month_xaxis(ax)
    ax.set_ylabel("NDVI", fontsize=11)
    ax.set_title(f"{zone} Zone — Raw vs Smoothed NDVI (SG filter)", fontsize=12)
    ax.legend(fontsize=9, title="Year")
    ax.set_xlim(1, 365)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, "Dashed = raw, Solid = SG smoothed",
            transform=ax.transAxes, fontsize=8, color="grey")
plt.suptitle("Savitzky-Golay Smoothing Example — Similipal Core", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/E5_smoothing_example.png", dpi=150)
plt.close()
print("  E5_smoothing_example.png")


# ════════════════════════════════════════════════════════════
# ██  VIZ BLOCK F — ECOSYSTEM FLUXES  ██
# ════════════════════════════════════════════════════════════
print("\n[INFO] [F] Plotting ecosystem flux analyses ...")

# F1: Annual GPP trend
annual_gpp = (
    ts.groupby(["zone", "pheno_year"])["GPP_gCm2_8day"]
    .sum()
    .reset_index(name="annual_GPP")
)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = annual_gpp[annual_gpp["zone"] == zone].sort_values("pheno_year")
    ax.bar(sub["pheno_year"], sub["annual_GPP"],
           color=ZONE_COLORS[zone], alpha=0.75, edgecolor="black", lw=0.3)
    mask = sub["annual_GPP"].notna()
    if mask.sum() >= 4:
        slope, p = add_linreg_trend(
            ax, sub["pheno_year"][mask].values,
            sub["annual_GPP"][mask].values, color="black", lw=2.0)
        ax.text(0.03, 0.95,
                f"trend: {slope:+.1f} gC/m²/yr\np={p:.3f} {sig_stars(p)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_xlabel("Phenological Year", fontsize=10)
    ax.set_ylabel("Annual GPP (gC/m²)", fontsize=11)
    ax.set_title(f"{zone} Zone — Annual GPP Trend", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
plt.suptitle("Annual Gross Primary Productivity Trend — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F1_annual_gpp_trend.png", dpi=150)
plt.close()
print("  F1_annual_gpp_trend.png")

# F2: ET stress seasonality
et_clim = ts.groupby(["zone", "pheno_doy"])["ET_stress"].mean().reset_index()
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ZONES_MAIN:
    sub = et_clim[et_clim["zone"] == zone].sort_values("pheno_doy")
    ax.plot(sub["pheno_doy"], sub["ET_stress"],
            color=ZONE_COLORS[zone], lw=2.5, label=zone)
    ax.fill_between(sub["pheno_doy"], 0, sub["ET_stress"],
                    color=ZONE_COLORS[zone], alpha=0.12)
set_month_xaxis(ax)
ax.set_ylabel("ET Stress Index (ET/PET)", fontsize=11)
ax.set_title(
    "Evapotranspiration Stress Seasonal Pattern — Similipal Core\n"
    "(1 = no stress, 0 = maximum stress)",
    fontsize=13
)
ax.legend(fontsize=10)
ax.set_xlim(1, 365)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F2_et_stress_seasonality.png", dpi=150)
plt.close()
print("  F2_et_stress_seasonality.png")

# F3: PsnNet seasonality
psn_clim = ts.groupby(["zone","pheno_doy"])["PsnNet_gCm2_8day"].mean().reset_index()
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ZONES_MAIN:
    sub = psn_clim[psn_clim["zone"]==zone].sort_values("pheno_doy")
    ax.plot(sub["pheno_doy"], sub["PsnNet_gCm2_8day"],
            color=ZONE_COLORS[zone], lw=2.5, label=zone)
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
ax.fill_between(sub["pheno_doy"], 0, sub["PsnNet_gCm2_8day"],
                where=sub["PsnNet_gCm2_8day"] > 0, alpha=0.10, color="green",
                label="Carbon gain")
ax.fill_between(sub["pheno_doy"], 0, sub["PsnNet_gCm2_8day"],
                where=sub["PsnNet_gCm2_8day"] < 0, alpha=0.10, color="red",
                label="Carbon loss")
set_month_xaxis(ax)
ax.set_ylabel("PsnNet (gC/m²/8day)", fontsize=11)
ax.set_title("Net Photosynthesis (PsnNet) Seasonal Pattern — Similipal Core", fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(1, 365)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F3_psnnet_seasonality.png", dpi=150)
plt.close()
print("  F3_psnnet_seasonality.png")

# F4: NDWI & NDMI annual trends
annual_idx = (
    ts.groupby(["zone","pheno_year"])[["NDWI","NDMI"]]
    .mean()
    .reset_index()
)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, idx in zip(axes, ["NDWI", "NDMI"]):
    for zone in ZONES_MAIN:
        sub = annual_idx[annual_idx["zone"] == zone].sort_values("pheno_year")
        ax.plot(sub["pheno_year"], sub[idx], "o-",
                color=ZONE_COLORS[zone], lw=1.5, ms=4, label=zone)
        mask = sub[idx].notna()
        if mask.sum() >= 4:
            add_linreg_trend(ax, sub["pheno_year"][mask].values,
                             sub[idx][mask].values,
                             color=ZONE_COLORS[zone], lw=1.8)
    ax.set_xlabel("Phenological Year", fontsize=10)
    ax.set_ylabel(idx, fontsize=11)
    ax.set_title(f"Annual Mean {idx} Trend", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle("Vegetation Moisture Index Trends — Similipal Core (2001–2024)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F4_moisture_index_trends.png", dpi=150)
plt.close()
print("  F4_moisture_index_trends.png")

# F5: LST DTR trend
annual_dtr = (
    ts.groupby(["zone","pheno_year"])["LST_DTR"]
    .mean()
    .reset_index(name="mean_DTR")
)
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ZONES_MAIN:
    sub = annual_dtr[annual_dtr["zone"] == zone].sort_values("pheno_year")
    ax.plot(sub["pheno_year"], sub["mean_DTR"], "o-",
            color=ZONE_COLORS[zone], lw=1.5, ms=5, label=zone)
    mask = sub["mean_DTR"].notna()
    if mask.sum() >= 4:
        add_linreg_trend(ax, sub["pheno_year"][mask].values,
                         sub["mean_DTR"][mask].values,
                         color=ZONE_COLORS[zone], lw=1.8)
ax.set_xlabel("Phenological Year", fontsize=11)
ax.set_ylabel("Mean LST DTR (°C)", fontsize=11)
ax.set_title(
    "Annual Mean Diurnal Temperature Range (LST Day − Night) Trend\n"
    "Similipal Core (2001–2024) — DTR linked to canopy structure & moisture",
    fontsize=13
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F5_lst_dtr_trend.png", dpi=150)
plt.close()
print("  F5_lst_dtr_trend.png")

# F6: Integrated NDVI trend (season-long productivity)
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ZONES_MAIN:
    sub = metrics[metrics["zone"] == zone].sort_values("pheno_year")
    ax.plot(sub["pheno_year"], sub["integrated_NDVI"], "o-",
            color=ZONE_COLORS[zone], lw=1.5, ms=5, label=zone)
    mask = ~sub["integrated_NDVI"].isna()
    if mask.sum() >= 4:
        slope, p = add_linreg_trend(
            ax, sub["pheno_year"][mask].values,
            sub["integrated_NDVI"][mask].values,
            color=ZONE_COLORS[zone], lw=1.8)
        trend, z, pval, _ = mann_kendall(sub["integrated_NDVI"][mask].values)
        ax.text(
            0.03 if zone == "Wet" else 0.03,
            0.97 if zone == "Wet" else 0.87,
            f"{zone}: {slope:+.0f}/yr  MK p={pval:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            color=ZONE_COLORS[zone],
            bbox=dict(boxstyle="round", fc="white", alpha=0.6)
        )
ax.set_xlabel("Phenological Year", fontsize=11)
ax.set_ylabel("Integrated NDVI (NDVI × DOY units)", fontsize=11)
ax.set_title(
    "Integrated Seasonal NDVI Trend (∫NDVI dt) — Similipal Core (2001–2024)\n"
    "Proxy for season-long vegetation productivity",
    fontsize=13
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F6_integrated_ndvi_trend.png", dpi=150)
plt.close()
print("  F6_integrated_ndvi_trend.png")

# F7: ET + NDVI dual-axis seasonality
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
for ax, zone in zip(axes, ZONES_MAIN):
    sub = ts[ts["zone"] == zone].groupby("pheno_doy")[
        ["NDVI_smooth", "ET_mm8day", "PET_mm8day"]
    ].mean()
    ax2 = ax.twinx()
    ln1, = ax.plot(sub.index, sub["NDVI_smooth"],
                   color=ZONE_COLORS[zone], lw=2.5, label="NDVI")
    ln2, = ax2.plot(sub.index, sub["ET_mm8day"],
                    color="#2ca02c", lw=2, ls="--", label="ET")
    ln3, = ax2.plot(sub.index, sub["PET_mm8day"],
                    color="#98df8a", lw=1.5, ls=":", label="PET")
    ax.set_ylabel("NDVI", fontsize=11, color=ZONE_COLORS[zone])
    ax2.set_ylabel("ET / PET (mm/8day)", fontsize=11, color="#2ca02c")
    ax.set_title(f"{zone} Zone — NDVI, ET & PET Seasonality", fontsize=12)
    set_month_xaxis(ax)
    ax.set_xlim(1, 365)
    ax.legend(handles=[ln1, ln2, ln3], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
plt.suptitle("NDVI vs ET and PET Seasonal Co-variation — Similipal Core", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/F7_ndvi_et_dual.png", dpi=150)
plt.close()
print("  F7_ndvi_et_dual.png")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY PRINT
# ════════════════════════════════════════════════════════════
print(f"""
══════════════════════════════════════════════════════
 ANALYSIS COMPLETE — Similipal Core Phenology Study
══════════════════════════════════════════════════════
 Output plots    : {OUT_DIR}/
   Block A (6)  : Seasonal climatological profiles
   Block B (9)  : Inter-annual phenological trends
   Block C (5)  : Heatmaps & anomaly analyses
   Block D (5)  : Climate–phenology relationships
   Block E (5)  : Statistical summaries
   Block F (7)  : Ecosystem flux analyses

 CSV outputs:
   {RESULTS_CSV}
   {MK_CSV}

 Total plots: ~37 publication-quality figures
══════════════════════════════════════════════════════
""")
