# 🌿 Modelling Phenological Shifts in Similipal National Park Using MODIS Satellite Data (2001–2024)

## 📌 Project Overview
This project comprehensively analyzes the seasonal vegetation dynamics and long-term phenological shifts in the core area of Similipal National Park over a 24-year period (2001–2024). Using MODIS satellite data, the study investigates how vegetation phenology (Start of Season, End of Season, Length of Season) responds to climate variables like rainfall, temperature (LST), evapotranspiration (ET), and gross primary productivity (GPP) across different ecological zones (Wet and Dry).

## 🌍 Google Earth Engine (GEE) Visualizations
The raw satellite imagery, multi-band indices, and exported visualizations (`.tif` format) that form the baseline of this analysis can be accessed via the following Google Drive repositories:

* 🔗 **[GEE Visualizations - Part 1](https://drive.google.com/drive/folders/1lCIWgGxC5x4JEQtRb2MO4RILhuAqzUxn)**
* 🔗 **[GEE Visualizations - Part 2](https://drive.google.com/drive/folders/1MG3pUYA5VxOcHwuG2aCWQRYo4Q19yQaU)**

## 📂 Project Structure

The data processing, phenology extraction, and machine learning prediction pipelines are split into three primary Python scripts:

### 1. `01_zone_classification.py`
**Purpose:** Spatially classifies the park into distinct moisture zones ("Wet", "Dry", and "Others").
* Loads multi-year MODIS pixel data.
* Calculates per-pixel statistics and creates a **moisture composite index** (derived from Normalized Difference Water Index [NDWI] and Normalized Difference Moisture Index [NDMI]).
* Assigns thresholds to classify each pixel into its respective zone.
* **Output:** `zone_labels.csv` and zone mapping visualizations.

### 2. `02_phenology_analysis.py`
**Purpose:** Core phenological extraction, trend analysis, and comprehensive visualization generation.
* Re-maps calendar dates to a March-anchored phenological year.
* Smooths noisy NDVI time-series data using a **Savitzky-Golay filter**.
* Detects phenological transition dates: Start of Season (SOS), End of Season (EOS), and Length of Season (LOS) using zone-specific amplitude thresholds.
* Runs **Mann-Kendall trend tests** and **Sen's slope estimation** to calculate decadal shifts.
* **Output:** Over 35 distinct visualizations (climatological profiles, inter-annual heatmaps, correlation matrices) stored in `plots/phenology/` and structured datasets (`phenology_metrics.csv`, `mannkendall_results.csv`).

### 3. `03_ml_phenology_prediction.py`
**Purpose:** Predictive modeling of phenological shifts using Machine Learning.
* Uses historical climate data (Rainfall, LST, ET) and past phenology states to predict future SOS and EOS dates.
* Implements robust tree-based regression models, primarily **Random Forest** and **XGBoost**.
* Evaluates model performance and feature importance to understand which climate drivers most heavily influence the changing seasons.

## 📊 Key Findings

Detailed interpretations of all generated plots can be found in `analysis.md`. A brief summary of findings:
* **Productivity is Increasing:** Both Wet and Dry zones show a statistically significant long-term greening trend (integrated NDVI).
* **Phenological Shifts:** The Start of Season (SOS) in the Dry zone is delaying by approximately +1 day/year, likely due to warming pre-monsoon temperatures. Overall, the End of Season (EOS) is also delaying, keeping vegetation greener for longer.
* **Climate Drivers:** Vegetation growth is heavily monsoon-driven, responding to rainfall with a ~1-month lag. While water availability drives growth, high pre-monsoon temperatures act as a limiting stress factor.

## 🚀 How to Run

1. **Install Dependencies:**
   Ensure you have all required Python libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the Pipeline:**
   Run the scripts sequentially:
   ```bash
   python 01_zone_classification.py
   python 02_phenology_analysis.py
   python 03_ml_phenology_prediction.py
   ```

*Note: Ensure the source MODIS `.csv` data files are placed in the `similipal_pixelwise/` directory before running.*
