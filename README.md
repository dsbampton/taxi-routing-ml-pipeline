# taxi-routing-ml-pipeline
A clean, automated 2-step Python pipeline that handles massive taxi data, removes bad logs, and builds polished feature tables for Machine Learning.
# Taxi Routing & Machine Learning Project

This repository contains a clean, automated 2-step pipeline designed to process massive taxi trip datasets. It automatically opens heavy raw files, filters out bad data entries, and calculates polished feature tables ready for Machine Learning models.

##  Key Features
* **Memory-Safe Loading:** Processes massive raw datasets in small chunks so your computer's RAM doesn't crash.
* **Bad Data Filtering:** Automatically removes missing paths and strips out unrealistic trip speeds or extreme outliers.
* **Smart Travel Math:** Calculates exact trip distances in kilometers using the geographic Haversine formula.
* **Route Corridor Tracking:** Automatically groups coordinates into rounded city grids to create unique origin-to-destination routes.

##  Project Structure
The code is split into modular scripts to keep it clean and professional:
```text
ROUTING PTOJECT/
├── data/                  # Folder where you store raw and generated datasets
├── src/                   # Source scripts holding the math and logic
│   ├── data_cleaning.py   # Step 1: Loops through raw data chunks and fixes formatting
│   └── ml_handover.py     # Step 2: Filters outliers and extracts final ML tables
└── main.py                # The Master Switch (Runs the whole system)
```

##  Requirements & Setup
1. Open your terminal or command prompt inside this folder.
2. Install the standard data libraries by running:
   ```bash
   pip install pandas numpy
   ```
3. Make sure your raw data is placed in this path: `data/train.csv/train.csv`.

##  How to Run the System
You don't need to manually click through different scripts. Just run the master controller file from your terminal, and it will execute the whole pipeline sequentially:

```bash
python main.py
```

### What You Will See in the Console:
```text
Starting Routing Project Pipeline...
Step 1: Cleaning Data
Checking raw data structure layout...
[Data columns list will display here]
Starting full dataset cleaning pipeline chunks...
Batch 1 processed and saved...
Success! Handoff file exported to: data/cleaned_taxi_data.csv

Step 2: Running ML Handover
Loaded clean records successfully!
Trip Duration Summary (in Minutes)
Records After Outlier Removal...
Unique Route Corridors found...
Success! Handoff file 'data/porto_taxi_ml_features.csv' exported.
Pipeline complete! All tables and features successfully generated.
```
