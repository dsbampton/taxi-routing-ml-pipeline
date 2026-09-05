import os
from src.data_cleaning import clean_data
from src.ml_handover import run_ml_pipeline

def main():
    print("Starting Routing Project Pipeline...")
    raw_data_path = "data/train.csv/train.csv"
    cleaned_data_path = "data/cleaned_taxi_data.csv"

    
    print("Step 1: Cleaning Data")
    clean_data(raw_data_path, cleaned_data_path)
    
    print("Step 2: Running ML Handover")
    run_ml_pipeline(cleaned_data_path)
    
    print("Pipeline complete! All tables and features successfully generated.")

if __name__ == "__main__":
    main()
