import pandas as pd
import numpy as np

def run_ml_pipeline(cleaned_data_path):
    df_clean = pd.read_csv(cleaned_data_path)
    print(f"Loaded {len(df_clean):,} clean records successfully!")

    duration_stats = df_clean["trip_duration_min"].describe()
    print("Trip Duration Summary (in Minutes):")
    print(duration_stats.apply(lambda x: f"{x:.2f}"))

    df_final = df_clean[
        (df_clean["trip_duration_min"].between(1, 120))
        & (df_clean["distance_km"] > 0.1)
    ].copy()

    print(f"Original Records: {len(df_clean):,}")
    print(f"Records After Outlier Removal: {len(df_final):,}")
    print(f"Max Duration Remaining: {df_final['trip_duration_min'].max():.2f} minutes")

    print(f"Unique Origin Zones: {df_final['origin_zone'].nunique():,}")
    print(f"Unique Destination Zones: {df_final['dest_zone'].nunique():,}")
    print(f"Unique Route Corridors: {df_final['route_id'].nunique():,}\n")
    print("Top 5 Most Frequent Route Corridors:")
    print(df_final["route_id"].value_counts().head(5))

    hourly_city_speed = df_final.groupby("hour")["speed_kmh"].mean()
    print("--- Average City Speed by Hour (km/h) ---")
    print(hourly_city_speed.round(2))
    
    route_counts = df_final["route_id"].value_counts()
    popular_routes = route_counts[route_counts >= 50].index
    congested_corridors = (
        df_final[df_final["route_id"].isin(popular_routes)]
        .groupby("route_id")["speed_kmh"]
        .mean()
        .sort_values()
        .head(5)
    )
    print("\n--- Top 5 Most Congested Route Corridors (Slowest Avg Speed in km/h) ---")
    print(congested_corridors.round(2))

    ml_feature_cols = [
        "TRIP_ID",
        "datetime",
        "hour",
        "day_of_week",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "origin_zone",
        "dest_zone",
        "route_id",
        "distance_km",
        "trip_duration_min",
        "speed_kmh",
        "avg_corridor_speed_kmh",
    ]
    df_ml_ready = df_final[ml_feature_cols].copy()
    output_handoff_path = "data/porto_taxi_ml_features.csv"
    df_ml_ready.to_csv(output_handoff_path, index=False)
    print(f"Success! Handoff file '{output_handoff_path}' exported.")
    print(f"Total Rows: {len(df_ml_ready):,}")
    print(f"Total Columns: {len(ml_feature_cols)}")
