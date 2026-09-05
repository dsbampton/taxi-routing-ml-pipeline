import json
import pandas as pd
import numpy as np

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def clean_data(input_path, output_path):
    print("Checking raw data structure layout...")
    df_preview = pd.read_csv(input_path, nrows=5)
    print(df_preview.dtypes)

    chunk_size = 100000
    first_chunk = True 
    lat_min, lat_max = 41.0, 41.3
    lon_min, lon_max = -8.7, -8.4
    
    temp_cleaned_path = "data/temp_cleaned_taxi_data.csv"
    
    print("Starting full dataset cleaning pipeline chunks...")
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        chunk["parsed_polyline"] = chunk["POLYLINE"].apply(json.loads)
        chunk["point_count"] = chunk["parsed_polyline"].apply(len)
        chunk = chunk[chunk["point_count"] >= 2].copy()
        chunk["trip_duration_sec"] = (chunk["point_count"] - 1) * 15
        chunk["datetime"] = pd.to_datetime(chunk["TIMESTAMP"], unit="s")
        chunk["hour"] = chunk["datetime"].dt.hour
        chunk["day_of_week"] = chunk["datetime"].dt.dayofweek
        chunk["origin_lon"] = chunk["parsed_polyline"].apply(lambda p: p[0][0])
        chunk["origin_lat"] = chunk["parsed_polyline"].apply(lambda p: p[0][1])
        chunk["dest_lon"] = chunk["parsed_polyline"].apply(lambda p: p[-1][0])
        chunk["dest_lat"] = chunk["parsed_polyline"].apply(lambda p: p[-1][1])
        
        chunk_clean = chunk[
            (chunk["origin_lat"].between(lat_min, lat_max))
            & (chunk["origin_lon"].between(lon_min, lon_max))
            & (chunk["dest_lat"].between(lat_min, lat_max))
            & (chunk["dest_lon"].between(lon_min, lon_max))
        ].copy()
        
        final_cols = [
            "TRIP_ID", "CALL_TYPE", "TAXI_ID", "datetime", "hour", 
            "day_of_week", "trip_duration_sec", "origin_lon", "origin_lat", 
            "dest_lon", "dest_lat"
        ]
        cleaned_batch = chunk_clean[final_cols]
        
        if first_chunk:
            cleaned_batch.to_csv(temp_cleaned_path, mode="w", index=False)
            first_chunk = False
        else:
            cleaned_batch.to_csv(temp_cleaned_path, mode="a", header=False, index=False)
        print(f"Batch {chunk_idx + 1} processed and saved.")

    print("Reloading batched data for outlier filters and grid grouping...")
    df_clean = pd.read_csv(temp_cleaned_path)
    
    df_clean["trip_duration_min"] = df_clean["trip_duration_sec"] / 60
    
    df_clean["distance_km"] = haversine_np(
        df_clean["origin_lon"], df_clean["origin_lat"],
        df_clean["dest_lon"], df_clean["dest_lat"]
    )
    
    df_final = df_clean[
        (df_clean["trip_duration_min"].between(1, 120))
        & (df_clean["distance_km"] > 0.1)
    ].copy()

    df_final["origin_grid_lat"] = df_final["origin_lat"].round(2)
    df_final["origin_grid_lon"] = df_final["origin_lon"].round(2)
    df_final["dest_grid_lat"] = df_final["dest_lat"].round(2)
    df_final["dest_grid_lon"] = df_final["dest_lon"].round(2)
    
    df_final["origin_zone"] = df_final["origin_grid_lat"].astype(str) + "_" + df_final["origin_grid_lon"].astype(str)
    df_final["dest_zone"] = df_final["dest_grid_lat"].astype(str) + "_" + df_final["dest_grid_lon"].astype(str)
    df_final["route_id"] = df_final["origin_zone"] + " -> " + df_final["dest_zone"]

    df_final["speed_kmh"] = df_final["distance_km"] / (df_final["trip_duration_min"] / 60)
    df_final = df_final[df_final["speed_kmh"].between(1, 140)].copy()
    
    corridor_hourly_stats = (
        df_final.groupby(["route_id", "hour"])["speed_kmh"]
        .mean()
        .reset_index()
        .rename(columns={"speed_kmh": "avg_corridor_speed_kmh"})
    )
    df_final = df_final.merge(corridor_hourly_stats, on=["route_id", "hour"], how="left")

    ml_feature_cols = [
        "TRIP_ID", "datetime", "hour", "day_of_week", "origin_lat", 
        "origin_lon", "dest_lat", "dest_lon", "origin_zone", "dest_zone", 
        "route_id", "distance_km", "trip_duration_min", "speed_kmh", "avg_corridor_speed_kmh"
    ]
    df_ml_ready = df_final[ml_feature_cols].copy()
    df_ml_ready.to_csv(output_path, index=False)
    
    print(f"Success! Handoff file exported to: {output_path}")
    print(f"Total Rows: {len(df_ml_ready):,}")
    print(f"Total Columns: {len(ml_feature_cols)}")
