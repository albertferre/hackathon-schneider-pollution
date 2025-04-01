import pandas as pd
import numpy as np


data_path = "data/raw/"
processed_path = "data/processed/"
predictions_path = "predictions/"

def read_files():
    measurement_data = pd.read_csv(f"{data_path}measurement_data.csv")
    instrument_data = pd.read_csv(f"{data_path}instrument_data.csv")
    pollutant_data = pd.read_csv(f"{data_path}pollutant_data.csv")

    return measurement_data, instrument_data, pollutant_data

def get_q1data():
    measurement_data, instrument_data, pollutant_data = read_files()
    # Merge datasets
    merged_data = instrument_data.merge(pollutant_data, on=["Item code"], how="left")

    # Filter only normal measurements
    merged_data = merged_data[merged_data["Instrument status"] == 0]

    conditions = [
        merged_data["Average value"] <= merged_data["Good"],
        (merged_data["Average value"] > merged_data["Good"]) & (merged_data["Average value"] <= merged_data["Normal"]),
        (merged_data["Average value"] > merged_data["Normal"]) & (merged_data["Average value"] <= merged_data["Bad"]),
        merged_data["Average value"] > merged_data["Bad"]
    ]

    categories = ["Good", "Normal", "Bad", "Very bad"]

    merged_data["Category"] = np.select(conditions, categories, default="Unknown")
    merged_data["Measurement date"] = pd.to_datetime(merged_data["Measurement date"])
    merged_data["season"] = merged_data["Measurement date"].dt.month.map({12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4})
    merged_data["hour"] = merged_data["Measurement date"].dt.hour

    return merged_data, instrument_data

def get_q2data():
    merged_data, instrument_data = get_q1data()

    df_pivot = merged_data[["Measurement date", "Station code", "Item name", "Average value"]].reset_index(drop=True)
    df_pivot["station_item"] = df_pivot["Item name"].astype(str) + "_" + df_pivot["Station code"].astype(str)
    df_pivot = df_pivot.pivot_table(index=["Measurement date"], columns=["station_item"], values="Average value").reset_index()

    df_pivot["season"] = df_pivot["Measurement date"].dt.month.map({12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4})
    df_pivot["hour"] = df_pivot["Measurement date"].dt.hour

    df_pivot["weekday"] = df_pivot["Measurement date"].dt.weekday

    # df_pivot["weekday"] = df_pivot["Measurement date"].dt.dayofweek

    df_pivot["sin_hour"] = np.sin(2 * np.pi * df_pivot["hour"] / 24)
    df_pivot["cos_hour"] = np.cos(2 * np.pi * df_pivot["hour"] / 24)
    df_pivot["sin_season"] = np.sin(2 * np.pi * df_pivot["season"] / 4)
    df_pivot["cos_season"] = np.cos(2 * np.pi * df_pivot["season"] / 4)
    df_pivot["sin_weekday"] = np.sin(2 * np.pi * df_pivot["weekday"] / 7)
    df_pivot["cos_weekday"] = np.cos(2 * np.pi * df_pivot["weekday"] / 7)

    df_pivot["weekday"].astype("category")
    df_pivot["season"].astype("category")

    df_pivot["SO2_mean"] = df_pivot.filter(like='SO2').mean(axis=1)
    df_pivot["NO2_mean"] = df_pivot.filter(like='NO2').mean(axis=1)
    df_pivot["O3_mean"] = df_pivot.filter(like='O3').mean(axis=1)
    df_pivot["CO_mean"] = df_pivot.filter(like='CO').mean(axis=1)
    df_pivot["PM10_mean"] = df_pivot.filter(like='PM10').mean(axis=1)
    df_pivot["PM2.5_mean"] = df_pivot.filter(like='PM2.5').mean(axis=1)

    df_pivot["SO2_std"] = df_pivot.filter(like='SO2').std(axis=1)
    df_pivot["NO2_std"] = df_pivot.filter(like='NO2').std(axis=1)
    df_pivot["O3_std"] = df_pivot.filter(like='O3').std(axis=1)
    df_pivot["CO_std"] = df_pivot.filter(like='CO').std(axis=1)
    df_pivot["PM10_std"] = df_pivot.filter(like='PM10').std(axis=1)
    df_pivot["PM2.5_std"] = df_pivot.filter(like='PM2.5').std(axis=1)

    df_pivot["SO2_median"] = df_pivot.filter(like='SO2').median(axis=1)
    df_pivot["NO2_median"] = df_pivot.filter(like='NO2').median(axis=1)
    df_pivot["O3_median"] = df_pivot.filter(like='O3').median(axis=1)
    df_pivot["CO_median"] = df_pivot.filter(like='CO').median(axis=1)
    df_pivot["PM10_median"] = df_pivot.filter(like='PM10').median(axis=1)
    df_pivot["PM2.5_median"] = df_pivot.filter(like='PM2.5').median(axis=1)

    df_pivot["SO2_90th"] = df_pivot.filter(like='SO2').quantile(0.9, axis=1)
    df_pivot["NO2_90th"] = df_pivot.filter(like='NO2').quantile(0.9, axis=1)
    df_pivot["O3_90th"] = df_pivot.filter(like='O3').quantile(0.9, axis=1)
    df_pivot["CO_90th"] = df_pivot.filter(like='CO').quantile(0.9, axis=1)
    df_pivot["PM10_90th"] = df_pivot.filter(like='PM10').quantile(0.9, axis=1)
    df_pivot["PM2.5_90th"] = df_pivot.filter(like='PM2.5').quantile(0.9, axis=1)

    df_pivot["SO2_10th"] = df_pivot.filter(like='SO2').quantile(0.1, axis=1)
    df_pivot["NO2_10th"] = df_pivot.filter(like='NO2').quantile(0.1, axis=1)
    df_pivot["O3_10th"] = df_pivot.filter(like='O3').quantile(0.1, axis=1)
    df_pivot["CO_10th"] = df_pivot.filter(like='CO').quantile(0.1, axis=1)
    df_pivot["PM10_10th"] = df_pivot.filter(like='PM10').quantile(0.1, axis=1)
    df_pivot["PM2.5_10th"] = df_pivot.filter(like='PM2.5').quantile(0.1, axis=1)


    df_pivot["SO2_hourly_mean"] = df_pivot.groupby("hour")["SO2_mean"].transform("mean")
    df_pivot["NO2_hourly_mean"] = df_pivot.groupby("hour")["NO2_mean"].transform("mean")
    df_pivot["O3_hourly_mean"] = df_pivot.groupby("hour")["O3_mean"].transform("mean")
    df_pivot["CO_hourly_mean"] = df_pivot.groupby("hour")["CO_mean"].transform("mean")
    df_pivot["PM10_hourly_mean"] = df_pivot.groupby("hour")["PM10_mean"].transform("mean")
    df_pivot["PM2.5_hourly_mean"] = df_pivot.groupby("hour")["PM2.5_mean"].transform("mean")


    df_pivot.drop(columns=["hour"], inplace=True)
    return df_pivot

def get_q3data():
    measurement_data, instrument_data, pollutant_data = read_files()

    task3_data = instrument_data.merge(pollutant_data, on=["Item code"], how="left")

    task3_data = task3_data.pivot_table(index=["Measurement date", "Station code"], columns=["Item name"], values=["Instrument status"]).reset_index()

    # Aplanar el MultiIndex de columnas
    task3_data.columns = [col[1] if col[1] else col[0] for col in task3_data.columns]

    # Mostrar el resultado
    task3_data["Measurement date"] = pd.to_datetime(task3_data["Measurement date"])
    measurement_data["Measurement date"] = pd.to_datetime(measurement_data["Measurement date"])
    task3_data = measurement_data.merge(task3_data, on=["Measurement date", "Station code"], how="left", suffixes=("", "anomalies"))
    task3_data.drop(columns=["Latitude", "Longitude"], inplace=True)
    task3_data["hour"] = task3_data["Measurement date"].dt.hour
    task3_data["season"] = task3_data["Measurement date"].dt.month.map({12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4})
    task3_data["weekday"] = task3_data["Measurement date"].dt.weekday

    # time variables as sin and cos
    task3_data["sin_hour"] = np.sin(2 * np.pi * task3_data["hour"] / 24)
    task3_data["cos_hour"] = np.cos(2 * np.pi * task3_data["hour"] / 24)
    task3_data["sin_season"] = np.sin(2 * np.pi * task3_data["season"] / 4)
    task3_data["cos_season"] = np.cos(2 * np.pi * task3_data["season"] / 4)
    task3_data["sin_weekday"] = np.sin(2 * np.pi * task3_data["weekday"] / 7)
    task3_data["cos_weekday"] = np.cos(2 * np.pi * task3_data["weekday"] / 7)

    task3_data.drop(columns=["hour", "season", "weekday"], inplace=True)

    # diff columns SO2	NO2	O3	CO	PM10	PM2.5
    task3_data["SO2_n1"] = task3_data["SO2"].diff(1)
    task3_data["SO2_n2"] = task3_data["SO2"].diff(2)
    task3_data["SO2_n3"] = task3_data["SO2"].diff(3)

    task3_data["NO2_n1"] = task3_data["SO2"].diff(1)
    task3_data["NO2_n2"] = task3_data["SO2"].diff(2)
    task3_data["NO2_n3"] = task3_data["SO2"].diff(3)

    task3_data["O3_n1"] = task3_data["SO2"].diff(1)
    task3_data["O3_n2"] = task3_data["SO2"].diff(2)
    task3_data["O3_n3"] = task3_data["SO2"].diff(3)

    task3_data["CO_n1"] = task3_data["SO2"].diff(1)
    task3_data["CO_n2"] = task3_data["SO2"].diff(2)
    task3_data["CO_n3"] = task3_data["SO2"].diff(3)

    task3_data["PM10_n1"] = task3_data["SO2"].diff(1)
    task3_data["PM10_n2"] = task3_data["SO2"].diff(2)
    task3_data["PM10_n3"] = task3_data["SO2"].diff(3)

    task3_data["PM2.5_n1"] = task3_data["SO2"].diff(1)
    task3_data["PM2.5_n2"] = task3_data["SO2"].diff(2)
    task3_data["PM2.5_n3"] = task3_data["SO2"].diff(3)

    # moving average
    task3_data["SO2_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["SO2_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["SO2_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    task3_data["NO2_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["NO2_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["NO2_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    task3_data["O3_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["O3_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["O3_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    task3_data["CO_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["CO_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["CO_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    task3_data["PM10_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["PM10_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["PM10_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    task3_data["PM2.5_moving_avg_1"] = task3_data["SO2"].rolling(window=1).mean()
    task3_data["PM2.5_moving_avg_2"] = task3_data["SO2"].rolling(window=2).mean()
    task3_data["PM2.5_moving_avg_3"] = task3_data["SO2"].rolling(window=3).mean()

    # z-score
    task3_data["SO2_z"] = (task3_data["SO2"] - task3_data["SO2"].mean()) / task3_data["SO2"].std()
    task3_data["NO2_z"] = (task3_data["NO2"] - task3_data["NO2"].mean()) / task3_data["NO2"].std()
    task3_data["O3_z"] = (task3_data["O3"] - task3_data["O3"].mean()) / task3_data["O3"].std()
    task3_data["CO_z"] = (task3_data["CO"] - task3_data["CO"].mean()) / task3_data["CO"].std()
    task3_data["PM10_z"] = (task3_data["PM10"] - task3_data["PM10"].mean()) / task3_data["PM10"].std()
    task3_data["PM2.5_z"] = (task3_data["PM2.5"] - task3_data["PM2.5"].mean()) / task3_data["PM2.5"].std()



    all_stations_avg = task3_data.reset_index().groupby(["Measurement date"])[["SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]].mean().reset_index()

    # add suffif _mean to columns
    task3_data = task3_data.merge(all_stations_avg, on=["Measurement date"], how="left", suffixes=("", "_mean"))

    # mean_diff
    task3_data["SO2_mean_diff"] = task3_data["SO2"] - task3_data["SO2_mean"]
    task3_data["NO2_mean_diff"] = task3_data["NO2"] - task3_data["NO2_mean"]
    task3_data["O3_mean_diff"] = task3_data["O3"] - task3_data["O3_mean"]
    task3_data["CO_mean_diff"] = task3_data["CO"] - task3_data["CO_mean"]
    task3_data["PM10_mean_diff"] = task3_data["PM10"] - task3_data["PM10_mean"]
    task3_data["PM2.5_mean_diff"] = task3_data["PM2.5"] - task3_data["PM2.5_mean"]

    return task3_data
