import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler # Scaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pandas.tseries.holiday import USFederalHolidayCalendar
import pytz
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fix_double_tracking(df, col, date_col="date"):
    df.loc[(df[date_col] <= pd.to_datetime("2018-06-12").date()) & (df[date_col] >= pd.to_datetime("2017-09-27").date()), col] = \
    df.loc[(df[date_col] <= pd.to_datetime("2018-06-12").date()) & (df[date_col] >= pd.to_datetime("2017-09-27").date()), col] / 2
    return df


def parse_xml_output(path):
    # Import the csv
    csv_df = pd.read_csv(path, low_memory=False)
    base_name = os.path.basename(path).split(".")[0]
    
    # BodyMassIndex we want to avg not sum
    if base_name == "BodyMassIndex":
        agg_func = "mean"
    else:
        agg_func = "sum"
    
    # Convert startDate and endDate columns to datetime objects
    csv_df["startDate"] = pd.to_datetime(csv_df["startDate"]).dt.tz_convert ("US/Eastern")
    csv_df["endDate"] = pd.to_datetime(csv_df["endDate"]).dt.tz_convert ("US/Eastern")
    # Create date & time column
    csv_df["date"] = pd.to_datetime(csv_df["startDate"]).dt.date
    csv_df["time"] = pd.to_datetime(csv_df["startDate"]).dt.time
    
    # Group by date and perform aggregations
    csv_df = csv_df.groupby("date").agg(
        max_start_time=pd.NamedAgg(column="startDate", aggfunc="max"),
        min_start_time=pd.NamedAgg(column="startDate", aggfunc="min"),
        max_end_time=pd.NamedAgg(column="endDate", aggfunc="max"),
        min_end_time=pd.NamedAgg(column="endDate", aggfunc="min"),
        value_sum=pd.NamedAgg(column="value", aggfunc=agg_func)
    ).reset_index()
    
    # Check if the dates match between min_start_time and max_end_time
    csv_df["dates_match"] = csv_df.apply(
        lambda row: row["min_start_time"].date() == row["max_end_time"].date(), axis=1)
    csv_df = csv_df.rename(columns={"value_sum": base_name})
    
    # Assert if dates match in all rows
    # assert csv_df["dates_match"].all(), "Dates do not match in some rows"
    
    # Trigonomic Hours
    csv_df[base_name+"_mx_st_hr_sin"] = np.sin(2 * np.pi * csv_df["max_start_time"].dt.hour / 24)
    csv_df[base_name+"_mx_st_hr_cos"] = np.cos(2 * np.pi * csv_df["max_start_time"].dt.hour / 24)
    csv_df[base_name+"_mn_st_hr_sin"] = np.sin(2 * np.pi * csv_df["min_start_time"].dt.hour / 24)
    csv_df[base_name+"_mn_st_hr_cos"] = np.cos(2 * np.pi * csv_df["min_start_time"].dt.hour / 24)
    csv_df[base_name+"_mx_et_hr_sin"] = np.sin(2 * np.pi * csv_df["max_end_time"].dt.hour / 24)
    csv_df[base_name+"_mx_et_hr_cos"] = np.cos(2 * np.pi * csv_df["max_end_time"].dt.hour / 24)
    csv_df[base_name+"_mn_et_hr_sin"] = np.sin(2 * np.pi * csv_df["min_end_time"].dt.hour / 24)
    csv_df[base_name+"_mn_et_hr_cos"] = np.cos(2 * np.pi * csv_df["min_end_time"].dt.hour / 24) 
    
    #! HIGHLY ILLEGAL FEATURE??? (Step Count basically calculates sleep time)
    # Hours between startDate and next startDate (lag = -1)
    csv_df[base_name+"_hours_between"] = (csv_df["min_start_time"].shift(-1) - csv_df["max_start_time"]).dt.total_seconds() / 3600
    
    #Fix double tracking
    csv_df = fix_double_tracking(csv_df, base_name)
    
    # Drop unnecessary columns
    csv_df = csv_df.drop(columns=["max_start_time", "min_start_time", "max_end_time", "min_end_time", "dates_match"])

    return csv_df
    

def create_data(path, type="train", preprocessor=None, xml_files_names = ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"], path_to_xmls = "./data/xml_export/", lookback = None):
    logging.info(f"Reading {type} data from {path}")
    df = pd.read_csv(path)
    
    # Format as date and sort
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date")
    
    # sleep_hours between 9/27/2017 and 6/12/2018 is doubled and needs to be divided by 2
    if type == "train":
        df = fix_double_tracking(df, "sleep_hours")

        ### Expand the date range to include all dates ###
        start_date = df["date"].min()
        logging.info(f"Start date: {start_date}")

        end_date = df["date"].max()
        logging.info(f"End date: {end_date}")

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        date_range = pd.DataFrame({"date": date_range})
        date_range["date"] = date_range["date"].dt.date

        df = date_range.merge(df, on="date", how="left")
        logging.info(f"missing days: {df.sleep_hours.isna().sum()}")
    
    ### Time series Data ###
    logging.info("Featurizing time series data")
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek # Day of the week
    df["dow_sin"] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df["dow_cos"] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
    df["doy_sin"] = np.sin(df['day_of_year'] * (2 * np.pi / 365))
    df["doy_cos"] = np.cos(df['day_of_year'] * (2 * np.pi / 365))
    
    df["month"] = pd.to_datetime(df["date"]).dt.month # Month
    df["month_sin"] = np.sin(df['month'] * (2 * np.pi / 12))
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0) # is_weekend

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start="2014-01-01", end="2023-12-31") 
    df["is_holiday"] = df["date"].apply(lambda x: 1 if x in holidays else 0) # is_holiday
    df["is_workday"] = 1-(df["is_weekend"] + df["is_holiday"]) # is_workday

    # is_daylight_savings
    def is_daylight_savings(date, timezone="US/Eastern"):
        tz = pytz.timezone(timezone)
        dt = tz.localize(datetime.combine(date, datetime.min.time()), is_dst=None)
        return int(dt.dst() != timedelta(0))

    df["is_daylight_savings"] = df["date"].apply(is_daylight_savings)
    
    ### XML DATA #
    logging.info("Featurizing XML data")
    # Append .csv to the end of each file name
    xml_files_names = [path_to_xmls + xml_file + ".csv" for xml_file in xml_files_names] 
    
    # Parse each xml file output and merge with the train data
    for xml_file in xml_files_names:
        xml = parse_xml_output(xml_file)
        
        # Merge with training data
        df = df.merge(xml, on="date", how="left")
    
    # Time series imputation
    logging.info("Imputing missing values...")
    df = df.fillna(method="ffill")
    
    # Create dow_median column
    median = df.groupby("day_of_week")["sleep_hours"].median()
    df["dow_median"] = df["day_of_week"].map(median)
    
    # Interactions
    logging.info("Creating interactions...")
    df["distance_per_step"] = df["DistanceWalkingRunning"] / df["StepCount"] # To account for jumping, hiking, etc.
    df["calorie_per_step"] = df["BasalEnergyBurned"] / df["StepCount"] # To account for intensity of exercise
    df["calorie_per_distance"] = df["BasalEnergyBurned"] / df["DistanceWalkingRunning"] # Gym days vs. Outdoor days
    


    logging.info("Saving to Parquet file...")
    df.to_parquet(f"./{type}_data.parquet")
    
    if type == "train":
        
        #! Drop not unique columns
        threshold = 0.05
        unique_pct = df.nunique() / len(df)

        df = df.drop(unique_pct[unique_pct <= threshold].index, axis=1)
        
        """ If training data, fit a preprocessor"""
        y = df["sleep_hours"]
        X = df.drop(columns=["date","sleep_hours"], axis =1 ) # Drop date column
        cols = X.columns
        
        imputer = SimpleImputer(strategy="median") 
        scaler = StandardScaler()

        f_preprocessor = Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", scaler)
        ])
        
        t_preprocessor = Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", scaler)
        ])
    else:
        f_preprocessor = None
        t_preprocessor = None
        cols=None
        
    return df, f_preprocessor, t_preprocessor, cols