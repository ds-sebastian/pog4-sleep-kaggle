import logging
import os
from datetime import datetime, timedelta
from typing import List
import warnings


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
import numpy as np
import datetime as dt


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s") # Configure logging

# Original columns = ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"]
# Activity Summary is by date!


class POG4_Dataset():
    """Initialize Dataset class."""
    def __init__(self, train_path: str = "./data/train.csv",  xml_export_path: str = "./data/xml_export", start_date = "2015-06-01") -> None:
        self.start_date = start_date
        self.xml_data = self.create_xml_data(xml_export_path, ['Workout', 'WalkingSpeed', 'ActiveEnergyBurned', 'RunningSpeed', 'AppleWalkingSteadiness', 'EnvironmentalAudioExposure', 'StairDescentSpeed', 'LowHeartRateEvent', 'RunningGroundContactTime', 'DistanceCycling', 'HandwashingEvent', 'NumberOfTimesFallen', 'BasalEnergyBurned', 'MindfulSession', 'SixMinuteWalkTestDistance', 'StairAscentSpeed', 'HKDataTypeSleepDurationGoal', 'HeartRateVariabilitySDNN', 'Height', 'OxygenSaturation', 'RunningStrideLength', 'HeartRateRecoveryOneMinute', 'WalkingStepLength', 'SwimmingStrokeCount', 'BodyMass', 'FlightsClimbed', 'DietaryEnergyConsumed', 'AudioExposureEvent', 'HeadphoneAudioExposure', 'StepCount', 'WalkingAsymmetryPercentage', 'RespiratoryRate', 'HeartRate', 'DietaryWater', 'BodyMassIndex', 'RunningPower', 'VO2Max', 'DistanceWalkingRunning', 'HeadphoneAudioExposureEvent', 'HighHeartRateEvent', 'WalkingDoubleSupportPercentage', 'AppleExerciseTime', 'RestingHeartRate', 'AppleStandTime', 'WalkingHeartRateAverage', 'DistanceSwimming', 'EnvironmentalSoundReduction', 'AppleStandHour', 'RunningVerticalOscillation'])
        self.activity_data = self.create_activity_data(os.path.join(xml_export_path, "ActivitySummary.csv"))
        self.train = self.create_train(train_path)

        self.preprocessor = None 
        
        self.sleep_times, self.awake_times = self.get_sleep_times() # Alternative targets
        
        # self.hr_intervals = self.process_interval_data(os.path.join(xml_export_path, "HeartRate.csv"))
        # self.step_intervals = self.process_interval_data(os.path.join(xml_export_path, "StepCount.csv"))
        # self.distance_intervals = self.process_interval_data(os.path.join(xml_export_path, "DistanceWalkingRunning.csv"))
        
        # self.train = self.train.merge(self.hr_intervals.add_prefix(f"hr_"), left_on='date', right_on = 'hr_date', how="left")
        # self.train = self.train.merge(self.step_intervals.add_prefix(f"steps_"), left_on='date', right_on = 'steps_date', how="left")
        # self.train = self.train.merge(self.distance_intervals.add_prefix(f"cal_"), left_on='date', right_on = 'cal_date', how="left")

        # self.train = self.train.drop(['hr_date', 'steps_date', 'cal_date'], axis=1)
        
        
        #         #Keep important features:
        # to_keep = ["date", "sleep_hours",
        #         "slp_DistanceWalkingRunning_hrs_max_max",
        #         "slp_FlightsClimbed_max_hrs_between",
        #         "day_of_week",
        #         "slp_StepCount_hrs_max_max",
        #         "distance_per_step",
        #         "hr_02:40:00",
        #         "is_workday",
        #         "slp_DistanceWalkingRunning_hrs_min_max",
        #         "steps_23:35:00",
        #         "hr_08:15:00",
        #         "slp_FlightsClimbed_hrs_min_max",
        #         "hr_02:00:00",
        #         "cal_08:25:00",
        #         "min_endDate_max_hr",
        #         "BodyMass",
        #         "cal_05:40:00",
        #         "hr_03:55:00",
        #         "cal_08:10:00",
        #         "BasalEnergyBurned",
        #         "dow_median",
        #         "FlightsClimbed",
        #         "avg_startDate_max_hr",
        #         "max_startDate_min_hr",
        #         "ActiveEnergyBurned",
        #         "avg_endDate_min_hr",
        #         "hr_03:25:00",
        #         "cal_08:15:00",
        #         "slp_AppleStandTime_max_hrs_between",
        #         "slp_StepCount_hrs_max_min",
        #         "day_of_year",
        #         "hr_06:55:00",
        #         "hr_06:45:00",
        #         "slp_StepCount_hrs_min_max",
        #         "doy_cos",
        #         "steps_08:10:00"]
        
        # self.train = self.train[to_keep]
        
        self.X = self.train.drop(columns=["sleep_hours", "date"])
        self.features = self.X.columns
        
        
        self.y = self.train["sleep_hours"].fillna(method="ffill")
        

    @staticmethod
    def _fix_doubling(df: pd.DataFrame, col: str, date_col: str = "date", start_date: str = "2017-09-27", end_date: str = "2018-06-12") -> pd.DataFrame:
        """Fixes doubling happening in a dataframe for a specific column"""
        logging.debug(f"Fixing doubling for {col}.")
        
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        date_range_mask = (df[date_col] <= end_date) & (df[date_col] >= start_date)
        df.loc[date_range_mask, col] = df.loc[date_range_mask, col] / 2

        return df

    @staticmethod
    def _is_daylight_savings(date, timezone: str = "US/Eastern") -> int:
        """Checks if a given date is in daylight savings time for a specific timezone."""
        # date = date.date()
        tz = pytz.timezone(timezone)
        dt = tz.localize(datetime.combine(date, datetime.min.time()), is_dst=None)
        return int(dt.dst() != timedelta(0))
    

    def get_sleep_times(self):
        # Read the CSV file
        df = pd.read_csv("./data/train_detailed.csv", low_memory=False)

        # Remove rows where "value" is HKCategoryValueSleepAnalysisInBed
        df = df[df["value"] != "HKCategoryValueSleepAnalysisInBed"]
        
        df = df.sort_values(by=['startDate', 'endDate'])

        # Convert startDate and endDate columns to datetime objects
        df["startDate"] = pd.to_datetime(df["startDate"])
        df["endDate"] = pd.to_datetime(df["endDate"])

        # Create adjusted_start_date column by subtracting 12 hours from startDate
        df['adjusted_start_date'] = (df['startDate'] - pd.DateOffset(hours=12)).dt.date

        # Save all unique adjusted_start_date values
        unique_dates = pd.DataFrame(self.train.date.unique(), columns=["adjusted_start_date"])

        # Filter rows with startDate hours >= 22 or endDate hours <= 10
        df = df[(df["startDate"].dt.hour >= 22) | (df["endDate"].dt.hour <= 10)]

        # Group by adjusted_start_date and get the min startDate and max endDate
        df = df.groupby("adjusted_start_date").agg({"startDate": "min", "endDate": "max"}).reset_index()

        # Convert startDate and endDate to hours since midnight
        df["startDate"] = df["startDate"].dt.hour + df["startDate"].dt.minute / 60 + df["startDate"].dt.second / 3600
        df["endDate"] = df["endDate"].dt.hour + df["endDate"].dt.minute / 60 + df["endDate"].dt.second / 3600

        # If startDate is less than 12, add 24 hours
        df.loc[df["startDate"] < 12, "startDate"] += 24

        # Merge the results with the unique_dates DataFrame
        final_df = unique_dates.merge(df, on="adjusted_start_date", how="left").fillna(method="ffill").fillna(method="bfill")
        
        return final_df.startDate, final_df.endDate

    @staticmethod
    def _process_interval_data(csv_df, interval_minutes=5):
        def round_time_to_nearest_interval(time):
            minutes = (time.hour * 60) + time.minute
            rounded_minutes = round(minutes / interval_minutes) * interval_minutes
            return dt.time(hour=(rounded_minutes // 60) % 24, minute=rounded_minutes % 60)

        def add_12_hours_to_time(time_obj):
            datetime_obj = dt.datetime.combine(dt.date(1, 1, 1), time_obj)
            datetime_obj += dt.timedelta(hours=12)
            return datetime_obj.time()

        #unique_dates = pd.DataFrame(self.train.date.unique(), columns=["date"])
        
        df = csv_df.copy()
        
        df = df[["startDate", "endDate", "value"]]
        df['startDate'] = pd.to_datetime(df['startDate']) - pd.Timedelta(hours=12)
        df['date'] = df['startDate'].dt.date
        df['time'] = df['startDate'].dt.time.map(round_time_to_nearest_interval)
        df_grouped = df.groupby(['date', 'time'])['value'].mean().reset_index()
        df_pivot = df_grouped.pivot_table(index='date', columns='time', values='value').reset_index()

        df_pivot.set_index('date', inplace=True)
        time_start = dt.time(hour=9) #9PM Start
        time_end = dt.time(hour=21) # 9AM End
        df_filtered = df_pivot.loc[:, (df_pivot.columns >= time_start) & (df_pivot.columns <= time_end)]
        df_filtered = df_filtered.rename(columns=add_12_hours_to_time)
        df_filtered.iloc[:, 1:] = df_filtered.iloc[:, 1:].interpolate(axis=1).ffill(axis=1).bfill(axis=1)
        df_filtered = df_filtered.reset_index()

        #final_df = unique_dates.merge(df_filtered, on="date", how="left")
        
        df_filtered.columns = [str(name).replace(':', '_') for name in df_filtered.columns]
        
        return df_filtered


    @staticmethod
    def _calculate_night_hours(df):
        
        df['startDate'] = pd.to_datetime(df['startDate']).dt.tz_localize(None)
        df['endDate'] = pd.to_datetime(df['endDate']).dt.tz_localize(None)
        
        df = df.sort_values(by=['startDate', 'endDate'])
        
        # Get the date range in the dataframe
        min_date = df['startDate'].min().date()
        max_date = df['endDate'].max().date()

        # Initialize an empty list to store the results
        results = []

        # Loop through each date in the range
        for date in pd.date_range(min_date, max_date):
            # startSleep time boundaries - Based on analysis of train_detailed
            start_day = pd.Timestamp.combine(date, pd.Timestamp('22:30:00').time())
            end_day = pd.Timestamp.combine(date + pd.DateOffset(1), pd.Timestamp('01:30:00').time())
            
            # endSleep time boundaries - Based on analysis of train_detailed
            start_night = pd.Timestamp.combine(date + pd.DateOffset(1), pd.Timestamp('06:30:00').time())
            end_night = pd.Timestamp.combine(date + pd.DateOffset(1), pd.Timestamp('9:30:00').time())

            # Filter the dataframe for max_endDate
            mask_endDate = (df['endDate'] >= start_day) & (df['endDate'] <= end_day)
            filtered_df_endDate = df[mask_endDate]

            # Filter the dataframe for min_startDate
            mask_startDate = (df['startDate'] >= start_night) & (df['startDate'] <= end_night)
            filtered_df_startDate = df[mask_startDate]

            # Find max_endDate and min_startDate
            min_endDate = filtered_df_endDate['endDate'].min() # if not filtered_df_endDate.empty else pd.to_datetime(start_day)
            max_endDate = filtered_df_endDate['endDate'].max() # if not filtered_df_endDate.empty else pd.to_datetime(end_day)
            min_startDate = filtered_df_startDate['startDate'].min() # if not filtered_df_startDate.empty else pd.to_datetime(start_night)
            max_startDate = filtered_df_startDate['startDate'].max() # if not filtered_df_startDate.empty else pd.to_datetime(end_night)
            
            # Append the results to the list
            results.append({
                'date': date,
                'min_endDate': min_endDate, # Min Possible Start Sleeping
                'max_endDate': max_endDate, # Max Possible Start Sleeping
                'min_startDate': min_startDate, # Min Possible End Sleeping
                'max_startDate': max_startDate, # Max Possible End Sleeping
            })

        # Convert the results to a dataframe and return
        result_df = pd.DataFrame(results)
        
        # Time Differences in hours # Attempt to manually calculate sleep time - doesn't work, but still useful
        result_df["hrs_min_min"] = (result_df["min_startDate"] - result_df["min_endDate"]).dt.total_seconds() / 3600
        result_df["hrs_min_max"] = (result_df["min_startDate"] - result_df["max_endDate"]).dt.total_seconds() / 3600
        result_df["hrs_max_min"] = (result_df["max_startDate"] - result_df["min_endDate"]).dt.total_seconds() / 3600
        result_df["hrs_max_max"] = (result_df["max_startDate"] - result_df["max_endDate"]).dt.total_seconds() / 3600
        
        # Hours
        result_df["min_endDate_hr"] = result_df["min_endDate"].dt.hour
        result_df["max_endDate_hr"] = result_df["max_endDate"].dt.hour
        result_df["min_startDate_hr"] = result_df["min_startDate"].dt.hour
        result_df["max_startDate_hr"] = result_df["max_startDate"].dt.hour

        
        result_df = result_df.drop(columns = ["min_endDate", "max_endDate", "min_startDate", "max_startDate", "avg_endDate", "avg_startDate"], errors="ignore").reset_index(drop=True)
        
        return result_df
        
    
    def _create_xml_features(self, path: str) -> pd.DataFrame:
        """Create XML features from the provided CSV file."""
        logging.debug(f"Featurizing {path}")
        csv_df = pd.read_csv(path, low_memory=False)
        base_name = os.path.basename(path).split(".")[0]
        
    
        value = "totalEnergyBurned" if base_name == "Workout" else "value"
        agg_func = "mean" if base_name == "BodyMassIndex" else "sum"
        
        csv_df["startDate"] = pd.to_datetime(csv_df["startDate"]).dt.tz_convert("US/Eastern")
        csv_df["endDate"] = pd.to_datetime(csv_df["endDate"]).dt.tz_convert("US/Eastern")
        csv_df["date"] = (pd.to_datetime(csv_df["startDate"])- pd.to_timedelta('12:00:00')).dt.date
        csv_df["time"] = pd.to_datetime(csv_df["startDate"]).dt.time
        
        csv_df = csv_df.sort_values(by=['startDate', 'endDate'])
        
        csv_df["hours_between"] = (csv_df["startDate"].shift(-1) - csv_df["endDate"]).dt.total_seconds() / 3600
        csv_df['is_night'] = (csv_df['startDate'] - pd.Timedelta(hours=12)).dt.date == csv_df['startDate'].dt.date
        
        groupby_agg = {
            "startDate": ["max", "min"],
            "endDate": ["max", "min"],
            f"{value}": agg_func,
            "hours_between" : "max"
        }

        df = csv_df.groupby("date").agg(groupby_agg).reset_index()
        df.columns = ["_".join(tup).rstrip("_") for tup in df.columns.values]

        df = df.rename(columns={f"{value}_{agg_func}": base_name})
        df = df.rename(columns={"hours_between_max": "slp_"+base_name+"_max_hrs_between"})
        
        # Sum hours between if is_night is True 
        def sum_night_hours(group):
            return group.loc[group['is_night'], 'hours_between'].sum()

        df[f"slp_{base_name}_sum_hrs_between"] = csv_df.groupby("date").apply(sum_night_hours).values
        
        # Count hours between if is_night is True (sleep interruptions)
        def count_night_hours(group):
            return group.loc[group['is_night'], 'hours_between'].count()
        
        df[f"slp_{base_name}_count_hrs_between"] = csv_df.groupby("date").apply(count_night_hours).values
        
        
        for time_col in ["startDate_max", "startDate_min", "endDate_max", "endDate_min"]:
            # Hours
            col_prefix = f"{base_name}_{time_col}_"
            df[col_prefix + "hr"] = df[time_col].dt.hour

        
        # Night Hours
        night_hours_df = self._calculate_night_hours(csv_df)
        night_hours_df = night_hours_df.add_prefix(f"slp_{base_name}_")
        night_hours_df = night_hours_df.rename(columns={f"slp_{base_name}_date": "date"})
        night_hours_df["date"] = pd.to_datetime(night_hours_df["date"]).dt.date
        df = df.merge(night_hours_df, how="left", on = "date")


        # Check if csv_df "value" is numeric, and if so, calculate interval data
        if pd.to_numeric(csv_df["value"], errors='coerce').notnull().all():

            # Time intervals
            intervals = self._process_interval_data(csv_df)
            intervals = intervals.add_prefix(f"slp_{base_name}_")
            intervals = intervals.rename(columns={f"slp_{base_name}_date": "date"})
            intervals["date"] = pd.to_datetime(intervals["date"]).dt.date
            df = df.merge(intervals, how="left", on = "date")


        df = self._fix_doubling(df, base_name)
        df = df.drop(columns=["startDate_max", "startDate_min", "endDate_max", "endDate_min"]) 

        return df

    def _workout_features(self, path: str) -> pd.DataFrame:
        logging.debug(f"Featurizing {path}")
        workout = pd.read_csv(path, low_memory=False)
        workout["date"] = pd.to_datetime(workout["startDate"]).dt.date
        workout = workout.groupby("date")[["duration","totalDistance","totalDistanceUnit","totalEnergyBurned","totalEnergyBurnedUnit"]].sum().reset_index()
        workout.columns = ["workout_" + col if col != "date" else col for col in workout.columns]
        
        return workout
    
    def create_xml_data(self, path: str, xml_files_names: List[str] = ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"]) -> pd.DataFrame:
        """Featurize XML data from given path."""
        logging.info("Creating XML data")
        
        xml_files_names = [os.path.join(path, f"{xml_file}.csv") for xml_file in xml_files_names]

        # Create DataFrame with date column from 1/1/2015 to 12/31/2023
        xml_data = pd.DataFrame({"date": pd.date_range(start="1/1/2015", end="12/31/2023", freq="D")})
        xml_data["date"] = xml_data["date"].dt.date

        # Parse each xml file output and merge with the train data
        for xml_file in xml_files_names:
            
            xml = self._workout_features(xml_file) if "Workout" in xml_file else self._create_xml_features(xml_file) 
            xml_data = pd.merge(xml_data, xml, on="date", how="outer")
        
        self.xml_data = xml_data
        
        return xml_data

    def create_activity_data(self, path):
        logging.info("Creating activity data")
        
        ad = pd.read_csv(path)
        
        logging.debug(f"Featurizing {path}")
        ad = pd.read_csv(path, low_memory=False)
        ad["date"] = pd.to_datetime(ad["dateComponents"]).dt.date
        
        ad = ad.groupby("date")[["activeEnergyBurned","appleExerciseTime","appleStandHours"]].sum().reset_index()
        
        return ad

    def _feature_engineering(self, df: pd.DataFrame, lookback: int = None) -> pd.DataFrame:
        """Feature engineering for time series data (Requires XML data)"""
        logging.info("Featurizing time series data")
        df = df.copy()       
        
        # Sort by date
        df = df.sort_values(by="date")
        
        ### Time series Data ###
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        # df["dow_sin"] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        # df["dow_cos"] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df["is_sunday"] = (df["day_of_week"] == 6).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_tuesday"] = (df["day_of_week"] == 1).astype(int)
        df["is_wednesday"] = (df["day_of_week"] == 2).astype(int)
        df["is_thursday"] = (df["day_of_week"] == 3).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_saturday"] = (df["day_of_week"] == 5).astype(int)
        # df["is_mwf"] = df["day_of_week"].isin([0,2,4])
        # df["is_tth"] = df["day_of_week"].isin([1,3])
        df["is_weekend"] = df["day_of_week"].isin([5,6])
        
        df["day_of_month"] = pd.to_datetime(df["date"]).dt.day
        # df["dom_sin"] = np.sin(df['day_of_month'] * (2 * np.pi / 30))
        # df["dom_cos"] = np.cos(df['day_of_month'] * (2 * np.pi / 30))

        df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
        df["doy_sin"] = np.sin(df['day_of_year'] * (2 * np.pi / 365))
        df["doy_cos"] = np.cos(df['day_of_year'] * (2 * np.pi / 365))

        df["month"] = pd.to_datetime(df["date"]).dt.month # Month
        df["month_sin"] = np.sin(df['month'] * (2 * np.pi / 12))
        df["month_cos"] = np.cos(df['month'] * (2 * np.pi / 12))
        
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start="2014-01-01", end="2023-12-31") 
        df["is_holiday"] = df["date"].apply(lambda x: 1 if x in holidays else 0)
        
        # Not holiday and workday (Mon-Fri)
        df["is_workday"] = (df["is_holiday"] == 0) & (df["day_of_week"] < 5)
        
        # df["is_daylight_savings"] = df["date"].apply(lambda x: self._is_daylight_savings(x))

        # Create dow_median column #? could create a bunch of useful summary statistics to use later?
        # DOW Median up until the current day
        df["dow_median"] = df.groupby('day_of_week').apply(lambda group: group.sort_values(by='date')['sleep_hours'].expanding().median().shift(1)).reset_index(level=0, drop=True)
        
        # DOY Mean up until that day
        #doy_mean = df.groupby("day_of_year")["sleep_hours"].mean()
        df["doy_mean"] = df.groupby('day_of_year').apply(lambda group: group.sort_values(by='date')['sleep_hours'].expanding().mean().shift(1)).reset_index(level=0, drop=True)

        #df = df.drop(columns = ["day_of_week", "day_of_year", "is_weekend", "is_holiday"])
        
        # Interactions
        logging.info("Creating interactions...")
        df["distance_per_step"] = df["DistanceWalkingRunning"] / df["StepCount"] # To account for jumping, hiking, etc.
        df["calorie_per_step"] = df["BasalEnergyBurned"] / df["StepCount"] # To account for intensity of exercise
        df["calorie_per_distance"] = df["BasalEnergyBurned"] / df["DistanceWalkingRunning"] # Gym days vs. Outdoor days
        
        logging.debug("Creating time averages...")
        for pattern in ['startDate_max_hr', 'startDate_min_hr', 'endDate_max_hr', 'endDate_min_hr']:
            filtered_columns = [col for col in df.columns if pattern in col]
            df_no_zeros = df[filtered_columns].replace(0, np.nan)
            df[f'avg_{pattern}'] = df_no_zeros.mean(axis=1)
            df[f'max_{pattern}'] = df_no_zeros.max(axis=1)
            df[f'min_{pattern}'] = df_no_zeros.min(axis=1)
            df = df.drop(columns = filtered_columns, errors = "ignore")
            
        return df

    def create_train(self, path: str, freq_threshold: float = 0.9) -> pd.DataFrame:
        """Create train dataset with provided path and frequency threshold."""
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Filter date on >= self.start_date
        #df = df[df["date"] >= pd.to_datetime(self.start_date)].reset_index(drop=True)
        df = df[df["date"] >= pd.to_datetime(self.start_date).date()].reset_index(drop=True)

        
        df = df.sort_values(by="date")
        df = self._fix_doubling(df, "sleep_hours")
        
        start_date, end_date = df["date"].min(), df["date"].max()
        logging.debug(f"Start date: {start_date}, End date: {end_date}")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        date_range = pd.DataFrame({"date": date_range})
        date_range["date"] = date_range["date"].dt.date
        
        df = date_range.merge(df, on="date", how="left")
        logging.info(f"Missing days: {df.sleep_hours.isna().sum()}")
        
        df = df.merge(self.xml_data, on="date", how="left") #Add XML data
        df = df.merge(self.activity_data, on="date", how="left") #Add Activity data
        df = self._feature_engineering(df) # Feature engineering
        
        # This is useless
        df = df.drop(columns = ["AppleStandHour"], errors = "ignore")

        # Drop columns that are mostly missing/nulls 75%
        # to_drop = [c for c in df.columns if df[c].isna().sum() > len(df) * 0.75]
        # df = df.drop(columns=to_drop, errors = "ignore")
        # logging.info(f"Dropped null columns: {to_drop}")
        
        # Drop columns that freq_threshold% is only one value
        # to_drop = [c for c in df.columns if df[c].value_counts(normalize=True, dropna=False).iloc[0] > freq_threshold]
        # df = df.drop(columns=to_drop, errors = "ignore")
        # logging.info(f"Dropped non-unique columns: {to_drop}")
    
        to_keep = ["date", "sleep_hours", "is_workday",
                    "slp_BasalEnergyBurned_04_15_00",
                    "slp_StepCount_07_00_00",
                    "slp_DistanceWalkingRunning_07_50_00",
                    "slp_ActiveEnergyBurned_07_10_00",
                    "slp_ActiveEnergyBurned_00_45_00",
                    "slp_StepCount_07_30_00",
                    "slp_ActiveEnergyBurned_06_25_00",
                    "slp_StepCount_23_00_00",
                    "slp_ActiveEnergyBurned_22_15_00",
                    "slp_EnvironmentalAudioExposure_00_55_00",
                    "slp_ActiveEnergyBurned_04_20_00",
                    "slp_HeartRate_06_55_00",
                    "slp_BasalEnergyBurned_23_25_00",
                    "FlightsClimbed",
                    "slp_ActiveEnergyBurned_01_35_00",
                    "slp_HeartRate_06_30_00",
                    "slp_ActiveEnergyBurned_sum_hrs_between",
                    "slp_HeartRate_03_45_00",
                    "slp_EnvironmentalAudioExposure_00_05_00",
                    "slp_ActiveEnergyBurned_01_10_00",
                    "min_endDate_max_hr",
                    "slp_ActiveEnergyBurned_07_20_00",
                    "slp_EnvironmentalAudioExposure_23_20_00",
                    "slp_ActiveEnergyBurned_22_45_00",
                    "slp_StepCount_hrs_min_max",
                    "slp_StepCount_hrs_max_max",
                    "slp_HeartRate_01_30_00",
                    "slp_DistanceWalkingRunning_07_00_00",
                    "day_of_week",
                    "slp_ActiveEnergyBurned_07_40_00",
                    "slp_EnvironmentalAudioExposure_00_50_00",
                    "slp_ActiveEnergyBurned_hrs_max_min",
                    "slp_HeartRate_07_00_00",
                    "slp_DistanceWalkingRunning_06_50_00",
                    "slp_ActiveEnergyBurned_23_45_00",
                    "slp_StepCount_00_00_00",
                    "slp_ActiveEnergyBurned_07_35_00",
                    "doy_cos",
                    "slp_ActiveEnergyBurned_00_05_00",
                    "slp_StepCount_21_20_00",
                    "slp_ActiveEnergyBurned_00_15_00",
                    "slp_EnvironmentalAudioExposure_03_25_00",
                    "BasalEnergyBurned",
                    "slp_AppleStandTime_max_hrs_between",
                    "slp_DistanceWalkingRunning_05_10_00",
                    "WalkingSpeed",
                    "slp_HeartRate_01_00_00",
                    "slp_BasalEnergyBurned_00_10_00",
                    "slp_BasalEnergyBurned_21_10_00",
                    "slp_DistanceWalkingRunning_count_hrs_between",
                    "slp_StepCount_08_00_00",
                    "slp_ActiveEnergyBurned_00_30_00",
                    "slp_StepCount_count_hrs_between",
                    "avg_endDate_min_hr",
                    "slp_ActiveEnergyBurned_23_05_00",
                    "avg_startDate_max_hr",
                    "slp_BasalEnergyBurned_01_00_00",
                    "slp_ActiveEnergyBurned_06_40_00",
                    "slp_StairAscentSpeed_sum_hrs_between",
                    "slp_EnvironmentalAudioExposure_00_45_00",
                    "slp_StepCount_23_10_00",
                    "slp_StepCount_sum_hrs_between",
                    "slp_StepCount_07_40_00",
                    "slp_ActiveEnergyBurned_23_35_00",
                    "day_of_year",
                    "slp_DistanceWalkingRunning_07_20_00",
                    "slp_ActiveEnergyBurned_count_hrs_between",
                    "slp_ActiveEnergyBurned_07_15_00",
                    "slp_EnvironmentalAudioExposure_00_40_00",
                    "slp_StepCount_22_30_00",
                    "slp_StepCount_max_endDate_hr",
                    "slp_HeartRate_02_00_00",
                    "slp_StepCount_23_30_00",
                    "slp_FlightsClimbed_max_hrs_between",
                    "slp_AppleStandTime_sum_hrs_between",
                    "slp_ActiveEnergyBurned_23_50_00",
                    "DistanceWalkingRunning",
                    "slp_FlightsClimbed_00_00_00",
                    "slp_StepCount_07_10_00",
                    "slp_BasalEnergyBurned_02_15_00",
                    "slp_ActiveEnergyBurned_00_00_00",
                    "slp_HeartRate_07_50_00",
                    "slp_StepCount_max_hrs_between",
                    "slp_StepCount_07_50_00",
                    "slp_FlightsClimbed_sum_hrs_between",
                    "slp_OxygenSaturation_01_00_00",
                    "slp_HeartRate_08_05_00",
                    "slp_BasalEnergyBurned_08_35_00",
                    "slp_DistanceWalkingRunning_07_30_00",
                    "slp_HeartRateVariabilitySDNN_max_hrs_between",
                    "doy_sin",
                    "slp_FlightsClimbed_07_00_00"
            ]
        
        df = df[to_keep]
        
        self.columns = df.columns
        self.train = df.reset_index(drop=True)
        self.target = self.train.pop("sleep_hours")
        
        return df

    def create_lags(self, lags = 7):
        X = self.X

        for lag in range(1, lags + 1):
            X[f"sleep_hours_lag_{lag}"] = self.y.shift(lag)
            X[f"sleep_hours_lag_{lag}"] = X[f"sleep_hours_lag_{lag}"].fillna(X[f"sleep_hours_lag_{lag}"].mean())
    
        self.X = X
        self.features = X.columns
        self.lags = True
        
    def train_test_split(self, train_size: float = 0.8):
        """Split data into train and test set"""
        logging.info("Splitting data into train and test set")
        X = self.X
        y = self.y
        
        train_size = int(len(X) * train_size)
        X_train, y_train = X[:train_size].reset_index(drop=True), y[:train_size].reset_index(drop=True),
        X_test, y_test = X[train_size:].reset_index(drop=True), y[train_size:].reset_index(drop=True),
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
    
    @staticmethod
    def _create_preprocessor(impute_strategy: str = "median"):
        """Create preprocessor for pipeline"""
        logging.debug("Creating preprocessor")
        imputer = SimpleImputer(strategy="median") 
        scaler = StandardScaler()
        preprocessor = Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", scaler)
        ])
        return preprocessor
    
    def preprocess_data(self):
        """Preprocess data using the preprocessor"""
        logging.info("Scaling and imputing data")
    
        X_train, X_test = self.X_train, self.X_test
        
        preprocessor = self._create_preprocessor()

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns = self.features)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns = self.features)
        
        self.X_train, self.X_test = X_train, X_test
        self.preprocessor = preprocessor
    
    def scale_target(self):
        """scales the target variable - sleep hours (useful for NNs)"""
        logging.info("Scaling target variable with minmax")
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(self.y_train.values.reshape(-1,1))
        y_test = scaler.transform(self.y_test.values.reshape(-1,1))
        
        self.target_scaler = scaler
        
        return y_train, y_test
        
    def create_submission(self, model, submission_path: str = "./data/test.csv", preprocess=False) -> pd.DataFrame:
        """Create submission dataset with provided path."""
        logging.info("Creating submission dataset")
        df = pd.read_csv(submission_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        df = df.merge(self.xml_data, on="date", how="left")
        df = df.merge(self.activity_data, on="date", how="left") #Add Activity data
        df = self._feature_engineering(df)
        df = df[self.columns] # Keep only columns that are in the train data
        print(df.columns)
        sub = pd.DataFrame({"date": df["date"]}) # Create dataframe with date column from df
        
        df = df.drop(columns=["date", "sleep_hours"], errors = 'ignore') # Drop date column from df
        logging.debug(f"Submission columns: {df.columns}")
        
        if preprocess:
            df = pd.DataFrame(self.preprocessor.transform(df), columns = self.features)   
        
        # Print df datatype
        logging.info(f"Submission data type: {df.dtypes}")
        preds = model.predict(df) # predictions
        
        # Create submission dataframe with date and predictions
        sub["sleep_hours"] = preds
        
        self.last_submission = sub
        return sub
    
    def to_parquet(self, train_path: str = "./train_data.parquet") -> None:
        """Save data to parquet files"""
        logging.info("Saving to Parquet file...")
        self.train.to_parquet(train_path)
