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

from hmmlearn import hmm


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s") # Configure logging

# Original columns = ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"]
# Activity Summary is by date!


class POG4_Dataset():
    """Initialize Dataset class."""
    def __init__(self, train_path: str = "./data/train.csv",  xml_export_path: str = "./data/xml_export", start_date = "2018-01-01") -> None:
        self.start_date = start_date
        self.xml_data = self.create_xml_data(xml_export_path, ['Workout', 'WalkingSpeed', 'ActiveEnergyBurned', 'RunningSpeed', 'AppleWalkingSteadiness', 'EnvironmentalAudioExposure', 'StairDescentSpeed', 'LowHeartRateEvent', 'RunningGroundContactTime', 'DistanceCycling', 'HandwashingEvent', 'NumberOfTimesFallen', 'BasalEnergyBurned', 'MindfulSession', 'SixMinuteWalkTestDistance', 'StairAscentSpeed', 'HKDataTypeSleepDurationGoal', 'HeartRateVariabilitySDNN', 'Height', 'OxygenSaturation', 'RunningStrideLength', 'HeartRateRecoveryOneMinute', 'WalkingStepLength', 'SwimmingStrokeCount', 'BodyMass', 'FlightsClimbed', 'DietaryEnergyConsumed', 'AudioExposureEvent', 'HeadphoneAudioExposure', 'StepCount', 'WalkingAsymmetryPercentage', 'RespiratoryRate', 'HeartRate', 'DietaryWater', 'BodyMassIndex', 'RunningPower', 'VO2Max', 'DistanceWalkingRunning', 'HeadphoneAudioExposureEvent', 'HighHeartRateEvent', 'WalkingDoubleSupportPercentage', 'AppleExerciseTime', 'RestingHeartRate', 'AppleStandTime', 'WalkingHeartRateAverage', 'DistanceSwimming', 'EnvironmentalSoundReduction', 'AppleStandHour', 'RunningVerticalOscillation'])
        self.activity_data = self.create_activity_data(os.path.join(xml_export_path, "ActivitySummary.csv"))
        self.train = self.create_train(train_path)
        self.X = self.train.drop(columns=["sleep_hours", "date"])
        self.features = self.X.columns
        self.y = self.train["sleep_hours"].fillna(method="ffill")
        self.preprocessor = None 
        
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
    
    def _create_xml_features(self, path: str) -> pd.DataFrame:
        """Create XML features from the provided CSV file."""
        logging.debug(f"Featurizing {path}")
        csv_df = pd.read_csv(path, low_memory=False)
        base_name = os.path.basename(path).split(".")[0]

        value = "totalEnergyBurned" if base_name == "Workout" else "value"
        agg_func = "mean" if base_name == "BodyMassIndex" else "sum"
        
        csv_df["startDate"] = pd.to_datetime(csv_df["startDate"]).dt.tz_convert("US/Eastern")
        csv_df["endDate"] = pd.to_datetime(csv_df["endDate"]).dt.tz_convert("US/Eastern")
        csv_df["date"] = pd.to_datetime(csv_df["startDate"]).dt.date
        csv_df["time"] = pd.to_datetime(csv_df["startDate"]).dt.time

        groupby_agg = {
            "startDate": ["max", "min"],
            "endDate": ["max", "min"],
            f"{value}": agg_func
        }

        df = csv_df.groupby("date").agg(groupby_agg).reset_index()
        df.columns = ["_".join(tup).rstrip("_") for tup in df.columns.values]

        df = df.rename(columns={f"{value}_{agg_func}": base_name})
        
        for time_col in ["startDate_max", "startDate_min", "endDate_max", "endDate_min"]:
            # Hours
            col_prefix = f"{base_name}_{time_col}_"
            df[col_prefix + "hr"] = df[time_col].dt.hour


        # Attempt to manually calculate sleep time - doesn't work, but still useful
        df[base_name+"_hrs_btween"] = (df["startDate_min"].shift(-1) - df["startDate_max"]).dt.total_seconds() / 3600

        df = self._fix_doubling(df, base_name)
        df = df.drop(columns=["startDate_max", "startDate_min", "endDate_max", "endDate_min"]) 
        
        
        # Sleep Modeling - HIGHLY EXPERIMENTAL
        if base_name in ["HeartRate", "RestingHeartRate", "StepCount","DistanceWalkingRunning"]:
            sleep_estimates = self._estimate_sleep_lengths_hmm(csv_df[["startDate", "value"]], "value")
            sleep_estimates = sleep_estimates.rename(columns={"sleep_hours": f"{base_name}_sleep_hours"})
            #FutureWarning: Comparison of Timestamp with datetime.date is deprecated in order to match the standard library behavior. In a future version these will be considered non-comparable. Use 'ts == pd.Timestamp(date)' or 'ts.date() == date' instead.
            df = df.merge(sleep_estimates, how="left", on = "date")

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
        
        ### Time series Data ###
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["dow_sin"] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df["dow_cos"] = np.cos(df['day_of_week'] * (2 * np.pi / 7))

        df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
        df["doy_sin"] = np.sin(df['day_of_year'] * (2 * np.pi / 365))
        df["doy_cos"] = np.cos(df['day_of_year'] * (2 * np.pi / 365))

        df["month"] = pd.to_datetime(df["date"]).dt.month # Month
        df["month_sin"] = np.sin(df['month'] * (2 * np.pi / 12))
        df["month_cos"] = np.cos(df['month'] * (2 * np.pi / 12))
        
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start="2014-01-01", end="2023-12-31") 
        df["is_holiday"] = df["date"].apply(lambda x: 1 if x in holidays else 0)
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        df["is_workday"] = 1-(df["is_weekend"] + df["is_holiday"])
        df["is_daylight_savings"] = df["date"].apply(lambda x: self._is_daylight_savings(x))

        # Create dow_median column #? could create a bunch of useful summary statistics to use later?
        median = df.groupby("day_of_week")["sleep_hours"].median()
        df["dow_median"] = df["day_of_week"].map(median)

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
            
        df["avg_startDate_max_sin"] = np.sin(df['avg_startDate_max_hr'] * (2 * np.pi / 24))
        df["avg_startDate_max_cos"] = np.cos(df['avg_startDate_max_hr'] * (2 * np.pi / 24))
        df["avg_startDate_min_sin"] = np.sin(df['avg_startDate_min_hr'] * (2 * np.pi / 24))
        df["avg_startDate_min_cos"] = np.cos(df['avg_startDate_min_hr'] * (2 * np.pi / 24))
        df["avg_endDate_max_sin"] = np.sin(df['avg_endDate_max_hr'] * (2 * np.pi / 24))
        df["avg_endDate_max_cos"] = np.cos(df['avg_endDate_max_hr'] * (2 * np.pi / 24))
        df["avg_endDate_min_sin"] = np.sin(df['avg_endDate_min_hr'] * (2 * np.pi / 24))
        df["avg_endDate_min_cos"] = np.cos(df['avg_endDate_min_hr'] * (2 * np.pi / 24))

        return df

    def create_train(self, path: str, freq_threshold: float = 0.9) -> pd.DataFrame:
        """Create train dataset with provided path and frequency threshold."""
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # Filter date on >= self.start_date
        df = df[df["date"] >= pd.to_datetime(self.start_date)].reset_index(drop=True)
        
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
        to_drop = [c for c in df.columns if df[c].isna().sum() > len(df) * 0.75]
        df = df.drop(columns=to_drop, errors = "ignore")
        logging.info(f"Dropped null columns: {to_drop}")
        
        # Drop columns that freq_threshold% is only one value
        to_drop = [c for c in df.columns if df[c].value_counts(normalize=True, dropna=False).iloc[0] > freq_threshold]
        df = df.drop(columns=to_drop, errors = "ignore")
        logging.info(f"Dropped non-unique columns: {to_drop}")
        
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
        
    def create_submission(self, model, submission_path: str = "./data/test.csv") -> pd.DataFrame:
        """Create submission dataset with provided path."""
        logging.info("Creating submission dataset")
        df = pd.read_csv(submission_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        df = df.merge(self.xml_data, on="date", how="left")
        df = df.merge(self.activity_data, on="date", how="left") #Add Activity data
        df = self._feature_engineering(df)
        df = df[self.columns] # Keep only columns that are in the train data
        
        sub = pd.DataFrame({"date": df["date"]}) # Create dataframe with date column from df
        
        df = df.drop(columns=["date", "sleep_hours"], errors = 'ignore') # Drop date column from df
        logging.debug(f"Submission columns: {df.columns}")
        
        # if self.lags:
        # # Iterate through prediction days and create lags using previous predictions
        # # Assume 0 for initial lags
        #     for i in range(1, 8):
        #         df[f"sleep_hours_lag_{i}"] = 0
        #         df[f"sleep_hours_lag_{i}"] = df[f"sleep_hours_lag_{i}"].fillna(df[f"sleep_hours_lag_{i}"].mean())


        
        if self.preprocessor is not None:
            df = pd.DataFrame(self.preprocessor.transform(df), columns = self.features)   
        
        preds = model.predict(df) # predictions
        
        # Create submission dataframe with date and predictions
        sub["sleep_hours"] = preds
    
        
        self.last_submission = sub
        return sub
    
    @staticmethod
    def _estimate_sleep_lengths_hmm(df, feature, sleep_start_hour=22, sleep_end_hour=9, resample_freq='3T', window_size=2, n_components=2):
        df['timestamp'] = pd.to_datetime(df['startDate'])
        df = df.drop_duplicates(subset=['timestamp']).sort_values(by=['timestamp'])

        # Create a new DataFrame with a fixed interval, merge and interpolate
        df_resampled = pd.DataFrame(pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq=resample_freq), columns=['timestamp'])
        df_resampled = pd.merge(df_resampled, df, on='timestamp', how='left').fillna(method='ffill')

        # Apply moving average filter to the feature
        df_resampled['filtered_feature'] = df_resampled[feature].ewm(span=window_size).mean()

        # Prepare the feature data for the HMM
        feature_data = df_resampled['filtered_feature'].to_numpy().reshape(-1, 1)

        # Define a 2-state Gaussian HMM (sleep and wake states)
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, init_params='stmcw')

        # Fit the HMM to the feature data
        model.fit(feature_data)

        # Get the most likely sleep/wake state sequence
        state_sequence = model.predict(feature_data)

        # Add the sleep/wake state sequence to the DataFrame
        df_resampled['sleep'] = state_sequence

        # Restrict sleep detection to the specified sleep window
        df_resampled['sleep'] = np.where((df_resampled['sleep'] == 1) & 
                                        ((df_resampled['timestamp'].dt.hour >= sleep_start_hour) | 
                                        (df_resampled['timestamp'].dt.hour < sleep_end_hour)), 1, 0)

        # Calculate estimated sleep hours for each day
        df_sleep = df_resampled.groupby(df_resampled['timestamp'].dt.floor('D')).agg({'sleep': 'sum'})
        df_sleep['sleep_hours'] = df_sleep['sleep'] * pd.to_timedelta(resample_freq).seconds / 3600

        df_sleep = df_sleep.reset_index()
        df_sleep['date'] = pd.to_datetime(df_sleep['timestamp']).dt.date
        df_sleep = df_sleep.drop(columns=['timestamp', 'sleep'])
        df_sleep.loc[df_sleep.sleep_hours == 0, 'sleep_hours'] = np.nan

        return df_sleep
    
    def to_parquet(self, train_path: str = "./train_data.parquet") -> None:
        """Save data to parquet files"""
        logging.info("Saving to Parquet file...")
        self.train.to_parquet(train_path)
