import logging
from datetime import datetime, timedelta
import os
from typing import List

import numpy as np
import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Configure logging

class POG4_Dataset():
    """Initialize Dataset class."""
    def __init__( self, train_path: str = "./data/train.csv",  xml_export_path: str = "./data/xml_export") -> None:
        self.xml_data = self.create_xml_data(xml_export_path, ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"])
        self.train = self.create_train(train_path)
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
    
    def _create_xml_features(self, path: str) -> pd.DataFrame:
        """Create XML features from the provided CSV file."""
        logging.debug(f"Featurizing {path}")
        csv_df = pd.read_csv(path, low_memory=False)
        base_name = os.path.basename(path).split(".")[0]

        agg_func = "mean" if base_name == "BodyMassIndex" else "sum"
        
        csv_df["startDate"] = pd.to_datetime(csv_df["startDate"]).dt.tz_convert("US/Eastern")
        csv_df["endDate"] = pd.to_datetime(csv_df["endDate"]).dt.tz_convert("US/Eastern")
        csv_df["date"] = pd.to_datetime(csv_df["startDate"]).dt.date
        csv_df["time"] = pd.to_datetime(csv_df["startDate"]).dt.time

        groupby_agg = {
            "startDate": ["max", "min"],
            "endDate": ["max", "min"],
            "value": agg_func
        }

        csv_df = csv_df.groupby("date").agg(groupby_agg).reset_index()
        csv_df.columns = ["_".join(tup).rstrip("_") for tup in csv_df.columns.values]

        csv_df = csv_df.rename(columns={f"value_{agg_func}": base_name})
        
        for time_col in ["startDate_max", "startDate_min", "endDate_max", "endDate_min"]:
            # Trigonomic Hours
            col_prefix = f"{base_name}_{time_col}_"
            csv_df[col_prefix + "hr_sin"] = np.sin(2 * np.pi * csv_df[time_col].dt.hour / 24)
            csv_df[col_prefix + "hr_cos"] = np.cos(2 * np.pi * csv_df[time_col].dt.hour / 24)

        # Attempt to manually calculate sleep time - doesn't work, but still useful
        csv_df[base_name+"_hrs_btween"] = (csv_df["startDate_min"].shift(-1) - csv_df["startDate_max"]).dt.total_seconds() / 3600

        csv_df = self._fix_doubling(csv_df, base_name)
        csv_df = csv_df.drop(columns=["startDate_max", "startDate_min", "endDate_max", "endDate_min"]) 

        return csv_df
    
    def create_xml_data(self, path: str, xml_files_names: List[str] = ["BasalEnergyBurned", "BodyMass", "FlightsClimbed", "StepCount", "BodyMassIndex", "DistanceWalkingRunning"]) -> pd.DataFrame:
        """Featurize XML data from given path."""
        logging.info("Creating XML data")
        
        xml_files_names = [os.path.join(path, f"{xml_file}.csv") for xml_file in xml_files_names]

        # Create DataFrame with date column from 1/1/2015 to 12/31/2023
        xml_data = pd.DataFrame({"date": pd.date_range(start="1/1/2015", end="12/31/2023", freq="D")})
        xml_data["date"] = xml_data["date"].dt.date

        # Parse each xml file output and merge with the train data
        for xml_file in xml_files_names:
            xml = self._create_xml_features(xml_file)
            xml_data = pd.merge(xml_data, xml, on="date", how="outer")
        
        self.xml_data = xml_data
        
        return xml_data

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
        
        return df

    def create_train(self, path: str, freq_threshold: float = 0.9) -> pd.DataFrame:
        """Create train dataset with provided path and frequency threshold."""
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
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
        df = self._feature_engineering(df) # Feature engineering
        
        # Drop columns that freq_threshold% is only one value
        to_drop = [c for c in df.columns if df[c].value_counts(normalize=True).iloc[0] > freq_threshold]
        df = df.drop(columns=to_drop)
        logging.info(f"Dropped non-unique columns: {to_drop}")
        
        # Drop columns that are mostly missing/nulls
        to_drop = [c for c in df.columns if df[c].isna().sum() > len(df) * freq_threshold]
        df = df.drop(columns=to_drop)
        logging.info(f"Dropped null columns: {to_drop}")
        
        self.columns = df.columns
        self.train = df.reset_index(drop=True)
        self.target = self.train.pop("sleep_hours")
        
        return df

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
    
    def to_parquet(self, train_path: str = "./train_data.parquet") -> None:
        """Save data to parquet files"""
        logging.info("Saving to Parquet file...")
        self.train.to_parquet(train_path)

    def train_test_split(self, train_size: float = 0.8):
        """Split data into train and test set"""
        logging.info("Splitting data into train and test set")
        X = self.X
        y = self.y
        
        train_size = int(len(X) * train_size)
        X_train, y_train = X[:train_size].reset_index(drop=True), y[:train_size].reset_index(drop=True),
        X_test, y_test = X[train_size:].reset_index(drop=True), y[train_size:].reset_index(drop=True),
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        return X_train, X_test, y_train, y_test

    def preprocess_data(self):
        """Preprocess data using the preprocessor"""
        logging.info("Scaling and imputing data")
    
        X_train, X_test = self.X_train, self.X_test
        
        preprocessor = self._create_preprocessor()

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns = self.features)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns = self.features)
        
        self.X_train, self.X_test = X_train, X_test
        self.preprocessor = preprocessor
        
        return X_train, X_test
    
    
    def create_submission(self, model, submission_path: str = "./data/test.csv") -> pd.DataFrame:
        """Create submission dataset with provided path."""
        logging.info("Creating submission dataset")
        df = pd.read_csv(submission_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        df = df.merge(self.xml_data, on="date", how="left")
        df = self._feature_engineering(df)
        df = df[self.columns] # Keep only columns that are in the train data
        
        sub = pd.DataFrame({"date": df["date"]}) # Create dataframe with date column from df
        
        df = df.drop(columns=["date", "sleep_hours"], errors = 'ignore') # Drop date column from df
        logging.debug(f"Submission columns: {df.columns}")
        
        if self.preprocessor is not None:
            df = pd.DataFrame(self.preprocessor.transform(df), columns = self.features)   
        
        preds = model.predict(df) # predictions
        
        # Create submission dataframe with date and predictions
        sub["sleep_hours"] = preds
    
        
        self.last_submission = sub
        return sub