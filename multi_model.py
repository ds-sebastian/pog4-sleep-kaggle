import os
import yaml
import json
import random
import pandas as pd
import numpy as np
import pytz
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def process_sleep_data(df, freq='1min', start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    #exclude where valud is HKCategoryValueSleepAnalysisInBed
    df = df.drop(df[df['value'] == 'HKCategoryValueSleepAnalysisInBed'].index)
    
    # Parse dates and times
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])

    # Create the date range
    expanded_df = pd.DataFrame()
    expanded_df["date"] = pd.date_range(start_date, end_date, freq=freq, tz=pytz.FixedOffset(-240))

    # 1 if between startDate and endDate, 0 otherwise
    expanded_df["value"] = 0
    for _, row in df.iterrows():
        mask = (expanded_df['date'] >= row['startDate']) & (expanded_df['date'] <= row['endDate'])
        expanded_df.loc[mask, 'value'] = 1
        
    expanded_df = expanded_df.rename(columns={'value': 'sleep'})

    return expanded_df

def preprocess_feature_data(df, col_name, freq='1min', smoothing = 2, start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    df = df[(df['startDate'] >= start_date) & (df['startDate'] <= end_date)]
    
    df = pd.melt(df, id_vars=['value'], value_vars=['startDate', 'endDate'], value_name='date')
    df = df.groupby('date', as_index=False).mean(numeric_only=True)
    df = df.sort_values(by='date')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.resample(freq).mean()
    
    df = df.interpolate().rolling(smoothing).mean()
    df = df.fillna(method="bfill")
    
    df = df.reset_index()
    df = df.rename(columns={'date': 'date', 'value': col_name})
    
    return df

def create_lags(df, column_name, n_lags):
    bckwd_columns = [df[column_name].shift(i).fillna(method="bfill").fillna(method="ffill") for i in range(1, n_lags+1)]
    fwd_columns = [df[column_name].shift(-i).fillna(method="bfill").fillna(method="ffill") for i in range(1, n_lags+1)]
    bckwd_names = [f"{column_name}_bckwd_{i}" for i in range(1, n_lags+1)]
    fwd_names = [f"{column_name}_fwd_{i}" for i in range(1, n_lags+1)]
    df_lags = pd.concat(bckwd_columns + fwd_columns, axis=1, keys=bckwd_names + fwd_names)
    return pd.concat([df, df_lags], axis=1)

def train_model(X, y):
    
    with open("sleep_model_best_params.json", "r") as f:
        sleep_model_best_params = json.load(f)
    
    xgb_params = {
        "learning_rate": sleep_model_best_params["learning_rate"],
        "max_depth": sleep_model_best_params["max_depth"],
        "n_estimators": sleep_model_best_params["n_estimators"],
        "subsample": sleep_model_best_params["subsample"],
        "colsample_bytree": sleep_model_best_params["colsample_bytree"],
        "gamma": sleep_model_best_params["gamma"],
        "min_child_weight": sleep_model_best_params["min_child_weight"],
        "objective": "binary:logistic",
        "seed": 42
    }

    if sleep_model_best_params["scaler"] == "minmax":
        scaler = MinMaxScaler()
    elif sleep_model_best_params["scaler"] == "standard":
        scaler = StandardScaler()
    elif sleep_model_best_params["scaler"] == "robust":
        scaler = RobustScaler()
    elif sleep_model_best_params["scaler"] == "none":
        scaler = None

    model = xgb.XGBClassifier(**xgb_params, gpu_id=0, tree_method="gpu_hist", random_state=42)
    pipeline = Pipeline(steps=[("scaler", scaler), ("model", model)])
    pipeline.fit(X, y)
    
    return pipeline

def create_submissions(sub_X, pipeline, sub_start_dates, feature_name):
    
    predictions = pipeline.predict_proba(sub_X) #! (or proba)
    predictions = pd.DataFrame({"startDate": sub_start_dates, "sleep_prob": predictions[:, 1]})
    predictions["startDate"] = pd.to_datetime(predictions["startDate"])
    predictions["shifted_date"] = predictions["startDate"] - pd.Timedelta(hours=12)
    predictions["date"] = predictions["shifted_date"].dt.date
    
    sleep_hours = predictions.groupby("date")["sleep_prob"].sum()
    sleep_hours = sleep_hours * 1 / 60
    sleep_hours = sleep_hours.reset_index()
    sleep_hours.columns = ["date", "sleep_hours_predicted"]
    sleep_hours.loc[sleep_hours['sleep_hours_predicted'] < 1, 'sleep_hours_predicted'] = 6.666 # median
    sleep_hours = sleep_hours.fillna(6.666)
    sleep_hours = sleep_hours.iloc[1:]
    
    submission = pd.read_csv("./data/sample_submission.csv")
    
    submission["date"] = submission["date"].astype(str)
    sleep_hours["date"] = sleep_hours["date"].astype(str)
    
    submission = submission.merge(sleep_hours, on="date", how="left")
    submission = submission.drop(columns=['sleep_hours'])
    submission = submission.rename(columns={'sleep_hours_predicted': f'sleep_hours_{feature_name}'})
    
    return submission


def main():
    seed = 42
    set_seed(seed)
    
    df_sleep = pd.read_csv('./data/train_detailed.csv', low_memory=False)
    df_sleep = process_sleep_data(df_sleep)
    
    features = [
        ('./data/xml_export/HeartRate.csv', "hr"),
        ('./data/xml_export/StepCount.csv', "steps"),
        ('./data/xml_export/DistanceWalkingRunning.csv', "distance"),
        ]
    
    final = pd.DataFrame()
    for file_path, col_name in features:
        raw = pd.read_csv(file_path, low_memory=False)
        preprocessed_df = preprocess_feature_data(raw, col_name, smoothing = 2, freq="1min", start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00')
        lagged_df = create_lags(preprocessed_df, col_name, 60)

        df = pd.merge(df_sleep, lagged_df, on='date', how='outer')
        df = df.set_index("date")
        df = df.astype('float32')
        df = df.fillna(method='ffill').fillna(method='bfill')

        train = df[:'2021-12-31'] 
        sub = df['2022-1-1':]

        X, y = train.drop(columns=["sleep"]), train["sleep"]
        sub_X = sub.drop(columns=["sleep"])

        pipeline = train_model(X, y)
        submission = create_submissions(sub_X, pipeline, sub.index, col_name)

        if final.empty:
            final = submission
        else:
            final = pd.merge(final, submission, on='date', how='outer')
        
    # mean of all sleep_hours_predicted
    final["sleep_hours"] = final.iloc[:, 1:].mean(axis=1)
    
    print("mean sleep hours: ", final["sleep_hours"].mean())
    
    # drop all sleep_hours_predicted
    final = final.drop(columns=final.columns[1:-1])
    final.to_csv("submission_multi_hr_model.csv", index=False)
    print(final.head(10))
    return final

if __name__ == "__main__":
    main()