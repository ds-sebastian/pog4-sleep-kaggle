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


import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def process_sleep_data(df, freq='1min', start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    #exclude where valud is HKCategoryValueSleepAnalysisInBed
    df = df.drop(df[df['value'] != 'HKCategoryValueSleepAnalysisInBed'].index)
    
    # Parse dates and times
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['adjusted_startDate'] = df['startDate'] - pd.to_timedelta('12:00:00') # Subtract 12 hours from startDate

    # Group by date and find min startDate and max endDate
    df = df.groupby(df['adjusted_startDate'].dt.date).agg(startDate=('startDate', 'min'),endDate=('endDate', 'max')).reset_index(drop=True)
    df["value"] = 1 
    
    date_range = pd.date_range(start_date, end_date, freq=freq, tz = pytz.FixedOffset(-240))
    expanded_df = pd.DataFrame(date_range, columns=['date'])
    expanded_df['value'] = 0 # Start with 0 and replace with 1s if in interval

    for _, row in df.iterrows():
        mask = (expanded_df['date'] >= row['startDate']) & (expanded_df['date'] <= row['endDate'])
        expanded_df.loc[mask, 'value'] = row['value']
        
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

def sweep():
    # Initialize the W&B run
    run = wandb.init()
    config = wandb.config
    
    xgb_params = {
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "n_estimators": config.n_estimators,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "gamma": config.gamma,
        "min_child_weight": config.min_child_weight,
        "objective": "binary:logistic",
        "seed": seed
    }
    
    if config.scaler == "minmax":
        scaler = MinMaxScaler()
    elif config.scaler == "standard":
        scaler = StandardScaler()
    elif config.scaler == "robust":
        scaler = RobustScaler()
    elif config.scaler == "none":
        scaler = None

    
    model = xgb.XGBClassifier(**xgb_params, gpu_id=0, tree_method="gpu_hist", random_state=seed)
    
    pipeline = Pipeline(steps=[("scaler", scaler), ("model", model)])
    
    # Set up the cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation and calculate metrics
    cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="roc_auc")
    auc = np.mean(cv_scores)

    # Log the metrics to W&B
    wandb.log({"AUC": auc, "CV_scores": cv_scores.tolist()})

    run.finish()

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    features = [
        ('../data/xml_export/HeartRate.csv', "hr"),
        ('../data/xml_export/StepCount.csv', "steps"),
        ('../data/xml_export/DistanceWalkingRunning.csv', "distance"),
        ]
    
    df = pd.DataFrame()
    for file_path, col_name in features:
        raw = pd.read_csv(file_path, low_memory=False)
        preprocessed_df = preprocess_feature_data(raw, col_name, smoothing = 2, freq="1min", start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00')
        lagged_df = create_lags(preprocessed_df, col_name, 60)
        if df.empty:
            df = lagged_df
        else:
            df = pd.merge(df, lagged_df, on='date', how='outer')

    df_sleep = pd.read_csv('../data/train_detailed.csv', low_memory=False)
    df_sleep = process_sleep_data(df_sleep)
    
    df = pd.merge(df_sleep, df, on='date', how='outer')
    df = df.set_index("date")
    df = df.astype('float32')
    
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(df.head(2))

    
    train = df[:'2021-12-31'] 
    sub = df['2022-1-1':]

    X, y = train.drop(columns=["sleep"]), train["sleep"]

    # Load the sweep configuration from the YAML file
    with open("sleep_model_sweep.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_sleep_model")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_sleep_model")

    best_run = min(runs, key=lambda run: run.summary.get('AUC', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("sleep_model_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")