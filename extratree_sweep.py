import os
import logging
import random
import yaml
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import wandb

from data import POG4_Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def sweep():
    # Initialize the W&B run
    run = wandb.init()
    config = wandb.config
    
    et_params = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_split": config.min_samples_split,
        "min_samples_leaf": config.min_samples_leaf,
    }

    model = ExtraTreesRegressor(**et_params, random_state=seed)
    
    # Set up the cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Scaler
    if config.scaler == "minmax":
        scaler = MinMaxScaler()
    elif config.scaler == "standard":
        scaler = StandardScaler()
    elif config.scaler == "robust":
        scaler = RobustScaler()
    
    # Imputer 
    if config.imputer == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif config.imputer == "median":
        imputer = SimpleImputer(strategy="median")
    elif config.imputer == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    
    pipeline = Pipeline(steps=[("imputer", imputer), ("scaler", scaler), ("model", model)])

    # Perform cross-validation and calculate metrics
    cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # Log the metrics to W&B
    wandb.log({"RMSE": avg_rmse, "CV_scores": cv_scores.tolist()})

    run.finish()
    
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    # Load the dataset
    data = POG4_Dataset()
    #data.create_lags()
    train = data.train[(data.train['date'] >= pd.to_datetime('2020-09-25').date()) & (data.train['date'] <= pd.to_datetime('2021-11-30').date())]

    # Using cross-validation so concat the train and test sets
    X = train.drop(['sleep_hours', 'date'], axis=1)
    y = train.sleep_hours.fillna(method="ffill").fillna(method="bfill").fillna(7.0)
    
    # Load the sweep configuration from the YAML file
    with open("extratree_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_et")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_et")

    best_run = min(runs, key=lambda run: run.summary.get('RMSE', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("et_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")