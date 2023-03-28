import os
import logging
import random
import yaml
import json

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

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
    
    xgb_params = {
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "n_estimators": config.n_estimators,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "objective": "reg:squarederror",
        "seed": 42
    }

    model = xgb.XGBRegressor(**xgb_params, gpu_id=0, tree_method="gpu_hist", random_state=seed)
    
    # Set up the cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Scaler
    if config.scaler == "minmax":
        scaler = MinMaxScaler()
    elif config.scaler == "standard":
        scaler = StandardScaler()
    elif config.scaler == "robust":
        scaler = RobustScaler()
    elif config.scaler == "none":
        scaler = None

    pipeline = Pipeline(steps=[("scaler", scaler), ("model", model)])

    # Perform cross-validation and calculate metrics
    cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # Fit the model to the entire dataset
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_
 
    # Log feature importances to wandb as a dictionary
    importances_dict = {f'feature/{feature_name}': importance for feature_name, importance in zip(X.columns, feature_importances)}
    wandb.log(importances_dict)

    # Log the metrics to W&B
    wandb.log({"RMSE": avg_rmse, "CV_scores": cv_scores.tolist()})

    run.finish()
    
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    # Load the dataset
    data = POG4_Dataset()
    #data.create_lags()
    data.train_test_split()
    data.preprocess_data()

    # Using cross-validation so concat the train and test sets
    X = pd.concat([data.X_train, data.X_test], axis = 0)
    y = pd.concat([data.y_train, data.y_test], axis = 0)
    
    # Load the sweep configuration from the YAML file
    with open("xgb_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_xgb")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_xgb")

    best_run = min(runs, key=lambda run: run.summary.get('RMSE', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("xgb_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")