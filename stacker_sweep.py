import os
import logging
import random
import yaml
import json

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


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
    
    #KNN
    knn_params = {
        "n_neighbors": config.knn_n_neighbors,
        "weights": config.knn_weights,
        "p": config.knn_p
    }

    knn_model = KNeighborsRegressor(**knn_params)
    
    # Random Forest
    rf_params = {
        "n_estimators": config.rf_n_estimators,
        "max_depth": config.rf_max_depth,
        "min_samples_split": config.rf_min_samples_split,
        "min_samples_leaf": config.rf_min_samples_leaf,
    }

    rf_model = RandomForestRegressor(**rf_params, random_state=seed, n_jobs=-1)
    
    # Extra Trees
    et_params = {
        "n_estimators": config.et_n_estimators,
        "max_depth": config.et_max_depth,
        "min_samples_split": config.et_min_samples_split,
        "min_samples_leaf": config.et_min_samples_leaf,
    }

    et_model = ExtraTreesRegressor(**et_params, random_state=seed, n_jobs=-1)
    
    # Model stacking
    stack_params = {
        "estimators": [('knn', knn_model), ('rf', rf_model), ('et', et_model)],
        "final_estimator": Ridge(alpha=config.stack_ridge_alpha),
    }
    model = StackingRegressor(**stack_params, n_jobs=-1)
    
    # Set up the cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

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
    cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    rmse_scores = np.sqrt(-cv_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # Log the metrics to W&B
    wandb.log({"RMSE": avg_rmse})

    run.finish()
    
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    # Load the dataset
    data = POG4_Dataset()

    # Using cross-validation so concat the train and test sets
    X = data.X
    y = data.y
    
    # Load the sweep configuration from the YAML file
    with open("stacker_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_stacker")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_stacker")

    best_run = min(runs, key=lambda run: run.summary.get('RMSE', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("stacker_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")