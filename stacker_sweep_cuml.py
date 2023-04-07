import os
import logging
import random
import yaml
import json

import cupy as np
import cudf as pd

from cuml.neighbors import KNeighborsRegressor
from cuml.svm import SVR
from cuml.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from cuml.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from cuml.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, SimpleImputer
from sklearn.pipeline import Pipeline
from cuml.linear_model import Ridge

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

    rf_model = RandomForestRegressor(**rf_params, random_state=seed)
    
    # SVM
    svm_params = {
        "kernel": config.svm_kernel,
        "C": config.svm_C,
        "epsilon": config.svm_epsilon,
        "gamma": config.svm_gamma
    }

    svm_model = SVR(**svm_params)
    
    # Extra Trees
    # et_params = {
    #     "n_estimators": config.et_n_estimators,
    #     "max_depth": config.et_max_depth,
    #     "min_samples_split": config.et_min_samples_split,
    #     "min_samples_leaf": config.et_min_samples_leaf,
    # }

    # et_model = ExtraTreesRegressor(**et_params, random_state=seed)
    
    # Model stacking
    stack_params = {
        "estimators": [('knn', knn_model), ('rf', rf_model), ('svm', svm_model)],
        "final_estimator": Ridge(alpha=config.stack_ridge_alpha),
    }
    model = StackingRegressor(**stack_params)
    
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
    cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # Log the metrics to W&B
    wandb.log({"RMSE": float(avg_rmse)})

    run.finish()
    
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    # Load the dataset
    data = POG4_Dataset()
    
    X = data.X.astype(np.float32)
    y = data.y.astype(np.float32)
    
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
