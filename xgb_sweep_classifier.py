import os
import logging
import random
import yaml
import json

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, TimeSeriesSplit
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
    
    xgb_params = {
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "n_estimators": config.n_estimators,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "objective": "binary:logistic",
        "seed": 42
    }

    model = xgb.XGBClassifier(**xgb_params, gpu_id=0, tree_method="gpu_hist", random_state=seed)
    
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
    
    # Imputer 
    if config.imputer == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif config.imputer == "median":
        imputer = SimpleImputer(strategy="median")
    elif config.imputer == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif config.imputer == "none":
        imputer = None 
    
    pipeline = Pipeline(steps=[("imputer", imputer), ("scaler", scaler), ("model", model)])

    # Perform cross-validation and calculate metrics
    cv_scores = cross_validate(pipeline, X, y, cv=tscv, scoring=["roc_auc", "accuracy", "f1", "precision", "recall"], n_jobs=-1)
    avg_auc_score = cv_scores["test_roc_auc"].mean()
    avg_accuracy_score = cv_scores["test_accuracy"].mean()
    avg_f1_score = cv_scores["test_f1"].mean()
    avg_precision_score = cv_scores["test_precision"].mean()
    avg_recall_score = cv_scores["test_recall"].mean()
    

    # # Fit the model to the entire dataset
    # model.fit(X, y)

    # # Get feature importances
    # feature_importances = model.feature_importances_
 
    # # Log feature importances to wandb as a dictionary
    # importances_dict = {f'feature/{feature_name}': importance for feature_name, importance in zip(X.columns, feature_importances)}
    # wandb.log(importances_dict)

    # Log the metrics to W&B
    wandb.log({"AUC": avg_auc_score, "Accuracy": avg_accuracy_score, "F1": avg_f1_score, "Precision": avg_precision_score, "Recall": avg_recall_score})

    run.finish()
    
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    # Load the dataset
    data = POG4_Dataset()
    #data.create_lags()
    # data.train_test_split()
    #data.preprocess_data()

    X = data.X
    y = data.y
    
    # Turn y into classification (greater or less than median)
    y = y > y.median()
    
    # Print y value counts
    print(y.value_counts())
    
    # Load the sweep configuration from the YAML file
    with open("xgb_sweep_config_classifier.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_xgb_classifier")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_xgb_classifier")

    best_run = max(runs, key=lambda run: run.summary.get('Accuracy', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("xgb_classifier_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")