import logging
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

import wandb
from helper import *

def run_sweep():
    wandb.init(project="pog4")
    config = wandb.config

    # Configure the logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset")
    X, y, f_transformer, t_transformer, cols = create_data("./data/train.csv", type="train")

    logger.info("Creating TimeSeriesSplit")
    tscv = TimeSeriesSplit(n_splits=5)

    logger.info("Setting XGBoost parameters")
    xgb_params = {
        "learning_rate": config.learning_rate,
        "max_depth": int(config.max_depth),
        "n_estimators": int(config.n_estimators),
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "objective": "reg:squarederror",
        "seed": 42
    }

    rmse_scores = []
    for i, (train_index, test_index) in enumerate(tqdm(tscv.split(X), total=tscv.get_n_splits()), start=1):
        logger.info(f"=== Iteration {i} ===")
        logger.info("Splitting data")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        logger.info("Preprocessing splits")
        X_train = f_transformer.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = f_transformer.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=cols)
        y_train = t_transformer.fit_transform(y_train.values.reshape(-1, 1))
        y_test = t_transformer.transform(y_test.values.reshape(-1, 1))

        logger.info("Fitting model on the full feature set")
        model = xgb.XGBRegressor(**xgb_params, tree_method="gpu_hist", gpu_id=0)
        model.fit(X_train, y_train)

        logger.info("Calculating feature importances")
        feature_importances = model.feature_importances_

        logger.info("Selecting top N features randomly")
        n_features = int(config.top_n_features)
        top_n_features = sorted(
            range(len(feature_importances)),
            key=lambda i: feature_importances[i],
            reverse=True,
        )[:n_features]

        logger.info("Randomly selecting N features")
        random_top_n_features = random.sample(top_n_features, n_features)

        logger.info(f"Using only the selected features: {random_top_n_features}")
        X_train_selected = X_train.iloc[:, random_top_n_features]
        X_test_selected = X_test.iloc[:, random_top_n_features]

        logger.info("Fitting model on the selected features")
        model_selected = xgb.XGBRegressor(**xgb_params, tree_method="gpu_hist", gpu_id=0)
        model_selected.fit(X_train_selected, y_train)

        logger.info("Predicting and calculating RMSE")
        y_pred = model_selected.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)

    logger.info("Calculating mean RMSE")
    mean_rmse = np.mean(rmse_scores)
    wandb.log({"rmse": mean_rmse})
    logger.info(f"Mean RMSE: {mean_rmse}")

if __name__ == "__main__":
    run_sweep()