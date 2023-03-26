
import itertools

import pandas as pd
import numpy as np
import json

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from data import POG4_Dataset

def main():
    # Load the dataset
    data = POG4_Dataset()

    df_p = data.train.copy()
    df_p = df_p.fillna(method="ffill").fillna(0) #Ok for Time Series
    df_p = df_p.rename(columns={'date': 'ds', 'sleep_hours': 'y'})
    df_p.head()

    m = Prophet()

    # Add the additional regressors
    for col in [col for col in df_p.columns if col not in ["ds", "y"]]:
        m.add_regressor(col)

    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 3, 6, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 3, 6, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here


    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(df_p)  # Fit model with given params
        df_cv = cross_validation(m, initial="1095 days", period="91 days", horizon="365 days", parallel="processes")
        df_perf = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_perf["rmse"].values[0])


    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results["rmse"] = rmses
    
    best_rmse = np.argmin(rmses)
    print("Best RMSE: ", rmses[best_rmse])
    best_params = all_params[best_rmse]


    with open("prophet_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best Parameters for Prophet: ", best_params)

if __name__ == "__main__":
    main()














