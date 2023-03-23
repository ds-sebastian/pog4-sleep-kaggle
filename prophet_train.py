# Prophet by META
# Testing a simple prophet model

import itertools
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

from helper import *

df, f_transformer, t_transformer, cols = create_data("./data/train.csv", type="train")
df_sub, _, _, _ = create_data("./data/test.csv", type="test")

print(df.head())
#Training data
df_p = df.copy()
df_p = df_p.rename(columns={'date': 'ds', 'sleep_hours': 'y'})
df_p = df_p.fillna(0)

#Submission Data
df_sub = df_sub.rename(columns={'date': 'ds'})
df_sub = df_sub.fillna(0)

# Instantiate the Prophet model with specified seasonality and holidays
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
    df_cv = cross_validation(m, initial='1825 days', period='91 days', horizon='365 days', parallel="processes")
    df_perf = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_perf['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

# Python
best_params = all_params[np.argmin(rmses)]
print(best_params)

# Fit Final Model
m = Prophet(**best_params)
m.fit(df_p)

sub_prophet = df_sub.copy()
forecast = m.predict(sub_prophet.rename(columns={'date': 'ds'}))

sub = pd.read_csv('./data/sample_submission.csv')
sub = sub.sort_values(by='date') # Make sure sorted by date
sub["date"] = pd.to_datetime(sub["date"]).dt.date # Format date as date

sub['sleep_hours'] = forecast['yhat'] # use forecast
sub.to_csv("./submissions/submission_prophet.csv", index=False)
print('Predicitons: ', sub.head())

# kaggle competitions submit -c kaggle-pog-series-s01e04 -f ./submissions/submission_prophet.csv -m "Trying Prophet"
# Score: __________________