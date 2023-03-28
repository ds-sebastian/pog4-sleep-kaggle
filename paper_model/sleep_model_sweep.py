
import yaml
import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

import wandb

from lstm import TimeSeriesDataset, LSTMModel, LSTMTrainer, set_seed

def preprocess_data(input_data_files, sleep_data_file, start_date='2020-09-25', end_date='2022-01-01', resample_minutes=15):
    sleep = pd.read_csv(sleep_data_file, low_memory=False)[["startDate", "endDate", "value"]]
    sleep['value'] = sleep['value'].replace({"HKCategoryValueSleepAnalysisInBed": 0.5, "HKCategoryValueSleepAnalysisAsleepUnspecified" : 1})
    sleep.rename(columns={'value': 'sleep_prob'}, inplace=True)
    sleep['startDate'] = pd.to_datetime(sleep['startDate'])
    sleep['endDate'] = pd.to_datetime(sleep['endDate'])
    sleep = sleep[(sleep['startDate'] > start_date) & (sleep['endDate'] < end_date)]

    merged_data = None
    for input_file, suffix in input_data_files:
        input_data = pd.read_csv(input_file, low_memory=False)[["startDate", "endDate", "value"]]
        input_data['startDate'] = pd.to_datetime(input_data['startDate'])
        input_data = input_data[(input_data['startDate'] > start_date) & (input_data['endDate'] < end_date)]
        input_data.set_index('startDate', inplace=True)
        input_data = input_data.resample(f'{resample_minutes}T').mean(numeric_only=True).ffill().reset_index()
        input_data.rename(columns={'value': suffix}, inplace=True)

        if merged_data is None:
            merged_data = input_data
        else:
            merged_data = pd.merge(merged_data, input_data, on='startDate', how='outer')

    merged_data.set_index('startDate', inplace=True)
    merged_data = merged_data.sort_index()
    sleep = sleep.sort_values('startDate').reset_index(drop=True)
    sleep_intervals = [(row['startDate'], row['endDate']) for _, row in sleep.iterrows()]
    sleep_prob_arr = np.zeros(len(merged_data), dtype=float)

    for idx, row in sleep.iterrows():
        mask = (merged_data.index >= row['startDate']) & (merged_data.index < row['endDate'])
        sleep_prob_arr[mask] = row['sleep_prob']

    merged_data['sleep_prob'] = sleep_prob_arr
    
    merged_data = merged_data.reset_index()
    merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    
    return merged_data

def sweep():
    # Initialize the W&B run
    run = wandb.init()
    config = run.config
    
    # Feature Config
    lookback = config.lookback
    output_size = 1 # Number of targets
    batch_size = config.batch_size

    train = pd.concat([X_train_prep, y_train], axis=1)
    test = pd.concat([X_test_prep, y_test], axis=1)
    
    input_size = train.shape[1]
    
    train = TimeSeriesDataset(train.to_numpy(), lookback, output_size)
    test = TimeSeriesDataset(test.to_numpy(), lookback, output_size)
    
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True)

    # Model Config
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    learning_rate = config.learning_rate
    dropout_rate = config.dropout_rate
    activation_function = config.activation_function

    model = LSTMModel(device, input_size, hidden_size, num_layers, output_size, dropout_rate, activation_function).to(device)

    # Training Config
    criterion = config.criterion
    optimizer = config.optimizer
    num_epochs = config.num_epochs

    trainer = LSTMTrainer(model, device, learning_rate, criterion, optimizer, target_scaler = None)

    for epoch in range(num_epochs):
        train_loss = trainer.train(train)
        val_loss = trainer.evaluate(test)
        print(f"Epoch {epoch+1}/{num_epochs}, train_{criterion}: {train_loss:.4f}, valid_rmse: {val_loss:.4f}")
        wandb.log({"val_rmse": val_loss, "train_loss": train_loss})

if __name__ == "__main__":
    set_seed(42)
    
    # Features to add: basal energy burned, flights climbed, step count, distance walk/run
    input_data_files = [
        ('./data/xml_export/HeartRate.csv', 'hr'), 
        ('./data/xml_export/HeartRateVariabilitySDNN.csv', 'hrv'),
        ('./data/xml_export/BasalEnergyBurned.csv', 'basal_energy'),
        ('./data/xml_export/FlightsClimbed.csv', 'flights_climbed'),
        ('./data/xml_export/StepCount.csv', 'step_count'),
        ('./data/xml_export/DistanceWalkingRunning.csv', 'distance')
                        ]

    data = preprocess_data(input_data_files, 'data/train_detailed.csv')
    data = data.drop(columns = ['startDate'], errors = 'ignore')

    y = data['sleep_prob']
    X = data.drop(columns = ['sleep_prob'])

    split = 0.8
    split_idx = int(len(X) * split)
    X_train, X_test = X[:split_idx].reset_index(drop=True), X[split_idx:].reset_index(drop=True)
    y_train, y_test = y[:split_idx].reset_index(drop=True), y[split_idx:].reset_index(drop=True)

    # Scale data
    scaler = StandardScaler()

    X_train_prep = scaler.fit_transform(X_train)
    X_train_prep = pd.DataFrame(X_train_prep, columns=X_train.columns)

    X_test_prep = scaler.transform(X_test)
    X_test_prep = pd.DataFrame(X_test_prep, columns=X_test.columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support
    
    # Load the sweep configuration from the YAML file
    with open("sleep_lstm_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_sleep_lstm")
    wandb.agent(sweep_id, function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_sleep_lstm")

    best_run = min(runs, key=lambda run: run.summary.get('val_rmse', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("sleep_lstm_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")