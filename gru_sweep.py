import logging
import random
import yaml
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

import wandb

from data import POG4_Dataset
from gru import TimeSeriesDataset, GRUModel, GRUTrainer, set_seed

def sweep():
    # Initialize the W&B run
    run = wandb.init()
    config = wandb.config
    
    # Feature Config
    lookback = config.lookback # Lookback window size
    batch_size = config.batch_size

    train = TimeSeriesDataset(df_train, lookback, output_size)
    test = TimeSeriesDataset(df_test, lookback, output_size)

    train = DataLoader(train, batch_size=batch_size, shuffle=False)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    # Model Config
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    learning_rate = config.learning_rate
    dropout_rate = config.dropout_rate
    activation_function = config.activation_function
    
    model = GRUModel(device, input_size, hidden_size, num_layers, output_size, dropout_rate, activation_function).to(device)
    
    # Training Config
    criterion = config.criterion
    optimizer = config.optimizer
    num_epochs = config.num_epochs  
    
    trainer = GRUTrainer(model, device, learning_rate, criterion, optimizer, target_scaler)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train(train)
        val_loss = trainer.evaluate(test)
        print(f"Epoch {epoch+1}/{num_epochs}, train_{criterion}: {train_loss:.4f}, valid_rmse: {val_loss:.4f}")
        wandb.log({"val_rmse": val_loss}) 

    run.finish()
    
if __name__ == "__main__":
    set_seed(42)
    
    # Load the dataset
    data = POG4_Dataset()
    data.train_test_split()

    imputer = SimpleImputer(strategy="median")
    scaler_f = RobustScaler() 
    scaler_t = RobustScaler() 

    X_train_imputed = pd.DataFrame(imputer.fit_transform(data.X_train), columns=data.X.columns) # Imputer 
    X_train_scaled = pd.DataFrame(scaler_f.fit_transform(X_train_imputed), columns=data.X.columns) # Scaler

    X_test_imputed = pd.DataFrame(imputer.transform(data.X_test), columns=data.X.columns) # Imputer
    X_test_scaled = pd.DataFrame(scaler_f.transform(X_test_imputed), columns=data.X.columns) # Scaler

    y_train_scaled = pd.DataFrame(scaler_t.fit_transform(data.y_train.values.reshape(-1, 1)), columns=["sleep_hours"])
    y_test_scaled = pd.DataFrame(scaler_t.transform(data.y_test.values.reshape(-1, 1)), columns=["sleep_hours"])

    target_scaler = scaler_t
    
    # Create train and test sets
    df_train = pd.concat([X_train_scaled, y_train_scaled], axis=1).to_numpy()
    df_test = pd.concat([X_test_scaled, y_test_scaled], axis=1).to_numpy()
    
    input_size = df_train.shape[1] # Number of features (plus 1 for the target)
    output_size = 1 # Number of targets
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support
    
    # Load the sweep configuration from the YAML file
    with open("gru_sweep_config.yml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project="pog4_gru")
    #sweep_id = "9mez8jf4"
    wandb.agent(sweep_id, project="pog4_gru", function=sweep)
    
    api = wandb.Api()
    runs = api.runs("sgobat/pog4_gru")

    best_run = min(runs, key=lambda run: run.summary.get('val_rmse', float('inf')))

    # Save the best parameters to a JSON file
    best_params = best_run.config
    with open("gru_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best run: {best_run.id}")
    print(f"Best parameters: {best_params}")