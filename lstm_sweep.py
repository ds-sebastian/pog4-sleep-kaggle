import logging
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from data import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_sequences(data, lookback, with_y=True):
    X = []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])

    X = np.array(X)

    if with_y:
        y = data[lookback:, 0]
        return X, np.array(y)
    else:
        return X


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob, activation_function='relu'):
        super(TimeSeriesModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.activation_function = activation_function

    def forward(self, x):
        _, hn = self.gru(x)  # Removed the extra unpacking variable
        out = hn[-1]
        out = self.dropout(out)
        out = self.fc(out)


        if self.activation_function == 'relu':
            out = torch.relu(out)
        elif self.activation_function == 'tanh':
            out = torch.tanh(out)
        elif self.activation_function == 'sigmoid':
            out = torch.sigmoid(out)

        return out

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float64).float()
        self.y = torch.as_tensor(y, dtype=torch.float64).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].squeeze()  # squeeze the target variable to make it 1D

class TimeSeriesPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        super().__init__()
        self.X = torch.as_tensor(X, dtype=torch.float64).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def train_model(model, train_loader, criterion, optimizer, device, wandb_log = False):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets are 2D tensors

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    if wandb_log:
        wandb.log({"Train Loss": epoch_loss})
    return epoch_loss


def evaluate_model(model, valid_loader, criterion, device, wandb_log = False):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets are 2D tensors

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    if wandb_log:
        wandb.log({"Validation Loss": epoch_loss})
    return epoch_loss


def predict_gru(model, data, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        # Convert the NumPy array to a PyTorch tensor and move it to the specified device
        inputs = torch.tensor(data, dtype=torch.float32).to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())

    return np.array(predictions)

def calc_rmse(y_test, y_pred):
    mse = torch.nn.functional.mse_loss(y_test, y_pred)
    rmse = torch.sqrt(mse)
    return rmse.item()

def run_sweep():
    wandb.init()
    config = wandb.config
    
    # create input sequences and corresponding target variables
    X, y = create_sequences(data, config.lookback)
    
    logger.info("Train Test Split")
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    logger.info("Preprocessing data")
    X_train = f_transformer.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = f_transformer.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    y_train = t_transformer.fit_transform(y_train.reshape(-1, 1))
    y_test = t_transformer.transform(y_test.reshape(-1, 1))

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    config.output_size = 1

    # set the device to use for training and evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the model
    model = TimeSeriesModel(input_size=X_train.shape[-1], hidden_size=config.hidden_size, output_size=1, num_layers=config.num_layers, dropout_prob=config.dropout_prob, activation_function=config.activation_function).to(device)
    
    #check if pytorch version is 2.0 or higher
    if torch.__version__ >= '2.0':
        try: 
            model = torch.compile(model)
        except:
            pass

    # define the loss function and optimizer
    
    if config.loss_function == 'mse':
        criterion  = nn.MSELoss()
    elif config.loss_function == 'mae':
        criterion = nn.L1Loss()
    elif config.loss_function == 'huber':
        criterion = nn.SmoothL1Loss()
    
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # move the training and testing data to the device
    X_train, y_train = torch.Tensor(X_train).float(), torch.Tensor(y_train).float()
    X_test, y_test = torch.Tensor(X_test).float(), torch.Tensor(y_test).float()


    # Train and evaluate the model
    epochs = 500
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        wandb.log({"Epoch": epoch + 1})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset")
    df, f_transformer, t_transformer, cols = create_data("./data/train.csv", type="train")
    df = df.drop(columns=["date"], axis=1)
    data = df.values
    
    run_sweep()
    