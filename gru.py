import random

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, output_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.lookback - self.output_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.output_size, -1] # ytrain in last column
        return x, y

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5, activation_function='relu'):
        super(GRUModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation_function = activation_function
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        
        if self.activation_function == 'relu':
            out = torch.relu(out)
        elif self.activation_function == 'tanh':
            out = torch.tanh(out)
        elif self.activation_function == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation_function == 'linear':
            pass
    
        return out

# GRU trainer
class GRUTrainer:
    def __init__(self, model, device, learning_rate, criterion = 'huber', optimizer = 'adam', target_scaler = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        self.target_scaler = target_scaler
        
        if criterion == 'mse':
            self.criterion  = nn.MSELoss()
        elif criterion == 'mae':
            self.criterion = nn.L1Loss()
        elif criterion == 'huber':
            self.criterion = nn.SmoothL1Loss()
            
        if optimizer == 'adam':
            self.optimizer = Adam(model.parameters(), lr=self.learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(model.parameters(), lr=self.learning_rate)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(model.parameters(), lr=self.learning_rate)

    def train(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)
    
    def _calculate_rmse(self, outputs, targets):
        
        if self.target_scaler is not None:
            outputs_inv = self.target_scaler.inverse_transform(outputs.cpu().numpy())
            targets_inv = self.target_scaler.inverse_transform(targets.cpu().numpy())

            outputs = torch.tensor(outputs_inv, device=self.device, dtype=torch.float)
            targets = torch.tensor(targets_inv, device=self.device, dtype=torch.float)

        mse_loss = F.mse_loss(outputs, targets)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss.item()

    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                rmse_loss = self._calculate_rmse(outputs, targets)

                running_loss += rmse_loss

        return running_loss / len(dataloader)
    
    def predict(self, x_input, lookback, output_size, num_predictions):
        self.model.eval()

        # Create a tensor to store the final predictions
        predictions = torch.zeros(num_predictions, output_size).to(self.device)

        # Initialize the input tensor with zeros
        input_tensor = torch.zeros(lookback, x_input.shape[-1] + 1).to(self.device)

        with torch.no_grad():
            for i in range(num_predictions):
                # Update the input tensor with x_input values
                if i < lookback:
                    # If we are still within the initial lookback period,
                    # only update the bottom portion of the input tensor with the x_input values
                    input_tensor[-(i + 1):, :-1] = x_input[:i + 1]
                else:
                    # After the initial lookback period, update the entire input tensor with the latest x_input values
                    input_tensor[:, :-1] = x_input[i - lookback + 1:i + 1]

                # Update the input tensor with past predictions
                # If i > 0 (i.e., not the first iteration), use the previous prediction;
                # otherwise, use 0 as the initial prediction
                input_tensor[-1, -1] = predictions[i - 1] if i > 0 else 0

                # Make a prediction using the model
                output = self.model(input_tensor.unsqueeze(0))

                # Store the prediction in the predictions tensor
                predictions[i] = output.squeeze()

        return predictions.cpu().numpy()

if __name__ == "__main__":
    # TEST ON REAL DATA
    from data import POG4_Dataset
    import pandas as pd
    
    data = POG4_Dataset()
    data.train_test_split()
    data.preprocess_data()
    
    # Feature Config
    lookback = 7 # Lookback window size
    input_size = data.train.shape[1]+1 # Number of features (plus 1 for the target)
    output_size = 1 # Number of targets
    batch_size = 32
    
    train = pd.concat([data.X_train, data.y_train], axis=1).to_numpy()
    test = pd.concat([data.X_test, data.y_test], axis=1).to_numpy()
    
    input_size = train.shape[1]
    
    train = TimeSeriesDataset(train, lookback, output_size)
    test = TimeSeriesDataset(test, lookback, output_size)
    
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    # Model Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    dropout_rate = 0.5
    activation_function = 'relu'
    
    model = GRUModel(device, input_size, hidden_size, num_layers, output_size, dropout_rate, activation_function).to(device)
    
    # Training Config
    criterion = 'huber'
    optimizer = 'adam'
    num_epochs = 50
    
    trainer = GRUTrainer(model, device, learning_rate, criterion, optimizer)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train(train)
        val_loss = trainer.evaluate(test)
        print(f"Epoch {epoch+1}/{num_epochs}, train_{criterion}: {train_loss:.4f}, valid_rmse: {val_loss:.4f}")
    
    # Predictions
    sub = pd.read_csv( "./data/test.csv")
    sub["date"] = pd.to_datetime(sub["date"]).dt.date
    sub = sub.merge(data.xml_data, on="date", how="left")
    sub = data._feature_engineering(sub)
    sub = sub[data.columns] # Keep only columns that are in the train data
    sub = sub.drop(columns=["date", "sleep_hours"], errors = 'ignore') # Drop date column from df
    
    
    print("x train cols ", len(data.X_train.columns))
    assert len(data.X_train.columns) == len(sub.columns), "Columns in test data do not match columns in train data"
    
    if data.preprocessor is not None:
        sub = pd.DataFrame(data.preprocessor.transform(sub), columns = data.features)   
    # sub now has the same columns as the x part of the train data
    
    sub = torch.from_numpy(sub.to_numpy()).float().to(device)
    
    predictions = trainer.predict(sub, lookback, output_size, input_size)
    print("Predictions:", predictions)
