import random

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np

from sklearn.metrics import roc_auc_score

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
    def __init__(self, device, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.batch_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

# GRU trainer
class Trainer:
    def __init__(self, model, device, learning_rate, criterion, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = criterion

    def train(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
                
            # Scale the loss and backpropagate
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()

        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Store targets and outputs for AUC calculation
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        auc = roc_auc_score(all_targets, all_outputs)

        return avg_loss, auc
    
    def predict(self, dataloader):
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)

                all_outputs.extend(outputs.cpu().numpy())

        return all_outputs

if __name__ == "__main__":
    # TEST ON REAL DATA
    from data import POG4_Dataset
    import pandas as pd
    
    data = POG4_Dataset()
    data.train_test_split()
    data.preprocess_data()
    
    y_train_mean = data.y_train.median()
    
    # Turn y_train and y_test into binary classification
    y_train = (data.y_train > y_train_mean).astype(int)
    y_test = (data.y_test > y_train_mean).astype(int)
    
    # Feature Config
    lookback = 7 # Lookback window size
    input_size = data.train.shape[1]+1 # Number of features (plus 1 for the target)
    output_size = 1 # Number of targets
    batch_size = 32
    
    train = pd.concat([data.X_train, y_train], axis=1).to_numpy()
    test = pd.concat([data.X_test, y_test], axis=1).to_numpy()
    
    input_size = train.shape[1]
    
    train = TimeSeriesDataset(train, lookback, output_size)
    test = TimeSeriesDataset(test, lookback, output_size)
    
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    # Model Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support
    hidden_size = 128
    num_layers = 1
    dropout_rate = 0.7
    
    model = GRUModel(device, input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
    
    # Training Config
    learning_rate = 0.0001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    num_epochs = 50
    
    trainer = Trainer(model, device, learning_rate, criterion, optimizer)
    
    for epoch in range(num_epochs):
        train_loss = trainer.train(train)
        val_loss, auc = trainer.evaluate(test)
        print(f"Epoch {epoch+1}/{num_epochs}, train_bce: {train_loss}, valid_bce: {val_loss}, valid_auc: {auc}")
    
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
