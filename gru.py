import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return hidden

# GRU trainer
class GRUTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

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

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

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

    
# TEST ON REAL DATA
from data import POG4_Dataset
import pandas as pd

data = POG4_Dataset()
data.train_test_split()
data.preprocess_data()

# Set CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lookback and feature configuration
lookback = 10

output_size = 1

train = pd.concat([data.X_train, data.y_train], axis=1).to_numpy()
test = pd.concat([data.X_test, data.y_test], axis=1).to_numpy()

num_features = train.shape[1]

train = TimeSeriesDataset(train, lookback, output_size)
test = TimeSeriesDataset(test, lookback, output_size)

# Training and evaluation
hidden_size = 64
num_layers = 2
batch_size = 32
num_epochs = 20
learning_rate = 0.001

train = DataLoader(train, batch_size=batch_size, shuffle=True)
test = DataLoader(test, batch_size=batch_size, shuffle=True)

model = GRUModel(num_features, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trainer = GRUTrainer(model, criterion, optimizer, device)

for epoch in range(num_epochs):
    train_loss = trainer.train(train)
    val_loss = trainer.evaluate(test)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

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

predictions = trainer.predict(sub, lookback, 1, num_predictions)
print("Predictions:", predictions)
