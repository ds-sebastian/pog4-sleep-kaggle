import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np

def preprocess_data(input_data_files, sleep_data_file, start_date='2020-09-25', end_date='2022-01-01', resample_minutes=1):
    sleep = pd.read_csv(sleep_data_file, low_memory=False)[["startDate", "endDate", "value"]]
    sleep['value'] = sleep['value'].replace({"HKCategoryValueSleepAnalysisInBed": 0, "HKCategoryValueSleepAnalysisAsleepUnspecified" : 1})
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

class CNNBlock(nn.Module):
    """The function of these blocks is to extract local features from the input heart rate time-series data"""
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class DilatedBlock(nn.Module):
    """ These filters have progressively increasing dilation rates of 2, 4, 8, 16, and 32, 
    which are responsible for increasing the network's field of view to capture long-range features from the input."""
    def __init__(self, in_channels, out_channels):
        super(DilatedBlock, self).__init__()

        layers = []
        for i in range(4):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2**i, dilation=2**i),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class MultiFeatureDataset(Dataset):
    """Use with >1 features"""
    def __init__(self, feature_data, labels, segment_length=24):
        self.feature_data = feature_data
        self.labels = labels
        self.segment_length = segment_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = idx - self.segment_length // 2
        end = start + self.segment_length
        segment = self.feature_data[start:end].to_numpy()
        label = self.labels[idx]

        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.float32) 

class HeartRateDataset(Dataset):
    """segment_length/2 fowards and backwards from the label time. The paper used 1-hour for both"""
    def __init__(self, hr_data, labels, segment_length=24):
        self.hr_data = hr_data
        self.labels = labels
        self.segment_length = segment_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = max(0, idx - self.segment_length // 2)
        end = min(len(self.hr_data), start + self.segment_length)
        segment = self.hr_data[start:end].to_numpy().reshape(1, -1)
        label = self.labels[idx]

        # Pad the segment with zeros if it's shorter than segment_length
        if segment.shape[1] < self.segment_length:
            segment = np.pad(segment, ((0, 0), (0, self.segment_length - segment.shape[1])), mode='constant')

        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class SleepStatusPredictor(nn.Module):
    def __init__(self):
        super(SleepStatusPredictor, self).__init__()

        self.cnn_blocks = nn.Sequential(
            CNNBlock(1, 64),
            CNNBlock(64, 128),
            CNNBlock(128, 128)
        )

        self.dilated_blocks = nn.Sequential(
            DilatedBlock(128, 128),
            DilatedBlock(128, 128)
        )

        self.final_conv = nn.Conv1d(128, 1, kernel_size=1, dilation=1)

    def forward(self, x):
        cnn_out = self.cnn_blocks(x)
        dilated_out = self.dilated_blocks(cnn_out)
        out = self.final_conv(dilated_out)
        return out
    
class FullyConvolutionalDNN(nn.Module):
    def __init__(self):
        super(FullyConvolutionalDNN, self).__init__()

        # First part of the network
        self.cnn_block1 = self.create_cnn_block(1, 2)
        self.cnn_block2 = self.create_cnn_block(2, 4)
        self.cnn_block3 = self.create_cnn_block(4, 8)

        # Second part of the network
        self.dilated_block1 = self.create_dilated_block()
        self.dilated_block2 = self.create_dilated_block()

        # Final convolutional layer
        self.conv_final = nn.Conv1d(128, 4, kernel_size=1)

    def create_cnn_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def create_dilated_block(self):
        layers = []
        dilation_rates = [2, 4, 8, 16, 32]
        for rate in dilation_rates:
            layers.extend([
                nn.Conv1d(128, 128, kernel_size=7, padding=rate, dilation=rate),
                nn.LeakyReLU(),
                nn.Dropout()
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # First part of the network
        x1 = self.cnn_block1(x)
        x = x1

        x2 = self.cnn_block2(x)
        x = x + x2[:, :, ::2]

        x3 = self.cnn_block3(x)
        x = x + x3[:, :, ::2]

        x = x.view(x.size(0), 128, -1)

        # Second part of the network
        x4 = self.dilated_block1(x)
        x = x + x4

        x5 = self.dilated_block2(x)
        x = x + x5

        # Final convolutional layer
        x = self.conv_final(x)

        return x


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
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        total_predictions = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))

                running_loss += loss.item()

                # Calculate the number of correct predictions
                predicted = (outputs > 0.5).float()
                correct = (predicted == targets.unsqueeze(1)).sum().item()
                correct_predictions += correct
                total_predictions += targets.numel()

            accuracy = correct_predictions / total_predictions
            avg_loss = running_loss / len(dataloader)

        return avg_loss, accuracy

    
    
########################################
input_data_files = [
    ('../data/xml_export/HeartRate.csv', 'hr'), 
    ('../data/xml_export/HeartRateVariabilitySDNN.csv', 'hrv'),
    ('../data/xml_export/BasalEnergyBurned.csv', 'basal_energy'),
    ('../data/xml_export/FlightsClimbed.csv', 'flights_climbed'),
    ('../data/xml_export/StepCount.csv', 'step_count'),
    ('../data/xml_export/DistanceWalkingRunning.csv', 'distance')        ]

data = preprocess_data(input_data_files, '../data/train_detailed.csv')

y = data['sleep_prob']
X = data["hr"]

split = 0.8
train_size = int(split * len(X))
test_size = len(X) - train_size
X_train, X_test = X[0:train_size].reset_index(drop=True), X[train_size:len(X)].reset_index(drop=True)
y_train, y_test = y[0:train_size].reset_index(drop=True), y[train_size:len(y)].reset_index(drop=True)

segment_length = 120
batch_size = 32

train = HeartRateDataset(X_train, y_train, segment_length=segment_length)
test = HeartRateDataset(X_test, y_test, segment_length=segment_length)

train = DataLoader(train, batch_size=batch_size, shuffle=False)
test = DataLoader(test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support

# Initialize the model, loss function, and optimizer
model = SleepStatusPredictor().to(device)

# if torch.__version__ >= '2.0':
#     model = torch.compile(model)

# Training Config
learning_rate = 0.0001
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.25)
num_epochs = 50

trainer = Trainer(model, device, learning_rate, criterion, optimizer)

for epoch in range(num_epochs):
    train_loss = trainer.train(train)
    val_loss, val_acc = trainer.evaluate(test)
    print(f"Epoch {epoch+1}/{num_epochs}, train_bce: {train_loss:.4f}, valid_bce: {val_loss:.4f}, valid_acc: {val_acc:.4f}")

