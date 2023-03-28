import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def process_sleep_data(df, freq='1min', start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    #exclude where valud is HKCategoryValueSleepAnalysisInBed
    df = df.drop(df[df['value'] != 'HKCategoryValueSleepAnalysisInBed'].index)
    
    # Parse dates and times
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['adjusted_startDate'] = df['startDate'] - pd.to_timedelta('12:00:00') # Subtract 12 hours from startDate

    # Group by date and find min startDate and max endDate
    df = df.groupby(df['adjusted_startDate'].dt.date).agg(startDate=('startDate', 'min'),endDate=('endDate', 'max')).reset_index(drop=True)
    df["value"] = 1 
    
    date_range = pd.date_range(start_date, end_date, freq=freq, tz = pytz.FixedOffset(-240))
    expanded_df = pd.DataFrame(date_range, columns=['date'])
    expanded_df['value'] = 0 # Start with 0 and replace with 1s if in interval

    for _, row in df.iterrows():
        mask = (expanded_df['date'] >= row['startDate']) & (expanded_df['date'] <= row['endDate'])
        expanded_df.loc[mask, 'value'] = row['value']
        
    expanded_df = expanded_df.rename(columns={'value': 'sleep'})

    return expanded_df

def preprocess_feature_data(df, col_name, time_interval='1min', start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    df = df[(df['startDate'] >= start_date) & (df['startDate'] <= end_date)]
    
    df = pd.melt(df, id_vars=['value'], value_vars=['startDate', 'endDate'], value_name='date')
    df = df.groupby('date', as_index=False).mean(numeric_only=True)
    df = df.sort_values(by='date')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.resample(time_interval).mean()
    
    df = df.interpolate().rolling(2).mean()
    df = df.fillna(method="bfill")
    
    df = df.reset_index()
    df = df.rename(columns={'date': 'date', 'value': col_name})
    
    return df

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
        for i in range(1): # Lowered from 4
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2**i, dilation=2**i),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class SleepStatusPredictor(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SleepStatusPredictor, self).__init__()

        self.cnn_blocks = nn.Sequential(
            CNNBlock(input_size, 128), # Lowered from 64, 128
            #CNNBlock(64, 128), # Lowered from 64, 128
            #CNNBlock(128, 128) # Lowered from 64, 128
        )

        self.dilated_blocks = nn.Sequential(
            DilatedBlock(128, 128), # lowered from 128
            #DilatedBlock(128, 128) # lowered from 128
        )

        self.final_conv = nn.Conv1d(128, output_size, kernel_size=1, dilation=1) #lowered from 128
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Add the global average pooling layer
        
    def forward(self, x):
        cnn_out = self.cnn_blocks(x)
        dilated_out = self.dilated_blocks(cnn_out)
        out = self.final_conv(dilated_out)
        out = self.global_avg_pool(out)  # Apply global average pooling
        out = out.squeeze(-1)  # Remove the temporal dimension
        out = out.squeeze(1)  # Remove the channel dimension
        return out
    
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
        segment = self.hr_data[start:end].reshape(1, -1)
        label = self.labels[idx]

        # Pad the segment with zeros if it's shorter than segment_length
        if segment.shape[1] < self.segment_length:
            segment = np.pad(segment, ((0, 0), (0, self.segment_length - segment.shape[1])), mode='constant')

        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)



    

class Trainer:
    def __init__(self, model, device, learning_rate, criterion, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.scaler = GradScaler()

    def train(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            
            # Use autocast to enable mixed precision
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            # Scale the loss and backpropagate
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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

                # Use autocast to enable mixed precision
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Store targets and outputs for AUC calculation
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        auc = roc_auc_score(all_targets, all_outputs)

        return avg_loss, auc
    
########################################
df_hr = pd.read_csv('../data/xml_export/HeartRate.csv', low_memory=False)
df_hr = preprocess_feature_data(df_hr, "hr")
# print(df_hr.head(2))
# print(df_hr.date.dt.tz)

df_sleep = pd.read_csv('../data/train_detailed.csv', low_memory=False)
df_sleep = process_sleep_data(df_sleep)
# print(df_sleep.head(2))
# print(df_sleep.date.dt.tz)

df = pd.merge(df_sleep, df_hr, on='date', how='outer')
df = df.fillna(method='ffill').fillna(method='bfill')
#df = create_lags(df, 60, "hr")
print(df.head(2))

df = df.set_index("date")
train = df[:'2021-9-30']
test = df['2021-10-1':'2021-12-31'] # Last three months as test
sub = df['2022-1-1':]
print(train.shape, test.shape, sub.shape)

X_train, y_train = train["hr"].to_numpy(), train["sleep"].to_numpy()
X_test, y_test = test["hr"].to_numpy(), test["sleep"].to_numpy()

#print("features: ", X_train.columns)

scaler = MinMaxScaler()
X_train_prep = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_prep = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

#print("Shape: ",X_train_prep.shape)
input_size = 1

segment_length = 60
batch_size = 128

train = HeartRateDataset(X_train_prep, y_train, segment_length)
test = HeartRateDataset(X_test_prep, y_test, segment_length)

train = DataLoader(train, batch_size=batch_size, shuffle=False)
test = DataLoader(test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support

# Initialize the model, loss function, and optimizer
model = SleepStatusPredictor(input_size=input_size).to(device)

if torch.__version__ >= '2.0':
    model = torch.compile(model)

# Training Config
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.25)
num_epochs = 50

trainer = Trainer(model, device, learning_rate, criterion, optimizer)

print("Training the model...")
for epoch in range(num_epochs):
    train_loss = trainer.train(train)
    val_loss, val_auc = trainer.evaluate(test)
    print(f"Epoch {epoch+1}/{num_epochs}, train_bce: {train_loss:.4f}, valid_bce: {val_loss:.4f}, valid_auc: {val_auc:.4f}")

