import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytz
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def process_sleep_data(df, freq='1min', start_date='2020-09-26 00:00:00', end_date='2023-03-17 00:00:00'):
    #exclude where valud is HKCategoryValueSleepAnalysisInBed
    df = df.drop(df[df['value'] == 'HKCategoryValueSleepAnalysisInBed'].index)
    
    # Parse dates and times
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['endDate'] = pd.to_datetime(df['endDate'])

    # Create the date range
    expanded_df = pd.DataFrame()
    expanded_df["date"] = pd.date_range(start_date, end_date, freq=freq, tz=pytz.FixedOffset(-240))

    # 1 if between startDate and endDate, 0 otherwise
    expanded_df["value"] = 0
    for _, row in df.iterrows():
        mask = (expanded_df['date'] >= row['startDate']) & (expanded_df['date'] <= row['endDate'])
        expanded_df.loc[mask, 'value'] = 1
        
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
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
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
        for i in range(5):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2**i, dilation=2**i),
                nn.BatchNorm1d(out_channels),
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
            CNNBlock(input_size, 64), 
            CNNBlock(64, 128),
            CNNBlock(128, 128)
        )

        self.dilated_blocks = nn.Sequential(
            DilatedBlock(128, 128),
            DilatedBlock(128, 128) 
        )

        self.final_conv = nn.Conv1d(128, output_size, kernel_size=1, dilation=1) 
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Add the global average pooling layer
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        cnn_out = self.cnn_blocks(x)
        dilated_out = self.dilated_blocks(cnn_out)
        out = self.final_conv(dilated_out)
        out = self.global_avg_pool(out)  # Apply global average pooling
        out = self.sigmoid(out)  # Apply sigmoid activation
        out = out.squeeze(-1)  # Remove the temporal dimension
        out = out.squeeze(1)  # Remove the channel dimension
        return out
    
class HeartRateDataset(Dataset):
    """segment_length/2 fowards and backwards from the label time. The paper used 1-hour for both"""
    def __init__(self, hr_data, labels, segment_length=120):
        self.hr_data = hr_data
        self.labels = labels
        self.segment_length = segment_length
        self.label_indices = np.arange(0, len(hr_data), len(hr_data) // len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_idx = self.label_indices[idx]
        start = max(0, label_idx - self.segment_length // 2)
        end = min(len(self.hr_data), label_idx + self.segment_length // 2)
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

    
########################################
def main_large(feature_path = "./data/xml_export/HeartRate.csv"):

    df_hr = pd.read_csv(feature_path, low_memory=False)
    df_hr = preprocess_feature_data(df_hr, "hr")

    df_sleep = pd.read_csv('./data/train_detailed.csv', low_memory=False)
    df_sleep = process_sleep_data(df_sleep)

    df = pd.merge(df_sleep, df_hr, on='date', how='outer')
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(df.head(2))

    df = df.set_index("date")
    train = df['2020-09-26':'2021-12-31']
    print(train.sleep.value_counts())

    test = df['2021-10-1':'2021-12-31'] # Last three months as test
    sub = df['2022-1-1':]
    print(train.shape, test.shape, sub.shape)

    X_train, y_train = train["hr"].to_numpy(), train["sleep"].to_numpy()
    X_test, y_test = test["hr"].to_numpy(), test["sleep"].to_numpy()
    X_sub = sub["hr"].to_numpy()
    sub_dates = sub.index
    #print("features: ", X_train.columns)

    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_prep = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    X_sub_prep = scaler.transform(X_sub.reshape(-1, 1)).reshape(X_sub.shape)
    #print("Shape: ",X_train_prep.shape)
    input_size = 1

    segment_length = 120 # 1 hour before and after 
    batch_size = 256

    train = HeartRateDataset(X_train_prep, y_train, segment_length)
    test = HeartRateDataset(X_test_prep, y_test, segment_length)
    sub = HeartRateDataset(X_sub_prep, np.zeros(len(X_sub_prep)), segment_length)

    train = DataLoader(train, batch_size=batch_size, shuffle=False)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    sub = DataLoader(sub, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA support

    # Initialize the model, loss function, and optimizer
    model = SleepStatusPredictor(input_size=input_size).to(device)

    if torch.__version__ >= '2.0':
        model = torch.compile(model)

    # Training Config
    learning_rate = 0.001
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    num_epochs = 50

    trainer = Trainer(model, device, learning_rate, criterion, optimizer)

    print("Training the model...")
    for epoch in range(num_epochs):
        train_loss = trainer.train(train)
        val_loss, val_auc = trainer.evaluate(test)
        print(f"Epoch {epoch+1}/{num_epochs}, train_bce: {train_loss:.4f}, valid_bce: {val_loss:.4f}, valid_auc: {val_auc:.4f}")
        

    probabilities = trainer.predict(sub)

    prediction_df = pd.DataFrame({"startDate": sub_dates, "sleep_prob": probabilities})
    prediction_df["startDate"] = pd.to_datetime(prediction_df["startDate"])
    prediction_df["shifted_date"] = prediction_df["startDate"] - pd.Timedelta(hours=12)
    prediction_df["date"] = prediction_df["shifted_date"].dt.date

    sleep_hours = prediction_df.groupby("date")["sleep_prob"].sum()
    sleep_hours = sleep_hours * 1 / 60
    sleep_hours = sleep_hours.reset_index()
    sleep_hours.columns = ["date", "sleep_hours_predicted"]

    print(sleep_hours.head())

    submission = pd.read_csv("./data/sample_submission.csv")

    submission["date"] = pd.to_datetime(submission["date"]).dt.date
    sleep_hours["date"] = pd.to_datetime(sleep_hours["date"]).dt.date

    # Replace the sleep hours in the submission file with the predicted values
    submission["sleep_hours"] = submission["date"].map(sleep_hours.set_index("date")["sleep_hours_predicted"])

    submission.loc[submission['sleep_hours'] < 1, 'sleep_hours'] = 6.666
    submission = submission.fillna(6.666)

    print("Mean sleep hours: ", submission["sleep_hours"].mean())

    # Save to csv
    # submission.to_csv("submission_big_sleep_model.csv", index=False)

    return submission

if __name__ == "__main__":
    main_large()