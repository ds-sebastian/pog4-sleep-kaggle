import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Read the data
df = pd.read_csv('../data/xml_export/HeartRate.csv', low_memory=False)[['startDate', 'value']]
#rhr = pd.read_csv('../data/xml_export/RestingHeartRate.csv', low_memory=False)
#df = pd.concat([hr, rhr], axis = 0, ignore_index = True)

# Rename column value to heartrate_value
df = df.rename(columns={'value': 'heartrate_value'})


# Assuming your DataFrame is named 'df'
# Step 1: Filter out anomalous values
mean_hr = df['heartrate_value'].mean()
std_hr = df['heartrate_value'].std()
threshold = 5 * std_hr
df = df[(df['heartrate_value'] >= mean_hr - threshold) & (df['heartrate_value'] <= mean_hr + threshold)]

# Step 2: Normalize the heart rate time-series
df['normalized_hr'] = (df['heartrate_value'] - df['heartrate_value'].mean()) / df['heartrate_value'].std()

# Step 3: Resample the time-series using linear interpolation to a 2 Hz sampling rate (0.5-second intervals)
df["startDate"] = pd.to_datetime(df["startDate"])

df = df.set_index('startDate')
df = df.groupby('startDate').mean()

resampled_df = df.resample('5T').interpolate(method='linear')
print("len: ", len(resampled_df))

# Assuming 'resampled_df' is your DataFrame after resampling and interpolating
target_length = 18000
current_length = len(resampled_df)
pad_length = max(target_length - current_length, 0)

if pad_length > 0:
    padding = pd.DataFrame({'normalized_hr': [0] * pad_length}, index=pd.date_range(start=resampled_df.index[-1] + pd.Timedelta('5T'), periods=pad_length, freq='5T'))
    resampled_df = pd.concat([resampled_df, padding])

print(resampled_df.reset_index())


"""In the paper the targets were sleep stages. I want to simplify things and only have awake vs sleep (binary).

A sample of the sleep data is below:
creationDate	startDate	endDate	value
4/19/2015 9:42	2/20/2015 1:45	2/20/2015 8:09	sleeping
4/19/2015 9:42	2/20/2015 1:39	2/20/2015 8:46	awake
4/19/2015 9:42	2/21/2015 1:59	2/21/2015 9:34	sleeping
4/19/2015 9:42	2/21/2015 1:52	2/21/2015 9:57	sleeping
4/19/2015 9:42	2/22/2015 2:50	2/22/2015 9:11	awake
4/19/2015 9:42	2/22/2015 2:36	2/22/2015 9:36	sleeping
4/19/2015 9:42	2/23/2015 1:19	2/23/2015 7:49	awake
4/19/2015 9:42	2/23/2015 1:12	2/23/2015 8:30	awake

This needs to be processed into a target  variable for the model to use. Will it have to match the time index with the heart rate data?"""