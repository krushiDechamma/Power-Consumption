#!/usr/bin/env python
# coding: utf-8

# # Data 

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Ignoring warnings for cleaner output

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Paths for the datset
input_txt_file = "C:\\Users\\Madhu Shree\\OneDrive\\Desktop\\Minor\\household_power_consumption.txt"
output_csv_file = "C:\\Users\\Madhu Shree\\OneDrive\\Desktop\\Minor\\data.csv"

# Read the text file, replacing '?' with NaN
data = pd.read_csv(input_txt_file, delimiter=';', na_values='?', low_memory=False)


# Convert relevant columns to float, ignoring errors for NaN values
cols_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Save as CSV
data.to_csv(output_csv_file, index=False)

print(f"Conversion completed. CSV file saved as {output_csv_file}")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


# Check for missing values before replacing
missing_values_before_replace = data.isnull().sum()
print("Missing values before replacing:")
print(missing_values_before_replace)


# In[8]:


# Replace missing values with NaN only if there are any missing values
if missing_values_before_replace.any():
    data.replace('?', np.nan, inplace=True)

# Convert numeric columns to float (excluding the 'Date' column)
numeric_columns = data.columns[data.dtypes != 'object']
numeric_columns = numeric_columns[numeric_columns != 'Date']
data[numeric_columns] = data[numeric_columns].astype(float)

# Handle the 'Date' column separately
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)


# Fill missing values with forward fill
data.fillna(method='ffill', inplace=True)


# In[9]:


# Check for missing values after filling
missing_values_after_fill = data.isnull().sum()
print("Missing values after filling:")
print(missing_values_after_fill)


# In[10]:


# Assuming 'Date' is the column containing your date values
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Resample data into day-wise bins and sum values
data = data.resample('D').sum()
# Convert 'Date' to string
data['Date'] = data.index.astype(str)




# In[11]:


# Plot all features
fig, axes = plt.subplots(len(data.columns[1:]), 1, figsize=(18, 18), sharex=True)

for i, col in enumerate(data.columns[1:]):  # Start from the second column
    axes[i].plot(data.index, data[col])  # Use data.index instead of data['Date']
    axes[i].set_title(col, loc='right')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].tick_params(axis='y', labelleft=False)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



# In[12]:


# Additional Analysis: Weekly Patterns


import calendar


# Create a new column 'day_of_week' based on the datetime index
data['day_of_week'] = data.index.day_name()

# Plot the average power consumption for each day of the week for selected columns
selected_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
custom_titles = ['Weekly average power consumption of kitchen', 'Weekly average power consumption of laundry room', 'Weekly average power consumption of Appliances']

# Adjust figsize and subplot height
fig, axes = plt.subplots(nrows=len(selected_columns), figsize=(12, 6 * len(selected_columns)))

# Define the order of days to ensure correct ordering in the plot
days_order = list(calendar.day_name)

# Group by 'day_of_week' and calculate the mean for each selected column
for i, column in enumerate(selected_columns):
    average_power_by_day = data.groupby('day_of_week')[column].mean().reindex(days_order)

    # Plotting with the same color for all days, with effects
    axes[i].plot(average_power_by_day.index, average_power_by_day, marker='o', linestyle='-', color='blue', label='Average Consumption')

    axes[i].set_xlabel('Day of the Week')
    axes[i].set_ylabel(f'Average {column} Consumption')
    axes[i].set_title(custom_titles[i])
    axes[i].legend()

    # Add grid lines for better readability
    axes[i].grid(True, linestyle='--', alpha=0.6)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



# In[13]:


# Additional Analysis: Correlation Matrix

# dropping columns with any non-numeric values
data = data.select_dtypes(include=[np.number])

# Visualize the correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()



# In[14]:


# Heatmap for Power Consumption Features


# Plot a heatmap for power consumption features
power_consumption_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
plt.figure(figsize=(14, 8))
sns.heatmap(data[power_consumption_features].resample('D').mean().corr(), annot=True, cmap='viridis')
plt.title('Power Consumption Features Correlation Heatmap')
plt.show()




# In[15]:


# Visualize daily power consumption trends
fig = plt.figure(figsize=(18, 18))

for i, feature in enumerate(power_consumption_features):
    plt.subplot(3, 3, i+1)
    data[feature].resample('D').mean().plot()
    plt.title(f'Daily {feature} Consumption')
    plt.xlabel('Date')
    plt.ylabel('Consumption')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[16]:


# Visualizing Weekly Power Consumption Trends


# Set the figure size
plt.figure(figsize=(18, 12))

# Calculate the number of rows and columns for subplots
num_features = len(power_consumption_features)
num_rows = 3
num_cols = (num_features // num_rows) + 1

# Plotting each feature
for i, feature in enumerate(power_consumption_features):
    plt.subplot(num_rows, num_cols, i+1)
    data[feature].resample('W').sum().plot()  # Resample data to weekly and sum for each week
    plt.title(f'Weekly {feature} Consumption')
    plt.xlabel('Week')
    plt.ylabel('Consumption')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()



# In[17]:


# generating histograms and kernel density plots


# Assuming 'data' is your DataFrame and 'feature' is the specific power consumption feature
plt.figure(figsize=(12, 6))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(data['Global_active_power'], kde=True, color='blue')
plt.title('Histogram of Global Active Power')

# Kernel Density Plot
plt.subplot(1, 2, 2)
sns.kdeplot(data['Global_active_power'], color='red')
plt.title('Kernel Density Plot of Global Active Power')

plt.show()



# In[18]:


# generating scatter plots and correlation matrix
plt.figure(figsize=(12, 6))

# Scatter Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x='Global_active_power', y='Global_reactive_power', data=data, color='green')
plt.title('Scatter Plot: Global Active Power vs. Global Reactive Power')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


# Extract relevant columns
features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
target = 'Global_active_power'

# Create a new DataFrame with selected columns
selected_data = data[[target] + features]


# In[20]:


def create_lstm_dataset(dataset, target_col, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset.iloc[i:(i + look_back)][features + [target_col]].values
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back][target_col])
    return np.array(dataX), np.array(dataY)


# In[21]:


# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(selected_data)


# In[22]:


# Set look-back window (adjust as needed)
look_back = 7

# Create LSTM datasets
X, y = create_lstm_dataset(pd.DataFrame(scaled_data, columns=[target] + features), target_col=target, look_back=look_back)


# In[23]:


# Split the dataset into training and testing sets
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Print the percentage of train and test data
print(f"Percentage of Train Data: {100 * train_size / len(X):.2f}%")
print(f"Percentage of Test Data: {100 * (1 - train_size / len(X)):.2f}%")


# In[24]:


# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))


# In[25]:


# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])


# In[26]:


# Display the model summary
model.summary()


# In[27]:


# Learning Rate Schedulers
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-8), metrics=["mae"])


# In[28]:


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


# In[29]:


# Define the build_lstm_model function
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3))
    return model


# In[30]:


# Creating the LSTM model
model = build_lstm_model()


# In[31]:


model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)


# In[32]:


pred = model.predict(X_test)


# In[33]:


# Training the models
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])


# In[34]:


# Extracting loss values from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting the loss curves
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[35]:


# Calculate RMSE for the training set
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

# Calculate RMSE for the testing set
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# Print the results
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')


# In[36]:


import plotly.graph_objects as go

# Create traces for true values and predicted values on the training set using Plotly
trace_true_train = go.Scatter(x=np.arange(look_back, len(y_train) + look_back),
                              y=y_train, mode='lines', name='True Values', line=dict(color='blue'))
trace_pred_train = go.Scatter(x=np.arange(look_back, len(y_train) + look_back),
                               y=model.predict(X_train).flatten(), mode='lines', name='Predicted Values (Model)', line=dict(color='red'))


# Create layout for the training set prediction plot
layout_train_pred = go.Layout(
    title='LSTM Model Predictions on Training Set',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Power Consumption'),
    showlegend=True,
    annotations=[
        dict(x=0.85, y=0, xref='paper', yref='paper', text='Training Set Predictions', showarrow=False),
    ]
)

# Create figure for the training set prediction plot
fig_train_pred = go.Figure(data=[trace_true_train, trace_pred_train], layout=layout_train_pred)

# Show the training set prediction plot using Plotly
fig_train_pred.show()


# In[37]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Create traces for true values and predicted values on the test set using Plotly
trace_true_test = go.Scatter(x=np.arange(len(y_train) + look_back, len(y_train) + len(y_test)),
                             y=y_test, mode='lines', name='True Values', line=dict(color='blue'))
trace_pred_test = go.Scatter(x=np.arange(len(y_train) + look_back, len(y_train) + len(y_test)),
                              y=pred.flatten(), mode='lines', name='Predicted Values (Model)', line=dict(color='red'))

# Create layout for the test set prediction plot
layout_test_pred = go.Layout(
    title='LSTM Model Predictions on Test Set',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Power Consumption'),
    showlegend=True,
    annotations=[
        dict(x=0.85, y=0, xref='paper', yref='paper', text='Test Set Predictions', showarrow=False),
    ]
)

# Create figure for the test set prediction plot
fig_test_pred = go.Figure(data=[trace_true_test, trace_pred_test], layout=layout_test_pred)

# Show the test set prediction plot using Plotly
fig_test_pred.show()


# In[38]:


model.save('lstm_model.h5')


# In[39]:


import os

# Get the current working directory
current_path = os.getcwd()

# Save the model with the full path
model.save(os.path.join(current_path, 'lstm_model.h5'))

# Print the path
print(f"Model saved in: {os.path.join(current_path, 'lstm_model.h5')}")


# In[ ]:





# In[ ]:





# In[ ]:




