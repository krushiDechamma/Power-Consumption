#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection using Isolation Forest

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[2]:


data = pd.read_csv("C:\\Users\\Madhu Shree\\OneDrive\\Desktop\\Minor\\data.csv", 
                   parse_dates={'dt': ['Date', 'Time']},  
                   infer_datetime_format=True,  
                   na_values=['nan', '?'],  
                   index_col='dt') 


# In[3]:


data.head()


# In[4]:


data = data.replace('?', np.nan)
data.fillna(method='ffill', inplace=True)


# In[5]:


# Resample data into day-wise bins and sum values
data = data.resample('D').sum()


# In[ ]:





# In[6]:


# Define the new target column
target_column = 'Global_active_power'


# In[7]:


# Select the target column for training the anomaly detection model
X_train_anomaly = data.loc[:'2009-12-31', target_column].values.reshape(-1, 1)


# In[8]:


# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination=0.3, random_state=42)
# Fit the model on the training data
iso_forest.fit(X_train_anomaly)


# In[9]:


# Predict anomalies on the test data
X_test_anomaly = data[data.index.year == 2010][target_column].values.reshape(-1, 1)
anomaly_preds = iso_forest.predict(X_test_anomaly)


# In[10]:


# Convert predictions to binary labels (1 for normal, -1 for anomaly)
anomaly_labels = np.where(anomaly_preds == 1, 0, 1)


# In[11]:


# Add the anomaly labels to the test data
data.loc[data.index.year == 2010, 'Anomaly'] = anomaly_labels


# In[12]:


import plotly.express as px

# Filter data for the year 2010 and anomalies
filtered_data = data[(data.index.year == 2010) & (data['Anomaly'] == 1)]

# Create an interactive plot using plotly
fig = px.line(data[data.index.year == 2010], x=data[data.index.year == 2010].index, y=target_column,
              labels={'x': 'Date', 'y': 'sub3'}, title='Anomaly Detection using Isolation Forest')

# Add scatter plot for anomalies
fig.add_scatter(x=filtered_data.index, y=filtered_data[target_column], mode='markers', name='Anomaly', marker=dict(color='red'))

# Update layout for better hover info
fig.update_layout(hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12))

# Set the order of categories for the x-axis
fig.update_xaxes(categoryorder="category ascending", tickmode="linear", dtick="M1")

# Show the plot
fig.show()


# In[13]:


from sklearn.metrics import precision_score, recall_score, f1_score

# True labels (actual anomalies)
true_labels = data.loc[data.index.year == 2010, 'Anomaly']

# Predicted labels from the anomaly detection model
predicted_labels = data.loc[data.index.year == 2010, 'Anomaly']

# Compute precision, recall, and F1-score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print the computed metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[14]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.title("Confusion Matrix for Anomaly Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[15]:


# Count the occurrences of value 1 in the 'Anomaly' column
num_anomalies = data['Anomaly'].value_counts().get(1, 0)

# Calculate the total number of rows in the DataFrame
total_rows = len(data)

# Calculate the percentage of anomalies
percentage_anomalies = (num_anomalies / total_rows) * 100

print("Percentage of anomalies:", percentage_anomalies, "%")


# In[16]:


# Count the occurrences of value 1 in the 'Anomaly' column
num_anomalies = data['Anomaly'].value_counts()[1]

print("Number of anomalies (value 1):", num_anomalies)


# In[17]:


# Summary statistics of anomaly labels
anomaly_summary = data['Anomaly'].value_counts()
print("Anomaly Summary:")
print(anomaly_summary)


# In[18]:


# Display DataFrame with anomaly labels
data[data.index.year == 2010]


# In[ ]:




