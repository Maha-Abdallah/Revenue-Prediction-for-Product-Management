#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# Load the dataset (replace 'your_dataset.csv' with the actual dataset path)
data = pd.read_csv('supply_chain_data.csv')

# Data Exploration and Cleaning
print("Data Overview:")
print(data.info())


# In[2]:


# Handling Missing Values
data = data.dropna()

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())


# In[3]:


# Data Visualization
# Pairplot for numeric variables
sns.pairplot(data[['Price', 'Availability', 'Number of products sold', 'Revenue generated', 'Stock levels']])
plt.show()



# In[4]:


# Distribution of 'Revenue generated'
plt.figure(figsize=(10, 6))
sns.histplot(data['Revenue generated'], bins=20, kde=True)
plt.title('Distribution of Revenue generated')
plt.show()


# In[12]:


# Countplot for categorical variables
plt.figure(figsize=(12, 8))
sns.countplot(x='Customer demographics', data=data)
plt.title('Count of Customer demographics')
plt.show()


# In[6]:


# Feature Engineering (if necessary)
# Example: Creating a new feature 'Profit' by subtracting 'Costs' from 'Revenue'
data['Profit'] = data['Revenue generated'] - data['Costs']


# In[7]:


# Machine Learning (Regression Example)
# Assuming we want to predict 'Revenue generated' based on other features
X = data.drop(['Revenue generated'], axis=1)
y = data['Revenue generated']


# In[8]:


# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)


# In[9]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Feature Scaling for numeric variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=np.number))

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

for model_name, model in models.items():
    print(f"\nTraining {model_name}:")
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    accuracy = model.score(X_test_scaled, y_test)  # Calculate accuracy (R^2 score)
    
    print(f"Mean Squared Error for {model_name}: {mse}")
    print(f"Accuracy for {model_name}: {accuracy:.2f}")


# Feature Importance for Random Forest Regressor
rf_model = models['Random Forest Regressor']
feature_importance_rf = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': rf_model.feature_importances_})
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
print("\nFeature Importance for Random Forest Regressor:")
print(feature_importance_rf)

