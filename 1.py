import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../../Desktop/data (2).csv')

# Display basic info
df.info()

# Check for null values
print(df.isnull().sum())

# Check for duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# Drop duplicates
df = df.drop_duplicates()

# Extract brand name from the 'name' column
df['brand_name'] = df['name'].str.split().str[0]
df['brand_name'] = df['brand_name'].replace('Land', 'Land Rover')

# Create a new dataframe with selected columns
df1 = df.drop(['fuel', 'seller_type', 'name'], axis=1)

# One-Hot Encoding for categorical columns
df1 = pd.get_dummies(df1, columns=['transmission', 'brand_name'], drop_first=True)

# Label Encoding for 'owner' column
le = LabelEncoder()
df1['owner'] = le.fit_transform(df1['owner'])

# Add 'car_age' column
current_year = 2025
df1['car_age'] = current_year - df1['year']
df1.drop('year', axis=1, inplace=True)

# Scale 'km_driven' column
scaler = StandardScaler()
df1['km_driven'] = scaler.fit_transform(df1[['km_driven']])

# Log transformation for 'selling_price' and 'km_driven' to handle outliers
df1['selling_price'] = np.log1p(df1['selling_price'])
df1['km_driven'] = np.log1p(df1['km_driven'])

# Fill NaN values in 'km_driven' with median
df1['km_driven'].fillna(df1['km_driven'].median(), inplace=True)

# Split data into features (x) and target (y)
x = df1.drop('selling_price', axis=1)
y = df1['selling_price']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Save model and columns
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(x_train.columns, f)
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
