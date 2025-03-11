import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load the dataset
file_path = './dataset recycled aggregate natural fiber.csv'
data = pd.read_csv(file_path)

# Rename 'Fiber Type' column to 'FT'
data.rename(columns={'Fiber Type': 'FT'}, inplace=True)

# One-hot encode the 'FT' column for all possible fiber types in the training data
data = pd.get_dummies(data, columns=['FT'], drop_first=True)

# Define features and target
features = data.drop(columns=['CS'])
target = data['CS']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the models
models = {
    "XGBoost": XGBRegressor(),
    "Random Forest": RandomForestRegressor(),
    "MLP": MLPRegressor(),
    "LightGBM": lgb.LGBMRegressor(),
    "CatBoost": CatBoostRegressor(learning_rate=0.1, iterations=1000, depth=6, verbose=0)
}

# Fit all models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)

# Create sliders for user input
st.sidebar.title("Input Variables:")

# Sliders for each feature
cement = st.sidebar.slider("Cement", min_value=float(features['Cem'].min()), max_value=float(features['Cem'].max()), value=380.0)
wb_ratio = st.sidebar.slider("Water-to-Binder Ratio", min_value=float(features['W/B'].min()), max_value=float(features['W/B'].max()), value=0.40)
fine_aggregate = st.sidebar.slider("Fine Aggregate", min_value=float(features['FA'].min()), max_value=float(features['FA'].max()), value=950.0)
coarse_aggregate = st.sidebar.slider("Coarse Aggregate", min_value=float(features['CA'].min()), max_value=float(features['CA'].max()), value=650.0)
recycled_coarse_aggregate = st.sidebar.slider("Recycled Coarse Aggregate", min_value=float(features['RCA'].min()), max_value=float(features['RCA'].max()), value=50.0)
superplasticizer = st.sidebar.slider("Superplasticizer", min_value=float(features['SP'].min()), max_value=float(features['SP'].max()), value=8.0)
natural_fiber = st.sidebar.slider("Natural Fiber", min_value=float(features['NF'].min()), max_value=float(features['NF'].max()), value=1.0)

# Sliders for Length and Age
length = st.sidebar.slider("Length", min_value=float(features['Length'].min()), max_value=float(features['Length'].max()), value=25.0)
age = st.sidebar.slider("Age", min_value=int(features['Age'].min()), max_value=int(features['Age'].max()), value=7)

# Select box for Fiber Type (Categorical input)
fiber_type = st.sidebar.selectbox("Fiber Type", ['Kenaf', 'Jute', 'Sisal', 'Ramie', 'Coir', 'Bamboo'])

# Prepare input data for prediction
input_data = {
    'Cem': [cement],
    'W/B': [wb_ratio],
    'FA': [fine_aggregate],
    'CA': [coarse_aggregate],
    'RCA': [recycled_coarse_aggregate],
    'SCM': [0.0],  # Assuming SCM is not provided, set to 0
    'SP': [superplasticizer],
    'NF': [natural_fiber],
    'Length': [length],
    'Age': [age],
    'FT_Kenaf': [1 if fiber_type == 'Kenaf' else 0],
    'FT_Jute': [1 if fiber_type == 'Jute' else 0],
    'FT_Sisal': [1 if fiber_type == 'Sisal' else 0],
    'FT_Ramie': [1 if fiber_type == 'Ramie' else 0],
    'FT_Coir': [1 if fiber_type == 'Coir' else 0],
    'FT_Bamboo': [1 if fiber_type == 'Bamboo' else 0]
}

# Convert input data into DataFrame
input_df = pd.DataFrame(input_data)

# Align the input data columns with the training data
missing_cols = set(X_train.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Reorder columns to match the training data columns
input_df = input_df[X_train.columns]

# Scale the input features using the same scaler used for training
input_scaled = scaler.transform(input_df)

# Make predictions using all models
predictions = {}
for model_name, model in models.items():
    predicted_cs = model.predict(input_scaled)
    predictions[model_name] = predicted_cs[0]

# Calculate the error and ±% for each model
errors = {}
for model_name, predicted_cs in predictions.items():
    error = abs(predicted_cs - target.mean())  # Absolute error
    error_percentage = (error / target.mean()) * 100  # ±% error
    errors[model_name] = (predicted_cs, error_percentage)

# Display the predicted CS value and error percentage from all models
st.subheader("Predicted Compressive Strength (CS) Values and ±% Error:")

for model_name, (cs_value, error_percentage) in errors.items():
    st.write(f"{model_name}: {cs_value:.2f} MPa ± {error_percentage:.2f}%")

# Optional: Show the input data for reference
st.subheader("Input Variables")
st.write(input_df)
