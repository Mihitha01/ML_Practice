# ===============================
# ===============================
# FEATURE SCALING WITH STANDARD SCALER
# EXPLAINED LINE BY LINE (COMMENT STYLE)
# ===============================

# Import pandas to load and work with CSV data
import pandas as pd

# Import Linear Regression model
from sklearn.linear_model import LinearRegression

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Import Mean Squared Error to evaluate model performance
from sklearn.metrics import mean_squared_error

# ---------------------------------
# LOAD THE DATASET
# ---------------------------------

# Read the CSV file and store it as a DataFrame
# The CSV must be in the same folder as this Python file
# or you must provide the full file path
df = pd.read_csv("house_prices_multi.csv")

# ---------------------------------
# SEPARATE FEATURES AND TARGET
# ---------------------------------

# Select input features (X)
# Double brackets ensure X is a 2D structure (required by ML models)
X = df[["size_sqft", "bedrooms", "age_years"]]

# Select the target variable (y)
# Single brackets give a 1D array, which is correct for labels
y = df["price_k"]

# ---------------------------------
# SPLIT DATA INTO TRAIN AND TEST SETS
# ---------------------------------

# Split data into training and testing portions
# test_size=0.2 → 20% testing data, 80% training data
# random_state ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# FEATURE SCALING (STANDARDIZATION)
# ---------------------------------

# Create a StandardScaler object
# This will standardize features to mean=0 and std=1
scaler = StandardScaler()

# Fit the scaler ONLY on training data
# This learns the mean and standard deviation of each feature
X_train_scaled = scaler.fit_transform(X_train)

# Apply the learned scaling to the test data
# IMPORTANT: Do NOT use fit_transform here
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# TRAIN THE MODEL
# ---------------------------------

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)

# ---------------------------------
# MAKE PREDICTIONS ON TEST DATA
# ---------------------------------

# Predict target values for unseen (test) data
y_pred = model.predict(X_test_scaled)

# ---------------------------------
# EVALUATE THE MODEL
# ---------------------------------

# Calculate Mean Squared Error between actual and predicted values
mse = mean_squared_error(y_test, y_pred)

# Print the error value
print("Mean Squared Error:", mse)

# ---------------------------------
# PREDICT FOR NEW, UNSEEN DATA
# ---------------------------------

# Define a new house with size, bedrooms, and age
new_house = [[1600, 3, 7]]

# Scale the new data using the SAME scaler
new_house_scaled = scaler.transform(new_house)

# Predict the price for the new house
prediction = model.predict(new_house_scaled)

# Print the predicted house price
print("Predicted price (in thousands):", prediction)

# ===============================
# END OF FEATURE SCALING EXAMPLE
# ===============================
