# ===============================
# REAL DATASET LINEAR REGRESSION EXAMPLE
# Example: Predicting house prices based on house size
# ===============================

# Import required libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# ---------------------------------
# STEP 1: CREATE A REALISTIC DATASET
# ---------------------------------
# House size in square feet (input feature)
# House price in thousands (target value)

data = {
    "size_sqft": [500, 750, 1000, 1250, 1500, 1750, 2000],
    "price_k": [50, 80, 110, 140, 170, 200, 230]
}

# Convert dictionary to a DataFrame (real-world format)
df = pd.DataFrame(data)

# ---------------------------------
# STEP 2: SPLIT FEATURES AND TARGET
# ---------------------------------
# X must be 2D (feature matrix)
X = df[["size_sqft"]]

# y is the target (what we want to predict)
y = df["price_k"]

# ---------------------------------
# STEP 3: CREATE THE MODEL
# ---------------------------------
model = LinearRegression()

# ---------------------------------
# STEP 4: TRAIN THE MODEL
# ---------------------------------
model.fit(X, y)

# ---------------------------------
# STEP 5: MAKE A PREDICTION
# ---------------------------------
# Predict house price for a 1800 sqft house
prediction = model.predict([[1800]])

print("Predicted price for 1800 sqft house (in thousands):", prediction)

# ---------------------------------
# STEP 6: MODEL PARAMETERS
# ---------------------------------
# Slope: price increase per square foot
print("Slope (price per sqft):", model.coef_)

# Intercept: base price when size is zero (theoretical)
print("Intercept:", model.intercept_)

# ===============================
# END OF CODE
# ===============================
