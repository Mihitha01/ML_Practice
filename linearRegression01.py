# Import the LinearRegression model from scikit-learn
# This model is used to find a straight-line relationship between input and output
from sklearn.linear_model import LinearRegression

# Import NumPy for numerical operations and array handling
import numpy as np

# Create input data (features)
# np.array creates a NumPy array
# reshape(-1, 1) converts it into a 2D array with one column (required by scikit-learn)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Create output data (target values / labels)
# These are the values the model will learn to predict
y = np.array([5, 7, 9, 11, 13])

# Create (initialize) the Linear Regression model
# At this stage, the model has not learned anything yet
model = LinearRegression()

# Train the model using the input (x) and output (y) data
# The model learns the best-fit line: y = mx + c
model.fit(x, y)

# Use the trained model to make a prediction
# We pass [[10]] because scikit-learn expects a 2D array
prediction = model.predict([[10]])

# Print the predicted value for x = 10
print(prediction)

# (Optional but useful)
# Print the slope (m) learned by the model
print("Slope (m):", model.coef_)

# Print the intercept (c) learned by the model
print("Intercept (c):", model.intercept_)
