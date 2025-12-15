import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load dataset
df = pd.read_csv("house_prices_multi.csv")

# 2. Separate features (X) and target (y)
X = df[["size_sqft", "bedrooms", "age_years"]]  # multiple features
y = df["price_k"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on unseen data
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 7. Inspect learned parameters
print("Coefficients (m1, m2, m3):", model.coef_)
print("Intercept (c):", model.intercept_)

# 8. Predict a new house
new_house = [[1600, 3, 7]]  # size, bedrooms, age
prediction = model.predict(new_house)
print("Predicted price (in thousands):", prediction)
