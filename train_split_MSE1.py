
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os

# Try to load CSV - check current directory first
csv_path = 'landPrice2.csv'
if not os.path.exists(csv_path):
    # Try alternate path
    csv_path = 'C:/Users/User/Downloads/landPrice2.csv'

data = pd.read_csv(csv_path)

df = pd.DataFrame(data)

x = df[["size_sqft"]]
y = df["price_k"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(pd.DataFrame({"size_sqft": [1800]}))

print(prediction)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)

