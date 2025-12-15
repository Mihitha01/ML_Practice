import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('C:/Users/User/Downloads/house_prices_multi.csv')

print(data.head())

df = pd.DataFrame(data)

x = df[['size_sqft', 'bedrooms', 'age_years']]
y = df['price_k']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)

print("Model Coefficients (slopes):", model.coef_)
print("Model Intercept:", model.intercept_)

new_house = pd.DataFrame({'size_sqft': [2100], 'bedrooms': [3], 'age_years': [10]})

print("Predicted price for new house (in thousands):", model.predict(new_house))

