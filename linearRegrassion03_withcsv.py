
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/User/Downloads/landPrice2.csv')

print(data.head())

df = pd.DataFrame(data)

X = df[['size_sqft']]
y = df['price_k']

model = LinearRegression()

model.fit(X, y)

prediction = model.predict(pd.DataFrame({'size_sqft': [1800]}))

print("Predicted price for 1800 sqft land (in thousands):", prediction)

print("Slope (price per sqft):", model.coef_)

print("Intercept:", model.intercept_)

