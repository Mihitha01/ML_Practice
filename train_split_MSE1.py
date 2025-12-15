
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/User/Downloads/landPrice2.csv')

df = pd.DataFrame(data)

x = df[["size_sqft"]]
y = df["price_k"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)