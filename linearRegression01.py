from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([5, 7, 9, 11, 13])

model = LinearRegression()

model.fit(x, y)

prediction = model.predict([[10]])

print(prediction)
