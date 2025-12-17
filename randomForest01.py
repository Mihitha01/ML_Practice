import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_prices_multi.csv")

x = df[["size_sqft", "bedrooms", "age_years"]]
y = df["price_k"]  

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

y_perd = model.predict(x_test)

mse = mean_squared_error(y_test, y_perd)

print("Mean Squared Error: ", mse)

new_house = [[2500, 4, 5]]  # Example: 2500 sqft, 4 bedrooms, 5 years old

prediction = model.predict(new_house)

print("Predicted price (in $1000s): ", prediction)

for feature, importance in zip(x.columns, model.feature_importances_):
    print(f"Feature: {feature}, Importance: {importance}")

# Evaluate feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

