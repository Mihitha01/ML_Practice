import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data with error handling
csv_path = 'house_prices_multi.csv'
if not os.path.exists(csv_path):
    csv_path = 'C:/Users/User/Downloads/house_prices_multi.csv'

data = pd.read_csv(csv_path)

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(data.head())
print(f"\nDataset shape: {data.shape}")
print(f"\nData statistics:\n{data.describe()}")

df = pd.DataFrame(data)

x = df[['size_sqft', 'bedrooms', 'age_years']]
y = df['price_k']

# Check for correlations
print("\n" + "=" * 60)
print("FEATURE CORRELATIONS WITH PRICE")
print("=" * 60)
correlations = df[['size_sqft', 'bedrooms', 'age_years', 'price_k']].corr()['price_k'].drop('price_k')
print(correlations)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature Scaling for better model performance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION (WITHOUT SCALING)")
print("=" * 60)

# Train model without scaling
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f} (explains {r2*100:.2f}% of variance)")
print(f"Model Coefficients (slopes): {model.coef_}")
print(f"Model Intercept: {model.intercept_:.4f}")

# Cross-validation for better generalization estimate (use 3-fold for small dataset)
if len(x_train) >= 3:
    cv_scores = cross_val_score(model, x_train, y_train, cv=min(3, len(x_train)), scoring='r2')
    print(f"\nCross-Validation R² Scores: {cv_scores}")
    print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
else:
    print("\nDataset too small for cross-validation")

print("\n" + "=" * 60)
print("MODEL WITH FEATURE SCALING")
print("=" * 60)

# Train model with scaling
model_scaled = LinearRegression()
model_scaled.fit(x_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(x_test_scaled)

mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print(f"Mean Squared Error: {mse_scaled:.4f}")
print(f"Root Mean Squared Error: {rmse_scaled:.4f}")
print(f"R² Score: {r2_scaled:.4f}")
print(f"Model Coefficients (slopes): {model_scaled.coef_}")

# Make predictions
new_house = pd.DataFrame({'size_sqft': [2100], 'bedrooms': [3], 'age_years': [10]})
new_house_scaled = scaler.transform(new_house)

pred_original = model.predict(new_house)[0]
pred_scaled = model_scaled.predict(new_house_scaled)[0]

print("\n" + "=" * 60)
print("PREDICTIONS FOR NEW HOUSE (2100 sqft, 3 bed, 10 yrs)")
print("=" * 60)
print(f"Predicted price (unscaled model): ${pred_original:.2f}k")
print(f"Predicted price (scaled model): ${pred_scaled:.2f}k")

# Residual Analysis
residuals = y_test - y_pred
print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)
print(f"Mean of residuals: {residuals.mean():.4f} (should be close to 0)")
print(f"Std of residuals: {residuals.std():.4f}")
print(f"Min residual: {residuals.min():.4f}")
print(f"Max residual: {residuals.max():.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price (k)')
axes[0, 0].set_ylabel('Predicted Price (k)')
axes[0, 0].set_title('Actual vs Predicted Prices')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price (k)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of residuals
axes[1, 0].hist(residuals, bins=10, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Residuals')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature importance
feature_importance = np.abs(model.coef_)
features = ['size_sqft', 'bedrooms', 'age_years']
axes[1, 1].barh(features, feature_importance)
axes[1, 1].set_xlabel('Absolute Coefficient Value')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=100, bbox_inches='tight')
print("\n✅ Visualization saved as 'model_analysis.png'")
plt.show()
