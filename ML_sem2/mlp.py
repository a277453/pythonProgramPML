import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the data
# Assuming 'x' is the feature and 'y' is the target
try:
    df = pd.read_csv('regression_data.csv')
    X = df[['x']].values
    y = df['y'].values
except FileNotFoundError:
    print("Error: regression_data.csv not found.")
    # Placeholder for demonstration if file is missing
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 2 + 1.5*X + 0.8*X**2 - 0.5*X**3 + np.random.normal(0, 1, X.shape)

# 2. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# 3. Build and evaluate models for degrees 1 through 6
degrees = range(1, 7)

for d in degrees:
    # Transform features to polynomial degree d
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate Metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    results.append({
        'Degree': d,
        'MSE Train': mse_train,
        'MSE Test': mse_test,
        'R2 Test': r2_test
    })

# 4. Display Results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 5. Plotting the Error Curves
plt.figure(figsize=(10, 6))
plt.plot(results_df['Degree'], results_df['MSE Train'], label='Train MSE', marker='o')
plt.plot(results_df['Degree'], results_df['MSE Test'], label='Test MSE', marker='s')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Testing Error')
plt.legend()
plt.grid(True)
plt.show()