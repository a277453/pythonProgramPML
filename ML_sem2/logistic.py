import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Create Dummy Data
# X: 5 rows, 2 columns (Features)
# y: 5 rows (Boolean/Binary Target)
X = np.array([
    [0.5, 1.2],
    [1.0, 3.1],
    [3.5, 0.5],
    [4.2, 2.2],
    [5.1, 4.8]
])

y = np.array([0, 0, 1, 1, 1])

# 2. Initialize and Train the Model
model = LogisticRegression()
model.fit(X, y)

# 3. Make a Prediction
# Testing with a new data point
new_data = np.array([[2.5, 2.5]])
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

# 4. Display Results
print(f"Weights (Coefficients): {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Prediction for {new_data[0]}: {'True' if prediction[0] == 1 else 'False'}")
print(f"Confidence (Probability): {probability[0]}")
